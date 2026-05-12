"""Training, inference, and active-learning helpers.

Per Section 7 of the spec. Highlights:

* **Image-level split** (Section 16 rule #1) is enforced in
  ``build_dataloaders`` by joining patches to images via
  ``utils.db.get_labeled_patches_with_split``. A patch's split comes from
  its source image's split — patches from one image cannot leak across.
* **Augmentation on train only** (HFlip, VFlip, rotation ±15°). Validation
  patches go through the same normalization (uint8/255.0) but no augment.
* **Class balance** has two modes (``weighted_loss`` and ``oversample``)
  exposed via the spec's ``class_balance_mode`` setting.
* **BCEWithLogitsLoss** for numerical stability — the model returns logits
  during training (``return_logits=True``).
* **Sidecar JSON** for every checkpoint, plus a ``training_runs`` row.
* **Active learning queue** with the burn-in / uncertainty-sampling
  two-phase approach from Section 15.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from learning.cnn import IntersectionCNN
from utils import db


log = logging.getLogger("training")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _PatchDataset(Dataset):
    """In-memory list of patch rows; loads PNGs lazily on __getitem__.

    Training rows go through random-flip + small rotation augmentation.
    Validation rows just normalize. Both branches finish with
    ``ToTensor`` (which divides uint8 by 255 → float32 in [0,1] and adds
    a channel dim, matching the spec's normalization rule).
    """

    def __init__(self, rows: list[dict], augment: bool):
        self.rows = rows
        self.augment = augment
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        # Open as grayscale to be safe even if a patch was somehow saved RGB.
        # The patch_extractor writes mode-L PNGs but we don't want a one-off
        # corruption to derail training.
        img = Image.open(row["patch_path"]).convert("L")
        x = self.transform(img)  # (1, H, W) float32 in [0,1]
        y = float(row["label"])
        return x, torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataloader construction (Section 7's build_dataloaders)
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size: int = 32,
    class_balance_mode: str = "weighted_loss",
) -> tuple[DataLoader, DataLoader, dict]:
    """Build training and validation DataLoaders.

    Returns ``(train_loader, val_loader, info)`` where ``info`` carries:
        - ``pos_weight``: scalar tensor for BCEWithLogitsLoss (only
          meaningful when ``class_balance_mode == 'weighted_loss'``)
        - ``class_balance``: dict ``{intersection: N, non_intersection: N}``
          across the train split
        - ``train_size``, ``val_size``: row counts

    Section 16 rule #1 enforcement: train rows come from images whose
    ``split == 'train'``; val rows come from the test split. The join
    happens at the SQL layer in ``utils.db.get_labeled_patches_with_split``.
    """
    if class_balance_mode not in ("weighted_loss", "oversample"):
        raise ValueError(
            f"class_balance_mode must be 'weighted_loss' or 'oversample', "
            f"got {class_balance_mode!r}"
        )

    train_rows = db.get_labeled_patches_with_split("train")
    val_rows = db.get_labeled_patches_with_split("test")

    if not train_rows:
        raise RuntimeError(
            "No labeled patches in the train split. Label some patches "
            "from training-split images before training."
        )

    pos_train = sum(1 for r in train_rows if r["label"] == 1)
    neg_train = len(train_rows) - pos_train
    log.info(
        "build_dataloaders: train=%d (pos=%d, neg=%d), val=%d, mode=%s",
        len(train_rows), pos_train, neg_train, len(val_rows), class_balance_mode,
    )

    train_ds = _PatchDataset(train_rows, augment=True)
    val_ds = _PatchDataset(val_rows, augment=False)

    # Worker count: 0 means same-process loading. Safer on Windows /
    # frozen apps and adequate for our patch sizes (64×64 PNGs are tiny).
    num_workers = 0

    if class_balance_mode == "oversample" and pos_train > 0 and neg_train > 0:
        # Per-sample weight = inverse class frequency. WeightedRandomSampler
        # then draws each minibatch with replacement, balancing classes on
        # average. We don't shuffle separately — the sampler itself is
        # stochastic.
        weight_pos = 1.0 / pos_train
        weight_neg = 1.0 / neg_train
        sample_weights = [
            weight_pos if r["label"] == 1 else weight_neg for r in train_rows
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_rows),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, drop_last=False,
        )
        pos_weight_tensor = torch.tensor(1.0)  # not used; sampler balances
    else:
        # weighted_loss path: ordinary shuffled training loader; pass the
        # class-imbalance ratio to BCEWithLogitsLoss as ``pos_weight``.
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=False,
        )
        # pos_weight = neg/pos so the positive class gets up-weighted in
        # proportion to its rarity. Guard against zero positives — if the
        # user trained without any intersections we'd hit 0/0.
        if pos_train == 0:
            pos_weight_value = 1.0
        else:
            pos_weight_value = float(neg_train) / float(pos_train)
        pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    info = {
        "pos_weight": pos_weight_tensor,
        "class_balance": {
            "intersection": pos_train,
            "non_intersection": neg_train,
        },
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
    return train_loader, val_loader, info


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Run one training epoch. Returns ``(avg_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)  # shape (B, 1) to match logits

        optimizer.zero_grad()
        logits = model(x, return_logits=True)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        # Accuracy at threshold 0.5 (i.e. logit > 0).
        preds = (logits > 0).float()
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Validate. Returns dict with loss, acc, and confusion-matrix counts."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            logits = model(x, return_logits=True)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            preds = (logits > 0).float()
            tp += int(((preds == 1) & (y == 1)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())

    if total_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    acc = (tp + tn) / total_samples
    return {
        "loss": total_loss / total_samples,
        "acc": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    """F1 score. Returns 0 when undefined (no positives anywhere)."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    progress_callback: Callable | None = None,
    pos_weight: torch.Tensor | None = None,
    class_balance: dict | None = None,
    settings_snapshot: dict | None = None,
) -> dict:
    """Train ``model`` and persist a checkpoint + sidecar metadata.

    Returns a history dict with per-epoch lists:
        ``train_loss, val_loss, train_acc, val_acc, tp, tn, fp, fn``.

    Per the spec:
      * Saves weights to ``models/checkpoints/model_epoch{N}_{timestamp}.pth``
      * Writes sidecar JSON to ``models/{run_id}_metadata.json``
      * Inserts a row into ``training_runs``
      * Calls ``progress_callback(epoch, total, metrics_dict)`` after each epoch
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "tp": [], "tn": [], "fp": [], "fn": [],
    }

    log.info(
        "train: epochs=%d lr=%g batch_size=%d device=%s",
        epochs, learning_rate, batch_size, device,
    )
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_metrics = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["tp"].append(val_metrics["tp"])
        history["tn"].append(val_metrics["tn"])
        history["fp"].append(val_metrics["fp"])
        history["fn"].append(val_metrics["fn"])

        elapsed = time.time() - t_epoch
        log.info(
            "epoch %d/%d: train_loss=%.4f train_acc=%.3f val_loss=%.4f "
            "val_acc=%.3f tp=%d fp=%d fn=%d (%.1fs)",
            epoch, epochs, train_loss, train_acc,
            val_metrics["loss"], val_metrics["acc"],
            val_metrics["tp"], val_metrics["fp"], val_metrics["fn"],
            elapsed,
        )

        if progress_callback:
            progress_callback(epoch, epochs, {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "tp": val_metrics["tp"],
                "tn": val_metrics["tn"],
                "fp": val_metrics["fp"],
                "fn": val_metrics["fn"],
            })

    # Final-epoch metrics for the sidecar / DB.
    final_tp = history["tp"][-1] if history["tp"] else 0
    final_fp = history["fp"][-1] if history["fp"] else 0
    final_fn = history["fn"][-1] if history["fn"] else 0
    final_tn = history["tn"][-1] if history["tn"] else 0
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    val_f1 = _f1(final_tp, final_fp, final_fn)

    # ---- Persist artifacts ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_filename = f"model_epoch{epochs}_{timestamp}.pth"
    ckpt_path_rel = f"models/checkpoints/{ckpt_filename}"

    # Save weights AND a small architecture descriptor so inference can
    # reconstruct the model without separately knowing patch_size.
    torch.save({
        "state_dict": model.state_dict(),
        "patch_size": getattr(model, "patch_size", 64),
        "trained_at": timestamp,
    }, ckpt_path_rel)
    log.info("train: checkpoint saved to %s", ckpt_path_rel)

    # Sidecar JSON per Section 14
    metadata_dir = Path("models")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_filename = f"{run_id}_metadata.json"
    metadata_path_rel = f"models/{metadata_filename}"

    # Pull "annotations_db_timestamp" as the file mtime of the SQLite DB —
    # it's the closest stable proxy to "DB state at time of training".
    db_mtime = (
        datetime.fromtimestamp(os.path.getmtime(db.DB_PATH)).isoformat()
        if os.path.exists(db.DB_PATH) else None
    )

    # Precision / recall for the sidecar 'performance' block.
    val_precision = (final_tp / (final_tp + final_fp)) if (final_tp + final_fp) else 0.0
    val_recall = (final_tp / (final_tp + final_fn)) if (final_tp + final_fn) else 0.0

    cb = class_balance or {"intersection": 0, "non_intersection": 0}
    train_imgs = sum(1 for i in db.get_images() if i["split"] == "train")
    test_imgs = sum(1 for i in db.get_images() if i["split"] == "test")

    sidecar = {
        "run_id": run_id,
        "checkpoint_path": ckpt_path_rel,
        "trained_at": datetime.now().isoformat(),
        "annotations_db_timestamp": db_mtime,
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patch_size": getattr(model, "patch_size", 64),
            "seed": 42,
        },
        "dataset": {
            "total_labeled": cb["intersection"] + cb["non_intersection"],
            "intersections": cb["intersection"],
            "non_intersections": cb["non_intersection"],
            "train_images": train_imgs,
            "test_images": test_imgs,
        },
        "performance": {
            "val_accuracy": final_val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
        },
        "settings_snapshot": settings_snapshot or {},
    }
    with open(metadata_path_rel, "w") as f:
        json.dump(sidecar, f, indent=2)
    log.info("train: sidecar JSON saved to %s", metadata_path_rel)

    # training_runs row
    db.insert_training_run(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        class_balance=cb,
        dataset_size=cb["intersection"] + cb["non_intersection"],
        train_accuracy=final_train_acc,
        val_accuracy=final_val_acc,
        val_f1=val_f1,
        checkpoint_path=ckpt_path_rel,
        metadata_path=metadata_path_rel,
    )

    total_elapsed = time.time() - t_start
    log.info("train: complete in %.1fs", total_elapsed)
    return history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _load_patch_tensor(patch_path: str) -> torch.Tensor:
    """Load PNG → ``(1, 1, H, W)`` float32 tensor in [0, 1].

    Same normalization as the validation pipeline (no augmentation), so
    inference matches what the model saw during validation.
    """
    img = Image.open(patch_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def run_inference(model: nn.Module, patch_path: str, device: str = "cpu") -> float:
    """Run the model on a single PNG. Returns sigmoid output in [0, 1]."""
    model.eval()
    x = _load_patch_tensor(patch_path).to(device)
    with torch.no_grad():
        prob = model(x, return_logits=False)
    return float(prob.item())


def run_inference_all(
    model: nn.Module,
    image_id: int,
    device: str = "cpu",
    batch_size: int = 64,
    progress_callback: Callable | None = None,
) -> int:
    """Run inference on every patch belonging to ``image_id``.

    Updates each patch's ``prediction`` column in the DB. Returns the
    number of patches scored.

    Note: this batches patches manually rather than using a DataLoader,
    because we want to keep the patch-id ↔ score mapping while writing
    back to the DB; a DataLoader would shuffle that mapping unless we
    carry indices through, which adds complexity for no gain at our scale.
    """
    rows = db.get_patches(image_id=image_id)
    if not rows:
        log.info("run_inference_all: no patches for image_id=%d", image_id)
        return 0

    model = model.to(device)
    model.eval()

    updates: list[tuple[int, float]] = []
    n_batches = (len(rows) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            chunk = rows[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            tensors = [_load_patch_tensor(r["patch_path"]).squeeze(0) for r in chunk]
            batch_x = torch.stack(tensors, dim=0).to(device)
            probs = model(batch_x, return_logits=False)
            for r, p in zip(chunk, probs):
                updates.append((int(r["id"]), float(p.item())))
            if progress_callback:
                progress_callback(
                    batch_idx + 1, n_batches,
                    f"Inference batch {batch_idx + 1}/{n_batches}",
                )

    db.set_patch_predictions_bulk(updates)
    log.info("run_inference_all: image_id=%d, %d patches scored", image_id, len(updates))
    return len(updates)


def load_latest_checkpoint(device: str = "cpu") -> nn.Module | None:
    """Load the most recently saved checkpoint, or ``None`` if none exists.

    Per Section 16 rule #7: the UI always uses the most recently saved
    .pth for inference. We look at ``training_runs`` first (which records
    the checkpoint path); if the table is empty, fall back to the newest
    .pth on disk in ``models/checkpoints/``.
    """
    latest = db.get_latest_training_run()
    if latest and latest.get("checkpoint_path") and os.path.exists(latest["checkpoint_path"]):
        ckpt_path = latest["checkpoint_path"]
    else:
        # Fallback: newest .pth on disk.
        ckpt_dir = Path("models/checkpoints")
        if not ckpt_dir.exists():
            return None
        candidates = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None
        ckpt_path = str(candidates[-1])

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    patch_size = payload.get("patch_size", 64)
    model = IntersectionCNN(patch_size=patch_size).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    log.info("load_latest_checkpoint: loaded %s (patch_size=%d)", ckpt_path, patch_size)
    return model


# ---------------------------------------------------------------------------
# Active-learning queue (Section 15)
# ---------------------------------------------------------------------------

def get_active_learning_queue(
    image_id: int,
    burn_in_complete: bool,
    burn_in_count: int = 50,
) -> list[dict]:
    """Return unlabeled patches in the order they should be presented.

    Phase 1 (``burn_in_complete=False``):
        Up to 100 random unlabeled patches for this image, seeded for
        reproducibility within a session.

    Phase 2 (``burn_in_complete=True``):
        Patches sorted by ``|prediction - 0.5|`` ascending (most uncertain
        first). Patches without a prediction are shuffled and appended
        last — they're either un-inferred or freshly extracted, and we
        prefer the user labels confidently-uncertain patches before
        un-known-confidence ones.

    The ``burn_in_count`` argument echoes the spec's user setting; the
    actual "is burn-in done?" decision lives in the UI based on the count
    of labeled patches, not in this function.
    """
    unlabeled = db.get_patches(image_id=image_id, unlabeled_only=True)
    if not unlabeled:
        return []

    if not burn_in_complete:
        # Random order, up to 100. Use a session-local RNG so the order is
        # the same as long as no relabeling happens, but doesn't disturb
        # the global RNG state.
        rng = random.Random(42)
        shuffled = list(unlabeled)
        rng.shuffle(shuffled)
        return shuffled[:100]

    # Uncertainty sampling. Patches WITH predictions sort by |p - 0.5|
    # ascending; those WITHOUT come last (random order so we don't bias).
    with_pred = [r for r in unlabeled if r["prediction"] is not None]
    without_pred = [r for r in unlabeled if r["prediction"] is None]

    if not with_pred:
        # Section 15: "If no inference has been run yet, fall back to
        # random ordering." This handles that case explicitly.
        rng = random.Random(42)
        shuffled = list(unlabeled)
        rng.shuffle(shuffled)
        return shuffled[:100]

    with_pred.sort(key=lambda r: abs(float(r["prediction"]) - 0.5))
    rng = random.Random(42)
    rng.shuffle(without_pred)
    return with_pred + without_pred
