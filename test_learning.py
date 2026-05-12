"""Smoke test for the learning layer.

Run from the project root after installing requirements:

    python test_learning.py

Builds a synthetic 64×64 dataset, trains for 2 epochs, runs inference, and
verifies that every Section 7 entry point works end-to-end. Cleans up
after itself. Useful for validating the install / ML stack before opening
the GUI.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image


def main() -> int:
    work_dir = tempfile.mkdtemp(prefix="inkjet_learning_test_")
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        for sub in ("data/raw", "data/patches", "data/outputs", "data/cache",
                    "models/checkpoints", "logs"):
            os.makedirs(sub, exist_ok=True)
        sys.path.insert(0, cwd)

        # Initialize a fresh DB
        from utils import db
        db.DB_PATH = os.path.join(work_dir, "data/annotations.db")
        db.init_db()

        # Insert two images, one in each split, with synthetic 64×64 patches.
        # "intersection" patches: cross pattern. "non-intersection": single line.
        train_id = db.insert_image("data/raw/train.tif", "train.tif", "train")
        test_id = db.insert_image("data/raw/test.tif", "test.tif", "test")

        rng = np.random.default_rng(0)

        def make_patch(label: int) -> np.ndarray:
            """Create a synthetic 64×64 grayscale patch."""
            img = np.full((64, 64), 240, dtype=np.uint8)
            if label == 1:
                # Cross pattern -> intersection
                img[30:34, :] = 30
                img[:, 30:34] = 30
            else:
                # Single horizontal line -> non-intersection
                img[30:34, :] = 30
            # Tiny noise so the model has something to learn beyond means
            noise = rng.integers(-15, 15, size=img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return img

        n_per_class = 30
        bulk_rows = []
        for img_id, split_name in [(train_id, "train"), (test_id, "test")]:
            for i in range(n_per_class):
                for label in (0, 1):
                    arr = make_patch(label)
                    p_path = f"data/patches/{img_id}_{split_name}_{i}_{label}.png"
                    Image.fromarray(arr).save(p_path)
                    bulk_rows.append({
                        "image_id": img_id, "patch_path": p_path,
                        "x_center": 32, "y_center": 32,
                        "inner_radius": 8, "outer_radius": 20,
                    })
        db.insert_patches_bulk(bulk_rows)

        # Label them all (we know the ground truth from filename suffix)
        for r in db.get_patches():
            label = 1 if r["patch_path"].endswith("_1.png") else 0
            db.set_patch_label(r["id"], label)
        print(f"Synthesized {len(bulk_rows)} patches (half intersections)")

        # ---- 1. cnn ----
        from learning.cnn import IntersectionCNN
        import torch

        model = IntersectionCNN(patch_size=64)
        x = torch.zeros(2, 1, 64, 64)
        y = model(x)
        assert y.shape == (2, 1)
        assert (y >= 0).all() and (y <= 1).all(), "sigmoid output must be in [0,1]"
        logits = model(x, return_logits=True)
        assert logits.shape == (2, 1)
        # logits should NOT be clipped to [0,1]
        print(f"cnn.IntersectionCNN: forward OK, output shape {y.shape}, "
              f"flat_dim={model._flat_dim}")
        # Test 32 and 128 patch sizes
        m32 = IntersectionCNN(patch_size=32)
        m32(torch.zeros(1, 1, 32, 32))
        m128 = IntersectionCNN(patch_size=128)
        m128(torch.zeros(1, 1, 128, 128))
        print(f"  patch_size 32 flat_dim={m32._flat_dim}, "
              f"patch_size 128 flat_dim={m128._flat_dim}")

        # ---- 2. trainer.build_dataloaders ----
        from learning.trainer import (
            build_dataloaders, train, run_inference, run_inference_all,
            get_active_learning_queue, load_latest_checkpoint,
        )

        train_loader, val_loader, info = build_dataloaders(
            batch_size=16, class_balance_mode="weighted_loss",
        )
        print(f"build_dataloaders: train={info['train_size']} val={info['val_size']} "
              f"pos_weight={info['pos_weight'].item():.3f}")
        assert info["train_size"] == n_per_class * 2  # 30 pos + 30 neg
        assert info["val_size"] == n_per_class * 2

        # Same with oversample mode
        _, _, info_os = build_dataloaders(
            batch_size=16, class_balance_mode="oversample",
        )
        print(f"  oversample mode: pos_weight (unused)={info_os['pos_weight'].item()}")

        # ---- 3. train ----
        epochs = []
        def cb(epoch, total, metrics):
            epochs.append(epoch)
            print(f"  epoch {epoch}/{total}: train_acc={metrics['train_acc']:.3f} "
                  f"val_acc={metrics['val_acc']:.3f}")

        history = train(
            model, train_loader, val_loader,
            epochs=3, learning_rate=1e-3, batch_size=16,
            device="cpu", progress_callback=cb,
            pos_weight=info["pos_weight"], class_balance=info["class_balance"],
            settings_snapshot={"test": True},
        )
        assert epochs == [1, 2, 3]
        assert len(history["train_loss"]) == 3
        # The task is very easy (cross vs line); model should reach high acc
        assert history["val_acc"][-1] > 0.7, \
            f"val_acc only {history['val_acc'][-1]:.3f} after 3 epochs on trivial task"
        print(f"train: final val_acc={history['val_acc'][-1]:.3f} "
              f"final val_loss={history['val_loss'][-1]:.3f}")

        # Verify checkpoint + sidecar were saved
        ckpts = list(__import__("pathlib").Path("models/checkpoints").glob("*.pth"))
        sidecars = list(__import__("pathlib").Path("models").glob("*_metadata.json"))
        assert len(ckpts) == 1, f"expected 1 checkpoint, found {len(ckpts)}"
        assert len(sidecars) == 1, f"expected 1 sidecar, found {len(sidecars)}"
        import json
        with open(sidecars[0]) as f:
            sc = json.load(f)
        assert sc["hyperparameters"]["epochs"] == 3
        assert sc["hyperparameters"]["seed"] == 42
        assert "performance" in sc and "val_accuracy" in sc["performance"]
        print(f"train: sidecar JSON has expected structure")

        # training_runs row
        runs = db.get_training_runs()
        assert len(runs) == 1
        print(f"train: training_runs row inserted, val_f1={runs[0]['val_f1']:.3f}")

        # ---- 4. run_inference + run_inference_all ----
        sample_patch = db.get_patches()[0]["patch_path"]
        score = run_inference(model, sample_patch)
        assert 0.0 <= score <= 1.0
        print(f"run_inference: single-patch score={score:.3f}")

        n_scored = run_inference_all(model, train_id, batch_size=16)
        assert n_scored == n_per_class * 2
        # Verify predictions are now in the DB
        preds = [r["prediction"] for r in db.get_patches(image_id=train_id)]
        assert all(p is not None for p in preds)
        print(f"run_inference_all: scored {n_scored} patches")

        # ---- 5. load_latest_checkpoint ----
        loaded = load_latest_checkpoint(device="cpu")
        assert loaded is not None
        score_after_reload = run_inference(loaded, sample_patch)
        assert abs(score_after_reload - score) < 1e-5, \
            f"reloaded model gives different score: {score_after_reload} vs {score}"
        print(f"load_latest_checkpoint: reload OK, score matches")

        # ---- 6. get_active_learning_queue ----
        # Burn-in phase
        # First clear labels on test image patches so we have unlabeled material
        for p in db.get_patches(image_id=test_id):
            # Set them all to NULL
            with __import__("sqlite3").connect(db.DB_PATH) as conn:
                conn.execute("UPDATE patches SET label = NULL WHERE id = ?", (p["id"],))
        q1 = get_active_learning_queue(test_id, burn_in_complete=False)
        assert len(q1) > 0 and len(q1) <= 100
        print(f"active_learning burn-in: {len(q1)} patches in random order")

        # Phase 2 — but no predictions on test_id patches yet, so it should fall back
        q2_fallback = get_active_learning_queue(test_id, burn_in_complete=True)
        assert len(q2_fallback) == len(q1)
        print(f"active_learning phase 2 (no predictions): falls back to random, "
              f"{len(q2_fallback)} patches")

        # Now add predictions to test set patches; phase 2 should sort by uncertainty
        run_inference_all(loaded, test_id, batch_size=16)
        q2 = get_active_learning_queue(test_id, burn_in_complete=True)
        # Verify ordering: uncertainties (|p - 0.5|) should be non-decreasing
        with_pred = [r for r in q2 if r["prediction"] is not None]
        uncertainties = [abs(r["prediction"] - 0.5) for r in with_pred]
        assert uncertainties == sorted(uncertainties), \
            f"phase 2 not uncertainty-sorted: {uncertainties[:5]}"
        print(f"active_learning phase 2 (with predictions): "
              f"first 3 uncertainties={uncertainties[:3]}")

        # ---- 7. evaluator ----
        from learning.evaluator import (
            compute_metrics, plot_learning_curve, plot_roc_curve,
        )
        # Re-label test set so compute_metrics has labels to compare against
        for r in db.get_patches(image_id=test_id):
            label = 1 if r["patch_path"].endswith("_1.png") else 0
            db.set_patch_label(r["id"], label)
        # Rebuild the val_loader (now that labels are restored)
        _, val_loader2, _ = build_dataloaders(batch_size=16)

        metrics = compute_metrics(val_loader2, model, device="cpu")
        assert "accuracy" in metrics and "confusion_matrix" in metrics
        print(f"compute_metrics: acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} TP={metrics['tp']} FP={metrics['fp']}")

        fig1 = plot_learning_curve(history)
        assert fig1 is not None
        print(f"plot_learning_curve: returned Figure")

        fig2 = plot_roc_curve(val_loader2, model, device="cpu")
        assert fig2 is not None
        print(f"plot_roc_curve: returned Figure")

        # Save them so the user can eyeball them
        fig1.savefig(os.path.join(work_dir, "learning_curve.png"))
        fig2.savefig(os.path.join(work_dir, "roc_curve.png"))
        print(f"  Plots saved to {work_dir}/learning_curve.png and roc_curve.png")
        print(f"  (cleanup will delete them — copy if you want to inspect)")

        print("\nAll learning layer tests pass.")
        return 0
    finally:
        os.chdir(cwd)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
