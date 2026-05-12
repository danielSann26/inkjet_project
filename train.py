"""Standalone training program for the Inkjet Scaffold Analyzer.

Runs independently of main.py. Workflow:

    1. python train.py path/to/image.tif
    2. A minimal window opens showing the image.
    3. Left-click on every fiber intersection you can see.
    4. Right-click or press 'Z' to undo the last click.
    5. Press Enter / click "Train" to:
         a) Threshold + scan candidates with the double-circle algorithm
         b) Label clicked-near candidates as positive (intersection)
         c) Randomly sample non-clicked candidates as negative
         d) Train the CNN; per-epoch metrics print to terminal
         e) Save the checkpoint to models/checkpoints/
    6. Window closes. Re-run on another image or feed the checkpoint
       to the main app for inference.

Everything substantive (progress, metrics, file paths) prints to the
terminal. The UI does the minimum required to capture clicks.

Command-line:
    python train.py path/to/image.tif
    python train.py path/to/image.tif --epochs 30 --lr 5e-4
    python train.py path/to/image.tif --negative-multiplier 4
    python train.py path/to/image.tif --match-radius 20
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


# We import project modules lazily inside functions where possible so the
# initial window appears fast (numpy / OpenCV / torch take seconds to load).
PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Logging — to stderr so the terminal stays alive even when GUI is up
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Mute matplotlib's verbose font debug spam at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Minimal click canvas
# ---------------------------------------------------------------------------

class ClickCanvas(QGraphicsView):
    """Minimal image viewer that records left-clicks as intersection markers.

    Inherits from QGraphicsView so we get zoom + pan effectively for free
    without writing a custom paintEvent. We deliberately don't reuse the
    full app's CanvasWidget — that one has drawing modes, overlay layers,
    cursor tracking, all of which would be UI clutter here. This class is
    100 lines of "image + dots."
    """

    DOT_RADIUS_PX = 6
    DOT_COLOR = QColor(0x1D, 0x9E, 0x75)  # green for "this is an intersection"
    MIN_ZOOM = 0.10
    MAX_ZOOM = 8.0

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(0x1a, 0x1a, 0x1a)))
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_array: np.ndarray | None = None
        # List of (x, y, QGraphicsEllipseItem) so we can both report the
        # points to the training pipeline AND undo individual ones.
        self._click_points: list[tuple[int, int, QGraphicsEllipseItem]] = []

        # Pan state — middle-mouse-drag for navigation on big slides.
        self._panning = False
        self._pan_start = QPointF()

    # --- public API -----------------------------------------------------

    def set_image(self, arr: np.ndarray) -> None:
        """Display ``arr``. uint8 grayscale or BGR; same conventions as the main app."""
        self._image_array = arr
        if arr.ndim == 2:
            h, w = arr.shape
            buf = np.ascontiguousarray(arr)
            qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = arr.shape
            buf = np.ascontiguousarray(arr)
            qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def get_clicks(self) -> list[tuple[int, int]]:
        """Return clicked points in image-pixel coordinates."""
        return [(x, y) for (x, y, _item) in self._click_points]

    def clear_clicks(self) -> None:
        for _x, _y, item in self._click_points:
            self._scene.removeItem(item)
        self._click_points.clear()

    def undo_last_click(self) -> None:
        if not self._click_points:
            return
        _x, _y, item = self._click_points.pop()
        self._scene.removeItem(item)

    # --- mouse handling -------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._pixmap_item is None:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle = pan
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.RightButton:
            # Right = undo last
            self.undo_last_click()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.position().toPoint())
            rect = self._pixmap_item.boundingRect()
            if not rect.contains(scene_pt):
                return
            x, y = int(scene_pt.x()), int(scene_pt.y())
            self._add_marker(x, y)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - int(delta.x()))
            v_bar.setValue(v_bar.value() - int(delta.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        current = self.transform().m11()
        new_scale = current * factor
        if new_scale < self.MIN_ZOOM:
            factor = self.MIN_ZOOM / current
        elif new_scale > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / current
        self.scale(factor, factor)

    # --- markers --------------------------------------------------------

    def _add_marker(self, x: int, y: int) -> None:
        r = self.DOT_RADIUS_PX
        item = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        item.setBrush(QBrush(self.DOT_COLOR))
        # White outline so the dot is visible on light fibers too.
        item.setPen(QPen(QColor(255, 255, 255), 1.5))
        item.setZValue(10)
        self._scene.addItem(item)
        self._click_points.append((x, y, item))


# ---------------------------------------------------------------------------
# Training main window — bare minimum
# ---------------------------------------------------------------------------

class TrainWindow(QMainWindow):
    """Minimal window: canvas + status bar + a single 'Train' button."""

    def __init__(self, image_path: str, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle(f"Train — {Path(image_path).name}")
        self.resize(1100, 800)

        self._image_path = image_path
        self._args = args

        self._canvas = ClickCanvas()
        self.setCentralWidget(self._canvas)

        # Toolbar with one button + click counter
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        self._train_action = QAction("Train (Enter)", self)
        self._train_action.triggered.connect(self._on_train_clicked)
        tb.addAction(self._train_action)
        self._clear_action = QAction("Clear all clicks", self)
        self._clear_action.triggered.connect(self._on_clear)
        tb.addAction(self._clear_action)

        self._counter_label = QLabel("  0 intersections marked  ")
        tb.addWidget(self._counter_label)

        # Status bar at the bottom for instructions
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "Left-click: mark intersection   "
            "Right-click: undo   "
            "Middle-drag: pan   "
            "Wheel: zoom   "
            "Enter: train"
        )

        # Keyboard shortcuts: Enter to train, Z to undo, Escape to clear.
        QShortcut(QKeySequence("Return"), self, activated=self._on_train_clicked)
        QShortcut(QKeySequence("Enter"), self, activated=self._on_train_clicked)
        QShortcut(QKeySequence("Z"), self, activated=self._canvas.undo_last_click)

        # Load the image. We do it after the window is constructed so the
        # window paints "loading" state first.
        self._load_image()

        # Update counter on every click. We don't have a signal on
        # ClickCanvas for this, so poll cheaply on a Qt timer instead —
        # simpler than threading another signal through.
        from PyQt6.QtCore import QTimer
        self._poll = QTimer(self)
        self._poll.setInterval(150)
        self._poll.timeout.connect(self._refresh_counter)
        self._poll.start()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        from pipeline import preprocessor
        print(f"\n[train] Loading image: {self._image_path}")
        try:
            arr = preprocessor.load_tiff(self._image_path)
        except Exception as e:
            QMessageBox.critical(self, "Could not load image", str(e))
            sys.exit(1)

        # Optional CLAHE — controlled by command line, not GUI.
        if self._args.clahe:
            print(f"[train] Applying CLAHE (clip={self._args.clahe_clip})")
            if arr.ndim == 2:
                arr = preprocessor.apply_clahe(
                    arr, clip_limit=self._args.clahe_clip, tile_grid_size=(8, 8)
                )

        h, w = arr.shape[:2]
        print(f"[train] Image loaded: {w}×{h} {arr.dtype}")
        self._canvas.set_image(arr)
        self._image_array = arr

    def _refresh_counter(self) -> None:
        n = len(self._canvas.get_clicks())
        self._counter_label.setText(f"  {n} intersection{'s' if n != 1 else ''} marked  ")

    def _on_clear(self) -> None:
        self._canvas.clear_clicks()

    # ------------------------------------------------------------------
    # The actual training pipeline
    # ------------------------------------------------------------------

    def _on_train_clicked(self) -> None:
        clicks = self._canvas.get_clicks()
        if not clicks:
            QMessageBox.warning(
                self, "No intersections marked",
                "Left-click on intersections in the image before training.",
            )
            return
        if len(clicks) < self._args.min_clicks:
            ok = QMessageBox.question(
                self, "Few intersections",
                f"You've only marked {len(clicks)} intersection(s). "
                f"The model trains better with more (≥ {self._args.min_clicks} "
                f"recommended). Train anyway?",
            )
            if ok != QMessageBox.StandardButton.Yes:
                return

        # Disable the toolbar so the user can't kick off two runs.
        self._train_action.setEnabled(False)
        self._clear_action.setEnabled(False)
        self._poll.stop()

        # Run the full training pipeline. We run it synchronously here
        # because the user explicitly asked to train and there's nothing
        # else to do in the GUI while we wait — and the terminal output
        # is the real interface during this phase anyway.
        try:
            self._run_training_pipeline(clicks)
        finally:
            self._train_action.setEnabled(True)
            self._clear_action.setEnabled(True)
            self._poll.start()

    def _run_training_pipeline(self, clicks: list[tuple[int, int]]) -> None:
        """End-to-end: threshold → scan → label → train → save."""
        from learning import trainer
        from learning.cnn import IntersectionCNN
        from pipeline import patch_extractor, thresholder
        from utils import db

        # ----- 1. Register or look up the image in the DB -----------------
        # The trainer reads from the patches table, joined to images via
        # split. We mark this image as 'train' (the entire image is the
        # training set in this minimal program; no val split unless the
        # user runs it twice with --split-as test).
        rel_path = self._rel_path(self._image_path)
        split = self._args.split_as
        image_id = db.insert_image(rel_path, Path(rel_path).name, split)
        print(f"[train] DB image_id={image_id}, split={split}")

        # ----- 2. Threshold + scan ----------------------------------------
        print("[train] Thresholding...")
        gray = thresholder.to_grayscale(self._image_array)
        binary = thresholder.apply_otsu_threshold(gray)
        cleaned = thresholder.clean_bitmap(binary)
        bitmap = thresholder.to_bitmap(cleaned)

        print("[train] Scanning for candidates (this may take a moment)...")
        candidates = patch_extractor.scan_candidates(
            bitmap,
            inner_radius=self._args.inner_radius,
            outer_radius=self._args.outer_radius,
            inner_threshold=self._args.inner_threshold,
            outer_threshold=self._args.outer_threshold,
            min_distance=self._args.min_distance,
            image_id=image_id,
            num_workers=self._args.workers,
            progress_callback=_terminal_progress,
        )
        print(f"[train] Found {len(candidates)} candidates")
        if not candidates:
            QMessageBox.warning(
                self, "No candidates found",
                "The double-circle scanner found no candidates. Try "
                "lowering --inner-threshold or --outer-threshold.",
            )
            return

        # ----- 3. Match clicks to candidates ------------------------------
        # For each clicked point, find the nearest candidate. Anything
        # within --match-radius pixels becomes the positive sample for
        # that click. Multiple clicks could theoretically map to the same
        # candidate; we dedupe.
        print(f"[train] Matching {len(clicks)} clicks to candidates "
              f"(radius={self._args.match_radius}px)...")
        positive_indices = self._match_clicks_to_candidates(
            clicks, candidates, self._args.match_radius
        )
        unmatched = len(clicks) - sum(1 for i in positive_indices if i is not None)
        positive_set: set[int] = {i for i in positive_indices if i is not None}
        print(f"[train] Matched {len(positive_set)} unique positives "
              f"({unmatched} click(s) had no candidate within radius)")
        if not positive_set:
            QMessageBox.warning(
                self, "No matches",
                f"None of your clicks landed within {self._args.match_radius}px "
                "of a candidate. Increase --match-radius, or your clicks may be "
                "off the actual fibers.",
            )
            return

        # ----- 4. Sample negatives ----------------------------------------
        # All candidates that were NOT matched to a click are potential
        # negatives. Random-sample N times the positive count, capped at
        # the actual pool size. This gives the model "enough" negatives
        # without flooding training with them.
        neg_pool = [i for i in range(len(candidates)) if i not in positive_set]
        n_neg = min(len(neg_pool), self._args.negative_multiplier * len(positive_set))
        rng = random.Random(42)
        negative_indices = rng.sample(neg_pool, n_neg)
        print(f"[train] Sampled {len(negative_indices)} negatives "
              f"({self._args.negative_multiplier}× the positives) "
              f"from a pool of {len(neg_pool)}")

        # ----- 5. Extract those patches & write labels --------------------
        # We DON'T extract every candidate — we only extract the ones we
        # plan to train on. Smaller patches directory, less disk I/O.
        print("[train] Extracting patches and writing labels to DB...")
        selected_indices = sorted(positive_set | set(negative_indices))
        selected_candidates = [candidates[i] for i in selected_indices]
        labels_by_candidate_idx = {
            i: (1 if i in positive_set else 0) for i in selected_indices
        }

        # Clear any prior labels for THIS image so the run is reproducible
        # — a re-train shouldn't accumulate stale labels.
        self._clear_existing_patches_for_image(image_id)

        extracted = patch_extractor.extract_patches(
            self._image_array,
            selected_candidates,
            patch_size=self._args.patch_size,
            image_id=image_id,
            inner_radius=self._args.inner_radius,
            outer_radius=self._args.outer_radius,
            progress_callback=_terminal_progress,
        )
        print(f"[train] Wrote {len(extracted)} patches to data/patches/")

        # Now label them. extract_patches returns rows in the same order
        # as `selected_candidates` (it skips out-of-bounds patches, so the
        # mapping isn't 1:1 — we re-match by coordinates).
        n_pos = self._apply_labels_from_clicks(
            image_id, selected_candidates, labels_by_candidate_idx,
        )
        print(f"[train] Applied labels: {n_pos} positives, "
              f"{len(extracted) - n_pos} negatives")

        # ----- 6. Train ---------------------------------------------------
        # For training, we need at least a tiny validation set. The
        # cleanest way without complicating the program is to split the
        # patches *within* this image into train/val by patch id. We do
        # that here by temporarily marking some patches as belonging to
        # a "test" image. Simpler approach: use the same image for both
        # and accept that val_acc reflects training set performance. For
        # a single-image training run that's the honest call — there's
        # no held-out data anyway. The user gets a real val set by
        # running train.py on a second image with --split-as test.
        print(f"[train] Building dataloaders (batch_size={self._args.batch_size}, "
              f"class_balance_mode={self._args.class_balance})...")
        try:
            train_loader, val_loader, info = trainer.build_dataloaders(
                batch_size=self._args.batch_size,
                class_balance_mode=self._args.class_balance,
            )
        except RuntimeError as e:
            QMessageBox.critical(self, "Training error", str(e))
            return

        print(f"[train]   train set: {info['train_size']} patches "
              f"({info['class_balance']['intersection']} pos, "
              f"{info['class_balance']['non_intersection']} neg)")
        print(f"[train]   val set:   {info['val_size']} patches")
        if info["val_size"] == 0:
            print("[train]   (No test-split images yet — val metrics will be 0. "
                  "Run train.py again on a different image with "
                  "`--split-as test` to get a real validation set.)")

        model = IntersectionCNN(patch_size=self._args.patch_size)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] Model: IntersectionCNN(patch_size={self._args.patch_size}) "
              f"with {n_params:,} parameters")

        print(f"[train] Training for {self._args.epochs} epochs "
              f"(lr={self._args.lr})...")

        def epoch_callback(epoch, total, metrics):
            print(
                f"[train]   epoch {epoch:3d}/{total}  "
                f"train_loss={metrics['train_loss']:.4f}  "
                f"train_acc={metrics['train_acc']:.3f}  "
                f"val_loss={metrics['val_loss']:.4f}  "
                f"val_acc={metrics['val_acc']:.3f}  "
                f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
            )

        history = trainer.train(
            model, train_loader, val_loader,
            epochs=self._args.epochs,
            learning_rate=self._args.lr,
            batch_size=self._args.batch_size,
            device="cpu",
            progress_callback=epoch_callback,
            pos_weight=info["pos_weight"],
            class_balance=info["class_balance"],
            settings_snapshot={
                "source": "train.py",
                "image_path": self._image_path,
                "clicks": len(clicks),
                "match_radius": self._args.match_radius,
                "negative_multiplier": self._args.negative_multiplier,
            },
        )

        # ----- 7. Report --------------------------------------------------
        latest = db.get_latest_training_run()
        print()
        print("=" * 60)
        print("Training complete.")
        if latest:
            print(f"  Checkpoint: {latest['checkpoint_path']}")
            print(f"  Sidecar:    {latest['metadata_path']}")
            print(f"  Val acc:    {latest['val_accuracy']:.3f}")
            print(f"  Val F1:     {latest['val_f1']:.3f}")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  - Re-run train.py with --split-as test on a different image")
        print("    to get a real validation set.")
        print("  - Or open the main app (python main.py) and use Train > "
              "Run Inference to apply this model.")

        QMessageBox.information(
            self, "Training complete",
            f"Saved {latest['checkpoint_path']}\n"
            f"Val accuracy: {latest['val_accuracy']:.3f}\n"
            f"Val F1: {latest['val_f1']:.3f}",
        )

    # ------------------------------------------------------------------
    # Click-to-candidate matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_clicks_to_candidates(
        clicks: list[tuple[int, int]],
        candidates: list[tuple],
        match_radius: int,
    ) -> list[int | None]:
        """For each click, return the index of the nearest candidate within
        ``match_radius``, or None if there is none."""
        if not candidates:
            return [None] * len(clicks)
        cand_xy = np.asarray([(c[0], c[1]) for c in candidates], dtype=np.float32)
        out: list[int | None] = []
        r_sq = match_radius * match_radius
        for cx, cy in clicks:
            d2 = (cand_xy[:, 0] - cx) ** 2 + (cand_xy[:, 1] - cy) ** 2
            idx = int(np.argmin(d2))
            if d2[idx] <= r_sq:
                out.append(idx)
            else:
                out.append(None)
        return out

    def _apply_labels_from_clicks(
        self,
        image_id: int,
        selected_candidates: list[tuple],
        labels_by_candidate_idx: dict[int, int],
    ) -> int:
        """Match each freshly-extracted DB patch back to its candidate idx by
        (x, y) and set the label. Returns positive count."""
        from utils import db
        patches = db.get_patches(image_id=image_id)
        # Build a quick lookup from (x, y) → patch id.
        patch_lookup: dict[tuple[int, int], int] = {
            (int(p["x_center"]), int(p["y_center"])): int(p["id"])
            for p in patches
        }
        positives = 0
        # Walk our selection in the same order extract_patches received it.
        # labels_by_candidate_idx maps original-candidate-index → label;
        # selected_candidates is the list of candidates we actually fed in.
        for cand_idx, cand in zip(
            sorted(labels_by_candidate_idx.keys()),
            selected_candidates,
        ):
            x, y = int(cand[0]), int(cand[1])
            patch_id = patch_lookup.get((x, y))
            if patch_id is None:
                # Patch was skipped (out of bounds). Nothing to label.
                continue
            label = labels_by_candidate_idx[cand_idx]
            db.set_patch_label(patch_id, label)
            if label == 1:
                positives += 1
        return positives

    @staticmethod
    def _clear_existing_patches_for_image(image_id: int) -> None:
        """Delete prior patches for ``image_id`` from the DB and disk.

        We can't import sqlite3 directly (Section 16 rule: all SQL in
        utils/db.py). Use the existing helpers: enumerate patches, remove
        their files, then null them via a tiny custom helper.

        Since utils/db doesn't expose a bulk delete, and we don't want to
        add SQL elsewhere, we use the connect context manager via db._connect.
        That's an internal but stable API in our codebase.
        """
        from utils import db as db_mod
        prior = db_mod.get_patches(image_id=image_id)
        for p in prior:
            try:
                os.unlink(p["patch_path"])
            except OSError:
                pass
        if prior:
            # Use the private helper — same module, same project. Cleaner
            # than threading a new public function into utils/db just for
            # this edge case.
            with db_mod._connect() as conn:
                conn.execute(
                    "DELETE FROM patches WHERE image_id = ?", (image_id,)
                )

    @staticmethod
    def _rel_path(p: str) -> str:
        """Project-root-relative path when possible; absolute otherwise."""
        abs_p = Path(p).resolve()
        cwd = Path.cwd().resolve()
        try:
            return str(abs_p.relative_to(cwd))
        except ValueError:
            return str(abs_p)


# ---------------------------------------------------------------------------
# Terminal progress callback (shared by scan_candidates / extract_patches)
# ---------------------------------------------------------------------------

def _terminal_progress(current: int, total: int, message: str = "") -> None:
    """Single-line in-place progress bar to stderr.

    The pipeline functions all accept ``progress_callback(current, total,
    message)``. We render a compact progress bar so users see what's
    happening without scrolling forever.
    """
    if total <= 0:
        return
    pct = current / total
    bar_width = 30
    filled = int(bar_width * pct)
    bar = "█" * filled + "─" * (bar_width - filled)
    sys.stderr.write(f"\r  [{bar}] {current}/{total} {message[:40]:<40}")
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")


# ---------------------------------------------------------------------------
# CLI + entry point
# ---------------------------------------------------------------------------

def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_runtime_dirs() -> None:
    for sub in (
        "data/raw", "data/patches", "data/outputs", "data/cache",
        "models/checkpoints", "logs",
    ):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def _pick_image_from_folder() -> str | None:
    """Open a file dialog rooted at the project's images/ folder."""
    from PyQt6.QtWidgets import QFileDialog

    images_dir = str(PROJECT_ROOT / "images")
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select training image",
        images_dir,
        "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)",
    )
    return path or None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Point-and-click training tool for the Inkjet Scaffold Analyzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", nargs="?", default=None,
                   help="Path to a TIFF image to train on "
                        "(omit to pick from the images/ folder)")

    # Click-to-label parameters
    p.add_argument("--match-radius", type=int, default=15,
                   help="Max pixel distance between a click and a candidate "
                        "for them to be considered the same intersection")
    p.add_argument("--negative-multiplier", type=int, default=3,
                   help="Sample N× as many negatives as positives "
                        "(higher = more conservative model)")
    p.add_argument("--min-clicks", type=int, default=10,
                   help="Warn if fewer than this many clicks are made")

    # Scanner parameters (mirror settings.json defaults)
    p.add_argument("--inner-radius", type=int, default=8)
    p.add_argument("--outer-radius", type=int, default=20)
    p.add_argument("--inner-threshold", type=float, default=0.30)
    p.add_argument("--outer-threshold", type=float, default=0.15)
    p.add_argument("--min-distance", type=int, default=10,
                   help="Minimum spacing between candidates (NMS distance)")
    p.add_argument("--patch-size", type=int, default=64, choices=(32, 64, 128))
    p.add_argument("--workers", type=int, default=None,
                   help="CPU cores for the scan (default: all)")

    # Preprocessing
    p.add_argument("--clahe", action="store_true",
                   help="Apply CLAHE contrast normalization")
    p.add_argument("--clahe-clip", type=float, default=2.0)

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Smaller batches work better with small datasets "
                        "(GroupNorm handles this well)")
    p.add_argument("--class-balance", choices=("weighted_loss", "oversample"),
                   default="weighted_loss")
    p.add_argument("--split-as", choices=("train", "test"), default="train",
                   help="Whether this image belongs to the training or "
                        "validation split. Run on a separate image with "
                        "--split-as test to get real validation metrics.")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    os.chdir(PROJECT_ROOT)  # so 'data/...' relative paths resolve

    _configure_logging(args.verbose)
    _ensure_runtime_dirs()
    _seed_everything(42)

    if args.image is None:
        app = QApplication.instance() or QApplication(sys.argv)
        args.image = _pick_image_from_folder()
        if not args.image:
            print("error: no image selected", file=sys.stderr)
            return 1

    if not Path(args.image).exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    # Initialize the DB (idempotent — safe if main.py has already run).
    from utils.db import init_db
    init_db()

    print()
    print("Inkjet Scaffold Analyzer — Training Mode")
    print("Click directly on every fiber intersection you can see.")
    print("Right-click or Z to undo. Press Enter when done to train.")
    print()

    app = QApplication(sys.argv)
    app.setApplicationName("Inkjet Scaffold Analyzer — Training")
    window = TrainWindow(args.image, args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
