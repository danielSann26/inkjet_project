"""Preprocessing program for the Inkjet Scaffold Analyzer.

A minimal GUI to rotate, crop, and optionally CLAHE-normalize a microscope
image before feeding it to train.py. Independent of main.py and train.py
in terms of code, but shares the pipeline/preprocessor functions.

Workflow:

    1.  python preprocess.py path/to/image.tif
    2.  Image loads in a window.
    3.  Click "Rotate" → click two points along the slide edge →
        image rotates so the edge is horizontal.
    4.  Click "Crop"   → click polygon vertices, double-click to close →
        image is masked & cropped to that region.
    5.  Optional: toggle CLAHE; slider adjusts the clip-limit live.
    6.  Click "Save"   → writes
            data/preprocessed/{original_stem}_preprocessed.png
        and prints the path to the terminal.

What it deliberately does NOT do:
    * No thresholding / bitmap conversion (train.py owns that).
    * No candidate scanning, labeling, training.
    * No database writes — preprocessing is stateless on disk.

Command-line:
    python preprocess.py path/to/image.tif
    python preprocess.py path/to/image.tif --output data/preprocessed/custom.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QStatusBar,
    QToolBar,
    QWidget,
)

from pipeline import preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Drawing modes
# ---------------------------------------------------------------------------

class Mode(Enum):
    NONE = "none"
    ROTATE = "rotate"    # 2 clicks → emit two-point line
    CROP = "crop"        # N clicks, double-click closes → polygon


# ---------------------------------------------------------------------------
# Canvas — image viewer with rotate/crop input
# ---------------------------------------------------------------------------

class PreprocessCanvas(QGraphicsView):
    """Image viewer that accepts two drawing modes.

    Two modes are enough to keep this distinct from train.py's
    ``ClickCanvas`` (which only knows "left-click adds a marker"). The
    rotate mode collects exactly 2 points and then commits. The crop
    mode collects N points and commits on double-click.
    """

    DRAW_COLOR = QColor(0xFF, 0x99, 0x00)
    MIN_ZOOM = 0.10
    MAX_ZOOM = 8.0

    def __init__(self, on_rotate, on_crop):
        super().__init__()
        # Owner provides callbacks rather than us emitting signals — keeps
        # the canvas/owner coupling explicit and avoids a tiny Qt signal
        # ceremony for one consumer each.
        self._on_rotate = on_rotate
        self._on_crop = on_crop

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(0x1a, 0x1a, 0x1a)))
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._mode: Mode = Mode.NONE
        self._drawing_points: list[tuple[float, float]] = []
        # Items that are part of an in-progress drawing — we remove them
        # when the drawing is committed or cancelled.
        self._preview_items: list = []

        # Pan state for middle-mouse-drag
        self._panning = False
        self._pan_start = QPointF()

    # --- public ---------------------------------------------------------

    def set_image(self, arr: np.ndarray) -> None:
        """Display ``arr`` (uint8). Auto-fits to the viewport on load."""
        if arr.ndim == 2:
            h, w = arr.shape
            buf = np.ascontiguousarray(arr)
            qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_Grayscale8)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            buf = np.ascontiguousarray(arr)
            qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
        else:
            raise ValueError(f"PreprocessCanvas: unsupported shape {arr.shape}")
        pixmap = QPixmap.fromImage(qimg.copy())

        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setTransformationMode(
                Qt.TransformationMode.FastTransformation
            )
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)

        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._cancel_drawing()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_mode(self, mode: Mode) -> None:
        self._cancel_drawing()
        self._mode = mode
        if mode in (Mode.ROTATE, Mode.CROP):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        log.debug("canvas mode = %s", mode.value)

    def cancel_drawing(self) -> None:
        """Public hook for the owner to drop an in-progress drawing."""
        self._cancel_drawing()

    # --- drawing internals ---------------------------------------------

    def _cancel_drawing(self) -> None:
        for item in self._preview_items:
            self._scene.removeItem(item)
        self._preview_items.clear()
        self._drawing_points.clear()

    def _add_marker(self, x: float, y: float) -> None:
        r = 4
        item = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        item.setBrush(QBrush(self.DRAW_COLOR))
        item.setPen(QPen(Qt.PenStyle.NoPen))
        item.setZValue(11)
        self._scene.addItem(item)
        self._preview_items.append(item)

    def _add_segment(self, p_prev, p_cur) -> None:
        seg = QGraphicsLineItem(p_prev[0], p_prev[1], p_cur[0], p_cur[1])
        seg.setPen(QPen(self.DRAW_COLOR, 1))
        seg.setZValue(11)
        self._scene.addItem(seg)
        self._preview_items.append(seg)

    # --- mouse handling --------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._pixmap_item is None:
            super().mousePressEvent(event)
            return

        # Middle = pan, regardless of mode.
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pt = self.mapToScene(event.position().toPoint())
        rect = self._pixmap_item.boundingRect()
        if not rect.contains(scene_pt):
            return
        x, y = scene_pt.x(), scene_pt.y()

        if self._mode == Mode.ROTATE:
            self._drawing_points.append((x, y))
            self._add_marker(x, y)
            if len(self._drawing_points) == 2:
                p1, p2 = self._drawing_points
                # Show the committed line briefly before the rotation
                # replaces the image.
                self._add_segment(p1, p2)
                # Hand off to owner. Owner will call set_image again
                # with the rotated array, which clears drawing state.
                self._on_rotate(p1, p2)
            return

        if self._mode == Mode.CROP:
            self._drawing_points.append((x, y))
            self._add_marker(x, y)
            if len(self._drawing_points) >= 2:
                self._add_segment(self._drawing_points[-2], self._drawing_points[-1])
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            # Restore the mode-appropriate cursor.
            self.set_mode(self._mode)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click in CROP mode closes the polygon."""
        if self._mode == Mode.CROP and event.button() == Qt.MouseButton.LeftButton:
            if len(self._drawing_points) >= 3:
                points = list(self._drawing_points)
                # Optionally show the closed polygon as a faint fill before
                # the image is replaced by the cropped version.
                poly = QPolygonF([QPointF(p[0], p[1]) for p in points])
                item = QGraphicsPolygonItem(poly)
                item.setPen(QPen(self.DRAW_COLOR, 2))
                item.setBrush(QBrush(QColor(255, 153, 0, 50)))
                item.setZValue(11)
                self._scene.addItem(item)
                self._preview_items.append(item)
                self._on_crop(points)
                return
        super().mouseDoubleClickEvent(event)

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


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PreprocessWindow(QMainWindow):
    """Minimal: toolbar (rotate / crop / clahe / save), canvas, status bar."""

    def __init__(self, image_path: str, output_path: str | None):
        super().__init__()
        self.setWindowTitle(f"Preprocess — {Path(image_path).name}")
        self.resize(1100, 800)

        self._image_path = image_path
        self._output_path = output_path

        # The "original" stays untouched so users can reset; ``current``
        # is the working copy that gets rotated / cropped. CLAHE is
        # applied as a render-time filter — we recompute it from
        # ``current`` whenever the slider changes, so toggling it on and
        # off is non-destructive.
        self._original: np.ndarray | None = None
        self._current: np.ndarray | None = None
        self._use_clahe: bool = False
        self._clahe_clip: float = 2.0

        self._canvas = PreprocessCanvas(
            on_rotate=self._on_rotate_committed,
            on_crop=self._on_crop_committed,
        )
        self.setCentralWidget(self._canvas)

        self._build_toolbar()
        self._build_status_bar()
        self._load_image()

    def _build_toolbar(self) -> None:
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        self._rotate_action = QAction("Rotate", self)
        self._rotate_action.setCheckable(True)
        self._rotate_action.triggered.connect(self._on_rotate_clicked)
        tb.addAction(self._rotate_action)

        self._crop_action = QAction("Crop", self)
        self._crop_action.setCheckable(True)
        self._crop_action.triggered.connect(self._on_crop_clicked)
        tb.addAction(self._crop_action)

        tb.addSeparator()

        # CLAHE controls. The clip-limit slider operates on a fixed 1-100
        # int range that we map to 0.1-10.0 floats; QSlider doesn't take
        # floats directly. 100 maps to 10.0, 20 maps to 2.0 (the default),
        # 1 maps to 0.1.
        self._clahe_checkbox = QCheckBox("CLAHE")
        self._clahe_checkbox.toggled.connect(self._on_clahe_toggled)
        tb.addWidget(self._clahe_checkbox)

        tb.addWidget(QLabel("  clip:"))
        self._clahe_slider = QSlider(Qt.Orientation.Horizontal)
        self._clahe_slider.setRange(1, 100)
        self._clahe_slider.setValue(20)  # corresponds to 2.0
        self._clahe_slider.setFixedWidth(120)
        self._clahe_slider.valueChanged.connect(self._on_clahe_clip_changed)
        tb.addWidget(self._clahe_slider)
        self._clahe_value_label = QLabel(" 2.0")
        tb.addWidget(self._clahe_value_label)

        tb.addSeparator()

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self._on_reset)
        tb.addAction(reset_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self._on_save)
        tb.addAction(save_action)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "Rotate: click 2 points on the slide edge  |  "
            "Crop: click polygon vertices, double-click to close  |  "
            "Middle-drag: pan  |  Wheel: zoom"
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        print(f"[preprocess] Loading: {self._image_path}")
        try:
            arr = preprocessor.load_tiff(self._image_path)
        except Exception as e:
            QMessageBox.critical(self, "Could not load image", str(e))
            sys.exit(1)
        # Force grayscale up front — the preprocessed output is always
        # single-channel, so we standardize on grayscale from the
        # moment we read the file. This keeps rotate/crop simpler too.
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        h, w = arr.shape[:2]
        print(f"[preprocess] Loaded: {w}×{h} {arr.dtype}")
        self._original = arr.copy()
        self._current = arr.copy()
        self._render()

    def _render(self) -> None:
        """Push the current image (with CLAHE applied if enabled) to the canvas."""
        if self._current is None:
            return
        display = self._current
        if self._use_clahe:
            display = preprocessor.apply_clahe(
                display, clip_limit=self._clahe_clip, tile_grid_size=(8, 8)
            )
        self._canvas.set_image(display)

    def _commit_clahe(self) -> np.ndarray | None:
        """Return the final image that should be saved — applies CLAHE if on.

        We do NOT bake CLAHE into ``self._current`` when it's toggled
        because we want toggling to be reversible. CLAHE is only baked
        in at save time (or when the user rotates/crops, since those
        operations should see the displayed image, not the raw one).
        """
        if self._current is None:
            return None
        if self._use_clahe:
            return preprocessor.apply_clahe(
                self._current, clip_limit=self._clahe_clip, tile_grid_size=(8, 8)
            )
        return self._current

    # ------------------------------------------------------------------
    # Toolbar handlers
    # ------------------------------------------------------------------

    def _on_rotate_clicked(self, checked: bool) -> None:
        if checked:
            self._crop_action.setChecked(False)
            self._canvas.set_mode(Mode.ROTATE)
        else:
            self._canvas.set_mode(Mode.NONE)

    def _on_crop_clicked(self, checked: bool) -> None:
        if checked:
            self._rotate_action.setChecked(False)
            self._canvas.set_mode(Mode.CROP)
        else:
            self._canvas.set_mode(Mode.NONE)

    def _on_clahe_toggled(self, checked: bool) -> None:
        self._use_clahe = checked
        self._render()

    def _on_clahe_clip_changed(self, raw_value: int) -> None:
        # Slider is integer 1-100 → float 0.1-10.0
        self._clahe_clip = raw_value / 10.0
        self._clahe_value_label.setText(f" {self._clahe_clip:.1f}")
        if self._use_clahe:
            self._render()

    def _on_reset(self) -> None:
        if self._original is None:
            return
        self._current = self._original.copy()
        self._use_clahe = False
        self._clahe_checkbox.setChecked(False)
        self._render()
        print("[preprocess] Reset to original")

    def _on_rotate_committed(self, p1: tuple, p2: tuple) -> None:
        if self._current is None:
            return
        # Apply rotation against the *current* image with CLAHE folded in,
        # so that if the user CLAHE-tunes for visibility before drawing
        # the rotation line, the rotation acts on the image they actually
        # saw. We bake CLAHE into ``current`` here so subsequent operations
        # build on the same pixel data.
        base = self._commit_clahe()
        if base is None:
            return
        # Disable CLAHE going forward; it's now part of ``current``.
        self._use_clahe = False
        self._clahe_checkbox.setChecked(False)

        angle = preprocessor.compute_rotation_angle(p1, p2)
        rotated = preprocessor.rotate_image(base, angle)
        self._current = rotated
        print(f"[preprocess] Rotated by {angle:.2f}°  →  shape {rotated.shape}")

        self._rotate_action.setChecked(False)
        self._canvas.set_mode(Mode.NONE)
        self._render()

    def _on_crop_committed(self, points: list) -> None:
        if self._current is None:
            return
        base = self._commit_clahe()
        if base is None:
            return
        self._use_clahe = False
        self._clahe_checkbox.setChecked(False)

        cropped, _mask = preprocessor.crop_polygon(base, points)
        self._current = cropped
        print(f"[preprocess] Cropped to {cropped.shape}")

        self._crop_action.setChecked(False)
        self._canvas.set_mode(Mode.NONE)
        self._render()

    def _on_save(self) -> None:
        if self._current is None:
            return
        final = self._commit_clahe()

        if self._output_path:
            target_path = Path(self._output_path)
        else:
            # Default location: data/preprocessed/{stem}_preprocessed.png
            stem = Path(self._image_path).stem
            target_path = Path("data/preprocessed") / f"{stem}_preprocessed.png"

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # PNG via OpenCV. cv2.imwrite expects BGR or grayscale; we have
        # grayscale, which is fine.
        ok = cv2.imwrite(str(target_path), final)
        if not ok:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not write to {target_path}. "
                "Check the path and your write permissions.",
            )
            return

        print(f"[preprocess] Saved to: {target_path}")
        print(f"[preprocess]    shape: {final.shape}, dtype: {final.dtype}")
        print()
        print("Next step:")
        print(f"   python train.py {target_path}")

        QMessageBox.information(
            self, "Saved",
            f"Wrote {target_path}\n\n"
            f"Now run:\n   python train.py {target_path}",
        )


# ---------------------------------------------------------------------------
# CLI + entry point
# ---------------------------------------------------------------------------

def _pick_image_from_folder() -> str | None:
    """Open a file dialog rooted at the project's images/ folder."""
    images_dir = str(PROJECT_ROOT / "images")
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select image to preprocess",
        images_dir,
        "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)",
    )
    return path or None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess a microscope image (rotate, crop, CLAHE) "
                    "before training. Output is grayscale PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", nargs="?", default=None,
                   help="Path to a TIFF/PNG image to preprocess "
                        "(omit to pick from the images/ folder)")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output path. Default: data/preprocessed/{stem}_preprocessed.png",
    )
    return p.parse_args()


def _ensure_runtime_dirs() -> None:
    for sub in ("data/preprocessed", "logs"):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = _parse_args()
    os.chdir(PROJECT_ROOT)

    _configure_logging()
    _ensure_runtime_dirs()

    if args.image is None:
        app = QApplication.instance() or QApplication(sys.argv)
        args.image = _pick_image_from_folder()
        if not args.image:
            print("error: no image selected", file=sys.stderr)
            return 1

    if not Path(args.image).exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    print()
    print("Inkjet Scaffold Analyzer — Preprocess")
    print("Rotate, crop, and (optionally) CLAHE-normalize an image.")
    print("Save when ready; the output goes to data/preprocessed/.")
    print()

    app = QApplication(sys.argv)
    app.setApplicationName("Inkjet Scaffold Analyzer — Preprocess")
    window = PreprocessWindow(args.image, args.output)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
