"""Evaluation: scikit-learn metrics + matplotlib figures.

Per Section 7 of the spec. All plot helpers return ``matplotlib.figure.Figure``
objects so the UI can embed them with ``FigureCanvasQTAgg`` rather than
calling ``plt.show()``. We also use the non-interactive 'Agg' backend in
case this module is ever imported in a context without a display.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # safe in headless contexts; QtAgg overrides per-canvas
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader


log = logging.getLogger("training")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _collect_predictions(
    val_loader: DataLoader,
    model: nn.Module,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the validation set through ``model`` once.

    Returns ``(y_true, y_pred_binary, y_score)`` numpy arrays. ``y_score``
    is the sigmoid output (continuous in [0, 1]); ``y_pred_binary`` is
    the threshold-0.5 classification.
    """
    model.eval()
    y_true_list: list[float] = []
    y_score_list: list[float] = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            scores = model(x, return_logits=False).squeeze(-1).cpu().numpy()
            y_score_list.extend(scores.tolist())
            y_true_list.extend(y.numpy().tolist())
    y_true = np.asarray(y_true_list, dtype=np.int64)
    y_score = np.asarray(y_score_list, dtype=np.float32)
    y_pred = (y_score > 0.5).astype(np.int64)
    return y_true, y_pred, y_score


def compute_metrics(
    val_loader: DataLoader,
    model: nn.Module,
    device: str = "cpu",
) -> dict:
    """Compute accuracy, precision, recall, F1, and confusion matrix.

    Returns:
        ``{accuracy, precision, recall, f1, tp, tn, fp, fn,
           confusion_matrix}``

    Confusion matrix is a numpy array shape ``(2, 2)`` in the sklearn
    convention: rows=true, cols=pred, ordered ``[0, 1]``.
    """
    y_true, y_pred, _ = _collect_predictions(val_loader, model, device)

    if len(y_true) == 0:
        log.warning("compute_metrics: empty val loader")
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
        }

    accuracy = accuracy_score(y_true, y_pred)
    # ``zero_division=0`` turns "0/0" into 0 instead of a warning + nan,
    # which happens when the val set has no positives or no predicted
    # positives — common with tiny early-stage datasets.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Plots — all return Figures, never call plt.show()
# ---------------------------------------------------------------------------

def plot_learning_curve(history: dict) -> Figure:
    """Two-panel figure: loss curves + confusion-matrix counts over epochs.

    Per Section 7:
        Panel 1 — train_loss and val_loss as line charts
        Panel 2 — TP, TN, FP, FN as line charts

    ``history`` must have the lists produced by ``trainer.train``.
    """
    epochs_axis = list(range(1, len(history.get("train_loss", [])) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: losses
    if epochs_axis:
        ax1.plot(epochs_axis, history["train_loss"], label="train loss", marker="o")
        ax1.plot(epochs_axis, history["val_loss"], label="val loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Panel 2: confusion-matrix counts.
    if epochs_axis:
        ax2.plot(epochs_axis, history["tp"], label="TP", marker="o")
        ax2.plot(epochs_axis, history["tn"], label="TN", marker="s")
        ax2.plot(epochs_axis, history["fp"], label="FP", marker="^")
        ax2.plot(epochs_axis, history["fn"], label="FN", marker="v")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Count")
    ax2.set_title("Validation Confusion-Matrix Counts")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    return fig


def plot_roc_curve(
    val_loader: DataLoader,
    model: nn.Module,
    device: str = "cpu",
) -> Figure:
    """Two-panel figure: ROC curve and Precision-Recall curve.

    The spec lists these as separate methods conceptually but groups them
    in one figure so the user sees both diagnostic views together. The UI
    opens this in a popup window.
    """
    y_true, _, y_score = _collect_predictions(val_loader, model, device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC needs both classes to be present in y_true; otherwise the FPR /
    # TPR formulae degenerate. Show an explanatory empty plot in that case
    # rather than a confusing diagonal.
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        ax1.text(0.5, 0.5, "ROC requires both classes\nin validation set",
                 ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "PR requires both classes\nin validation set",
                 ha="center", va="center", transform=ax2.transAxes)
        ax1.set_title("ROC Curve")
        ax2.set_title("Precision-Recall Curve")
        fig.tight_layout()
        return fig

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="chance")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    return fig
