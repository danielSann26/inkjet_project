"""Entry point for the Inkjet Scaffold Analyzer.

Per Section 17 of the specification. Sets reproducibility seeds (Section 16
rule #8), configures logging (Section 13), initializes the SQLite database,
and launches the PyQt6 main window.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from utils.db import init_db


# Project root is the folder containing this file. All paths in the spec are
# relative to this root (Section 16 rule #3).
PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_runtime_dirs() -> None:
    """Create directories the spec assumes exist before any module writes to them."""
    for sub in (
        "data/raw",
        "data/patches",
        "data/outputs",
        "data/cache",
        "models/checkpoints",
        "logs",
    ):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def _configure_logging() -> None:
    """Two handlers per Section 13: rotating file (DEBUG+) and console (WARNING+)."""
    log_dir = PROJECT_ROOT / "logs"
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Wipe any handlers a parent process may have attached so we don't double-log.
    root.handlers.clear()

    # Rotating file handler for the application logger (logs/app.log).
    app_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(fmt)
    root.addHandler(app_handler)

    # Console: warnings and errors only, so the terminal stays usable.
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Dedicated training logger writes to logs/training.log (Section 13).
    train_log = logging.getLogger("training")
    train_log.setLevel(logging.DEBUG)
    train_log.propagate = False  # don't double-write to app.log
    train_handler = logging.handlers.RotatingFileHandler(
        log_dir / "training.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    train_handler.setLevel(logging.DEBUG)
    train_handler.setFormatter(fmt)
    train_log.addHandler(train_handler)


def _seed_everything(seed: int = 42) -> None:
    """Section 16 rule #8: fixed seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Honor a deterministic env var some libs read.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def main() -> int:
    # Run the working dir from the project root so all relative paths resolve.
    os.chdir(PROJECT_ROOT)

    _ensure_runtime_dirs()
    _configure_logging()
    _seed_everything(42)

    log = logging.getLogger("app")
    log.info("Starting Inkjet Scaffold Analyzer")

    init_db()

    app = QApplication(sys.argv)
    app.setApplicationName("Inkjet Scaffold Analyzer")
    app.setOrganizationName("Inkjet Project")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
