"""SQLite access layer.

**All SQL in this project lives in this file** (Section 16 rule). Other
modules import these helpers; they never write raw SQL. This keeps the
schema in one place and makes migrations / tests tractable.

Schema is defined in Section 5 of the specification. We open a fresh
connection per call (``with sqlite3.connect(...)``); SQLite handles this
efficiently and it sidesteps the cross-thread connection-sharing rules,
which matters because pipeline workers run inside QThreads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


log = logging.getLogger("app")

# Database path is relative to the project root. ``main.py`` chdirs to the
# project root at startup, so this resolves correctly without a hardcoded
# absolute path (Section 16 rule #3).
DB_PATH = "data/annotations.db"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a configured connection inside a transaction.

    - ``Row`` factory so we can return list-of-dicts.
    - Foreign keys are off in SQLite by default; we enable them.
    - The ``with`` block on the connection commits on success and rolls back
      on exception, which is exactly the semantics we want.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

# DDL is kept as module-level strings so it's easy to inspect and to add
# future migrations next to the originals.
_DDL_IMAGES = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath    TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    split       TEXT NOT NULL CHECK(split IN ('train', 'test')),
    processed   INTEGER DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_DDL_PATCHES = """
CREATE TABLE IF NOT EXISTS patches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id),
    patch_path      TEXT NOT NULL,
    x_center        INTEGER NOT NULL,
    y_center        INTEGER NOT NULL,
    inner_radius    INTEGER NOT NULL,
    outer_radius    INTEGER NOT NULL,
    inner_density   REAL,
    outer_density   REAL,
    label           INTEGER DEFAULT NULL CHECK(label IS NULL OR label IN (0, 1)),
    prediction      REAL DEFAULT NULL,
    is_golden       INTEGER DEFAULT 0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_DDL_TRAINING_RUNS = """
CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    epochs          INTEGER,
    learning_rate   REAL,
    batch_size      INTEGER,
    class_balance   TEXT,
    dataset_size    INTEGER,
    train_accuracy  REAL,
    val_accuracy    REAL,
    val_f1          REAL,
    checkpoint_path TEXT,
    metadata_path   TEXT
)
"""

# Indexes that the queries below benefit from. Created idempotently.
_DDL_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_patches_image_id ON patches(image_id)",
    "CREATE INDEX IF NOT EXISTS idx_patches_label    ON patches(label)",
    "CREATE INDEX IF NOT EXISTS idx_images_split     ON images(split)",
)


def init_db() -> None:
    """Create all tables and indexes if they don't already exist."""
    with _connect() as conn:
        conn.execute(_DDL_IMAGES)
        conn.execute(_DDL_PATCHES)
        conn.execute(_DDL_TRAINING_RUNS)
        for stmt in _DDL_INDEXES:
            conn.execute(stmt)
    log.debug("init_db: schema ensured at %s", DB_PATH)


# ---------------------------------------------------------------------------
# images table
# ---------------------------------------------------------------------------

def insert_image(filepath: str, filename: str, split: str) -> int:
    """Insert an image row. Idempotent on ``filepath`` (UNIQUE).

    Returns the row id of the newly inserted image, or the existing id if
    ``filepath`` was already present. ``loader.load_folder`` relies on this
    to skip duplicates on re-scan.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO images (filepath, filename, split) VALUES (?, ?, ?)",
            (filepath, filename, split),
        )
        if cur.lastrowid:
            return cur.lastrowid
        # The row already existed; look up its id.
        existing = conn.execute(
            "SELECT id FROM images WHERE filepath = ?", (filepath,)
        ).fetchone()
        return int(existing["id"])


def get_images(split: str | None = None) -> list[dict]:
    """Return image rows, optionally filtered by ``split``.

    Each dict carries ``id, filepath, filename, split, processed, created_at``.
    """
    with _connect() as conn:
        if split is None:
            rows = conn.execute("SELECT * FROM images ORDER BY id ASC").fetchall()
        else:
            if split not in ("train", "test"):
                raise ValueError(f"split must be 'train' or 'test', got {split!r}")
            rows = conn.execute(
                "SELECT * FROM images WHERE split = ? ORDER BY id ASC", (split,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_image_by_id(image_id: int) -> dict | None:
    """Single-row lookup by id. Returns ``None`` if not found."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return _row_to_dict(row)


def mark_image_processed(image_id: int) -> None:
    """Set ``processed=1`` after the bitmap has been cached for this image."""
    with _connect() as conn:
        conn.execute("UPDATE images SET processed = 1 WHERE id = ?", (image_id,))


# ---------------------------------------------------------------------------
# patches table
# ---------------------------------------------------------------------------

def insert_patch(
    image_id: int,
    patch_path: str,
    x_center: int,
    y_center: int,
    inner_radius: int,
    outer_radius: int,
    inner_density: float | None = None,
    outer_density: float | None = None,
) -> int:
    """Insert a single patch row. Returns the new patch id.

    ``label``, ``prediction``, and ``is_golden`` start at their schema
    defaults (NULL, NULL, 0). They are mutated later by setter helpers.
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO patches
                (image_id, patch_path, x_center, y_center,
                 inner_radius, outer_radius, inner_density, outer_density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                patch_path,
                int(x_center),
                int(y_center),
                int(inner_radius),
                int(outer_radius),
                inner_density,
                outer_density,
            ),
        )
        return int(cur.lastrowid)


def insert_patches_bulk(rows: list[dict]) -> None:
    """Bulk-insert patches in a single transaction.

    ``rows`` is a list of dicts with the same keys as ``insert_patch``'s
    parameters. ``patch_extractor.extract_patches`` calls this once per
    image to avoid one transaction per patch.
    """
    if not rows:
        return
    payload = [
        (
            int(r["image_id"]),
            r["patch_path"],
            int(r["x_center"]),
            int(r["y_center"]),
            int(r["inner_radius"]),
            int(r["outer_radius"]),
            r.get("inner_density"),
            r.get("outer_density"),
        )
        for r in rows
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO patches
                (image_id, patch_path, x_center, y_center,
                 inner_radius, outer_radius, inner_density, outer_density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def get_patches(
    image_id: int | None = None,
    labeled_only: bool = False,
    unlabeled_only: bool = False,
) -> list[dict]:
    """Return patches, optionally filtered.

    ``labeled_only`` and ``unlabeled_only`` are mutually exclusive; passing
    both is a programming error and raises ``ValueError``.
    """
    if labeled_only and unlabeled_only:
        raise ValueError("labeled_only and unlabeled_only cannot both be True")

    where_clauses: list[str] = []
    params: list[Any] = []
    if image_id is not None:
        where_clauses.append("image_id = ?")
        params.append(image_id)
    if labeled_only:
        where_clauses.append("label IS NOT NULL")
    if unlabeled_only:
        where_clauses.append("label IS NULL")

    sql = "SELECT * FROM patches"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " ORDER BY id ASC"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def get_patch_by_id(patch_id: int) -> dict | None:
    """Single-row lookup by id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM patches WHERE id = ?", (patch_id,)
        ).fetchone()
        return _row_to_dict(row)


def set_patch_label(patch_id: int, label: int) -> None:
    """Set the ``label`` column for a patch. Must be 0 or 1."""
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    with _connect() as conn:
        conn.execute(
            "UPDATE patches SET label = ? WHERE id = ?", (int(label), patch_id)
        )


def undo_last_label(image_id: int) -> int | None:
    """Revert the most recent label for ``image_id`` back to NULL.

    Note on "most recent": the schema in Section 5 has no ``labeled_at``
    column, so we use the highest-id labeled patch as a pragmatic proxy.
    For the human-in-the-loop UX (one label at a time, sequential), this is
    correct in every realistic case: the just-labeled patch is the one with
    the largest id whose label is set. If we ever need true "label time"
    semantics (e.g. relabeling out of order), we'd add ``labeled_at`` and
    order by it instead.

    Returns the patch id that was reverted, or ``None`` if there was nothing
    to undo for this image.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM patches
            WHERE image_id = ? AND label IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (image_id,),
        ).fetchone()
        if row is None:
            return None
        patch_id = int(row["id"])
        conn.execute("UPDATE patches SET label = NULL WHERE id = ?", (patch_id,))
        return patch_id


def set_patch_prediction(patch_id: int, prediction: float) -> None:
    """Store a sigmoid-output confidence in [0, 1]."""
    p = float(prediction)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"prediction must be in [0, 1], got {p}")
    with _connect() as conn:
        conn.execute(
            "UPDATE patches SET prediction = ? WHERE id = ?", (p, patch_id)
        )


def set_patch_predictions_bulk(updates: list[tuple[int, float]]) -> None:
    """Bulk-update predictions: list of ``(patch_id, prediction)`` pairs.

    Used by ``run_inference_all`` so we don't open one transaction per patch
    when running across an entire image.
    """
    if not updates:
        return
    payload = [(float(p), int(pid)) for pid, p in updates]
    with _connect() as conn:
        conn.executemany(
            "UPDATE patches SET prediction = ? WHERE id = ?", payload
        )


def get_labeled_patch_counts() -> dict:
    """Return ``{total, intersections, non_intersections}`` across the DB.

    Used by the annotation panel's status footer.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                  AS total,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS intersections,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS non_intersections
            FROM patches
            WHERE label IS NOT NULL
            """
        ).fetchone()
        # SUM over an empty set returns NULL in SQLite; coerce to 0.
        return {
            "total": int(row["total"] or 0),
            "intersections": int(row["intersections"] or 0),
            "non_intersections": int(row["non_intersections"] or 0),
        }


def get_labeled_patch_count_for_image(image_id: int) -> int:
    """Count of labeled patches for one image — used by the active-learning
    burn-in check and the per-image progress in the left panel."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM patches WHERE image_id = ? AND label IS NOT NULL",
            (image_id,),
        ).fetchone()
        return int(row["n"])


def get_patch_count_for_image(image_id: int) -> int:
    """Total patches for an image, regardless of label state."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM patches WHERE image_id = ?", (image_id,)
        ).fetchone()
        return int(row["n"])


# ---------------------------------------------------------------------------
# Joins for the trainer (image-level split enforcement)
# ---------------------------------------------------------------------------

def get_labeled_patches_with_split(split: str) -> list[dict]:
    """Return all labeled patches whose source image is in ``split``.

    This is the join the trainer relies on to enforce Section 16 rule #1
    (image-level train/test split): patches from a given image only ever
    appear in one split because the image's row dictates it.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT p.*
            FROM patches AS p
            JOIN images  AS i ON p.image_id = i.id
            WHERE p.label IS NOT NULL AND i.split = ?
            ORDER BY p.id ASC
            """,
            (split,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

def mark_golden_set(patch_ids: list[int]) -> None:
    """Flag the given patches as golden-set members (``is_golden = 1``)."""
    if not patch_ids:
        return
    with _connect() as conn:
        conn.executemany(
            "UPDATE patches SET is_golden = 1 WHERE id = ?",
            [(int(pid),) for pid in patch_ids],
        )


def unmark_golden_set(patch_ids: list[int]) -> None:
    """Inverse of ``mark_golden_set`` — clears the flag."""
    if not patch_ids:
        return
    with _connect() as conn:
        conn.executemany(
            "UPDATE patches SET is_golden = 0 WHERE id = ?",
            [(int(pid),) for pid in patch_ids],
        )


def get_golden_set_patches() -> list[dict]:
    """Return all patches with ``is_golden = 1`` for consistency testing."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM patches WHERE is_golden = 1 ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# training_runs table
# ---------------------------------------------------------------------------

# Allowed kwargs for ``insert_training_run``. Defined as a tuple so we can
# both validate and build the SQL programmatically without a long if-chain.
_TRAINING_RUN_FIELDS = (
    "epochs",
    "learning_rate",
    "batch_size",
    "class_balance",
    "dataset_size",
    "train_accuracy",
    "val_accuracy",
    "val_f1",
    "checkpoint_path",
    "metadata_path",
)


def insert_training_run(**kwargs: Any) -> int:
    """Insert a training-run row. Returns the new run id.

    Accepts any subset of the columns listed in ``_TRAINING_RUN_FIELDS``;
    omitted columns get their schema defaults. ``class_balance`` may be
    passed as a dict and will be JSON-encoded before storage (the schema
    column is ``TEXT``).
    """
    unknown = set(kwargs) - set(_TRAINING_RUN_FIELDS)
    if unknown:
        raise ValueError(f"Unknown training_runs columns: {sorted(unknown)}")

    payload = dict(kwargs)
    if isinstance(payload.get("class_balance"), dict):
        payload["class_balance"] = json.dumps(payload["class_balance"])

    cols = list(payload.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    values = [payload[c] for c in cols]

    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO training_runs ({col_list}) VALUES ({placeholders})",
            values,
        )
        return int(cur.lastrowid)


def get_training_runs() -> list[dict]:
    """Return all training runs, most recent first.

    The metrics panel shows the latest run; older rows are useful for
    historical comparison.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_training_run() -> dict | None:
    """Convenience: most recent training run, or ``None`` if none exist."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _row_to_dict(row)
