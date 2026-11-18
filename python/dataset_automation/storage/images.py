"""Image files of a folder and their rows in the shared `images` table.

Stopgap until person C's ingestion exists: it will own the `images` table (RAW, ROS bags,
timestamps, metadata). Until then every module registers plain JPEG and PNG files this way.
"""

import sqlite3
from pathlib import Path

from dataset_automation.storage.database import start_run

# Suffix to the `images.source` value.
IMAGE_SUFFIXES = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}


def image_files(root: Path) -> list[Path]:
    """JPEG and PNG files below `root`, any depth, in a stable order."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def register_image_files(
    connection: sqlite3.Connection, image_root: Path, module: str
) -> int:
    """Add the image files below `image_root` missing from `images`; return how many.

    `module` names the caller in the `runs` table.
    """
    paths = [path.resolve() for path in image_files(image_root)]
    run_id = start_run(connection, module, {"image_root": str(image_root.resolve())})
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO images (path, source, run_id) VALUES (?, ?, ?)",
        [(str(path), IMAGE_SUFFIXES[path.suffix.lower()], run_id) for path in paths],
    )
    connection.commit()
    return connection.total_changes - before
