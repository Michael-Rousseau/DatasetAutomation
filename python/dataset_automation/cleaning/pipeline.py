"""Run a segmenter over images and record the masks (files + `masks` rows), traceably.

Layout under `output_root/run_<id>/`:
- `colmap/`: one merged mask per image, every unwanted class ignored. Pass this folder to
  COLMAP as `--ImageReader.mask_path`, with `image_root` as `--image_path`.
- `<class>/`: one mask per image where that class was found, for analysis and evaluation.

In the `masks` table, the merged mask has `class_name = 'all'` and is written for every image,
even when nothing was found; per-class rows exist only where the class was found. Since the
run's parameters list the classes searched, a missing class row means "searched, not found".
"""

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from dataset_automation.cleaning.masks import colmap_mask_path, keep_mask, write_mask
from dataset_automation.cleaning.segmenter import Segmenter
from dataset_automation.storage.database import RunId, start_run

MODULE = "cleaning"
COMBINED = "all"
IMAGE_SUFFIXES = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageToMask:
    image_id: int
    path: Path


@dataclass(frozen=True)
class MaskingReport:
    run_id: RunId
    masked: int
    unreadable: list[Path] = field(default_factory=list)


def image_files(root: Path) -> list[Path]:
    """JPEG and PNG files below `root`, any depth, in a stable order."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def images_under(connection: sqlite3.Connection, image_root: Path) -> list[ImageToMask]:
    """Image files registered in the `images` table that live under `image_root`.

    Only JPEG and PNG files are returned: ROS bag frames are not files COLMAP can pair a mask
    with, and RAW images need their own reader.
    """
    root = image_root.resolve()
    rows = connection.execute(
        "SELECT id, path FROM images WHERE frame_index = 0 AND source IN ('jpeg', 'png') "
        "ORDER BY id"
    )
    return [
        ImageToMask(image_id, Path(path))
        for image_id, path in rows
        if Path(path).resolve().is_relative_to(root)
    ]


def mask_images(
    connection: sqlite3.Connection,
    segmenter: Segmenter,
    images: Iterable[ImageToMask],
    image_root: Path,
    output_root: Path,
) -> MaskingReport:
    run_id = start_run(
        connection,
        MODULE,
        {
            "method": segmenter.name,
            "classes": list(segmenter.classes),
            **segmenter.run_parameters(),
        },
    )
    run_root = output_root.resolve() / f"run_{run_id}"
    root = image_root.resolve()
    masked = 0
    unreadable: list[Path] = []

    for image in images:
        pixels = cv2.imread(str(image.path))
        # A corrupted file must never stop a long run: log it and move on.
        if pixels is None:
            logger.warning("cannot read %s, skipped", image.path)
            unreadable.append(image.path)
            continue

        unwanted = segmenter.segment(pixels)
        if set(unwanted) != set(segmenter.classes):
            raise ValueError(
                f"{segmenter.name} returned classes {sorted(unwanted)}, "
                f"expected {sorted(segmenter.classes)}"
            )
        shape = pixels.shape[:2]
        path = image.path.resolve()

        rows: list[tuple[str, Path]] = []
        for name in segmenter.classes:
            if unwanted[name].any():
                class_path = colmap_mask_path(run_root / name, path, root)
                write_mask(class_path, keep_mask([unwanted[name]], shape))
                rows.append((name, class_path))
        combined_path = colmap_mask_path(run_root / "colmap", path, root)
        write_mask(combined_path, keep_mask(unwanted.values(), shape))
        rows.append((COMBINED, combined_path))

        connection.executemany(
            "INSERT INTO masks (image_id, class_name, mask_path, method, run_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (image.image_id, name, str(mask_path), segmenter.name, run_id)
                for name, mask_path in rows
            ],
        )
        # One commit per image: an interrupted run keeps every mask already recorded.
        connection.commit()
        masked += 1

    return MaskingReport(run_id=run_id, masked=masked, unreadable=unreadable)
