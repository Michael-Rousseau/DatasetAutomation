"""Produce masks for every image under a folder and record them in the shared database.

    uv run python scripts/cleaning/make_masks.py data/mermaid/images --method rule
    uv sync --extra cleaning
    uv run python scripts/cleaning/make_masks.py data/mermaid/images --method zero-shot
    uv run python scripts/cleaning/make_masks.py data/mermaid/images --method segformer \\
        --model outputs/cleaning/segformer

Images must be in the `images` table. Until person C's ingestion exists, `--register` adds the
JPEG and PNG files found under the folder. The merged masks for COLMAP land in
`<output>/run_<id>/colmap/`.
"""

import argparse
import sqlite3
from pathlib import Path

from dataset_automation.cleaning.pipeline import (
    IMAGE_SUFFIXES,
    image_files,
    images_under,
    mask_images,
)
from dataset_automation.cleaning.segmenter import METHODS, load_segmenter
from dataset_automation.storage.database import open_database, start_run

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/cleaning/masks")


def register_images(connection: sqlite3.Connection, image_root: Path) -> int:
    """Stopgap until ingestion exists: add the image files not already in `images`."""
    paths = [path.resolve() for path in image_files(image_root)]
    run_id = start_run(
        connection, "cleaning-register", {"image_root": str(image_root.resolve())}
    )
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO images (path, source, run_id) VALUES (?, ?, ?)",
        [(str(path), IMAGE_SUFFIXES[path.suffix.lower()], run_id) for path in paths],
    )
    connection.commit()
    return connection.total_changes - before


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--method", choices=METHODS, default="rule")
    parser.add_argument("--model", type=Path, help="trained model, for segformer")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--register",
        action="store_true",
        help="add image files missing from the images table first",
    )
    arguments = parser.parse_args()

    arguments.database.parent.mkdir(parents=True, exist_ok=True)
    connection = open_database(arguments.database)
    if arguments.register:
        print(f"registered {register_images(connection, arguments.image_root)} images")

    images = images_under(connection, arguments.image_root)
    if not images:
        raise SystemExit(
            f"no image under {arguments.image_root} in {arguments.database}: "
            "run ingestion first, or pass --register"
        )

    segmenter = load_segmenter(arguments.method, arguments.model, arguments.device)
    print(f"masking {len(images)} images with {segmenter.name}")
    report = mask_images(
        connection, segmenter, images, arguments.image_root, arguments.output
    )
    print(
        f"run {report.run_id}: {report.masked} images masked, "
        f"{len(report.unreadable)} unreadable"
    )
    print(
        f"COLMAP masks: {arguments.output.resolve() / f'run_{report.run_id}' / 'colmap'}"
    )


if __name__ == "__main__":
    main()
