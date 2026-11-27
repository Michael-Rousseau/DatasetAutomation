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
from pathlib import Path

from dataset_automation.cleaning.pipeline import images_under, mask_images
from dataset_automation.cleaning.segmenter import METHODS, load_segmenter
from dataset_automation.storage.database import open_database
from dataset_automation.storage.images import register_image_files

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/cleaning/masks")


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
        added = register_image_files(
            connection, arguments.image_root, "cleaning-register"
        )
        print(f"registered {added} images")

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
