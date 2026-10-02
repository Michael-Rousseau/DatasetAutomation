"""Step 1: which unwanted elements appear in a dataset, and how often.

    uv sync --extra cleaning
    uv run python scripts/cleaning/inventory.py data/mermaid/images --every 25

Runs a segmenter (zero-shot by default) on every n-th image, prints the presence rate per class,
and writes tinted overlays to outputs/cleaning/inventory/ to check the candidates by eye.
Classes that never really appear are not worth handling (docs/02_cleaning_tracking.md).
"""

import argparse
from pathlib import Path

import cv2

from dataset_automation.cleaning.inventory import every_nth, overlay, presence
from dataset_automation.cleaning.pipeline import image_files
from dataset_automation.cleaning.segmenter import METHODS, load_segmenter

OUTPUT_DIR = Path("outputs/cleaning/inventory")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--every", type=int, default=25, help="keep one image in N")
    parser.add_argument("--method", choices=METHODS, default="zero-shot")
    parser.add_argument("--model", type=Path, help="trained model, for segformer")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    segmenter = load_segmenter(arguments.method, arguments.model, arguments.device)
    paths = every_nth(image_files(arguments.image_root), arguments.every)
    print(f"{len(paths)} images sampled from {arguments.image_root}")

    results = []
    for index, path in enumerate(paths, start=1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"cannot read {path}, skipped")
            continue
        masks = segmenter.segment(image)
        results.append(masks)
        preview = arguments.output / path.relative_to(arguments.image_root)
        preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview.with_suffix(".jpg")), overlay(image, masks))
        print(f"{index}/{len(paths)} {path.name}", end="\r", flush=True)

    print()
    print(f"{'class':<12} {'images':>8} {'rate':>7} {'mean area':>10}")
    for row in presence(results, segmenter.classes).values():
        area = (
            f"{row.mean_area_fraction:.1%}"
            if row.mean_area_fraction is not None
            else "-"
        )
        print(
            f"{row.class_name:<12} {row.images_with_class:>8} "
            f"{row.presence_rate:>7.1%} {area:>10}"
        )
    print(f"overlays in {arguments.output}/")


if __name__ == "__main__":
    main()
