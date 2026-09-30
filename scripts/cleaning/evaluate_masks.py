"""Step 3: IoU per class of a segmenter against hand-annotated images (COCO export).

    uv run python scripts/cleaning/evaluate_masks.py annotations/test.json data/mermaid/images \\
        --method rule
    uv run python scripts/cleaning/evaluate_masks.py annotations/test.json data/mermaid/images \\
        --method zero-shot --category-map "Swim fin=fin"

Annotate 50 to 100 images in CVAT or Label Studio and export them as COCO. Only the classes
both annotated and produced by the segmenter are scored. Results are also saved as JSON in
outputs/cleaning/evaluation/, to compare methods in the report.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dataset_automation.cleaning.annotations import annotated_images, parse_category_map
from dataset_automation.cleaning.evaluation import score_classes
from dataset_automation.cleaning.segmenter import CLASSES, METHODS, load_segmenter
from dataset_automation.storage.database import current_code_version

OUTPUT_DIR = Path("outputs/cleaning/evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("annotations", type=Path, help="COCO JSON export")
    parser.add_argument("image_root", type=Path, help="folder the file names refer to")
    parser.add_argument("--method", choices=METHODS, default="zero-shot")
    parser.add_argument("--model", type=Path, help="trained model, for segformer")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument(
        "--category-map",
        nargs="*",
        default=[],
        metavar="CATEGORY=CLASS",
        help="rename annotation categories to class names",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    segmenter = load_segmenter(arguments.method, arguments.model, arguments.device)
    classes = tuple(name for name in CLASSES if name in segmenter.classes)
    samples = []
    for file_name, image, truth in annotated_images(
        arguments.annotations,
        arguments.image_root,
        classes,
        parse_category_map(arguments.category_map),
    ):
        samples.append((segmenter.segment(image), truth))
        print(f"{len(samples)} {file_name}", end="\r", flush=True)
    print()

    scores = score_classes(samples, classes)
    print(f"{segmenter.name} on {len(samples)} images")
    print(
        f"{'class':<12} {'mean IoU':>9} {'pooled':>7} {'scored':>7} {'false +':>8} {'missed':>7}"
    )
    for score in scores.values():
        mean = f"{score.mean_iou:.3f}" if score.mean_iou is not None else "-"
        pooled = f"{score.pooled_iou:.3f}" if score.pooled_iou is not None else "-"
        print(
            f"{score.class_name:<12} {mean:>9} {pooled:>7} {score.images_scored:>7} "
            f"{score.false_positive_images:>8} {score.missed_images:>7}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_path = arguments.output / f"{segmenter.name}_{stamp}.json"
    result_path.write_text(
        json.dumps(
            {
                "method": segmenter.name,
                "parameters": segmenter.run_parameters(),
                "code_version": current_code_version(),
                "annotations": str(arguments.annotations.resolve()),
                "images": len(samples),
                "scores": [asdict(score) for score in scores.values()],
            },
            indent=2,
        )
    )
    print(f"saved {result_path}")


if __name__ == "__main__":
    main()
