"""Step 4: fine-tune SegFormer on hand-annotated images, if zero-shot is not good enough.

    uv sync --extra cleaning
    uv run python scripts/cleaning/train_segformer.py annotations/train.json data/mermaid/images \\
        --classes fin diver --epochs 30

Keep the evaluation images out of the training export, then compare with
scripts/cleaning/evaluate_masks.py --method segformer --model outputs/cleaning/segformer.
The order of --classes is the priority where annotations overlap (first wins).
"""

import argparse
import json
from pathlib import Path

from dataset_automation.cleaning.annotations import annotated_images, parse_category_map
from dataset_automation.cleaning.labels import label_image
from dataset_automation.cleaning.segformer import (
    PRETRAINED_ID,
    TrainingParameters,
    build_model,
    train,
    training_run_parameters,
)
from dataset_automation.cleaning.segmenter import CLASSES
from dataset_automation.storage.database import current_code_version

OUTPUT_DIR = Path("outputs/cleaning/segformer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("annotations", type=Path, help="COCO JSON export")
    parser.add_argument("image_root", type=Path, help="folder the file names refer to")
    parser.add_argument("--classes", nargs="+", default=list(CLASSES))
    parser.add_argument(
        "--category-map", nargs="*", default=[], metavar="CATEGORY=CLASS"
    )
    parser.add_argument("--pretrained", default=PRETRAINED_ID)
    parser.add_argument("--epochs", type=int, default=TrainingParameters.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingParameters.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=TrainingParameters.learning_rate
    )
    parser.add_argument("--image-size", type=int, default=TrainingParameters.image_size)
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    classes = tuple(arguments.classes)
    parameters = TrainingParameters(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        image_size=arguments.image_size,
    )
    samples = [
        (image, label_image(masks, classes))
        for _, image, masks in annotated_images(
            arguments.annotations,
            arguments.image_root,
            classes,
            parse_category_map(arguments.category_map),
        )
    ]
    print(f"training on {len(samples)} images, classes {', '.join(classes)}")

    model = build_model(classes, pretrained=arguments.pretrained)
    losses = train(model, samples, parameters, arguments.device)
    for epoch, loss in enumerate(losses, start=1):
        print(f"epoch {epoch:>3}  loss {loss:.4f}")

    model.save_pretrained(arguments.output)
    # Saved beside the weights, so every model can be traced back to its data and settings.
    (arguments.output / "training.json").write_text(
        json.dumps(
            {
                **training_run_parameters(parameters, classes, arguments.pretrained),
                "annotations": str(arguments.annotations.resolve()),
                "images": len(samples),
                "code_version": current_code_version(),
                "epoch_losses": losses,
            },
            indent=2,
        )
    )
    print(f"model saved in {arguments.output}/")


if __name__ == "__main__":
    main()
