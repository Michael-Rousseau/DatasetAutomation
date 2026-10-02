# pyright: reportMissingImports=false
# torch and transformers come from the optional `cleaning` extra, which CI does not install.
"""Step 4: fine-tune SegFormer on hand-annotated underwater images, then use it as a segmenter.

SegFormer is a semantic segmentation model: it predicts one label per pixel (background or one
of the classes), which maps directly onto per-class masks. The smallest variant (MiT-b0, 3.7 M
parameters) trains on a few hundred images on a laptop.

Images are resized to a square `image_size` for training and inference. It distorts the aspect
ratio, which the model learns from; predictions are scaled back to the original size.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from dataset_automation.cleaning.device import move_to, pick_device
from dataset_automation.cleaning.labels import BACKGROUND, masks_from_labels

PRETRAINED_ID = "nvidia/mit-b0"
# ImageNet statistics, which the MiT encoder was pre-trained with.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class TrainingParameters:
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 6e-5
    image_size: int = 512
    seed: int = 0


DEFAULT_TRAINING = TrainingParameters()


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 image → RGB uint8 image of (size, size)."""
    return cv2.cvtColor(
        cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
    )


def resize_labels(labels: np.ndarray, size: int) -> np.ndarray:
    # Nearest neighbour: interpolating label ids would invent classes between two others.
    return cv2.resize(labels, (size, size), interpolation=cv2.INTER_NEAREST)


def to_pixel_values(images_rgb: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB → (N, 3, H, W) normalised float tensor."""
    normalised = (images_rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(normalised).permute(0, 3, 1, 2).contiguous()


def build_model(
    classes: tuple[str, ...],
    pretrained: str = PRETRAINED_ID,
    config: SegformerConfig | None = None,
) -> SegformerForSemanticSegmentation:
    """A SegFormer whose head predicts background + `classes`.

    With `config`, the model is randomly initialised from it (tests, experiments); otherwise the
    encoder comes from `pretrained` and only the new head starts from scratch.
    """
    id2label = {BACKGROUND: "background"} | {
        index + 1: name for index, name in enumerate(classes)
    }
    label2id = {name: index for index, name in id2label.items()}
    if config is not None:
        config.num_labels = len(id2label)
        config.id2label = id2label
        config.label2id = label2id
        return SegformerForSemanticSegmentation(config)
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )


def model_classes(model: SegformerForSemanticSegmentation) -> tuple[str, ...]:
    if model.config.id2label is None:
        raise ValueError("model config has no id2label: not a model from build_model")
    id2label = {int(index): name for index, name in model.config.id2label.items()}
    return tuple(id2label[index] for index in sorted(id2label) if index != BACKGROUND)


def train(
    model: SegformerForSemanticSegmentation,
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    parameters: TrainingParameters = DEFAULT_TRAINING,
    device: str | None = None,
) -> list[float]:
    """Fine-tune on (BGR image, label image) pairs; return the mean loss of each epoch."""
    if not samples:
        raise ValueError("no training samples")
    target = pick_device(device)
    generator = np.random.default_rng(parameters.seed)
    torch.manual_seed(parameters.seed)

    # Resized once, kept as uint8: a few hundred 512 px images fit in memory this way.
    images = np.stack(
        [resize_image(image, parameters.image_size) for image, _ in samples]
    )
    labels = np.stack(
        [resize_labels(label, parameters.image_size) for _, label in samples]
    )

    move_to(model, target).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=parameters.learning_rate)
    epoch_losses = []
    for _ in range(parameters.epochs):
        order = generator.permutation(len(samples))
        losses = []
        for start in range(0, len(order), parameters.batch_size):
            batch = order[start : start + parameters.batch_size]
            batch_images, batch_labels = images[batch], labels[batch]
            # Random horizontal flip: fins and divers appear on either side.
            flip = generator.random(len(batch)) < 0.5
            batch_images[flip] = batch_images[flip, :, ::-1]
            batch_labels[flip] = batch_labels[flip, :, ::-1]

            outputs = model(
                pixel_values=to_pixel_values(batch_images).to(target),
                labels=torch.from_numpy(batch_labels.astype(np.int64)).to(target),
            )
            optimiser.zero_grad()
            outputs.loss.backward()
            optimiser.step()
            losses.append(outputs.loss.item())
        epoch_losses.append(float(np.mean(losses)))
    return epoch_losses


def predict_labels(
    model: SegformerForSemanticSegmentation, image: np.ndarray, image_size: int
) -> np.ndarray:
    """Label image of the original size for one BGR image."""
    height, width = image.shape[:2]
    pixel_values = to_pixel_values(resize_image(image, image_size)[None])
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=pixel_values.to(model.device)).logits
        # SegFormer predicts at 1/4 resolution: upsample scores, then pick the best class.
        logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


class SegformerSegmenter:
    name = "segformer"

    def __init__(
        self, model_dir: Path, image_size: int = 512, device: str | None = None
    ) -> None:
        self.model_dir = model_dir
        self.image_size = image_size
        self.model = move_to(
            SegformerForSemanticSegmentation.from_pretrained(model_dir),
            pick_device(device),
        ).eval()
        self.classes = model_classes(self.model)

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return masks_from_labels(
            predict_labels(self.model, image, self.image_size), self.classes
        )

    def run_parameters(self) -> dict[str, object]:
        return {"model": str(self.model_dir.resolve()), "image_size": self.image_size}


def training_run_parameters(
    parameters: TrainingParameters, classes: tuple[str, ...], pretrained: str
) -> dict[str, object]:
    return {**asdict(parameters), "classes": list(classes), "pretrained": pretrained}
