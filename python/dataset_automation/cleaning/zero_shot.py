# pyright: reportMissingImports=false
# torch and transformers come from the optional `cleaning` extra (`uv sync --extra cleaning`),
# which CI does not install; they are imported inside functions so the rest of the package
# stays importable without them.
"""Zero-shot masks: Grounding DINO finds boxes from text prompts, SAM2 turns boxes into masks.

Grounding DINO knows *what* to look for but only draws boxes; SAM2 draws precise outlines but
needs to be told where. Chained, they need no training data. Both have mostly seen terrestrial
images, so their quality underwater is exactly what the evaluation step has to measure.

Each class is queried separately with its own phrasings: every box then belongs to one class
without guessing from the partial phrase Grounding DINO returns ("diver" could come from
"scuba diver" or "diver fin").
"""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from dataset_automation.cleaning.device import move_to, pick_device

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam2.1-hiera-tiny"

# Several phrasings per class, to compare during evaluation (docs/02_cleaning_tracking.md, step 2).
DEFAULT_PROMPTS: dict[str, tuple[str, ...]] = {
    "fin": ("diver fin", "swim fin", "flipper"),
    "diver": ("scuba diver", "person"),
    "robot_part": ("robot arm", "metal frame", "cable"),
    "surface": ("water surface",),
    "sky": ("sky",),
}


def detect_boxes(
    processor,
    model,
    image_rgb: np.ndarray,
    phrases: Sequence[str],
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (K, 4) as (x0, y0, x1, y1) pixels and their scores (K,) for one text query."""
    import torch

    inputs = processor(images=image_rgb, text=list(phrases), return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_rgb.shape[:2]],
    )[0]
    return (
        result["boxes"].cpu().numpy().reshape(-1, 4),
        result["scores"].cpu().numpy().reshape(-1),
    )


def segment_boxes(
    processor, model, image_rgb: np.ndarray, boxes: np.ndarray
) -> np.ndarray:
    """One boolean (H, W) mask per box, as a (K, H, W) array."""
    height, width = image_rgb.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, height, width), dtype=bool)

    import torch

    inputs = processor(
        images=image_rgb, input_boxes=[boxes.tolist()], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        # One mask per box: the multimask variants only help with ambiguous point prompts.
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0]
    return masks[:, 0].numpy().astype(bool)


def drop_oversized_boxes(
    boxes: np.ndarray, image_shape: tuple[int, int], max_area_fraction: float
) -> np.ndarray:
    """Remove boxes covering more than `max_area_fraction` of the image.

    Given a prompt it cannot find, Grounding DINO tends to return the whole frame; masking it
    would make COLMAP and tracking ignore the entire image.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[areas <= max_area_fraction * image_shape[0] * image_shape[1]]


class ZeroShotSegmenter:
    name = "zero-shot"

    def __init__(
        self,
        prompts: Mapping[str, Sequence[str]] = DEFAULT_PROMPTS,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        max_box_area_fraction: float = 0.8,
        detector_id: str = DETECTOR_ID,
        sam_id: str = SAM_ID,
        device: str | None = None,
    ) -> None:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self.prompts = {name: tuple(phrases) for name, phrases in prompts.items()}
        self.classes = tuple(self.prompts)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_box_area_fraction = max_box_area_fraction
        self.detector_id = detector_id
        self.sam_id = sam_id

        target = pick_device(device)
        self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = move_to(
            AutoModelForZeroShotObjectDetection.from_pretrained(detector_id), target
        ).eval()
        self.sam_processor = Sam2Processor.from_pretrained(sam_id)
        self.sam = move_to(Sam2Model.from_pretrained(sam_id), target).eval()

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image_rgb.shape[:2]

        boxes_by_class = {
            name: drop_oversized_boxes(
                detect_boxes(
                    self.detector_processor,
                    self.detector,
                    image_rgb,
                    phrases,
                    self.box_threshold,
                    self.text_threshold,
                )[0],
                shape,
                self.max_box_area_fraction,
            )
            for name, phrases in self.prompts.items()
        }

        # All boxes in one SAM2 call: the image encoder, the costly part, runs once per image.
        all_boxes = np.concatenate(list(boxes_by_class.values()))
        masks = segment_boxes(self.sam_processor, self.sam, image_rgb, all_boxes)

        result = {}
        start = 0
        for name, boxes in boxes_by_class.items():
            result[name] = masks[start : start + len(boxes)].any(axis=0)
            start += len(boxes)
        return result

    def run_parameters(self) -> dict[str, object]:
        return {
            "prompts": {name: list(phrases) for name, phrases in self.prompts.items()},
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "max_box_area_fraction": self.max_box_area_fraction,
            "detector": self.detector_id,
            "sam": self.sam_id,
        }
