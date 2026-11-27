# pyright: reportMissingImports=false
# transformers comes from the optional `cleaning` extra; these tests skip without it.
import os

import numpy as np
import pytest

from dataset_automation.cleaning.zero_shot import drop_oversized_boxes, segment_boxes

# Downloads Grounding DINO and SAM2 weights (~850 MB) on first run.
MODEL_TESTS = os.environ.get("DATASET_AUTOMATION_MODEL_TESTS") == "1"


def test_drops_boxes_covering_most_of_the_image() -> None:
    boxes = np.array([[0, 0, 100, 100], [10, 10, 30, 30]], dtype=np.float32)

    kept = drop_oversized_boxes(boxes, (100, 100), max_area_fraction=0.8)

    assert kept.tolist() == [[10, 10, 30, 30]]


def test_segment_boxes_returns_one_full_size_mask_per_box() -> None:
    pytest.importorskip("torchvision")
    from transformers import Sam2Config, Sam2Model, Sam2Processor
    from transformers.models.sam2.image_processing_sam2 import Sam2ImageProcessor

    # Random weights: this checks shapes and plumbing, not mask quality.
    processor = Sam2Processor(image_processor=Sam2ImageProcessor())
    model = Sam2Model(Sam2Config()).eval()
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    boxes = np.array([[5, 5, 40, 30], [20, 10, 70, 50]], dtype=np.float32)

    masks = segment_boxes(processor, model, image, boxes)

    assert masks.shape == (2, 60, 80)
    assert masks.dtype == bool


def test_segment_boxes_without_boxes_returns_no_mask() -> None:
    masks = segment_boxes(None, None, np.zeros((6, 8, 3), np.uint8), np.zeros((0, 4)))

    assert masks.shape == (0, 6, 8)


@pytest.mark.skipif(not MODEL_TESTS, reason="set DATASET_AUTOMATION_MODEL_TESTS=1")
def test_pretrained_models_return_every_class_at_image_size() -> None:
    from dataset_automation.cleaning.zero_shot import ZeroShotSegmenter

    segmenter = ZeroShotSegmenter()
    image = np.full((120, 160, 3), (120, 90, 20), dtype=np.uint8)

    masks = segmenter.segment(image)

    assert set(masks) == set(segmenter.classes)
    assert all(mask.shape == (120, 160) for mask in masks.values())
