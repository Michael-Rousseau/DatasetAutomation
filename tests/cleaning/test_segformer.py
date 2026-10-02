# pyright: reportMissingImports=false
# transformers comes from the optional `cleaning` extra; these tests skip without it.
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import SegformerConfig

from dataset_automation.cleaning.segformer import (
    SegformerSegmenter,
    TrainingParameters,
    build_model,
    model_classes,
    train,
)

CLASSES = ("fin", "diver")


def tiny_config() -> SegformerConfig:
    """A few thousand parameters, randomly initialised: no download, trains in seconds."""
    return SegformerConfig(
        hidden_sizes=[8, 16, 24, 32],
        depths=[1, 1, 1, 1],
        num_attention_heads=[1, 1, 1, 1],
        decoder_hidden_size=16,
    )


def left_half_fin(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Bright left half labelled "fin", dark right half background: trivially learnable."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 40, size=(48, 64, 3), dtype=np.uint8)
    image[:, :32] += 200
    labels = np.zeros((48, 64), dtype=np.uint8)
    labels[:, :32] = 1
    return image, labels


def test_model_head_covers_background_and_classes() -> None:
    model = build_model(CLASSES, config=tiny_config())

    assert model.config.num_labels == 3
    assert model_classes(model) == CLASSES


def test_training_lowers_the_loss() -> None:
    model = build_model(CLASSES, config=tiny_config())
    samples = [left_half_fin(seed) for seed in range(8)]

    losses = train(
        model,
        samples,
        TrainingParameters(epochs=15, batch_size=4, learning_rate=3e-3, image_size=64),
        device="cpu",
    )

    assert len(losses) == 15
    assert losses[-1] < 0.5 * losses[0]


def test_saved_model_segments_at_original_size(tmp_path: Path) -> None:
    model = build_model(CLASSES, config=tiny_config())
    model.save_pretrained(tmp_path / "model")

    segmenter = SegformerSegmenter(tmp_path / "model", image_size=64, device="cpu")
    masks = segmenter.segment(left_half_fin(0)[0])

    assert segmenter.classes == CLASSES
    assert set(masks) == set(CLASSES)
    assert masks["fin"].shape == (48, 64)
