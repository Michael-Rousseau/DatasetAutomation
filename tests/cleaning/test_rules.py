import numpy as np
import pytest

from dataset_automation.cleaning.rules import (
    SurfaceRule,
    SurfaceRuleParameters,
    surface_mask,
)

HEIGHT, WIDTH = 120, 160


def seafloor(seed: int = 0) -> np.ndarray:
    """Textured, mid-grey image: nothing on it looks like the surface."""
    rng = np.random.default_rng(seed)
    grey = rng.integers(40, 160, size=(HEIGHT, WIDTH), dtype=np.uint8)
    return np.dstack([grey, grey, grey])


def test_detects_bright_smooth_band_at_the_top() -> None:
    image = seafloor()
    image[:30] = 250

    mask = surface_mask(image)

    # Exactly the band: the rows next to the seafloor, where the local std is high, included.
    assert mask[:30].all()
    assert not mask[30:].any()


def test_ignores_bright_patch_not_touching_the_top() -> None:
    image = seafloor()
    image[20:50, 60:100] = 250

    assert not surface_mask(image).any()


def test_ignores_bright_but_textured_top() -> None:
    image = seafloor()
    rng = np.random.default_rng(1)
    image[:30] = rng.integers(200, 256, size=(30, WIDTH, 1), dtype=np.uint8)

    assert not surface_mask(image).any()


def test_never_marks_pixels_below_the_top_fraction() -> None:
    image = np.full((HEIGHT, WIDTH, 3), 250, dtype=np.uint8)

    mask = surface_mask(image, SurfaceRuleParameters(top_fraction=0.25))

    assert mask[:30].all()
    assert not mask[30:].any()


def test_rejects_even_window() -> None:
    with pytest.raises(ValueError):
        SurfaceRuleParameters(window=4)


def test_rule_segmenter_returns_only_the_surface_class() -> None:
    rule = SurfaceRule()

    masks = rule.segment(seafloor())

    assert set(masks) == set(rule.classes) == {"surface"}
    assert rule.run_parameters()["min_brightness"] == 200
