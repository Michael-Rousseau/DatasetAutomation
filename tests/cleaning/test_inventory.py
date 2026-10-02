import numpy as np
import pytest

from dataset_automation.cleaning.inventory import every_nth, overlay, presence


def test_presence_counts_images_above_the_area_threshold() -> None:
    covered = np.zeros((10, 10), dtype=bool)
    covered[:5] = True
    speck = np.zeros((10, 10), dtype=bool)
    speck[0, 0] = True
    results = [{"fin": covered}, {"fin": speck}, {"fin": np.zeros((10, 10), bool)}]

    fin = presence(results, ("fin",), min_area_fraction=0.05)["fin"]

    assert (fin.images_with_class, fin.images_seen) == (1, 3)
    assert fin.presence_rate == pytest.approx(1 / 3)
    assert fin.mean_area_fraction == pytest.approx(0.5)


def test_absent_class_has_no_mean_area() -> None:
    fin = presence([{"fin": np.zeros((2, 2), bool)}], ("fin",))["fin"]

    assert fin.images_with_class == 0
    assert fin.mean_area_fraction is None


def test_every_nth_keeps_evenly_spaced_items() -> None:
    assert every_nth(list(range(10)), 4) == [0, 4, 8]


def test_overlay_tints_only_masked_pixels_and_leaves_input_untouched() -> None:
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    mask = np.array([[True, False]])

    tinted = overlay(image, {"fin": mask}, opacity=0.5)

    assert tinted[0, 0].tolist() == [0, 0, 128]
    assert tinted[0, 1].tolist() == [0, 0, 0]
    assert not image.any()
