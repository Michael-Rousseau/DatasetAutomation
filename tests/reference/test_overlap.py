import pytest
from shapely import Polygon, box

from dataset_automation.reference.overlap import pair_overlap


def test_identical_footprints_overlap_fully() -> None:
    overlap = pair_overlap(box(0, 0, 2, 1), box(0, 0, 2, 1))

    assert (overlap.a_in_b, overlap.b_in_a, overlap.iou) == (1.0, 1.0, 1.0)


def test_disjoint_footprints_do_not_overlap() -> None:
    overlap = pair_overlap(box(0, 0, 1, 1), box(5, 5, 6, 6))

    assert (overlap.a_in_b, overlap.b_in_a, overlap.iou) == (0.0, 0.0, 0.0)


def test_small_footprint_inside_large_one_is_fully_covered_but_iou_is_low() -> None:
    small, large = box(1, 1, 2, 2), box(0, 0, 4, 4)

    overlap = pair_overlap(small, large)

    assert overlap.a_in_b == pytest.approx(1.0)
    assert overlap.b_in_a == pytest.approx(1 / 16)
    assert overlap.iou == pytest.approx(1 / 16)


def test_half_shifted_equal_squares_share_a_third_of_their_union() -> None:
    overlap = pair_overlap(box(0, 0, 2, 2), box(1, 0, 3, 2))

    assert overlap.a_in_b == pytest.approx(0.5)
    assert overlap.b_in_a == pytest.approx(0.5)
    assert overlap.iou == pytest.approx(1 / 3)


def test_swapping_footprints_swaps_directional_overlaps() -> None:
    a, b = box(0, 0, 2, 2), box(1, 1, 5, 5)

    forward, backward = pair_overlap(a, b), pair_overlap(b, a)

    assert (forward.a_in_b, forward.b_in_a) == (backward.b_in_a, backward.a_in_b)
    assert forward.iou == backward.iou


def test_rejects_degenerate_footprint() -> None:
    flat = Polygon([(0, 0), (1, 0), (2, 0)])

    with pytest.raises(ValueError, match="non-zero area"):
        pair_overlap(flat, box(0, 0, 1, 1))
