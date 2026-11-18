import pytest

from dataset_automation.overlap.extraction import (
    CurvePoint,
    SelectionRow,
    WeakLinks,
    count_weak_links,
    extract_subset,
    extraction_curve,
    selection_rows,
)


def test_skips_the_images_that_still_overlap_the_kept_one() -> None:
    overlaps = {(0, 1): 0.95, (0, 2): 0.85, (0, 3): 0.70, (2, 3): 0.90}

    assert extract_subset(4, overlaps, threshold=0.80) == [0, 2, 3]


def test_keeps_every_image_when_neighbours_overlap_too_little() -> None:
    overlaps = {(0, 1): 0.60, (1, 2): 0.70, (0, 2): 0.30}

    assert extract_subset(3, overlaps, threshold=0.80) == [0, 1, 2]


def test_overlap_is_measured_from_the_kept_image_not_the_previous_one() -> None:
    # Each image overlaps its predecessor by 0.90, but image 2 has moved away from image 0.
    overlaps = {(0, 1): 0.90, (1, 2): 0.90, (0, 2): 0.70, (1, 3): 0.85, (2, 3): 0.90}

    assert extract_subset(4, overlaps, threshold=0.80) == [0, 1, 3]


def test_threshold_itself_is_enough_overlap() -> None:
    assert extract_subset(3, {(0, 1): 0.90, (0, 2): 0.80}, threshold=0.80) == [0, 2]


def test_an_unknown_overlap_stops_the_jump() -> None:
    # (0, 2) is absent: unknown, or never compared. Image 3 is not trusted past it.
    overlaps = {(0, 1): 0.95, (0, 3): 0.90, (1, 2): 0.95, (1, 3): 0.90}

    assert extract_subset(4, overlaps, threshold=0.80) == [0, 1, 3]


def test_keeps_the_next_image_when_nothing_is_known() -> None:
    assert extract_subset(3, {}, threshold=0.80) == [0, 1, 2]


def test_keeps_the_last_image_even_when_it_could_be_skipped() -> None:
    overlaps = {(0, 1): 0.95, (0, 2): 0.90}

    assert extract_subset(3, overlaps, threshold=0.80) == [0, 2]


def test_a_lower_threshold_removes_more_images() -> None:
    overlaps = {
        (0, 1): 0.90,
        (0, 2): 0.75,
        (0, 3): 0.55,
        (1, 2): 0.90,
        (1, 3): 0.75,
        (2, 3): 0.90,
    }

    assert extract_subset(4, overlaps, threshold=0.80) == [0, 1, 2, 3]
    assert extract_subset(4, overlaps, threshold=0.70) == [0, 2, 3]
    assert extract_subset(4, overlaps, threshold=0.50) == [0, 3]


@pytest.mark.parametrize("image_count, kept", [(0, []), (1, [0])])
def test_sequences_too_short_to_extract_from(image_count: int, kept: list[int]) -> None:
    assert extract_subset(image_count, {}, threshold=0.80) == kept


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_rejects_a_threshold_outside_zero_one(threshold: float) -> None:
    with pytest.raises(ValueError):
        extract_subset(2, {(0, 1): 0.9}, threshold)


def test_curve_counts_the_images_kept_at_each_threshold() -> None:
    overlaps = {
        (0, 1): 0.90,
        (0, 2): 0.75,
        (0, 3): 0.55,
        (1, 2): 0.90,
        (1, 3): 0.75,
        (2, 3): 0.90,
    }

    curve = extraction_curve(4, overlaps, [0.80, 0.70, 0.50])

    assert curve == [
        CurvePoint(threshold=0.80, image_count=4, kept_count=4),
        CurvePoint(threshold=0.70, image_count=4, kept_count=3),
        CurvePoint(threshold=0.50, image_count=4, kept_count=2),
    ]
    assert [point.removed_count for point in curve] == [0, 1, 2]
    assert curve[2].removed_share == 0.5


def test_curve_of_an_empty_sequence_removes_nothing() -> None:
    (point,) = extraction_curve(0, {}, [0.80])

    assert point.kept_count == 0
    assert point.removed_share == 0.0


def test_rows_compare_each_image_with_the_kept_image_before_it() -> None:
    overlaps = {(0, 1): 0.95, (0, 2): 0.85, (0, 3): 0.70, (2, 3): 0.90}
    kept = extract_subset(4, overlaps, threshold=0.80)

    assert selection_rows(4, overlaps, kept) == [
        SelectionRow(0, is_kept=True, reference=None, overlap_with_reference=None),
        SelectionRow(1, is_kept=False, reference=0, overlap_with_reference=0.95),
        SelectionRow(2, is_kept=True, reference=0, overlap_with_reference=0.85),
        SelectionRow(3, is_kept=True, reference=2, overlap_with_reference=0.90),
    ]


def test_row_of_an_image_kept_without_a_known_overlap_has_none() -> None:
    rows = selection_rows(2, {}, extract_subset(2, {}, threshold=0.80))

    assert rows[1] == SelectionRow(
        1, is_kept=True, reference=0, overlap_with_reference=None
    )


def test_counts_the_kept_links_the_sequence_cannot_satisfy() -> None:
    # 0 → 1 is too low, 1 → 2 is unknown, 2 → 3 is fine but 3 is the last image.
    overlaps = {(0, 1): 0.40, (2, 3): 0.90}
    rows = selection_rows(4, overlaps, extract_subset(4, overlaps, threshold=0.80))

    assert count_weak_links(rows, threshold=0.80) == WeakLinks(
        below_threshold=1, unknown=1
    )


def test_a_subset_whose_links_all_reach_the_threshold_has_no_weak_link() -> None:
    overlaps = {(0, 1): 0.95, (0, 2): 0.85}
    rows = selection_rows(3, overlaps, extract_subset(3, overlaps, threshold=0.80))

    assert count_weak_links(rows, threshold=0.80) == WeakLinks(0, 0)
