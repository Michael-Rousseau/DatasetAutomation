import pytest

from dataset_automation.overlap.deduplication import (
    NearDuplicate,
    find_near_duplicates,
    mutual_overlap,
)


def test_flags_an_image_nearly_identical_to_the_one_before() -> None:
    duplicates = find_near_duplicates(2, {(0, 1): 0.97}, threshold=0.95)

    assert duplicates == [NearDuplicate(index=1, duplicate_of=0, mutual_overlap=0.97)]


def test_keeps_an_image_below_the_threshold() -> None:
    assert find_near_duplicates(2, {(0, 1): 0.94}, threshold=0.95) == []


def test_threshold_itself_is_a_duplicate() -> None:
    duplicates = find_near_duplicates(2, {(0, 1): 0.95}, threshold=0.95)

    assert [duplicate.index for duplicate in duplicates] == [1]


def test_flags_several_images_as_duplicates_of_the_same_kept_image() -> None:
    duplicates = find_near_duplicates(
        3, {(0, 1): 0.99, (0, 2): 0.96, (1, 2): 0.99}, threshold=0.95
    )

    assert [(d.index, d.duplicate_of) for d in duplicates] == [(1, 0), (2, 0)]


def test_slow_drift_is_measured_from_a_kept_image() -> None:
    # Each image duplicates its predecessor, but image 2 has moved away from image 0.
    overlaps = {(0, 1): 0.97, (1, 2): 0.97, (0, 2): 0.90, (2, 3): 0.97}

    duplicates = find_near_duplicates(4, overlaps, threshold=0.95)

    assert [(d.index, d.duplicate_of) for d in duplicates] == [(1, 0), (3, 2)]


def test_keeps_an_image_whose_overlap_with_the_kept_image_is_unknown() -> None:
    # (0, 1) is absent: unknown, or never compared.
    duplicates = find_near_duplicates(3, {(1, 2): 0.99}, threshold=0.95)

    assert [(d.index, d.duplicate_of) for d in duplicates] == [(2, 1)]


def test_keeps_an_image_beyond_the_window_compared_with_the_kept_image() -> None:
    # Two neighbours per image: image 3 was never compared with image 0.
    overlaps = {(0, 1): 0.99, (0, 2): 0.99, (1, 2): 0.99, (1, 3): 0.99, (2, 3): 0.99}

    duplicates = find_near_duplicates(4, overlaps, threshold=0.95)

    assert [d.index for d in duplicates] == [1, 2]


def test_flags_a_duplicate_of_an_image_kept_much_earlier() -> None:
    # A survey grid: image 40 comes back over image 3, on the next line.
    duplicates = find_near_duplicates(41, {(3, 40): 0.97}, threshold=0.95)

    assert duplicates == [NearDuplicate(index=40, duplicate_of=3, mutual_overlap=0.97)]


def test_reports_the_most_similar_kept_image() -> None:
    overlaps = {(0, 1): 0.50, (0, 2): 0.96, (1, 2): 0.98}

    duplicates = find_near_duplicates(3, overlaps, threshold=0.95)

    assert duplicates == [NearDuplicate(index=2, duplicate_of=1, mutual_overlap=0.98)]


def test_rejects_a_pair_not_ordered_earlier_then_later() -> None:
    with pytest.raises(ValueError):
        find_near_duplicates(2, {(1, 0): 0.97}, threshold=0.95)


def test_close_up_inside_a_wide_view_is_not_a_duplicate() -> None:
    overlap = mutual_overlap(a_in_b=1.0, b_in_a=0.10)

    assert overlap == 0.10
    assert find_near_duplicates(2, {(0, 1): overlap}, threshold=0.95) == []


@pytest.mark.parametrize("image_count", [0, 1])
def test_fewer_than_two_images_have_no_duplicate(image_count: int) -> None:
    assert find_near_duplicates(image_count, {}, threshold=0.95) == []


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_rejects_a_threshold_outside_zero_one(threshold: float) -> None:
    with pytest.raises(ValueError):
        find_near_duplicates(2, {(0, 1): 0.97}, threshold)
