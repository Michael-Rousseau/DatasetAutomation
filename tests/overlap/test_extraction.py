import pytest

from dataset_automation.overlap.extraction import extract_subset


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
