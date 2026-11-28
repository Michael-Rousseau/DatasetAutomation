import pytest

from dataset_automation.overlap.support_selection import (
    Pair,
    select_supported_subset,
)


def line(image_count: int, reach: int, overlap: float = 0.8) -> dict[Pair, float]:
    """Images along a line, each overlapping the `reach` images before and after it."""
    return {
        (first, second): overlap
        for first in range(image_count)
        for second in range(first + 1, min(first + reach, image_count - 1) + 1)
    }


def views(image: int, kept: list[int], overlaps: dict[Pair, float]) -> int:
    return sum(
        other == image or (min(image, other), max(image, other)) in overlaps
        for other in kept
    )


def test_every_image_stays_seen_by_enough_kept_images() -> None:
    overlaps = line(12, reach=3)

    kept = select_supported_subset(12, overlaps, min_views=3)

    assert len(kept) < 12
    assert all(views(image, kept, overlaps) >= 3 for image in range(12))


def test_asking_for_more_views_than_the_images_have_keeps_them_all() -> None:
    assert select_supported_subset(8, line(8, reach=2), min_views=6) == list(range(8))


def test_image_that_nothing_overlaps_is_kept() -> None:
    assert select_supported_subset(3, {}, min_views=1) == [0, 1, 2]


def test_overlap_too_low_to_show_the_same_place_does_not_count() -> None:
    overlaps = line(8, reach=3, overlap=0.4)

    assert select_supported_subset(8, overlaps, min_views=2) == list(range(8))


def test_first_and_last_images_are_kept() -> None:
    kept = select_supported_subset(10, line(10, reach=4), min_views=2)

    assert kept[0] == 0
    assert kept[-1] == 9


def test_kept_images_around_a_removed_one_still_overlap() -> None:
    # Image 2 overlaps its neighbours, but images 1 and 3 do not overlap each other.
    overlaps = {
        (0, 1): 0.8,
        (1, 2): 0.8,
        (2, 3): 0.8,
        (3, 4): 0.8,
        (0, 2): 0.8,
        (2, 4): 0.8,
    }

    kept = select_supported_subset(5, overlaps, min_views=2)

    assert 2 in kept


def test_image_seen_from_another_line_can_be_removed() -> None:
    # A line of 5 images, each overlapping only the next one: nothing can go.
    alone = {(0, 1): 0.8, (1, 2): 0.8, (2, 3): 0.8, (3, 4): 0.8, (1, 3): 0.4}
    assert select_supported_subset(8, alone, min_views=3) == list(range(8))

    # Another line, images 5 to 7, sees images 1, 2 and 3 as well.
    other_line = {(5, 6): 0.8, (6, 7): 0.8, (5, 7): 0.8}
    loops = {(1, 6): 0.7, (2, 6): 0.7, (2, 7): 0.7, (3, 6): 0.7}

    assert 2 not in select_supported_subset(8, alone | other_line | loops, min_views=3)


def test_two_views_must_be_far_enough_apart_when_angles_are_known() -> None:
    everyone = {
        (first, second): 0.8 for first in range(6) for second in range(first + 1, 6)
    }
    angles = dict.fromkeys(everyone, 1.0) | {(1, 4): 20.0}

    kept = select_supported_subset(6, everyone, min_views=2, view_angles=angles)

    # Images 1 and 4 are the only two views far apart: removing either leaves none.
    assert kept == [0, 1, 4, 5]


def test_nothing_is_removed_when_no_two_views_are_far_apart() -> None:
    everyone = {
        (first, second): 0.8 for first in range(6) for second in range(first + 1, 6)
    }
    angles = dict.fromkeys(everyone, 1.0)

    kept = select_supported_subset(6, everyone, min_views=2, view_angles=angles)

    assert kept == list(range(6))


def test_no_image_selects_nothing() -> None:
    assert select_supported_subset(0, {}, min_views=3) == []


def test_rejects_a_number_of_views_below_one() -> None:
    with pytest.raises(ValueError):
        select_supported_subset(5, line(5, reach=2), min_views=0)


def test_protected_images_are_kept() -> None:
    overlaps = line(12, reach=3)
    freely = select_supported_subset(12, overlaps, min_views=3)
    removed = sorted(set(range(12)) - set(freely))

    kept = select_supported_subset(12, overlaps, min_views=3, protected=removed)

    assert set(removed) <= set(kept)
