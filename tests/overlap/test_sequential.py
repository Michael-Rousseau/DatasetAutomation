from collections.abc import Iterator

import numpy as np
import pytest

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.interface import Features, Matches
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.pairwise import PairOverlapEstimate
from dataset_automation.overlap.sequential import (
    ImageSize,
    SequenceImage,
    estimate_sequential_overlaps,
)
from tests.overlap.scenes import HEIGHT, WIDTH, scene, view_pair

SIZE = (WIDTH, HEIGHT)
STEP_PX = 80
EXTRACTOR = SiftExtractor(scale=1.0)


class NoMatcher:
    name = "none"

    def match(self, features_a: Features, features_b: Features) -> Matches:
        return Matches(index_pairs=np.empty((0, 2), dtype=np.int64), scores=np.empty(0))


def placeholder_estimate(
    features_a: Features,
    features_b: Features,
    matches: Matches,
    size_a: ImageSize,
    size_b: ImageSize,
    intrinsics: Intrinsics | None,
) -> PairOverlapEstimate:
    return PairOverlapEstimate("placeholder", 0, 0, 1.0, 1.0, 1.0, None)


def empty_images(count: int) -> Iterator[SequenceImage]:
    features = Features(
        keypoints=np.empty((0, 2)), descriptors=np.empty((0, 128)), scores=np.empty(0)
    )
    for _ in range(count):
        yield SequenceImage(features, SIZE)


def index_pairs(image_count: int, neighbour_count: int) -> list[tuple[int, int]]:
    overlaps = estimate_sequential_overlaps(
        empty_images(image_count),
        neighbour_count,
        NoMatcher(),
        estimate_pair=placeholder_estimate,
    )
    return [(overlap.index_a, overlap.index_b) for overlap in overlaps]


def sliding_camera(image_count: int, world: np.ndarray) -> list[SequenceImage]:
    """A camera moving right by STEP_PX per image over a flat scene."""
    images = []
    for position in range(image_count):
        shift = np.array(
            [[1.0, 0.0, -STEP_PX * position], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        _, image = view_pair(shift, world)
        images.append(SequenceImage(EXTRACTOR.extract(image), SIZE))
    return images


def test_pairs_each_image_with_its_next_neighbours() -> None:
    assert index_pairs(image_count=4, neighbour_count=2) == [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_pairs_all_images_when_the_sequence_is_shorter_than_the_window() -> None:
    assert index_pairs(image_count=3, neighbour_count=10) == [(0, 1), (0, 2), (1, 2)]


def test_single_image_has_no_pair() -> None:
    assert index_pairs(image_count=1, neighbour_count=5) == []


def test_reads_images_only_as_pairs_are_requested() -> None:
    images_read = 0

    def counted_images() -> Iterator[SequenceImage]:
        nonlocal images_read
        for image in empty_images(100):
            images_read += 1
            yield image

    overlaps = estimate_sequential_overlaps(
        counted_images(), 1, NoMatcher(), estimate_pair=placeholder_estimate
    )
    next(overlaps)

    assert images_read == 2


def test_rejects_neighbour_count_below_one_before_reading_images() -> None:
    with pytest.raises(ValueError):
        estimate_sequential_overlaps(empty_images(3), 0, NoMatcher())


def test_overlap_decreases_with_the_gap_as_the_camera_moves() -> None:
    overlaps = estimate_sequential_overlaps(
        sliding_camera(3, scene()), 2, RatioTestMatcher()
    )

    a_in_b = {
        (overlap.index_a, overlap.index_b): overlap.estimate.a_in_b
        for overlap in overlaps
    }
    assert a_in_b[0, 1] == pytest.approx(1 - STEP_PX / WIDTH, abs=0.01)
    assert a_in_b[1, 2] == pytest.approx(1 - STEP_PX / WIDTH, abs=0.01)
    assert a_in_b[0, 2] == pytest.approx(1 - 2 * STEP_PX / WIDTH, abs=0.01)


def test_unrelated_consecutive_images_are_unknown_not_zero() -> None:
    images = [sliding_camera(1, scene(seed))[0] for seed in (0, 1)]

    (overlap,) = estimate_sequential_overlaps(images, 1, RatioTestMatcher())

    assert not overlap.estimate.is_known
    assert overlap.estimate.a_in_b is None
