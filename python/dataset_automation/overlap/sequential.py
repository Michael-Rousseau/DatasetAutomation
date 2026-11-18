"""Overlap of each image with the ones that follow it (01_overlap_engine.md, step 3).

The cost is linear in the number of images, and only the last `neighbour_count` images are held
in memory: the input can be a generator reading features from disk.
"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.interface import (
    FeatureMatcher,
    Features,
    Matches,
)
from dataset_automation.overlap.displacement import (
    DisplacementEstimate,
    estimate_displacement,
)
from dataset_automation.overlap.pairwise import (
    PairOverlapEstimate,
    estimate_homography_overlap,
)

ImageSize = tuple[int, int]
PairEstimator = Callable[
    [Features, Features, Matches, ImageSize, ImageSize, Intrinsics | None],
    PairOverlapEstimate,
]


@dataclass(frozen=True, eq=False)
class SequenceImage:
    features: Features
    size: ImageSize  # (width, height)


@dataclass(frozen=True)
class SequentialOverlap:
    """`index_a` < `index_b` are positions in capture order, matching the `pairs` table.

    `displacement` is None without a calibration: rays are not defined without it.
    """

    index_a: int
    index_b: int
    estimate: PairOverlapEstimate
    displacement: DisplacementEstimate | None = None


def estimate_sequential_overlaps(
    images: Iterable[SequenceImage],
    neighbour_count: int,
    matcher: FeatureMatcher,
    intrinsics: Intrinsics | None = None,
    estimate_pair: PairEstimator = estimate_homography_overlap,
) -> Iterator[SequentialOverlap]:
    """Yield one overlap per pair (i, i + 1 … i + neighbour_count), grouped by second image.

    `images` must be in capture order. Unknown overlaps are yielded too, with their reason.
    With `intrinsics`, the camera displacement of each pair is estimated from the same matches.
    """
    # Checked here, not in the generator: a generator body only runs at the first `next()`.
    if neighbour_count < 1:
        raise ValueError(f"neighbour_count must be at least 1, got {neighbour_count}")
    return _sequential_overlaps(
        images, neighbour_count, matcher, intrinsics, estimate_pair
    )


def _sequential_overlaps(
    images: Iterable[SequenceImage],
    neighbour_count: int,
    matcher: FeatureMatcher,
    intrinsics: Intrinsics | None,
    estimate_pair: PairEstimator,
) -> Iterator[SequentialOverlap]:
    previous: deque[tuple[int, SequenceImage]] = deque(maxlen=neighbour_count)
    for index_b, image_b in enumerate(images):
        for index_a, image_a in previous:
            matches = matcher.match(image_a.features, image_b.features)
            estimate = estimate_pair(
                image_a.features,
                image_b.features,
                matches,
                image_a.size,
                image_b.size,
                intrinsics,
            )
            displacement = (
                None
                if intrinsics is None
                else estimate_displacement(
                    image_a.features, image_b.features, matches, intrinsics
                )
            )
            yield SequentialOverlap(index_a, index_b, estimate, displacement)
        previous.append((index_b, image_b))
