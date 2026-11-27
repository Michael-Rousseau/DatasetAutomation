"""Confirm loop closure candidates by matching them.

Resemblance only proposes: on Mermaid about half of the proposed pairs do not overlap. Each
candidate goes through the same estimate as two neighbours, with the same answer when the
images cannot be related: unknown, with its reason. A candidate whose overlap is known is a
link between two survey lines or two passes.
"""

from collections.abc import Callable, Iterable, Iterator

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.interface import FeatureMatcher
from dataset_automation.overlap.loop_candidates import CandidatePair
from dataset_automation.overlap.pairwise import estimate_homography_overlap
from dataset_automation.overlap.sequential import (
    MeasuredPair,
    PairEstimator,
    SequenceImage,
    measure_pair,
)

ImageLoader = Callable[[int], SequenceImage]


def verify_candidates(
    candidates: Iterable[CandidatePair],
    image_at: ImageLoader,
    matcher: FeatureMatcher,
    intrinsics: Intrinsics | None = None,
    estimate_pair: PairEstimator = estimate_homography_overlap,
) -> Iterator[MeasuredPair]:
    """Yield one measure per candidate, in the order given, unknown ones included.

    `image_at(position)` returns the image at that position in capture order. The features
    of a dive do not fit in memory at once (about 4 MB per image), so they are asked for one
    image at a time; the first image of a pair is kept while consecutive candidates share it,
    which they do when sorted as `propose_far_pairs` returns them.
    """
    held_index: int | None = None
    held_image: SequenceImage | None = None
    for candidate in candidates:
        if held_image is None or held_index != candidate.index_a:
            held_index, held_image = candidate.index_a, image_at(candidate.index_a)
        yield measure_pair(
            candidate.index_a,
            held_image,
            candidate.index_b,
            image_at(candidate.index_b),
            matcher,
            intrinsics,
            estimate_pair,
        )
