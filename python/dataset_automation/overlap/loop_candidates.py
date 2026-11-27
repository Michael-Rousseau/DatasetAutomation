"""Pairs of far-apart images worth matching: candidates for loop closures.

A camera coming back over a place it has seen (the next survey line, a later pass) produces
images that overlap without being neighbours in time. Matching every pair is out of reach:
1,244 images make 773,000 pairs, about a day of matching. Image resemblance proposes a short
list for each image, and only those pairs are matched.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidatePair:
    """`index_a` < `index_b` are positions in capture order.

    `rank` is the best place the pair holds in either image's list, starting at 0.
    """

    index_a: int
    index_b: int
    similarity: float
    rank: int


def propose_far_pairs(
    image_vectors: np.ndarray, candidates_per_image: int, min_gap: int
) -> list[CandidatePair]:
    """For each image, its `candidates_per_image` most resembling images more than `min_gap`
    positions away, as distinct pairs sorted by position.

    `image_vectors` is (N, D), one unit vector per image in capture order; their dot product
    is the resemblance. Images `min_gap` or fewer apart are left to the sequential mode: they
    resemble most and would fill every list.
    """
    if candidates_per_image < 1:
        raise ValueError(
            f"candidates_per_image must be at least 1, got {candidates_per_image}"
        )
    if min_gap < 0:
        raise ValueError(f"min_gap must be at least 0, got {min_gap}")

    image_count = len(image_vectors)
    similarity = image_vectors @ image_vectors.T
    positions = np.arange(image_count)
    too_close = np.abs(positions[:, None] - positions[None, :]) <= min_gap
    ranked = np.where(too_close, -np.inf, similarity)

    best: dict[tuple[int, int], int] = {}
    for index in range(image_count):
        order = np.argsort(-ranked[index], kind="stable")[:candidates_per_image]
        for rank, other in enumerate(order):
            other = int(other)
            # Fewer far images than requested: the rest of the list is the excluded ones.
            if too_close[index, other]:
                break
            pair = (min(index, other), max(index, other))
            best[pair] = min(rank, best.get(pair, rank))
    return [
        CandidatePair(index_a, index_b, float(similarity[index_a, index_b]), rank)
        for (index_a, index_b), rank in sorted(best.items())
    ]
