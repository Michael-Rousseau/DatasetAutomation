"""Smallest subset of a sequence that keeps a chosen overlap (01_overlap_engine.md, step 5).

Non-destructive: the result is the list of images to keep, nothing is removed.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    image_count: int
    kept_count: int

    @property
    def removed_count(self) -> int:
        return self.image_count - self.kept_count

    @property
    def removed_share(self) -> float:
        return self.removed_count / self.image_count if self.image_count else 0.0


def extract_subset(
    image_count: int,
    overlaps: Mapping[tuple[int, int], float],
    threshold: float,
) -> list[int]:
    """Positions of the images to keep so that consecutive kept images overlap by `threshold`.

    Greedy: keep an image, jump to the last following image that still overlaps it enough,
    keep that one, repeat. `overlaps` maps (earlier position, later position) to the known
    overlaps only, in the measure the caller chose.

    The jump stops at the first pair that is below the threshold, unknown or never compared:
    without evidence the overlap is not assumed. The image right after a kept one is then kept
    whatever its overlap, so the subset never has a gap the full sequence did not have. The
    first and last images are always kept.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if image_count <= 0:
        return []

    kept = [0]
    last_position = image_count - 1
    while kept[-1] < last_position:
        current = kept[-1]
        furthest = current + 1
        for candidate in range(current + 1, image_count):
            overlap = overlaps.get((current, candidate))
            if overlap is None or overlap < threshold:
                break
            furthest = candidate
        kept.append(furthest)
    return kept


def extraction_curve(
    image_count: int,
    overlaps: Mapping[tuple[int, int], float],
    thresholds: Iterable[float],
) -> list[CurvePoint]:
    """How many images each threshold keeps, in the order the thresholds are given.

    No matching is involved: every point reuses the same overlaps.
    """
    return [
        CurvePoint(
            threshold,
            image_count,
            len(extract_subset(image_count, overlaps, threshold)),
        )
        for threshold in thresholds
    ]
