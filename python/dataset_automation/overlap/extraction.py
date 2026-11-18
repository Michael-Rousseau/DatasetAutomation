"""Smallest subset of a sequence that keeps a chosen overlap (01_overlap_engine.md, step 5).

Non-destructive: the result is the list of images to keep, nothing is removed.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionRow:
    """`reference` is the kept image this one is compared with: the one it was skipped for,
    or, for a kept image, the kept image before it. Both are None for the first image."""

    position: int
    is_kept: bool
    reference: int | None
    overlap_with_reference: float | None


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


def selection_rows(
    image_count: int, overlaps: Mapping[tuple[int, int], float], kept: list[int]
) -> list[SelectionRow]:
    """One row per image, saying whether it is kept and what it overlaps the kept image by."""
    kept_positions = set(kept)
    rows: list[SelectionRow] = []
    reference: int | None = None
    for position in range(image_count):
        is_kept = position in kept_positions
        overlap = None if reference is None else overlaps.get((reference, position))
        rows.append(SelectionRow(position, is_kept, reference, overlap))
        if is_kept:
            reference = position
    return rows


@dataclass(frozen=True)
class WeakLinks:
    """Consecutive kept images that do not reach the requested overlap.

    They exist because the image after a kept one is always kept: there the sequence itself
    lacks the overlap, and no selection can restore it.
    """

    below_threshold: int
    unknown: int


def count_weak_links(rows: list[SelectionRow], threshold: float) -> WeakLinks:
    links = [row for row in rows if row.is_kept and row.reference is not None]
    unknown = sum(row.overlap_with_reference is None for row in links)
    below = sum(
        row.overlap_with_reference is not None
        and row.overlap_with_reference < threshold
        for row in links
    )
    return WeakLinks(below_threshold=below, unknown=unknown)
