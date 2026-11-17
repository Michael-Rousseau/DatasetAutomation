"""Near-duplicate images of a sequence (01_overlap_engine.md, step 4).

Non-destructive: images are flagged, never removed.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NearDuplicate:
    """`index` and `duplicate_of` are positions in capture order."""

    index: int
    duplicate_of: int
    mutual_overlap: float


def mutual_overlap(a_in_b: float, b_in_a: float) -> float:
    """Smaller of the two directional overlaps.

    A close-up contained in a wide view overlaps it by 1.00 one way and 0.10 the other: the
    close-up is not a duplicate, it carries a finer ground resolution.
    """
    return min(a_in_b, b_in_a)


def find_near_duplicates(
    image_count: int,
    mutual_overlaps: Mapping[tuple[int, int], float],
    threshold: float,
) -> list[NearDuplicate]:
    """Flag each image whose mutual overlap with an image already kept reaches `threshold`.

    `mutual_overlaps` maps (earlier position, later position) to the known overlaps only. A
    pair that is unknown or was never compared is absent: without evidence an image is not a
    duplicate, so it is kept. Pairs need not be consecutive: an image is also flagged when it
    duplicates one kept on an earlier survey line or pass.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    earlier_images: dict[int, list[tuple[float, int]]] = {}
    for (earlier, later), overlap in mutual_overlaps.items():
        if earlier >= later:
            raise ValueError(f"pair ({earlier}, {later}) is not (earlier, later)")
        earlier_images.setdefault(later, []).append((overlap, earlier))

    # Only kept images count, never the previous one as such: a slow camera makes every image
    # a duplicate of its predecessor while drifting arbitrarily far from where it started.
    kept: set[int] = set()
    duplicates: list[NearDuplicate] = []
    for index in range(image_count):
        matches = [
            (overlap, earlier)
            for overlap, earlier in earlier_images.get(index, [])
            if earlier in kept and overlap >= threshold
        ]
        if not matches:
            kept.add(index)
            continue
        overlap, most_similar = max(matches)
        duplicates.append(NearDuplicate(index, most_similar, overlap))
    return duplicates
