from dataclasses import dataclass

from shapely import Polygon


@dataclass(frozen=True)
class PairOverlap:
    """Overlap of two ground footprints A and B.

    `a_in_b` is the share of A's area also covered by B; it is what feature-based
    estimators (homography of A into B) measure and what deduplication of A needs.
    `iou` is the symmetric view, derivable from the two directional values.
    """

    a_in_b: float
    b_in_a: float
    iou: float


def pair_overlap(footprint_a: Polygon, footprint_b: Polygon) -> PairOverlap:
    if footprint_a.area == 0 or footprint_b.area == 0:
        raise ValueError("footprints must have a non-zero area")

    # Clamped: when one footprint contains the other, shapely's intersection area can exceed
    # the smaller area by ~1e-16, which would read as "more than 100 % covered".
    intersection_area = min(
        footprint_a.intersection(footprint_b).area, footprint_a.area, footprint_b.area
    )
    union_area = footprint_a.area + footprint_b.area - intersection_area
    return PairOverlap(
        a_in_b=intersection_area / footprint_a.area,
        b_in_a=intersection_area / footprint_b.area,
        iou=intersection_area / union_area,
    )
