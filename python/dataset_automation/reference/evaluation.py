"""Judge a selection of images against the reference poses.

Two questions, because one alone misleads: what can still be reconstructed, and whether the
kept images still chain together. On Mermaid a random subset keeps as much reconstructible area
as an overlap-based selection, but breaks the sequence in places.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from dataset_automation.reference.footprints import footprint_polygon
from dataset_automation.reference.observability import (
    SeafloorGrid,
    depth_noise_ratio,
    inside,
    observe,
    reconstructible,
)
from dataset_automation.reference.overlap import pair_overlap

# The rim of the reconstructible area is seen by few images, from wide and tilted views where
# the flat-seafloor reference is least reliable: it is reported apart from the inside.
RIM_M = 1.0
WEAK_LINK_OVERLAPS = (0.5, 0.3)


@dataclass(frozen=True)
class SelectionEvaluation:
    """Areas are seafloor areas reconstructible with all the images.

    Depth noise is relative to all the images (1 = unchanged), over the cells the kept images
    still reconstruct; NaN when there is none. `weak_links` maps a mutual overlap to the number
    of consecutive images below it, with all the images and with the kept ones.
    """

    reconstructible_m2: float
    kept_reconstructible_m2: float
    inside_kept_share: float
    rim_kept_share: float
    median_depth_noise: float
    worst_tenth_depth_noise: float
    median_views: tuple[float, float]
    weak_links: dict[float, tuple[int, int]]

    @property
    def kept_share(self) -> float:
        return _share(self.kept_reconstructible_m2, self.reconstructible_m2)


def evaluate_selection(
    grid: SeafloorGrid,
    footprints: np.ndarray,
    camera_centres: np.ndarray,
    seafloor_height_m: float,
    cameras: Sequence[int],
    kept_cameras: Sequence[int],
) -> SelectionEvaluation:
    """`cameras` are the camera indices of all the images in capture order, `kept_cameras`
    those of the selection, in capture order too."""
    everything = observe(grid, footprints, camera_centres, seafloor_height_m, cameras)
    selection = observe(
        grid, footprints, camera_centres, seafloor_height_m, kept_cameras
    )
    area = reconstructible(everything)
    kept_area = reconstructible(selection) & area
    core = inside(area, grid, RIM_M)
    rim = area & ~core

    noise = depth_noise_ratio(everything, selection)[kept_area]
    seen = everything.views > 0
    return SelectionEvaluation(
        reconstructible_m2=float(area.sum() * grid.cell_area_m2),
        kept_reconstructible_m2=float(kept_area.sum() * grid.cell_area_m2),
        inside_kept_share=_share(int((kept_area & core).sum()), int(core.sum())),
        rim_kept_share=_share(int((kept_area & rim).sum()), int(rim.sum())),
        median_depth_noise=float(np.median(noise)) if noise.size else float("nan"),
        worst_tenth_depth_noise=(
            float(np.percentile(noise, 90)) if noise.size else float("nan")
        ),
        median_views=(
            float(np.median(everything.views[seen])),
            float(np.median(selection.views[seen])),
        ),
        weak_links={
            overlap: (
                _weak_link_count(footprints, cameras, overlap),
                _weak_link_count(footprints, kept_cameras, overlap),
            )
            for overlap in WEAK_LINK_OVERLAPS
        },
    )


def _weak_link_count(
    footprints: np.ndarray, cameras: Sequence[int], below: float
) -> int:
    count = 0
    for first, second in pairwise(cameras):
        footprint_a = footprint_polygon(footprints, first)
        footprint_b = footprint_polygon(footprints, second)
        # A camera without a footprint has no reference overlap: the link is not judged.
        if footprint_a is None or footprint_b is None:
            continue
        overlap = pair_overlap(footprint_a, footprint_b)
        count += min(overlap.a_in_b, overlap.b_in_a) < below
    return count


def _share(part: float, whole: float) -> float:
    return float(part / whole) if whole else float("nan")
