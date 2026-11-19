"""Revisits according to the poses: far-apart images that see the same place.

The ground truth for loop closures. "Their footprints touch" is true for three quarters of
the far pairs of Mermaid and says nothing; a revisit is defined by how much the two images
share and described by the angle between the two views.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely import STRtree

from dataset_automation.reference.footprints import footprint_polygon

# Each image sees at least half of the other: the strictest level at which nearly every image
# of Mermaid still has revisits (96 %, 66 per image in median). A working choice.
REVISIT_MUTUAL_OVERLAP = 0.5


@dataclass(frozen=True, eq=False)
class FarPairs:
    """Pairs of images more than a given gap apart whose footprints intersect, one entry each.

    `index_a` < `index_b` are camera indices. `mutual_overlap` is the smaller of the two
    shares each image sees of the other. `view_angle_deg` is the angle under which the two
    cameras see the middle of what they share: 0 for the same viewpoint.
    """

    index_a: np.ndarray
    index_b: np.ndarray
    mutual_overlap: np.ndarray
    iou: np.ndarray
    view_angle_deg: np.ndarray

    def __len__(self) -> int:
        return len(self.index_a)

    def revisits(self, mutual_overlap: float = REVISIT_MUTUAL_OVERLAP) -> np.ndarray:
        """Boolean mask of the pairs sharing at least `mutual_overlap`."""
        return self.mutual_overlap >= mutual_overlap


def far_pairs(
    footprints: np.ndarray,
    camera_centres: np.ndarray,
    seafloor_height_m: float,
    min_gap: int,
) -> FarPairs:
    """Cameras are in capture order; pairs `min_gap` or fewer apart are neighbours, left out."""
    valid = np.flatnonzero(~np.isnan(footprints).any(axis=(1, 2)))
    polygons = np.array(
        [footprint_polygon(footprints, int(index)) for index in valid], dtype=object
    )
    first, second = STRtree(polygons).query(polygons, predicate="intersects")
    far = valid[second] - valid[first] > min_gap
    first, second = first[far], second[far]
    index_a, index_b = valid[first], valid[second]

    shared = shapely.intersection(polygons[first], polygons[second])
    shared_area = shapely.area(shared)
    area_a, area_b = shapely.area(polygons[first]), shapely.area(polygons[second])
    middle = shapely.centroid(shared)
    point = np.column_stack(
        [
            shapely.get_x(middle),
            shapely.get_y(middle),
            np.full(len(first), seafloor_height_m),
        ]
    )
    ray_a, ray_b = camera_centres[index_a] - point, camera_centres[index_b] - point
    cosine = np.sum(ray_a * ray_b, axis=1) / (
        np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
    )
    return FarPairs(
        index_a=index_a,
        index_b=index_b,
        mutual_overlap=shared_area / np.maximum(area_a, area_b),
        iou=shared_area / (area_a + area_b - shared_area),
        view_angle_deg=np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))),
    )
