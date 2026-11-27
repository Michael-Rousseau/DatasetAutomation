"""3D scene of a survey: trajectory, cameras, ground footprints and links between images.

Pure geometry: every layer is a set of coloured line segments in world coordinates, so it can
be checked without a display and drawn by any viewer (Open3D in `view.render`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import shapely
from matplotlib import colormaps

from dataset_automation.reference.footprints import footprint_polygon


class ImageStatus(Enum):
    KEPT = "kept"
    REMOVED = "removed"
    # Not in the selection read, or no selection read at all.
    UNKNOWN = "unknown"


STATUS_COLOURS: dict[ImageStatus, tuple[float, float, float]] = {
    ImageStatus.KEPT: (0.13, 0.55, 0.13),
    ImageStatus.REMOVED: (0.84, 0.19, 0.15),
    ImageStatus.UNKNOWN: (0.45, 0.45, 0.45),
}
TRAJECTORY_COLOUR = (0.1, 0.1, 0.1)
UNKNOWN_ANGLE_COLOUR = (0.75, 0.75, 0.75)
ANGLE_COLOURMAP = "viridis"
# Revisits of Mermaid are seen 27° apart in median: 40° keeps most of them inside the scale.
DEFAULT_MAX_ANGLE_DEG = 40.0
# Cameras fly 1 to 3 m above the seafloor: a 25 cm pyramid is visible without hiding it.
DEFAULT_FRUSTUM_DEPTH_M = 0.25

# The four edges from the apex (index 0) to the base corners, then the base loop.
_FRUSTUM_EDGES = np.array(
    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
)


@dataclass(frozen=True, eq=False)
class LineSet:
    """`segments` (S, 2) are indices into `points` (P, 3); `colours` (S, 3) are RGB in [0, 1],
    one per segment."""

    points: np.ndarray
    segments: np.ndarray
    colours: np.ndarray

    def __len__(self) -> int:
        return len(self.segments)


def trajectory(centres: np.ndarray) -> LineSet:
    """Camera centres (N, 3) joined in capture order."""
    steps = np.arange(max(len(centres) - 1, 0))
    segments = np.column_stack([steps, steps + 1])
    return LineSet(
        points=centres.copy(),
        segments=segments,
        colours=np.tile(TRAJECTORY_COLOUR, (len(segments), 1)),
    )


def camera_frustums(
    camera_to_world: np.ndarray,
    corner_rays: np.ndarray,
    depth_m: float,
    colours: np.ndarray,
) -> LineSet:
    """One pyramid per camera: apex at the centre, base where the image corners are seen at
    `depth_m` along the optical axis.

    `camera_to_world` is (N, 4, 4); `corner_rays` (4, 3) are camera-frame rays (x, y, 1) of the
    image corners, in order around the image; `colours` is (N, 3).
    """
    rotations, centres = camera_to_world[:, :3, :3], camera_to_world[:, :3, 3]
    corners = centres[:, None, :] + depth_m * np.einsum(
        "nij,kj->nki", rotations, corner_rays
    )
    points = np.concatenate([centres[:, None, :], corners], axis=1)
    return _repeat_per_item(points, _FRUSTUM_EDGES, colours)


def footprint_outlines(
    footprints: np.ndarray, height_m: float, colours: np.ndarray
) -> LineSet:
    """Closed outline of each footprint (N, M, 2) on the plane z = `height_m`, with `colours`
    (N, 3). Undefined footprints (NaN) are left out rather than drawn truncated."""
    valid = ~np.isnan(footprints).any(axis=(1, 2))
    kept = footprints[valid]
    vertex_count = footprints.shape[1]
    points = np.concatenate([kept, np.full((*kept.shape[:2], 1), height_m)], axis=2)
    around = np.arange(vertex_count)
    loop = np.column_stack([around, (around + 1) % vertex_count])
    return _repeat_per_item(points, loop, colours[valid])


def links(
    centres: np.ndarray,
    pairs: np.ndarray,
    angles_deg: np.ndarray,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
) -> LineSet:
    """A segment between the centres of each pair (K, 2) of camera indices, coloured by the
    angle between the two views (K,), NaN where unknown."""
    return LineSet(
        points=centres.copy(),
        segments=pairs.reshape(-1, 2).astype(np.int64),
        colours=angle_colours(angles_deg, max_angle_deg),
    )


def angle_colours(angles_deg: np.ndarray, max_angle_deg: float) -> np.ndarray:
    """Colour of each angle on `ANGLE_COLOURMAP` from 0 to `max_angle_deg` (clipped above);
    unknown angles (NaN) get `UNKNOWN_ANGLE_COLOUR`, never a colour of the scale."""
    unknown = np.isnan(angles_deg)
    scaled = np.clip(np.where(unknown, 0.0, angles_deg) / max_angle_deg, 0.0, 1.0)
    colours = colormaps[ANGLE_COLOURMAP](scaled)[:, :3]
    colours[unknown] = UNKNOWN_ANGLE_COLOUR
    return colours


def status_colours(statuses: Sequence[ImageStatus]) -> np.ndarray:
    return np.array([STATUS_COLOURS[status] for status in statuses]).reshape(-1, 3)


def faded(colours: np.ndarray, towards_white: float) -> np.ndarray:
    """`colours` mixed with white: hundreds of overlapping footprints drawn at full strength
    hide the cameras and links above them."""
    return colours + towards_white * (1.0 - colours)


def sample_pairs(
    pairs: np.ndarray, angles_deg: np.ndarray, count: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """At most `count` pairs drawn at random, the same ones for the same seed. Tens of
    thousands of links fill the whole view; a sample still shows where they go and their
    angles."""
    if len(pairs) <= count:
        return pairs, angles_deg
    chosen = np.sort(
        np.random.default_rng(seed).choice(len(pairs), size=count, replace=False)
    )
    return pairs[chosen], angles_deg[chosen]


def reference_view_angles(
    footprints: np.ndarray,
    centres: np.ndarray,
    height_m: float,
    pairs: np.ndarray,
) -> np.ndarray:
    """Angle under which the two cameras of each pair see the middle of what their footprints
    share: the measure of `reference.revisits.FarPairs.view_angle_deg`, for chosen pairs.

    NaN when a footprint is undefined or the two share nothing: no place to measure it from.
    """
    pairs = pairs.reshape(-1, 2)
    angles = np.full(len(pairs), np.nan)
    for row, (first, second) in enumerate(pairs):
        polygon_a = footprint_polygon(footprints, int(first))
        polygon_b = footprint_polygon(footprints, int(second))
        if polygon_a is None or polygon_b is None:
            continue
        shared = shapely.intersection(polygon_a, polygon_b)
        if shared.is_empty:
            continue
        middle = shared.centroid
        point = np.array([middle.x, middle.y, height_m])
        ray_a, ray_b = centres[first] - point, centres[second] - point
        cosine = ray_a @ ray_b / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
        angles[row] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angles


def _repeat_per_item(
    points: np.ndarray, local_segments: np.ndarray, colours: np.ndarray
) -> LineSet:
    """One copy of `local_segments` per item of `points` (N, V, 3), indices shifted to it."""
    item_count, vertex_count = points.shape[:2]
    offsets = vertex_count * np.arange(item_count)
    segments = local_segments[None, :, :] + offsets[:, None, None]
    return LineSet(
        points=points.reshape(-1, 3),
        segments=segments.reshape(-1, 2),
        colours=np.repeat(colours.reshape(-1, 3), len(local_segments), axis=0),
    )
