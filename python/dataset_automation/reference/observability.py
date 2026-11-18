"""What a set of images lets a 3D reconstruction see, from known camera poses.

A point is reconstructed from views taken at different positions: many images from the same
place add nothing. For every cell of the seafloor this counts the images seeing it and measures
how wide the fan of their viewing directions is. It is an evaluation tool: it needs the poses,
and says what the geometry allows, not whether the images actually connect.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import shapely

from dataset_automation.reference.footprints import footprint_polygon

DEFAULT_CELL_M = 0.10
# First guesses, not measured against a real reconstruction.
MIN_VIEWS = 3
MIN_FAN_DEG = 10.0


@dataclass(frozen=True, eq=False)
class SeafloorGrid:
    """Cell centres of a regular grid over the seafloor plane, in metres."""

    x_m: np.ndarray
    y_m: np.ndarray
    cell_m: float

    @property
    def cell_area_m2(self) -> float:
        return self.cell_m**2


@dataclass(frozen=True, eq=False)
class Observation:
    """Per grid cell (row = y, column = x): images seeing it and the fan of their directions.

    `fan_deg` is 0 for a cell seen from a single position. `depth_information` grades what the
    fan only thresholds: the scatter of the viewing directions around their mean, summed over
    the images. The depth of a point is known with a noise proportional to 1 / sqrt of it, so
    it grows with both the number of images and how far apart they stand; it is 0 from a
    single position.
    """

    views: np.ndarray
    fan_deg: np.ndarray
    depth_information: np.ndarray


def grid_over(footprints: np.ndarray, cell_m: float = DEFAULT_CELL_M) -> SeafloorGrid:
    """Grid covering every valid footprint of `footprints`, (N, M, 2) with NaN where undefined."""
    low = np.nanmin(footprints, axis=(0, 1))
    high = np.nanmax(footprints, axis=(0, 1))
    return SeafloorGrid(
        x_m=np.arange(low[0] + cell_m / 2, high[0], cell_m),
        y_m=np.arange(low[1] + cell_m / 2, high[1], cell_m),
        cell_m=cell_m,
    )


def observe(
    grid: SeafloorGrid,
    footprints: np.ndarray,
    camera_centres: np.ndarray,
    seafloor_height_m: float,
    cameras: Iterable[int],
) -> Observation:
    """Views and fan of viewing directions per cell, for the cameras at indices `cameras`."""
    shape = (len(grid.y_m), len(grid.x_m))
    views = np.zeros(shape, dtype=np.int32)
    # A viewing direction is stored as its slopes along x and y (tangent of the angle to the
    # vertical); the fan is measured across the box holding every direction of the cell.
    lowest = np.full((2, *shape), np.inf)
    highest = np.full((2, *shape), -np.inf)
    slope_sum = np.zeros((2, *shape))
    squared_slope_sum = np.zeros(shape)

    for camera in cameras:
        polygon = footprint_polygon(footprints, camera)
        if polygon is None:
            continue
        min_x, min_y, max_x, max_y = polygon.bounds
        columns = slice(
            np.searchsorted(grid.x_m, min_x), np.searchsorted(grid.x_m, max_x)
        )
        rows = slice(np.searchsorted(grid.y_m, min_y), np.searchsorted(grid.y_m, max_y))
        cell_x, cell_y = np.meshgrid(grid.x_m[columns], grid.y_m[rows])
        seen_here = shapely.contains_xy(polygon, cell_x, cell_y)

        centre = camera_centres[camera]
        height = centre[2] - seafloor_height_m
        slopes = np.stack(
            [(centre[0] - cell_x) / height, (centre[1] - cell_y) / height]
        )
        views[rows, columns] += seen_here
        slope_sum[:, rows, columns] += np.where(seen_here, slopes, 0.0)
        squared_slope_sum[rows, columns] += np.where(
            seen_here, slopes[0] ** 2 + slopes[1] ** 2, 0.0
        )
        lowest[:, rows, columns] = np.where(
            seen_here,
            np.minimum(lowest[:, rows, columns], slopes),
            lowest[:, rows, columns],
        )
        highest[:, rows, columns] = np.where(
            seen_here,
            np.maximum(highest[:, rows, columns], slopes),
            highest[:, rows, columns],
        )

    fan_deg = np.zeros(shape)
    seen = views > 0
    extent = highest[:, seen] - lowest[:, seen]
    fan_deg[seen] = np.degrees(np.arctan(np.hypot(extent[0], extent[1])))
    # Sum of squared distances to the mean direction: Σ|s|² − |Σs|² / n.
    depth_information = np.zeros(shape)
    depth_information[seen] = (
        squared_slope_sum[seen]
        - (slope_sum[0, seen] ** 2 + slope_sum[1, seen] ** 2) / views[seen]
    )
    # Rounding leaves tiny negative values where every view comes from the same place.
    np.maximum(depth_information, 0.0, out=depth_information)
    return Observation(
        views=views, fan_deg=fan_deg, depth_information=depth_information
    )


def reconstructible(
    observation: Observation,
    min_views: int = MIN_VIEWS,
    min_fan_deg: float = MIN_FAN_DEG,
) -> np.ndarray:
    """Cells seen by enough images from directions spread widely enough."""
    return (observation.views >= min_views) & (observation.fan_deg >= min_fan_deg)


def depth_noise_ratio(everything: Observation, subset: Observation) -> np.ndarray:
    """How much noisier the depth of each cell is with `subset` than with `everything`.

    1 means unchanged, 2 twice the noise, infinity that the subset cannot give a depth there.
    NaN where even `everything` gives none.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.sqrt(everything.depth_information / subset.depth_information)
    ratio[everything.depth_information == 0] = np.nan
    return ratio


def inside(area: np.ndarray, grid: SeafloorGrid, margin_m: float) -> np.ndarray:
    """Cells of the boolean `area` further than `margin_m` from the nearest cell outside it."""
    # Padded so that the border of the grid counts as outside too.
    padded = np.pad(area.astype(np.uint8), 1)
    distance_m = (
        cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        * grid.cell_m
    )
    return area & (distance_m > margin_m)
