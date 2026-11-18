"""What a set of images lets a 3D reconstruction see, from known camera poses.

A point is reconstructed from views taken at different positions: many images from the same
place add nothing. For every cell of the seafloor this counts the images seeing it and measures
how wide the fan of their viewing directions is. It is an evaluation tool: it needs the poses,
and says what the geometry allows, not whether the images actually connect.
"""

from collections.abc import Iterable
from dataclasses import dataclass

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

    `fan_deg` is 0 for a cell seen from a single position.
    """

    views: np.ndarray
    fan_deg: np.ndarray


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
        inside = shapely.contains_xy(polygon, cell_x, cell_y)

        centre = camera_centres[camera]
        height = centre[2] - seafloor_height_m
        slopes = np.stack(
            [(centre[0] - cell_x) / height, (centre[1] - cell_y) / height]
        )
        views[rows, columns] += inside
        lowest[:, rows, columns] = np.where(
            inside,
            np.minimum(lowest[:, rows, columns], slopes),
            lowest[:, rows, columns],
        )
        highest[:, rows, columns] = np.where(
            inside,
            np.maximum(highest[:, rows, columns], slopes),
            highest[:, rows, columns],
        )

    fan_deg = np.zeros(shape)
    seen = views > 0
    extent = highest[:, seen] - lowest[:, seen]
    fan_deg[seen] = np.degrees(np.arctan(np.hypot(extent[0], extent[1])))
    return Observation(views=views, fan_deg=fan_deg)


def reconstructible(
    observation: Observation,
    min_views: int = MIN_VIEWS,
    min_fan_deg: float = MIN_FAN_DEG,
) -> np.ndarray:
    """Cells seen by enough images from directions spread widely enough."""
    return (observation.views >= min_views) & (observation.fan_deg >= min_fan_deg)
