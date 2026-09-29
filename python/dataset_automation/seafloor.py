from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SeafloorSurface(Protocol):
    """Anything rays can be cast onto: a plane today, a mesh later."""

    def intersect(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Hit points of rays `origin + t·direction` with t > 0.

        `origins` and `directions` have shape (..., 3) and broadcast together.
        Returns (..., 3), with NaN rows for rays that never reach the surface.
        """
        ...


@dataclass(frozen=True)
class HorizontalPlane:
    height_m: float

    def intersect(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        origins, directions = np.broadcast_arrays(origins, directions)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.height_m - origins[..., 2]) / directions[..., 2]

        t = np.where(np.isfinite(t) & (t > 0), t, np.nan)
        hits = origins + t[..., None] * directions

        return hits


def altitude_from_gsd(gsd_m: float, focal_px: float) -> float:
    return gsd_m * focal_px


def plane_below_lowest_cameras(
    camera_heights_m: np.ndarray,
    lowest_altitude_m: float,
    lowest_percentile: float = 1.0,
) -> HorizontalPlane:
    # A low percentile rather than the minimum, so a single outlier pose does not sink the plane.
    lowest_height_m = float(np.percentile(camera_heights_m, lowest_percentile))
    return HorizontalPlane(height_m=lowest_height_m - lowest_altitude_m)
