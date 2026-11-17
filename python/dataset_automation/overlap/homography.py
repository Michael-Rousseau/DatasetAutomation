from dataclasses import dataclass

import cv2
import numpy as np
from shapely import Polygon

# Largest footprint area ratio between Mermaid images that overlap by ≥ 20 % is 5.2 (p99 3.0,
# from the pose-based reference): a homography scaling areas by more than this is not a view
# of the same seafloor.
MAX_AREA_RATIO = 8.0


@dataclass(frozen=True, eq=False)
class HomographyFit:
    """Maps undistorted (or raw) pixels of image A to those of image B."""

    matrix: np.ndarray
    inliers: np.ndarray  # boolean, one entry per match

    @property
    def inlier_count(self) -> int:
        return int(self.inliers.sum())


def fit_homography(
    points_a: np.ndarray, points_b: np.ndarray, threshold_px: float
) -> HomographyFit | None:
    if len(points_a) < 4:
        return None
    matrix, inlier_mask = cv2.findHomography(
        points_a, points_b, cv2.USAC_MAGSAC, threshold_px
    )
    if matrix is None or inlier_mask is None:
        return None
    return HomographyFit(matrix=matrix, inliers=inlier_mask.ravel().astype(bool))


def transform(matrix: np.ndarray, polygon: Polygon) -> Polygon:
    vertices = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    return Polygon(
        cv2.perspectiveTransform(vertices.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    )


def is_plausible(matrix: np.ndarray, frame_a: Polygon, frame_b: Polygon) -> bool:
    """Whether the homography can be a view change between two images of the same seafloor.

    With few inliers MAGSAC can return a homography that sends part of an image beyond the
    horizon, mirrors it, or blows up its scale: the overlap computed from it would be garbage.
    """
    return _maps_frame_sensibly(matrix, frame_a) and _maps_frame_sensibly(
        np.linalg.inv(matrix), frame_b
    )


def _maps_frame_sensibly(matrix: np.ndarray, frame: Polygon) -> bool:
    vertices = np.asarray(frame.exterior.coords[:-1], dtype=np.float64)
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))]) @ matrix.T
    # A vertex at or beyond the horizon (w ≤ 0) wraps the projected outline through infinity.
    if (homogeneous[:, 2] <= 0).any():
        return False
    projected = transform(matrix, frame)
    if not projected.is_valid or projected.area == 0:
        return False
    # A mirrored view reverses the outline's orientation: no camera motion does that.
    if _signed_area(projected) * _signed_area(frame) < 0:
        return False
    return 1 / MAX_AREA_RATIO <= projected.area / frame.area <= MAX_AREA_RATIO


def _signed_area(polygon: Polygon) -> float:
    x, y = np.asarray(polygon.exterior.coords).T
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
