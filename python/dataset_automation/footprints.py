import cv2
import numpy as np
from shapely import Polygon

from dataset_automation.intrinsics import Intrinsics, camera_matrix, opencv_distortion
from dataset_automation.poses import CameraPoses, camera_centres
from dataset_automation.seafloor import SeafloorSurface

# OpenCV's default (5 iterations) leaves ~0.1 px of round-trip error at the corners of this
# wide-angle lens; these settings bring it below 1e-9 px.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)
DEFAULT_SAMPLES_PER_SIDE = 25


def image_border_pixels(
    width_px: int, height_px: int, samples_per_side: int
) -> np.ndarray:
    """Closed loop along the outer edge of the image, clockwise, as (4·samples_per_side, 2).

    Coordinates follow OpenCV (pixel centres at integers), so the outer edge runs from −0.5
    to size − 0.5. Each side starts at a corner and stops before the next one.
    """
    left, top = -0.5, -0.5
    right, bottom = width_px - 0.5, height_px - 0.5
    steps = np.linspace(0.0, 1.0, samples_per_side, endpoint=False)
    top_edge = np.column_stack(
        [left + steps * (right - left), np.full_like(steps, top)]
    )
    right_edge = np.column_stack(
        [np.full_like(steps, right), top + steps * (bottom - top)]
    )
    bottom_edge = np.column_stack(
        [right - steps * (right - left), np.full_like(steps, bottom)]
    )
    left_edge = np.column_stack(
        [np.full_like(steps, left), bottom - steps * (bottom - top)]
    )
    return np.concatenate([top_edge, right_edge, bottom_edge, left_edge])


def camera_rays(intrinsics: Intrinsics, pixels: np.ndarray) -> np.ndarray:
    """Directions (M, 3) in the camera frame, as (x, y, 1), of the rays seen at `pixels`."""
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64),
        camera_matrix(intrinsics),
        opencv_distortion(intrinsics),
        None,
        None,
        None,
        UNDISTORT_CRITERIA,
    ).reshape(-1, 2)
    return np.column_stack([normalized, np.ones(len(normalized))])


def compute_footprints(
    poses: CameraPoses, rays_in_camera: np.ndarray, surface: SeafloorSurface
) -> np.ndarray:
    """Ground footprint of every camera as (N, M, 2) xy vertices, NaN where it is undefined.

    The rays are the same for every camera (shared intrinsics), so they are only rotated
    per camera. A footprint with a single ray missing the surface is dropped entirely:
    a truncated polygon would silently under-report overlap.
    """
    rotations = poses.camera_to_world[:, :3, :3]
    rays_in_world = np.einsum("nij,mj->nmi", rotations, rays_in_camera)
    hits = surface.intersect(camera_centres(poses)[:, None, :], rays_in_world)
    footprints = hits[..., :2].copy()
    footprints[np.isnan(footprints).any(axis=(1, 2))] = np.nan
    return footprints


def invalid_footprints(footprints: np.ndarray) -> np.ndarray:
    return np.isnan(footprints).any(axis=(1, 2))


def footprint_polygon(footprints: np.ndarray, camera_index: int) -> Polygon | None:
    vertices = footprints[camera_index]
    if np.isnan(vertices).any():
        return None
    return Polygon(vertices)
