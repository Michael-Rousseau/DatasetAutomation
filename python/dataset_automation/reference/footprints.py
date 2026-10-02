import numpy as np
from shapely import Polygon

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.camera.undistortion import undistort_to_normalized
from dataset_automation.reference.poses import CameraPoses, camera_centres
from dataset_automation.reference.seafloor import SeafloorSurface

DEFAULT_SAMPLES_PER_SIDE = 25


def camera_rays(intrinsics: Intrinsics, pixels: np.ndarray) -> np.ndarray:
    """Directions (M, 3) in the camera frame, as (x, y, 1), of the rays seen at `pixels`."""
    normalized = undistort_to_normalized(intrinsics, pixels)
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
