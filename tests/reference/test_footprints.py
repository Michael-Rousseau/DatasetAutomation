import numpy as np
import pytest

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.reference.footprints import (
    camera_rays,
    compute_footprints,
    footprint_polygon,
    invalid_footprints,
)
from dataset_automation.reference.poses import CameraPoses
from dataset_automation.reference.seafloor import HorizontalPlane

PINHOLE = Intrinsics(
    width_px=400,
    height_px=300,
    focal_px=200.0,
    cx_offset_px=0.0,
    cy_offset_px=0.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
)
# Camera frame is x right, y down, z forward. Columns below are those axes in world coordinates.
LOOKING_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
LOOKING_ALONG_X = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def make_poses(rotations: list[np.ndarray], centres: list[list[float]]) -> CameraPoses:
    transforms = np.tile(np.eye(4), (len(rotations), 1, 1))
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = centres
    unknown = np.full((len(rotations), 3, 3), np.nan)
    return CameraPoses(
        labels=[f"camera_{i}" for i in range(len(rotations))],
        camera_to_world=transforms,
        rotation_covariance=unknown,
        location_covariance=unknown,
    )


def test_undistorted_nadir_footprint_is_the_image_rectangle_scaled_by_altitude() -> (
    None
):
    poses = make_poses([LOOKING_DOWN], [[10.0, 20.0, 2.0]])
    rays = camera_rays(PINHOLE, image_border_pixels(400, 300, 8))

    footprint = footprint_polygon(
        compute_footprints(poses, rays, HorizontalPlane(height_m=0.0)), 0
    )

    # Seen from 2 m with f = 200 px, 400 × 300 px covers 4 m × 3 m.
    assert footprint is not None
    minx, miny, maxx, maxy = footprint.bounds
    assert (maxx - minx, maxy - miny) == (pytest.approx(4.0), pytest.approx(3.0))
    assert footprint.centroid.x == pytest.approx(10.0, abs=0.01)
    assert footprint.centroid.y == pytest.approx(20.0, abs=0.01)


def test_camera_looking_at_horizon_gets_no_footprint() -> None:
    poses = make_poses(
        [LOOKING_DOWN, LOOKING_ALONG_X], [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    )
    rays = camera_rays(PINHOLE, image_border_pixels(400, 300, 8))

    footprints = compute_footprints(poses, rays, HorizontalPlane(height_m=0.0))

    np.testing.assert_array_equal(invalid_footprints(footprints), [False, True])
    assert footprint_polygon(footprints, 1) is None
