import cv2
import numpy as np
import pytest

from dataset_automation.footprints import (
    camera_rays,
    compute_footprints,
    footprint_polygon,
    image_border_pixels,
    invalid_footprints,
)
from dataset_automation.intrinsics import (
    MERMAID_INTRINSICS,
    Intrinsics,
    camera_matrix,
    opencv_distortion,
)
from dataset_automation.poses import CameraPoses
from dataset_automation.seafloor import HorizontalPlane

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


def test_border_starts_at_top_left_corner_and_runs_clockwise() -> None:
    border = image_border_pixels(400, 300, samples_per_side=4)

    assert border.shape == (16, 2)
    np.testing.assert_allclose(border[0], [-0.5, -0.5])
    np.testing.assert_allclose(border[4], [399.5, -0.5])
    np.testing.assert_allclose(border[8], [399.5, 299.5])
    np.testing.assert_allclose(border[12], [-0.5, 299.5])


def test_opencv_distortion_swaps_metashape_tangential_coefficients() -> None:
    k1, k2, p1, p2, k3 = opencv_distortion(MERMAID_INTRINSICS)

    assert (k1, k2, k3) == (
        MERMAID_INTRINSICS.k1,
        MERMAID_INTRINSICS.k2,
        MERMAID_INTRINSICS.k3,
    )
    assert (p1, p2) == (MERMAID_INTRINSICS.p2, MERMAID_INTRINSICS.p1)


def test_principal_point_moves_half_pixel_into_opencv_convention() -> None:
    matrix = camera_matrix(MERMAID_INTRINSICS)

    assert matrix[0, 2] == pytest.approx(1920 - 12.752 - 0.5)
    assert matrix[1, 2] == pytest.approx(1440 - 16.6962 - 0.5)


def test_mermaid_border_rays_project_back_onto_border_pixels() -> None:
    border = image_border_pixels(
        MERMAID_INTRINSICS.width_px, MERMAID_INTRINSICS.height_px, 25
    )

    rays = camera_rays(MERMAID_INTRINSICS, border)
    projected, _ = cv2.projectPoints(
        rays,
        np.zeros(3),
        np.zeros(3),
        camera_matrix(MERMAID_INTRINSICS),
        opencv_distortion(MERMAID_INTRINSICS),
    )

    np.testing.assert_allclose(projected.reshape(-1, 2), border, atol=1e-6)


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
