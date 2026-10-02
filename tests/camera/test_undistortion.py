import cv2
import numpy as np

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.camera.intrinsics import (
    Intrinsics,
    camera_matrix,
    opencv_distortion,
)
from dataset_automation.camera.undistortion import (
    undistort_to_normalized,
    undistort_to_pixels,
)
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS

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


def test_mermaid_border_rays_project_back_onto_border_pixels() -> None:
    border = image_border_pixels(
        MERMAID_INTRINSICS.width_px, MERMAID_INTRINSICS.height_px, 25
    )

    normalized = undistort_to_normalized(MERMAID_INTRINSICS, border)
    projected, _ = cv2.projectPoints(
        np.column_stack([normalized, np.ones(len(normalized))]),
        np.zeros(3),
        np.zeros(3),
        camera_matrix(MERMAID_INTRINSICS),
        opencv_distortion(MERMAID_INTRINSICS),
    )

    np.testing.assert_allclose(projected.reshape(-1, 2), border, atol=1e-6)


def test_undistortion_is_identity_without_lens_distortion() -> None:
    pixels = np.array([[0.0, 0.0], [123.4, 56.7], [399.5, 299.5]])

    np.testing.assert_allclose(undistort_to_pixels(PINHOLE, pixels), pixels, atol=1e-9)


def test_principal_point_does_not_move() -> None:
    centre = camera_matrix(MERMAID_INTRINSICS)[:2, 2]

    np.testing.assert_allclose(
        undistort_to_pixels(MERMAID_INTRINSICS, centre[None]), [centre], atol=1e-9
    )


def test_undoing_barrel_distortion_pushes_edge_midpoints_outwards() -> None:
    # Mermaid's lens squeezes the field towards the centre by ~65 px at the edge midpoints.
    # Not at the corners: with k2 > 0 the polynomial turns back there (they move ~24 px inwards).
    centre = camera_matrix(MERMAID_INTRINSICS)[:2, 2]
    edge_midpoints = np.array([[1919.5, -0.5], [3839.5, 1439.5]])

    undistorted = undistort_to_pixels(MERMAID_INTRINSICS, edge_midpoints)

    pushed = np.linalg.norm(undistorted - centre, axis=1) - np.linalg.norm(
        edge_midpoints - centre, axis=1
    )
    assert (pushed > 50).all()


def test_accepts_no_points() -> None:
    assert undistort_to_pixels(MERMAID_INTRINSICS, np.empty((0, 2))).shape == (0, 2)
