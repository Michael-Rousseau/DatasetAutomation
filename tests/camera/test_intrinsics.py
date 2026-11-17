import pytest

from dataset_automation.camera.intrinsics import camera_matrix, opencv_distortion
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS


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
