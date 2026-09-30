from dataclasses import dataclass

import numpy as np

# Metashape puts the centre of the top-left pixel at (0.5, 0.5); OpenCV puts it at (0, 0).
METASHAPE_TO_OPENCV_PIXEL_SHIFT = -0.5


@dataclass(frozen=True)
class Intrinsics:
    """Camera calibration stored raw, in Agisoft Metashape's frame-camera convention.

    Reference: Metashape Pro 2.1 manual, appendix D "Camera models":
        x' = x(1 + K1r² + K2r⁴ + K3r⁶) + P1(r² + 2x²) + 2·P2·xy
        y' = y(1 + K1r² + K2r⁴ + K3r⁶) + P2(r² + 2y²) + 2·P1·xy
        u = w/2 + cx + x'·f,   v = h/2 + cy + y'·f
    Use `camera_matrix` and `opencv_distortion` to get OpenCV's equivalents.
    """

    width_px: int
    height_px: int
    focal_px: float
    cx_offset_px: float
    cy_offset_px: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float


def camera_matrix(intrinsics: Intrinsics) -> np.ndarray:
    principal_x = (
        intrinsics.width_px / 2
        + intrinsics.cx_offset_px
        + METASHAPE_TO_OPENCV_PIXEL_SHIFT
    )
    principal_y = (
        intrinsics.height_px / 2
        + intrinsics.cy_offset_px
        + METASHAPE_TO_OPENCV_PIXEL_SHIFT
    )
    return np.array(
        [
            [intrinsics.focal_px, 0.0, principal_x],
            [0.0, intrinsics.focal_px, principal_y],
            [0.0, 0.0, 1.0],
        ]
    )


def opencv_distortion(intrinsics: Intrinsics) -> np.ndarray:
    # OpenCV order is (k1, k2, p1, p2, k3), and its p1/p2 play the roles of Metashape's P2/P1.
    return np.array(
        [intrinsics.k1, intrinsics.k2, intrinsics.p2, intrinsics.p1, intrinsics.k3]
    )


# GoPro Hero 3 Silver, from the Mermaid dataset description.
MERMAID_INTRINSICS = Intrinsics(
    width_px=3840,
    height_px=2880,
    focal_px=2334.29,
    cx_offset_px=-12.752,
    cy_offset_px=-16.6962,
    k1=-0.222446,
    k2=0.310621,
    k3=-0.0835057,
    p1=-0.000995472,
    p2=-7.8498e-05,
)
