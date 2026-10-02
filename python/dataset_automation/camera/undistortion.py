import cv2
import numpy as np

from dataset_automation.camera.intrinsics import (
    Intrinsics,
    camera_matrix,
    opencv_distortion,
)

# OpenCV's default (5 iterations) leaves ~0.1 px of round-trip error at the corners of a
# wide-angle lens like Mermaid's GoPro; these settings bring it below 1e-9 px.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)


def undistort_to_normalized(intrinsics: Intrinsics, pixels: np.ndarray) -> np.ndarray:
    """Normalized camera coordinates (M, 2): the ray seen at each pixel is (x, y, 1)."""
    if len(pixels) == 0:
        return np.empty((0, 2))
    return cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64),
        camera_matrix(intrinsics),
        opencv_distortion(intrinsics),
        None,
        None,
        None,
        UNDISTORT_CRITERIA,
    ).reshape(-1, 2)


def undistort_to_pixels(intrinsics: Intrinsics, pixels: np.ndarray) -> np.ndarray:
    """Where each pixel would be seen by the same camera without lens distortion (M, 2).

    Keeps pixel units and the same focal length and principal point, so that pixel thresholds
    (e.g. RANSAC's reprojection error) keep their meaning after undistortion.
    """
    matrix = camera_matrix(intrinsics)
    normalized = undistort_to_normalized(intrinsics, pixels)
    return normalized * [matrix[0, 0], matrix[1, 1]] + [matrix[0, 2], matrix[1, 2]]
