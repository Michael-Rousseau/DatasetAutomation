"""Image frames and matched points, undistorted when the camera is known."""

import numpy as np
from shapely import Polygon

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.camera.undistortion import undistort_to_pixels
from dataset_automation.features.interface import Features, Matches

# Undistortion curves the image edges: sample them densely enough to follow the curve.
BORDER_SAMPLES_PER_SIDE = 25


def image_frame(size: tuple[int, int], intrinsics: Intrinsics | None) -> Polygon:
    """Outline of a (width, height) image, in undistorted pixels if `intrinsics` is given.

    For a nadir view, areas in undistorted pixels are proportional to areas on the ground,
    which is what the pose-based reference overlap measures.
    """
    border = image_border_pixels(size[0], size[1], BORDER_SAMPLES_PER_SIDE)
    if intrinsics is not None:
        border = undistort_to_pixels(intrinsics, border)
    return Polygon(border)


def matched_points(
    features_a: Features,
    features_b: Features,
    matches: Matches,
    intrinsics: Intrinsics | None,
) -> tuple[np.ndarray, np.ndarray]:
    points_a = features_a.keypoints[matches.index_pairs[:, 0]]
    points_b = features_b.keypoints[matches.index_pairs[:, 1]]
    if intrinsics is None:
        return points_a, points_b
    return undistort_to_pixels(intrinsics, points_a), undistort_to_pixels(
        intrinsics, points_b
    )
