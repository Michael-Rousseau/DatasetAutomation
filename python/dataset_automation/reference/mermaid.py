"""The Mermaid reference: poses, seafloor plane and ground footprints, built one way only.

Every evaluation against Mermaid must use the same seafloor plane and the same footprints, or
their numbers cannot be compared.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.reference.footprints import (
    DEFAULT_SAMPLES_PER_SIDE,
    camera_rays,
    compute_footprints,
)
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.reference.poses import (
    CameraPoses,
    camera_centres,
    load_camera_poses,
)
from dataset_automation.reference.seafloor import (
    altitude_from_gsd,
    plane_below_lowest_cameras,
)

# Smallest ground sampling distance announced for the dataset (report 107174): the lowest
# cameras see about 0.5 mm per pixel, which fixes their altitude and anchors the seafloor plane.
MIN_GSD_M = 0.0005


@dataclass(frozen=True, eq=False)
class MermaidReference:
    """`footprints` is (N, M, 2), NaN for a camera whose view misses the plane.

    The seafloor is a horizontal plane although the relief exceeds 2.7 m: an approximation,
    least reliable for tilted views.
    """

    poses: CameraPoses
    centres: np.ndarray
    seafloor_height_m: float
    footprints: np.ndarray


def load_mermaid_reference(poses_path: Path) -> MermaidReference:
    poses = load_camera_poses(poses_path)
    centres = camera_centres(poses)
    plane = plane_below_lowest_cameras(
        centres[:, 2], altitude_from_gsd(MIN_GSD_M, MERMAID_INTRINSICS.focal_px)
    )
    border = image_border_pixels(
        MERMAID_INTRINSICS.width_px,
        MERMAID_INTRINSICS.height_px,
        DEFAULT_SAMPLES_PER_SIDE,
    )
    footprints = compute_footprints(
        poses, camera_rays(MERMAID_INTRINSICS, border), plane
    )
    return MermaidReference(
        poses=poses,
        centres=centres,
        seafloor_height_m=plane.height_m,
        footprints=footprints,
    )
