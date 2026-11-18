"""How far the camera moved between two images, relative to the distance of the scene.

Overlap says what two images see in common; this says whether they see it from different
places. Only the second makes a 3D reconstruction possible: two images taken from the same
spot overlap fully and carry no depth.

The measure is the triangulation angle: for a matched point, the angle between the ray that
sees it in A and the ray that sees it in B. It is roughly displacement / distance, so it needs
no metric scale, which a pair of images cannot give.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.camera.undistortion import undistort_to_normalized
from dataset_automation.features.interface import Features, Matches
from dataset_automation.overlap.pairwise import MIN_INLIERS, UnknownReason

# 3 px with MAGSAC, on the 1,243 consecutive Mermaid pairs against the reference poses: angle
# off by 0.17 degrees in median (p90 0.72), rotation by 0.29 degrees. Not measured again on
# Mermaid since the estimator below replaced MAGSAC.
RANSAC_THRESHOLD_PX = 3.0
# MAGSAC returns the essential matrix of its best sample without refining it on the inliers:
# on synthetic points with a third of wrong matches the rotation is off by 0.28 degrees
# (0.33 with 0.5 px of noise), which biases the angle by as much. USAC_ACCURATE refines it:
# 0.00 and 0.07 degrees on the same data, for the same time.
ESSENTIAL_METHOD = cv2.USAC_ACCURATE
RANSAC_CONFIDENCE = 0.999
# The five-point algorithm needs five matches; fewer gives no model at all.
MIN_MATCHES = 5
# Share of the matches fitting an essential matrix that a single homography also explains,
# above which the scene is treated as a plane (or the camera as only turning). First guess,
# not measured yet: relief on Mermaid leaves a homography about a third of the matches.
PLANE_INLIER_SHARE = 0.8


class RotationSource(Enum):
    """Which model gave the camera's rotation, needed to isolate its displacement."""

    ESSENTIAL_MATRIX = "essential_matrix"
    # Points on one plane (flat sand seen from above) fit several essential matrices equally
    # well, and a camera that only turns fits none properly: a homography is decomposed
    # instead, keeping the plane that faces the camera.
    PLANE = "plane"


@dataclass(frozen=True)
class DisplacementEstimate:
    """None when unknown, never an invented value.

    `triangulation_angle_deg` is the median over the inliers. `rotation_deg` is how much the
    camera turned between the two images, whatever its displacement.
    """

    match_count: int
    inlier_count: int
    triangulation_angle_deg: float | None
    rotation_deg: float | None
    rotation_source: RotationSource | None
    unknown_reason: UnknownReason | None

    @property
    def is_known(self) -> bool:
        return self.unknown_reason is None


def estimate_displacement(
    features_a: Features,
    features_b: Features,
    matches: Matches,
    intrinsics: Intrinsics,
    min_inliers: int = MIN_INLIERS,
) -> DisplacementEstimate:
    """Needs the calibration (same camera for both images): rays are not defined without it."""
    if len(matches) < MIN_MATCHES:
        return _unknown(len(matches), 0, UnknownReason.TOO_FEW_MATCHES)

    points_a = undistort_to_normalized(
        intrinsics, features_a.keypoints[matches.index_pairs[:, 0]]
    )
    points_b = undistort_to_normalized(
        intrinsics, features_b.keypoints[matches.index_pairs[:, 1]]
    )
    # Points are already normalized, so the camera matrix is the identity and the pixel
    # threshold is divided by the focal length.
    identity = np.eye(3)
    essential, essential_mask = cv2.findEssentialMat(
        points_a,
        points_b,
        identity,
        method=ESSENTIAL_METHOD,
        prob=RANSAC_CONFIDENCE,
        threshold=RANSAC_THRESHOLD_PX / intrinsics.focal_px,
    )
    # OpenCV stacks several 3 × 3 candidates when the solution is ambiguous.
    if essential is None or essential.shape != (3, 3):
        return _unknown(len(matches), 0, UnknownReason.NO_MODEL)

    threshold = RANSAC_THRESHOLD_PX / intrinsics.focal_px
    homography, homography_mask = cv2.findHomography(
        points_a, points_b, cv2.USAC_MAGSAC, threshold
    )
    is_plane = (
        homography is not None
        and homography_mask.sum() >= PLANE_INLIER_SHARE * essential_mask.sum()
    )
    if is_plane:
        inliers = homography_mask.ravel() > 0
        rotation_a_to_b = _rotation_from_plane(
            homography, points_a[inliers], points_b[inliers]
        )
        source = RotationSource.PLANE
    else:
        # Of the four poses an essential matrix allows, keep the one putting the points in
        # front of both cameras; its mask drops the points that end up behind.
        _, rotation_a_to_b, _, pose_mask = cv2.recoverPose(
            essential, points_a, points_b, identity, mask=essential_mask
        )
        inliers = pose_mask.ravel() > 0
        source = RotationSource.ESSENTIAL_MATRIX
    inlier_count = int(inliers.sum())
    if inlier_count < min_inliers:
        return _unknown(len(matches), inlier_count, UnknownReason.TOO_FEW_INLIERS)

    rays_a = _unit_rays(points_a[inliers])
    # Row-wise Rᵀ·r: each ray of B expressed in A's frame, which removes the camera's rotation
    # and leaves only the effect of its displacement.
    rays_b_in_a = _unit_rays(points_b[inliers]) @ rotation_a_to_b
    angles = np.arccos(np.clip(np.sum(rays_a * rays_b_in_a, axis=1), -1.0, 1.0))
    return DisplacementEstimate(
        match_count=len(matches),
        inlier_count=inlier_count,
        triangulation_angle_deg=float(np.degrees(np.median(angles))),
        rotation_deg=_rotation_angle_deg(rotation_a_to_b),
        rotation_source=source,
        unknown_reason=None,
    )


def _rotation_from_plane(
    homography: np.ndarray, points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """Rotation from A to B among the decompositions of a homography between normalized points.

    A homography has up to four decompositions (rotation, displacement, plane normal). Those
    putting the points behind a camera are impossible; of the two that usually remain, the
    plane facing camera A most directly is kept. That is an assumption: right for a seafloor
    seen from above, not guaranteed for a wall seen at a grazing angle.
    """
    _, rotations, translations, normals = cv2.decomposeHomographyMat(
        homography, np.eye(3)
    )
    rays_a = np.column_stack([points_a, np.ones(len(points_a))])
    candidates: list[tuple[int, float, np.ndarray]] = []
    for rotation, translation, normal in zip(
        rotations, translations, normals, strict=True
    ):
        normal = normal.ravel()
        # The plane is n·X = 1 in A's frame, so the point seen along ray m is m / (n·m).
        scale = rays_a @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            points_in_a = rays_a / scale[:, None]
        depth_in_b = (points_in_a @ rotation.T + translation.ravel())[:, 2]
        in_front_of_both = int(((scale > 0) & (depth_in_b > 0)).sum())
        candidates.append((in_front_of_both, float(normal[2]), rotation))
    return max(candidates, key=lambda candidate: candidate[:2])[2]


def _unit_rays(normalized_points: np.ndarray) -> np.ndarray:
    rays = np.column_stack([normalized_points, np.ones(len(normalized_points))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _unknown(
    match_count: int, inlier_count: int, reason: UnknownReason
) -> DisplacementEstimate:
    return DisplacementEstimate(
        match_count=match_count,
        inlier_count=inlier_count,
        triangulation_angle_deg=None,
        rotation_deg=None,
        rotation_source=None,
        unknown_reason=reason,
    )
