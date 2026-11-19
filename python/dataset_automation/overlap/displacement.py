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

# Measured on the 1,243 consecutive Mermaid pairs against the reference poses (see docs/overlap_journal.md).
RANSAC_THRESHOLD_PX = 3.0
RANSAC_CONFIDENCE = 0.999
# The five-point algorithm needs five matches; fewer gives no model at all.
MIN_MATCHES = 5
# No single estimator is reliable: on about 1 % of the consecutive Mermaid pairs each of them
# returns an essential matrix that fits the matches along the epipolar lines but puts half of
# them behind a camera, and reads a 5 degree displacement as 1 degree. They rarely fail on the
# same pairs, so each one proposes its poses and the pose supported by most points wins.
# MAGSAC alone also leaves the rotation 0.3 degrees off (it does not refine on the inliers).
ESSENTIAL_METHODS = (cv2.USAC_ACCURATE, cv2.USAC_MAGSAC, cv2.LMEDS)
# Share of the points supporting the best essential-matrix pose that one homography also maps,
# above which the scene is a plane (or the camera only turned). First guess, to settle on a
# dataset with flat seafloor: relief on Mermaid leaves a homography about a third of the matches.
PLANE_INLIER_SHARE = 0.8
# On a plane several poses are supported equally and counting cannot choose: the plane model
# is then preferred when supported by nearly as many points, and so is the plane facing the
# camera among the decompositions of a homography. Elsewhere the plane model must be clearly
# better supported to win: with a small displacement even a wrong pose fits the epipolar
# lines of every match (measured: 0.84 degrees of invented rotation for a 10 cm step at 2 m).
NEAR_TIE = 0.98
# Below this (in units of the distance to the plane) a decomposed displacement is noise: the
# camera only turned, and the direction of the displacement is undefined.
NO_DISPLACEMENT = 1e-6


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


@dataclass(frozen=True, eq=False)
class _Pose:
    """Camera B relative to camera A: a point X of A's frame is R·X + t in B's."""

    rotation: np.ndarray
    translation: np.ndarray
    supported: np.ndarray  # boolean, one entry per match

    @property
    def support(self) -> int:
        return int(self.supported.sum())


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
    # Points are normalized, so the pixel threshold is divided by the focal length.
    threshold = RANSAC_THRESHOLD_PX / intrinsics.focal_px

    essential_pose = _best_essential_pose(points_a, points_b, threshold)
    plane = _best_plane_pose(points_a, points_b, threshold)
    if essential_pose is None and plane is None:
        return _unknown(len(matches), 0, UnknownReason.NO_MODEL)
    if plane is not None and _prefers_plane(plane, essential_pose):
        pose, source = plane.pose, RotationSource.PLANE
    else:
        assert essential_pose is not None
        pose, source = essential_pose, RotationSource.ESSENTIAL_MATRIX
    if pose.support < min_inliers:
        return _unknown(len(matches), pose.support, UnknownReason.TOO_FEW_INLIERS)

    rays_a = _unit_rays(points_a[pose.supported])
    # Row-wise Rᵀ·r: each ray of B expressed in A's frame, which removes the camera's rotation
    # and leaves only the effect of its displacement.
    rays_b_in_a = _unit_rays(points_b[pose.supported]) @ pose.rotation
    angles = np.arccos(np.clip(np.sum(rays_a * rays_b_in_a, axis=1), -1.0, 1.0))
    return DisplacementEstimate(
        match_count=len(matches),
        inlier_count=pose.support,
        triangulation_angle_deg=float(np.degrees(np.median(angles))),
        rotation_deg=_rotation_angle_deg(pose.rotation),
        rotation_source=source,
        unknown_reason=None,
    )


@dataclass(frozen=True, eq=False)
class _PlanePose:
    pose: _Pose
    mapped_count: int  # matches the homography itself maps within the threshold


def _prefers_plane(plane: _PlanePose, essential_pose: _Pose | None) -> bool:
    if essential_pose is None:
        return True
    is_plane = plane.mapped_count >= PLANE_INLIER_SHARE * essential_pose.support
    if is_plane:
        return plane.pose.support >= NEAR_TIE * essential_pose.support
    return plane.pose.support * NEAR_TIE > essential_pose.support


def _best_essential_pose(
    points_a: np.ndarray, points_b: np.ndarray, threshold: float
) -> _Pose | None:
    poses: list[_Pose] = []
    for method in ESSENTIAL_METHODS:
        essential, _ = cv2.findEssentialMat(
            points_a,
            points_b,
            np.eye(3),
            method=method,
            prob=RANSAC_CONFIDENCE,
            threshold=threshold,
        )
        # OpenCV stacks several 3 × 3 candidates when the solution is ambiguous.
        if essential is None or essential.shape != (3, 3):
            continue
        # An essential matrix allows four poses; only one puts the points in front of both
        # cameras, and counting the supported points finds it.
        first_rotation, second_rotation, direction = cv2.decomposeEssentialMat(
            essential
        )
        for rotation in (first_rotation, second_rotation):
            for translation in (direction, -direction):
                poses.append(
                    _pose(rotation, translation, points_a, points_b, threshold)
                )
    return max(poses, key=lambda pose: pose.support, default=None)


def _best_plane_pose(
    points_a: np.ndarray, points_b: np.ndarray, threshold: float
) -> _PlanePose | None:
    """Pose from the decompositions of a homography, keeping the plane that faces camera A.

    Facing the camera is an assumption: right for a seafloor seen from above, not guaranteed
    for a wall seen at a grazing angle.
    """
    if len(points_a) < 4:
        return None
    homography, mapped = cv2.findHomography(
        points_a, points_b, cv2.USAC_MAGSAC, threshold
    )
    if homography is None:
        return None
    _, rotations, translations, normals = cv2.decomposeHomographyMat(
        homography, np.eye(3)
    )
    candidates = [
        (
            _pose(rotation, translation, points_a, points_b, threshold),
            float(normal[2, 0]),
        )
        for rotation, translation, normal in zip(
            rotations, translations, normals, strict=True
        )
    ]
    most_supported = max(pose.support for pose, _ in candidates)
    near_ties = [
        candidate
        for candidate in candidates
        if candidate[0].support >= NEAR_TIE * most_supported
    ]
    facing_camera = max(near_ties, key=lambda candidate: candidate[1])[0]
    return _PlanePose(pose=facing_camera, mapped_count=int(mapped.sum()))


def _pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold: float,
) -> _Pose:
    translation = translation.ravel()
    rays_a_in_b = _homogeneous(points_a) @ rotation.T
    rays_b = _homogeneous(points_b)
    length = float(np.linalg.norm(translation))
    if length < NO_DISPLACEMENT:
        # The camera only turned: a match is supported when the rotation alone maps it.
        with np.errstate(divide="ignore", invalid="ignore"):
            predicted = rays_a_in_b[:, :2] / rays_a_in_b[:, 2:]
        distance = np.linalg.norm(predicted - points_b, axis=1)
        supported = (rays_a_in_b[:, 2] > 0) & (distance < threshold)
        return _Pose(rotation, translation, supported)

    direction = translation / length
    # Sampson distance to the epipolar geometry E = [t]× R, in normalized image units.
    epipolar_lines_in_b = np.cross(direction, rays_a_in_b)
    epipolar_lines_in_a = np.cross(rays_b, direction) @ rotation
    residual = np.abs(np.sum(rays_b * epipolar_lines_in_b, axis=1)) / np.sqrt(
        np.sum(epipolar_lines_in_b[:, :2] ** 2, axis=1)
        + np.sum(epipolar_lines_in_a[:, :2] ** 2, axis=1)
    )
    # Depths along both rays: least squares of depth_a·(R·a) − depth_b·b = −t per match.
    aa = np.sum(rays_a_in_b * rays_a_in_b, axis=1)
    bb = np.sum(rays_b * rays_b, axis=1)
    ab = np.sum(rays_a_in_b * rays_b, axis=1)
    at, bt = rays_a_in_b @ direction, rays_b @ direction
    with np.errstate(divide="ignore", invalid="ignore"):
        determinant = aa * bb - ab**2
        depth_a = (bt * ab - at * bb) / determinant
        depth_b = (bt * aa - at * ab) / determinant
    supported = (residual < threshold) & (depth_a > 0) & (depth_b > 0)
    return _Pose(rotation, translation, supported)


def _homogeneous(normalized_points: np.ndarray) -> np.ndarray:
    return np.column_stack([normalized_points, np.ones(len(normalized_points))])


def _unit_rays(normalized_points: np.ndarray) -> np.ndarray:
    rays = _homogeneous(normalized_points)
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
