import cv2
import numpy as np
import pytest

from dataset_automation.camera.intrinsics import Intrinsics, camera_matrix
from dataset_automation.features.interface import Features, Matches
from dataset_automation.overlap.displacement import (
    DisplacementEstimate,
    RotationSource,
    estimate_displacement,
)
from dataset_automation.overlap.pairwise import UnknownReason

# No distortion: pixels are then an exact projection, with no lens model to invert.
PINHOLE = Intrinsics(
    width_px=1600,
    height_px=1200,
    focal_px=1000.0,
    cx_offset_px=0.0,
    cy_offset_px=0.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
)
POINT_COUNT = 200
NO_ROTATION = np.eye(3)


def seafloor_points(relief_m: float = 0.5, seed: int = 0) -> np.ndarray:
    """Points about 2 m in front of camera A, in A's frame, spread over the image."""
    random = np.random.default_rng(seed)
    return np.column_stack(
        [
            random.uniform(-1.0, 1.0, POINT_COUNT),
            random.uniform(-0.8, 0.8, POINT_COUNT),
            2.0 + random.uniform(-relief_m, relief_m, POINT_COUNT),
        ]
    )


def rotation_about_y(degrees: float) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.array([0.0, np.radians(degrees), 0.0]))
    return rotation


def project(points: np.ndarray) -> np.ndarray:
    pixels = points @ camera_matrix(PINHOLE).T
    return pixels[:, :2] / pixels[:, 2:]


def as_features(pixels: np.ndarray) -> Features:
    return Features(
        keypoints=pixels,
        descriptors=np.zeros((len(pixels), 1), dtype=np.float32),
        scores=np.ones(len(pixels)),
    )


def identity_matches(count: int) -> Matches:
    indices = np.arange(count)
    return Matches(
        index_pairs=np.column_stack([indices, indices]), scores=np.ones(count)
    )


def displacement(
    points: np.ndarray, centre_b: np.ndarray, rotation_a_to_b: np.ndarray
) -> DisplacementEstimate:
    """Camera A at the origin; camera B at `centre_b` (in A's frame), turned by the rotation."""
    points_in_b = (points - centre_b) @ rotation_a_to_b.T
    return estimate_displacement(
        as_features(project(points)),
        as_features(project(points_in_b)),
        identity_matches(len(points)),
        PINHOLE,
    )


def true_angle_deg(points: np.ndarray, centre_b: np.ndarray) -> float:
    to_a = points / np.linalg.norm(points, axis=1, keepdims=True)
    to_b = points - centre_b
    to_b /= np.linalg.norm(to_b, axis=1, keepdims=True)
    return float(np.degrees(np.median(np.arccos(np.sum(to_a * to_b, axis=1)))))


def test_measures_the_angle_under_which_a_sideways_step_is_seen() -> None:
    points, step = seafloor_points(), np.array([0.2, 0.0, 0.0])

    estimate = displacement(points, step, NO_ROTATION)

    assert estimate.rotation_source is RotationSource.ESSENTIAL_MATRIX
    assert estimate.triangulation_angle_deg == pytest.approx(
        true_angle_deg(points, step), abs=0.02
    )
    # 0.2 m seen from about 2 m.
    assert estimate.triangulation_angle_deg == pytest.approx(5.5, abs=0.5)


def test_turning_the_camera_does_not_change_the_angle() -> None:
    points, step = seafloor_points(), np.array([0.2, 0.0, 0.0])

    estimate = displacement(points, step, rotation_about_y(8.0))

    assert estimate.triangulation_angle_deg == pytest.approx(
        true_angle_deg(points, step), abs=0.02
    )
    assert estimate.rotation_deg == pytest.approx(8.0, abs=0.02)


@pytest.mark.parametrize("step_m", [0.05, 0.1])
def test_small_step_over_relief_is_not_explained_away_by_a_rotation(
    step_m: float,
) -> None:
    # With a small step a wrong pose still fits every epipolar line: the plane model once won
    # here and invented 0.8 degrees of rotation.
    points, step = seafloor_points(), np.array([step_m, 0.0, 0.0])

    estimate = displacement(points, step, rotation_about_y(4.0))

    assert estimate.triangulation_angle_deg == pytest.approx(
        true_angle_deg(points, step), abs=0.02
    )
    assert estimate.rotation_deg == pytest.approx(4.0, abs=0.02)


def test_a_larger_step_gives_a_larger_angle() -> None:
    points = seafloor_points()

    small = displacement(points, np.array([0.1, 0.0, 0.0]), NO_ROTATION)
    large = displacement(points, np.array([0.4, 0.0, 0.0]), NO_ROTATION)

    assert small.triangulation_angle_deg is not None
    assert large.triangulation_angle_deg == pytest.approx(
        4 * small.triangulation_angle_deg, rel=0.05
    )


def test_camera_turning_on_the_spot_has_no_triangulation_angle() -> None:
    estimate = displacement(seafloor_points(), np.zeros(3), rotation_about_y(8.0))

    assert estimate.rotation_source is RotationSource.PLANE
    assert estimate.triangulation_angle_deg == pytest.approx(0.0, abs=0.01)
    assert estimate.rotation_deg == pytest.approx(8.0, abs=0.01)


@pytest.mark.parametrize("turn_deg", [0.0, 8.0])
def test_flat_seafloor_seen_from_above_is_measured_like_relief(turn_deg: float) -> None:
    # Coplanar points fit several essential matrices: the plane model must take over.
    points, step = seafloor_points(relief_m=0.0), np.array([0.2, 0.0, 0.0])

    estimate = displacement(points, step, rotation_about_y(turn_deg))

    assert estimate.rotation_source is RotationSource.PLANE
    assert estimate.triangulation_angle_deg == pytest.approx(
        true_angle_deg(points, step), abs=0.02
    )
    assert estimate.rotation_deg == pytest.approx(turn_deg, abs=0.02)


def test_flat_seafloor_seen_at_an_angle_is_measured_too() -> None:
    points = seafloor_points(relief_m=0.0) @ rotation_about_y(25.0).T + [0.0, 0.0, 0.5]
    step = np.array([0.2, 0.05, 0.0])

    estimate = displacement(points, step, rotation_about_y(3.0))

    assert estimate.rotation_source is RotationSource.PLANE
    assert estimate.triangulation_angle_deg == pytest.approx(
        true_angle_deg(points, step), abs=0.02
    )


def test_wrong_matches_do_not_move_the_angle() -> None:
    points, step = seafloor_points(), np.array([0.2, 0.0, 0.0])
    pixels_b = project(points - step)
    # A third of the matches point to another keypoint of B.
    wrong = np.arange(len(points)) % 3 == 0
    pixels_b[wrong] = np.roll(pixels_b[wrong], 7, axis=0)

    estimate = estimate_displacement(
        as_features(project(points)),
        as_features(pixels_b),
        identity_matches(len(points)),
        PINHOLE,
    )

    assert estimate.triangulation_angle_deg == pytest.approx(
        true_angle_deg(points[~wrong], step), abs=0.02
    )
    assert estimate.inlier_count <= (~wrong).sum() + 3


def test_too_few_matches_are_unknown() -> None:
    points = seafloor_points()[:4]

    estimate = displacement(points, np.array([0.2, 0.0, 0.0]), NO_ROTATION)

    assert estimate.unknown_reason is UnknownReason.TOO_FEW_MATCHES
    assert estimate.triangulation_angle_deg is None
    assert estimate.rotation_deg is None
    assert estimate.rotation_source is None


def test_too_few_inliers_are_unknown() -> None:
    points = seafloor_points()[:10]

    estimate = displacement(points, np.array([0.2, 0.0, 0.0]), NO_ROTATION)

    assert estimate.unknown_reason is UnknownReason.TOO_FEW_INLIERS
    assert estimate.inlier_count <= 10
