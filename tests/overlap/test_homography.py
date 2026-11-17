import numpy as np
from shapely import box

from dataset_automation.overlap.homography import fit_homography, is_plausible

FRAME = box(-0.5, -0.5, 639.5, 479.5)


def test_identity_is_plausible() -> None:
    assert is_plausible(np.eye(3), FRAME, FRAME)


def test_moderate_rotation_and_zoom_is_plausible() -> None:
    angle = np.radians(30)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    assert is_plausible(rotation @ np.diag([1.5, 1.5, 1.0]), FRAME, FRAME)


def test_rejects_mirrored_view() -> None:
    mirror = np.array([[-1.0, 0.0, 639.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert not is_plausible(mirror, FRAME, FRAME)


def test_rejects_extreme_scale_change() -> None:
    assert not is_plausible(np.diag([10.0, 10.0, 1.0]), FRAME, FRAME)


def test_rejects_homography_sending_part_of_the_image_beyond_the_horizon() -> None:
    # w = 1 − x / 300 turns negative on the right half of the image.
    beyond = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1 / 300, 0.0, 1.0]])

    assert not is_plausible(beyond, FRAME, FRAME)


def test_fit_needs_four_matches() -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    assert fit_homography(points, points, threshold_px=3.0) is None


def test_fit_recovers_a_translation_and_flags_outliers() -> None:
    rng = np.random.default_rng(0)
    points_a = rng.uniform(0, 600, size=(50, 2))
    points_b = points_a + [25.0, -10.0]
    points_b[:5] = rng.uniform(0, 600, size=(5, 2))  # five wrong matches

    fit = fit_homography(points_a, points_b, threshold_px=3.0)

    assert fit is not None
    np.testing.assert_allclose(
        fit.matrix / fit.matrix[2, 2], [[1, 0, 25], [0, 1, -10], [0, 0, 1]], atol=1e-4
    )
    assert fit.inlier_count == 45
    assert not fit.inliers[:5].any()
