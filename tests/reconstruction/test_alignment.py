import numpy as np
import pytest

from dataset_automation.reconstruction.alignment import fit_similarity


def rotation_about_z(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def points() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(40, 3))


def test_recovers_scale_rotation_and_translation(points: np.ndarray) -> None:
    rotation, translation = rotation_about_z(35.0), np.array([4.0, -2.0, 7.0])
    target = 2.5 * points @ rotation.T + translation

    similarity = fit_similarity(points, target)

    assert similarity.scale == pytest.approx(2.5)
    np.testing.assert_allclose(similarity.rotation, rotation, atol=1e-9)
    np.testing.assert_allclose(similarity.translation, translation, atol=1e-9)
    np.testing.assert_allclose(similarity.apply(points), target, atol=1e-9)


def test_noise_on_the_points_is_what_is_left_after_the_fit(points: np.ndarray) -> None:
    noise = np.random.default_rng(1).normal(scale=0.01, size=points.shape)
    target = 3.0 * points @ rotation_about_z(-120.0).T + noise

    similarity = fit_similarity(points, target)

    error = np.linalg.norm(similarity.apply(points) - target, axis=1)
    assert similarity.scale == pytest.approx(3.0, abs=0.01)
    assert np.median(error) < 0.03


def test_mirrored_points_are_not_fitted_by_a_reflection(points: np.ndarray) -> None:
    mirrored = points * np.array([1.0, 1.0, -1.0])

    similarity = fit_similarity(points, mirrored)

    assert np.linalg.det(similarity.rotation) == pytest.approx(1.0)
    assert np.linalg.norm(similarity.apply(points) - mirrored, axis=1).max() > 0.1


def test_points_on_a_plane_are_enough() -> None:
    flat = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 1.0, 0.0]]
    )
    target = flat @ rotation_about_z(90.0).T + np.array([1.0, 1.0, 5.0])

    similarity = fit_similarity(flat, target)

    np.testing.assert_allclose(similarity.apply(flat), target, atol=1e-9)


def test_rejects_too_few_or_mismatched_points(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_similarity(points[:2], points[:2])
    with pytest.raises(ValueError):
        fit_similarity(points, points[:-1])
    with pytest.raises(ValueError):
        fit_similarity(np.zeros((5, 3)), points[:5])
