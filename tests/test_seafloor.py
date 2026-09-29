import numpy as np
import pytest

from dataset_automation.seafloor import (
    HorizontalPlane,
    altitude_from_gsd,
    plane_below_lowest_cameras,
)


def test_vertical_ray_hits_plane_right_below_origin() -> None:
    hits = HorizontalPlane(height_m=-2.0).intersect(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, -1.0])
    )

    np.testing.assert_allclose(hits, [1.0, 2.0, -2.0])


def test_oblique_ray_travels_horizontally_before_hitting_plane() -> None:
    # 45° downwards from 1 m above the plane: lands 1 m further along x.
    hits = HorizontalPlane(height_m=0.0).intersect(
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0])
    )

    np.testing.assert_allclose(hits, [1.0, 0.0, 0.0])


def test_direction_length_does_not_change_hit_point() -> None:
    plane = HorizontalPlane(height_m=0.0)
    origin = np.array([0.0, 0.0, 2.0])

    np.testing.assert_allclose(
        plane.intersect(origin, np.array([0.3, 0.1, -1.0])),
        plane.intersect(origin, np.array([3.0, 1.0, -10.0])),
    )


def test_ray_pointing_up_misses_plane() -> None:
    hits = HorizontalPlane(height_m=0.0).intersect(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    )

    assert np.isnan(hits).all()


def test_horizontal_ray_misses_plane() -> None:
    hits = HorizontalPlane(height_m=0.0).intersect(
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    )

    assert np.isnan(hits).all()


def test_intersects_many_cameras_and_rays_at_once() -> None:
    origins = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 2.0]])[
        :, None, :
    ]  # (2 cameras, 1, 3)
    directions = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]])[
        None, :, :
    ]  # (1, 3 rays, 3)

    hits = HorizontalPlane(height_m=0.0).intersect(origins, directions)

    assert hits.shape == (2, 3, 3)
    np.testing.assert_allclose(hits[1, 1], [7.0, 0.0, 0.0])
    assert np.isnan(hits[:, 2]).all()
    assert not np.isnan(hits[:, :2]).any()


def test_altitude_from_gsd_scales_with_focal_length() -> None:
    assert altitude_from_gsd(0.0005, 2000.0) == pytest.approx(1.0)


def test_plane_sits_lowest_altitude_below_low_percentile_camera() -> None:
    heights = np.linspace(-3.0, 1.0, 101)

    plane = plane_below_lowest_cameras(
        heights, lowest_altitude_m=1.5, lowest_percentile=0.0
    )

    assert plane.height_m == pytest.approx(-4.5)


def test_hit_behind_the_origin_is_rejected() -> None:
    # A camera below the plane looking down would otherwise "see" the plane behind it.
    hits = HorizontalPlane(height_m=0.0).intersect(
        np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0])
    )

    assert np.isnan(hits).all()


def test_origin_below_plane_hits_it_looking_up() -> None:
    hits = HorizontalPlane(height_m=0.0).intersect(
        np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    )

    np.testing.assert_allclose(hits, [0.0, 0.0, 0.0])
