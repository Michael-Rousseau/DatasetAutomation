import numpy as np
import pytest

from dataset_automation.reference.observability import (
    depth_noise_ratio,
    grid_over,
    inside,
    observe,
    reconstructible,
)

SEAFLOOR_HEIGHT = 0.0
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def scene(camera_x: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Cameras 1 m above the seafloor at the given x, all seeing the same unit square."""
    footprints = np.stack([SQUARE] * len(camera_x))
    centres = np.array([[x, 0.5, 1.0] for x in camera_x])
    return footprints, centres


def centre_cell(values: np.ndarray) -> float:
    return float(values[values.shape[0] // 2, values.shape[1] // 2])


def test_counts_the_images_seeing_each_cell() -> None:
    footprints, centres = scene([0.0, 0.5, 1.0])
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    assert observation.views.shape == (5, 5)
    assert (observation.views == 3).all()


def test_fan_is_the_angle_spanned_by_the_viewing_directions() -> None:
    # Cameras 1 m apart at 1 m of altitude: slopes differ by 1, so the fan is 45 degrees.
    footprints, centres = scene([0.0, 0.5, 1.0])
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    assert centre_cell(observation.fan_deg) == pytest.approx(45.0)


def test_images_taken_from_the_same_place_reconstruct_nothing() -> None:
    footprints, centres = scene([0.5, 0.5, 0.5])
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    assert (observation.views == 3).all()
    assert (observation.fan_deg == 0).all()
    assert not reconstructible(observation).any()


def test_two_views_are_not_enough_by_default() -> None:
    footprints, centres = scene([0.0, 0.5, 1.0])
    grid = grid_over(footprints, cell_m=0.2)

    assert reconstructible(
        observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))
    ).all()
    assert not reconstructible(
        observe(grid, footprints, centres, SEAFLOOR_HEIGHT, [0, 2])
    ).any()


def test_cells_outside_every_footprint_are_not_seen() -> None:
    far_square = SQUARE + [3.0, 0.0]
    footprints = np.stack([SQUARE, far_square])
    centres = np.array([[0.5, 0.5, 1.0], [3.5, 0.5, 1.0]])
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(2))

    between = (grid.x_m > 1.0) & (grid.x_m < 3.0)
    assert (observation.views[:, between] == 0).all()
    assert (observation.views[:, ~between] == 1).all()


def test_camera_without_a_footprint_is_ignored() -> None:
    footprints, centres = scene([0.0, 1.0])
    footprints[1] = np.nan
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(2))

    assert (observation.views == 1).all()


def test_depth_information_grows_with_the_spread_of_the_views() -> None:
    footprints, centres = scene([0.0, 0.5, 1.0])
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    # Slopes −0.5, 0 and 0.5 around their mean 0.
    assert centre_cell(observation.depth_information) == pytest.approx(0.5)


def test_views_from_the_same_place_carry_no_depth_information() -> None:
    footprints, centres = scene([0.5, 0.5, 0.5])
    grid = grid_over(footprints, cell_m=0.2)

    observation = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    assert (observation.depth_information == 0).all()


def test_depth_noise_depends_on_which_view_is_removed() -> None:
    footprints, centres = scene([0.0, 0.5, 1.0])
    grid = grid_over(footprints, cell_m=0.2)
    everything = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    without_the_middle = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, [0, 2])
    without_an_end = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, [0, 1])

    # Seen from the centre cell, the middle camera sits at the mean direction: it adds nothing.
    assert centre_cell(depth_noise_ratio(everything, without_the_middle)) == (
        pytest.approx(1.0)
    )
    assert centre_cell(depth_noise_ratio(everything, without_an_end)) == (
        pytest.approx(2.0)
    )


def test_depth_noise_is_infinite_where_a_single_position_is_left() -> None:
    footprints, centres = scene([0.0, 0.5, 1.0])
    grid = grid_over(footprints, cell_m=0.2)
    everything = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, range(3))

    one_view = observe(grid, footprints, centres, SEAFLOOR_HEIGHT, [1])

    assert np.isinf(depth_noise_ratio(everything, one_view)).all()


def test_inside_drops_the_cells_near_the_edge_of_an_area() -> None:
    footprints, _ = scene([0.5])
    grid = grid_over(footprints * 2.0, cell_m=0.2)  # 10 × 10 cells of 20 cm
    area = np.ones((10, 10), dtype=bool)

    core = inside(area, grid, margin_m=0.5)

    # 0.5 m is 2.5 cells: the two outer rings are too close to a cell outside the area.
    assert core.sum() == 36
    assert core[2:8, 2:8].all()
