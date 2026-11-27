import numpy as np
import pytest

from dataset_automation.reference.revisits import far_pairs

SEAFLOOR_HEIGHT = 0.0
SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def cameras(
    positions: list[tuple[float, float]], altitude_m: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """One camera per (x, y), each seeing a 2 m square centred under it."""
    footprints = np.stack([SQUARE + [x - 1.0, y - 1.0] for x, y in positions])
    centres = np.array([[x, y, altitude_m] for x, y in positions])
    return footprints, centres


def test_camera_coming_back_over_a_place_makes_a_far_pair() -> None:
    # Out along x, then back: the last camera is 0.5 m from the first.
    footprints, centres = cameras([(0, 0), (3, 0), (6, 0), (3, 5), (0.5, 0)])

    pairs = far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=1)

    assert list(zip(pairs.index_a, pairs.index_b, strict=True)) == [(0, 4)]
    # Squares of 2 m shifted by 0.5 m share 1.5 × 2 of their 2 × 2.
    assert pairs.mutual_overlap[0] == pytest.approx(0.75)
    assert pairs.iou[0] == pytest.approx(3.0 / 5.0)
    assert pairs.revisits().tolist() == [True]


def test_neighbours_in_time_are_left_out() -> None:
    footprints, centres = cameras([(0, 0), (0.5, 0), (1.0, 0)])

    assert len(far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=2)) == 0
    assert len(far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=1)) == 1


def test_view_angle_is_the_angle_between_the_two_cameras_seen_from_the_ground() -> None:
    # 1 m apart at 1 m of altitude, seen from the middle of what they share: 2 × atan(0.5).
    footprints, centres = cameras([(0, 0), (9, 9), (1, 0)])

    pairs = far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=1)

    assert pairs.view_angle_deg[0] == pytest.approx(2 * np.degrees(np.arctan(0.5)))


def test_same_viewpoint_again_has_no_view_angle() -> None:
    footprints, centres = cameras([(0, 0), (9, 9), (0, 0)])

    pairs = far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=1)

    assert pairs.mutual_overlap[0] == pytest.approx(1.0)
    assert pairs.view_angle_deg[0] == pytest.approx(0.0, abs=1e-4)


def test_mutual_overlap_is_the_smaller_share() -> None:
    # A close-up inside a wide view: fully seen one way, a quarter the other way.
    footprints, centres = cameras([(0, 0), (9, 9), (0, 0)])
    footprints[2] = SQUARE * 2.0 - 2.0

    pairs = far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=1)

    assert pairs.mutual_overlap[0] == pytest.approx(0.25)
    assert pairs.revisits().tolist() == [False]


def test_camera_without_a_footprint_makes_no_pair() -> None:
    footprints, centres = cameras([(0, 0), (9, 9), (0.5, 0)])
    footprints[2] = np.nan

    assert len(far_pairs(footprints, centres, SEAFLOOR_HEIGHT, min_gap=1)) == 0
