import numpy as np
import pytest

from dataset_automation.view.scene import (
    STATUS_COLOURS,
    UNKNOWN_ANGLE_COLOUR,
    ImageStatus,
    angle_colours,
    camera_frustums,
    faded,
    footprint_outlines,
    links,
    reference_view_angles,
    sample_pairs,
    status_colours,
    trajectory,
)

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
# Camera-frame rays of the four corners of an image seeing ±0.5 of its depth sideways.
CORNER_RAYS = np.array(
    [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0]]
)


def looking_down_from(x: float, y: float, z: float) -> np.ndarray:
    """Camera-to-world transform of a nadir camera: camera z (forward) is world −z."""
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [x, y, z]
    return transform


def segment_ends(lines) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    return [
        (tuple(lines.points[first]), tuple(lines.points[second]))
        for first, second in lines.segments
    ]


def test_trajectory_joins_cameras_in_capture_order() -> None:
    centres = np.array([[0.0, 0, 1], [1, 0, 1], [1, 1, 1]])

    path = trajectory(centres)

    assert path.segments.tolist() == [[0, 1], [1, 2]]
    assert path.colours.shape == (2, 3)


def test_trajectory_of_a_single_camera_has_no_segment() -> None:
    assert len(trajectory(np.array([[0.0, 0.0, 1.0]]))) == 0


def test_frustum_apex_is_the_camera_centre_and_base_lies_below_it() -> None:
    frustums = camera_frustums(
        np.stack([looking_down_from(3.0, 4.0, 2.0)]),
        CORNER_RAYS,
        depth_m=1.0,
        colours=np.array([[1.0, 0.0, 0.0]]),
    )

    assert len(frustums) == 8
    apex, *base = frustums.points
    assert apex.tolist() == [3.0, 4.0, 2.0]
    # One metre down the optical axis, the corners spread ±0.5 m around the centre.
    assert np.allclose([corner[2] for corner in base], 1.0)
    assert sorted((corner[0], corner[1]) for corner in base) == [
        (2.5, 3.5),
        (2.5, 4.5),
        (3.5, 3.5),
        (3.5, 4.5),
    ]
    assert frustums.colours.tolist() == [[1.0, 0.0, 0.0]] * 8


def test_frustums_of_two_cameras_do_not_share_points() -> None:
    frustums = camera_frustums(
        np.stack([looking_down_from(0, 0, 1), looking_down_from(5, 0, 1)]),
        CORNER_RAYS,
        depth_m=0.5,
        colours=np.zeros((2, 3)),
    )

    first, second = frustums.segments[:8], frustums.segments[8:]
    assert set(first.ravel()) == {0, 1, 2, 3, 4}
    assert set(second.ravel()) == {5, 6, 7, 8, 9}


def test_footprint_outline_is_closed_on_the_seafloor_plane() -> None:
    outlines = footprint_outlines(
        SQUARE[None], height_m=-4.0, colours=np.array([[0.0, 1.0, 0.0]])
    )

    assert outlines.segments.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]
    assert np.all(outlines.points[:, 2] == -4.0)


def test_undefined_footprint_is_left_out_with_its_colour() -> None:
    undefined = np.full_like(SQUARE, np.nan)
    colours = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    outlines = footprint_outlines(np.stack([undefined, SQUARE]), 0.0, colours)

    assert not np.isnan(outlines.points).any()
    assert len(outlines) == 4
    assert outlines.colours.tolist() == [[0.0, 0.0, 1.0]] * 4


def test_link_joins_the_two_camera_centres() -> None:
    centres = np.array([[0.0, 0, 1], [1, 0, 1], [0, 3, 1]])

    linked = links(centres, np.array([[0, 2]]), np.array([10.0]))

    assert segment_ends(linked) == [((0.0, 0.0, 1.0), (0.0, 3.0, 1.0))]


def test_unknown_angle_gets_its_own_colour_not_one_of_the_scale() -> None:
    colours = angle_colours(np.array([np.nan, 0.0]), max_angle_deg=40.0)

    assert colours[0].tolist() == list(UNKNOWN_ANGLE_COLOUR)
    assert colours[1].tolist() != list(UNKNOWN_ANGLE_COLOUR)


def test_angles_beyond_the_scale_share_its_last_colour() -> None:
    colours = angle_colours(np.array([40.0, 75.0]), max_angle_deg=40.0)

    assert colours[0].tolist() == colours[1].tolist()


def test_status_colours_follow_the_statuses() -> None:
    colours = status_colours([ImageStatus.REMOVED, ImageStatus.KEPT])

    assert colours.tolist() == [
        list(STATUS_COLOURS[ImageStatus.REMOVED]),
        list(STATUS_COLOURS[ImageStatus.KEPT]),
    ]


def test_view_angle_is_seen_from_the_middle_of_the_shared_footprint() -> None:
    # Two cameras 1 m apart at 1 m of altitude, each seeing a 2 m square centred under it:
    # the middle of what they share is halfway between them, seen at 2 × atan(0.5).
    footprints = np.stack([SQUARE - 1.0, SQUARE - [0.0, 1.0]])
    centres = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    angles = reference_view_angles(footprints, centres, 0.0, np.array([[0, 1]]))

    assert angles[0] == pytest.approx(np.degrees(2 * np.arctan(0.5)))


def test_view_angle_is_unknown_when_footprints_share_nothing() -> None:
    footprints = np.stack([SQUARE, SQUARE + 10.0])
    centres = np.array([[1.0, 1.0, 1.0], [11.0, 11.0, 1.0]])

    angles = reference_view_angles(footprints, centres, 0.0, np.array([[0, 1]]))

    assert np.isnan(angles[0])


def test_fading_moves_colours_towards_white() -> None:
    colours = faded(np.array([[0.0, 0.5, 1.0]]), towards_white=0.5)

    assert colours.tolist() == [[0.5, 0.75, 1.0]]


def test_sample_keeps_pairs_and_angles_together() -> None:
    pairs = np.column_stack([np.arange(100), np.arange(100) + 200])
    angles = np.arange(100, dtype=float)

    sampled_pairs, sampled_angles = sample_pairs(pairs, angles, count=10)

    assert len(sampled_pairs) == 10
    assert sampled_angles.tolist() == sampled_pairs[:, 0].astype(float).tolist()
    # The same seed draws the same links, so two screenshots can be compared.
    assert sample_pairs(pairs, angles, count=10)[0].tolist() == sampled_pairs.tolist()


def test_sample_returns_everything_when_under_the_limit() -> None:
    pairs = np.array([[0, 1], [2, 3]])

    sampled_pairs, _ = sample_pairs(pairs, np.array([1.0, 2.0]), count=10)

    assert sampled_pairs.tolist() == pairs.tolist()
