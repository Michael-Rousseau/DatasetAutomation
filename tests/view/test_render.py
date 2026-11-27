import numpy as np
import pytest

from dataset_automation.view.scene import LineSet

# Open3D is an optional extra: CI installs the project without it.
pytest.importorskip("open3d")

from dataset_automation.view.render import (
    to_open3d,
    viewpoint,
)


def test_open3d_line_set_keeps_points_segments_and_colours() -> None:
    lines = LineSet(
        points=np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        segments=np.array([[0, 1], [1, 2]]),
        colours=np.array([[1.0, 0, 0], [0, 0, 1]]),
    )

    converted = to_open3d(lines)

    assert np.asarray(converted.points).tolist() == lines.points.tolist()
    assert np.asarray(converted.lines).tolist() == lines.segments.tolist()
    assert np.asarray(converted.colors).tolist() == lines.colours.tolist()


def test_viewpoint_looks_at_the_middle_from_above_and_south() -> None:
    points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 2.0]])

    centre, eye = viewpoint(points)

    assert centre.tolist() == [5.0, 5.0, 1.0]
    assert eye[1] < centre[1]
    assert eye[2] > centre[2]
