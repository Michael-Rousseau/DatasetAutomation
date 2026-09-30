from pathlib import Path

import numpy as np
import pytest

from dataset_automation.reference.poses import (
    camera_centres,
    load_camera_poses,
    optical_axes,
    parse_camera_poses,
)

MERMAID_POSES = Path(__file__).parents[2] / "data" / "mermaid" / "107177.xml"

# Camera A: 90° rotation about z, centre (1, 2, 3). Camera B: looks straight down, no covariance.
TWO_CAMERAS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<camera_poses>
  <camera id="0" label="A">
    <transform>0 -1 0 1  1 0 0 2  0 0 1 3  0 0 0 1</transform>
    <rotation_covariance>1 0 0 0 1 0 0 0 1</rotation_covariance>
    <location_covariance>2 0 0 0 2 0 0 0 2</location_covariance>
    <orientation>1</orientation>
  </camera>
  <camera id="1" label="B">
    <transform>1 0 0 0  0 -1 0 0  0 0 -1 5  0 0 0 1</transform>
    <orientation>1</orientation>
  </camera>
</camera_poses>
"""


def test_parses_transform_as_row_major_4x4() -> None:
    poses = parse_camera_poses(TWO_CAMERAS_XML)

    assert poses.labels == ["A", "B"]
    assert poses.camera_to_world.shape == (2, 4, 4)
    np.testing.assert_array_equal(poses.camera_to_world[0, 0], [0, -1, 0, 1])


def test_missing_covariance_becomes_nan() -> None:
    poses = parse_camera_poses(TWO_CAMERAS_XML)

    np.testing.assert_array_equal(poses.location_covariance[0], 2 * np.eye(3))
    assert np.isnan(poses.location_covariance[1]).all()
    assert np.isnan(poses.rotation_covariance[1]).all()


def test_camera_centre_is_translation_column() -> None:
    poses = parse_camera_poses(TWO_CAMERAS_XML)

    np.testing.assert_array_equal(camera_centres(poses), [[1, 2, 3], [0, 0, 5]])


def test_optical_axis_is_camera_z_in_world() -> None:
    poses = parse_camera_poses(TWO_CAMERAS_XML)

    np.testing.assert_array_equal(optical_axes(poses)[1], [0, 0, -1])


def test_rejects_non_rigid_transform() -> None:
    scaled = TWO_CAMERAS_XML.replace("1 0 0 0  0 -1 0 0", "2 0 0 0  0 -1 0 0")

    with pytest.raises(ValueError, match="camera B: transform is not a rigid motion"):
        parse_camera_poses(scaled)


def test_rejects_reflection() -> None:
    mirrored = TWO_CAMERAS_XML.replace("0 0 -1 5", "0 0 1 5")

    with pytest.raises(ValueError, match="camera B"):
        parse_camera_poses(mirrored)


def test_rejects_unsupported_orientation() -> None:
    rotated_exif = TWO_CAMERAS_XML.replace(
        "<orientation>1</orientation>", "<orientation>6</orientation>", 1
    )

    with pytest.raises(ValueError, match="unsupported orientation"):
        parse_camera_poses(rotated_exif)


def test_rejects_duplicate_labels() -> None:
    duplicated = TWO_CAMERAS_XML.replace('label="B"', 'label="A"')

    with pytest.raises(ValueError, match="not unique"):
        parse_camera_poses(duplicated)


@pytest.mark.skipif(not MERMAID_POSES.exists(), reason="Mermaid dataset not downloaded")
def test_loads_all_1244_mermaid_cameras() -> None:
    poses = load_camera_poses(MERMAID_POSES)

    assert len(poses) == 1244
    assert poses.labels[0] == "SR202204_LDM-S_D01_0000"
    # Camera 511 is the only one exported without covariances.
    assert np.isnan(poses.location_covariance).any(axis=(1, 2)).sum() == 1
