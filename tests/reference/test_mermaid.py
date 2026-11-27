from pathlib import Path

import numpy as np
import pytest

from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.reference.mermaid import MIN_GSD_M, load_mermaid_reference

# Two cameras looking straight down, 2 m and 3 m high, 1 m apart.
TWO_NADIR_CAMERAS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<camera_poses>
  <camera id="0" label="low">
    <transform>1 0 0 0  0 -1 0 0  0 0 -1 2  0 0 0 1</transform>
    <orientation>1</orientation>
  </camera>
  <camera id="1" label="high">
    <transform>1 0 0 1  0 -1 0 0  0 0 -1 3  0 0 0 1</transform>
    <orientation>1</orientation>
  </camera>
</camera_poses>
"""


@pytest.fixture
def poses_path(tmp_path: Path) -> Path:
    path = tmp_path / "poses.xml"
    path.write_text(TWO_NADIR_CAMERAS_XML)
    return path


def test_seafloor_sits_below_the_lowest_camera_by_the_altitude_of_the_finest_gsd(
    poses_path: Path,
) -> None:
    reference = load_mermaid_reference(poses_path)

    lowest_altitude = MIN_GSD_M * MERMAID_INTRINSICS.focal_px
    assert reference.seafloor_height_m == pytest.approx(2.0 - lowest_altitude, abs=0.02)


def test_footprint_lies_under_its_camera_and_grows_with_altitude(
    poses_path: Path,
) -> None:
    reference = load_mermaid_reference(poses_path)

    assert reference.footprints.shape[0] == 2
    low, high = reference.footprints
    assert low.mean(axis=0) == pytest.approx(reference.centres[0, :2], abs=0.05)
    assert np.ptp(high[:, 0]) > np.ptp(low[:, 0])
