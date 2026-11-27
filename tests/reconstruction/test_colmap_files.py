import io
from pathlib import Path

import numpy as np
import pytest

from dataset_automation.reconstruction.colmap_files import (
    read_camera_centres,
    write_match_list,
)

HEADER = (
    "# Image list with two lines of data per image:\n"
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    "# Number of images: 3, mean observations per image: 1\n"
)


def test_match_list_has_one_pair_of_names_per_line() -> None:
    file = io.StringIO()

    count = write_match_list(file, [("a/1.jpg", "a/2.jpg"), ("a/1.jpg", "b/7.jpg")])

    assert count == 2
    assert file.getvalue() == "a/1.jpg a/2.jpg\na/1.jpg b/7.jpg\n"


def test_centre_is_where_the_camera_is_not_its_translation(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    half = np.sqrt(0.5)
    images.write_text(
        HEADER
        # No rotation: the camera sits at −t.
        + "1 1 0 0 0 1 2 3 1 line/0001.JPG\n"
        + "10.5 20.5 -1 30.0 40.0 7\n"
        # World-to-camera rotation of 90° about z, and no 2D point at all.
        + f"2 {half} 0 0 {half} 1 0 0 1 line/0002.JPG\n"
        + "\n"
        + "3 1 0 0 0 0 0 0 1 name with spaces.jpg\n"
        + "1.0 2.0 5\n"
    )

    centres = read_camera_centres(images)

    assert list(centres) == ["line/0001.JPG", "line/0002.JPG", "name with spaces.jpg"]
    np.testing.assert_allclose(centres["line/0001.JPG"], [-1.0, -2.0, -3.0])
    # Rᵀ turns t = (1, 0, 0) into (0, −1, 0); the centre is its opposite.
    np.testing.assert_allclose(centres["line/0002.JPG"], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(centres["name with spaces.jpg"], [0.0, 0.0, 0.0])


def test_model_without_any_image_gives_no_centre(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text(HEADER)

    assert read_camera_centres(images) == {}


def test_file_that_is_not_a_model_is_refused(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text("1 2 3\n\n")

    with pytest.raises(ValueError):
        read_camera_centres(images)
