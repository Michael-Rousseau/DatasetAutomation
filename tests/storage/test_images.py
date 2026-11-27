import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset_automation.storage.database import open_database
from dataset_automation.storage.images import (
    image_files,
    register_image_files,
    registered_images_under,
)


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


def write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))
    return path


def test_image_files_finds_jpeg_and_png_at_any_depth(tmp_path: Path) -> None:
    first = write_image(tmp_path / "b" / "IMG_2.JPG")
    second = write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("not an image")

    assert image_files(tmp_path) == [second, first]


def test_registers_image_files_with_their_source_and_run(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    root = tmp_path / "images"
    jpeg = write_image(root / "dive" / "a.jpg")
    png = write_image(root / "b.png")
    (root / "notes.txt").write_text("not an image")

    added = register_image_files(database, root, "overlap-register")

    assert added == 2
    assert database.execute(
        "SELECT images.path, images.source, runs.module FROM images "
        "JOIN runs ON runs.id = images.run_id ORDER BY images.path"
    ).fetchall() == [
        (str(png.resolve()), "png", "overlap-register"),
        (str(jpeg.resolve()), "jpeg", "overlap-register"),
    ]


def test_registering_again_adds_only_the_new_files(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    write_image(tmp_path / "a.jpg")
    register_image_files(database, tmp_path, "overlap-register")
    write_image(tmp_path / "b.jpg")

    added = register_image_files(database, tmp_path, "cleaning-register")

    assert added == 1
    assert database.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2


def test_registered_images_are_only_those_below_the_root(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    inside = write_image(tmp_path / "images" / "a.jpg")
    write_image(tmp_path / "elsewhere" / "b.jpg")
    register_image_files(database, tmp_path, "test-register")

    found = registered_images_under(database, tmp_path / "images")

    assert [image.path for image in found] == [inside.resolve()]
