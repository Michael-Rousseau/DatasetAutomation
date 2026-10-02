import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset_automation.cleaning.masks import read_unwanted
from dataset_automation.cleaning.pipeline import (
    COMBINED,
    ImageToMask,
    image_files,
    images_under,
    mask_images,
)
from dataset_automation.storage.database import open_database, start_run


class LeftColumnFin:
    """Fake segmenter: a fin on the left column, never a diver."""

    name = "fake"
    classes = ("fin", "diver")

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        fin = np.zeros(image.shape[:2], dtype=bool)
        fin[:, 0] = True
        return {"fin": fin, "diver": np.zeros_like(fin)}

    def run_parameters(self) -> dict[str, object]:
        return {"threshold": 0.3}


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


def register(database: sqlite3.Connection, paths: list[Path]) -> list[ImageToMask]:
    run_id = start_run(database, "ingestion", {}, code_version="test")
    images = []
    for path in paths:
        cursor = database.execute(
            "INSERT INTO images (path, source, run_id) VALUES (?, 'jpeg', ?)",
            (str(path), run_id),
        )
        assert cursor.lastrowid is not None
        images.append(ImageToMask(cursor.lastrowid, path))
    return images


def write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((4, 3, 3), 90, dtype=np.uint8))
    return path


def test_writes_colmap_and_class_masks_and_records_them(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    root = tmp_path / "images"
    (image,) = register(database, [write_image(root / "dive" / "a.jpg")])

    report = mask_images(database, LeftColumnFin(), [image], root, tmp_path / "masks")

    rows = dict(
        database.execute(
            "SELECT class_name, mask_path FROM masks WHERE image_id = ?",
            (image.image_id,),
        ).fetchall()
    )
    assert set(rows) == {"fin", COMBINED}
    combined = Path(rows[COMBINED])
    assert (
        combined
        == (
            tmp_path
            / "masks"
            / f"run_{report.run_id}"
            / "colmap"
            / "dive"
            / "a.jpg.png"
        ).resolve()
    )
    assert read_unwanted(combined)[:, 0].all()
    assert not read_unwanted(combined)[:, 1:].any()
    assert report.masked == 1


def test_records_method_and_parameters_in_the_run(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    root = tmp_path / "images"
    images = register(database, [write_image(root / "a.jpg")])

    report = mask_images(database, LeftColumnFin(), images, root, tmp_path / "masks")

    module, parameters = database.execute(
        "SELECT module, parameters FROM runs WHERE id = ?", (report.run_id,)
    ).fetchone()
    assert module == "cleaning"
    assert json.loads(parameters) == {
        "method": "fake",
        "classes": ["fin", "diver"],
        "threshold": 0.3,
    }


def test_unreadable_image_is_skipped_not_fatal(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    root = tmp_path / "images"
    broken = root / "broken.jpg"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a jpeg")
    images = register(database, [broken, write_image(root / "good.jpg")])

    report = mask_images(database, LeftColumnFin(), images, root, tmp_path / "masks")

    assert report.unreadable == [broken]
    assert report.masked == 1


def test_rejects_segmenter_that_forgets_a_class(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    class Forgetful(LeftColumnFin):
        def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
            return {"fin": np.zeros(image.shape[:2], dtype=bool)}

    root = tmp_path / "images"
    images = register(database, [write_image(root / "a.jpg")])

    with pytest.raises(ValueError):
        mask_images(database, Forgetful(), images, root, tmp_path / "masks")


def test_images_under_keeps_only_files_below_the_root(
    database: sqlite3.Connection, tmp_path: Path
) -> None:
    inside = write_image(tmp_path / "images" / "a.jpg")
    outside = write_image(tmp_path / "elsewhere" / "b.jpg")
    register(database, [inside, outside])

    found = images_under(database, tmp_path / "images")

    assert [image.path for image in found] == [inside]


def test_image_files_finds_jpeg_and_png_at_any_depth(tmp_path: Path) -> None:
    first = write_image(tmp_path / "b" / "IMG_2.JPG")
    second = write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("not an image")

    assert image_files(tmp_path) == [second, first]
