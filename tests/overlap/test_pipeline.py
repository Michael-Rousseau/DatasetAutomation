import sqlite3
from pathlib import Path

import cv2
import pytest

from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.pipeline import SequenceFile, record_file_sequence
from dataset_automation.storage.database import RunId, open_database, start_run
from tests.overlap.scenes import STEP_PX, WIDTH, scene, sliding_views

EXTRACTOR = SiftExtractor(scale=1.0)


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


@pytest.fixture
def run_id(database: sqlite3.Connection) -> RunId:
    return start_run(database, "overlap", {}, code_version="test")


def register(database: sqlite3.Connection, run_id: RunId, path: Path) -> SequenceFile:
    cursor = database.execute(
        "INSERT INTO images (path, source, run_id) VALUES (?, 'png', ?)",
        (str(path), run_id),
    )
    assert cursor.lastrowid is not None
    return SequenceFile(cursor.lastrowid, path)


def sliding_files(
    database: sqlite3.Connection, run_id: RunId, folder: Path, count: int
) -> list[SequenceFile]:
    files = []
    for position, view in enumerate(sliding_views(count, scene())):
        path = folder / f"{position}.png"
        cv2.imwrite(str(path), view)
        files.append(register(database, run_id, path))
    return files


def test_records_the_overlap_of_consecutive_image_files(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    files = sliding_files(database, run_id, tmp_path, 3)

    report = record_file_sequence(
        database, run_id, files, EXTRACTOR, RatioTestMatcher(), 1
    )

    rows = database.execute(
        "SELECT image_a_id, image_b_id, overlap_a_in_b FROM pairs ORDER BY id"
    ).fetchall()
    assert [row[:2] for row in rows] == [
        (files[0].image_id, files[1].image_id),
        (files[1].image_id, files[2].image_id),
    ]
    assert [row[2] for row in rows] == pytest.approx(
        [1 - STEP_PX / WIDTH] * 2, abs=0.01
    )
    assert report.recording.known == 2
    assert report.unreadable == []


def test_skips_an_unreadable_file_and_pairs_its_neighbours(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    first, second = sliding_files(database, run_id, tmp_path, 2)
    corrupted_path = tmp_path / "corrupted.png"
    corrupted_path.write_bytes(b"not an image")
    corrupted = register(database, run_id, corrupted_path)

    report = record_file_sequence(
        database,
        run_id,
        [first, corrupted, second],
        EXTRACTOR,
        RatioTestMatcher(),
        1,
    )

    assert database.execute("SELECT image_a_id, image_b_id FROM pairs").fetchall() == [
        (first.image_id, second.image_id)
    ]
    assert report.unreadable == [corrupted_path]
