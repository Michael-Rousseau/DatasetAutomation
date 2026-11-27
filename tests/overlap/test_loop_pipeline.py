import csv
import sqlite3
from pathlib import Path

import cv2
import pytest

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.features.vlad import load_codebook
from dataset_automation.overlap.loop_pipeline import (
    MODULE,
    LoopReport,
    record_loop_closures,
)
from dataset_automation.overlap.pipeline import SequenceFile
from dataset_automation.storage.database import RunId, open_database, start_run
from tests.overlap.scenes import HEIGHT, WIDTH, scene, sliding_views

EXTRACTOR = SiftExtractor(scale=1.0)
PINHOLE = Intrinsics(WIDTH, HEIGHT, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LINE_LENGTH = 5
MIN_GAP = 2


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


@pytest.fixture
def run_id(database: sqlite3.Connection) -> RunId:
    return start_run(database, MODULE, {}, code_version="test")


def register(database: sqlite3.Connection, run_id: RunId, path: Path) -> SequenceFile:
    cursor = database.execute(
        "INSERT INTO images (path, source, run_id) VALUES (?, 'png', ?)",
        (str(path), run_id),
    )
    assert cursor.lastrowid is not None
    return SequenceFile(cursor.lastrowid, path)


def out_and_back(
    database: sqlite3.Connection, run_id: RunId, folder: Path
) -> list[SequenceFile]:
    """A camera going along a line, then coming back over the same places: image i and
    image 2 × LINE_LENGTH − 1 − i see the same place."""
    line = sliding_views(LINE_LENGTH, scene())
    files = []
    for position, view in enumerate(line + line[::-1]):
        path = folder / f"{position:02d}.png"
        cv2.imwrite(str(path), view)
        files.append(register(database, run_id, path))
    return files


def find_loops(
    database: sqlite3.Connection,
    run_id: RunId,
    files: list[SequenceFile],
    tmp_path: Path,
    intrinsics: Intrinsics | None = None,
    output_directory: Path | None = None,
) -> LoopReport:
    return record_loop_closures(
        database,
        run_id,
        files,
        tmp_path / "cache",
        EXTRACTOR,
        RatioTestMatcher(),
        candidates_per_image=2,
        min_gap=MIN_GAP,
        intrinsics=intrinsics,
        output_directory=output_directory,
    )


def test_links_each_image_to_the_one_taken_on_the_way_back(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    files = out_and_back(database, run_id, tmp_path)

    report = find_loops(database, run_id, files, tmp_path)

    known = {
        (image_a_id, image_b_id): overlap
        for image_a_id, image_b_id, overlap in database.execute(
            "SELECT image_a_id, image_b_id, overlap_a_in_b FROM pairs "
            "WHERE run_id = ? AND overlap_a_in_b IS NOT NULL",
            (run_id,),
        )
    }
    twins = [
        (position, 2 * LINE_LENGTH - 1 - position) for position in range(LINE_LENGTH)
    ]
    # The two images at the turn are neighbours in time, left to the sequential mode.
    far_twins = [(first, second) for first, second in twins if second - first > MIN_GAP]
    assert len(far_twins) == LINE_LENGTH - 1
    for first, second in far_twins:
        pair = (files[first].image_id, files[second].image_id)
        assert known[pair] == pytest.approx(1.0, abs=0.01)
    assert report.image_count == 2 * LINE_LENGTH
    assert report.recording.known == len(known)
    assert report.recording.known + sum(report.recording.unknown.values()) == (
        report.candidate_count
    )
    assert files[0].path in report.linked_images


def test_never_proposes_neighbours_in_time(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    files = out_and_back(database, run_id, tmp_path)
    position = {file.image_id: index for index, file in enumerate(files)}

    find_loops(database, run_id, files, tmp_path)

    rows = database.execute("SELECT image_a_id, image_b_id FROM pairs").fetchall()
    assert rows
    assert all(abs(position[a] - position[b]) > MIN_GAP for a, b in rows)


def test_skips_an_unreadable_file(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    files = out_and_back(database, run_id, tmp_path)
    corrupted_path = tmp_path / "corrupted.png"
    corrupted_path.write_bytes(b"not an image")
    files.insert(3, register(database, run_id, corrupted_path))

    report = find_loops(database, run_id, files, tmp_path)

    assert report.unreadable == [corrupted_path]
    assert report.image_count == 2 * LINE_LENGTH
    assert report.recording.known > 0


def test_keeps_the_codebook_and_the_displacements_next_to_the_run(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    files = out_and_back(database, run_id, tmp_path)
    output = tmp_path / "run"

    report = find_loops(database, run_id, files, tmp_path, PINHOLE, output)

    assert load_codebook(output / "codebook.npy").words.shape == (64, 128)
    with (output / "displacements.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == report.candidate_count


def test_two_files_with_the_same_name_are_refused(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    files = out_and_back(database, run_id, tmp_path)
    (tmp_path / "other").mkdir()
    duplicate = tmp_path / "other" / "00.png"
    duplicate.write_bytes(files[0].path.read_bytes())
    files.append(register(database, run_id, duplicate))

    with pytest.raises(ValueError, match="same file name"):
        find_loops(database, run_id, files, tmp_path)


def test_no_image_gives_an_empty_report(
    database: sqlite3.Connection, run_id: RunId, tmp_path: Path
) -> None:
    report = find_loops(database, run_id, [], tmp_path)

    assert report.candidate_count == 0
    assert report.linked_images == []
