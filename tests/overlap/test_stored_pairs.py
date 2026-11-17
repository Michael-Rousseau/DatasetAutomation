import sqlite3
from pathlib import Path

import pytest

from dataset_automation.overlap.stored_pairs import load_mutual_overlaps
from dataset_automation.storage.database import RunId, open_database, start_run


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


@pytest.fixture
def run_id(database: sqlite3.Connection) -> RunId:
    return start_run(database, "overlap", {}, code_version="test")


def register(database: sqlite3.Connection, run_id: RunId, path: str) -> int:
    cursor = database.execute(
        "INSERT INTO images (path, source, run_id) VALUES (?, 'jpeg', ?)",
        (path, run_id),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def store_pair(
    database: sqlite3.Connection,
    run_id: RunId,
    image_ids: tuple[int, int],
    overlaps: tuple[float, float] | None,
) -> None:
    a_in_b, b_in_a = overlaps if overlaps else (None, None)
    database.execute(
        "INSERT INTO pairs (image_a_id, image_b_id, method, inlier_count, "
        "overlap_a_in_b, overlap_b_in_a, run_id) VALUES (?, ?, 'homography', 50, ?, ?, ?)",
        (min(image_ids), max(image_ids), a_in_b, b_in_a, run_id),
    )


def test_keys_mutual_overlaps_by_position_in_path_order(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    # Registered out of capture order: database ids do not follow the paths.
    third = register(database, run_id, "dive/0003.jpg")
    first = register(database, run_id, "dive/0001.jpg")
    second = register(database, run_id, "dive/0002.jpg")
    store_pair(database, run_id, (first, second), (0.9, 0.8))
    store_pair(database, run_id, (second, third), (0.7, 0.95))

    sequence = load_mutual_overlaps(database, run_id)

    assert sequence.paths == [
        Path("dive/0001.jpg"),
        Path("dive/0002.jpg"),
        Path("dive/0003.jpg"),
    ]
    assert sequence.mutual_overlaps == {(0, 1): 0.8, (1, 2): 0.7}


def test_leaves_unknown_overlaps_out_but_keeps_their_images(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    first = register(database, run_id, "0001.jpg")
    second = register(database, run_id, "0002.jpg")
    store_pair(database, run_id, (first, second), None)

    sequence = load_mutual_overlaps(database, run_id)

    assert len(sequence.paths) == 2
    assert sequence.mutual_overlaps == {}


def test_reads_only_the_requested_run(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    other_run = start_run(database, "overlap", {}, code_version="test")
    first = register(database, run_id, "0001.jpg")
    second = register(database, run_id, "0002.jpg")
    store_pair(database, run_id, (first, second), (0.9, 0.9))
    store_pair(database, other_run, (first, second), (0.2, 0.2))

    assert load_mutual_overlaps(database, run_id).mutual_overlaps == {(0, 1): 0.9}
