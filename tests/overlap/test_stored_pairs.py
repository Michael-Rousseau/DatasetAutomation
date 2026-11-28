import sqlite3
from pathlib import Path

import pytest

from dataset_automation.overlap.stored_pairs import (
    latest_overlap_run,
    load_mutual_overlaps,
    load_stored_sequence,
    weakly_matched_images,
)
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


def test_keys_overlaps_by_position_in_path_order(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    # Registered out of capture order: database ids do not follow the paths.
    third = register(database, run_id, "dive/0003.jpg")
    first = register(database, run_id, "dive/0001.jpg")
    second = register(database, run_id, "dive/0002.jpg")
    store_pair(database, run_id, (first, second), (0.9, 0.8))
    store_pair(database, run_id, (second, third), (0.7, 0.95))

    sequence = load_stored_sequence(database, run_id)

    assert sequence.paths == [
        Path("dive/0001.jpg"),
        Path("dive/0002.jpg"),
        Path("dive/0003.jpg"),
    ]
    assert sequence.image_ids == [first, second, third]
    assert sequence.mutual_overlaps == {(0, 1): 0.8, (1, 2): 0.7}
    assert sequence.unknown_pair_count == 0


def test_directional_overlap_follows_capture_order_not_database_ids(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    # The later image was registered first, so it is image_a in the stored row.
    later = register(database, run_id, "0002.jpg")
    earlier = register(database, run_id, "0001.jpg")
    database.execute(
        "INSERT INTO pairs (image_a_id, image_b_id, method, inlier_count, "
        "overlap_a_in_b, overlap_b_in_a, run_id) VALUES (?, ?, 'homography', 50, 0.3, 0.9, ?)",
        (later, earlier, run_id),
    )

    sequence = load_stored_sequence(database, run_id)

    assert sequence.earlier_in_later == {(0, 1): 0.9}


def test_leaves_unknown_overlaps_out_but_keeps_their_images(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    first = register(database, run_id, "0001.jpg")
    second = register(database, run_id, "0002.jpg")
    store_pair(database, run_id, (first, second), None)

    sequence = load_stored_sequence(database, run_id)

    assert len(sequence.paths) == 2
    assert sequence.mutual_overlaps == {}
    assert sequence.earlier_in_later == {}
    assert sequence.unknown_pair_count == 1


def test_reads_only_the_requested_run(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    other_run = start_run(database, "overlap", {}, code_version="test")
    first = register(database, run_id, "0001.jpg")
    second = register(database, run_id, "0002.jpg")
    store_pair(database, run_id, (first, second), (0.9, 0.9))
    store_pair(database, other_run, (first, second), (0.2, 0.2))

    assert load_stored_sequence(database, run_id).mutual_overlaps == {(0, 1): 0.9}


def test_latest_overlap_run_ignores_other_modules(database: sqlite3.Connection) -> None:
    assert latest_overlap_run(database) is None

    overlap_run = start_run(database, "overlap", {}, code_version="test")
    start_run(database, "deduplication", {}, code_version="test")

    assert latest_overlap_run(database) == overlap_run


def test_overlaps_of_another_run_are_keyed_by_the_positions_asked_for(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    first = register(database, run_id, "dive/0001.jpg")
    second = register(database, run_id, "dive/0002.jpg")
    third = register(database, run_id, "dive/0003.jpg")
    elsewhere = register(database, run_id, "other_dive/0001.jpg")
    loops = start_run(database, "overlap-loops", {}, code_version="test")
    store_pair(database, loops, (first, third), (0.6, 0.7))
    store_pair(database, loops, (first, second), None)
    store_pair(database, loops, (second, elsewhere), (0.9, 0.9))

    # Capture order given by the caller, here the reverse of the database ids.
    overlaps = load_mutual_overlaps(database, loops, [third, second, first])

    assert overlaps == {(0, 2): 0.6}


def test_latest_run_of_another_module(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    assert latest_overlap_run(database, "overlap-loops") is None

    loops = start_run(database, "overlap-loops", {}, code_version="test")
    start_run(database, "overlap", {}, code_version="test")

    assert latest_overlap_run(database, "overlap-loops") == loops


def test_images_tied_to_a_neighbour_by_few_matches(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    ids = [register(database, run_id, f"dive/000{index}.jpg") for index in range(5)]
    for first, second, inliers in ((0, 1, 900), (1, 2, 80), (2, 3, 700), (3, 4, 600)):
        database.execute(
            "INSERT INTO pairs (image_a_id, image_b_id, method, inlier_count, "
            "overlap_a_in_b, overlap_b_in_a, run_id) VALUES (?, ?, 'homography', ?, 0.8, "
            "0.8, ?)",
            (ids[first], ids[second], inliers, run_id),
        )
    # Few matches between images two apart say nothing about the chain.
    database.execute(
        "INSERT INTO pairs (image_a_id, image_b_id, method, inlier_count, run_id) "
        "VALUES (?, ?, 'homography', 10, ?)",
        (ids[2], ids[4], run_id),
    )

    assert weakly_matched_images(database, run_id, ids, min_inliers=200) == {1, 2}
