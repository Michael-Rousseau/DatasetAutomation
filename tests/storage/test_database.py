import json
import sqlite3
from pathlib import Path

import pytest

from dataset_automation.storage.database import (
    SchemaVersionError,
    open_database,
    start_run,
)
from dataset_automation.storage.schema import SCHEMA_VERSION

SHARED_TABLES = {
    "runs",
    "images",
    "features",
    "pairs",
    "tracks",
    "track_observations",
    "masks",
    "targets",
}


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


def insert_image(database: sqlite3.Connection, path: str, run_id: int) -> int:
    cursor = database.execute(
        "INSERT INTO images (path, source, run_id) VALUES (?, 'jpeg', ?)",
        (path, run_id),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def test_creates_all_shared_tables_on_first_open(database: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert tables == SHARED_TABLES
    assert database.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopening_keeps_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    start_run(open_database(path), "ingestion", {}, code_version="test")

    assert open_database(path).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_refuses_database_from_another_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    open_database(path).execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(SchemaVersionError):
        open_database(path)


def test_run_stores_module_version_and_parameters(database: sqlite3.Connection) -> None:
    run_id = start_run(
        database, "overlap", {"k": 5, "method": "homography"}, code_version="abc123"
    )

    module, code_version, parameters = database.execute(
        "SELECT module, code_version, parameters FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert (module, code_version) == ("overlap", "abc123")
    assert json.loads(parameters) == {"k": 5, "method": "homography"}


def test_run_records_a_code_version_by_default(database: sqlite3.Connection) -> None:
    run_id = start_run(database, "overlap", {})

    (code_version,) = database.execute(
        "SELECT code_version FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert code_version


def test_rejects_rows_pointing_to_unknown_run(database: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert_image(database, "a.jpg", run_id=42)


def test_stores_each_pair_once_in_ascending_image_order(
    database: sqlite3.Connection,
) -> None:
    run_id = start_run(database, "overlap", {}, code_version="test")
    first, second = (
        insert_image(database, "a.jpg", run_id),
        insert_image(database, "b.jpg", run_id),
    )

    with pytest.raises(sqlite3.IntegrityError):
        database.execute(
            "INSERT INTO pairs (image_a_id, image_b_id, method, run_id) VALUES (?, ?, 'test', ?)",
            (second, first, run_id),
        )


def test_accepts_pair_with_unknown_overlap(database: sqlite3.Connection) -> None:
    run_id = start_run(database, "overlap", {}, code_version="test")
    first, second = (
        insert_image(database, "a.jpg", run_id),
        insert_image(database, "b.jpg", run_id),
    )

    database.execute(
        "INSERT INTO pairs (image_a_id, image_b_id, method, inlier_count, run_id) VALUES (?, ?, 'sift', 3, ?)",
        (first, second, run_id),
    )


def test_rejects_overlap_above_one(database: sqlite3.Connection) -> None:
    run_id = start_run(database, "overlap", {}, code_version="test")
    first, second = (
        insert_image(database, "a.jpg", run_id),
        insert_image(database, "b.jpg", run_id),
    )

    with pytest.raises(sqlite3.IntegrityError):
        database.execute(
            "INSERT INTO pairs (image_a_id, image_b_id, method, overlap_a_in_b, run_id) VALUES (?, ?, 'test', 1.2, ?)",
            (first, second, run_id),
        )


def test_rejects_track_seeing_two_points_in_the_same_image(
    database: sqlite3.Connection,
) -> None:
    run_id = start_run(database, "tracking", {}, code_version="test")
    image = insert_image(database, "a.jpg", run_id)
    track = database.execute(
        "INSERT INTO tracks (detector, length, run_id) VALUES ('sift', 2, ?)", (run_id,)
    ).lastrowid
    database.execute(
        "INSERT INTO track_observations (track_id, image_id, keypoint_index) VALUES (?, ?, 0)",
        (track, image),
    )

    with pytest.raises(sqlite3.IntegrityError):
        database.execute(
            "INSERT INTO track_observations (track_id, image_id, keypoint_index) VALUES (?, ?, 7)",
            (track, image),
        )
