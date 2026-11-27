import io
import sqlite3
from pathlib import Path

import pytest

from dataset_automation.storage.database import RunId, open_database, start_run
from dataset_automation.view.inputs import (
    pair_indices,
    read_confirmed_links,
    read_displacement_angles,
    read_selection,
    statuses_for,
    stem_pair,
)
from dataset_automation.view.scene import ImageStatus


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "shared.sqlite")
    run_id = start_run(connection, "test", {})
    connection.executemany(
        "INSERT INTO images (id, path, source, run_id) VALUES (?, ?, 'jpeg', ?)",
        [(id_, f"/data/images/IMG_{id_}.JPG", run_id) for id_ in (1, 2, 3)],
    )
    connection.commit()
    return connection


def insert_pair(
    connection: sqlite3.Connection,
    image_a_id: int,
    image_b_id: int,
    overlaps: tuple[float, float] | None,
    run_id: RunId,
) -> None:
    a_in_b, b_in_a = overlaps or (None, None)
    connection.execute(
        "INSERT INTO pairs (image_a_id, image_b_id, method, overlap_a_in_b, overlap_b_in_a, "
        "run_id) VALUES (?, ?, 'homography', ?, ?, ?)",
        (image_a_id, image_b_id, a_in_b, b_in_a, run_id),
    )


def test_selection_is_keyed_by_image_stem() -> None:
    selection = read_selection(
        io.StringIO(
            "image,status,compared_with,overlap\n"
            "data/images/IMG_1.JPG,kept,,\n"
            "data/images/IMG_2.JPG,removed,data/images/IMG_1.JPG,0.9100\n"
        )
    )

    assert selection == {"IMG_1": ImageStatus.KEPT, "IMG_2": ImageStatus.REMOVED}


def test_image_missing_from_the_selection_is_unknown() -> None:
    statuses = statuses_for(["IMG_1", "IMG_9"], {"IMG_1": ImageStatus.KEPT})

    assert statuses == [ImageStatus.KEPT, ImageStatus.UNKNOWN]


def test_only_confirmed_links_of_the_run_are_read(
    database: sqlite3.Connection,
) -> None:
    loops = start_run(database, "overlap-loops", {})
    other = start_run(database, "overlap-sequential", {})
    insert_pair(database, 1, 3, (0.6, 0.7), loops)
    insert_pair(database, 1, 2, None, loops)  # proposed, not confirmed
    insert_pair(database, 2, 3, (0.9, 0.9), other)

    assert read_confirmed_links(database, loops) == [("IMG_1", "IMG_3")]


def test_displacement_angles_are_keyed_by_stem_pair_in_any_order() -> None:
    angles = read_displacement_angles(
        io.StringIO(
            "image_a_id,image_b_id,triangulation_angle_deg,rotation_deg,"
            "rotation_source,inlier_count,unknown_reason\n"
            "3,1,12.5000,3.0,essential,80,\n"
            "1,2,,,,4,too_few_inliers\n"
            "1,7,8.0,2.0,essential,50,\n"
        ),
        {1: "IMG_1", 2: "IMG_2", 3: "IMG_3"},
    )

    assert angles == {("IMG_1", "IMG_3"): 12.5}


def test_pairs_without_a_pose_are_dropped() -> None:
    indices, placed = pair_indices([("A", "C"), ("A", "Z")], labels=["A", "B", "C"])

    assert indices.tolist() == [[0, 2]]
    assert placed == [("A", "C")]


def test_no_pair_gives_an_empty_index_array() -> None:
    indices, _ = pair_indices([], labels=["A"])

    assert indices.shape == (0, 2)


def test_stem_pair_is_ordered() -> None:
    assert stem_pair("b", "a") == ("a", "b")
