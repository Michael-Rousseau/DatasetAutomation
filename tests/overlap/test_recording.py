import sqlite3
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.overlap.pairwise import PairOverlapEstimate, UnknownReason
from dataset_automation.overlap.recording import record_sequential_overlaps
from dataset_automation.overlap.sequential import (
    SequentialOverlap,
    estimate_sequential_overlaps,
)
from dataset_automation.storage.database import RunId, open_database, start_run
from tests.overlap.scenes import STEP_PX, WIDTH, scene
from tests.overlap.test_sequential import sliding_camera

PAIR_COLUMNS = (
    "SELECT image_a_id, image_b_id, method, inlier_count, overlap_a_in_b, "
    "overlap_b_in_a, iou, run_id FROM pairs ORDER BY id"
)


@pytest.fixture
def database(tmp_path: Path) -> sqlite3.Connection:
    return open_database(tmp_path / "shared.sqlite")


@pytest.fixture
def run_id(database: sqlite3.Connection) -> RunId:
    return start_run(database, "overlap", {}, code_version="test")


def register_images(
    database: sqlite3.Connection, run_id: RunId, count: int
) -> list[int]:
    image_ids = []
    for position in range(count):
        cursor = database.execute(
            "INSERT INTO images (path, source, run_id) VALUES (?, 'jpeg', ?)",
            (f"{position}.jpg", run_id),
        )
        assert cursor.lastrowid is not None
        image_ids.append(cursor.lastrowid)
    return image_ids


def known(a_in_b: float, b_in_a: float, iou: float) -> PairOverlapEstimate:
    return PairOverlapEstimate("homography", 200, 120, a_in_b, b_in_a, iou, None)


def unknown(reason: UnknownReason, inlier_count: int) -> PairOverlapEstimate:
    return PairOverlapEstimate("homography", 40, inlier_count, None, None, None, reason)


def test_stores_a_known_overlap_with_its_run(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    first, second = register_images(database, run_id, 2)

    record_sequential_overlaps(
        database,
        run_id,
        [first, second],
        [SequentialOverlap(0, 1, known(0.9, 0.6, 0.5))],
    )

    assert database.execute(PAIR_COLUMNS).fetchall() == [
        (first, second, "homography", 120, 0.9, 0.6, 0.5, run_id)
    ]


def test_stores_an_unknown_overlap_as_null_with_its_inlier_count(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    image_ids = register_images(database, run_id, 2)

    record_sequential_overlaps(
        database,
        run_id,
        image_ids,
        [SequentialOverlap(0, 1, unknown(UnknownReason.TOO_FEW_INLIERS, 6))],
    )

    (row,) = database.execute(PAIR_COLUMNS).fetchall()
    assert row[3:7] == (6, None, None, None)


def test_swaps_directional_overlaps_when_ids_do_not_follow_capture_order(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    lower_id, higher_id = register_images(database, run_id, 2)

    record_sequential_overlaps(
        database,
        run_id,
        [higher_id, lower_id],
        [SequentialOverlap(0, 1, known(0.9, 0.6, 0.5))],
    )

    (row,) = database.execute(PAIR_COLUMNS).fetchall()
    assert row[:2] == (lower_id, higher_id)
    assert row[4:6] == (0.6, 0.9)


def test_reports_unknown_overlaps_by_reason(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    image_ids = register_images(database, run_id, 4)

    report = record_sequential_overlaps(
        database,
        run_id,
        image_ids,
        [
            SequentialOverlap(0, 1, known(0.9, 0.9, 0.8)),
            SequentialOverlap(1, 2, unknown(UnknownReason.TOO_FEW_INLIERS, 6)),
            SequentialOverlap(2, 3, unknown(UnknownReason.TOO_FEW_INLIERS, 3)),
            SequentialOverlap(0, 2, unknown(UnknownReason.IMPLAUSIBLE_HOMOGRAPHY, 30)),
        ],
    )

    assert report.known == 1
    assert report.unknown == Counter(
        {
            UnknownReason.TOO_FEW_INLIERS: 2,
            UnknownReason.IMPLAUSIBLE_HOMOGRAPHY: 1,
        }
    )


def test_keeps_pairs_recorded_before_an_interruption(
    tmp_path: Path, run_id: RunId, database: sqlite3.Connection
) -> None:
    image_ids = register_images(database, run_id, 3)
    database.commit()

    def interrupted() -> Iterator[SequentialOverlap]:
        yield SequentialOverlap(0, 1, known(0.9, 0.9, 0.8))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        record_sequential_overlaps(database, run_id, image_ids, interrupted())
    database.close()

    reopened = open_database(tmp_path / "shared.sqlite")
    assert reopened.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 1


def test_records_overlaps_estimated_from_real_features(
    database: sqlite3.Connection, run_id: RunId
) -> None:
    image_ids = register_images(database, run_id, 2)
    overlaps = estimate_sequential_overlaps(
        sliding_camera(2, scene()), 1, RatioTestMatcher()
    )

    report = record_sequential_overlaps(database, run_id, image_ids, overlaps)

    (row,) = database.execute(PAIR_COLUMNS).fetchall()
    assert report.known == 1
    assert row[4] == pytest.approx(1 - STEP_PX / WIDTH, abs=0.01)
