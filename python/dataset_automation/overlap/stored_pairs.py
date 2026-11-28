"""Read the overlaps of a run back from the shared `pairs` table."""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dataset_automation.overlap.deduplication import mutual_overlap
from dataset_automation.overlap.pipeline import MODULE
from dataset_automation.storage.database import RunId


@dataclass(frozen=True)
class StoredSequence:
    """Images of a run in capture order, and their known overlaps keyed by positions.

    `image_ids[i]` is the `images.id` of the image at position i. Keys are (earlier position,
    later position). `earlier_in_later` is the share of the
    earlier image seen by the later one; `mutual_overlaps` is the smaller of both directions.
    """

    paths: list[Path]
    image_ids: list[int]
    earlier_in_later: dict[tuple[int, int], float]
    mutual_overlaps: dict[tuple[int, int], float]
    unknown_pair_count: int


def latest_overlap_run(
    connection: sqlite3.Connection, module: str = MODULE
) -> RunId | None:
    """Latest run of `module`: the sequential overlaps by default."""
    row = connection.execute(
        "SELECT MAX(id) FROM runs WHERE module = ?", (module,)
    ).fetchone()
    return None if row[0] is None else RunId(row[0])


def load_stored_sequence(
    connection: sqlite3.Connection, run_id: RunId
) -> StoredSequence:
    """Capture order is the order of the file paths, as in the sequential run that wrote the
    pairs, until ingestion fills `images.timestamp_s`."""
    images = connection.execute(
        "SELECT id, path FROM images WHERE id IN ("
        "SELECT image_a_id FROM pairs WHERE run_id = :run "
        "UNION SELECT image_b_id FROM pairs WHERE run_id = :run) ORDER BY path",
        {"run": run_id},
    ).fetchall()
    position = {image_id: index for index, (image_id, _) in enumerate(images)}

    earlier_in_later: dict[tuple[int, int], float] = {}
    mutual_overlaps: dict[tuple[int, int], float] = {}
    unknown_pair_count = 0
    pairs = connection.execute(
        "SELECT image_a_id, image_b_id, overlap_a_in_b, overlap_b_in_a FROM pairs "
        "WHERE run_id = ?",
        (run_id,),
    )
    for image_a_id, image_b_id, a_in_b, b_in_a in pairs:
        if a_in_b is None or b_in_a is None:
            unknown_pair_count += 1
            continue
        position_a, position_b = position[image_a_id], position[image_b_id]
        # Rows are ordered by database id, which need not be capture order.
        if position_a > position_b:
            position_a, position_b = position_b, position_a
            a_in_b, b_in_a = b_in_a, a_in_b
        earlier_in_later[position_a, position_b] = a_in_b
        mutual_overlaps[position_a, position_b] = mutual_overlap(a_in_b, b_in_a)
    return StoredSequence(
        paths=[Path(path) for _, path in images],
        image_ids=[image_id for image_id, _ in images],
        earlier_in_later=earlier_in_later,
        mutual_overlaps=mutual_overlaps,
        unknown_pair_count=unknown_pair_count,
    )


def load_mutual_overlaps(
    connection: sqlite3.Connection, run_id: RunId, image_ids: Sequence[int]
) -> dict[tuple[int, int], float]:
    """Known mutual overlaps of a run, keyed by (lower position, higher position) in
    `image_ids`. Pairs with an image outside `image_ids` are left out: the run may cover
    other images than the sequence they are read for."""
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    pairs = connection.execute(
        "SELECT image_a_id, image_b_id, overlap_a_in_b, overlap_b_in_a FROM pairs "
        "WHERE run_id = ? AND overlap_a_in_b IS NOT NULL AND overlap_b_in_a IS NOT NULL",
        (run_id,),
    )
    return {
        _ordered(position[image_a_id], position[image_b_id]): mutual_overlap(
            a_in_b, b_in_a
        )
        for image_a_id, image_b_id, a_in_b, b_in_a in pairs
        if image_a_id in position and image_b_id in position
    }


def weakly_matched_images(
    connection: sqlite3.Connection,
    run_id: RunId,
    image_ids: Sequence[int],
    min_inliers: int,
) -> set[int]:
    """Positions of the images tied to the image before or after them by fewer than
    `min_inliers` matches that agree.

    Where the water is thick or the camera high, neighbours still overlap but few points
    say so: on Mermaid these are the images a reconstruction places worst.
    """
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    pairs = connection.execute(
        "SELECT image_a_id, image_b_id FROM pairs WHERE run_id = ? AND inlier_count < ?",
        (run_id, min_inliers),
    )
    weak: set[int] = set()
    for image_a_id, image_b_id in pairs:
        first, second = position.get(image_a_id), position.get(image_b_id)
        if first is not None and second is not None and abs(first - second) == 1:
            weak.update((first, second))
    return weak


def _ordered(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)
