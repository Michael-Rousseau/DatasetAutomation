"""Read the overlaps of a run back from the shared `pairs` table."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from dataset_automation.overlap.deduplication import mutual_overlap
from dataset_automation.storage.database import RunId


@dataclass(frozen=True)
class StoredSequence:
    """`paths[i]` is the image at position i; `mutual_overlaps` is keyed by positions."""

    paths: list[Path]
    mutual_overlaps: dict[tuple[int, int], float]


def load_mutual_overlaps(
    connection: sqlite3.Connection, run_id: RunId
) -> StoredSequence:
    """Images compared by `run_id`, in capture order, with their known mutual overlaps.

    Capture order is the order of the file paths, as in the sequential run that wrote the
    pairs, until ingestion fills `images.timestamp_s`.
    """
    images = connection.execute(
        "SELECT id, path FROM images WHERE id IN ("
        "SELECT image_a_id FROM pairs WHERE run_id = :run "
        "UNION SELECT image_b_id FROM pairs WHERE run_id = :run) ORDER BY path",
        {"run": run_id},
    ).fetchall()
    position = {image_id: index for index, (image_id, _) in enumerate(images)}

    mutual_overlaps: dict[tuple[int, int], float] = {}
    known_pairs = connection.execute(
        "SELECT image_a_id, image_b_id, overlap_a_in_b, overlap_b_in_a FROM pairs "
        "WHERE run_id = ? AND overlap_a_in_b IS NOT NULL AND overlap_b_in_a IS NOT NULL",
        (run_id,),
    )
    for image_a_id, image_b_id, a_in_b, b_in_a in known_pairs:
        # Rows are ordered by database id, which need not be capture order.
        first, second = sorted((position[image_a_id], position[image_b_id]))
        mutual_overlaps[first, second] = mutual_overlap(a_in_b, b_in_a)
    return StoredSequence(
        paths=[Path(path) for _, path in images], mutual_overlaps=mutual_overlaps
    )
