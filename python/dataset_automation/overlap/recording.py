"""Write sequential overlaps to the shared `pairs` table."""

import sqlite3
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from dataset_automation.overlap.pairwise import UnknownReason
from dataset_automation.overlap.sequential import SequentialOverlap
from dataset_automation.storage.database import RunId


@dataclass(frozen=True)
class RecordingReport:
    known: int = 0
    # `pairs` has no column for the reason: this count is the only place it survives.
    unknown: Counter[UnknownReason] = field(default_factory=Counter)


def record_sequential_overlaps(
    connection: sqlite3.Connection,
    run_id: RunId,
    image_ids: Sequence[int],
    overlaps: Iterable[SequentialOverlap],
) -> RecordingReport:
    """`image_ids[i]` is the `images.id` of the image at position i in capture order.

    Unknown overlaps are stored as NULL with their inlier count: "compared, could not tell" is
    different from "never compared".
    """
    known = 0
    unknown: Counter[UnknownReason] = Counter()
    for overlap in overlaps:
        connection.execute(
            "INSERT INTO pairs (image_a_id, image_b_id, method, inlier_count, "
            "overlap_a_in_b, overlap_b_in_a, iou, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            _pair_row(overlap, image_ids, run_id),
        )
        # One commit per pair: an interrupted run keeps every pair already compared, and
        # matching (~0.1 s per pair) dwarfs the cost of the commit.
        connection.commit()
        reason = overlap.estimate.unknown_reason
        if reason is None:
            known += 1
        else:
            unknown[reason] += 1
    return RecordingReport(known=known, unknown=unknown)


def _pair_row(
    overlap: SequentialOverlap, image_ids: Sequence[int], run_id: RunId
) -> tuple[int, int, str, int, float | None, float | None, float | None, RunId]:
    estimate = overlap.estimate
    first_id, second_id = image_ids[overlap.index_a], image_ids[overlap.index_b]
    first_in_second, second_in_first = estimate.a_in_b, estimate.b_in_a
    # The table stores one row per unordered pair with image_a_id < image_b_id, but nothing
    # makes database ids follow capture order: swap the directional values with the ids.
    if first_id > second_id:
        first_id, second_id = second_id, first_id
        first_in_second, second_in_first = second_in_first, first_in_second
    return (
        first_id,
        second_id,
        estimate.method,
        estimate.inlier_count,
        first_in_second,
        second_in_first,
        estimate.iou,
        run_id,
    )
