"""Write sequential overlaps to the shared `pairs` table, and displacements to a CSV file."""

import csv
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TextIO

from dataset_automation.overlap.pairwise import UnknownReason
from dataset_automation.overlap.sequential import MeasuredPair
from dataset_automation.storage.database import RunId

DISPLACEMENT_COLUMNS = (
    "image_a_id",
    "image_b_id",
    "triangulation_angle_deg",
    "rotation_deg",
    "rotation_source",
    "inlier_count",
    "unknown_reason",
)


@dataclass(frozen=True)
class RecordingReport:
    known: int = 0
    # `pairs` has no column for the reason: this count is the only place it survives.
    unknown: Counter[UnknownReason] = field(default_factory=Counter)


def record_sequential_overlaps(
    connection: sqlite3.Connection,
    run_id: RunId,
    image_ids: Sequence[int],
    overlaps: Iterable[MeasuredPair],
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
    overlap: MeasuredPair, image_ids: Sequence[int], run_id: RunId
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


def write_displacements(
    file: TextIO, image_ids: Sequence[int], overlaps: Iterable[MeasuredPair]
) -> Iterator[MeasuredPair]:
    """Pass `overlaps` through, writing the displacement of each pair to `file` as CSV.

    The `pairs` table has no column for the displacement, and adding one changes the shared
    schema: it lives in a file next to the run until the measure has settled. Rows keep the
    capture order (image A is the earlier one), unlike `pairs`, which orders by database id.
    Each row is flushed, so an interrupted run keeps the pairs already measured.
    """
    writer = csv.writer(file)
    writer.writerow(DISPLACEMENT_COLUMNS)
    for overlap in overlaps:
        displacement = overlap.displacement
        if displacement is not None:
            writer.writerow(
                [
                    image_ids[overlap.index_a],
                    image_ids[overlap.index_b],
                    _text(displacement.triangulation_angle_deg),
                    _text(displacement.rotation_deg),
                    _value(displacement.rotation_source),
                    displacement.inlier_count,
                    _value(displacement.unknown_reason),
                ]
            )
            file.flush()
        yield overlap


def read_view_angles(
    file: TextIO, image_ids: Sequence[int]
) -> dict[tuple[int, int], float]:
    """Angle between the two views of each pair of a file written by `write_displacements`,
    keyed by (lower position, higher position) in `image_ids`. Unknown angles and pairs with
    an image outside `image_ids` are left out."""
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    angles: dict[tuple[int, int], float] = {}
    for row in csv.DictReader(file):
        first = position.get(int(row["image_a_id"]))
        second = position.get(int(row["image_b_id"]))
        if first is None or second is None or not row["triangulation_angle_deg"]:
            continue
        angles[min(first, second), max(first, second)] = float(
            row["triangulation_angle_deg"]
        )
    return angles


def _text(number: float | None) -> str:
    return "" if number is None else f"{number:.4f}"


def _value(member: Enum | None) -> str:
    return "" if member is None else str(member.value)
