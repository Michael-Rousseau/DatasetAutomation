"""Read what the overlap engine wrote, keyed by image file stem so it meets the pose labels.

Poses name images by stem (Mermaid labels), while the database and the CSV files name them by
path or database id: the stem is the only key the three share.
"""

import csv
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import PurePath
from typing import TextIO

import numpy as np

from dataset_automation.storage.database import RunId
from dataset_automation.view.scene import ImageStatus

StemPair = tuple[str, str]


def read_selection(file: TextIO) -> dict[str, ImageStatus]:
    """Status of each image of an `images.csv` written by `select_images.py`."""
    return {
        PurePath(row["image"]).stem: ImageStatus(row["status"])
        for row in csv.DictReader(file)
    }


def statuses_for(
    labels: Sequence[str], selection: dict[str, ImageStatus]
) -> list[ImageStatus]:
    return [selection.get(label, ImageStatus.UNKNOWN) for label in labels]


def image_stems(connection: sqlite3.Connection) -> dict[int, str]:
    return {
        image_id: PurePath(path).stem
        for image_id, path in connection.execute("SELECT id, path FROM images")
    }


def read_confirmed_links(
    connection: sqlite3.Connection, run_id: RunId
) -> list[StemPair]:
    """Pairs of a run whose overlap is known: for `find_loops.py`, the confirmed loop closures.
    Candidates that matching could not confirm are stored with NULL overlaps and left out."""
    stems = image_stems(connection)
    rows = connection.execute(
        "SELECT image_a_id, image_b_id FROM pairs WHERE run_id = ? "
        "AND overlap_a_in_b IS NOT NULL AND overlap_b_in_a IS NOT NULL",
        (run_id,),
    )
    return [stem_pair(stems[first], stems[second]) for first, second in rows]


def read_displacement_angles(
    file: TextIO, stems: dict[int, str]
) -> dict[StemPair, float]:
    """Angle between the two views of each pair of a `displacements.csv`; unknown angles and
    images missing from `stems` are left out."""
    angles: dict[StemPair, float] = {}
    for row in csv.DictReader(file):
        first = stems.get(int(row["image_a_id"]))
        second = stems.get(int(row["image_b_id"]))
        if first is None or second is None or not row["triangulation_angle_deg"]:
            continue
        angles[stem_pair(first, second)] = float(row["triangulation_angle_deg"])
    return angles


def pair_indices(
    pairs: Iterable[StemPair], labels: Sequence[str]
) -> tuple[np.ndarray, list[StemPair]]:
    """Camera indices (K, 2) of the pairs whose two images have a pose, with those pairs."""
    index = {label: position for position, label in enumerate(labels)}
    placed = [pair for pair in pairs if pair[0] in index and pair[1] in index]
    indices = np.array(
        [[index[first], index[second]] for first, second in placed], dtype=np.int64
    )
    return indices.reshape(-1, 2), placed


def stem_pair(first: str, second: str) -> StemPair:
    return (first, second) if first < second else (second, first)
