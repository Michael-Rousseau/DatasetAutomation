"""Sequential overlaps of image files: read, extract, estimate, record."""

import logging
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.interface import FeatureExtractor, FeatureMatcher
from dataset_automation.overlap.recording import (
    RecordingReport,
    record_sequential_overlaps,
)
from dataset_automation.overlap.sequential import (
    SequenceImage,
    estimate_sequential_overlaps,
)
from dataset_automation.storage.database import RunId

MODULE = "overlap"
PROGRESS_EVERY = 100

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceFile:
    image_id: int
    path: Path


@dataclass(frozen=True)
class SequenceReport:
    recording: RecordingReport
    unreadable: list[Path]


def record_file_sequence(
    connection: sqlite3.Connection,
    run_id: RunId,
    files: Iterable[SequenceFile],
    extractor: FeatureExtractor,
    matcher: FeatureMatcher,
    neighbour_count: int,
    intrinsics: Intrinsics | None = None,
) -> SequenceReport:
    """`files` must be in capture order. An unreadable file is skipped: its two neighbours
    become consecutive."""
    # Filled by the reader as it goes: position i of the sequence the estimator sees is the
    # i-th readable file, which is only known once the files before it have been read.
    read_image_ids: list[int] = []
    unreadable: list[Path] = []
    overlaps = estimate_sequential_overlaps(
        _readable_images(files, extractor, read_image_ids, unreadable),
        neighbour_count,
        matcher,
        intrinsics,
    )
    recording = record_sequential_overlaps(connection, run_id, read_image_ids, overlaps)
    return SequenceReport(recording=recording, unreadable=unreadable)


def _readable_images(
    files: Iterable[SequenceFile],
    extractor: FeatureExtractor,
    read_image_ids: list[int],
    unreadable: list[Path],
) -> Iterator[SequenceImage]:
    for file in files:
        pixels = cv2.imread(str(file.path), cv2.IMREAD_GRAYSCALE)
        # A corrupted file must never stop a long run: imread returns None, log and move on.
        if pixels is None:
            logger.warning("cannot read %s, skipped", file.path)
            unreadable.append(file.path)
            continue
        height, width = pixels.shape
        features = extractor.extract(np.asarray(pixels))
        read_image_ids.append(file.image_id)
        if len(read_image_ids) % PROGRESS_EVERY == 0:
            logger.info("%d images read", len(read_image_ids))
        yield SequenceImage(features, (width, height))
