"""Loop closures of image files: propose far pairs by resemblance, confirm them, record them."""

import logging
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.interface import FeatureExtractor, FeatureMatcher
from dataset_automation.features.store import load_or_extract
from dataset_automation.features.vlad import (
    Codebook,
    root_sift,
    save_codebook,
    train_codebook,
    vlad,
)
from dataset_automation.overlap.loop_candidates import propose_far_pairs
from dataset_automation.overlap.loop_verification import verify_candidates
from dataset_automation.overlap.pipeline import SequenceFile
from dataset_automation.overlap.recording import (
    RecordingReport,
    record_sequential_overlaps,
    write_displacements,
)
from dataset_automation.overlap.sequential import MeasuredPair, SequenceImage
from dataset_automation.storage.database import RunId

# Not the sequential module's name: the latest sequential run is what a selection reads.
MODULE = "overlap-loops"
CODEBOOK_SAMPLES_PER_IMAGE = 120
PROGRESS_EVERY = 100
PAIR_PROGRESS_EVERY = 500

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopReport:
    """`linked_images` are the files with at least one confirmed far pair."""

    image_count: int
    candidate_count: int
    recording: RecordingReport
    linked_images: list[Path]
    unreadable: list[Path]


def record_loop_closures(
    connection: sqlite3.Connection,
    run_id: RunId,
    files: Sequence[SequenceFile],
    cache_directory: Path,
    extractor: FeatureExtractor,
    matcher: FeatureMatcher,
    candidates_per_image: int,
    min_gap: int,
    intrinsics: Intrinsics | None = None,
    output_directory: Path | None = None,
) -> LoopReport:
    """`files` must be in capture order. Features are read from `cache_directory`, extracted
    there on first use. An unreadable file is skipped.

    With `output_directory`, the codebook is saved there and, with `intrinsics`, the camera
    displacement of each candidate is written to `displacements.csv`.
    """
    _require_distinct_names(files)
    sample, readable, unreadable = _codebook_sample(files, cache_directory, extractor)
    if not readable:
        return LoopReport(0, 0, RecordingReport(), [], unreadable)
    codebook = train_codebook(sample)
    if output_directory is not None:
        save_codebook(output_directory / "codebook.npy", codebook)

    vectors = _image_vectors(readable, cache_directory, extractor, codebook)
    candidates = propose_far_pairs(vectors, candidates_per_image, min_gap)
    logger.info("%d candidate pairs to match", len(candidates))

    def image_at(position: int) -> SequenceImage:
        stored = load_or_extract(readable[position].path, cache_directory, extractor)
        return SequenceImage(stored.features, stored.image_size)

    image_ids = [file.image_id for file in readable]
    linked: set[int] = set()
    measures = _noting_links(
        verify_candidates(candidates, image_at, matcher, intrinsics), linked
    )
    if intrinsics is None or output_directory is None:
        recording = record_sequential_overlaps(connection, run_id, image_ids, measures)
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
        with (output_directory / "displacements.csv").open("w", newline="") as file:
            recording = record_sequential_overlaps(
                connection,
                run_id,
                image_ids,
                write_displacements(file, image_ids, measures),
            )
    return LoopReport(
        image_count=len(readable),
        candidate_count=len(candidates),
        recording=recording,
        linked_images=[readable[position].path for position in sorted(linked)],
        unreadable=unreadable,
    )


def _require_distinct_names(files: Sequence[SequenceFile]) -> None:
    # The feature cache is keyed by file name: two images sharing one would share features.
    seen: dict[str, Path] = {}
    for file in files:
        other = seen.setdefault(file.path.stem, file.path)
        if other != file.path:
            raise ValueError(f"{other} and {file.path} have the same file name")


def _codebook_sample(
    files: Sequence[SequenceFile], cache_directory: Path, extractor: FeatureExtractor
) -> tuple[np.ndarray, list[SequenceFile], list[Path]]:
    """A few descriptors of every image, with the files that could be read and the others.

    The descriptors of a dive do not fit in memory at once, hence a sample here and a second
    reading to build the vectors.
    """
    random = np.random.default_rng(0)
    samples: list[np.ndarray] = []
    readable: list[SequenceFile] = []
    unreadable: list[Path] = []
    for file in files:
        try:
            stored = load_or_extract(file.path, cache_directory, extractor)
        except OSError:
            # A corrupted file must never stop a long run.
            logger.warning("cannot read %s, skipped", file.path)
            unreadable.append(file.path)
            continue
        descriptors = root_sift(stored.features.descriptors)
        kept = random.choice(
            len(descriptors),
            min(CODEBOOK_SAMPLES_PER_IMAGE, len(descriptors)),
            replace=False,
        )
        samples.append(descriptors[kept])
        readable.append(file)
        if len(readable) % PROGRESS_EVERY == 0:
            logger.info("%d images read", len(readable))
    sample = np.vstack(samples) if samples else np.empty((0, 0), dtype=np.float32)
    return sample, readable, unreadable


def _image_vectors(
    files: Sequence[SequenceFile],
    cache_directory: Path,
    extractor: FeatureExtractor,
    codebook: Codebook,
) -> np.ndarray:
    return np.stack(
        [
            vlad(
                root_sift(
                    load_or_extract(
                        file.path, cache_directory, extractor
                    ).features.descriptors
                ),
                codebook,
            )
            for file in files
        ]
    )


def _noting_links(
    measures: Iterable[MeasuredPair], linked: set[int]
) -> Iterator[MeasuredPair]:
    """Pass `measures` through, adding to `linked` the positions of the confirmed pairs."""
    for count, measure in enumerate(measures, start=1):
        if measure.estimate.unknown_reason is None:
            linked.update((measure.index_a, measure.index_b))
        if count % PAIR_PROGRESS_EVERY == 0:
            logger.info("%d pairs matched", count)
        yield measure
