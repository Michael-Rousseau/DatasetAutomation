"""Overlap of each image with the ones that follow it, recorded in the shared `pairs` table.

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --neighbours 5

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --register

Images must be in the `images` table; until person C's ingestion exists, `--register` adds the
JPEG and PNG files found under the folder. Capture order is the order of the file paths, until
ingestion fills `images.timestamp_s`.

Without a calibration, overlaps are measured in raw pixels. There is no calibration file format
yet: `--mermaid-calibration` applies Mermaid's, and then also writes the camera displacement of
each pair to `<output>/run_<id>/displacements.csv`.
"""

import argparse
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path

from dataset_automation.cleaning.pipeline import images_under
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.pairwise import (
    MIN_INLIER_SPREAD,
    MIN_INLIERS,
    RANSAC_THRESHOLD_PX,
    SPREAD_INLIER_SHARE,
)
from dataset_automation.overlap.pipeline import (
    MODULE,
    SequenceFile,
    SequenceReport,
    record_file_sequence,
)
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.storage.database import open_database, start_run
from dataset_automation.storage.images import register_image_files

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/overlap")
DEFAULT_NEIGHBOURS = 5


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments.database.parent.mkdir(parents=True, exist_ok=True)
    connection = open_database(arguments.database)
    files = sequence_files(connection, arguments.image_root, arguments.register)

    extractor, matcher = SiftExtractor(), RatioTestMatcher()
    run_id = start_run(
        connection, MODULE, run_parameters(arguments, extractor, matcher)
    )
    print(f"run {run_id}: {len(files)} images, {arguments.neighbours} neighbours each")
    intrinsics = MERMAID_INTRINSICS if arguments.mermaid_calibration else None
    displacements_path = (
        arguments.output / f"run_{run_id}" / "displacements.csv"
        if arguments.mermaid_calibration
        else None
    )
    report = record_file_sequence(
        connection,
        run_id,
        files,
        extractor,
        matcher,
        arguments.neighbours,
        intrinsics,
        displacements_path,
    )
    print_report(report, displacements_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--neighbours", type=int, default=DEFAULT_NEIGHBOURS)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mermaid-calibration",
        action="store_true",
        help="undistort with Mermaid's camera and measure the camera displacement",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="add image files missing from the images table first",
    )
    return parser.parse_args()


def sequence_files(
    connection: sqlite3.Connection, image_root: Path, register_first: bool
) -> list[SequenceFile]:
    """Registered image files under `image_root`, in capture order (the order of the paths)."""
    if register_first:
        added = register_image_files(connection, image_root, f"{MODULE}-register")
        print(f"registered {added} images")
    files = sorted(
        (
            SequenceFile(image.image_id, image.path)
            for image in images_under(connection, image_root)
        ),
        key=lambda file: file.path,
    )
    if not files:
        raise SystemExit(
            f"no registered image under {image_root}: run ingestion first, "
            "or pass --register"
        )
    return files


def run_parameters(
    arguments: argparse.Namespace, extractor: SiftExtractor, matcher: RatioTestMatcher
) -> dict[str, object]:
    return {
        "mode": "sequential",
        "image_root": str(arguments.image_root.resolve()),
        "neighbour_count": arguments.neighbours,
        "extractor": asdict(extractor),
        "matcher": asdict(matcher),
        "method": "homography",
        "calibration": "mermaid" if arguments.mermaid_calibration else None,
        "min_inliers": MIN_INLIERS,
        "ransac_threshold_px": RANSAC_THRESHOLD_PX,
        "min_inlier_spread": MIN_INLIER_SPREAD,
        "spread_inlier_share": SPREAD_INLIER_SHARE,
    }


def print_report(report: SequenceReport, displacements_path: Path | None) -> None:
    recording = report.recording
    unknown_count = sum(recording.unknown.values())
    print(f"{recording.known} known overlaps, {unknown_count} unknown")
    for reason, count in recording.unknown.most_common():
        print(f"  {reason.value}: {count}")
    print(f"{len(report.unreadable)} unreadable images")
    if displacements_path is not None:
        print(f"camera displacements: {displacements_path}")


if __name__ == "__main__":
    main()
