"""Overlap of each image with the ones that follow it, recorded in the shared `pairs` table.

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --neighbours 5

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --register

Images must be in the `images` table; until person C's ingestion exists, `--register` adds the
JPEG and PNG files found under the folder. Capture order is the order of the file paths, until
ingestion fills `images.timestamp_s`. No calibration is applied: overlaps are measured in raw
pixels.
"""

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from dataset_automation.cleaning.pipeline import images_under
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.pairwise import (
    MIN_INLIER_SPREAD,
    MIN_INLIERS,
    RANSAC_THRESHOLD_PX,
)
from dataset_automation.overlap.pipeline import (
    MODULE,
    SequenceFile,
    record_file_sequence,
)
from dataset_automation.storage.database import open_database, start_run
from dataset_automation.storage.images import register_image_files

DATABASE = Path("outputs/shared.sqlite")
DEFAULT_NEIGHBOURS = 5


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--neighbours", type=int, default=DEFAULT_NEIGHBOURS)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument(
        "--register",
        action="store_true",
        help="add image files missing from the images table first",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arguments.database.parent.mkdir(parents=True, exist_ok=True)
    connection = open_database(arguments.database)
    if arguments.register:
        added = register_image_files(
            connection, arguments.image_root, f"{MODULE}-register"
        )
        print(f"registered {added} images")
    files = sorted(
        (
            SequenceFile(image.image_id, image.path)
            for image in images_under(connection, arguments.image_root)
        ),
        key=lambda file: file.path,
    )
    if not files:
        raise SystemExit(
            f"no image under {arguments.image_root} in {arguments.database}: "
            "run ingestion first, or pass --register"
        )

    extractor, matcher = SiftExtractor(), RatioTestMatcher()
    run_id = start_run(
        connection,
        MODULE,
        {
            "mode": "sequential",
            "image_root": str(arguments.image_root.resolve()),
            "neighbour_count": arguments.neighbours,
            "extractor": asdict(extractor),
            "matcher": asdict(matcher),
            "method": "homography",
            "calibrated": False,
            "min_inliers": MIN_INLIERS,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "min_inlier_spread": MIN_INLIER_SPREAD,
        },
    )
    print(f"run {run_id}: {len(files)} images, {arguments.neighbours} neighbours each")
    report = record_file_sequence(
        connection, run_id, files, extractor, matcher, arguments.neighbours
    )

    recording = report.recording
    unknown_count = sum(recording.unknown.values())
    print(f"{recording.known} known overlaps, {unknown_count} unknown")
    for reason, count in recording.unknown.most_common():
        print(f"  {reason.value}: {count}")
    print(f"{len(report.unreadable)} unreadable images")


if __name__ == "__main__":
    main()
