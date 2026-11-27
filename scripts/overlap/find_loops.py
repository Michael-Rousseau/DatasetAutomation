"""Loop closures: links between images far apart in time that see the same place.

    uv run python scripts/overlap/find_loops.py data/mermaid/images

For each image, the far images that resemble it most are proposed, then confirmed by matching.
Confirmed and unconfirmed pairs go to the shared `pairs` table under a run of their own, the
codebook to `<output>/run_<id>/codebook.npy`. About 25 min for 1,244 images with 10
candidates each, once the features are cached; extracting them first adds about 10 min.

Images must be in the `images` table: `--register` adds the files found under the folder.
`--mermaid-calibration` applies Mermaid's calibration and also writes the camera displacement
of each pair, the angle between the two views, to `<output>/run_<id>/displacements.csv`.
"""

import argparse
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path

from run_sequential import sequence_files

from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.features.vlad import DEFAULT_WORD_COUNT
from dataset_automation.overlap.loop_pipeline import (
    CODEBOOK_SAMPLES_PER_IMAGE,
    MODULE,
    LoopReport,
    record_loop_closures,
)
from dataset_automation.overlap.pairwise import (
    MIN_INLIER_SPREAD,
    MIN_INLIERS,
    RANSAC_THRESHOLD_PX,
    SPREAD_INLIER_SHARE,
)
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.storage.database import open_database, start_run

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/loops")
CACHE_ROOT = Path("outputs/cache")
DEFAULT_CANDIDATES = 10
# Images this close in time are neighbours: the sequential mode compares them.
DEFAULT_MIN_GAP = 30


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments.database.parent.mkdir(parents=True, exist_ok=True)
    connection = open_database(arguments.database)
    files = sequence_files(connection, arguments.image_root, arguments.register)
    extractor, matcher = SiftExtractor(), RatioTestMatcher()
    cache = arguments.cache or default_cache(arguments.image_root, extractor)

    run_id = start_run(
        connection, MODULE, run_parameters(arguments, extractor, matcher, cache)
    )
    output = arguments.output / f"run_{run_id}"
    print(
        f"run {run_id}: {len(files)} images, {arguments.candidates} candidates each, "
        f"more than {arguments.min_gap} images apart"
    )
    report = record_loop_closures(
        connection,
        run_id,
        files,
        cache,
        extractor,
        matcher,
        arguments.candidates,
        arguments.min_gap,
        MERMAID_INTRINSICS if arguments.mermaid_calibration else None,
        output,
    )
    print_report(report, output)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--min-gap", type=int, default=DEFAULT_MIN_GAP)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--cache",
        type=Path,
        help="folder of the features on disk; by default one per image folder and "
        "extractor setting, under outputs/cache/",
    )
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


def default_cache(image_root: Path, extractor: SiftExtractor) -> Path:
    """The cache is keyed by file name: a folder per dataset and per extractor setting keeps
    two dives, or two settings, from reading each other's features."""
    root = image_root.resolve()
    key = hashlib.sha256(f"{root}|{asdict(extractor)}".encode()).hexdigest()[:8]
    return CACHE_ROOT / f"{root.name}_{extractor.name}_{key}"


def run_parameters(
    arguments: argparse.Namespace,
    extractor: SiftExtractor,
    matcher: RatioTestMatcher,
    cache: Path,
) -> dict[str, object]:
    return {
        "mode": "loops",
        "image_root": str(arguments.image_root.resolve()),
        "candidates_per_image": arguments.candidates,
        "min_gap": arguments.min_gap,
        "word_count": DEFAULT_WORD_COUNT,
        "codebook_samples_per_image": CODEBOOK_SAMPLES_PER_IMAGE,
        "feature_cache": str(cache.resolve()),
        "extractor": asdict(extractor),
        "matcher": asdict(matcher),
        "method": "homography",
        "calibration": "mermaid" if arguments.mermaid_calibration else None,
        "min_inliers": MIN_INLIERS,
        "ransac_threshold_px": RANSAC_THRESHOLD_PX,
        "min_inlier_spread": MIN_INLIER_SPREAD,
        "spread_inlier_share": SPREAD_INLIER_SHARE,
    }


def print_report(report: LoopReport, output: Path) -> None:
    recording = report.recording
    print(f"{report.candidate_count} candidate pairs, {recording.known} confirmed")
    for reason, count in recording.unknown.most_common():
        print(f"  {reason.value}: {count}")
    if report.image_count:
        share = len(report.linked_images) / report.image_count
        print(
            f"{len(report.linked_images)} of {report.image_count} images linked "
            f"to a far image ({share:.0%})"
        )
    print(f"{len(report.unreadable)} unreadable images")
    print(f"codebook and displacements: {output}")


if __name__ == "__main__":
    main()
