"""Is the overlap-based selection better than taking one image in k, or images at random?

    uv run python scripts/overlap/compare_selection_rules.py   # ~40 s

Judged on Mermaid from the reference poses, on two counts: what the kept images still let a
reconstruction see (area, depth noise) and whether they still chain together (weak links).
Reads the overlaps of a sequential run in the `pairs` table; no matching.
"""

import argparse
from pathlib import Path

import numpy as np

from dataset_automation.overlap.extraction import extract_subset
from dataset_automation.overlap.stored_pairs import (
    latest_overlap_run,
    load_stored_sequence,
)
from dataset_automation.reference.evaluation import evaluate_selection
from dataset_automation.reference.mermaid import load_mermaid_reference
from dataset_automation.reference.observability import grid_over
from dataset_automation.storage.database import RunId, open_database

DATABASE = Path("outputs/shared.sqlite")
POSES_PATH = Path("data/mermaid/107177.xml")
OVERLAP_THRESHOLDS = (0.90, 0.80, 0.75, 0.70, 0.60, 0.50)
DECIMATIONS = (2, 3, 4)
RANDOM_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--overlap-run", type=int, help="run that wrote the pairs (default: latest)"
    )
    parser.add_argument("--database", type=Path, default=DATABASE)
    arguments = parser.parse_args()

    connection = open_database(arguments.database)
    overlap_run = (
        RunId(arguments.overlap_run)
        if arguments.overlap_run is not None
        else latest_overlap_run(connection)
    )
    if overlap_run is None:
        raise SystemExit("no overlap run yet: run scripts/overlap/run_sequential.py")
    sequence = load_stored_sequence(connection, overlap_run)
    reference = load_mermaid_reference(POSES_PATH)
    camera_of_stem = {label: i for i, label in enumerate(reference.poses.labels)}
    cameras = [camera_of_stem[path.stem] for path in sequence.paths]
    grid = grid_over(reference.footprints)

    print(f"overlaps of run {overlap_run}, {len(cameras)} images")
    print(
        f"{'rule':<18} kept | inside    rim | depth noise: median, worst tenth "
        "| links < 0.5, < 0.3"
    )
    for name, kept in subsets(len(cameras), sequence.earlier_in_later):
        evaluation = evaluate_selection(
            grid,
            reference.footprints,
            reference.centres,
            reference.seafloor_height_m,
            cameras,
            [cameras[position] for position in kept],
        )
        print(
            f"{name:<18} {len(kept):4d} | {evaluation.inside_kept_share:6.1%} "
            f"{evaluation.rim_kept_share:6.1%} "
            f"| x{evaluation.median_depth_noise:.2f}, "
            f"x{evaluation.worst_tenth_depth_noise:.2f}"
            f"{'':18}| {evaluation.weak_links[0.5][1]:4d}, "
            f"{evaluation.weak_links[0.3][1]:3d}"
        )


def subsets(
    image_count: int, overlaps: dict[tuple[int, int], float]
) -> list[tuple[str, list[int]]]:
    """Overlap-based selections, then decimations and random subsets of comparable sizes."""
    random = np.random.default_rng(RANDOM_SEED)
    by_overlap = [
        (
            f"overlap >= {threshold:.2f}",
            extract_subset(image_count, overlaps, threshold),
        )
        for threshold in OVERLAP_THRESHOLDS
    ]
    decimated = [
        (f"1 image in {step}", list(range(0, image_count, step)))
        for step in DECIMATIONS
    ]
    sizes = sorted({len(kept) for _, kept in (by_overlap[1], *decimated)}, reverse=True)
    randomly = [
        ("random", sorted(random.choice(image_count, size, replace=False).tolist()))
        for size in sizes
    ]
    return [
        ("all images", list(range(image_count))),
        *by_overlap,
        *decimated,
        *randomly,
    ]


if __name__ == "__main__":
    main()
