"""Judge the loop closures of a `find_loops.py` run on Mermaid against the reference poses.

    uv run python scripts/overlap/evaluate_loops.py   # ~15 s

Reads the pairs of the run in the `pairs` table and, when the run was calibrated, the angles
of its `displacements.csv`. No matching. The reference is the overlap of the footprints on a
flat seafloor: "no overlap" there means the two images cannot see the same place.
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from dataset_automation.overlap.deduplication import mutual_overlap
from dataset_automation.overlap.loop_pipeline import MODULE as LOOP_MODULE
from dataset_automation.overlap.recording import read_view_angles
from dataset_automation.overlap.stored_pairs import latest_overlap_run
from dataset_automation.reference.mermaid import load_mermaid_reference
from dataset_automation.reference.revisits import REVISIT_MUTUAL_OVERLAP, far_pairs
from dataset_automation.storage.database import RunId, open_database

DATABASE = Path("outputs/shared.sqlite")
LOOPS_DIR = Path("outputs/loops")
POSES_PATH = Path("data/mermaid/107177.xml")
OVERLAP_BANDS = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))
LARGE_OVERLAP_ERRORS = (0.2, 0.4)
ANGLE_LIMITS_DEG = (10.0, 20.0)

Pair = tuple[int, int]


def main() -> None:
    arguments = parse_arguments()
    connection = open_database(arguments.database)
    run = arguments.loop_run or latest_overlap_run(connection, LOOP_MODULE)
    if run is None:
        raise SystemExit("no loop run yet: run scripts/overlap/find_loops.py")
    reference = load_mermaid_reference(arguments.mermaid_poses)
    image_ids = image_ids_in_pose_order(connection, reference.poses.labels)
    estimates = stored_pairs(connection, RunId(run), image_ids)
    if not estimates:
        raise SystemExit(f"run {run} has no pair between Mermaid images")

    min_gap = min(second - first for first, second in estimates) - 1
    image_count = len(image_ids)
    far = far_pairs(
        reference.footprints, reference.centres, reference.seafloor_height_m, min_gap
    )
    reference_overlap = np.zeros((image_count, image_count))
    reference_overlap[far.index_a, far.index_b] = far.mutual_overlap
    reference_angle = np.full((image_count, image_count), np.nan)
    reference_angle[far.index_a, far.index_b] = far.view_angle_deg

    print(f"loop run {run}: {image_count} images, pairs more than {min_gap} apart")
    print_confirmation(estimates, reference_overlap, int(far.revisits().sum()))
    print_linked_images(estimates, reference_overlap, image_count)
    print_overlap_error(estimates, reference_overlap)
    angles_path = arguments.loops_directory / f"run_{run}" / "displacements.csv"
    if angles_path.exists():
        with angles_path.open(newline="") as file:
            angles = read_view_angles(file, image_ids)
        print_angles(estimates, angles, reference_angle)
    else:
        print(f"no {angles_path}: the run was not calibrated, angles are not judged")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--loop-run", type=int, help="run to judge (default: latest)")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--loops-directory", type=Path, default=LOOPS_DIR)
    parser.add_argument("--mermaid-poses", type=Path, default=POSES_PATH)
    return parser.parse_args()


def image_ids_in_pose_order(
    connection: sqlite3.Connection, labels: list[str]
) -> list[int]:
    id_of_stem = {
        Path(path).stem: image_id
        for image_id, path in connection.execute("SELECT id, path FROM images")
    }
    missing = [label for label in labels if label not in id_of_stem]
    if missing:
        raise SystemExit(f"{len(missing)} poses have no registered image")
    return [id_of_stem[label] for label in labels]


def stored_pairs(
    connection: sqlite3.Connection, run: RunId, image_ids: list[int]
) -> dict[Pair, float | None]:
    """Every pair the run compared, by positions; None where the overlap is unknown."""
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    rows = connection.execute(
        "SELECT image_a_id, image_b_id, overlap_a_in_b, overlap_b_in_a FROM pairs "
        "WHERE run_id = ?",
        (run,),
    )
    return {
        (
            min(position[image_a], position[image_b]),
            max(position[image_a], position[image_b]),
        ): None if a_in_b is None else mutual_overlap(a_in_b, b_in_a)
        for image_a, image_b, a_in_b, b_in_a in rows
        if image_a in position and image_b in position
    }


def print_confirmation(
    estimates: dict[Pair, float | None],
    reference_overlap: np.ndarray,
    revisit_count: int,
) -> None:
    confirmed = [pair for pair, overlap in estimates.items() if overlap is not None]
    print(f"\ncandidate pairs: {len(estimates)}, confirmed: {len(confirmed)}")
    print("reference overlap | candidates | confirmed")
    without = [pair for pair in estimates if reference_overlap[pair] == 0]
    false_links = sum(estimates[pair] is not None for pair in without)
    print(f"{'none':>17} | {len(without):10d} | {false_links} (false links)")
    for low, high in OVERLAP_BANDS:
        band = [
            pair
            for pair in estimates
            if low <= reference_overlap[pair] < high and reference_overlap[pair] > 0
        ]
        kept = sum(estimates[pair] is not None for pair in band)
        share = f"{kept / len(band):.0%}" if band else "-"
        label = f"{low:.1f} to {min(high, 1.0):.1f}"
        print(f"{label:>17} | {len(band):10d} | {kept} ({share})")
    revisits = [
        pair for pair in confirmed if reference_overlap[pair] >= REVISIT_MUTUAL_OVERLAP
    ]
    print(
        f"revisits (reference overlap of {REVISIT_MUTUAL_OVERLAP} or more) confirmed: "
        f"{len(revisits)} of the {revisit_count} of the dive "
        f"({len(revisits) / revisit_count:.0%})"
    )


def print_linked_images(
    estimates: dict[Pair, float | None],
    reference_overlap: np.ndarray,
    image_count: int,
) -> None:
    links = np.zeros(image_count, dtype=int)
    for (first, second), overlap in estimates.items():
        if overlap is not None:
            links[first] += 1
            links[second] += 1
    symmetric = np.maximum(reference_overlap, reference_overlap.T)
    has_revisit = (symmetric >= REVISIT_MUTUAL_OVERLAP).any(axis=1)
    unlinked = links == 0
    print(
        f"\nimages linked to a far image: {int((~unlinked).sum())} of {image_count} "
        f"({(~unlinked).mean():.0%}), {np.median(links):.0f} links each in median"
    )
    print(
        f"images without a link: {int(unlinked.sum())}, of which "
        f"{int((unlinked & has_revisit).sum())} have a revisit in the reference"
    )


def print_overlap_error(
    estimates: dict[Pair, float | None], reference_overlap: np.ndarray
) -> None:
    error = np.array(
        [
            abs(overlap - reference_overlap[pair])
            for pair, overlap in estimates.items()
            if overlap is not None
        ]
    )
    beyond = ", ".join(
        f"by more than {limit}: {int((error > limit).sum())} ({(error > limit).mean():.1%})"
        for limit in LARGE_OVERLAP_ERRORS
    )
    print(
        f"\nmutual overlap of a confirmed pair: off by {np.median(error):.3f} in median; "
        f"{beyond}"
    )


def print_angles(
    estimates: dict[Pair, float | None],
    angles: dict[Pair, float],
    reference_angle: np.ndarray,
) -> None:
    confirmed = [pair for pair, overlap in estimates.items() if overlap is not None]
    measured = np.array([angles.get(pair, np.nan) for pair in confirmed])
    expected = np.array([reference_angle[pair] for pair in confirmed])
    known = ~np.isnan(measured)
    both = known & ~np.isnan(expected)
    print(
        f"\nangle between the two views of a confirmed pair: known for "
        f"{int(known.sum())} of {len(confirmed)}"
    )
    print(
        f"  measured {np.median(measured[known]):.1f} deg in median, reference "
        f"{np.median(expected[both]):.1f} deg; off by "
        f"{np.median(np.abs(measured[both] - expected[both])):.1f} deg in median"
    )
    for limit in ANGLE_LIMITS_DEG:
        print(
            f"  {limit:.0f} deg or more: {(measured[known] >= limit).mean():.0%} "
            "of the pairs"
        )


if __name__ == "__main__":
    main()
