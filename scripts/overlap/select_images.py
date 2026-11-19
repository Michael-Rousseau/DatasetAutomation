"""Select the images to keep so that consecutive kept images still overlap enough.

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --register
    uv run python scripts/overlap/select_images.py --overlap 0.80

Reads the overlaps already in `pairs` and does no matching, so another threshold takes a
second. Nothing is deleted. Written to `<output>/run_<id>/`:

    kept_images.txt   one image path per line, for Metashape or COLMAP
    images.csv        one line per image: kept or removed, and its overlap with the kept image
    curve.csv, .png   images removed for a range of thresholds
    summary.txt       totals and the parameters used

The overlap compared with the threshold is the share of the kept image seen by the next one.

On Mermaid, `--mermaid-poses data/mermaid/107177.xml` adds an evaluation from the reference
poses to the summary: the seafloor area a 3D reconstruction can still see with the kept images.
"""

import argparse
import csv
from pathlib import Path

from matplotlib.figure import Figure

from dataset_automation.overlap.extraction import (
    CurvePoint,
    SelectionRow,
    count_weak_links,
    extract_subset,
    extraction_curve,
    selection_rows,
)
from dataset_automation.overlap.stored_pairs import (
    StoredSequence,
    latest_overlap_run,
    load_stored_sequence,
)
from dataset_automation.reference.evaluation import RIM_M, evaluate_selection
from dataset_automation.reference.images import match_poses_to_images
from dataset_automation.reference.mermaid import load_mermaid_reference
from dataset_automation.reference.observability import (
    MIN_FAN_DEG,
    MIN_VIEWS,
    grid_over,
)
from dataset_automation.storage.database import RunId, open_database, start_run

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/selection")
MODULE = "selection"
DEFAULT_OVERLAP = 0.80
CURVE_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--overlap-run", type=int, help="run that wrote the pairs (default: latest)"
    )
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mermaid-poses",
        type=Path,
        help="reference poses of Mermaid (XML): evaluate what the kept images still see",
    )
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
    if not sequence.paths:
        raise SystemExit(f"run {overlap_run} has no pair in {arguments.database}")

    image_count, overlaps = len(sequence.paths), sequence.earlier_in_later
    kept = extract_subset(image_count, overlaps, arguments.overlap)
    rows = selection_rows(image_count, overlaps, kept)
    curve = extraction_curve(image_count, overlaps, CURVE_THRESHOLDS)

    run_id = start_run(
        connection,
        MODULE,
        {
            "overlap_run": overlap_run,
            "threshold": arguments.overlap,
            "measure": "earlier_in_later",
        },
    )
    run_directory = arguments.output / f"run_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "kept_images.txt").write_text(
        "".join(f"{sequence.paths[position]}\n" for position in kept)
    )
    write_images(run_directory / "images.csv", sequence.paths, rows)
    write_curve(run_directory / "curve.csv", curve)
    plot_curve(curve, arguments.overlap).savefig(run_directory / "curve.png", dpi=110)
    summary = summarize(sequence, rows, arguments.overlap, overlap_run, run_id)
    if arguments.mermaid_poses is not None:
        summary += evaluate_with_mermaid_poses(
            arguments.mermaid_poses, sequence.paths, kept
        )
    (run_directory / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"written to {run_directory}")


def write_images(
    report_path: Path, paths: list[Path], rows: list[SelectionRow]
) -> None:
    with report_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "status", "compared_with", "overlap"])
        for row in rows:
            writer.writerow(
                [
                    paths[row.position],
                    "kept" if row.is_kept else "removed",
                    "" if row.reference is None else paths[row.reference],
                    ""
                    if row.overlap_with_reference is None
                    else f"{row.overlap_with_reference:.4f}",
                ]
            )


def write_curve(report_path: Path, curve: list[CurvePoint]) -> None:
    with report_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["overlap", "images", "kept", "removed", "removed_share"])
        for point in curve:
            writer.writerow(
                [
                    point.threshold,
                    point.image_count,
                    point.kept_count,
                    point.removed_count,
                    f"{point.removed_share:.4f}",
                ]
            )


def plot_curve(curve: list[CurvePoint], chosen_overlap: float) -> Figure:
    figure = Figure(figsize=(7, 4.5))
    axis = figure.add_subplot()
    axis.plot(
        [100 * point.threshold for point in curve],
        [100 * point.removed_share for point in curve],
        "o-",
    )
    axis.axvline(100 * chosen_overlap, color="grey", linestyle=":", linewidth=1)
    # Highest overlap on the left: the curve then reads "removing more" from left to right.
    axis.invert_xaxis()
    axis.set_xlabel("overlap kept between consecutive selected images (%)")
    axis.set_ylabel("images removed (%)")
    axis.set_title(f"Images removed by the selection ({curve[0].image_count} images)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def summarize(
    sequence: StoredSequence,
    rows: list[SelectionRow],
    threshold: float,
    overlap_run: RunId,
    run_id: RunId,
) -> str:
    image_count = len(rows)
    kept_count = sum(row.is_kept for row in rows)
    removed = image_count - kept_count
    weak = count_weak_links(rows, threshold)
    compared = len(sequence.earlier_in_later) + sequence.unknown_pair_count
    return (
        f"selection run {run_id}, from the overlaps of run {overlap_run}\n"
        f"overlap kept between consecutive selected images: {threshold:.2f}\n"
        f"images: {image_count}\n"
        f"kept: {kept_count}\n"
        f"removed: {removed} ({removed / image_count:.1%})\n"
        f"pairs compared: {compared}, of which unknown: {sequence.unknown_pair_count}\n"
        f"links between kept images below {threshold:.2f}: {weak.below_threshold}, "
        f"unknown: {weak.unknown} (the sequence itself lacks the overlap there)\n"
        "only neighbours in time are compared: redundancy between survey lines or passes "
        "is not removed\n"
    )


def evaluate_with_mermaid_poses(
    poses_path: Path, paths: list[Path], kept: list[int]
) -> str:
    """Seafloor area a 3D reconstruction can see with all images and with the kept ones."""
    reference = load_mermaid_reference(poses_path)
    labels = reference.poses.labels
    pose_of_stem = {label: index for index, label in enumerate(labels)}
    match = match_poses_to_images(labels, paths)
    if match.images_without_pose:
        raise SystemExit(
            f"{len(match.images_without_pose)} images have no pose in {poses_path}"
        )
    grid = grid_over(reference.footprints)
    cameras = [pose_of_stem[path.stem] for path in paths]
    evaluation = evaluate_selection(
        grid,
        reference.footprints,
        reference.centres,
        reference.seafloor_height_m,
        cameras,
        [cameras[position] for position in kept],
    )
    links = "; ".join(
        f"{kept_count} below {overlap:.2f} ({all_count} with all images)"
        for overlap, (all_count, kept_count) in evaluation.weak_links.items()
    )
    return (
        "evaluation from the reference poses (flat seafloor, cells of "
        f"{grid.cell_m * 100:.0f} cm; reconstructible = at least {MIN_VIEWS} images "
        f"from directions at least {MIN_FAN_DEG:.0f} degrees apart)\n"
        f"reconstructible area: {evaluation.reconstructible_m2:.0f} m2 with all images, "
        f"{evaluation.kept_reconstructible_m2:.0f} m2 with the kept ones "
        f"({evaluation.kept_share:.1%})\n"
        f"  more than {RIM_M:.0f} m inside that area: "
        f"{evaluation.inside_kept_share:.1%} kept; on its rim: "
        f"{evaluation.rim_kept_share:.1%} kept\n"
        "depth noise where still reconstructible (1.00 = as with all images): "
        f"median x{evaluation.median_depth_noise:.2f}, "
        f"one place in ten above x{evaluation.worst_tenth_depth_noise:.2f}\n"
        f"images seeing a place (median): {evaluation.median_views[0]:.0f} "
        f"-> {evaluation.median_views[1]:.0f}\n"
        f"consecutive kept images whose mutual overlap is low: {links}\n"
    )


if __name__ == "__main__":
    main()
