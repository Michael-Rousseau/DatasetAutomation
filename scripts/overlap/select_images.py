"""Select the images to keep, by one of two rules.

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --register
    uv run python scripts/overlap/select_images.py --overlap 0.80
    uv run python scripts/overlap/select_images.py --min-views 5

`--overlap`: consecutive kept images still overlap that much, measured as the share of the
kept image seen by the next one. `--min-views`: every image stays seen by that many kept
images; it thins the middle of a well-covered area more and its rim less. With `--min-views`,
the loop closures of the latest `find_loops.py` run are read too (`--loop-run`, or
`--no-loops`), and `--angles` takes the displacement files of calibrated runs so that two of
the views must also be far enough apart:

    uv run python scripts/overlap/select_images.py --min-views 5 \
        --angles outputs/overlap/run_21/displacements.csv outputs/loops/run_31/displacements.csv

Reads the overlaps already in `pairs` and does no matching, so another setting takes a
second. Nothing is deleted. Written to `<output>/run_<id>/`:

    kept_images.txt   one image path per line, for Metashape or COLMAP
    images.csv        one line per image: kept or removed, and its overlap with the kept image
    curve.csv, .png   images removed for a range of settings
    summary.txt       totals and the parameters used

On Mermaid, `--mermaid-poses data/mermaid/107177.xml` adds an evaluation from the reference
poses to the summary: the seafloor area a 3D reconstruction can still see with the kept images.
"""

import argparse
import csv
import sqlite3
from dataclasses import dataclass
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
from dataset_automation.overlap.loop_pipeline import MODULE as LOOP_MODULE
from dataset_automation.overlap.recording import read_view_angles
from dataset_automation.overlap.stored_pairs import (
    StoredSequence,
    latest_overlap_run,
    load_mutual_overlaps,
    load_stored_sequence,
)
from dataset_automation.overlap.support_selection import (
    CHAIN_OVERLAP,
    MIN_ANGLE_DEG,
    SUPPORT_OVERLAP,
    select_supported_subset,
)
from dataset_automation.reference.evaluation import (
    RIM_M,
    SelectionEvaluation,
    evaluate_selection,
)
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
CURVE_MIN_VIEWS = (10, 8, 6, 5, 4, 3)


@dataclass(frozen=True)
class Selection:
    """What a rule decided: the images kept, its curve, and how to report both.

    `setting_label` names the setting the curve varies; `parameters` go to the `runs` table.
    """

    kept: list[int]
    curve: list[CurvePoint]
    chosen_setting: float
    setting_label: str
    parameters: dict[str, object]
    description: str


def main() -> None:
    arguments = parse_arguments()
    connection = open_database(arguments.database)
    overlap_run = requested_or_latest_run(connection, arguments.overlap_run)
    sequence = load_stored_sequence(connection, overlap_run)
    if not sequence.paths:
        raise SystemExit(f"run {overlap_run} has no pair in {arguments.database}")

    selection = (
        select_by_overlap(sequence, arguments.overlap or DEFAULT_OVERLAP)
        if arguments.min_views is None
        else select_by_support(connection, sequence, arguments)
    )
    kept = selection.kept
    rows = selection_rows(len(sequence.paths), sequence.earlier_in_later, kept)
    run_id = start_run(
        connection, MODULE, {"overlap_run": overlap_run, **selection.parameters}
    )
    summary = summarize(sequence, rows, selection, overlap_run, run_id)
    if arguments.mermaid_poses is not None:
        summary += evaluate_with_mermaid_poses(
            arguments.mermaid_poses, sequence.paths, kept
        )

    run_directory = arguments.output / f"run_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "kept_images.txt").write_text(
        "".join(f"{sequence.paths[position]}\n" for position in kept)
    )
    write_images(run_directory / "images.csv", sequence.paths, rows)
    write_curve(run_directory / "curve.csv", selection)
    plot_curve(selection).savefig(run_directory / "curve.png", dpi=110)
    (run_directory / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"written to {run_directory}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    rule = parser.add_mutually_exclusive_group()
    rule.add_argument(
        "--overlap",
        type=float,
        help=f"overlap kept between consecutive selected images (default {DEFAULT_OVERLAP})",
    )
    rule.add_argument(
        "--min-views",
        type=int,
        help="kept images that must still see each image, instead of --overlap",
    )
    loops = parser.add_mutually_exclusive_group()
    loops.add_argument(
        "--loop-run", type=int, help="find_loops.py run to read (default: latest)"
    )
    loops.add_argument(
        "--no-loops", action="store_true", help="use the neighbours in time only"
    )
    parser.add_argument(
        "--angles",
        type=Path,
        nargs="+",
        default=[],
        help="displacements.csv of calibrated runs: two views must then be "
        f"{MIN_ANGLE_DEG:.0f} degrees apart",
    )
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
    return parser.parse_args()


def requested_or_latest_run(
    connection: sqlite3.Connection, requested: int | None
) -> RunId:
    if requested is not None:
        return RunId(requested)
    latest = latest_overlap_run(connection)
    if latest is None:
        raise SystemExit("no overlap run yet: run scripts/overlap/run_sequential.py")
    return latest


def select_by_overlap(sequence: StoredSequence, threshold: float) -> Selection:
    image_count, overlaps = len(sequence.paths), sequence.earlier_in_later
    return Selection(
        kept=extract_subset(image_count, overlaps, threshold),
        curve=extraction_curve(image_count, overlaps, CURVE_THRESHOLDS),
        chosen_setting=threshold,
        setting_label="overlap",
        parameters={
            "rule": "overlap",
            "threshold": threshold,
            "measure": "earlier_in_later",
        },
        description=(
            f"overlap kept between consecutive selected images: {threshold:.2f}\n"
            "only neighbours in time are compared: redundancy between survey lines or "
            "passes is not removed\n"
        ),
    )


def select_by_support(
    connection: sqlite3.Connection,
    sequence: StoredSequence,
    arguments: argparse.Namespace,
) -> Selection:
    image_count = len(sequence.paths)
    loop_run = (
        None
        if arguments.no_loops
        else arguments.loop_run or latest_overlap_run(connection, LOOP_MODULE)
    )
    loops = (
        {}
        if loop_run is None
        else load_mutual_overlaps(connection, RunId(loop_run), sequence.image_ids)
    )
    overlaps = sequence.mutual_overlaps | loops
    angles: dict[tuple[int, int], float] | None = None
    for angles_path in arguments.angles:
        with angles_path.open(newline="") as file:
            angles = (angles or {}) | read_view_angles(file, sequence.image_ids)

    return Selection(
        kept=select_supported_subset(
            image_count, overlaps, arguments.min_views, angles
        ),
        curve=[
            CurvePoint(
                min_views,
                image_count,
                len(select_supported_subset(image_count, overlaps, min_views, angles)),
            )
            for min_views in CURVE_MIN_VIEWS
        ],
        chosen_setting=arguments.min_views,
        setting_label="min_views",
        parameters={
            "rule": "support",
            "min_views": arguments.min_views,
            "loop_run": loop_run,
            "angles": [str(path) for path in arguments.angles],
            "support_overlap": SUPPORT_OVERLAP,
            "min_angle_deg": MIN_ANGLE_DEG,
            "chain_overlap": CHAIN_OVERLAP,
        },
        description=(
            f"kept images that still see each image: at least {arguments.min_views}\n"
            f"two images see the same place from a mutual overlap of {SUPPORT_OVERLAP}\n"
            + (
                "loop closures: none read, neighbours in time only\n"
                if loop_run is None
                else f"loop closures: {len(loops)} links of run {loop_run}\n"
            )
            + (
                "angles between views: not read, only the number of views counts\n"
                if angles is None
                else f"angles between views: known for {len(angles)} pairs, two views "
                f"must be {MIN_ANGLE_DEG:.0f} degrees apart\n"
            )
        ),
    )


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


def write_curve(report_path: Path, selection: Selection) -> None:
    with report_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [selection.setting_label, "images", "kept", "removed", "removed_share"]
        )
        for point in selection.curve:
            writer.writerow(
                [
                    point.threshold,
                    point.image_count,
                    point.kept_count,
                    point.removed_count,
                    f"{point.removed_share:.4f}",
                ]
            )


def plot_curve(selection: Selection) -> Figure:
    curve = selection.curve
    by_overlap = selection.setting_label == "overlap"
    scale = 100 if by_overlap else 1
    figure = Figure(figsize=(7, 4.5))
    axis = figure.add_subplot()
    axis.plot(
        [scale * point.threshold for point in curve],
        [100 * point.removed_share for point in curve],
        "o-",
    )
    axis.axvline(
        scale * selection.chosen_setting, color="grey", linestyle=":", linewidth=1
    )
    # Strictest setting on the left: the curve then reads "removing more" from left to right.
    axis.invert_xaxis()
    axis.set_xlabel(
        "overlap kept between consecutive selected images (%)"
        if by_overlap
        else "kept images that must still see each image"
    )
    axis.set_ylabel("images removed (%)")
    axis.set_title(f"Images removed by the selection ({curve[0].image_count} images)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def summarize(
    sequence: StoredSequence,
    rows: list[SelectionRow],
    selection: Selection,
    overlap_run: RunId,
    run_id: RunId,
) -> str:
    image_count = len(rows)
    kept_count = sum(row.is_kept for row in rows)
    removed = image_count - kept_count
    # The overlap rule promises its threshold between kept neighbours; the other rule
    # promises only that they still chain.
    promised = (
        selection.chosen_setting
        if selection.setting_label == "overlap"
        else CHAIN_OVERLAP
    )
    weak = count_weak_links(rows, promised)
    compared = len(sequence.earlier_in_later) + sequence.unknown_pair_count
    return (
        f"selection run {run_id}, from the overlaps of run {overlap_run}\n"
        + selection.description
        + f"images: {image_count}\n"
        f"kept: {kept_count}\n"
        f"removed: {removed} ({removed / image_count:.1%})\n"
        f"pairs compared: {compared}, of which unknown: {sequence.unknown_pair_count}\n"
        f"links between consecutive kept images below {promised:.2f}: "
        f"{weak.below_threshold}, unknown: {weak.unknown}\n"
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
    return describe(evaluation, grid.cell_m)


def describe(evaluation: SelectionEvaluation, cell_m: float) -> str:
    links = "; ".join(
        f"{kept_count} below {overlap:.2f} ({all_count} with all images)"
        for overlap, (all_count, kept_count) in evaluation.weak_links.items()
    )
    return (
        "evaluation from the reference poses (flat seafloor, cells of "
        f"{cell_m * 100:.0f} cm; reconstructible = at least {MIN_VIEWS} images "
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
