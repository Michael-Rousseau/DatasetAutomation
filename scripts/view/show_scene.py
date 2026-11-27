"""3D view of a Mermaid survey: trajectory, cameras, footprints, loop links and their angles,
and the images a selection removed.

    uv sync --extra view
    uv run python scripts/view/show_scene.py                          # poses only
    uv run python scripts/view/show_scene.py --reference-loops        # + revisits from the poses
    uv run python scripts/view/show_scene.py \\
        --selection outputs/selection/run_40/images.csv \\
        --database outputs/shared.sqlite \\
        --angles outputs/loops/run_31/displacements.csv \\
        --screenshot outputs/view/scene.png

Cameras are placed by the reference poses, so only Mermaid can be drawn for now. Layers:

    trajectory   camera centres in capture order, black
    cameras      a small pyramid per camera: green kept, red removed, grey without selection
    footprints   what each image sees of the seafloor plane, same colours as the cameras
    links        loop closures, coloured by the angle between the two views

Links come from the latest `find_loops.py` run of `--database` (confirmed ones only), or from
the poses with `--reference-loops`. Their angle is the measured one with `--angles`, else the
one given by the poses. Without `--screenshot`, a window opens: drag to turn, scroll to zoom.
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.overlap.loop_pipeline import MODULE as LOOP_MODULE
from dataset_automation.overlap.stored_pairs import latest_overlap_run
from dataset_automation.reference.footprints import camera_rays
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.reference.mermaid import (
    MermaidReference,
    load_mermaid_reference,
)
from dataset_automation.reference.revisits import far_pairs
from dataset_automation.storage.database import RunId
from dataset_automation.view.inputs import (
    StemPair,
    image_stems,
    pair_indices,
    read_confirmed_links,
    read_displacement_angles,
    read_selection,
    statuses_for,
)
from dataset_automation.view.legend import LegendEntry, legend_figure
from dataset_automation.view.scene import (
    DEFAULT_FRUSTUM_DEPTH_M,
    DEFAULT_MAX_ANGLE_DEG,
    STATUS_COLOURS,
    TRAJECTORY_COLOUR,
    ImageStatus,
    LineSet,
    camera_frustums,
    faded,
    footprint_outlines,
    links,
    reference_view_angles,
    sample_pairs,
    status_colours,
    trajectory,
)

POSES = Path("data/mermaid/107177.xml")
# Same gap as `find_loops.py`: closer images are neighbours in time, not revisits.
REFERENCE_MIN_GAP = 30
# Beyond a few thousand, links fill the view and nothing can be read from them.
DEFAULT_MAX_LINKS = 1500
FOOTPRINT_FADING = 0.65


def main() -> None:
    arguments = parse_arguments()
    print(f"loading the poses of {arguments.poses} (about 15 s)")
    reference = load_mermaid_reference(arguments.poses)
    labels = reference.poses.labels

    statuses = [ImageStatus.UNKNOWN] * len(labels)
    if arguments.selection is not None:
        with arguments.selection.open(newline="") as file:
            statuses = statuses_for(labels, read_selection(file))
    print(describe_statuses(statuses))

    pairs, angles, links_label = load_links(arguments, reference)
    if len(pairs) > arguments.max_links:
        print(f"drawing {arguments.max_links} links of {len(pairs)}, chosen at random")
        links_label += f" ({arguments.max_links} of {len(pairs)} drawn)"
        pairs, angles = sample_pairs(pairs, angles, arguments.max_links)
    layers = build_layers(arguments, reference, statuses, pairs, angles)

    title = f"Mermaid, {len(labels)} images" + (
        f", {links_label}" if links_label else ""
    )
    if arguments.screenshot is None:
        from dataset_automation.view.render import show

        show(layers, title)
        return

    from dataset_automation.view.render import render_image

    # Framed on the cameras: footprints on the flat seafloor spread far beyond them.
    image = render_image(layers, reference.centres, arguments.width, arguments.height)
    figure = legend_figure(
        image,
        title,
        legend_entries(arguments, statuses),
        arguments.max_angle if len(pairs) else None,
        has_unknown_angle=bool(np.isnan(angles).any()),
    )
    arguments.screenshot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.screenshot)
    print(f"written to {arguments.screenshot}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--poses", type=Path, default=POSES, help="Mermaid poses (XML)")
    parser.add_argument(
        "--selection", type=Path, help="images.csv written by select_images.py"
    )
    loops = parser.add_mutually_exclusive_group()
    loops.add_argument(
        "--database", type=Path, help="shared.sqlite holding a find_loops.py run"
    )
    loops.add_argument(
        "--reference-loops",
        action="store_true",
        help="draw the revisits given by the poses instead (mutual overlap 0.5 or more)",
    )
    parser.add_argument(
        "--loop-run", type=int, help="find_loops.py run to draw (default: latest)"
    )
    parser.add_argument(
        "--angles",
        type=Path,
        nargs="+",
        default=[],
        help="displacements.csv files: measured angles instead of the poses' ones",
    )
    parser.add_argument(
        "--footprints",
        choices=("all", "kept", "none"),
        default="all",
        help="footprints to draw (default: all)",
    )
    parser.add_argument(
        "--hide-removed",
        action="store_true",
        help="leave out the cameras and footprints of removed images",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=DEFAULT_MAX_LINKS,
        help=f"links drawn at most, chosen at random (default {DEFAULT_MAX_LINKS})",
    )
    parser.add_argument("--max-angle", type=float, default=DEFAULT_MAX_ANGLE_DEG)
    parser.add_argument(
        "--screenshot", type=Path, help="save an image instead of opening a window"
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    arguments = parser.parse_args()
    if arguments.footprints == "kept" and arguments.selection is None:
        parser.error("--footprints kept needs --selection")
    if arguments.angles and arguments.database is None:
        parser.error("--angles needs --database, which maps their image ids to files")
    return arguments


def load_links(
    arguments: argparse.Namespace, reference: MermaidReference
) -> tuple[np.ndarray, np.ndarray, str]:
    """Camera index pairs (K, 2), their angles (K,) and how to name them in the title."""
    if arguments.reference_loops:
        print("finding the revisits in the poses (about 20 s)")
        found = far_pairs(
            reference.footprints,
            reference.centres,
            reference.seafloor_height_m,
            REFERENCE_MIN_GAP,
        )
        revisit = found.revisits()
        pairs = np.column_stack([found.index_a[revisit], found.index_b[revisit]])
        print(f"links: {len(pairs)} revisits from the poses")
        return pairs, found.view_angle_deg[revisit], "revisits from the poses"

    if arguments.database is None:
        return np.empty((0, 2), dtype=np.int64), np.empty(0), ""

    connection = sqlite3.connect(arguments.database)
    run_id = arguments.loop_run or latest_overlap_run(connection, LOOP_MODULE)
    if run_id is None:
        raise SystemExit(f"no find_loops.py run in {arguments.database}")
    stem_pairs = read_confirmed_links(connection, RunId(run_id))
    pairs, placed = pair_indices(stem_pairs, reference.poses.labels)
    print(
        f"links: {len(pairs)} confirmed loop closures of run {run_id}"
        + skipped_note(len(stem_pairs) - len(placed))
    )

    if not arguments.angles:
        print("angles: from the poses (no --angles given)")
        angles = reference_view_angles(
            reference.footprints,
            reference.centres,
            reference.seafloor_height_m,
            pairs,
        )
        return pairs, angles, f"loop closures of run {run_id}, angles from the poses"

    measured = read_measured_angles(connection, arguments.angles)
    angles = np.array([measured.get(pair, np.nan) for pair in placed])
    print(f"angles: measured for {np.count_nonzero(~np.isnan(angles))} of {len(pairs)}")
    return pairs, angles, f"loop closures of run {run_id}, measured angles"


def read_measured_angles(
    connection: sqlite3.Connection, paths: list[Path]
) -> dict[StemPair, float]:
    stems = image_stems(connection)
    angles: dict[StemPair, float] = {}
    for path in paths:
        with path.open(newline="") as file:
            angles |= read_displacement_angles(file, stems)
    return angles


def build_layers(
    arguments: argparse.Namespace,
    reference: MermaidReference,
    statuses: list[ImageStatus],
    pairs: np.ndarray,
    angles: np.ndarray,
) -> list[LineSet]:
    shown = np.array(
        [
            not (arguments.hide_removed and status is ImageStatus.REMOVED)
            for status in statuses
        ]
    )
    with_footprint = shown & (
        np.array([status is ImageStatus.KEPT for status in statuses])
        if arguments.footprints == "kept"
        else np.full(len(statuses), arguments.footprints == "all")
    )
    colours = status_colours(statuses)
    corners = camera_rays(
        MERMAID_INTRINSICS,
        image_border_pixels(
            MERMAID_INTRINSICS.width_px, MERMAID_INTRINSICS.height_px, 1
        ),
    )
    return [
        footprint_outlines(
            reference.footprints[with_footprint],
            reference.seafloor_height_m,
            faded(colours[with_footprint], FOOTPRINT_FADING),
        ),
        trajectory(reference.centres),
        camera_frustums(
            reference.poses.camera_to_world[shown],
            corners,
            DEFAULT_FRUSTUM_DEPTH_M,
            colours[shown],
        ),
        links(reference.centres, pairs, angles, arguments.max_angle),
    ]


def legend_entries(
    arguments: argparse.Namespace, statuses: list[ImageStatus]
) -> list[LegendEntry]:
    entries: list[LegendEntry] = [("trajectory", TRAJECTORY_COLOUR)]
    present = set(statuses)
    for status, label in (
        (ImageStatus.KEPT, "kept image"),
        (ImageStatus.REMOVED, "removed image"),
        (
            ImageStatus.UNKNOWN,
            "image" if arguments.selection is None else "not selected",
        ),
    ):
        if status in present and not (
            arguments.hide_removed and status is ImageStatus.REMOVED
        ):
            entries.append((label, STATUS_COLOURS[status]))
    return entries


def describe_statuses(statuses: list[ImageStatus]) -> str:
    counts = {status: statuses.count(status) for status in ImageStatus}
    return (
        f"images: {counts[ImageStatus.KEPT]} kept, {counts[ImageStatus.REMOVED]} removed, "
        f"{counts[ImageStatus.UNKNOWN]} without a selection status"
    )


def skipped_note(count: int) -> str:
    return "" if count == 0 else f" ({count} left out: an image has no pose)"


if __name__ == "__main__":
    main()
