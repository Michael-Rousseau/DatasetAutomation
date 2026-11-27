"""Write what COLMAP needs to reconstruct a set of images from the pairs already matched.

    uv run python scripts/reconstruction/export_colmap_pairs.py data/mermaid/images
    uv run python scripts/reconstruction/export_colmap_pairs.py data/mermaid/images \
        --kept outputs/selection/run_32/kept_images.txt --output outputs/colmap/min_views_5

Written to `<output>/`:

    image_list.txt   the images to reconstruct, one name per line
    pairs.txt        the pairs to match, two names per line

The pairs are those whose overlap is known in the `pairs` table: neighbours in time
(`run_sequential.py`) and loop closures (`find_loops.py`). COLMAP then matches these instead
of every pair. With `--kept`, the list written by `select_images.py`, only those images and
the pairs between them are written. Names are paths relative to the image folder.
"""

import argparse
from pathlib import Path

from dataset_automation.overlap.loop_pipeline import MODULE as LOOP_MODULE
from dataset_automation.overlap.stored_pairs import (
    latest_overlap_run,
    load_mutual_overlaps,
    load_stored_sequence,
)
from dataset_automation.reconstruction.colmap_files import write_match_list
from dataset_automation.storage.database import RunId, open_database

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/colmap/all_images")


def main() -> None:
    arguments = parse_arguments()
    connection = open_database(arguments.database)
    overlap_run = arguments.overlap_run or latest_overlap_run(connection)
    if overlap_run is None:
        raise SystemExit("no overlap run yet: run scripts/overlap/run_sequential.py")
    sequence = load_stored_sequence(connection, RunId(overlap_run))
    pairs = set(sequence.mutual_overlaps)
    loop_run = (
        None
        if arguments.no_loops
        else arguments.loop_run or latest_overlap_run(connection, LOOP_MODULE)
    )
    if loop_run is not None:
        pairs |= set(
            load_mutual_overlaps(connection, RunId(loop_run), sequence.image_ids)
        )

    image_root = arguments.image_root.resolve()
    names = [
        path.resolve().relative_to(image_root).as_posix() for path in sequence.paths
    ]
    wanted = kept_positions(arguments.kept, sequence.paths)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "image_list.txt").write_text(
        "".join(f"{names[position]}\n" for position in sorted(wanted))
    )
    with (arguments.output / "pairs.txt").open("w") as file:
        pair_count = write_match_list(
            file,
            (
                (names[first], names[second])
                for first, second in sorted(pairs)
                if first in wanted and second in wanted
            ),
        )
    print(
        f"{len(wanted)} of {len(names)} images, {pair_count} pairs "
        f"(overlap run {overlap_run}, loop run {loop_run})"
    )
    print(f"written to {arguments.output}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--kept", type=Path, help="kept_images.txt of a selection")
    parser.add_argument(
        "--overlap-run", type=int, help="sequential run (default: latest)"
    )
    loops = parser.add_mutually_exclusive_group()
    loops.add_argument(
        "--loop-run", type=int, help="find_loops.py run (default: latest)"
    )
    loops.add_argument(
        "--no-loops", action="store_true", help="neighbours in time only"
    )
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def kept_positions(kept_file: Path | None, paths: list[Path]) -> set[int]:
    """Positions of the images to reconstruct: all of them without a selection."""
    if kept_file is None:
        return set(range(len(paths)))
    position = {path.resolve(): index for index, path in enumerate(paths)}
    kept = [Path(line).resolve() for line in kept_file.read_text().splitlines() if line]
    unknown = [path for path in kept if path not in position]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} images of {kept_file} are not in the overlap run, "
            f"first: {unknown[0]}"
        )
    return {position[path] for path in kept}


if __name__ == "__main__":
    main()
