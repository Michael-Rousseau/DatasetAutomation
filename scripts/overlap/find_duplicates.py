"""Flag the near-duplicate images of a sequence from the overlaps already in `pairs`.

    uv run python scripts/overlap/run_sequential.py data/mermaid/images --register
    uv run python scripts/overlap/find_duplicates.py --threshold 0.95

Nothing is deleted: the result is a CSV with one line per image, `kept` or `duplicate` of
another image. No matching is done here, so trying another threshold takes a second.
"""

import argparse
import csv
import sqlite3
from pathlib import Path

from dataset_automation.overlap.deduplication import (
    NearDuplicate,
    find_near_duplicates,
)
from dataset_automation.overlap.pipeline import MODULE as OVERLAP_MODULE
from dataset_automation.overlap.stored_pairs import load_mutual_overlaps
from dataset_automation.storage.database import RunId, open_database, start_run

DATABASE = Path("outputs/shared.sqlite")
OUTPUT_DIR = Path("outputs/deduplication")
MODULE = "deduplication"
DEFAULT_THRESHOLD = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--overlap-run", type=int, help="run that wrote the pairs (default: latest)"
    )
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    connection = open_database(arguments.database)
    overlap_run = (
        RunId(arguments.overlap_run)
        if arguments.overlap_run is not None
        else latest_overlap_run(connection)
    )
    sequence = load_mutual_overlaps(connection, overlap_run)
    if not sequence.paths:
        raise SystemExit(f"run {overlap_run} has no pair in {arguments.database}")

    duplicates = find_near_duplicates(
        len(sequence.paths), sequence.mutual_overlaps, arguments.threshold
    )
    run_id = start_run(
        connection,
        MODULE,
        {"overlap_run": overlap_run, "threshold": arguments.threshold},
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output / f"run_{run_id}.csv"
    write_report(report_path, sequence.paths, duplicates)

    share = len(duplicates) / len(sequence.paths)
    print(
        f"run {run_id} (overlaps of run {overlap_run}, threshold {arguments.threshold}): "
        f"{len(duplicates)} near-duplicates among {len(sequence.paths)} images ({share:.1%})"
    )
    print(f"report: {report_path}")


def latest_overlap_run(connection: sqlite3.Connection) -> RunId:
    row = connection.execute(
        "SELECT MAX(id) FROM runs WHERE module = ?", (OVERLAP_MODULE,)
    ).fetchone()
    if row[0] is None:
        raise SystemExit("no overlap run yet: run scripts/overlap/run_sequential.py")
    return RunId(row[0])


def write_report(
    report_path: Path, paths: list[Path], duplicates: list[NearDuplicate]
) -> None:
    by_position = {duplicate.index: duplicate for duplicate in duplicates}
    with report_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "status", "duplicate_of", "mutual_overlap"])
        for position, path in enumerate(paths):
            duplicate = by_position.get(position)
            if duplicate is None:
                writer.writerow([path, "kept", "", ""])
                continue
            writer.writerow(
                [
                    path,
                    "duplicate",
                    paths[duplicate.duplicate_of],
                    f"{duplicate.mutual_overlap:.4f}",
                ]
            )


if __name__ == "__main__":
    main()
