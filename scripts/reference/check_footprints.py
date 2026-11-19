"""Sanity check of the Mermaid ground footprints on the horizontal-plane seafloor model."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from dataset_automation.reference.footprints import (
    footprint_polygon,
    invalid_footprints,
)
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.reference.mermaid import load_mermaid_reference

POSES_PATH = Path("data/mermaid/107177.xml")
OUTPUT_DIR = Path("outputs")
CONSECUTIVE_SHOWN = range(0, 40, 4)


def plot_footprints(footprints: np.ndarray, centres: np.ndarray) -> Figure:
    figure = Figure(figsize=(10, 8))
    ax = figure.subplots()
    ax.scatter(
        centres[:, 0], centres[:, 1], s=2, c="lightgray", label="all camera centres"
    )
    for i in CONSECUTIVE_SHOWN:
        polygon = footprint_polygon(footprints, i)
        if polygon is None:
            continue
        xs, ys = polygon.exterior.xy
        (line,) = ax.plot(xs, ys, linewidth=1)
        ax.plot(*centres[i, :2], "o", color=line.get_color(), markersize=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Footprints of images {CONSECUTIVE_SHOWN.start}–{CONSECUTIVE_SHOWN.stop - 1} (every {CONSECUTIVE_SHOWN.step})"
    )
    ax.legend(loc="upper right")
    return figure


def main() -> None:
    reference = load_mermaid_reference(POSES_PATH)
    poses, centres, footprints = (
        reference.poses,
        reference.centres,
        reference.footprints,
    )

    invalid = invalid_footprints(footprints)
    polygons = [footprint_polygon(footprints, i) for i in range(len(poses))]
    areas = np.array([p.area if p is not None else np.nan for p in polygons])
    altitudes = centres[:, 2] - reference.seafloor_height_m
    gsd_mm = 1000 * altitudes / MERMAID_INTRINSICS.focal_px

    print(f"seafloor plane z = {reference.seafloor_height_m:.2f} m")
    print(f"invalid footprints: {invalid.sum()} / {len(poses)}")
    print(
        f"non-simple polygons: {sum(p is not None and not p.is_valid for p in polygons)}"
    )
    print(
        f"altitude (m)   p1/median/p99: {np.percentile(altitudes, [1, 50, 99]).round(2)}"
    )
    print(
        f"nadir GSD (mm) p1/median/p99: {np.percentile(gsd_mm, [1, 50, 99]).round(2)}"
    )
    print(
        f"area (m²)      p1/median/p99: {np.nanpercentile(areas, [1, 50, 99]).round(2)}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_footprints(footprints, centres).savefig(
        OUTPUT_DIR / "step2b_footprints.png", dpi=150
    )
    print(f"figure written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
