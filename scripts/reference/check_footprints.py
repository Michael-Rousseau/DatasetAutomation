"""Sanity check of the Mermaid ground footprints on the horizontal-plane seafloor model."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from dataset_automation.reference.footprints import (
    DEFAULT_SAMPLES_PER_SIDE,
    camera_rays,
    compute_footprints,
    footprint_polygon,
    image_border_pixels,
    invalid_footprints,
)
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.reference.poses import camera_centres, load_camera_poses
from dataset_automation.reference.seafloor import (
    altitude_from_gsd,
    plane_below_lowest_cameras,
)

POSES_PATH = Path("data/mermaid/107177.xml")
OUTPUT_DIR = Path("outputs")
# Smallest GSD announced for the dataset (report 107174): the lowest cameras see ~0.5 mm/px.
MIN_GSD_M = 0.0005
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
    poses = load_camera_poses(POSES_PATH)
    centres = camera_centres(poses)
    plane = plane_below_lowest_cameras(
        centres[:, 2], altitude_from_gsd(MIN_GSD_M, MERMAID_INTRINSICS.focal_px)
    )
    rays = camera_rays(
        MERMAID_INTRINSICS,
        image_border_pixels(
            MERMAID_INTRINSICS.width_px,
            MERMAID_INTRINSICS.height_px,
            DEFAULT_SAMPLES_PER_SIDE,
        ),
    )
    footprints = compute_footprints(poses, rays, plane)

    invalid = invalid_footprints(footprints)
    polygons = [footprint_polygon(footprints, i) for i in range(len(poses))]
    areas = np.array([p.area if p is not None else np.nan for p in polygons])
    altitudes = centres[:, 2] - plane.height_m
    gsd_mm = 1000 * altitudes / MERMAID_INTRINSICS.focal_px

    print(f"seafloor plane z = {plane.height_m:.2f} m")
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
