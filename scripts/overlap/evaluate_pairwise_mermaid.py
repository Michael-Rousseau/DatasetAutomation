"""Evaluate feature-based pairwise overlap against the pose-based reference on Mermaid.

    uv run python scripts/overlap/evaluate_pairwise_mermaid.py [--starts 30]

Pairs: `--starts` positions spread over the dive, each paired with the images 1, 2, 5, 10 and
20 later, so the reference overlap spans 0–100 %. For each method and reference-overlap band:
share of pairs reported unknown, then median and 90th-percentile absolute error and median
signed error (bias) on the known ones. Draws outputs/overlap_estimate_vs_reference.png.

The reference is itself approximate (horizontal seafloor plane, ≥ 2.7 m of relief: see
docs/step0.md), so small errors measure agreement with it, not with the truth.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.features.interface import Features, Matches
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.pairwise import (
    PairOverlapEstimate,
    estimate_homography_overlap,
    estimate_hull_overlap,
)
from dataset_automation.reference.footprints import (
    camera_rays,
    compute_footprints,
    footprint_polygon,
)
from dataset_automation.reference.images import match_poses_to_images
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from dataset_automation.reference.overlap import pair_overlap
from dataset_automation.reference.poses import camera_centres, load_camera_poses
from dataset_automation.reference.seafloor import (
    altitude_from_gsd,
    plane_below_lowest_cameras,
)

POSES_PATH = Path("data/mermaid/107177.xml")
IMAGES_DIRECTORY = Path("data/mermaid/images")
OUTPUT_DIR = Path("outputs")
GAPS = (1, 2, 5, 10, 20)
# Same seafloor model as the reference footprints (docs/step0.md, step 2a).
MIN_GSD_M = 0.0005
BANDS = ((0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2))
SIZE = (MERMAID_INTRINSICS.width_px, MERMAID_INTRINSICS.height_px)

Estimator = Callable[[Features, Features, Matches], PairOverlapEstimate]
METHODS: dict[str, Estimator] = {
    "homography, raw pixels": lambda a, b, m: estimate_homography_overlap(
        a, b, m, SIZE, SIZE
    ),
    "homography, undistorted": lambda a, b, m: estimate_homography_overlap(
        a, b, m, SIZE, SIZE, MERMAID_INTRINSICS
    ),
    "inlier hull, undistorted": lambda a, b, m: estimate_hull_overlap(
        a, b, m, SIZE, SIZE, MERMAID_INTRINSICS
    ),
}


def reference_overlaps(pairs: list[tuple[int, int]]) -> np.ndarray:
    poses = load_camera_poses(POSES_PATH)
    plane = plane_below_lowest_cameras(
        camera_centres(poses)[:, 2],
        altitude_from_gsd(MIN_GSD_M, MERMAID_INTRINSICS.focal_px),
    )
    rays = camera_rays(MERMAID_INTRINSICS, image_border_pixels(*SIZE, 25))
    footprints = compute_footprints(poses, rays, plane)
    values = []
    for a, b in pairs:
        polygon_a, polygon_b = (
            footprint_polygon(footprints, a),
            footprint_polygon(footprints, b),
        )
        if polygon_a is None or polygon_b is None:
            raise SystemExit(f"no reference footprint for pair {a}–{b}")
        values.append(pair_overlap(polygon_a, polygon_b).a_in_b)
    return np.array(values)


def images_in_capture_order() -> list[Path]:
    labels = load_camera_poses(POSES_PATH).labels
    match = match_poses_to_images(labels, sorted(IMAGES_DIRECTORY.rglob("*")))
    if match.poses_without_image:
        raise SystemExit(
            f"{len(match.poses_without_image)} poses have no image: run scripts/fetch_mermaid.py --images"
        )
    return [match.matched[label] for label in labels]


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # imread signals an unreadable file by returning None, not by raising.
    if image is None:
        raise OSError(f"cannot read image {path}")
    return np.asarray(image)


def report(
    name: str, reference: np.ndarray, estimates: list[PairOverlapEstimate]
) -> None:
    known = np.array([estimate.is_known for estimate in estimates])
    values = np.array(
        [
            np.nan if estimate.a_in_b is None else estimate.a_in_b
            for estimate in estimates
        ]
    )
    print(f"\n{name}")
    print(
        f"  {'reference':>11} {'pairs':>6} {'unknown':>8} {'|error| median':>15} {'p90':>6} {'bias':>7}"
    )
    for low, high in BANDS:
        band = (reference >= low) & ((reference < high) | (high == 1.0))
        scored = band & known
        errors = values[scored] - reference[scored]
        summary = (
            f"{np.median(np.abs(errors)):15.3f} {np.percentile(np.abs(errors), 90):6.3f} {np.median(errors):+7.3f}"
            if scored.any()
            else f"{'-':>15} {'-':>6} {'-':>7}"
        )
        print(
            f"  {f'{low:.1f}–{high:.1f}':>11} {band.sum():6d} {100 * (band & ~known).sum() / max(band.sum(), 1):7.0f}% {summary}"
        )


def plot(
    reference: np.ndarray, results: dict[str, list[PairOverlapEstimate]]
) -> Figure:
    figure = Figure(figsize=(6 * len(results), 6))
    for ax, (name, estimates) in zip(figure.subplots(1, len(results)), results.items()):
        known = [
            (r, e.a_in_b) for r, e in zip(reference, estimates) if e.a_in_b is not None
        ]
        unknown = [r for r, e in zip(reference, estimates) if e.a_in_b is None]
        if known:
            ax.scatter(*zip(*known), s=10, label="estimate")
        ax.scatter(
            unknown,
            [-0.05] * len(unknown),
            s=10,
            marker="x",
            color="red",
            label="unknown",
        )
        ax.plot([0, 1], [0, 1], color="grey", linewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.1, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("reference overlap (poses)")
        ax.set_ylabel("estimated overlap (features)")
        ax.set_title(name)
        ax.legend(loc="upper left")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--starts", type=int, default=30, help="positions sampled over the dive"
    )
    arguments = parser.parse_args()

    paths = images_in_capture_order()
    starts = (
        np.linspace(0, len(paths) - 1 - max(GAPS), arguments.starts)
        .astype(int)
        .tolist()
    )
    pairs = [(start, start + gap) for start in starts for gap in GAPS]
    reference = reference_overlaps(pairs)

    extractor, matcher = SiftExtractor(), RatioTestMatcher()
    features: dict[int, Features] = {}
    for index in sorted({image for pair in pairs for image in pair}):
        features[index] = extractor.extract(read_grayscale(paths[index]))
    matches = [matcher.match(features[a], features[b]) for a, b in pairs]
    print(f"{len(pairs)} pairs over {len(features)} images")

    results = {
        name: [
            estimate(features[a], features[b], m) for (a, b), m in zip(pairs, matches)
        ]
        for name, estimate in METHODS.items()
    }
    for name, estimates in results.items():
        report(name, reference, estimates)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot(reference, results).savefig(
        OUTPUT_DIR / "overlap_estimate_vs_reference.png", dpi=110
    )
    print(f"\nfigure written to {OUTPUT_DIR}/overlap_estimate_vs_reference.png")


if __name__ == "__main__":
    main()
