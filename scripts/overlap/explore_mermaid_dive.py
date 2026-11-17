"""What feature-based overlap achieves on the whole Mermaid dive.

    uv run python scripts/overlap/explore_mermaid_dive.py sequential   # each image vs its 5 next
    uv run python scripts/overlap/explore_mermaid_dive.py revisits     # passes 0–200 vs 800–1000

`sequential` plots estimate and reference overlap along the dive, with the texture available
(SIFT keypoints): where does it break, and why? `revisits` asks the n → n question: the diver
came back over the same area hundreds of images later (trajectory in report 107174); among the
pairs the poses say overlap, how many do the images alone find, and how many false alarms?

SIFT features are cached in outputs/cache/ so both runs extract each image once.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure
from shapely import Polygon, STRtree

from dataset_automation.camera.frame import image_border_pixels
from dataset_automation.features.interface import Features
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.pairwise import (
    PairOverlapEstimate,
    estimate_homography_overlap,
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
CACHE_DIR = OUTPUT_DIR / "cache" / "sift_defaults"
SIZE = (MERMAID_INTRINSICS.width_px, MERMAID_INTRINSICS.height_px)
MIN_GSD_M = 0.0005
NEIGHBOURS = 5
FIRST_PASS, SECOND_PASS = range(200), range(800, 1000)
# Pairs the reference says do not overlap at all, matched too to count false alarms.
NEGATIVE_SAMPLES = 400
# 68 % of the 40,000 cross-pass pairs overlap (the diver gridded the same area): matching them
# all takes ~45 min, so each reference-overlap band is sampled instead.
POSITIVE_BANDS = ((0.6, 1.01), (0.4, 0.6), (0.2, 0.4), (1e-9, 0.2))
SAMPLES_PER_BAND = 400


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=("sequential", "revisits"))
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    if arguments.experiment == "sequential":
        sequential()
    else:
        revisits()


def sequential() -> None:
    paths = images_in_capture_order()
    footprints = reference_footprints()
    features = [cached_features(index, path) for index, path in enumerate(paths)]
    matcher = RatioTestMatcher()

    gaps = range(1, NEIGHBOURS + 1)
    estimates = np.full((len(paths), NEIGHBOURS), np.nan)
    references = np.full((len(paths), NEIGHBOURS), np.nan)
    for a in range(len(paths)):
        for gap in gaps:
            b = a + gap
            if b >= len(paths):
                continue
            references[a, gap - 1] = reference_overlap(footprints, a, b)
            estimate = overlap(features[a], features[b], matcher)
            if estimate.a_in_b is not None:
                estimates[a, gap - 1] = estimate.a_in_b
        if a % 100 == 0:
            print(f"  {a}/{len(paths)}", flush=True)

    np.savez(
        OUTPUT_DIR / "sequential_overlaps.npz",
        estimates=estimates,
        references=references,
    )
    keypoints = np.array([len(f) for f in features])
    summarize_sequential(estimates, references)
    plot_sequential(estimates, references, keypoints).savefig(
        OUTPUT_DIR / "sequential_overlaps.png", dpi=90
    )
    print(f"figure written to {OUTPUT_DIR}/sequential_overlaps.png")


def summarize_sequential(estimates: np.ndarray, references: np.ndarray) -> None:
    print(f"\n{'gap':>4} {'pairs':>6} {'unknown':>8} {'|error| median':>15} {'p90':>6}")
    for gap in range(NEIGHBOURS):
        reference, estimate = references[:, gap], estimates[:, gap]
        valid = ~np.isnan(reference)
        known = valid & ~np.isnan(estimate)
        errors = np.abs(estimate[known] - reference[known])
        print(
            f"{gap + 1:4d} {valid.sum():6d} {100 * (valid & ~known).sum() / valid.sum():7.1f}% "
            f"{np.median(errors):15.3f} {np.percentile(errors, 90):6.3f}"
        )


def plot_sequential(
    estimates: np.ndarray, references: np.ndarray, keypoints: np.ndarray
) -> Figure:
    figure = Figure(figsize=(22, 11))
    top, middle, bottom = figure.subplots(3, 1, sharex=True)
    index = np.arange(len(keypoints))
    for ax, gap in ((top, 1), (middle, 5)):
        reference, estimate = references[:, gap - 1], estimates[:, gap - 1]
        ax.plot(index, reference, color="grey", linewidth=1, label="reference (poses)")
        ax.scatter(
            index, estimate, s=4, color="tab:blue", label="estimate (SIFT + homography)"
        )
        unknown = ~np.isnan(reference) & np.isnan(estimate)
        ax.scatter(
            index[unknown],
            np.full(unknown.sum(), -0.05),
            s=4,
            marker="|",
            color="red",
            label="unknown",
        )
        ax.set_ylim(-0.1, 1.05)
        ax.set_ylabel(f"overlap with image +{gap}")
        ax.legend(loc="lower left", fontsize=8)
    bottom.semilogy(index, np.maximum(keypoints, 1), color="tab:green", linewidth=1)
    bottom.set_ylabel("SIFT keypoints")
    bottom.set_xlabel("image index (capture order)")
    figure.tight_layout()
    return figure


def revisits() -> None:
    paths = images_in_capture_order()
    footprints = reference_footprints()
    polygons = {
        i: required_footprint(footprints, i) for i in (*FIRST_PASS, *SECOND_PASS)
    }
    tree = STRtree([polygons[j] for j in SECOND_PASS])

    positives: list[tuple[int, int, float]] = []
    for a in FIRST_PASS:
        for candidate in tree.query(polygons[a]):
            b = SECOND_PASS[int(candidate)]
            value = reference_overlap(footprints, a, b)
            if value > 0:
                positives.append((a, b, value))
    overlapping = {(a, b) for a, b, _ in positives}
    rng = np.random.default_rng(0)
    print(f"{len(positives)} overlapping cross-pass pairs (reference > 0)")
    positives = sample_per_band(positives, rng)
    negatives: list[tuple[int, int, float]] = []
    while len(negatives) < NEGATIVE_SAMPLES:
        a, b = int(rng.choice(FIRST_PASS)), int(rng.choice(SECOND_PASS))
        if (a, b) not in overlapping and not polygons[a].intersects(polygons[b]):
            negatives.append((a, b, 0.0))
    print(
        f"matching {len(positives)} sampled overlapping pairs and {len(negatives)} non-overlapping ones"
    )

    features = {
        i: cached_features(i, paths[i])
        for i in sorted({i for a, b, _ in positives + negatives for i in (a, b)})
    }
    matcher = RatioTestMatcher()
    results = [
        (a, b, value, overlap(features[a], features[b], matcher))
        for a, b, value in positives + negatives
    ]
    summarize_revisits(results)
    plot_revisits(results, footprints).savefig(OUTPUT_DIR / "revisits.png", dpi=90)
    print(f"figure written to {OUTPUT_DIR}/revisits.png")


def sample_per_band(
    pairs: list[tuple[int, int, float]], rng: np.random.Generator
) -> list[tuple[int, int, float]]:
    sampled = []
    for low, high in POSITIVE_BANDS:
        band = [pair for pair in pairs if low <= pair[2] < high]
        chosen = rng.choice(
            len(band), size=min(SAMPLES_PER_BAND, len(band)), replace=False
        )
        sampled.extend(band[i] for i in chosen)
    return sampled


def summarize_revisits(
    results: list[tuple[int, int, float, PairOverlapEstimate]],
) -> None:
    print(f"\n{'reference':>11} {'pairs':>6} {'found':>7} {'|error| median':>15}")
    for low, high in POSITIVE_BANDS:
        band = [
            (value, estimate)
            for _, _, value, estimate in results
            if low <= value < high
        ]
        found = [
            (value, estimate.a_in_b)
            for value, estimate in band
            if estimate.a_in_b is not None
        ]
        error = (
            f"{np.median([abs(e - v) for v, e in found]):15.3f}"
            if found
            else f"{'-':>15}"
        )
        print(
            f"{f'{low:.1f}–{min(high, 1):.1f}':>11} {len(band):6d} {100 * len(found) / max(len(band), 1):6.1f}% {error}"
        )
    false_alarms = [
        (a, b, e) for a, b, value, e in results if value == 0 and e.a_in_b is not None
    ]
    print(
        f"false alarms on non-overlapping pairs: {len(false_alarms)} / {NEGATIVE_SAMPLES}"
    )
    for a, b, estimate in false_alarms[:10]:
        print(
            f"  {a}->{b}: estimate {estimate.a_in_b:.2f}, {estimate.inlier_count} inliers"
        )


def plot_revisits(
    results: list[tuple[int, int, float, PairOverlapEstimate]], footprints: np.ndarray
) -> Figure:
    centres = camera_centres(load_camera_poses(POSES_PATH))
    figure = Figure(figsize=(11, 10))
    ax = figure.subplots()
    ax.plot(
        *centres[FIRST_PASS.start : FIRST_PASS.stop, :2].T,
        color="tab:blue",
        linewidth=1,
        label="images 0–199",
    )
    ax.plot(
        *centres[SECOND_PASS.start : SECOND_PASS.stop, :2].T,
        color="tab:orange",
        linewidth=1,
        label="images 800–999",
    )
    for a, b, value, estimate in results:
        if value >= 0.2 and estimate.a_in_b is None:
            ax.plot(
                *np.array([centres[a, :2], centres[b, :2]]).T,
                color="red",
                linewidth=0.3,
                alpha=0.3,
            )
    for a, b, value, estimate in results:
        if estimate.a_in_b is not None:
            color = "lime" if value > 0 else "magenta"
            ax.plot(
                *np.array([centres[a, :2], centres[b, :2]]).T, color=color, linewidth=1
            )
    ax.plot([], [], color="lime", label="revisit found by the images")
    ax.plot([], [], color="red", label="revisit (reference ≥ 0.2) missed")
    ax.plot([], [], color="magenta", label="false alarm")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Cross-pass links between camera centres")
    return figure


def overlap(
    features_a: Features, features_b: Features, matcher: RatioTestMatcher
) -> PairOverlapEstimate:
    matches = matcher.match(features_a, features_b)
    return estimate_homography_overlap(
        features_a, features_b, matches, SIZE, SIZE, MERMAID_INTRINSICS
    )


def reference_footprints() -> np.ndarray:
    poses = load_camera_poses(POSES_PATH)
    plane = plane_below_lowest_cameras(
        camera_centres(poses)[:, 2],
        altitude_from_gsd(MIN_GSD_M, MERMAID_INTRINSICS.focal_px),
    )
    return compute_footprints(
        poses, camera_rays(MERMAID_INTRINSICS, image_border_pixels(*SIZE, 25)), plane
    )


def required_footprint(footprints: np.ndarray, index: int) -> Polygon:
    polygon = footprint_polygon(footprints, index)
    if polygon is None:
        raise SystemExit(f"image {index} has no reference footprint")
    return polygon


def reference_overlap(footprints: np.ndarray, a: int, b: int) -> float:
    polygon_a, polygon_b = (
        footprint_polygon(footprints, a),
        footprint_polygon(footprints, b),
    )
    if polygon_a is None or polygon_b is None:
        return float("nan")
    return pair_overlap(polygon_a, polygon_b).a_in_b


def cached_features(index: int, path: Path) -> Features:
    cache = CACHE_DIR / f"{path.stem}.npz"
    if cache.exists():
        stored = np.load(cache)
        return Features(stored["keypoints"], stored["descriptors"], stored["scores"])
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # imread signals an unreadable file by returning None, not by raising.
    if image is None:
        raise OSError(f"cannot read image {path}")
    features = SiftExtractor().extract(np.asarray(image))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        keypoints=features.keypoints,
        descriptors=features.descriptors,
        scores=features.scores,
    )
    if index % 100 == 0:
        print(f"  extracted {index}", flush=True)
    return features


def images_in_capture_order() -> list[Path]:
    labels = load_camera_poses(POSES_PATH).labels
    match = match_poses_to_images(labels, sorted(IMAGES_DIRECTORY.rglob("*")))
    if match.poses_without_image:
        raise SystemExit(
            f"{len(match.poses_without_image)} poses have no image: run scripts/fetch_mermaid.py --images"
        )
    return [match.matched[label] for label in labels]


if __name__ == "__main__":
    main()
