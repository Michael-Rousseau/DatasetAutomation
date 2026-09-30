"""Measure SIFT settings on Mermaid images sampled over the whole dive.

    uv run python scripts/features/check_sift_mermaid.py [--samples 40]

For each setting: keypoints per image, and matches between images 1 and 5 apart (the
sequential mode compares each image with its k = 5 neighbours). Low percentiles matter most:
they are the low-contrast sand images. Also draws the matches of one consecutive pair in
outputs/features_matches.png. This is the measurement behind SiftExtractor's defaults.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from dataset_automation.features.interface import Features, Matches
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.reference.images import match_poses_to_images
from dataset_automation.reference.poses import load_camera_poses

POSES_PATH = Path("data/mermaid/107177.xml")
IMAGES_DIRECTORY = Path("data/mermaid/images")
OUTPUT_DIR = Path("outputs")
GAPS = (1, 5)
SETTINGS = {
    "OpenCV defaults at half resolution": SiftExtractor(
        max_keypoints=0, contrast_threshold=0.04
    ),
    "SiftExtractor defaults": SiftExtractor(),
    "defaults without keypoint cap": SiftExtractor(max_keypoints=0),
}
MATCHES_DRAWN = 300


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


def percentiles(values: list[int]) -> str:
    low, median = np.percentile(values, [5, 50]).astype(int)
    return f"p5 {low:5d} / median {median:5d}"


def measure(
    name: str,
    extractor: SiftExtractor,
    images: dict[int, np.ndarray],
    starts: list[int],
) -> None:
    matcher = RatioTestMatcher()
    start = time.perf_counter()
    features = {index: extractor.extract(image) for index, image in images.items()}
    extract_s = (time.perf_counter() - start) / len(images)

    print(f"{name}: {extractor}")
    print(
        f"  keypoints          {percentiles([len(features[i]) for i in starts])}   {extract_s:.2f} s/image"
    )
    for gap in GAPS:
        start = time.perf_counter()
        counts = [len(matcher.match(features[i], features[i + gap])) for i in starts]
        match_s = (time.perf_counter() - start) / len(starts)
        print(f"  matches, gap {gap}     {percentiles(counts)}   {match_s:.2f} s/pair")


def plot_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    features_a: Features,
    features_b: Features,
    matches: Matches,
) -> Figure:
    shown = np.random.default_rng(0).permutation(len(matches))[:MATCHES_DRAWN]
    points_a = features_a.keypoints[matches.index_pairs[shown, 0]]
    points_b = features_b.keypoints[matches.index_pairs[shown, 1]] + [
        image_a.shape[1],
        0,
    ]

    figure = Figure(figsize=(16, 6))
    ax = figure.subplots()
    ax.imshow(np.hstack([image_a, image_b]), cmap="gray")
    for (xa, ya), (xb, yb) in zip(points_a, points_b):
        ax.plot([xa, xb], [ya, yb], linewidth=0.5)
    ax.set_axis_off()
    ax.set_title(
        f"{len(shown)} of {len(matches)} SIFT matches between consecutive images"
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--samples", type=int, default=40, help="images sampled evenly over the dive"
    )
    arguments = parser.parse_args()

    paths = images_in_capture_order()
    starts = (
        np.linspace(0, len(paths) - 1 - max(GAPS), arguments.samples)
        .astype(int)
        .tolist()
    )
    needed = sorted({i + gap for i in starts for gap in (0, *GAPS)})
    images = {i: read_grayscale(paths[i]) for i in needed}
    print(f"{len(starts)} images sampled over {len(paths)}, {len(images)} loaded\n")

    for name, extractor in SETTINGS.items():
        measure(name, extractor, images, starts)

    extractor, first = SiftExtractor(), starts[0]
    features_a, features_b = (
        extractor.extract(images[first]),
        extractor.extract(images[first + 1]),
    )
    matches = RatioTestMatcher().match(features_a, features_b)
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_matches(
        images[first], images[first + 1], features_a, features_b, matches
    ).savefig(OUTPUT_DIR / "features_matches.png", dpi=120)
    print(f"\nfigure written to {OUTPUT_DIR}/features_matches.png")


if __name__ == "__main__":
    main()
