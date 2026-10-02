"""Which verification settings recover Mermaid revisits (n → n), and at what false-alarm cost.

    uv run python scripts/overlap/compare_verification_settings.py   # ~20 min

200 cross-pass revisits (passes 0–200 vs 800–1000, reference overlap ≥ 0.6) and 200 non-overlapping
pairs among the same images. Loop-closure candidates are few, so verification can afford settings
too costly for the sequential mode (full resolution, no keypoint cap, looser ratio test).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import explore_mermaid_dive as ex

from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor

fp = ex.reference_footprints()
paths = ex.images_in_capture_order()
P = {i: ex.required_footprint(fp, i) for i in (*ex.FIRST_PASS, *ex.SECOND_PASS)}
rng = np.random.default_rng(1)
candidates = [
    (a, b) for a in ex.FIRST_PASS for b in ex.SECOND_PASS if P[a].intersects(P[b])
]
positives = []
for i in rng.permutation(len(candidates)):
    a, b = candidates[i]
    if ex.reference_overlap(fp, a, b) >= 0.6:
        positives.append((a, b))
    if len(positives) == 200:
        break
used_a, used_b = sorted({a for a, _ in positives}), sorted({b for _, b in positives})
negatives = []
while len(negatives) < 200:
    a, b = int(rng.choice(used_a)), int(rng.choice(used_b))
    if not P[a].intersects(P[b]) and (a, b) not in negatives:
        negatives.append((a, b))
images = sorted({i for pair in positives + negatives for i in pair})
print(
    f"{len(positives)} revisits (reference >= 0.6), {len(negatives)} non-overlapping pairs, {len(images)} images",
    flush=True,
)


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # imread signals an unreadable file by returning None, not by raising.
    if image is None:
        raise OSError(f"cannot read image {path}")
    return np.asarray(image)


gray = {i: read_grayscale(paths[i]) for i in images}

settings = [
    ("half res, cap 8000, ratio 0.75 (current)", SiftExtractor(), 0.75),
    ("half res, no cap,   ratio 0.75", SiftExtractor(max_keypoints=0), 0.75),
    ("half res, no cap,   ratio 0.80", SiftExtractor(max_keypoints=0), 0.80),
    (
        "full res, cap 20000, ratio 0.80",
        SiftExtractor(scale=1.0, max_keypoints=20000),
        0.80,
    ),
]
cache = {}
for name, extractor, ratio in settings:
    key = (extractor.scale, extractor.max_keypoints)
    if key not in cache:
        t = time.perf_counter()
        cache[key] = {i: extractor.extract(g) for i, g in gray.items()}
        print(f"  extracted {key} in {time.perf_counter() - t:.0f} s", flush=True)
    feats, matcher = cache[key], RatioTestMatcher(ratio=ratio)
    t = time.perf_counter()
    est_pos = [
        ex.estimate_homography_overlap(
            feats[a],
            feats[b],
            matcher.match(feats[a], feats[b]),
            ex.SIZE,
            ex.SIZE,
            ex.MERMAID_INTRINSICS,
        )
        for a, b in positives
    ]
    est_neg = [
        ex.estimate_homography_overlap(
            feats[a],
            feats[b],
            matcher.match(feats[a], feats[b]),
            ex.SIZE,
            ex.SIZE,
            ex.MERMAID_INTRINSICS,
        )
        for a, b in negatives
    ]
    per_pair = (time.perf_counter() - t) / 400
    found = [e for e in est_pos if e.is_known]
    errors = [
        abs(e.a_in_b - ex.reference_overlap(fp, a, b))
        for (a, b), e in zip(positives, est_pos)
        if e.a_in_b is not None
    ]
    alarms = sum(e.is_known for e in est_neg)
    print(
        f"{name}: found {100 * len(found) / 200:5.1f} %, |error| median {np.median(errors) if errors else float('nan'):.3f}, false alarms {alarms}/200, {per_pair:.2f} s/pair",
        flush=True,
    )
