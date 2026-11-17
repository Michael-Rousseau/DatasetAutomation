# CLAUDE.md — Dataset Automation, Person A: reference overlap

## Context

Project: platform that turns large underwater image datasets into structured data.
I am Person A (overlap engine). **Current scope is narrow on purpose**: compute the
**reference overlap between images from known camera poses**, on the Mermaid dataset only,
first in Python, then ported to Rust with PyO3.

This reference overlap will later serve as ground truth to evaluate feature-based overlap.
Nothing else is in scope for now (no feature matching, no dedup, no extraction, no other datasets).

Background specs (for context only): @docs/01_overlap_engine.md

## Working mode — IMPORTANT

I want to understand everything we build. My Rust level: intermediate. My Python level: good.

- Work **one small step at a time**. Never start the next step without my explicit "go".
- **Before coding**: explain what you are about to do and why, in a few lines. If there are several options, list them with trade-offs and let me choose.
- **After coding**: summarize what changed, which files, and how to run/test it.
- Explain every new concept the first time it appears: geometry (ray casting, homogeneous coordinates, distortion), Rust (ownership, borrowing), PyO3 (`#[pyfunction]`, `#[pymodule]`, numpy arrays across the boundary), rayon.
- Prefer small, readable code over clever code. No big code dumps.
- Sometimes ask me to write a small part myself, then review it.
- Talk to me in French. Code, comments, docstrings and commit messages in English.

## Tech stack

- Python: numpy, opencv-python (undistortion), shapely (polygons), matplotlib (plots), pytest.
- Rust: pyo3, numpy crate, rayon. Built with maturin, mixed Python/Rust layout.

## Rules

- **Python reference first, Rust port second**, with a test proving both give the same results (within tolerance).
- **Pure functions**: clear inputs and outputs, no global state.
- Data lives in `data/mermaid/`, is **gitignored**, and is never modified.
- Don't install anything globally or download large files without asking me first.
- Run tests after each change. One git commit per completed step.

## Mermaid dataset — `data/mermaid/`

- 1,244 JPG images, 3840 × 2880 px, ~18.7 m depth, **mainly nadir views**, ~150 m² area.
- Relief: a statue, a sandy plain with stones, a rocky area. A flat-seafloor assumption is wrong near the statue: keep that in mind when interpreting results.
- Camera: GoPro Hero 3 Silver. Calibration:
  F = 2334.29, Cx = -12.752, Cy = -16.6962,
  K1 = -0.222446, K2 = 0.310621, K3 = -0.0835057, P1 = -0.000995472, P2 = -7.8498e-05.
- Probably Agisoft Metashape convention (F in pixels, Cx/Cy offsets from the image centre).
  **Verify** it, including how Metashape distortion coefficients map to OpenCV (check the P1/P2 order).
- Poses XML: per camera, a 4×4 `transform`, `rotation_covariance`, `location_covariance`, `orientation`.
  The transform direction (camera→world or world→camera) must be **verified** with a sanity check.
- Licence CC BY-NC-ND.

## Roadmap (one step at a time)

0. **Skeleton**: maturin + PyO3 project, a "hello world" Rust function called from Python, pytest and cargo test running.

1. **Loader (Python)**: parse the poses XML and the intrinsics, match poses to image files.
   Sanity check: plot camera positions (top view) and viewing directions. Nadir cameras should all look roughly the same way, towards the seafloor.

2. **Reference overlap from poses (Python)** — the core of this scope. Sub-steps:
   - 2a. **Seafloor plane**: discuss the options with me (e.g. plane fitted under the camera centres with an estimated altitude, horizontal plane at a fixed depth…) and their limits. I choose.
   - 2b. **Image footprint**: sample points along the image border (not just the 4 corners: distortion curves the edges), undistort them, cast rays from the camera centre, intersect with the plane → footprint polygon.
   - 2c. **Overlap ratio**: define it clearly. `area(A ∩ B) / area(A)` is asymmetric; also consider IoU. Explain the difference and let me choose what to store.
   - 2d. **Candidate pairs**: 1,244 images ≈ 773,000 pairs. Avoid brute force: pre-filter with a spatial index (e.g. shapely STRtree) so only footprints that can intersect are compared.
   - 2e. **Sanity checks**: consecutive images should overlap strongly, distant ones not at all. Plot footprints for a few pairs over each other. Histogram of overlap values.
   - Output: a table (image_a, image_b, overlap, iou) saved to a file (Parquet or CSV) in `outputs/`.

3. **Rust port of step 2 (PyO3 + rayon)**: port the footprint intersection over candidate pairs.
   Equality test against the Python version, then a benchmark (Python vs Rust, single-thread vs rayon).
   Explain what crosses the Python/Rust boundary and why it matters for performance.

**Stop after step 3.**

## Decisions and findings

- **Step 1 — pose convention**: `transform` is **camera→world** (camera centre = translation column).
  Evidence (`scripts/reference/check_mermaid_poses.py`): max step between consecutive images is 1.66 m vs 19.16 m
  for the inverse hypothesis. Camera frame: x right, y down, z forward; all optical axes have z in [−1, −0.8].
- **Pose frame is local**: centres span ~11 × 9.4 m, z from −4 to +2 m. Camera 511 has no covariances.
- **STL (107176)** = the 5 GCP markers of the micro geodesic network (report 107175), as vertical prisms
  of height 19.701 − depth. They are **not** in the pose frame. Where the report is inconsistent, the STL
  confirms the table: y(GCP 33) = 3.891, y(GCP 26) = −0.084.
- **Seafloor relief**: GCP depths range 17.0–19.7 m, so the seafloor varies by ≥ 2.7 m. At the same xy,
  camera heights differ by 2.6 m (median over 1 m cells): the altitude is not constant.
- **Step 2a — seafloor model**: horizontal plane z = z_floor (option A), behind an interface that a mesh
  can replace later. z_floor is anchored on the smallest announced GSD (0.5 mm ⇒ altitude ≈ 0.5 mm × F
  ≈ 1.17 m for the lowest cameras), i.e. ≈ −4.9 m; kept as a parameter.
  **Known limit**: with ≥ 2.7 m of relief and 1–3 m typical altitude, footprint sizes can be off by tens
  of percent. This is an approximation, not ground truth; sensitivity to z_floor is measured in 2e.
  A true reference needs the scene surface (unpublished Metashape mesh — ask the authors — or a
  triangulation with the known poses once the images are downloaded).
- **Step 2b — distortion convention** (Metashape Pro 2.1 manual, appendix D): OpenCV p1 = Metashape P2,
  OpenCV p2 = Metashape P1; OpenCV principal point = w/2 + Cx − 0.5 (Metashape puts the first pixel
  centre at (0.5, 0.5)). Default `cv2.undistortPoints` leaves ~0.1 px error at the corners: use explicit
  criteria. Union of all footprints is 390–510 m² vs ~150 m² announced: either the announced area is the
  cropped mosaic, or the poses are not exactly metric. To check once images exist: triangulate the GCP
  markers and compare with the measured slope distances.
- **Step 2c — overlap definition**: one row per unordered pair (a < b) with
  `overlap_a_in_b = |A∩B|/|A|`, `overlap_b_in_a = |A∩B|/|B|` and `iou = |A∩B|/|A∪B|`.
  Directional values match the feature-based (homography) estimator and deduplication; IoU is kept for
  symmetric thresholds. Non-intersecting pairs are not stored. Example: pair 24→268 gives 1.00 / 0.10 /
  0.10 (24 is fully inside 268 but has a 3× finer GSD, so footprint overlap alone must not drive dedup).

## Commands

```sh
uv sync                                   # install deps, rebuild the Rust extension
uv run pytest                             # Python tests
cargo test                                # Rust unit tests
uv run ruff check . && uv run ruff format --check .
uv run maturin develop --uv               # fast Rust rebuild without a full sync
uv run python scripts/fetch_mermaid.py              # Mermaid poses and reports into data/mermaid/
uv run pyright                                      # type check
uv run python scripts/reference/check_mermaid_poses.py   # step 1: pose convention check + plots in outputs/
uv run python scripts/reference/check_footprints.py      # step 2b: footprint stats + plot in outputs/
```

## Scope extension — feature layer (branch `a/sift-features`)

The reference-overlap roadmap above is paused after step 2c: the feature layer comes first because
persons B and C depend on it (`01_overlap_engine.md`, step 1). Steps 2d/2e resume when the
feature-based overlap needs to be evaluated.

- **SIFT localization bias**: OpenCV's default SIFT upsamples the first octave in a way that shifts
  every keypoint by ~+0.25 px (measured on Gaussian blobs with known centres). `SiftExtractor` uses
  `enable_precise_upscale=True`, which brings the error to ~0.00 px.
- **Downscaled detection**: keypoints are mapped back with `(x + 0.5) / f − 0.5` (pixel centres on
  integers); a test checks there is no residual bias between scale 1 and scale 0.5.
- **`nfeatures` is approximate**: OpenCV keeps ties at the cut-off, so `SiftExtractor` enforces
  `max_keypoints` itself by keeping the strongest responses.
- **SEANOE downloads**: a single large request stalls, and each connection is throttled to
  ~0.3 MB/s. `scripts/fetch_mermaid.py` fetches 4 MB byte ranges over 8 connections (~1.6 MB/s)
  and resumes interrupted downloads.
- **SIFT defaults** (`scripts/features/check_features_mermaid.py`, 40 images sampled over the dive,
  gaps of 1 and 5 images): OpenCV's contrast threshold 0.04 leaves low-contrast sand with 16 keypoints
  (p5) and 1 match at gap 5; full resolution barely helps (4 matches). A threshold of 0.01 gives 56
  matches at gap 5 (p5). Capping at the 8000 strongest keypoints keeps ~all gap-1 matches (865 vs 888
  at p5) for 0.08 s/pair instead of 0.54 s. Defaults: `scale=0.5`, `contrast_threshold=0.01`,
  `max_keypoints=8000`; sequential mode with k = 5 on Mermaid ≈ 8 min of matching.
- **Images**: `match_poses_to_images` pairs all 1,244 poses with their JPG (0 missing, 0 extra).

## Harris baseline (branch `a/harris-features`)

- `HarrisSiftExtractor`: Harris corners (`goodFeaturesToTrack`) described by upright SIFT
  descriptors, sharing SIFT's preprocessing (`features/preprocessing.py`) and matcher.
- **Sub-pixel refinement**: `cv2.cornerSubPix` models a corner as two straight edges. On
  seafloor-like texture it left half the corners unmoved and threw 10 % more than 4.5 px away
  (median error 0.67 px on a sub-pixel shift, no better than integer corners, 40 % fewer matches).
  A per-axis parabola fitted to the Harris response gives 0.23–0.28 px and never moves a corner by
  more than half a pixel.
- **Result** (`scripts/features/check_features_mermaid.py`, homography inliers, MAGSAC 3 px):

  | Extractor | inliers gap 1 (p5 / median) | inliers gap 5 (p5 / median) |
  |---|---|---|
  | SIFT defaults | 128 / 457 | 7 / 64 |
  | Harris + upright SIFT (best setting) | 11 / 221 | 0 / 4 |

  Without orientation or scale, Harris descriptors break between images 5 apart (heading and
  altitude change). SIFT stays the default; Harris is kept as a comparison baseline.
- **For pairwise overlap**: even with SIFT only ~28 % of consecutive matches fit one homography
  (457 inliers of 1,604 matches): relief breaks the planar-scene assumption, as expected in
  `01_overlap_engine.md`.

## Pairwise overlap (branch `a/pairwise-overlap`)

- **`camera/`** (new shared package): the Metashape camera model, undistortion and image outline
  moved out of `reference/`; `MERMAID_INTRINSICS` stays in `reference/`. Mermaid's lens is barrel
  in the field (edge midpoints move ~65 px outwards when undistorted) but turns back near the
  corners (k2 > 0: corners move ~24 px inwards).
- **`overlap/pairwise.py`**: `estimate_homography_overlap` (MAGSAC, undistorted pixels) and
  `estimate_hull_overlap` (fundamental-matrix inliers, so no planar assumption). Each share is
  measured in its own image's frame; unknown overlap is `None` with a reason, never a number.
- **Validation of the homography**: no vertex beyond the horizon, no mirroring, area ratio within
  ×8 (largest footprint area ratio between overlapping Mermaid images: 5.2), ≥ 15 inliers.
  MAGSAC threshold 8 px: relief creates parallax a single homography cannot explain; 3 → 8 px cut
  unknowns at 20–40 % overlap from 80 % to 73 % with the same error, 15 px accepted false overlaps.
- **Repeated targets trap**: GCP plates 28 and 30 share the same pattern; matched to each other
  they gave a consistent homography and a 79 % overlap for two images that do not overlap. All
  inliers sat on the plate: hull / common area 0.005, vs ≥ 0.026 for the 102 correct estimates.
  Pairs below `MIN_INLIER_SPREAD = 0.01` are unknown. Only one such case observed so far.
- **Result** (`scripts/overlap/evaluate_pairwise_mermaid.py`, 150 pairs, |error| median):

  | Reference overlap | homography raw | homography undistorted | inlier hull | unknown (undistorted) |
  |---|---|---|---|---|
  | 0.8–1.0 | 0.023 | 0.031 | 0.174 (bias −0.17) | 0 % |
  | 0.6–0.8 | 0.038 | 0.044 | 0.212 (bias −0.21) | 11 % |
  | 0.4–0.6 | 0.059 | 0.060 | 0.187 | 30 % |
  | 0.2–0.4 | 0.115 | 0.076 | 0.086 | 73 % |
  | 0.0–0.2 | – | 0.058 (1 pair) | 0.035 | 95 % |

  (raw column measured at 3 px, the others at 8 px.)

  Undistortion gives +23 % inliers (median) and more inliers in 85 % of pairs, so the model fits
  the images better; it does not improve agreement with the reference, whose own error (horizontal
  plane vs ≥ 2.7 m of relief) dominates at this level. The inlier hull underestimates by ~20 points
  (points cover 15–30 % of the common area). Five pairs keep an error > 0.15 with a normal inlier
  spread (e.g. 168→170: reference 0.86, estimate 0.52): not explained yet, candidates are relief and
  the reference plane.
- **Honest limit**: reliable only for nearly consecutive images. Enough for deduplication (≥ 95 %),
  borderline for the ~80 % extraction, not enough below ~40 % overlap (60–73 % unknown).
- **Keypoint cap trade-off** (measured, not applied): without the 8000 cap, unknowns at 40–60 %
  overlap drop from 30 % to 17 % (60–80 %: 11 % → 5 %), but brute-force matching is 7× slower
  (0.54 vs 0.08 s/pair, ~1 h vs 8 min for k = 5 on Mermaid). To revisit with a faster matcher
  (FLANN) in its own branch, since it changes the SIFT default used by B. Other levers to measure:
  masking GCP plates (they dominate matches), better descriptors, a 3D model instead of a homography.

## Notes for the n → n mode (loop closures)

- The robot / diver path must be encoded, both ways:
  - **sequence graph**: capture order as edges between consecutive images, the structure the
    sequential overlaps live on;
  - **positions** when available: reference poses (Mermaid, Eiffel Tower), navigation from the ROS
    bags (AQUALOC IMU and pressure, C's ingestion), or a trajectory chained from sequential overlaps
    (visual odometry).
- Candidate loop pairs are then images close in space but far apart in time.
- **Mermaid has real revisits**: per the trajectory in report 107174, passes 0–207 and 832–1039
  fly over the same areas (`scripts/overlap/explore_mermaid_dive.py revisits` tests them).
- **The encoded path drifts** (navigation or chained odometry accumulates error): use it to
  pre-select candidates, never to decide that two images overlap. The images decide.
- **Read the target IDs** (for later, with C): each GCP plate carries a unique number (26, 28, 30
  read in the images). The same ID in two images links them regardless of drift, and different IDs
  defuse the identical-plates trap found in pairwise overlap.
- The shared schema has no trajectory table yet (`images` only has `sequence` and `timestamp_s`):
  adding one is a change to the `storage/` contract, to agree with B and C.

## Overlap and loop closures on the whole dive (exploration, 2026-10-01)

`scripts/overlap/explore_mermaid_dive.py` (SIFT features cached in `outputs/cache/`):

- **Sequential** (1,244 images, each vs its 5 next, 6,205 pairs): gap 1 → 0 % unknown, |error|
  median 0.025; gap 5 → 12.6 % unknown, 0.054. Unknowns concentrate on images ~50–500, where
  overlap with +5 is only 0.3–0.6; texture is not the cause (8,000 keypoints almost everywhere).
- **Revisits** (passes 0–200 vs 800–1000): 68 % of the 40,000 cross-pass pairs overlap. Sampled
  400 per band: found 14.8 % of revisits at ≥ 0.6 overlap (|error| 0.005), 4.8 % at 0.4–0.6, 1.5 %
  at 0.2–0.4; **0 false alarms** on 400 non-overlapping pairs. Precise, but poor recall.
- **Why**: between passes the diver flies the other way (heading difference > 90° for 63 % of the
  pairs, recall 11 % vs 33 % under 20°) and much higher (altitude ratio > 2 for 91 %, recall 13 %
  vs 35 % between ×1.3 and ×2; ratios relative to the approximate reference plane). Only ~30
  matches survive per pair with the sequential settings.
- **Costlier verification does not help** (`scripts/overlap/compare_verification_settings.py`,
  200 revisits ≥ 0.6, 200 non-overlapping pairs): current settings 14.0 % found; no keypoint cap
  14.5 %; no cap + ratio 0.80 15.5 %; full resolution (cap 20,000, ratio 0.80) 11.0 %. Always 0 false
  alarms, 6–8× slower. The bottleneck is not the matcher's settings.
- **What the revisits really are**: between the two passes no pair has a similar footprint
  (reference IoU ≥ 0.5: 0 %, ≥ 0.3: 2.5 %). Almost every "revisit" is a close-up from the first
  pass (~1–2 m above the seafloor) contained in a wide view from the second, ~3 m higher, seen
  through more turbid water (pale, blurred, GCP plates a few pixels wide). Recall by reference IoU:
  0.30–0.50 → 36 %, 0.15–0.30 → 21 %, < 0.15 → 3 %. Finding a close-up inside a hazy wide view is
  the hardest case for local features; the 14 % measured mostly that.
- **Next**: define what a useful loop closure is (similar footprints, e.g. IoU or min(a_in_b,
  b_in_a), rather than a_in_b alone); measure contrast enhancement (e.g. CLAHE) against turbidity
  on the IoU 0.15–0.5 band; then choose the image-based candidate pre-selection (VLAD on SIFT,
  CPU only) and test it on other datasets with real revisits at similar altitude (AQUALOC,
  UnderLoc / SQUIDLE+ VPR).
