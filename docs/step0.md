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
- **SIFT defaults** (`scripts/features/check_sift_mermaid.py`, 40 images sampled over the dive,
  gaps of 1 and 5 images): OpenCV's contrast threshold 0.04 leaves low-contrast sand with 16 keypoints
  (p5) and 1 match at gap 5; full resolution barely helps (4 matches). A threshold of 0.01 gives 56
  matches at gap 5 (p5). Capping at the 8000 strongest keypoints keeps ~all gap-1 matches (865 vs 888
  at p5) for 0.08 s/pair instead of 0.54 s. Defaults: `scale=0.5`, `contrast_threshold=0.01`,
  `max_keypoints=8000`; sequential mode with k = 5 on Mermaid ≈ 8 min of matching.
- **Images**: `match_poses_to_images` pairs all 1,244 poses with their JPG (0 missing, 0 extra).
