# CLAUDE.md — Dataset Automation, Person A: reference overlap

> The scope described below is where this work started. It has widened since (feature layer,
> sequential overlap, selection, camera displacement, loop closures): see
> [`overlap_status.md`](overlap_status.md).

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

Moved to [`overlap_journal.md`](overlap_journal.md), the detailed record of what was decided
and measured. [`overlap_status.md`](overlap_status.md) says where the work stands today.

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
uv run python scripts/overlap/run_sequential.py data/mermaid/images --register   # sequential overlaps into outputs/shared.sqlite
uv run python scripts/overlap/select_images.py --overlap 0.80   # images to keep + curve, from the stored overlaps
```
