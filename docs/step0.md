# CLAUDE.md — Dataset Automation, Person A: overlap engine

## Context

Project: platform that turns large underwater image datasets into structured data.
I am Person A (overlap engine). The goal, set with the supervisors on 2026-11-17: from a folder
of images alone, propose **the smallest set of images that keeps the same 3D reconstruction
quality**. Removing 2D duplicates is the easy part; a view of the same place from another
angle is information, not redundancy.

Two parts:

- **Overlap and selection**: overlap between images, camera displacement, images to keep for a
  chosen overlap, with the curve of images removed against that overlap.
- **Loop closures**: links between survey lines and between passes, with the angle between the
  views.

The method must work on any dataset: no pose and no position as input. The Mermaid reference
poses are used **to evaluate only**. Targets and a 3D viewer (Open3D) come later.

Where the work stands: [`overlap_status.md`](overlap_status.md). Background specs (for
context only): @docs/01_overlap_engine.md

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

- Python: numpy, opencv-python (features, geometry), shapely (polygons), matplotlib (plots), SQLite (shared database), pytest.
- Rust: pyo3, numpy crate, rayon. Built with maturin, mixed Python/Rust layout.

## Rules

- **Python reference first, Rust port second**, with a test proving both give the same results (within tolerance).
- **Unknown rather than invented**: an estimate that cannot be supported is stored as unknown, with its reason.
- No computation of several minutes without my go; never two heavy runs at once.
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

Done, in order: project skeleton, Mermaid loader, reference overlap from the poses, feature
layer (SIFT, matching), pairwise overlap, sequential overlap into the shared database,
deduplication, selection with its curve, camera displacement, a pose-based judge of a
selection, loop closure candidates.

Next (the project is down to one person since 2026-11-27, and to this plan):

1. A real 3D reconstruction (COLMAP or Metashape) of the full set and of reduced sets, to
   check the judge of 3D quality.
2. Measure the sequential overlaps again on Mermaid after the last fix of the inlier spread
   filter.
3. A calibration file format, then the same measurements on a second dataset.

Set aside: targets, masks (`archive/cleaning/`), the Open3D viewer, the Rust port of the slow
parts (the Rust extension is still the "hello world" of the skeleton).

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
uv run python scripts/overlap/find_loops.py data/mermaid/images   # loop closures into outputs/shared.sqlite
```
