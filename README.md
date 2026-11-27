# DatasetAutomation

From a folder of underwater images, with no pose and no position: how much the images overlap,
which ones can be removed, and which far-apart images see the same place (loop closures
between survey lines and passes). The aim is the smallest set of images that keeps the same 3D
reconstruction quality. Python first; measured hotspots move to Rust (PyO3).

Where the work stands, with the measured results: [`docs/overlap_status.md`](docs/overlap_status.md).
Decisions and measurements in order: [`docs/overlap_journal.md`](docs/overlap_journal.md).

The project started wider, with three people ([`docs/00_project.md`](docs/00_project.md),
[`CONTRIBUTING.md`](CONTRIBUTING.md)); it is now one person's work on the scope above. Code
outside that scope is in [`archive/`](archive/README.md).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Rust toolchain (`cargo`), needed to build the native extension

## Setup

```sh
uv sync                                    # .venv, Python deps, Rust extension (maturin)
uv run python scripts/fetch_mermaid.py     # Mermaid poses and reports into data/mermaid/ (~5 MB)
uv run python scripts/fetch_mermaid.py --images   # + the images (3.69 GB)
```

## Usage

```sh
uv run python scripts/overlap/run_sequential.py <folder> --register   # overlap between neighbours, ~10 min for 1,244 images
uv run python scripts/overlap/find_loops.py <folder>                  # loop closures, ~20 min for 1,244 images
uv run python scripts/overlap/select_images.py --overlap 0.80         # images to keep + curve, 1 s
uv run python scripts/overlap/select_images.py --min-views 5          # the other rule, reads the loop closures
uv run python scripts/overlap/find_duplicates.py --threshold 0.95     # near-copies, 1 s
```

The first two commands match images and are the long ones; the others read their results from
`outputs/shared.sqlite`. Nothing is ever deleted: a selection is a list of images to keep.
Each script documents its options and outputs at the top of its file.

## Checks (the same ones CI runs on every pull request)

```sh
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
cargo fmt --check && cargo clippy --all-targets -- -D warnings
cargo test
```

## Layout

```
python/dataset_automation/
  features/        SIFT and Harris, matching, VLAD image vectors, features on disk
  camera/          camera model (Metashape convention), undistortion
  overlap/         pairwise overlap, camera displacement, sequential mode, selection, loop closures
  reference/       Mermaid reference poses and the judge of a selection (evaluation only)
  storage/         SQLite schema, run traceability, image registration
  _core.pyi        type stub of the Rust extension
src/               Rust extension, one file per module, imported as dataset_automation._core
tests/<package>/   tests, mirroring the package layout
scripts/           runnable entry points; scripts/<package>/ for package-specific ones
docs/              status, journal, and the original project sheets
archive/           code set aside, not installed, linted or tested (see archive/README.md)
data/, outputs/    local only, gitignored
```

## Adding Rust code

1. Create `src/<module>.rs` with your `#[pyfunction]`s and a `pub fn register(module)` that adds them.
2. In `src/lib.rs`, add `mod <module>;` and `<module>::register(module)?;`.
3. Declare the new functions in `python/dataset_automation/_core.pyi`, so pyright and editors know them.
4. Run `uv sync` (it rebuilds when `Cargo.toml` or `src/**/*.rs` change), then
   `from dataset_automation import _core`.

To iterate faster without a full sync, run `uv run maturin develop --uv`. Add `--release` for optimized builds.
