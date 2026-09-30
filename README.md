# DatasetAutomation

Automated analysis of large underwater image datasets: overlap between images, deduplication,
tracking metrics, masks, targets and GSD. Python first; measured hotspots move to Rust (PyO3).

Project scope and work split: [`docs/00_project.md`](docs/00_project.md).
How to work on the repository together: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Rust toolchain (`cargo`), needed to build the native extension

## Setup

```sh
uv sync                                    # .venv, Python deps, Rust extension (maturin)
uv run python scripts/fetch_mermaid.py     # Mermaid poses and reports into data/mermaid/ (~5 MB)
uv run python scripts/fetch_mermaid.py --images   # + the images (3.69 GB)
uv sync --extra cleaning                   # + torch/transformers for model-based masks (B)
```

The `cleaning` extra is optional: CI and the other packages do not need it. Tests that need it
skip without it; the one downloading Grounding DINO and SAM2 (~850 MB) also needs
`DATASET_AUTOMATION_MODEL_TESTS=1`.

## Masks of unwanted elements (B)

```sh
uv run python scripts/cleaning/inventory.py data/mermaid/images --every 25       # step 1
uv run python scripts/cleaning/make_masks.py data/mermaid/images --register \
    --method zero-shot                                                             # step 2
uv run python scripts/cleaning/evaluate_masks.py test.json data/mermaid/images \
    --method zero-shot                                                             # step 3
uv run python scripts/cleaning/train_segformer.py train.json data/mermaid/images  # step 4
```

`--method` is `rule` (surface only, no extra needed), `zero-shot` (Grounding DINO + SAM2) or
`segformer` (a model trained by step 4, `--model <dir>`). Annotations are COCO exports from CVAT
or Label Studio. Masks follow COLMAP's convention (0 = ignored, `<image name>.png`).

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
  features/        shared feature interface (Features, Matches, extractor/matcher protocols)
  storage/         shared SQLite schema and run traceability
  cleaning/        masks of unwanted elements: rule, zero-shot, SegFormer, IoU    — B
  reference/       reference overlap from known camera poses (Mermaid)             — A
  _core.pyi        type stub of the Rust extension
src/               Rust extension, one file per module, imported as dataset_automation._core
tests/<package>/   tests, mirroring the package layout
scripts/           runnable entry points; scripts/<package>/ for package-specific ones
docs/              project reference and per-person sheets
data/, outputs/    local only, gitignored
```

## Adding Rust code

1. Create `src/<module>.rs` with your `#[pyfunction]`s and a `pub fn register(module)` that adds them.
2. In `src/lib.rs`, add `mod <module>;` and `<module>::register(module)?;`.
3. Declare the new functions in `python/dataset_automation/_core.pyi`, so pyright and editors know them.
4. Run `uv sync` (it rebuilds when `Cargo.toml` or `src/**/*.rs` change), then
   `from dataset_automation import _core`.

To iterate faster without a full sync, run `uv run maturin develop --uv`. Add `--release` for optimized builds.
