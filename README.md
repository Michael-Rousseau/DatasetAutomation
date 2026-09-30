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
```

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
