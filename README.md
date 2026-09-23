# DatasetAutomation

Image dataset exploration in Python with OpenCV. Hot paths can be moved to Rust and exposed to Python through PyO3.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Rust toolchain (`cargo`), needed to build the native extension

## Setup

```sh
uv sync
```

This creates `.venv`, installs the Python dependencies and builds the Rust extension with maturin.

## Usage

```sh
uv run python main.py data/image.png
```

Put local images in `data/`. Everything in it is ignored by git.

## Layout

```
main.py                         exploration entry point
python/dataset_automation/      Python package
src/lib.rs                      Rust extension, imported as dataset_automation._core
pyproject.toml                  Python project and maturin config
Cargo.toml                      Rust crate
```

## Adding Rust code

1. Add a `#[pyfunction]` in `src/lib.rs` and register it in the `_core` module.
2. Run `uv sync`. It rebuilds the extension whenever `Cargo.toml` or `src/**/*.rs` changes.
3. Call it from Python:

   ```python
   from dataset_automation import _core
   ```

To iterate faster without a full sync, run `uv run maturin develop --uv`. Add `--release` for optimized builds.
