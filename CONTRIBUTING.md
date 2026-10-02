# Contributing

Three people work in parallel (see `docs/00_project.md`, section 7). These rules keep branches
independent and `main` always green.

## Ownership

Each block lives in its own package, so parallel branches touch different files.

| Package | Owner | Content |
|---|---|---|
| `features/` | **shared** | `Features`, `Matches`, `FeatureExtractor`, `FeatureMatcher` |
| `storage/` | **shared** | SQLite schema, `open_database`, `start_run` |
| `camera/` | **shared** | camera model (Metashape convention), undistortion, image outline |
| `reference/` | A | reference overlap from camera poses (Mermaid) |
| `overlap/` | A | feature-based pairwise overlap; dedup and extraction to come |
| `cleaning/`, `tracking/` | B | masks, tracks (to create) |
| `ingestion/`, `targets/`, `export/` | C | readers, targets and GSD, iFDO (to create) |

Create your package (and `tests/<package>/`) in your own branch when you start. Detectors
added by B (ORB, SuperPoint) go in `features/<detector>.py`: one file each, behind the shared
interface.

**Shared packages are contracts.** A change to `features/`, `storage/` or `camera/` needs an approval from
the two other people. Changing an existing table bumps `SCHEMA_VERSION` in `storage/schema.py`.

## Branches and pull requests

- Never commit to `main` directly. Branch from an up-to-date `main`:
  `a/<topic>`, `b/<topic>` or `c/<topic>` (e.g. `a/sequential-overlap`, `c/rosbag-reader`).
- Keep branches short-lived: small pull requests, merged within a few days. Rebase on `main`
  (`git pull --rebase origin main`) rather than merging `main` into your branch.
- Every pull request is **reviewed by someone else** (`docs/00_project.md`, section 9), and CI
  must be green. The reviewer should be able to explain the change at the defense.
- Commit messages: `<type>(<scope>): <summary>`, e.g. `feat(reference): footprint polygons`.
  Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

## Dependencies

- Add them with `uv add <package>` (`uv add --dev <package>` for tooling): never edit versions by hand.
  Announce new dependencies in the pull request description.
- `uv.lock` conflicts are expected when two branches add dependencies. Resolve by taking `main`'s
  version, then re-running your `uv add` (or `uv lock`), never by hand-merging the file.
- GPU is not available: flag any GPU-only dependency before adding it.

## Data

- Datasets live in `data/` and results in `outputs/`, both gitignored. **Never commit data**:
  the repository is public and Mermaid is CC BY-NC-ND.
- Fetch Mermaid with `uv run python scripts/fetch_mermaid.py` (see README).
- Tests needing a dataset skip when it is absent (`pytest.mark.skipif`): CI has no data.
- The original images are never modified. Masks, keypoints, etc. are written next to the
  database and referenced by path.

## Code

- One module = pure functions: explicit inputs and outputs, no global state.
- Every processing step records a run: `run_id = start_run(db, "<module>", parameters)` and
  stores `run_id` on every row it writes.
- Unknown is not zero: store `NULL` (e.g. overlap with too few inliers), never an invented value.
- Tests go in `tests/<package>/test_<module>.py`, named after the behaviour
  (`test_rejects_pair_with_unknown_image`, not `test_insert`).
- Code, comments, docstrings and commit messages in English. Comment *why*, not *what*.
- Rust: one file per module in `src/`, keep the logic free of PyO3 types so `cargo test` covers it,
  and update `python/dataset_automation/_core.pyi` for every new `#[pyfunction]`.

Run the full check list from the README before pushing; CI runs the same commands.
