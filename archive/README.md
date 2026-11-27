# Archive

Code set aside: not installed with the package, not linted, type-checked or tested by CI.
It is kept in the tree rather than only in the git history so that it stays easy to find.

## `cleaning/` — masks of unwanted elements (set aside on 2026-11-27)

Masks of divers, fish and the water surface, in COLMAP's convention (0 = ignored,
`<image name>.png`), by three methods: a rule (water surface only), zero-shot (Grounding DINO
+ SAM2) and a fine-tuned SegFormer. Specification: `docs/02_cleaning_tracking.md`, workstream 1.
Written by the person in charge of block VI, who left the project; it came in with pull
request #2.

Why it is set aside: nothing used it. No mask was ever produced in the shared database, and
the only link with the rest of the code was a function listing registered images, now in
`storage/images.py`. The project's scope was reduced to the overlap engine, loop closures and
3D reconstruction.

When it could be worth bringing back: a diver crossing the field of view is what makes
neighbouring images impossible to relate on `data_test_roman/data/set_003`, and it degrades a
reconstruction. The feature extractors already accept a mask.

To bring it back:

```sh
git mv archive/cleaning/package python/dataset_automation/cleaning
git mv archive/cleaning/scripts scripts/cleaning
git mv archive/cleaning/tests tests/cleaning
```

then restore the optional dependencies in `pyproject.toml` (the model-based methods need
them; the rule does not) and run `uv lock`:

```toml
[project.optional-dependencies]
cleaning = [
    "torch>=2.14.0",
    "torchvision>=0.29.0",
    "transformers>=5.17.0",
]
```

The `masks` table of the shared schema is unchanged.
