# Overlap engine — where it stands

Person A's part of the project, as of 2026-11-19. Short on purpose: the measurements, their
method and the dead ends are in [`overlap_journal.md`](overlap_journal.md).

## What it does

From a folder of images, with no pose and no position:

1. measures how much each image overlaps the ones that follow it, and answers "unknown" rather
   than a number it cannot support;
2. selects the images to keep for a chosen overlap, and draws the curve of images removed
   against that overlap;
3. with a camera calibration, measures how far the camera moved between two images, as an
   angle;
4. proposes, for each image, the far-apart images that look like it: candidates for loop
   closures (links between survey lines and between passes).

Nothing is ever deleted: the output is a list of images to keep. Every run records its
parameters and the version of the code.

## How to run it

```sh
uv run python scripts/overlap/run_sequential.py <folder> --register   # overlaps, ~10 min for 1,244 images
uv run python scripts/overlap/select_images.py --overlap 0.80         # images to keep + curve, 1 s
uv run python scripts/overlap/find_duplicates.py --threshold 0.95     # near-copies, 1 s
uv run python scripts/overlap/find_loops.py <folder>                  # loop closures, ~25 min for 1,244 images
```

The first and the last commands are the long ones. The two in between read the result of the
first from `outputs/shared.sqlite`, so another threshold costs nothing. `find_loops.py` writes
its links to the same `pairs` table under a run of its own, and keeps the features of each
image on disk, so a second run does not extract them again. `select_images.py` writes
`kept_images.txt`, `images.csv`, `curve.csv`, `curve.png` and `summary.txt`.

On Mermaid only, where a calibration and reference poses exist:

```sh
uv run python scripts/overlap/run_sequential.py data/mermaid/images --mermaid-calibration
uv run python scripts/overlap/select_images.py --overlap 0.80 --mermaid-poses data/mermaid/107177.xml
```

The first adds the camera displacement of each pair; the second judges the selection against
the poses.

## What is measured

On Mermaid (1,244 images of a diver's survey grid), against the reference poses.

| Question | Result |
|---|---|
| Overlap between neighbours | known for 100 %, off by 0.025 in median |
| Overlap 5 images apart | known for 87 %, off by 0.054 |
| Images removed at 0.80 of overlap | 385 of 1,244 (31 %) |
| … at 0.70, 0.60, 0.50 | 55 %, 68 %, 75 % |
| What that selection costs | reconstructible area intact inside, 28 % of a 1 m rim lost; depth noise ×1.22 |
| Camera displacement between neighbours | known for all pairs, off by 0.13° in median for 3.6° |
| Far images proposed by resemblance (10 per image) | nearly half are true revisits; 85 % of the images get one |
| Proposed pairs confirmed by matching | 95 % of the images linked to another line or pass; no false link among 982 candidates that do not overlap |
| Overlap of a confirmed far pair | off by 0.056 in median, by more than 0.2 for 2 % |

On a second dataset without poses or calibration (`data_test_roman/data/set_003`, 328 images)
the first two commands run unchanged: 98 % of the neighbours are estimated, the only unknown
ones being where a diver passes in front of the camera.

## What was learned

- **The relief breaks the flat-scene model.** A homography is reliable between neighbours, not
  below about 40 % of overlap. The camera displacement does not assume a flat scene and is
  unknown three times less often (83 pairs against 279).
- **Between neighbours, no rule beats another on precision.** Depth precision follows the
  number of images kept, whatever the rule. What the overlap rule guarantees is that the kept
  images still chain together, which taking one image in k or a random subset does not.
- **A revisit is almost never the same view again.** Between survey lines, 91 % of the revisits
  are seen 10° apart or more. They are the wide baselines a reconstruction needs, not
  duplicates to remove.
- **Loop closures are feasible from the images alone**: about a minute to propose, 20 min to confirm, for
  1,244 images instead of a day for all pairs.
- **Identical target plates trap the matching.** Two filters handle it; the second was added
  after a false link got through.

## Known limits

- **One dataset with ground truth.** Every threshold was set on Mermaid.
- **The reference is approximate**: a flat seafloor under at least 2.7 m of relief.
- **No calibration file format**: the displacement works on Mermaid only, through an option
  named after it.
- **The judge of 3D quality is a model**, never compared with a real reconstruction.
- **Loop closures are stored but not used yet**: `find_loops.py` proposes, confirms and
  records them; the selection does not read them. The check of the stored links against the
  poses was done by hand and is not in the repository.
- **Sequential numbers predate the last fix** of the inlier spread filter (2026-11-19) and were
  not measured again; the loop closure numbers were, on 2026-11-27.
- Features are extracted again at every sequential run, and an interrupted run cannot resume.

## Next

1. A script that reads the stored links back and judges them against the poses.
2. Measure the sequential overlaps again on Mermaid after the last fix.
3. A selection that uses the links between survey lines: remove an image only where it adds
   neither coverage nor a new viewpoint.
4. A calibration format, then the same measurements on a second dataset.

## Where the code is

| Package | Content |
|---|---|
| `features/` | SIFT, matching, VLAD image vectors, features on disk |
| `overlap/` | pairwise overlap, camera displacement, sequential mode, selection, loop candidates |
| `reference/` | Mermaid poses, footprints, revisits, and the judge of a selection |
| `scripts/overlap/` | the commands above and the evaluations behind the numbers |
