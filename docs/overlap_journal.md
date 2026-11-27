# Overlap engine — journal of decisions and measurements

The detailed record of person A's work: what was decided, what was measured and how, in the
order it happened. Every number here is for the Mermaid dataset unless stated otherwise.

For where the work stands today, read [`overlap_status.md`](overlap_status.md) first.

A result marked "one-off" came from a throwaway script that is not in the repository. Those
that matter were added later and can be run again:

| Result | Script |
|---|---|
| pose convention, camera top view | `scripts/reference/check_mermaid_poses.py` |
| footprints on the seafloor plane | `scripts/reference/check_footprints.py` |
| SIFT against Harris | `scripts/features/check_features_mermaid.py` |
| overlap error by reference overlap | `scripts/overlap/evaluate_pairwise_mermaid.py` |
| whole-dive sequential overlap and revisits between two passes | `scripts/overlap/explore_mermaid_dive.py` |
| selection, curve and its evaluation from the poses | `scripts/overlap/select_images.py --mermaid-poses` |
| overlap rule against decimation and random subsets | `scripts/overlap/compare_selection_rules.py` |
| loop closure candidates from resemblance | `scripts/overlap/evaluate_loop_candidates.py` |

Not reproducible from the repository yet: the per-image recall of revisits between two passes,
the comparison of angle-driven rules, the study of the inlier spread, the checks of the
displacement against the poses, and the verification of loop closure candidates by matching
(to come with the loop closure script).

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
- **SIFT defaults** (`scripts/features/check_features_mermaid.py`, 40 images sampled over the dive,
  gaps of 1 and 5 images): OpenCV's contrast threshold 0.04 leaves low-contrast sand with 16 keypoints
  (p5) and 1 match at gap 5; full resolution barely helps (4 matches). A threshold of 0.01 gives 56
  matches at gap 5 (p5). Capping at the 8000 strongest keypoints keeps ~all gap-1 matches (865 vs 888
  at p5) for 0.08 s/pair instead of 0.54 s. Defaults: `scale=0.5`, `contrast_threshold=0.01`,
  `max_keypoints=8000`; sequential mode with k = 5 on Mermaid ≈ 8 min of matching.
- **Images**: `match_poses_to_images` pairs all 1,244 poses with their JPG (0 missing, 0 extra).

## Harris baseline (branch `a/harris-features`)

- `HarrisSiftExtractor`: Harris corners (`goodFeaturesToTrack`) described by upright SIFT
  descriptors, sharing SIFT's preprocessing (`features/preprocessing.py`) and matcher.
- **Sub-pixel refinement**: `cv2.cornerSubPix` models a corner as two straight edges. On
  seafloor-like texture it left half the corners unmoved and threw 10 % more than 4.5 px away
  (median error 0.67 px on a sub-pixel shift, no better than integer corners, 40 % fewer matches).
  A per-axis parabola fitted to the Harris response gives 0.23–0.28 px and never moves a corner by
  more than half a pixel.
- **Result** (`scripts/features/check_features_mermaid.py`, homography inliers, MAGSAC 3 px):

  | Extractor | inliers gap 1 (p5 / median) | inliers gap 5 (p5 / median) |
  |---|---|---|
  | SIFT defaults | 128 / 457 | 7 / 64 |
  | Harris + upright SIFT (best setting) | 11 / 221 | 0 / 4 |

  Without orientation or scale, Harris descriptors break between images 5 apart (heading and
  altitude change). SIFT stays the default; Harris is kept as a comparison baseline.
- **For pairwise overlap**: even with SIFT only ~28 % of consecutive matches fit one homography
  (457 inliers of 1,604 matches): relief breaks the planar-scene assumption, as expected in
  `01_overlap_engine.md`.

## Pairwise overlap (branch `a/pairwise-overlap`)

- **`camera/`** (new shared package): the Metashape camera model, undistortion and image outline
  moved out of `reference/`; `MERMAID_INTRINSICS` stays in `reference/`. Mermaid's lens is barrel
  in the field (edge midpoints move ~65 px outwards when undistorted) but turns back near the
  corners (k2 > 0: corners move ~24 px inwards).
- **`overlap/pairwise.py`**: `estimate_homography_overlap` (MAGSAC, undistorted pixels) and
  `estimate_hull_overlap` (fundamental-matrix inliers, so no planar assumption). Each share is
  measured in its own image's frame; unknown overlap is `None` with a reason, never a number.
- **Validation of the homography**: no vertex beyond the horizon, no mirroring, area ratio within
  ×8 (largest footprint area ratio between overlapping Mermaid images: 5.2), ≥ 15 inliers.
  MAGSAC threshold 8 px: relief creates parallax a single homography cannot explain; 3 → 8 px cut
  unknowns at 20–40 % overlap from 80 % to 73 % with the same error, 15 px accepted false overlaps.
- **Repeated targets trap**: GCP plates 28 and 30 share the same pattern; matched to each other
  they gave a consistent homography and a 79 % overlap for two images that do not overlap. All
  inliers sat on the plate: hull / common area 0.005, vs ≥ 0.026 for the 102 correct estimates.
  Pairs below `MIN_INLIER_SPREAD = 0.01` are unknown. Only one such case observed so far.
- **Result** (`scripts/overlap/evaluate_pairwise_mermaid.py`, 150 pairs, |error| median):

  | Reference overlap | homography raw | homography undistorted | inlier hull | unknown (undistorted) |
  |---|---|---|---|---|
  | 0.8–1.0 | 0.023 | 0.031 | 0.174 (bias −0.17) | 0 % |
  | 0.6–0.8 | 0.038 | 0.044 | 0.212 (bias −0.21) | 11 % |
  | 0.4–0.6 | 0.059 | 0.060 | 0.187 | 30 % |
  | 0.2–0.4 | 0.115 | 0.076 | 0.086 | 73 % |
  | 0.0–0.2 | – | 0.058 (1 pair) | 0.035 | 95 % |

  (raw column measured at 3 px, the others at 8 px.)

  Undistortion gives +23 % inliers (median) and more inliers in 85 % of pairs, so the model fits
  the images better; it does not improve agreement with the reference, whose own error (horizontal
  plane vs ≥ 2.7 m of relief) dominates at this level. The inlier hull underestimates by ~20 points
  (points cover 15–30 % of the common area). Five pairs keep an error > 0.15 with a normal inlier
  spread (e.g. 168→170: reference 0.86, estimate 0.52): not explained yet, candidates are relief and
  the reference plane.
- **Honest limit**: reliable only for nearly consecutive images. Enough for deduplication (≥ 95 %),
  borderline for the ~80 % extraction, not enough below ~40 % overlap (60–73 % unknown).
- **Keypoint cap trade-off** (measured, not applied): without the 8000 cap, unknowns at 40–60 %
  overlap drop from 30 % to 17 % (60–80 %: 11 % → 5 %), but brute-force matching is 7× slower
  (0.54 vs 0.08 s/pair, ~1 h vs 8 min for k = 5 on Mermaid). To revisit with a faster matcher
  (FLANN) in its own branch, since it changes the SIFT default used by B. Other levers to measure:
  masking GCP plates (they dominate matches), better descriptors, a 3D model instead of a homography.

## Notes for the n → n mode (loop closures)

- The robot / diver path must be encoded, both ways:
  - **sequence graph**: capture order as edges between consecutive images, the structure the
    sequential overlaps live on;
  - **positions** when available: reference poses (Mermaid, Eiffel Tower), navigation from the ROS
    bags (AQUALOC IMU and pressure, C's ingestion), or a trajectory chained from sequential overlaps
    (visual odometry).
- Candidate loop pairs are then images close in space but far apart in time.
- **Mermaid has real revisits**: per the trajectory in report 107174, passes 0–207 and 832–1039
  fly over the same areas (`scripts/overlap/explore_mermaid_dive.py revisits` tests them).
- **The encoded path drifts** (navigation or chained odometry accumulates error): use it to
  pre-select candidates, never to decide that two images overlap. The images decide.
- **Read the target IDs** (for later, with C): each GCP plate carries a unique number (26, 28, 30
  read in the images). The same ID in two images links them regardless of drift, and different IDs
  defuse the identical-plates trap found in pairwise overlap.
- The shared schema has no trajectory table yet (`images` only has `sequence` and `timestamp_s`):
  adding one is a change to the `storage/` contract, to agree with B and C.

## Overlap and loop closures on the whole dive (exploration, 2026-10-01)

`scripts/overlap/explore_mermaid_dive.py` (SIFT features cached in `outputs/cache/`):

- **Sequential** (1,244 images, each vs its 5 next, 6,205 pairs): gap 1 → 0 % unknown, |error|
  median 0.025; gap 5 → 12.6 % unknown, 0.054. Unknowns concentrate on images ~50–500, where
  overlap with +5 is only 0.3–0.6; texture is not the cause (8,000 keypoints almost everywhere).
- **Revisits** (passes 0–200 vs 800–1000): 68 % of the 40,000 cross-pass pairs overlap. Sampled
  400 per band: found 14.8 % of revisits at ≥ 0.6 overlap (|error| 0.005), 4.8 % at 0.4–0.6, 1.5 %
  at 0.2–0.4; **0 false alarms** on 400 non-overlapping pairs. Precise, but poor recall.
- **Why**: between passes the diver flies the other way (heading difference > 90° for 63 % of the
  pairs, recall 11 % vs 33 % under 20°) and much higher (altitude ratio > 2 for 91 %, recall 13 %
  vs 35 % between ×1.3 and ×2; ratios relative to the approximate reference plane). Only ~30
  matches survive per pair with the sequential settings.
- **Costlier verification does not help** (`scripts/overlap/compare_verification_settings.py`,
  200 revisits ≥ 0.6, 200 non-overlapping pairs): current settings 14.0 % found; no keypoint cap
  14.5 %; no cap + ratio 0.80 15.5 %; full resolution (cap 20,000, ratio 0.80) 11.0 %. Always 0 false
  alarms, 6–8× slower. The bottleneck is not the matcher's settings.
- **What the revisits really are**: between the two passes no pair has a similar footprint
  (reference IoU ≥ 0.5: 0 %, ≥ 0.3: 2.5 %). Almost every "revisit" is a close-up from the first
  pass (~1–2 m above the seafloor) contained in a wide view from the second, ~3 m higher, seen
  through more turbid water (pale, blurred, GCP plates a few pixels wide). Recall by reference IoU:
  0.30–0.50 → 36 %, 0.15–0.30 → 21 %, < 0.15 → 3 %. Finding a close-up inside a hazy wide view is
  the hardest case for local features; the 14 % measured mostly that.
- **Next**: define what a useful loop closure is (similar footprints, e.g. IoU or min(a_in_b,
  b_in_a), rather than a_in_b alone); measure contrast enhancement (e.g. CLAHE) against turbidity
  on the IoU 0.15–0.5 band; then choose the image-based candidate pre-selection (VLAD on SIFT,
  CPU only) and test it on other datasets with real revisits at similar altitude (AQUALOC,
  UnderLoc / SQUIDLE+ VPR).

## Sequential mode into `pairs` (branch `a/sequential-overlap`)

- **`overlap/sequential.py`**: each image against its next `neighbour_count` images, over a sliding
  window. Only the window is in memory and pairs are yielded one at a time, so the input can be
  read from disk as it goes. Pairs come out grouped by their second image.
- **`overlap/recording.py`**: one `pairs` row per pair. An unknown overlap is a row with NULL
  overlaps and its inlier count: "compared, could not tell" differs from "never compared".
  Database ids are not assumed to follow capture order: when they do not, the directional values
  are swapped with the ids (the table requires `image_a_id < image_b_id`).
- **`overlap/pipeline.py`** and `scripts/overlap/run_sequential.py`: image files → features →
  overlaps → `pairs`. An unreadable file is skipped and its two neighbours become consecutive.
- **`storage/images.py`**: image file registration moved out of `scripts/cleaning/make_masks.py`
  so that every module can use it until ingestion exists.
- **Check on Mermaid** (calibrated, 1,244 images, 5 neighbours, 6,205 pairs): same numbers as the
  exploration script, which this pipeline replaces for the sequential mode.

  | Gap | unknown | \|error\| median |
  |---|---|---|
  | 1 | 0.0 % | 0.025 |
  | 2 | 0.3 % | 0.039 |
  | 3 | 2.3 % | 0.047 |
  | 4 | 7.3 % | 0.052 |
  | 5 | 12.6 % | 0.054 |

  279 unknown pairs: 222 too few inliers, 53 implausible homography, 4 inliers too concentrated.
  The last 4 (322→327, 323→327, 1144→1149, 1147→1152, all at gap 4 or 5) are not the
  identical-plates trap: the two images do overlap (reference 0.56–0.73) and show the same GCP
  plate. But every inlier sits on that plate (0.15–0.5 % of the image, spread 0.007–0.008): the
  seafloor around it did not match. A homography fitted on the plate alone extrapolates badly to
  the whole image: it would have claimed 0.20, 0.29, 0.64 and 0.37 against 0.56, 0.62, 0.73 and
  0.65. So `MIN_INLIER_SPREAD` rejects them rightly, for a second reason: too small a support,
  not only a repeated target.
- **Limits of the script**:
  - capture order is the order of the file paths (true on Mermaid: it matches the pose order),
    until ingestion fills `images.timestamp_s`;
  - no calibration: there is no calibration file format yet, so the script measures raw pixels
    (the library function accepts intrinsics);
  - features are extracted again at every run (the `features` table is not filled);
  - an interrupted run keeps its pairs but cannot be resumed;
  - the reason of an unknown overlap is only printed: `pairs` has no column for it.
- **Cost**: extraction 0.15 s per image, matching ~0.08 s per pair, both inside OpenCV. About
  10 min for Mermaid on 8 cores; brute-force matching is the first thing to speed up.

## Deduplication (branch `a/deduplication`)

- **Definition**: an image is a near-duplicate when its **mutual overlap**, `min(a_in_b, b_in_a)`,
  with an image already kept reaches the threshold. The minimum avoids flagging a close-up
  contained in a wide view (1.00 one way, 0.10 the other).
- **Compared with kept images only**, never with the previous image as such: a slow camera makes
  every image a duplicate of its predecessor while drifting away from where it started.
- **Unknown or missing pair → the image is kept.** Pairs need not be consecutive: overlaps found
  later between survey lines or passes (n → n mode) will be used without changing this code.
- `scripts/overlap/find_duplicates.py` reads `pairs`, does no matching, and writes one CSV line
  per image (`kept`, or `duplicate` of which image) in `outputs/deduplication/`. A table would be
  a schema change: to propose once the format has settled.
- **Result on Mermaid** (overlaps of the calibrated sequential run, 1,244 images):

  | Threshold | near-duplicates | reference mutual overlap of the flagged pairs (min / median) |
  |---|---|---|
  | 0.95 | 37 (3.0 %) | 0.89 / 0.95 |
  | 0.90 | 254 (20.4 %) | 0.79 / 0.92 |
  | 0.85 | 500 (40.2 %) | not measured |
  | 0.80 | 683 (54.9 %) | not measured |

  No flagged pair is far from the threshold according to the poses (|error| median 0.008 at
  0.95). The reference counts 44 consecutive pairs at ≥ 0.95 and 400 at ≥ 0.90, which is not
  directly comparable: a chain of duplicates is flagged against one kept image, not pair by pair.
- **Limits**: with k neighbours compared, at most k images in a row can be flagged, since the
  next one was never compared with the kept image (to fix by comparing on demand). Only temporal
  neighbours are compared, so duplicates between survey lines are not seen yet. Not validated on
  a manually annotated sample, as `01_overlap_engine.md` asks.

## Selection of a minimal subset (branch `a/extraction`)

- **Rule** (`overlap/extraction.py`): keep an image, jump to the last following image that still
  overlaps it by the threshold, keep that one, repeat. The jump stops at the first pair that is
  below the threshold, unknown or never compared. The image after a kept one is always kept, so
  the subset never has a gap the sequence did not have; first and last images are always kept.
- **Measure**: the share of the kept image seen by the next one (`earlier_in_later`), as in
  `01_overlap_engine.md`. The mutual overlap gives nearly the same counts (357 instead of 385
  removed at 0.80).
- `scripts/overlap/select_images.py` reads `pairs`, does no matching (0.6 s on Mermaid) and writes
  `kept_images.txt`, `images.csv`, `curve.csv`, `curve.png` and `summary.txt` in
  `outputs/selection/run_<id>/`.
- **Result on Mermaid** (calibrated sequential run, 5 neighbours, 1,244 images). The last column
  applies the same rule to the pose-based reference overlap, with no limit on the jump.

  | Overlap kept | removed, from the images | removed, from the poses |
  |---|---|---|
  | 0.95 | 8 (0.6 %) | 9 (0.7 %) |
  | 0.90 | 67 (5.4 %) | 126 (10.1 %) |
  | 0.85 | 224 (18.0 %) | 339 (27.3 %) |
  | 0.80 | 385 (30.9 %) | 542 (43.6 %) |
  | 0.75 | 544 (43.7 %) | 713 (57.3 %) |
  | 0.70 | 682 (54.8 %) | 816 (65.6 %) |
  | 0.60 | 847 (68.1 %) | 969 (77.9 %) |
  | 0.50 | 928 (74.6 %) | 1,050 (84.4 %) |

  The image-based selection is conservative: an unknown overlap ends the jump. The limit of 5
  neighbours is hardly involved (reached 28 times at 0.80).
- **Weak links**: at 0.80, 205 of the 858 links between consecutive kept images are below 0.80
  by our own estimate, and none is unknown. They are not selection errors: there the next image
  already overlaps the kept one by less than 0.80, and it is kept so as not to open a gap. The
  sequence itself lacks the overlap (median 0.87 between neighbours), and `summary.txt` reports
  the count. Against the poses, 23 links are below 0.60 (minimum 0.35): all are consecutive
  images, and our estimate also puts them low (0.40–0.75), 0.1–0.2 above the reference. They are
  tilted views (26° from the vertical in median, against 17° for the other links) or sharp
  turns (875→876: heading change of 144°), where the horizontal-plane reference is least
  reliable too.
- **Limits**: Mermaid's neighbours overlap by 0.87 in median, so little can be skipped at 0.80.
  Only neighbours in time are compared, while most of the redundancy is between survey lines
  (28 images per place in median according to the poses). The criterion is 2D: nothing yet says
  the 3D reconstruction stays as good, which is the question raised by the supervisors
  (2026-11-17): remove stationary images, keep different viewpoints.

## Direction after the progress review (2026-11-17)

Supervisors' feedback, as noted during the meeting:

- Removing 2D duplicates is the easy part; the question is redundancy **for 3D**.
- Likely target: remove stationary images and produce a **minimal set with the same 3D quality**.
- The same area seen from another angle (relief) may not be redundant: open.
- A loop closure needs the 3D relation between the two views, not only a 2D overlap.
- Wanted figures: images removed at ~80 % overlap, and compute time against redundancy.
- Targets: to be discussed again.

Open questions for them: how "same 3D quality" is measured, with which software, whether they
can reconstruct a subset we provide, and which compute time is meant.

## Findings from one-off explorations (2026-11-17 and 18)

Computed with throwaway scripts, not in the repository. Numbers are for Mermaid only.

- **Revisits, recall per image.** 50 images of pass 0–199 (one in four) against all 200 images of
  pass 800–999, sequential settings, 10,000 pairs. Per pair the recall is the known one (14.2 %
  at reference overlap ≥ 0.6, 7.5 % of all overlapping pairs), but **33 of the 50 images (66 %)
  get at least one link** to the other pass, 26 get at least three, and none gets a false link
  (0 false alarms on 3,227 non-overlapping pairs). An image has 146 overlapping partners in
  median, so missing nine in ten still leaves several. Found links are accurate (|error| median
  0.012, 49 inliers in median). The 17 images without a link form zones (0–24, 68–80, 96–104,
  160–168) and depend on how similar the best partner is: best reference IoU ≥ 0.30 → 16 of 19
  images linked, 0.15–0.30 → 14 of 20, < 0.15 → 3 of 11. Linking two passes is feasible with
  the current matcher; the open problem is finding the candidates without the poses.
- **"Any overlap" is not a revisit.** From the poses, 74 % of all image pairs overlap (569,077 of
  773,146), every image overlaps one taken more than 30 images away, and a place is seen by 28
  images in median (up to 583). Footprints are several metres wide while survey lines are less
  than a metre apart. A useful revisit needs a stricter definition (similar footprints).
- **Pose-based indicator of what a subset keeps for 3D (proof of concept).** For every 10 cm
  cell of the seafloor plane: how many kept images see it, and how wide the fan of their viewing
  directions is. A cell counts as reconstructible with at least 3 views spread over at least
  10° (both thresholds are a first guess).

  | Subset | images | reconstructible area kept | views (median) | fan (median) |
  |---|---|---|---|---|
  | all images | 1,244 | 100 % | 28 | 58° |
  | selection 0.90 | 1,177 | 98.4 % | 25 | 58° |
  | selection 0.80 | 859 | 94.2 % | 15 | 57° |
  | selection 0.70 | 562 | 88.9 % | 9 | 52° |
  | selection 0.50 | 316 | 80.0 % | 6 | 46° |
  | 316 random images | 316 | 78.6 % | 6 | 45° |
  | 1 image in 10 | 125 | 67.3 % | 3 | 36° |
  | 1 image in 50 | 25 | 35.7 % | 1 | 0° |

  The indicator behaves as expected: it decreases steadily and collapses when too much is
  removed, and it separates coverage from viewpoint diversity (images 1040–1243 cover 77 % of
  the area with a 30° fan, against 58° for the whole dive). It also shows that **the sequential
  2D selection barely beats a random subset of the same size** (80.0 % against 78.6 %): the
  argument for measuring camera displacement and for using links between survey lines.
  Limits: flat seafloor, so areas and angles are wrong on the relief; it says what the geometry
  allows, not whether the images actually connect; it has not been compared with a real
  reconstruction.
- **The indicator is now in the repository**: `reference/observability.py`, used by
  `scripts/overlap/select_images.py --mermaid-poses data/mermaid/107177.xml`, which appends the
  reconstructible area kept by the selection to the summary (same numbers as the table above,
  2 s). It needs the poses, so it is an evaluation on Mermaid, not part of the generic output.
- **Camera displacement between consecutive images, as a triangulation angle** (exploration for
  the next step, 1,243 pairs, 153 s). Essential matrix from the matches (calibrated, MAGSAC,
  3 px), then for each inlier the angle between its ray in A and its ray in B rotated back to A;
  the pair's value is the median. Compared with the same angle computed with the rotation from
  the poses: estimate available for 1,235 pairs (99.4 %), |error| median 0.17° (p90 0.72°,
  p99 2.8°), correlation 0.97; rotation error 0.29° in median. The angle is 3.6° in median
  between neighbours (p5 1.3°, p95 8.7°) for a 0.24 m step. Mermaid has almost no truly
  stationary images: 28 pairs under 1° (2.3 %), 176 under 2° (14 %), in runs mostly between
  images 511 and 937, where the camera is high and the step is small against the distance.
  Overlap and angle are related but not the same (correlation −0.58): of the 176 pairs under 2°,
  81 have a mutual overlap below 0.90, and of the 340 pairs at overlap ≥ 0.90, 95 are under 2°
  and 19 are at 5° or more. Deciding "under 2°" from the images finds 153 of the 176, with 74
  false alarms: the threshold needs a margin.

## Camera displacement (branch `a/camera-displacement`)

- **`overlap/displacement.py`**: `estimate_displacement` returns the triangulation angle between
  two images (median over the inliers, rotation removed), how much the camera turned, and which
  model gave the rotation. It needs the calibration.
- **Flat scenes and pure rotation** (found with synthetic scenes, hidden by Mermaid's relief):
  coplanar points fit several essential matrices, and the estimator answered 0.7° for a true
  5.2° on a flat seafloor; a camera only turning gave no answer. When one homography explains
  at least 80 % of the matches that fit the essential matrix, the homography is decomposed
  instead and the plane facing the camera is kept. That choice is an assumption (right for a
  seafloor seen from above); `rotation_source` records when it was used.
- **MAGSAC does not refine the essential matrix**: with wrong matches the rotation was off by
  0.28° on exact synthetic points; `USAC_ACCURATE` gives 0.00° (0.07° with 0.5 px of noise).
- **Gross errors and their cause.** With one estimator, about 1 % of the consecutive Mermaid
  pairs got an angle near 1° where the truth is near 5° (316, 78, 266…). The estimate was the
  same whatever the random seed, and the matches were fine: the true pose fitted more of them
  within 3 px than the returned one (935 against 726 for pair 316). The returned essential matrix
  fitted the epipolar lines but put about half of the points behind a camera. Each estimator
  (`USAC_ACCURATE`, MAGSAC, LMEDS) fails this way, rarely on the same pairs.
- **Fix: several candidates, one count.** Every estimator proposes its essential matrix, each
  decomposed into its four poses, and the homography adds its own; the pose supported by most
  matches wins, a match supporting a pose when it is within 3 px of its epipolar line **and** in
  front of both cameras. On the 25 worst pairs the error drops from 3.0° to 0.4° in median (21
  fixed, 4 left), with no change on 25 ordinary pairs.
- **The plane model must not win on support alone**: with a small step, a wrong pose fits the
  epipolar line of every match (0.84° of invented rotation for a 10 cm step at 2 m on synthetic
  relief). It is preferred only when the homography itself maps at least 80 % of the supported
  points, and must otherwise be clearly better supported.
- **On Mermaid** (1,243 consecutive pairs, 201 s for matching and estimation):

  | | one estimator (MAGSAC) | candidates |
  |---|---|---|
  | pairs with an estimate | 1,235 | 1,243 (all) |
  | \|error\| median | 0.17° | 0.13° |
  | \|error\| p90 | 0.72° | 0.56° |
  | \|error\| p99 | 2.83° | 1.63° |
  | correlation with the truth | 0.971 | 0.986 |
  | "under 2°": found / false alarms | 153 of 176 / 74 | 145 of 176 / 30 |

  The plane model is used for 26 pairs (2.1 %), less accurately (0.47° in median).
- **Left**: about 8 pairs (0.6 %) still read 0.2–0.7° for a truth of 2.7–4.0° (1065, 794, 264,
  420, 776…): no candidate finds the right pose. The plane share of 80 % and the plane facing
  the camera are still unmeasured assumptions, to settle on a dataset with flat seafloor. The
  truth uses the rotation of the poses on our own matches: it checks the rotation, not the
  matches.
- **Where the 0.80 selection loses reconstructible area** (5.8 %, 23 of 397 m²): on the rim.
  99 % of the lost cells are within 1 m of the edge of the reconstructible area and 83 % within
  0.5 m; more than 1 m inside, 0.07 % is lost (0.22 of 316 m²). The lost cells were seen by 5
  images in median with all images, against 61 for the whole reconstructible area: they were
  just above the threshold of 3 views over 10°. 74 % are lost for having fewer than 3 views
  left, 26 % for a fan under 10°, and 5 % are no longer seen at all. The rim comes from wide,
  tilted footprints, the least reliable part of the flat-seafloor reference.
- **Displacement over the whole dive** (`outputs/overlap/run_21/displacements.csv`, 6,205 pairs,
  18 min 28 s with matching): the angle grows by about 3.5° per image of gap (3.6°, 7.0°, 10.0°,
  13.0°, 15.4° in median for gaps 1 to 5), so three images of gap are needed to reach 10°. It is
  unknown for 83 pairs, against 279 for the overlap (4.4 % against 12.6 % at gap 5): it does not
  assume a flat scene.

## Selection rules compared (one-off experiment, 2026-11-18)

Greedy rules along the sequence, judged from the poses. Overlaps and angles of run 21.

- **Using the angle between neighbours does not help.** Jumping until the camera has moved by a
  target angle (with a floor on the overlap), or skipping an image the camera barely moved for,
  gives the same reconstructible area as the overlap rule at equal size or less (e.g. 636 images
  → 89.5 %, 856 → 93.5 %).
- **Uniform decimation and random subsets keep as much reconstructible area as the overlap
  rule, or more**: 1 image in 2 (622) → 92.9 % against 90.4 % for overlap ≥ 0.75 (700); 1 in 4
  (311) → 86.3 % against 80.0 % for overlap ≥ 0.50 (316); random 859 → 96.5 % against 94.2 %.
  The overlap rule removes most where neighbours overlap most, i.e. in the high, wide views
  that cover the rim; and the interior is nearly intact for every subset (≥ 98 % down to ~400
  images). **The indicator saturates**: with tens of views per place, "3 views over 10°" only
  discriminates on the rim.
- **What the overlap rule buys is the chain.** Mutual overlap between consecutive kept images,
  from the poses:

  | Subset | images | median | p5 | links < 0.5 | < 0.3 | < 0.1 |
  |---|---|---|---|---|---|---|
  | all images | 1,244 | 0.87 | 0.67 | 15 | 1 | 1 |
  | overlap ≥ 0.80 | 859 | 0.84 | 0.61 | 15 | 1 | 1 |
  | overlap ≥ 0.70 | 562 | 0.76 | 0.55 | 15 | 1 | 1 |
  | overlap ≥ 0.50 | 316 | 0.63 | 0.38 | 65 | 3 | 1 |
  | 1 image in 2 | 622 | 0.79 | 0.48 | 35 | 5 | 0 |
  | 1 image in 4 | 311 | 0.66 | 0.27 | 61 | 20 | 1 |
  | random | 562 | 0.81 | 0.38 | 45 | 14 | 4 |
  | random | 316 | 0.74 | 0.19 | 68 | 28 | 13 |

  Down to 0.70 the overlap rule adds no weak link to those the dive already has; decimation and
  random subsets break the sequence in places.
- **Three definitions of the fan give the same ranking**: diagonal of the box of viewing
  directions (397 m² reconstructible with all images), widest pair of directions, i.e. a real
  triangle ground point – camera – camera (396 m²), and that plus directions not aligned, i.e.
  three cameras forming a triangle rather than a line (374 m²: 5.5 % of the area is seen along
  one line only).
- **Consequences.** A quality measure needs two parts, what can be reconstructed and whether the
  images stay connected; the first one must grade instead of thresholding (e.g. expected depth
  precision from the number of views and their spread). And the gain is not in a smarter rule
  between neighbours: it is between survey lines, which no sequential rule sees.

## A judge with two parts (branch `a/selection-evaluation`)

`reference/evaluation.py`, shown by `select_images.py --mermaid-poses`:

- **What can be reconstructed**, now graded. `depth_information` is the scatter of the viewing
  directions of a cell around their mean, summed over the images; depth noise is proportional to
  1 / sqrt of it, so it grows with fewer images and with images standing closer together. The
  summary gives the depth noise relative to all the images, and the reconstructible area kept
  inside and on the rim (within 1 m of its edge) separately.
- **Whether the images stay chained**: consecutive kept images whose mutual overlap, from the
  poses, is below 0.50 and 0.30, next to the same count for all the images.

  | Selection | images | inside kept | rim kept | depth noise (median / worst tenth) | links < 0.50 | < 0.30 |
  |---|---|---|---|---|---|---|
  | all | 1,244 | 100 % | 100 % | ×1.00 | 15 | 1 |
  | overlap ≥ 0.80 | 859 | 99.9 % | 71.9 % | ×1.22 / ×1.48 | 15 | 1 |
  | overlap ≥ 0.50 | 316 | 95.8 % | 18.7 % | ×2.05 / ×2.80 | 65 | 3 |

  The depth noise is a model, not a measure: it assumes independent pixel noise and a flat
  seafloor, and has not been compared with a real reconstruction.
- **Rules compared again with this judge** (one-off, run 21):

  | Rule | images | inside kept | rim kept | depth noise (median / worst tenth) | links < 0.50 | < 0.30 |
  |---|---|---|---|---|---|---|
  | overlap ≥ 0.90 | 1,177 | 100.0 % | 92.3 % | ×1.03 / ×1.09 | 15 | 1 |
  | overlap ≥ 0.80 | 859 | 99.9 % | 71.9 % | ×1.22 / ×1.48 | 15 | 1 |
  | random | 859 | 100.0 % | 83.7 % | ×1.19 / ×1.29 | 33 | 7 |
  | overlap ≥ 0.75 | 700 | 99.8 % | 54.0 % | ×1.39 / ×1.88 | 13 | 1 |
  | 1 image in 2 | 622 | 100.0 % | 65.6 % | ×1.42 / ×1.64 | 35 | 5 |
  | random | 622 | 99.9 % | 66.9 % | ×1.41 / ×1.83 | 41 | 7 |
  | overlap ≥ 0.70 | 562 | 99.5 % | 48.0 % | ×1.56 / ×2.27 | 15 | 1 |
  | angle ≤ 12°, overlap ≥ 0.5 | 524 | 99.3 % | 31.8 % | ×1.74 / ×2.71 | 36 | 2 |
  | 1 image in 3 | 415 | 99.8 % | 50.8 % | ×1.75 / ×2.25 | 52 | 13 |
  | overlap ≥ 0.50 | 316 | 95.8 % | 18.7 % | ×2.05 / ×2.80 | 65 | 3 |
  | 1 image in 4 | 311 | 98.5 % | 38.8 % | ×2.05 / ×2.77 | 61 | 20 |
  | random | 316 | 96.1 % | 23.9 % | ×2.07 / ×2.94 | 64 | 26 |

  **Depth precision follows the number of images, whatever the rule**: the median noise is
  close to sqrt(1,244 / kept) everywhere (×1.20 expected at 859, ×1.41 at 622, ×1.98 at 316). No
  sequential rule buys precision; the overlap rule is even a little more uneven (worst tenth
  ×1.48 against ×1.29 for a random subset of 859) and loses more of the rim. **What it does buy
  is the chain**: down to 0.70 it keeps the dive's own weak links (15 and 1) where decimation
  and random subsets double the links below 0.50 and multiply those below 0.30 by 5 to 26. The
  angle-driven rules are worse on both counts.
  To remove images without losing precision, a rule must know where a place is already seen
  many times, i.e. use the overlaps between survey lines.

## Second dataset: `data_test_roman/data/set_003` (2026-11-18)

328 JPEG images of 4056 × 3040, one every 2.2 s over 12 minutes (2024-09-23), a diver over rocky
seafloor, very wide lens behind a port that darkens the corners. No poses, no calibration.
`run_sequential.py` was run unchanged (5 neighbours, 1,625 pairs, 3 min 29 s).

| Gap | unknown | overlap (median) | inliers (median) |
|---|---|---|---|
| 1 | 1.8 % | 0.74 | 427 |
| 2 | 8.3 % | 0.58 | 130 |
| 3 | 37.5 % | 0.50 | 32 |
| 4 | 62.7 % | 0.44 | 11 |
| 5 | 85.8 % | 0.39 | 5 |

- **Neighbours are estimated all along the sequence** (98 %), including where the camera rises
  and the water turns blue. There is no ground truth: only consistency and a look at the images.
- **The port's rim does not pollute the matches**: on 5 pairs no inlier stayed at the same pixel
  from one image to the next, and the dark corners are 0.4–1 % of the image.
- **The only unknown neighbours are images 5–9 (and 264)**: a diver passes in front of the
  camera. "Unknown" was the right answer, and the case is one for the masks of block VI.
- **A truly stationary camera**: images 0–5 overlap by 0.95–0.99, the case Mermaid lacks.
- **Less redundancy than Mermaid**: 0.74 between neighbours against 0.87, and the estimate
  collapses beyond 2 images of gap, so comparing 5 neighbours is mostly wasted here. Fewer
  matches too (350–600 per pair against ~1,600, for the same 8,000 keypoints).
- **Selection**: 4 images removed at 0.80 (the stationary start), 22 at 0.70, 84 at 0.60, 153 at
  0.50; 253 links are below 0.80 in the dive itself. A fixed 80 % makes no sense here: the
  threshold belongs to the user, and the curve is what lets them choose.
- **The curve is not monotonic**: 3 images removed at 0.90, 1 at 0.85. The greedy rule starts
  from the first image and does not stop on the same images for every threshold.
- **Missing most**: a calibration (no displacement, unknown distortion error on the overlaps),
  then flagging images that do not show the seafloor, and a number of neighbours that adapts.
- The images are committed in the repository (678 MB), unlike `data/`, which is ignored.

## Loop closures, step 1: what a useful revisit is (branch `a/loop-candidates`)

From the poses (flat seafloor), every pair of images more than 30 images apart in time: 736,291
pairs, of which 537,363 (73 %) have intersecting footprints. The angle is the one under which
the two cameras see the middle of what they share.

| Definition | pairs | images with ≥ 1 | per image (median, p10–p90) | angle (median) | altitude ratio | heading difference |
|---|---|---|---|---|---|---|
| any overlap | 537,363 (73 %) | 100 % | 908 (551–1,113) | 47° | 1.41 | 88° |
| mutual overlap ≥ 0.3 | 145,918 (19.8 %) | 99 % | 248 (58–403) | 32° | 1.23 | 81° |
| **mutual overlap ≥ 0.5** | 44,014 (6.0 %) | 96 % | 66 (9–146) | 27° | 1.15 | 78° |
| mutual overlap ≥ 0.7 | 6,369 (0.9 %) | 77 % | 6 (0–26) | 23° | 1.09 | 71° |
| mutual overlap ≥ 0.8 | 913 (0.1 %) | 39 % | 0 (0–5) | 19° | 1.06 | 23° |
| IoU ≥ 0.3 | 102,840 (14.0 %) | 99 % | 173 (37–286) | 29° | 1.25 | 79° |
| IoU ≥ 0.5 | 23,789 (3.2 %) | 93 % | 33 (3–81) | 25° | 1.17 | 77° |

- **Working definition: mutual overlap ≥ 0.5, more than 30 images apart.** Each image sees at
  least half of the other. It keeps 6 % of the far pairs while nearly every image still has
  some (1,195 of 1,244, 66 in median); the 49 without are mostly the first images (0–26).
- **A revisit is almost never the same view again.** At mutual ≥ 0.5 the two cameras see the
  shared area 27° apart in median (p10 10°, p90 50°): 91 % of the revisits are 10° apart or
  more, and only 944 pairs (2 %) are under 5°, concerning 545 images. Between survey lines the
  redundancy is in coverage, not in viewpoint: these pairs are the wide baselines a
  reconstruction wants, not duplicates. This answers the supervisors' open point for Mermaid.
- **It is also why they are hard to find**: 68 % of them are 20° apart or more and the heading
  differs by 78° in median, far from the 3.6° between neighbours on which the matcher was tuned.
- Time gap at mutual ≥ 0.5: 258 images in median (p10 91, p90 586): adjacent lines and later
  passes both contribute.
- Limits: flat seafloor, so areas and angles are approximate, least reliable for tilted views.

## Loop closures, step 2: candidates from image resemblance (one-off, 2026-11-19)

One VLAD vector per image from the cached SIFT features: RootSIFT descriptors, a codebook of 64
words (k-means on 120 descriptors per image), residuals summed per word, each word normalised,
signed square root, L2. Similarity is the dot product; images 30 or fewer apart in time are
excluded. No new dependency, CPU only: 21 s for the 1,244 images, codebook included.

Judged against the revisits of step 1 (mutual overlap ≥ 0.5, from the poses). A random far pair
is such a revisit 6.0 % of the time.

| k candidates per image | images with a true revisit among them | candidates that are revisits | … at mutual ≥ 0.3 | angle of the revisits found (median) |
|---|---|---|---|---|
| 1 | 56 % | 53.7 % | 79.0 % | 12° |
| 5 | 76 % | 50.3 % | 75.0 % | 13° |
| 10 | 85 % | 47.3 % | 72.3 % | 15° |
| 20 | 91 % | 43.0 % | 68.5 % | 16° |
| 50 | 95 % | 35.6 % | 62.1 % | 19° |

- **Resemblance proposes well**: with 10 candidates, 85 % of the images that have a revisit get
  at least one, and nearly half of the candidates are revisits, eight times chance. 91 % of the
  candidates at least touch the image.
- **It favours the views that look alike**, as expected: share of the true revisits proposed in
  the top 10 of either image, by angle between the two views: under 5° → 44 %, 5–10° → 33 %,
  10–20° → 16 %, 20–40° → 8 %, over 40° → 3 % (11 % overall, 36 % in the top 50). The revisits
  found are 12–15° apart in median against 27° for all of them. That bias suits the next step,
  since those are the pairs the matcher can verify, but the widest baselines are not proposed.
- 180 images with revisits get none in their 10 candidates, mostly early in the dive (67 in
  images 0–199, 39 in 200–399).
- Not measured yet: how many candidates survive verification by matching, and whether other
  settings (more words, fewer keypoints, a learned descriptor) do better.

## Loop closures, step 3: candidates verified by matching (one-off, 2026-11-19)

The 10 most resembling far images of every image (VLAD, no poses) give 10,916 distinct pairs.
Each was matched, then overlap (homography) and displacement were estimated with the sequential
settings: 1,499 s (25 min), 0.14 s per pair. Poses are used only to judge.

- **Of the candidates**, 4,915 (45 %) are true revisits (mutual ≥ 0.5 from the poses), 5,019
  touch the image without reaching 0.5, and 982 do not touch it.
- **Verification accepts the right ones**: an overlap is estimated for 90 % of the true
  revisits, 56 % of the merely touching ones, and **1 of the 982 that do not touch** (320–831,
  0.72 claimed with 17 inliers; image 320 shows GCP plates, the known repeated-target trap, not
  checked). Accepted links: |error| on the overlap 0.047 in median (p90 0.126); 642 have a
  reference mutual overlap below 0.3.
- **By angle between the two views** (true revisits): verified 100 % under 10°, 96 % at 10–20°,
  78 % at 20–40°, 69 % beyond 40°. Far better than the 14 % measured between passes 0–200 and
  800–1000 on all overlapping pairs: resemblance pre-selects the pairs that can be matched.
- **Per image, without any pose**: 1,189 of 1,244 images (96 %) get at least one verified link
  to an image more than 30 apart in time, 1,124 (90 %) at least three, 10 in median. The 55
  without are mostly the first images (35 in 0–199). The far links alone join 1,187 images
  (95 %) in one connected group.
- **Displacement** is known for 75 % of the candidates against 66 % for the overlap, and for 98 %
  of the images on at least one far pair. On verified links the estimated angle is 17.5° in
  median against 16.8° from the poses (|difference| 2.1° in median, the two being measured
  differently: median over matched points against the middle of the shared footprint).
  2,984 verified links are 20° apart or more.
- **End to end**: finding the links between survey lines and passes is feasible from the images
  alone, at about 21 s + 25 min for 1,244 images, with a false-link rate of 1 in 982 on
  non-overlapping candidates.
- Limits: one dataset; one setting of the descriptor; the 30-image gap and k = 10 are choices;
  wide baselines are under-represented among the candidates; throwaway scripts, nothing in the
  library yet.

## Loop closures in the pipeline (branch `a/loop-verification`, 2026-11-27)

`scripts/overlap/find_loops.py` chains what steps 2 and 3 did by hand: features read from
disk, codebook, one vector per image, far candidates, confirmation by matching, recording.

Decisions:

- **One measure for both modes.** A candidate goes through `measure_pair`, the function the
  sequential mode uses, and gives a `MeasuredPair` (formerly `SequentialOverlap`): the code
  writing to `pairs` and to the displacements file is shared, not copied.
- **A run of its own**, with the module name `overlap-loops`. `select_images.py` reads the
  latest run of module `overlap`: a loop run under the same name would have replaced the
  sequential overlaps it expects.
- **Unconfirmed candidates are stored too**, as unknown: "compared, could not tell" stays
  different from "never compared".
- **Features on disk, with the image size.** Comparing two images needs their frames, and the
  cache did not hold them. Files written before are completed at first use by reading the
  image again, without extracting.
- **One cache folder per image folder and extractor setting**, since the cache is keyed by file
  name. Two files with the same name in one run are refused. The Mermaid cache made before
  this script is `outputs/cache/sift_defaults`: pass it with `--cache`.
- **Two readings of the features**: a sample of each image for the codebook, then the
  vectors. The descriptors of 1,244 images do not fit in memory together.

First full run on Mermaid (run 31, 2026-11-27, 20 min, features already on disk), judged
against the poses by a one-off check that is not in the repository:

| | |
|---|---|
| Candidate pairs (10 per image, more than 30 images apart) | 10,916 |
| Confirmed | 7,154 |
| Confirmed pairs whose footprints do not overlap at all (false links) | 0 |
| Unconfirmed pairs whose footprints do not overlap | 982 |
| Unconfirmed pairs that are true revisits (mutual overlap of 0.5 or more), missed | 546 |
| Images linked to a far image | 1,188 of 1,244 (95 %) |
| Mutual overlap of a confirmed pair against the reference | off by 0.056 in median |
| … off by more than 0.2 | 164 pairs (2.3 %), 6 % of those with fewer than 30 inliers |
| … off by more than 0.4 | 5 pairs |
| Angle between the two views of a confirmed pair (reference) | 17° in median, 74 % at 10° or more |

This is the first measure after the fix of the inlier spread. The pair 320–831 (two identical
target plates) is proposed again and no longer confirmed. A link can be right while its overlap
is wrong: images 0 and 65 are given 0.95 where the reference says 0.17.

## A selection by support, and what loop closures add to it (branch `a/support-selection`, 2026-11-27)

Question: with the loop closures stored, can more images be removed for the same quality?
Answer on Mermaid: no. A new rule does better than the overlap rule on the rim of the area,
and it owes little to the loop closures.

The rule (`overlap/support_selection.py`, `select_images.py --min-views`): an image may be
removed when it, and every image that overlaps it by 0.5, stays seen by `min_views` kept
images, two of them 10° apart when angles are known; the kept images around it must still
overlap by 0.3. Judged against the poses, for comparable numbers of images:

| Rule | Kept | Inside | Rim | Depth noise (median) |
|---|---|---|---|---|
| support, 8 views, with loops | 849 | 100 % | 91 % | ×1.17 |
| overlap ≥ 0.80 | 859 | 99.9 % | 72 % | ×1.22 |
| support, 5 views, with loops | 606 | 99.8 % | 74 % | ×1.42 |
| 1 image in 2 | 622 | 100 % | 66 % | ×1.42 |
| overlap ≥ 0.70 | 562 | 99.5 % | 48 % | ×1.56 |
| support, 3 views, with loops | 439 | 99.4 % | 61 % | ×1.64 |
| 1 image in 3 | 415 | 99.8 % | 51 % | ×1.75 |
| overlap ≥ 0.60 | 397 | 98.3 % | 30 % | ×1.84 |

- **Depth noise still follows the number of images.** To stay at the noise of the overlap rule
  at 0.80, about as many images must be kept. No removal comes for free.
- **For the same number of images the rim is better kept**, by 15 to 20 points: the rule
  thins the middle of the area, seen by many images, and spares the ends of the lines.
- **The gain is the form of the rule, not the loops.** With the neighbours in time only, 5
  views keep 799 images for 80 % of the rim and ×1.23; with the loops, 6 views keep 699 for
  80 % and ×1.31: about the same curve. What the loops may add is a kept set that
  holds together: around 550 images, it is in 6 groups without them (566 images) and in 1 or
  2 with them (606 and 520 images), groups being judged from reference overlaps of 0.5. At
  439 images it is in 6 groups with them too.

Reservations: the rule uses the criterion of the judge (number of views, 10°), so part of its
advantage is by construction; one dataset; nothing tuned; differences of 0.05 in noise are not
to be read. The kept sequence chains less well in time than with the overlap rule (44
consecutive kept images below 0.5 of reference overlap at 606 images, against 15). The
comparison table came from a one-off script; `select_images.py --min-views 5 --loop-run 31
--angles … --mermaid-poses …` reproduces its lines one at a time.

Decision: both rules stay in `select_images.py`. What neither the judge nor more rules can
settle is whether the model of quality is right: next is a real reconstruction of the full set
and of reduced sets.

Also on 2026-11-27: the project is down to one person and to the overlap engine, loop closures
and 3D reconstruction. The unused `cleaning/` package moved to `archive/`.
