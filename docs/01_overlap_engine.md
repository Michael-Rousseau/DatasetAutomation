# Person A — Overlap engine

**Assigned to:** 
**Blocks:** I (overlap), IV (loop closure, optional), deduplication, ~80% extraction
**Profile:** geometry, algorithms

You are building the central component of the project. Dedup, image extraction, loops and B's tracking all depend on your engine. Your number one priority is therefore to deliver **a simple version that works, early**, and improve it afterwards.

---

## 1. Mission

For two images A and B, measure what proportion of A appears in B. From there, build the overlap graph of an entire sequence and derive three services from it: deduplication, extraction of a subset at ~80% overlap, and the optimization curve.

---

## 2. Steps

### Step 1 — Feature layer (shared with B)

Define a common interface for all detectors, for example:

```python
def extract(image: np.ndarray, mask: np.ndarray | None = None) -> Features:
    """Returns keypoints (N×2), descriptors (N×D), scores."""


def match(fa: Features, fb: Features) -> Matches:
    """Returns matched index pairs and a score per pair."""
```

- Implement **SIFT** first (OpenCV): robust, well documented, no GPU needed.
- The `mask` parameter lets you use B's masks as soon as they exist.
- B will add ORB and SuperPoint + LightGlue behind the same interface.

**Deliver early**: B and C depend on it.

### Step 2 — Overlap between two images

1. Match the keypoints of A and B.
2. Filter out wrong matches with RANSAC: `cv2.findHomography(..., cv2.USAC_MAGSAC)`.
3. Compute the overlap ratio. Two methods to compare:
   - **Homography-based**: project the 4 corners of A into B, compute the area of the intersection of the two polygons (`shapely` library) divided by the area of A. Accurate if the scene is planar or distant.
   - **Convex hull of inliers**: the area of the hull of correctly matched points, divided by the image area. Less accurate, but more robust when there is relief.
4. Also return a **confidence indicator** (number of inliers, inlier / match ratio). If there are too few points: report "unknown overlap" rather than a wrong value.

### Step 3 — Sequential mode

- Compare each image with its k neighbours (configurable k, e.g. 5).
- Linear cost: scales well.
- Write each result to the `pairs` table.
- Build the graph with `networkx`.

### Step 4 — Deduplication

- User parameter: overlap threshold (e.g. 95%).
- Walk through the sequence: if an image overlaps the last kept image beyond the threshold, it is flagged as a near-duplicate.
- **Non-destructive**: we flag, we don't delete anything.
- Optional: an ultra-fast pre-filter using perceptual hashing (`imagehash` library) for near-identical copies, before matching.

### Step 5 — ~80% extraction

- Greedy pass: keep an image, then move forward to the last image whose overlap with the kept image is still ≥ 80%, keep it, and repeat.
- Configurable threshold.
- Output: the list of kept images, directly usable by COLMAP (C, phase 2).

### Step 6 — Optimization curve

- Vary the extraction threshold (95%, 90%, 80%, 70%, 60%…).
- For each threshold: % of images kept and coverage preserved.
- **Measuring coverage** is the tricky part.
  - On Mermaid and Eiffel Tower, the reference poses let you compute the actual ground area covered: that's your ground truth.
  - Without poses: approximate it as the share of pixels of each removed image that remains visible in a kept image, via homographies.
- Deliverable: the "% of images removed" versus "% of coverage lost" curve. It's a key figure for the defense.

### Step 7 (optional) — n → n mode and loop closures

Comparing all pairs is impossible at scale (1 million images ≈ 500 billion pairs). Two-stage procedure:
1. **Pre-selection**: one global descriptor per image (DINOv2, NetVLAD or MegaLoc like UnderLoc), then nearest-neighbour search with `faiss`.
2. **Verification**: precise matching (step 2) only on the top-k candidates.

A non-consecutive pair with good overlap = a loop closure. Reminder: revisits are **not** duplicates, we keep them as links.

---

## 3. Datasets

| Dataset | Purpose |
|---|---|
| AQUALOC | Development: continuous sequences, high overlap |
| Mermaid | Coverage ground truth (XML poses), relief (statue) |
| Eiffel Tower | Coverage ground truth, very deep |
| UnderLoc / SQUIDLE+ VPR | n → n option: known revisits |

---

## 4. Evaluation

- **Overlap accuracy**: compare your estimate with the overlap computed from reference poses (Mermaid, Eiffel Tower).
- **Dedup**: on a manually annotated sample, true and false duplicate rates.
- **Extraction**: check that COLMAP (C) reconstructs correctly from your subset.
- **Performance**: images per second, in sequential and n → n modes.
- **n → n option**: compare your detected loops with the UnderLoc baseline.

---

## 5. Interfaces with the others

| You provide | To whom | When |
|---|---|---|
| Feature interface + SIFT | B, C | Very start of phase 1 |
| Pairwise matches | B (tracking) | Phase 1 |
| 80% subset | C (COLMAP) | Phase 2 |

| You receive | From whom |
|---|---|
| `images` table, file reading | C |
| Masks | B |

---

## 6. Risks

- **Textureless seafloor**: few keypoints, wrong homography. Always return the confidence, never an invented value.
- **Strong relief**: the homography assumes a planar scene. Compare with the convex hull method on Mermaid (statue).
- **Compute time**: profile before optimizing. If matching is the bottleneck, SuperPoint + LightGlue on GPU may be faster than SIFT on CPU; measure it.

---

## 7. Done when…

- [ ] Sequential overlap runs on a full AQUALOC sequence
- [ ] Overlap error is measured on Mermaid against the reference poses
- [ ] Threshold-based dedup works and is validated on an annotated sample
- [ ] 80% extraction produces a subset that COLMAP reconstructs
- [ ] The optimization curve is produced on at least two datasets
- [ ] (Optional) n → n mode detects loops on SQUIDLE+ VPR
