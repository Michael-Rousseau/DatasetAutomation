# Person B — AI cleaning and tracking metrics

**Assigned to:** 
**Blocks:** VI (masks of unwanted elements), II (tracking metrics)
**Profile:** AI / segmentation, results analysis

You have two complementary workstreams. Cleaning starts on day one, without depending on anyone. Tracking builds on A's matches. And the two meet: good masks should reduce false tracks, which gives a concrete measure of how useful your cleaning is.

---

## 1. Mission

1. Detect and mask unwanted elements in images: fins, divers, robot parts, water surface, sky.
2. Measure how well points are tracked across a sequence, and compare several detectors.

---

## Workstream 1 — Mask-based cleaning (block VI)

### Is it easy with AI?

Partly. Modern segmentation models are powerful, but:
- **SAM2** segments very well, but doesn't know *what* to segment: you have to give it a point or a box.
- **Open-vocabulary** models (for example Grounding DINO, which takes a text prompt such as "diver fin") can find objects on their own, then SAM2 refines the outline. But they have mostly seen terrestrial images: on a blurry fin in green water, their reliability is uncertain.

**Strategy: try zero-shot first, fine-tune only if needed.**

### Step 1 — Inventory

- Go through Mermaid and AQUALOC: which unwanted elements actually appear, and how often?
- If a type of element never appears in your data, don't handle it.

### Step 2 — Zero-shot approach

- Grounding DINO (text prompts) → boxes → SAM2 → masks.
- Try several prompt phrasings per class.
- Surface and sky: a simple rule may be enough (top area of the image, very bright, little texture), to compare against the model.

### Step 3 — Evaluation

- Manually annotate a small test set (50 to 100 images) with CVAT or Label Studio.
- Measure **IoU** (overlap between predicted and true mask) per class.

### Step 4 — Fine-tuning (if zero-shot isn't enough)

- Annotate a few hundred additional images.
- Fine-tune a lightweight segmentation model (for example Ultralytics YOLO-seg or SegFormer).
- Compare IoU again.

### Output format

- One PNG mask per image, stored separately: **the original image is never modified**.
- Use COLMAP's convention so that C can use them directly: black areas (value 0) = ignored areas. Check the COLMAP documentation for the exact expected mask file naming.
- Record in the `masks` table: image, class, path, method.

---

## Workstream 2 — Tracking metrics (block II)

### The principle

A **track** is the sequence of observations of the same physical point across several consecutive images. If point `p` in image 1 is matched to point `q` in image 2, itself matched to point `r` in image 3, then (p, q, r) is a track of length 3.

### Step 1 — Build the tracks

- Start from the pairwise matches provided by A's engine.
- Chain them with a **union-find** structure over (image, keypoint) pairs.
- Remove inconsistent tracks: a track containing two different points from the same image is necessarily wrong.

### Step 2 — Compute the metrics

- Mean and median track length, and their distribution
- Proportion of points tracked over at least 3 images
- Number of active tracks per image along the sequence: a sudden drop signals a problem (fast motion, turbid water, textureless seafloor)
- Compute time per image

### Step 3 — Compare detectors

Add behind A's feature interface:
- **ORB** (OpenCV): very fast, less robust
- **SuperPoint + LightGlue** (`lightglue` library): learned, on GPU

Compare all three (SIFT, ORB, SuperPoint) on the same sequences. An interesting question: does the advantage of learned methods, well established on terrestrial images, hold underwater?

### Step 4 — Connect the two workstreams

Recompute the tracks **with and without masks**. If the masks remove fins and the surface, false tracks on these moving objects should disappear. That's quantified proof of the value of cleaning.

### Validation

COLMAP also computes its own tracks. On a small subset, compare your results with COLMAP's.

---

## 3. Datasets

| Dataset | Purpose |
|---|---|
| Mermaid | Divers, so possible fins; relief; natural light |
| AQUALOC | Continuous ROV sequences, possible robot parts, shallow to deep |

To check in step 1: whether unwanted elements actually appear in these two datasets.

---

## 4. Interfaces with the others

| You provide | To whom | When |
|---|---|---|
| Masks | A (features), C (COLMAP) | Phase 1 (zero-shot), phase 2 (fine-tuned) |
| ORB and SuperPoint in the feature interface | A | Phase 2 |
| Tracking metrics per sequence | Everyone (report, defense) | Phase 2 |

| You receive | From whom |
|---|---|
| Feature interface, pairwise matches | A |
| `images` table, file reading | C |

---

## 5. Risks

- **Zero-shot not good enough**: plan annotation time from the start, even if we hope not to need it.
- **No unwanted elements in the data**: workstream 1 shrinks. Shift the effort to the detector comparison.
- **Dependency on A for tracking**: while waiting for the engine, you can match consecutive pairs yourself with SIFT to prototype.
- **GPU**: Grounding DINO, SAM2 and SuperPoint need one. To confirm with the supervisor.

---

## 6. Done when…

- [ ] The inventory of unwanted elements is done on Mermaid and AQUALOC
- [ ] Zero-shot masks are produced and evaluated (IoU per class)
- [ ] If needed, a fine-tuned model outperforms zero-shot
- [ ] Tracks are built from A's matches
- [ ] Tracking metrics are computed on several sequences
- [ ] ORB, SIFT and SuperPoint are compared on the same sequences
- [ ] The effect of masks on tracks is quantified
