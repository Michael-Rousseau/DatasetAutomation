# Dataset Automation — Scale Up UW Vision

> *Turn terabytes of images into data.* Develop automated tools to process large underwater datasets: coverage estimation, feature tracking, trajectory extraction, and more.

The team's reference document. It describes what we are building, why, how the parts fit together, and who does what. Each person then has their own detailed sheet.

---

## 1. The problem

An underwater acquisition campaign (ROV, AUV, diver) produces tens of thousands to millions of images. A large share of them is **redundant**: when the camera moves slowly and captures several images per second, two consecutive images overlap almost entirely. Nothing is indexed: we don't know which images overlap, which area has been covered, or at what scale each image was taken.

Public datasets are **finished products**. Tools such as Marimba or Paidiverpy handle metadata, packaging and preprocessing, but they don't perform geometric analysis of the images (overlap, tracking, scale). Deduplication is announced in Marimba but not yet integrated.

**Our positioning: an intelligent analysis layer, not yet another management platform.**

---

## 2. What we are building

A **modular platform** used by the lab: it takes a folder of images as input, and the user chooses which processing steps to apply.

| Item | Decision |
|---|---|
| Users | The lab |
| Inputs | JPEG / PNG, RAW, ROS bags (no video) |
| Output | iFDO-compatible, without depending on Marimba |
| Principle | Non-destructive: the original image is never modified |
| Traceability | Every processing step logs its parameters and version |
| Language | Python first, Rust (PyO3) later on measured hotspots |

---

## 3. Functional blocks

### Kept

**Block I — Overlap engine** *(central component)*
Measure what proportion of one image appears in another, using keypoint matching. Produces an **overlap graph**: each image is a node, each overlap a weighted edge. Works in sequential mode (each image against its k neighbours) and, optionally, in n → n mode.
Sub-features:
- **Extraction at ~80% overlap**: keep the minimal subset of images that preserves enough overlap for photogrammetry.
- **Optimization curve**: % of images removed versus % of coverage lost.
- **Configurable deduplication**: above an overlap threshold chosen by the user (e.g. 95%), an image is a near-duplicate.

**Block IV — Loop closure** *(option of block I)*
A loop closure is an overlap between two non-consecutive images: the camera comes back to a place it has already seen. It is the n → n mode of the engine, with a pre-selection step to avoid comparing every pair.

**Block II — Tracking metrics**
For each physical point in the scene, over how many consecutive images it remains recognized (track length). Comparison of ORB, SIFT and SuperPoint. A quality indicator of a sequence for 3D reconstruction.

**Block VI — Mask-based cleaning**
Detect and mask unwanted elements (fins, divers, robot parts, water surface, sky). Masks are stored separately and can be reused by COLMAP and by tracking to ignore these areas.

**Block VII — Targets and GSD**
Detect images that contain a target (an object of known size, ideally a coded marker with a unique ID) and derive the **GSD** (real size of one pixel, e.g. 0.5 mm/pixel) at the target's location.

### Later

**Block III — Trajectory with COLMAP**
Structure-from-Motion on the subset extracted by block I, scaled using the targets from block VII.

### Set aside

**Block V — Quality metrics** (blur, usability, grey image detection). Set aside for now. Note: the overlap engine provides a close indicator for free, the number of keypoints per image, which already flags textureless images.

---

## 4. How the blocks fit together

```
                 ┌─────────────────────────┐
  Ingestion  ──► │   Feature extraction    │  (SIFT, ORB, SuperPoint…)
 (JPEG, RAW,     └────────────┬────────────┘
  ROS bags)                   │
      │           ┌───────────▼────────────┐
      │           │   I. Overlap engine    │──► Dedup (% threshold)
      │           │     image graph        │──► ~80% extraction
      │           └──┬──────────────────┬──┘──► Optimization curve
      │              │                  │
      │     IV. n → n mode        II. Tracking
      │     (loops)               (chaining matches)
      │
      ├──► VI. Masks ─────────► improve I, II and III
      │
      └──► VII. Targets / GSD ──► III. COLMAP (later): scaling
                                        ▲
                    ~80% subset ────────┘ (from I)
```

Key points:
- **The overlap engine (I) is the central component.** Dedup, extraction, loops and tracking all derive from it.
- **Masks (VI) and targets (VII) are independent**: they can start on day one.
- **The feature layer is shared**: A builds it with a common interface, B adds the detectors to compare.

---

## 5. Shared data

All modules read from and write to the same storage (SQLite or Parquet), with a schema defined together in phase 0.

| Table | Main content | Written by |
|---|---|---|
| `images` | id, path, sequence, timestamp, source, metadata | Ingestion (C) |
| `features` | image, detector, number of keypoints, path to keypoint file | Features (A, B) |
| `pairs` | image_a, image_b, method, number of inliers, overlap ratio | Overlap (A) |
| `tracks` | detector, length, list of (image, keypoint) | Tracking (B) |
| `masks` | image, class, mask path, method | Cleaning (B) |
| `targets` | image, target ID, pixel corners, GSD | Targets (C) |
| `runs` | module, version, parameters, date | Everyone (traceability) |

The iFDO export (C) reads these tables and produces the iFDO file. Minimal required fields: iFDO schema version and the filename of each image. For our metrics, check in the schema how to declare additional fields.

---

## 6. Datasets

| Dataset | Depth | Platform | What it provides | Used by |
|---|---|---|---|---|
| [Mermaid](https://www.seanoe.org/data/00868/97987/) | ~20 m | Diver | Raw images, reference poses (XML), geodetic network, sub-millimetre GSD, calibration. Produced by SEAL/LRE/EPITA. CC BY-NC-ND licence | A, B, C |
| [AQUALOC](https://www.lirmm.fr/aqualoc/) | 5 m, 270 m, 380 m | ROV | Continuous sequences, ROS bags, IMU, pressure, SfM trajectory | A, B, C |
| [Eiffel Tower](https://www.seanoe.org/data/00810/92226/) | 1,700 m | ROV | 4 campaigns, camera poses | A |
| [UnderLoc / SQUIDLE+ VPR](https://github.com/bev-gorry/underloc) | 10 to 885 m | AUV, towed camera | Revisits years apart, relocalization baseline | A (option IV) |
| [BenthicNet](https://www.frdr-dfdr.ca/repo/dataset/24c6c813-d0ff-4461-8174-3f960f1edc0d) | Mixed | Mixed | Very large volume, heterogeneous sources | C (ingestion, scaling) |

BenthicNet mostly contains isolated images from many sources, with few continuous sequences: it is less useful for overlap than for testing ingestion robustness at scale.

---

## 7. Work split

| | Person A | Person B | Person C |
|---|---|---|---|
| Mission | Overlap engine | AI cleaning and tracking | Targets, GSD and data foundation |
| Blocks | I, IV (option), dedup, 80% extraction | VI (masks), II | VII, ingestion, iFDO export, then III |
| Profile | Geometry, algorithms | AI / segmentation, analysis | Classical vision, data engineering |
| Sheet | `person_A_overlap.md` | `person_B_cleaning_tracking.md` | `person_C_targets_foundation.md` |

Assignment: A = Michael · B =  · C = 

---

## 8. Phased plan

The project duration is not known yet: phases need to be mapped to the actual calendar.

**Phase 0 — Scoping and foundation (together)**
- Meeting with the supervisor (questions in section 10)
- Git repository, package structure, data schema, conventions
- Download a small subset of each dataset
- Naive end-to-end pipeline: ingestion → features → sequential overlap → export. Each step can be basic; what matters is that the chain runs.

**Phase 1 — Blocks in parallel**
- A: sequential overlap + threshold-based dedup
- B: zero-shot masks (no training) + first tracking on SIFT
- C: RAW and ROS bag ingestion + target detection and GSD

**Phase 2 — Going deeper**
- A: 80% extraction, optimization curve, n → n option
- B: ORB / SIFT / SuperPoint comparison, mask fine-tuning if needed, impact of masks on tracking
- C: full iFDO export, then COLMAP on A's subset, scaled with the targets

**Phase 3 — Integration and defense (together)**
- Scaling on full datasets, profiling, optimization (PyO3 if useful)
- Documentation for the lab
- Defense preparation

---

## 9. Working rules

- **One module = pure functions**: clear input, clear output, no global state. Makes testing and Rust porting easier.
- **Measure before optimizing**: fixed benchmark set, throughput measured from the first version.
- **Cross reviews**: every pull request is reviewed by someone else, so that everyone can answer questions on every part during the defense.
- **Short syncs** between you several times a week, weekly meeting with the supervisor.

---

## 10. Open questions (for the supervisor)

1. Which lab data should the platform run on first? What volume, what formats?
2. Do the lab's images contain targets? Which ones (ArUco, AprilTag, checkerboard, scale bar)?
3. Is there a GPU machine or server access?
4. Is iFDO relevant if only the lab uses the tool, or would an internal format be enough?
5. Is the refocus on overlap / tracking / targets, with the COLMAP trajectory postponed, acceptable?
6. The Mermaid dataset comes from the lab: are its authors available for questions about the acquisition?

---

## 11. Main risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Textureless seafloor (sand, open water) | Few keypoints, wrong overlap | Detect and flag rather than output a wrong value |
| n → n mode too expensive | Impossible on large volumes | Pre-selection with global descriptors, n → n only as an option |
| Unreliable zero-shot masks | Unusable cleaning | Annotate a few hundred images and fine-tune a model |
| No targets in the data | Block VII has no test data | Check the Mermaid dataset, otherwise print markers and test in a tank |
| B and C blocked on A's engine | Cascading delays | Feature interface fixed in phase 0, minimal version delivered early |
