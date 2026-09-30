# Person C — Targets, GSD and data foundation

**Assigned to:** 
**Blocks:** VII (targets and GSD), data ingestion, iFDO export, then III (COLMAP, later)
**Profile:** classical vision, data engineering

You own both ends of the platform: **the input** (reading every format) and **the output** (iFDO export, traceability). In between, you carry the most original block of the project: scaling images to real-world units using targets. In phase 2, both roles come together in COLMAP.

---

## 1. Mission

1. Robustly read JPEG, PNG, RAW and ROS bags, and fill the `images` table.
2. Detect targets in images and compute the GSD.
3. Export the results of all modules in iFDO format, with traceability.
4. Later: run COLMAP on A's subset and scale it using the targets.

---

## Workstream 1 — Ingestion (phases 0 and 1)

This is the first deliverable of the whole project: A and B need it to start.

### Formats

| Format | Library | Note |
|---|---|---|
| JPEG, PNG | OpenCV or Pillow | Also read EXIF metadata |
| RAW | `rawpy` | Keep the raw data, don't apply any default correction |
| ROS bags | `rosbags` | Reads ROS1 and ROS2 without installing ROS. Extract images and their timestamps |

### Rules

- **A corrupted file must never stop processing**: log it and move on.
- **Resume after interruption**: if processing stops, it restarts where it left off.
- Record for each image: ID, path, sequence, timestamp, source, available metadata.
- For AQUALOC: also retrieve IMU and pressure data if present in the bag.

### Robustness test

Run ingestion on a significant part of **BenthicNet**: many heterogeneous sources, large volume. Measure throughput and error count.

---

## Workstream 2 — Targets and GSD (block VII)

### What is a target?

A reference object of known size placed in the scene. "Unique target" suggests **coded markers** with a unique ID (ArUco, AprilTag): each one can be recognized individually. This is the simplest case.

### Step 1 — Find out which targets exist

- **Question to ask the supervisor first**: which targets does the lab use?
- Read the geodetic network PDF report of the Mermaid dataset: are the control points physically marked by targets visible in the images?
- If no data contains targets: print ArUco markers and run tests in a tank.

### Step 2 — Detection

- Coded markers: `cv2.aruco.ArucoDetector` (OpenCV). Returns the ID and the 4 corners in pixels.
- Non-coded targets (checkerboard, scale bar): `cv2.findChessboardCorners` for checkerboards, otherwise a detector to train. More complex: only do it if the lab uses them.
- Record in the `targets` table: image, ID, pixel corners.

### Step 3 — Computing the GSD

**Simple method**: if the real side of the target measures `L` cm and spans `n` pixels, then GSD ≈ `L / n` cm per pixel, **at the target's location**.

**More accurate method**: with the camera calibration (provided for Mermaid), `cv2.solvePnP` estimates the distance `Z` between the camera and the target. For a front-facing view, GSD ≈ `Z / f`, with `f` the focal length in pixels. This method accounts for the target's tilt.

### Limitations to document honestly

- The GSD is only valid **at the target's distance**. An object twice as far has a GSD twice as large.
- Refraction through the housing port distorts the image: use a calibration done underwater.
- A target seen at a steep angle gives a less reliable measurement.

### Validation

The Mermaid dataset states a sub-millimetre GSD and provides a surveyed geodetic network: compare your values with these references.

---

## Workstream 3 — iFDO export and traceability

- Read the shared tables (`images`, `pairs`, `tracks`, `masks`, `targets`) and produce the iFDO file.
- Minimal required fields: iFDO schema version and the filename of each image.
- For our metrics (overlap, GSD, tracks): check in the iFDO schema how to declare additional fields.
- `runs` table: for each processing step, module, code version, parameters and date. Every result must be traceable.

---

## Workstream 4 — COLMAP (block III, phase 2)

1. Take the ~80% subset produced by A (COLMAP can't handle hundreds of thousands of images).
2. Use B's masks to ignore unwanted elements.
3. Run COLMAP with sequential matching (suited to continuous sequences).
4. **Scaling**: a single-camera reconstruction has an arbitrary scale. Targets detected in a few images provide real distances, which allow the whole reconstruction to be rescaled.
5. **Evaluation**: compare the resulting trajectory with Mermaid's reference poses (XML file) using the `evo` tool, after alignment.

---

## 3. Datasets

| Dataset | Purpose |
|---|---|
| Mermaid | Targets or control points, reference GSD, calibration, poses |
| AQUALOC | ROS bag ingestion, IMU, pressure |
| BenthicNet | Ingestion robustness at scale |

---

## 4. Interfaces with the others

| You provide | To whom | When |
|---|---|---|
| Ingestion and `images` table | A, B | Phase 0 (minimal version), phase 1 (all formats) |
| GSD per image | Everyone, iFDO export | Phase 1 |
| iFDO export and `runs` table | Everyone | Phase 2 |
| Scaled COLMAP trajectory | Report, defense | Phase 2 |

| You receive | From whom |
|---|---|
| 80% subset | A |
| Masks | B |

---

## 5. Risks

- **No targets in the data**: this is the main risk. Resolve it in phase 0 with the supervisor.
- **Large ROS bags**: stream them, don't load everything into memory.
- **COLMAP fails underwater** (textureless seafloor, turbid water): it may produce several disconnected reconstructions. Document it rather than forcing it.
- **Workload**: ingestion + targets + export is a lot. Minimal ingestion must ship very early; advanced formats can follow.

---

## 6. Done when…

- [ ] Ingestion reads JPEG, PNG, RAW and ROS bags, without stopping on a corrupted file
- [ ] Ingestion has been tested at scale on BenthicNet
- [ ] Targets are detected and identified
- [ ] The GSD is computed and compared with the Mermaid dataset references
- [ ] The iFDO export produces a valid file with the results of all modules
- [ ] Every result is traceable in the `runs` table
- [ ] (Phase 2) COLMAP reconstructs A's subset at real-world scale
