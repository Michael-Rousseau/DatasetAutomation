"""Shared database schema (docs/00_project.md, section 5).

Changing this module affects persons A, B and C: it needs a review from all three, and any
change to existing tables must bump SCHEMA_VERSION.

Every result row points to the `runs` row that produced it: that is the traceability rule.
Overlap values are NULL when unknown (e.g. too few inliers): never store an invented value.
"""

SCHEMA_VERSION = 1

SCHEMA_SQL = f"""
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    module TEXT NOT NULL,
    code_version TEXT NOT NULL,
    parameters TEXT NOT NULL CHECK (json_valid(parameters)),
    started_at TEXT NOT NULL  -- ISO 8601, UTC
);

CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    -- Position of the image inside `path`: 0 for image files, message index for ROS bags.
    frame_index INTEGER NOT NULL DEFAULT 0 CHECK (frame_index >= 0),
    sequence TEXT,
    timestamp_s REAL,  -- seconds since the Unix epoch, UTC; NULL when unknown
    source TEXT NOT NULL CHECK (source IN ('jpeg', 'png', 'raw', 'rosbag')),
    metadata TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(metadata)),
    run_id INTEGER NOT NULL REFERENCES runs (id),
    UNIQUE (path, frame_index)
);

CREATE TABLE features (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images (id),
    detector TEXT NOT NULL,
    keypoint_count INTEGER NOT NULL CHECK (keypoint_count >= 0),
    keypoints_path TEXT NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs (id),
    UNIQUE (image_id, detector, run_id)
);

CREATE TABLE pairs (
    id INTEGER PRIMARY KEY,
    image_a_id INTEGER NOT NULL REFERENCES images (id),
    image_b_id INTEGER NOT NULL REFERENCES images (id),
    method TEXT NOT NULL,
    inlier_count INTEGER CHECK (inlier_count >= 0),
    overlap_a_in_b REAL CHECK (overlap_a_in_b BETWEEN 0 AND 1),
    overlap_b_in_a REAL CHECK (overlap_b_in_a BETWEEN 0 AND 1),
    iou REAL CHECK (iou BETWEEN 0 AND 1),
    run_id INTEGER NOT NULL REFERENCES runs (id),
    -- One row per unordered pair: the directional columns carry the asymmetry.
    CHECK (image_a_id < image_b_id),
    UNIQUE (image_a_id, image_b_id, method, run_id)
);
CREATE INDEX pairs_image_b ON pairs (image_b_id);

CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    detector TEXT NOT NULL,
    length INTEGER NOT NULL CHECK (length >= 2),
    run_id INTEGER NOT NULL REFERENCES runs (id)
);

CREATE TABLE track_observations (
    track_id INTEGER NOT NULL REFERENCES tracks (id),
    image_id INTEGER NOT NULL REFERENCES images (id),
    keypoint_index INTEGER NOT NULL CHECK (keypoint_index >= 0),
    -- A physical point is seen at most once per image; two keypoints means a broken track.
    PRIMARY KEY (track_id, image_id)
);
CREATE INDEX track_observations_image ON track_observations (image_id);

CREATE TABLE masks (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images (id),
    class_name TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    method TEXT NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs (id),
    UNIQUE (image_id, class_name, run_id)
);

CREATE TABLE targets (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images (id),
    marker_id TEXT NOT NULL,
    corners_px TEXT NOT NULL CHECK (json_valid(corners_px)),  -- [[x, y], ...] in pixels
    gsd_m_per_px REAL CHECK (gsd_m_per_px > 0),
    run_id INTEGER NOT NULL REFERENCES runs (id)
);
CREATE INDEX targets_image ON targets (image_id);

PRAGMA user_version = {SCHEMA_VERSION};
"""
