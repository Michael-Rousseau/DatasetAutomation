from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

RIGIDITY_TOLERANCE = 1e-9
# Metashape's `orientation` is the EXIF orientation; only "normal" is understood so far.
SUPPORTED_ORIENTATION = "1"


# eq=False: the generated __eq__ would compare numpy arrays element-wise and fail.
@dataclass(frozen=True, eq=False)
class CameraPoses:
    """Poses of N cameras, stored as parallel arrays indexed like `labels`.

    `camera_to_world` maps camera coordinates to world coordinates. That direction is an
    assumption checked by `scripts/check_mermaid_poses.py`. Covariances are NaN for
    cameras that have none in the source file.
    """

    labels: list[str]
    camera_to_world: np.ndarray
    rotation_covariance: np.ndarray
    location_covariance: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def load_camera_poses(path: Path) -> CameraPoses:
    return parse_camera_poses(path.read_text(encoding="utf-8"))


def parse_camera_poses(xml_text: str) -> CameraPoses:
    cameras = ElementTree.fromstring(xml_text).findall("camera")
    labels = [_parse_label(camera) for camera in cameras]
    _check_unique(labels)

    for camera, label in zip(cameras, labels):
        orientation = camera.findtext("orientation")
        if orientation != SUPPORTED_ORIENTATION:
            raise ValueError(f"camera {label}: unsupported orientation {orientation!r}")

    camera_to_world = np.stack(
        [
            _parse_matrix(camera.findtext("transform"), (4, 4), label)
            for camera, label in zip(cameras, labels)
        ]
    )
    _check_rigid(camera_to_world, labels)

    return CameraPoses(
        labels=labels,
        camera_to_world=camera_to_world,
        rotation_covariance=_parse_covariances(cameras, labels, "rotation_covariance"),
        location_covariance=_parse_covariances(cameras, labels, "location_covariance"),
    )


def camera_centres(poses: CameraPoses) -> np.ndarray:
    return poses.camera_to_world[:, :3, 3]


def optical_axes(poses: CameraPoses) -> np.ndarray:
    # The camera looks along its +z axis (Metashape and OpenCV share x right, y down, z forward).
    return poses.camera_to_world[:, :3, 2]


def _parse_label(camera: ElementTree.Element) -> str:
    label = camera.get("label")
    if not label:
        raise ValueError(f"camera id={camera.get('id')!r} has no label")
    return label


def _check_unique(labels: list[str]) -> None:
    if len(set(labels)) != len(labels):
        raise ValueError("camera labels are not unique")


def _parse_matrix(text: str | None, shape: tuple[int, int], label: str) -> np.ndarray:
    if text is None:
        raise ValueError(f"camera {label}: missing matrix")
    values = np.array(text.split(), dtype=np.float64)
    if values.size != shape[0] * shape[1]:
        raise ValueError(
            f"camera {label}: expected {shape[0] * shape[1]} values, got {values.size}"
        )
    return values.reshape(shape)


def _parse_covariances(
    cameras: list[ElementTree.Element], labels: list[str], tag: str
) -> np.ndarray:
    missing = np.full((3, 3), np.nan)
    return np.stack(
        [
            missing
            if (text := camera.findtext(tag)) is None
            else _parse_matrix(text, (3, 3), label)
            for camera, label in zip(cameras, labels)
        ]
    )


def _check_rigid(camera_to_world: np.ndarray, labels: list[str]) -> None:
    rotations = camera_to_world[:, :3, :3]
    orthonormality_error = np.abs(
        np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3)
    ).max(axis=(1, 2))
    determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
    bottom_row_error = np.abs(camera_to_world[:, 3, :] - [0.0, 0.0, 0.0, 1.0]).max(
        axis=1
    )

    invalid = (
        (orthonormality_error > RIGIDITY_TOLERANCE)
        | (determinant_error > RIGIDITY_TOLERANCE)
        | (bottom_row_error > RIGIDITY_TOLERANCE)
    )
    if invalid.any():
        first = int(np.argmax(invalid))
        raise ValueError(f"camera {labels[first]}: transform is not a rigid motion")
