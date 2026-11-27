"""Text files exchanged with COLMAP: the pairs to match, and the cameras it placed.

Formats are COLMAP's own (https://colmap.github.io/format.html). Images are named by their
path relative to the image folder given to COLMAP.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

import numpy as np

# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
IMAGE_LINE_FIELDS = 10
# POINT3D_ID, X, Y, Z, R, G, B, then the error and the track, which are not read.
POINT_LINE_FIELDS = 7


def write_match_list(file: TextIO, pairs: Iterable[tuple[str, str]]) -> int:
    """One pair of image names per line, the list COLMAP matches instead of every pair.

    Returns the number of pairs written.
    """
    count = 0
    for first, second in pairs:
        file.write(f"{first} {second}\n")
        count += 1
    return count


def read_camera_centres(images_file: Path) -> dict[str, np.ndarray]:
    """Centre of each registered camera of a text model (`images.txt`), by image name.

    COLMAP stores the world-to-camera rotation (a quaternion, w first) and translation: the
    centre is −Rᵀt. Images it could not place are absent from the file.
    """
    centres: dict[str, np.ndarray] = {}
    lines = [
        line
        for line in images_file.read_text().splitlines()
        if not line.startswith("#")
    ]
    # Two lines per image: its pose, then its 2D points, which may be an empty line.
    for line in lines[::2]:
        fields = line.split(maxsplit=IMAGE_LINE_FIELDS - 1)
        if len(fields) != IMAGE_LINE_FIELDS:
            raise ValueError(f"not an image line of a COLMAP text model: {line!r}")
        quaternion = np.array(fields[1:5], dtype=float)
        translation = np.array(fields[5:8], dtype=float)
        centres[fields[9]] = -_rotation(quaternion).T @ translation
    return centres


def read_points(points_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions, (N, 3), and colours, (N, 3) in 0-255, of the 3D points of a text model
    (`points3D.txt`)."""
    positions: list[list[float]] = []
    colours: list[list[int]] = []
    for line in points_file.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=POINT_LINE_FIELDS)
        if len(fields) < POINT_LINE_FIELDS:
            raise ValueError(f"not a point line of a COLMAP text model: {line!r}")
        positions.append([float(value) for value in fields[1:4]])
        colours.append([int(value) for value in fields[4:7]])
    return (
        np.array(positions, dtype=float).reshape(-1, 3),
        np.array(colours, dtype=np.uint8).reshape(-1, 3),
    )


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
