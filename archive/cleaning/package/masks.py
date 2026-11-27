"""Mask files on disk, in COLMAP's convention so persons A and C can use them unchanged.

Every mask file is a single-channel PNG of the image's size where 0 marks pixels to ignore and
255 pixels to keep. COLMAP looks up the mask of `<image_path>/sub/img.jpg` at
`<mask_path>/sub/img.jpg.png`: the image's relative path plus `.png`.
"""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

IGNORE = 0
KEEP = 255


def keep_mask(unwanted: Iterable[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Merge boolean "unwanted" masks into one uint8 keep mask of `shape` (H, W)."""
    ignored = np.zeros(shape, dtype=bool)
    for mask in unwanted:
        if mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} does not match image {shape}")
        ignored |= mask
    return np.where(ignored, IGNORE, KEEP).astype(np.uint8)


def colmap_mask_path(mask_root: Path, image_path: Path, image_root: Path) -> Path:
    relative = image_path.relative_to(image_root)
    return mask_root / relative.with_name(relative.name + ".png")


def write_mask(path: Path, mask: np.ndarray) -> None:
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError(
            f"mask must be a 2D uint8 array, got {mask.dtype} {mask.shape}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite reports failure by returning False, not by raising.
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"cannot write mask {path}")


def read_unwanted(path: Path) -> np.ndarray:
    """Read a mask file back as a boolean array, True where pixels are ignored."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"cannot read mask {path}")
    return mask == IGNORE
