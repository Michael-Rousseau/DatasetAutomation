"""Image preparation shared by the OpenCV-based extractors (SIFT, Harris)."""

from dataclasses import dataclass

import cv2
import numpy as np

from dataset_automation.features.interface import Features

SIFT_DESCRIPTOR_SIZE = 128


@dataclass(frozen=True)
class DetectionImage:
    """The grayscale image and mask a detector runs on, possibly downscaled."""

    image: np.ndarray
    mask: np.ndarray | None
    full_shape: tuple[int, int]

    def to_full_resolution(self, xy: np.ndarray) -> np.ndarray:
        # Per-axis factors of the actual resize: the rounded output size can differ from `scale`.
        factors = np.array(
            [
                self.image.shape[1] / self.full_shape[1],
                self.image.shape[0] / self.full_shape[0],
            ]
        )
        # OpenCV puts pixel centres on integers, so resizing maps x to (x + 0.5)·f − 0.5;
        # inverting without the 0.5 terms would shift every keypoint by half a pixel.
        return (xy + 0.5) / factors - 0.5


def prepare(image: np.ndarray, mask: np.ndarray | None, scale: float) -> DetectionImage:
    grayscale = to_grayscale(image)
    if mask is not None:
        check_mask(mask, grayscale.shape)
    full_shape = (grayscale.shape[0], grayscale.shape[1])
    if scale == 1:
        return DetectionImage(grayscale, mask, full_shape)

    height, width = full_shape
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small_image = cv2.resize(grayscale, size, interpolation=cv2.INTER_AREA)
    # Nearest keeps the mask binary: interpolated values would half-mask the borders.
    small_mask = (
        None
        if mask is None
        else cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    )
    return DetectionImage(small_image, small_mask, full_shape)


def check_scale(scale: float) -> None:
    if not 0 < scale <= 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"feature extraction needs an 8-bit image, got {image.dtype}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"expected a grayscale or BGR image, got shape {image.shape}")


def check_mask(mask: np.ndarray, image_shape: tuple[int, ...]) -> None:
    if mask.dtype != np.uint8 or mask.shape != image_shape:
        raise ValueError(
            f"mask must be uint8 of shape {image_shape}, got {mask.dtype} {mask.shape}"
        )


def no_features(descriptor_size: int = SIFT_DESCRIPTOR_SIZE) -> Features:
    return Features(
        keypoints=np.empty((0, 2)),
        descriptors=np.empty((0, descriptor_size), dtype=np.float32),
        scores=np.empty(0),
    )
