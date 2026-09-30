from dataclasses import dataclass, field

import cv2
import numpy as np

from dataset_automation.features.interface import Features

DESCRIPTOR_SIZE = 128


@dataclass(frozen=True)
class SiftExtractor:
    """SIFT keypoints, optionally detected on a downscaled copy of the image.

    Keypoints are always returned in full-resolution pixel coordinates, whatever `scale` is.
    `max_keypoints` = 0 keeps every keypoint SIFT finds.

    Defaults measured on 40 Mermaid images spread over the dive (docs/step0.md): OpenCV's
    contrast threshold (0.04) leaves low-contrast sand with ~16 keypoints and ~1 match between
    images 5 apart; 0.01 raises that to ~56 matches. Full resolution adds keypoints but almost
    no matches, and the 8000 cap keeps brute-force matching at ~0.08 s per pair.
    """

    max_keypoints: int = 8000
    scale: float = 0.5
    contrast_threshold: float = 0.01
    name: str = field(default="sift", init=False)

    def __post_init__(self) -> None:
        if not 0 < self.scale <= 1:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if self.max_keypoints < 0:
            raise ValueError(f"max_keypoints must be >= 0, got {self.max_keypoints}")
        if self.contrast_threshold <= 0:
            raise ValueError(
                f"contrast_threshold must be > 0, got {self.contrast_threshold}"
            )

    def extract(self, image: np.ndarray, mask: np.ndarray | None = None) -> Features:
        grayscale = _to_grayscale(image)
        if mask is not None:
            _check_mask(mask, grayscale.shape)

        small_image, small_mask = self._downscale(grayscale, mask)
        # The default upscaling of the first octave biases every keypoint by ~+0.25 px;
        # precise upscaling removes it (measured on blobs with known centres).
        detector = cv2.SIFT.create(
            nfeatures=self.max_keypoints,
            contrastThreshold=self.contrast_threshold,
            enable_precise_upscale=True,
        )
        keypoints, descriptors = detector.detectAndCompute(small_image, small_mask)
        if not keypoints:
            return _no_features()
        # `nfeatures` keeps ties at the cut-off, so it can return a few extra keypoints.
        if self.max_keypoints:
            strongest = np.argsort(
                [-keypoint.response for keypoint in keypoints], kind="stable"
            )[: self.max_keypoints]
            keypoints = [keypoints[i] for i in strongest]
            descriptors = descriptors[strongest]

        small_xy = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float64)
        # Per-axis factors of the actual resize: the rounded output size can differ from `scale`.
        factors = np.array(
            [
                small_image.shape[1] / grayscale.shape[1],
                small_image.shape[0] / grayscale.shape[0],
            ]
        )
        return Features(
            # OpenCV puts pixel centres on integers, so resizing maps x to (x + 0.5)·f − 0.5;
            # inverting without the 0.5 terms would shift every keypoint by half a pixel.
            keypoints=(small_xy + 0.5) / factors - 0.5,
            descriptors=descriptors,
            scores=np.array(
                [keypoint.response for keypoint in keypoints], dtype=np.float64
            ),
        )

    def _downscale(
        self, grayscale: np.ndarray, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self.scale == 1:
            return grayscale, mask
        height, width = grayscale.shape
        size = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        small_image = cv2.resize(grayscale, size, interpolation=cv2.INTER_AREA)
        # Nearest keeps the mask binary: interpolated values would half-mask the borders.
        small_mask = (
            None
            if mask is None
            else cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        )
        return small_image, small_mask


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"SIFT needs an 8-bit image, got {image.dtype}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"expected a grayscale or BGR image, got shape {image.shape}")


def _check_mask(mask: np.ndarray, image_shape: tuple[int, ...]) -> None:
    if mask.dtype != np.uint8 or mask.shape != image_shape:
        raise ValueError(
            f"mask must be uint8 of shape {image_shape}, got {mask.dtype} {mask.shape}"
        )


def _no_features() -> Features:
    return Features(
        keypoints=np.empty((0, 2)),
        descriptors=np.empty((0, DESCRIPTOR_SIZE), dtype=np.float32),
        scores=np.empty(0),
    )
