"""Features on disk, one file per image, so that an image is extracted once.

Changing this module affects persons A, B and C: it needs a review from all three.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset_automation.features.interface import FeatureExtractor, Features

ImageSize = tuple[int, int]


@dataclass(frozen=True, eq=False)
class StoredFeatures:
    """`image_size` is (width, height) in pixels: comparing two images needs their frames,
    and reading it back from the image file costs as much as decoding it."""

    features: Features
    image_size: ImageSize


def save_features(path: Path, features: Features, image_size: ImageSize) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keypoints=features.keypoints,
        descriptors=features.descriptors,
        scores=features.scores,
        image_size=np.array(image_size),
    )


def load_features(path: Path) -> StoredFeatures:
    features, image_size = _read(path)
    if image_size is None:
        raise ValueError(f"{path} was written without the image size")
    return StoredFeatures(features, image_size)


def load_or_extract(
    image_path: Path, cache_directory: Path, extractor: FeatureExtractor
) -> StoredFeatures:
    """Features of `image_path`, read from `cache_directory` or extracted and stored there.

    The cache is keyed by the image's file name only: use one directory per extractor
    setting and per dataset.
    """
    cache_path = cache_directory / f"{image_path.stem}.npz"
    if cache_path.exists():
        features, image_size = _read(cache_path)
        if image_size is None:
            # Written before the size was stored: complete the file, keep the features.
            image_size = _size(_pixels(image_path))
            save_features(cache_path, features, image_size)
        return StoredFeatures(features, image_size)
    pixels = _pixels(image_path)
    stored = StoredFeatures(extractor.extract(pixels), _size(pixels))
    save_features(cache_path, stored.features, stored.image_size)
    return stored


def _read(path: Path) -> tuple[Features, ImageSize | None]:
    with np.load(path) as stored:
        features = Features(
            stored["keypoints"], stored["descriptors"], stored["scores"]
        )
        if "image_size" not in stored:
            return features, None
        width, height = stored["image_size"]
        return features, (int(width), int(height))


def _pixels(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # imread signals an unreadable file by returning None, not by raising.
    if image is None:
        raise OSError(f"cannot read image {image_path}")
    return np.asarray(image)


def _size(pixels: np.ndarray) -> ImageSize:
    height, width = pixels.shape
    return width, height
