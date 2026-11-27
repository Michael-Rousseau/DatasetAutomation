"""Features on disk, one file per image, so that an image is extracted once.

Changing this module affects persons A, B and C: it needs a review from all three.
"""

from pathlib import Path

import cv2
import numpy as np

from dataset_automation.features.interface import FeatureExtractor, Features


def save_features(path: Path, features: Features) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keypoints=features.keypoints,
        descriptors=features.descriptors,
        scores=features.scores,
    )


def load_features(path: Path) -> Features:
    with np.load(path) as stored:
        return Features(stored["keypoints"], stored["descriptors"], stored["scores"])


def load_or_extract(
    image_path: Path, cache_directory: Path, extractor: FeatureExtractor
) -> Features:
    """Features of `image_path`, read from `cache_directory` or extracted and stored there.

    The cache is keyed by the image's file name only: use one directory per extractor
    setting and per dataset.
    """
    cache_path = cache_directory / f"{image_path.stem}.npz"
    if cache_path.exists():
        return load_features(cache_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    # imread signals an unreadable file by returning None, not by raising.
    if image is None:
        raise OSError(f"cannot read image {image_path}")
    features = extractor.extract(np.asarray(image))
    save_features(cache_path, features)
    return features
