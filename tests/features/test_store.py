from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset_automation.features.interface import Features
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.features.store import (
    load_features,
    load_or_extract,
    save_features,
)
from tests.features.textures import textured_image


class CountingExtractor:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0
        self.sift = SiftExtractor(scale=1.0)

    def extract(self, image: np.ndarray, mask: np.ndarray | None = None) -> Features:
        self.calls += 1
        return self.sift.extract(image, mask)


def write_image(path: Path) -> Path:
    cv2.imwrite(str(path), textured_image(120, 160))
    return path


def test_features_come_back_as_they_were_saved(tmp_path: Path) -> None:
    features = Features(
        keypoints=np.array([[1.5, 2.5], [3.0, 4.0]]),
        descriptors=np.arange(8, dtype=np.float32).reshape(2, 4),
        scores=np.array([0.9, 0.1]),
    )

    save_features(tmp_path / "nested" / "image.npz", features)
    loaded = load_features(tmp_path / "nested" / "image.npz")

    np.testing.assert_array_equal(loaded.keypoints, features.keypoints)
    np.testing.assert_array_equal(loaded.descriptors, features.descriptors)
    np.testing.assert_array_equal(loaded.scores, features.scores)
    assert loaded.descriptors.dtype == np.float32


def test_an_image_is_extracted_only_once(tmp_path: Path) -> None:
    image_path = write_image(tmp_path / "dive_0001.png")
    extractor = CountingExtractor()

    first = load_or_extract(image_path, tmp_path / "cache", extractor)
    second = load_or_extract(image_path, tmp_path / "cache", extractor)

    assert extractor.calls == 1
    assert len(first) > 0
    np.testing.assert_array_equal(second.keypoints, first.keypoints)


def test_unreadable_image_is_an_error_and_leaves_no_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(OSError):
        load_or_extract(image_path, tmp_path / "cache", CountingExtractor())

    assert not (tmp_path / "cache" / "broken.npz").exists()
