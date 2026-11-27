from pathlib import Path

import numpy as np
import pytest

from dataset_automation.features.sift import SiftExtractor
from dataset_automation.features.vlad import (
    Codebook,
    load_codebook,
    root_sift,
    save_codebook,
    train_codebook,
    vlad,
)
from tests.features.textures import textured_image

EXTRACTOR = SiftExtractor(scale=1.0)
WORD_COUNT = 8


def sift_descriptors(image: np.ndarray) -> np.ndarray:
    return root_sift(EXTRACTOR.extract(image).descriptors)


def shifted(image: np.ndarray, columns: int) -> np.ndarray:
    return image[:, columns : columns + 320]


@pytest.fixture(scope="module")
def scenes() -> tuple[np.ndarray, np.ndarray]:
    return textured_image(240, 400, seed=1), textured_image(240, 400, seed=2)


@pytest.fixture(scope="module")
def codebook(scenes: tuple[np.ndarray, np.ndarray]) -> Codebook:
    sample = np.vstack([sift_descriptors(scene) for scene in scenes])
    return train_codebook(sample, WORD_COUNT)


def test_root_sift_rows_have_unit_length() -> None:
    descriptors = np.array([[1.0, 3.0, 0.0, 12.0], [5.0, 5.0, 5.0, 5.0]])

    rooted = root_sift(descriptors)

    assert np.linalg.norm(rooted, axis=1) == pytest.approx([1.0, 1.0])
    assert rooted[1] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_vector_has_unit_length_and_one_block_per_word(
    scenes: tuple[np.ndarray, np.ndarray], codebook: Codebook
) -> None:
    vector = vlad(sift_descriptors(scenes[0]), codebook)

    assert vector.shape == (WORD_COUNT * 128,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_two_views_of_a_scene_resemble_more_than_two_scenes(
    scenes: tuple[np.ndarray, np.ndarray], codebook: Codebook
) -> None:
    first, second = scenes
    view = vlad(sift_descriptors(shifted(first, 0)), codebook)
    same_scene = vlad(sift_descriptors(shifted(first, 60)), codebook)
    other_scene = vlad(sift_descriptors(shifted(second, 0)), codebook)

    assert view @ same_scene > view @ other_scene + 0.2


def test_vector_does_not_depend_on_the_order_of_the_descriptors(
    scenes: tuple[np.ndarray, np.ndarray], codebook: Codebook
) -> None:
    descriptors = sift_descriptors(scenes[0])

    assert vlad(descriptors[::-1], codebook) == pytest.approx(
        vlad(descriptors, codebook), abs=1e-5
    )


def test_image_without_descriptors_resembles_nothing(codebook: Codebook) -> None:
    vector = vlad(np.empty((0, 128), dtype=np.float32), codebook)

    assert vector.shape == (WORD_COUNT * 128,)
    assert not vector.any()


def test_same_descriptors_and_seed_give_the_same_codebook(
    scenes: tuple[np.ndarray, np.ndarray],
) -> None:
    sample = sift_descriptors(scenes[0])

    first = train_codebook(sample, WORD_COUNT, seed=3)
    second = train_codebook(sample, WORD_COUNT, seed=3)

    assert np.array_equal(first.words, second.words)


def test_rejects_fewer_descriptors_than_words() -> None:
    with pytest.raises(ValueError):
        train_codebook(np.zeros((3, 128), dtype=np.float32), word_count=8)


def test_codebook_read_back_gives_the_same_vectors(
    scenes: tuple[np.ndarray, np.ndarray], codebook: Codebook, tmp_path: Path
) -> None:
    descriptors = sift_descriptors(scenes[0])

    save_codebook(tmp_path / "run" / "codebook.npy", codebook)
    read_back = load_codebook(tmp_path / "run" / "codebook.npy")

    np.testing.assert_array_equal(
        vlad(descriptors, read_back), vlad(descriptors, codebook)
    )
