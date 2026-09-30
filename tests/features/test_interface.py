import numpy as np
import pytest

from dataset_automation.features.interface import Features, Matches


def make_features(count: int, descriptor_size: int = 128) -> Features:
    return Features(
        keypoints=np.zeros((count, 2)),
        descriptors=np.zeros((count, descriptor_size)),
        scores=np.zeros(count),
    )


def test_features_accept_consistent_arrays() -> None:
    assert len(make_features(5)) == 5


def test_features_accept_an_image_without_keypoints() -> None:
    # Textureless images must yield empty features, not an error.
    assert len(make_features(0)) == 0


def test_features_reject_keypoints_that_are_not_xy_pairs() -> None:
    with pytest.raises(ValueError, match="keypoints"):
        Features(
            keypoints=np.zeros((5, 3)), descriptors=np.zeros((5, 8)), scores=np.zeros(5)
        )


def test_features_reject_descriptor_count_mismatch() -> None:
    with pytest.raises(ValueError, match="descriptors"):
        Features(
            keypoints=np.zeros((5, 2)), descriptors=np.zeros((4, 8)), scores=np.zeros(5)
        )


def test_features_reject_score_count_mismatch() -> None:
    with pytest.raises(ValueError, match="scores"):
        Features(
            keypoints=np.zeros((5, 2)), descriptors=np.zeros((5, 8)), scores=np.zeros(4)
        )


def test_matches_accept_integer_index_pairs() -> None:
    matches = Matches(
        index_pairs=np.array([[0, 3], [2, 1]]), scores=np.array([0.9, 0.4])
    )

    assert len(matches) == 2


def test_matches_reject_float_indices() -> None:
    with pytest.raises(ValueError, match="integers"):
        Matches(index_pairs=np.array([[0.0, 3.0]]), scores=np.array([0.9]))


def test_matches_reject_score_count_mismatch() -> None:
    with pytest.raises(ValueError, match="scores"):
        Matches(index_pairs=np.array([[0, 3], [2, 1]]), scores=np.array([0.9]))
