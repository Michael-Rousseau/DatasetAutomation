import numpy as np
import pytest

from dataset_automation.features.interface import FeatureMatcher, Features
from dataset_automation.features.matching import RatioTestMatcher, lowe_ratio_filter
from dataset_automation.features.sift import SiftExtractor
from tests.features.textures import textured_image


def test_ratio_test_matcher_satisfies_the_shared_interface() -> None:
    matcher: FeatureMatcher = RatioTestMatcher()

    assert matcher.name == "ratio-test"


def test_keeps_distinctive_match() -> None:
    keep, scores = lowe_ratio_filter(np.array([1.0]), np.array([4.0]), ratio=0.75)

    assert keep.tolist() == [True]
    assert scores.tolist() == pytest.approx([0.75])


def test_drops_ambiguous_match() -> None:
    keep, scores = lowe_ratio_filter(np.array([0.9]), np.array([1.0]), ratio=0.75)

    assert keep.tolist() == [False]
    assert scores.size == 0


def test_drops_match_when_second_distance_is_zero() -> None:
    keep, _ = lowe_ratio_filter(np.array([0.0]), np.array([0.0]), ratio=0.75)

    assert keep.tolist() == [False]


def test_score_grows_with_distinctiveness() -> None:
    _, scores = lowe_ratio_filter(
        np.array([1.0, 1.0]), np.array([2.0, 10.0]), ratio=0.75
    )

    assert scores[1] > scores[0]


def test_matches_an_image_with_itself_one_to_one() -> None:
    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(
        textured_image(240, 320)
    )

    matches = RatioTestMatcher().match(features, features)

    assert len(matches) > 0.9 * len(features)
    assert (matches.index_pairs[:, 0] == matches.index_pairs[:, 1]).mean() > 0.99


def test_returns_no_matches_when_image_has_no_keypoints() -> None:
    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(
        textured_image(240, 320)
    )
    empty = SiftExtractor(max_keypoints=0, scale=1.0).extract(
        np.full((240, 320), 128, dtype=np.uint8)
    )

    assert len(RatioTestMatcher().match(empty, features)) == 0
    assert len(RatioTestMatcher().match(features, empty)) == 0


def test_returns_no_matches_when_second_image_has_a_single_keypoint() -> None:
    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(
        textured_image(240, 320)
    )
    single = Features(
        keypoints=features.keypoints[:1],
        descriptors=features.descriptors[:1],
        scores=features.scores[:1],
    )

    assert len(RatioTestMatcher().match(features, single)) == 0
