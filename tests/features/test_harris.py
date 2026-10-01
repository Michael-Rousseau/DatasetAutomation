import cv2
import numpy as np
import pytest

from dataset_automation.features.harris import (
    HarrisSiftExtractor,
    refine_to_response_peak,
)
from dataset_automation.features.interface import FeatureExtractor
from dataset_automation.features.matching import RatioTestMatcher
from tests.features.textures import textured_image


def test_harris_extractor_satisfies_the_shared_interface() -> None:
    extractor: FeatureExtractor = HarrisSiftExtractor()

    assert extractor.name == "harris-sift"


def test_finds_corners_described_by_sift_on_textured_image() -> None:
    features = HarrisSiftExtractor(scale=1.0).extract(textured_image(240, 320))

    assert len(features) > 100
    assert features.descriptors.shape == (len(features), 128)
    assert (features.scores > 0).all()


def test_returns_empty_features_on_uniform_image() -> None:
    features = HarrisSiftExtractor(scale=1.0).extract(
        np.full((240, 320), 128, dtype=np.uint8)
    )

    assert len(features) == 0
    assert features.descriptors.shape == (0, 128)


def test_caps_corner_count() -> None:
    features = HarrisSiftExtractor(max_keypoints=50, scale=1.0).extract(
        textured_image(240, 320)
    )

    assert 0 < len(features) <= 50


def test_ignores_masked_out_region() -> None:
    image = textured_image(240, 320)
    mask = np.full(image.shape, 255, dtype=np.uint8)
    mask[:, :160] = 0

    features = HarrisSiftExtractor(scale=1.0).extract(image, mask)

    # Sub-pixel refinement moves a corner by at most half a pixel.
    assert len(features) > 0
    assert features.keypoints[:, 0].min() >= 159.5


def test_rejects_mask_of_wrong_shape() -> None:
    with pytest.raises(ValueError, match="mask"):
        HarrisSiftExtractor().extract(
            textured_image(240, 320), np.ones((100, 100), dtype=np.uint8)
        )


def test_rejects_quality_level_outside_zero_one() -> None:
    with pytest.raises(ValueError, match="quality_level"):
        HarrisSiftExtractor(quality_level=1.5)


def test_refines_to_the_peak_of_a_sampled_parabola() -> None:
    # Response sampled from a parabola peaking at (5.3, 4.8): the fit must recover the vertex.
    rows, columns = np.mgrid[0:10, 0:10].astype(float)
    response = 100 - (columns - 5.3) ** 2 - (rows - 4.8) ** 2

    refined = refine_to_response_peak(response, np.array([[5, 5]]))

    np.testing.assert_allclose(refined, [[5.3, 4.8]], atol=1e-9)


def test_keeps_integer_position_where_there_is_no_peak() -> None:
    flat = np.zeros((10, 10))

    refined = refine_to_response_peak(flat, np.array([[5, 5], [0, 9]]))

    np.testing.assert_array_equal(refined, [[5.0, 5.0], [0.0, 9.0]])


def test_recovers_sub_pixel_shift_at_full_resolution() -> None:
    shift = np.array([10.4, 6.7])
    image_a = textured_image(480, 640)
    # warpAffine maps output pixel p to input p + shift: a scene point at x appears at x − shift.
    image_b = cv2.warpAffine(
        image_a,
        np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32),
        (640, 480),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
    )
    extractor = HarrisSiftExtractor(scale=1.0)

    features_a, features_b = extractor.extract(image_a), extractor.extract(image_b)
    matches = RatioTestMatcher().match(features_a, features_b)

    errors = np.linalg.norm(
        features_b.keypoints[matches.index_pairs[:, 1]]
        - features_a.keypoints[matches.index_pairs[:, 0]]
        + shift,
        axis=1,
    )
    # Integer corners would give ~0.7 px here.
    assert len(matches) > 1000
    assert np.median(errors) < 0.35


def test_recovers_known_translation_at_full_resolution_when_downscaled() -> None:
    shift_x, shift_y = 21, 13
    scene = textured_image(520, 700)
    image_a = scene[:480, :640]
    image_b = scene[shift_y : shift_y + 480, shift_x : shift_x + 640]
    extractor = HarrisSiftExtractor(scale=0.5)

    features_a, features_b = extractor.extract(image_a), extractor.extract(image_b)
    matches = RatioTestMatcher().match(features_a, features_b)

    displacements = (
        features_b.keypoints[matches.index_pairs[:, 1]]
        - features_a.keypoints[matches.index_pairs[:, 0]]
    )
    assert len(matches) > 50
    np.testing.assert_allclose(
        np.median(displacements, axis=0), [-shift_x, -shift_y], atol=0.5
    )


def test_upright_descriptors_do_not_match_a_strongly_rotated_image() -> None:
    # Documents the known limitation: no orientation, so a 90° turn breaks matching.
    image = textured_image(320, 320)
    extractor = HarrisSiftExtractor(scale=1.0)

    same = RatioTestMatcher().match(
        extractor.extract(image), extractor.extract(image.copy())
    )
    rotated = RatioTestMatcher().match(
        extractor.extract(image), extractor.extract(np.rot90(image).copy())
    )

    assert len(rotated) < 0.1 * len(same)
