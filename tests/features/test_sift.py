import numpy as np
import pytest

from dataset_automation.features.interface import FeatureExtractor
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from tests.features.textures import textured_image


def test_sift_extractor_satisfies_the_shared_interface() -> None:
    extractor: FeatureExtractor = SiftExtractor(max_keypoints=0, scale=1.0)

    assert extractor.name == "sift"


def test_finds_keypoints_on_textured_image() -> None:
    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(
        textured_image(240, 320)
    )

    assert len(features) > 100
    assert features.descriptors.shape == (len(features), 128)


def test_accepts_bgr_images() -> None:
    gray = textured_image(240, 320)
    bgr = np.dstack([gray, gray, gray])

    assert len(SiftExtractor(max_keypoints=0, scale=1.0).extract(bgr)) > 100


def test_returns_empty_features_on_uniform_image() -> None:
    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(
        np.full((240, 320), 128, dtype=np.uint8)
    )

    assert len(features) == 0
    assert features.descriptors.shape == (0, 128)


def test_caps_keypoint_count() -> None:
    features = SiftExtractor(max_keypoints=50, scale=1.0).extract(
        textured_image(240, 320)
    )

    assert 0 < len(features) <= 50


def test_ignores_masked_out_region() -> None:
    image = textured_image(240, 320)
    mask = np.full(image.shape, 255, dtype=np.uint8)
    mask[:, :160] = 0

    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(image, mask)

    assert len(features) > 0
    assert features.keypoints[:, 0].min() >= 159


def test_rejects_mask_of_wrong_shape() -> None:
    with pytest.raises(ValueError, match="mask"):
        SiftExtractor(max_keypoints=0, scale=1.0).extract(
            textured_image(240, 320), np.ones((100, 100), dtype=np.uint8)
        )


def test_rejects_non_8_bit_image() -> None:
    with pytest.raises(ValueError, match="8-bit"):
        SiftExtractor(max_keypoints=0, scale=1.0).extract(
            np.zeros((240, 320), dtype=np.uint16)
        )


def test_rejects_scale_above_one() -> None:
    with pytest.raises(ValueError, match="scale"):
        SiftExtractor(max_keypoints=0, scale=2.0)


def test_recovers_known_translation_at_full_resolution_when_downscaled() -> None:
    shift_x, shift_y = 21, 13
    scene = textured_image(520, 700)
    image_a = scene[:480, :640]
    image_b = scene[shift_y : shift_y + 480, shift_x : shift_x + 640]
    extractor = SiftExtractor(max_keypoints=0, scale=0.5)
    features_a, features_b = extractor.extract(image_a), extractor.extract(image_b)

    matches = RatioTestMatcher().match(features_a, features_b)

    # A scene point at x in A appears at x − shift in B.
    displacements = (
        features_b.keypoints[matches.index_pairs[:, 1]]
        - features_a.keypoints[matches.index_pairs[:, 0]]
    )
    assert len(matches) > 50
    np.testing.assert_allclose(
        np.median(displacements, axis=0), [-shift_x, -shift_y], atol=0.5
    )


def test_downscaled_keypoints_have_no_half_pixel_bias() -> None:
    image = textured_image(480, 640)
    full = SiftExtractor(max_keypoints=0, scale=1.0).extract(image)
    half = SiftExtractor(max_keypoints=0, scale=0.5).extract(image)

    matches = RatioTestMatcher().match(half, full)

    # Without the pixel-centre correction the median offset would be 0.5 px on each axis.
    offsets = (
        half.keypoints[matches.index_pairs[:, 0]]
        - full.keypoints[matches.index_pairs[:, 1]]
    )
    assert len(matches) > 50
    np.testing.assert_allclose(np.median(offsets, axis=0), [0.0, 0.0], atol=0.2)


def test_locates_blob_centre_without_bias() -> None:
    # A single Gaussian blob with a known centre: SIFT must put its keypoint there.
    rows, columns = np.mgrid[0:160, 0:200].astype(float)
    centre_x, centre_y = 100.0, 80.0
    blob = 128 + 100 * np.exp(
        -((columns - centre_x) ** 2 + (rows - centre_y) ** 2) / (2 * 4.0**2)
    )

    features = SiftExtractor(max_keypoints=0, scale=1.0).extract(blob.astype(np.uint8))

    nearest = features.keypoints[
        np.argmin(np.linalg.norm(features.keypoints - [centre_x, centre_y], axis=1))
    ]
    np.testing.assert_allclose(nearest, [centre_x, centre_y], atol=0.05)


def test_low_contrast_threshold_finds_keypoints_on_faint_texture() -> None:
    # Underwater sand: texture is present but faint, which OpenCV's default threshold discards.
    faint = (128 + (textured_image(240, 320).astype(float) - 128) * 0.1).astype(
        np.uint8
    )

    opencv_default = SiftExtractor(
        max_keypoints=0, scale=1.0, contrast_threshold=0.04
    ).extract(faint)
    underwater = SiftExtractor(max_keypoints=0, scale=1.0).extract(faint)

    assert len(underwater) > 5 * max(len(opencv_default), 1)


def test_rejects_non_positive_contrast_threshold() -> None:
    with pytest.raises(ValueError, match="contrast_threshold"):
        SiftExtractor(contrast_threshold=0.0)
