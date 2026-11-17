import numpy as np
import pytest
from shapely import affinity

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.geometry import image_frame
from dataset_automation.overlap.homography import transform
from dataset_automation.overlap.pairwise import (
    MIN_INLIERS,
    PairOverlapEstimate,
    UnknownReason,
    estimate_homography_overlap,
    estimate_hull_overlap,
)
from tests.overlap.scenes import (
    HEIGHT,
    SMALL_GOPRO,
    WIDTH,
    distorted_view,
    scene,
    view_pair,
)

SIZE = (WIDTH, HEIGHT)
EXTRACTOR = SiftExtractor(scale=1.0)


def homography_estimate(
    image_a: np.ndarray,
    image_b: np.ndarray,
    intrinsics: Intrinsics | None = None,
    min_inliers: int = MIN_INLIERS,
) -> PairOverlapEstimate:
    features_a, features_b = EXTRACTOR.extract(image_a), EXTRACTOR.extract(image_b)
    matches = RatioTestMatcher().match(features_a, features_b)
    return estimate_homography_overlap(
        features_a, features_b, matches, SIZE, SIZE, intrinsics, min_inliers
    )


def exact_overlap(a_to_b: np.ndarray) -> tuple[float, float]:
    frame = image_frame(SIZE, None)
    a_in_b = (
        frame.intersection(transform(np.linalg.inv(a_to_b), frame)).area / frame.area
    )
    b_in_a = frame.intersection(transform(a_to_b, frame)).area / frame.area
    return a_in_b, b_in_a


def translation(shift_x: float, shift_y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]])


def test_half_image_translation_overlaps_by_half() -> None:
    estimate = homography_estimate(*view_pair(translation(-320, 0), scene()))

    assert estimate.is_known
    assert estimate.a_in_b == pytest.approx(0.5, abs=0.01)
    assert estimate.b_in_a == pytest.approx(0.5, abs=0.01)
    assert estimate.iou == pytest.approx(1 / 3, abs=0.01)


@pytest.mark.parametrize(
    "description, a_to_b",
    [
        ("rotation of 30° about the centre", "rotation"),
        ("zoom ×1.5 about the centre", "zoom"),
    ],
)
def test_recovers_exact_overlap_of_a_known_homography(
    description: str, a_to_b: str
) -> None:
    centre = translation(WIDTH / 2, HEIGHT / 2)
    angle = np.radians(30)
    transforms = {
        "rotation": np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        ),
        "zoom": np.diag([1.5, 1.5, 1.0]),
    }
    homography = (
        centre @ transforms[a_to_b] @ np.linalg.inv(centre) @ translation(-60, 40)
    )

    estimate = homography_estimate(*view_pair(homography, scene()))

    assert estimate.is_known, description
    expected_a_in_b, expected_b_in_a = exact_overlap(homography)
    assert estimate.a_in_b == pytest.approx(expected_a_in_b, abs=0.01)
    assert estimate.b_in_a == pytest.approx(expected_b_in_a, abs=0.01)


def test_overlap_is_measured_in_each_image_own_frame() -> None:
    # Zooming in ×2: B sees a quarter of A, while all of B is inside A.
    centre = translation(WIDTH / 2, HEIGHT / 2)

    estimate = homography_estimate(
        *view_pair(centre @ np.diag([2.0, 2.0, 1.0]) @ np.linalg.inv(centre), scene())
    )

    assert estimate.a_in_b == pytest.approx(0.25, abs=0.01)
    assert estimate.b_in_a == pytest.approx(1.0, abs=0.01)


def test_undistortion_recovers_the_overlap_of_distorted_images() -> None:
    world = scene()
    image_a, image_b = distorted_view(world, 0.0), distorted_view(world, 160.0)
    frame = image_frame(SIZE, SMALL_GOPRO)
    # Undistorted, B is A translated by 160 px: A's share in B is |F ∩ (F + 160)| / |F|.
    expected = (
        frame.intersection(affinity.translate(frame, -160.0, 0.0)).area / frame.area
    )

    corrected = homography_estimate(image_a, image_b, intrinsics=SMALL_GOPRO)
    raw = homography_estimate(image_a, image_b)

    assert corrected.a_in_b == pytest.approx(expected, abs=0.01)
    # Ignoring the lens bends straight lines, so a homography fits worse and keeps fewer inliers.
    assert corrected.inlier_count > raw.inlier_count


def test_reports_unknown_overlap_without_texture() -> None:
    flat = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)

    estimate = homography_estimate(flat, flat)

    assert not estimate.is_known
    assert estimate.unknown_reason is UnknownReason.TOO_FEW_MATCHES
    assert (estimate.a_in_b, estimate.b_in_a, estimate.iou) == (None, None, None)


def test_reports_unknown_overlap_with_too_few_inliers_and_keeps_the_count() -> None:
    estimate = homography_estimate(
        *view_pair(translation(-100, 0), scene()), min_inliers=100_000
    )

    assert estimate.unknown_reason is UnknownReason.TOO_FEW_INLIERS
    assert estimate.inlier_count > 0
    assert 0 < estimate.inlier_ratio <= 1


def test_hull_underestimates_when_texture_is_patchy() -> None:
    # Texture only in a central patch: points cover a fraction of the true overlap.
    world = scene()
    patchy = np.full_like(world, 128)
    rows, columns = (
        slice(world.shape[0] // 2 - 120, world.shape[0] // 2 + 120),
        slice(world.shape[1] // 2 - 160, world.shape[1] // 2 + 160),
    )
    patchy[rows, columns] = world[rows, columns]
    image_a, image_b = view_pair(translation(-80, 0), patchy)
    features_a, features_b = EXTRACTOR.extract(image_a), EXTRACTOR.extract(image_b)
    matches = RatioTestMatcher().match(features_a, features_b)

    homography = estimate_homography_overlap(
        features_a, features_b, matches, SIZE, SIZE
    )
    hull = estimate_hull_overlap(features_a, features_b, matches, SIZE, SIZE)

    assert homography.a_in_b == pytest.approx(0.875, abs=0.01)
    assert hull.a_in_b is not None
    assert hull.a_in_b < 0.5
    assert hull.iou is None


def test_refuses_overlap_supported_only_by_a_repeated_object() -> None:
    # Two scenes that share nothing but one identical planar patch, like two GCP plates with the
    # same pattern: the patch alone fits a homography, which claims a large overlap.
    patch = scene(seed=7)[:60, :60]
    image_a = scene(seed=1)[:HEIGHT, :WIDTH].copy()
    image_b = scene(seed=2)[:HEIGHT, :WIDTH].copy()
    image_a[300:360, 200:260] = patch
    image_b[280:340, 230:290] = patch

    estimate = homography_estimate(image_a, image_b, min_inliers=5)

    assert estimate.unknown_reason is UnknownReason.INLIERS_TOO_CONCENTRATED
    assert estimate.a_in_b is None
