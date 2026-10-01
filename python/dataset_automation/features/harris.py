from dataclasses import dataclass, field

import cv2
import numpy as np

from dataset_automation.features.interface import Features
from dataset_automation.features.preprocessing import check_scale, no_features, prepare

# OpenCV's usual Harris settings: 3×3 gradient covariance window, 3×3 Sobel, k = 0.04.
HARRIS_BLOCK_SIZE = 3
HARRIS_SOBEL_SIZE = 3
HARRIS_K = 0.04


@dataclass(frozen=True)
class HarrisSiftExtractor:
    """Harris corners, located to sub-pixel precision, described by upright SIFT descriptors.

    Harris gives neither scale nor orientation: every corner gets the same descriptor diameter
    and angle 0. Descriptors are therefore *not* rotation- or scale-invariant: fine between
    consecutive images, unreliable for revisits seen under another heading or altitude.
    `quality_level` is relative to the strongest corner of the image, so it adapts to contrast.
    Lengths (`min_distance_px`, `descriptor_diameter_px`) are in detection-image pixels, i.e.
    after downscaling by `scale`. `max_keypoints` = 0 keeps every corner.

    A baseline, not a replacement for SIFT: on 40 Mermaid images sampled over the dive, these
    defaults (the best of scale 0.5/1, quality 0.01/0.001, diameter 16/32) give 4 homography
    inliers between images 5 apart (median) against 64 for SiftExtractor.
    """

    max_keypoints: int = 8000
    scale: float = 0.5
    quality_level: float = 0.001
    min_distance_px: float = 3.0
    descriptor_diameter_px: float = 16.0
    name: str = field(default="harris-sift", init=False)

    def __post_init__(self) -> None:
        check_scale(self.scale)
        if self.max_keypoints < 0:
            raise ValueError(f"max_keypoints must be >= 0, got {self.max_keypoints}")
        if not 0 < self.quality_level < 1:
            raise ValueError(
                f"quality_level must be in (0, 1), got {self.quality_level}"
            )
        if self.descriptor_diameter_px <= 0:
            raise ValueError(
                f"descriptor_diameter_px must be > 0, got {self.descriptor_diameter_px}"
            )

    def extract(self, image: np.ndarray, mask: np.ndarray | None = None) -> Features:
        detection = prepare(image, mask, self.scale)
        # Returned strongest first; maxCorners = 0 means no limit, as for max_keypoints.
        corners = cv2.goodFeaturesToTrack(
            detection.image,
            self.max_keypoints,
            self.quality_level,
            self.min_distance_px,
            mask=detection.mask,
            blockSize=HARRIS_BLOCK_SIZE,
            useHarrisDetector=True,
            k=HARRIS_K,
        )
        if corners is None:
            return no_features()

        response = cv2.cornerHarris(
            detection.image, HARRIS_BLOCK_SIZE, HARRIS_SOBEL_SIZE, HARRIS_K
        )
        integer_xy = corners.reshape(-1, 2).astype(np.int64)
        scores = response[integer_xy[:, 1], integer_xy[:, 0]]
        refined_xy = refine_to_response_peak(response, integer_xy)

        # class_id carries the corner's index: SIFT drops keypoints too close to the border.
        keypoints = [
            cv2.KeyPoint(
                float(x),
                float(y),
                self.descriptor_diameter_px,
                0.0,
                float(score),
                0,
                index,
            )
            for index, ((x, y), score) in enumerate(zip(refined_xy, scores))
        ]
        described, descriptors = cv2.SIFT.create().compute(detection.image, keypoints)
        if not described:
            return no_features()
        kept = np.array([keypoint.class_id for keypoint in described])
        return Features(
            keypoints=detection.to_full_resolution(refined_xy[kept]),
            descriptors=descriptors,
            scores=scores[kept].astype(np.float64),
        )


def refine_to_response_peak(response: np.ndarray, integer_xy: np.ndarray) -> np.ndarray:
    """Sub-pixel corner positions: vertex of a parabola fitted per axis to the Harris response.

    cv2.cornerSubPix models a corner as two straight edges; on seafloor-like texture it left half
    the corners unmoved and threw others several pixels away (median error 0.67 px on a
    sub-pixel shift). Fitting the response peak gives 0.23–0.28 px and never moves a corner
    by more than half a pixel. Corners on the image border stay at their integer position.
    """
    height, width = response.shape
    x, y = integer_xy[:, 0], integer_xy[:, 1]
    inside_x = (x > 0) & (x < width - 1)
    inside_y = (y > 0) & (y < height - 1)
    safe_x, safe_y = np.clip(x, 1, width - 2), np.clip(y, 1, height - 2)

    offset_x = _parabola_vertex(
        response[y, safe_x - 1], response[y, safe_x], response[y, safe_x + 1]
    )
    offset_y = _parabola_vertex(
        response[safe_y - 1, x], response[safe_y, x], response[safe_y + 1, x]
    )
    return np.column_stack(
        [x + np.where(inside_x, offset_x, 0.0), y + np.where(inside_y, offset_y, 0.0)]
    ).astype(np.float64)


def _parabola_vertex(
    before: np.ndarray, centre: np.ndarray, after: np.ndarray
) -> np.ndarray:
    # A maximum has negative curvature; anything else (flat, saddle) keeps the integer position.
    curvature = before - 2 * centre + after
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(curvature < 0, (before - after) / (2 * curvature), 0.0)
    return np.clip(offset, -0.5, 0.5)
