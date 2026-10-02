"""Overlap between two images from their matched features (01_overlap_engine.md, step 2).

Two methods, compared in scripts/overlap/evaluate_pairwise_mermaid.py:
- homography: project one image outline into the other. Exact for a planar or distant scene,
  and it does not need points everywhere: the outline carries the geometry.
- inlier hull: the area covered by correctly matched points. Robust to relief, but it can only
  see the overlap where there are points, so it underestimates on patchy texture.

Both report "unknown" (None) rather than a number they cannot support.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from shapely import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.features.interface import Features, Matches
from dataset_automation.overlap.geometry import image_frame, matched_points
from dataset_automation.overlap.homography import (
    fit_homography,
    is_plausible,
    transform,
)

# Measured on 150 Mermaid pairs: at 60–80 % reference overlap the 10th percentile is 13 SIFT
# inliers, and below ~40 % overlap the 4–8 inliers left are noise around the 4-point minimum.
MIN_INLIERS = 15
# Relief makes parallax a single homography cannot explain. On 150 Mermaid pairs, 8 px instead of
# 3 px cut unknowns from 80 % to 73 % at 20–40 % overlap with the same error; 15 px started
# accepting overlaps for pairs that share nothing.
RANSAC_THRESHOLD_PX = 8.0
# Area of the inliers' convex hull over the estimated common area. Identical GCP target plates
# (same pattern, different number) produced a consistent homography with all inliers on the
# plate: spread 0.005 for a claimed 79 % overlap of images that do not overlap. The 102 correct
# Mermaid estimates never went below 0.026 (p1 0.044). One false case observed: revisit with data.
MIN_INLIER_SPREAD = 0.01


class UnknownReason(Enum):
    TOO_FEW_MATCHES = "too_few_matches"
    NO_MODEL = "no_model"
    TOO_FEW_INLIERS = "too_few_inliers"
    IMPLAUSIBLE_HOMOGRAPHY = "implausible_homography"
    INLIERS_TOO_CONCENTRATED = "inliers_too_concentrated"


@dataclass(frozen=True)
class PairOverlapEstimate:
    """Share of each image seen by the other; None when unknown, never an invented value.

    `a_in_b` is measured in image A's frame (|A ∩ B| / |A| in A's pixels) and `b_in_a` in B's:
    a perspective change does not preserve areas, so each share uses its own image's pixels.
    """

    method: str
    match_count: int
    inlier_count: int
    a_in_b: float | None
    b_in_a: float | None
    iou: float | None
    unknown_reason: UnknownReason | None

    @property
    def is_known(self) -> bool:
        return self.unknown_reason is None

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.match_count if self.match_count else 0.0


def estimate_homography_overlap(
    features_a: Features,
    features_b: Features,
    matches: Matches,
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    intrinsics: Intrinsics | None = None,
    min_inliers: int = MIN_INLIERS,
) -> PairOverlapEstimate:
    """Sizes are (width, height). `intrinsics` (same camera for both) undistorts first."""
    method = "homography"
    points_a, points_b = matched_points(features_a, features_b, matches, intrinsics)
    fit = fit_homography(points_a, points_b, RANSAC_THRESHOLD_PX)
    if fit is None:
        return _unknown(method, len(matches), 0, _no_model_reason(len(matches)))
    if fit.inlier_count < min_inliers:
        return _unknown(
            method, len(matches), fit.inlier_count, UnknownReason.TOO_FEW_INLIERS
        )

    frame_a, frame_b = image_frame(size_a, intrinsics), image_frame(size_b, intrinsics)
    if not is_plausible(fit.matrix, frame_a, frame_b):
        return _unknown(
            method, len(matches), fit.inlier_count, UnknownReason.IMPLAUSIBLE_HOMOGRAPHY
        )

    a_in_b_frame = transform(fit.matrix, frame_a)
    b_in_a_frame = transform(np.linalg.inv(fit.matrix), frame_b)
    shared_in_b = _area(a_in_b_frame.intersection(frame_b), a_in_b_frame, frame_b)
    shared_in_a = _area(frame_a.intersection(b_in_a_frame), frame_a, b_in_a_frame)
    # Inliers but no common area: the homography contradicts its own support.
    if shared_in_a == 0:
        return _unknown(
            method, len(matches), fit.inlier_count, UnknownReason.IMPLAUSIBLE_HOMOGRAPHY
        )
    # A repeated planar object (e.g. two identical target plates) fits a homography perfectly
    # from a few square centimetres: a real overlap spreads its inliers over the common area.
    inlier_hull = MultiPoint(points_a[fit.inliers]).convex_hull
    if inlier_hull.area / shared_in_a < MIN_INLIER_SPREAD:
        return _unknown(
            method,
            len(matches),
            fit.inlier_count,
            UnknownReason.INLIERS_TOO_CONCENTRATED,
        )
    return PairOverlapEstimate(
        method=method,
        match_count=len(matches),
        inlier_count=fit.inlier_count,
        a_in_b=shared_in_a / frame_a.area,
        b_in_a=shared_in_b / frame_b.area,
        # IoU has no natural frame under perspective: it is measured in B's.
        iou=shared_in_b / (a_in_b_frame.area + frame_b.area - shared_in_b),
        unknown_reason=None,
    )


def estimate_hull_overlap(
    features_a: Features,
    features_b: Features,
    matches: Matches,
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    intrinsics: Intrinsics | None = None,
    min_inliers: int = MIN_INLIERS,
) -> PairOverlapEstimate:
    """Convex hull of the inliers over the image area, in each image.

    Inliers come from a fundamental matrix, which holds for any rigid scene: a homography would
    reimpose the planar assumption this method is meant to avoid. No IoU: the two hulls live in
    different images.
    """
    method = "inlier-hull"
    points_a, points_b = matched_points(features_a, features_b, matches, intrinsics)
    # The 8-point algorithm needs 8 matches; fewer means no model at all.
    if len(matches) < 8:
        return _unknown(method, len(matches), 0, UnknownReason.TOO_FEW_MATCHES)
    _, inlier_mask = cv2.findFundamentalMat(
        points_a, points_b, cv2.USAC_MAGSAC, RANSAC_THRESHOLD_PX
    )
    if inlier_mask is None:
        return _unknown(method, len(matches), 0, UnknownReason.NO_MODEL)
    inliers = inlier_mask.ravel().astype(bool)
    if inliers.sum() < min_inliers:
        return _unknown(
            method, len(matches), int(inliers.sum()), UnknownReason.TOO_FEW_INLIERS
        )

    frame_a, frame_b = image_frame(size_a, intrinsics), image_frame(size_b, intrinsics)
    hull_a = MultiPoint(points_a[inliers]).convex_hull.intersection(frame_a)
    hull_b = MultiPoint(points_b[inliers]).convex_hull.intersection(frame_b)
    return PairOverlapEstimate(
        method=method,
        match_count=len(matches),
        inlier_count=int(inliers.sum()),
        a_in_b=min(hull_a.area / frame_a.area, 1.0),
        b_in_a=min(hull_b.area / frame_b.area, 1.0),
        iou=None,
        unknown_reason=None,
    )


def _area(intersection: BaseGeometry, first: Polygon, second: Polygon) -> float:
    # Clamped: shapely can return an intersection a hair larger than the smaller polygon.
    return min(intersection.area, first.area, second.area)


def _no_model_reason(match_count: int) -> UnknownReason:
    return UnknownReason.TOO_FEW_MATCHES if match_count < 4 else UnknownReason.NO_MODEL


def _unknown(
    method: str, match_count: int, inlier_count: int, reason: UnknownReason
) -> PairOverlapEstimate:
    return PairOverlapEstimate(
        method=method,
        match_count=match_count,
        inlier_count=inlier_count,
        a_in_b=None,
        b_in_a=None,
        iou=None,
        unknown_reason=reason,
    )
