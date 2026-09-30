from dataclasses import dataclass, field

import cv2
import numpy as np

from dataset_automation.features.interface import Features, Matches

# Lowe (2004), "Distinctive Image Features from Scale-Invariant Keypoints".
LOWE_RATIO = 0.75


def lowe_ratio_filter(
    best_distances: np.ndarray, second_distances: np.ndarray, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep matches whose nearest neighbour is clearly closer than the second one.

    Returns the boolean mask of kept matches and their scores, 1 − d1/d2: 0 for an ambiguous
    match, 1 for a perfectly distinctive one. A zero second distance means two identical
    candidates, so the match is ambiguous and dropped.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        distance_ratios = best_distances / second_distances
    keep = np.isfinite(distance_ratios) & (distance_ratios < ratio)
    return keep, 1.0 - distance_ratios[keep]


@dataclass(frozen=True)
class RatioTestMatcher:
    """Exact nearest neighbours (brute force, L2) filtered by Lowe's ratio test."""

    ratio: float = LOWE_RATIO
    name: str = field(default="ratio-test", init=False)

    def match(self, features_a: Features, features_b: Features) -> Matches:
        # The ratio test needs two candidates in B for every keypoint of A.
        if len(features_a) == 0 or len(features_b) < 2:
            return _no_matches()

        neighbours = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            features_a.descriptors.astype(np.float32),
            features_b.descriptors.astype(np.float32),
            k=2,
        )
        best_distances = np.array([best.distance for best, _ in neighbours])
        second_distances = np.array([second.distance for _, second in neighbours])
        keep, scores = lowe_ratio_filter(best_distances, second_distances, self.ratio)

        index_pairs = np.array(
            [(best.queryIdx, best.trainIdx) for best, _ in neighbours], dtype=np.int64
        )
        return Matches(index_pairs=index_pairs[keep].reshape(-1, 2), scores=scores)


def _no_matches() -> Matches:
    return Matches(index_pairs=np.empty((0, 2), dtype=np.int64), scores=np.empty(0))
