"""Bring a reconstruction into the frame of a reference, to compare camera positions.

A reconstruction from images alone has its own origin, orientation and scale. The similarity
(scale, rotation, translation) that best maps its camera centres onto the reference ones
removes that freedom; what is left is the error of the reconstruction.
"""

from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3


@dataclass(frozen=True, eq=False)
class Similarity:
    """Maps a point p to `scale * rotation @ p + translation`."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.rotation.T + self.translation


def fit_similarity(source: np.ndarray, target: np.ndarray) -> Similarity:
    """Least-squares similarity from `source` to `target`, both (N, 3), row i with row i
    (Umeyama, 1991, "Least-squares estimation of transformation parameters between two point
    patterns").

    Every point weighs the same: a few cameras placed far from where they should be pull
    the fit, and show up as a larger error everywhere. Read the median error, not the mean.
    """
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"expected two (N, 3) arrays, got {source.shape} and {target.shape}"
        )
    if len(source) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} points, got {len(source)}")

    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centred, target_centred = source - source_mean, target - target_mean
    covariance = target_centred.T @ source_centred / len(source)
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    # A reflection would fit as well as a rotation: forbid it.
    handedness = np.ones(3)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0:
        handedness[2] = -1.0
    rotation = left @ np.diag(handedness) @ right_transposed
    source_variance = (source_centred**2).sum() / len(source)
    if source_variance == 0:
        raise ValueError("the source points are all the same point")
    scale = float((singular_values * handedness).sum() / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    return Similarity(scale, rotation, translation)
