"""Shared feature interface: every detector (SIFT, ORB, SuperPoint…) and matcher plugs in here.

Changing this module affects persons A, B and C: it needs a review from all three.

Conventions:
- Keypoints are (x, y) pixel coordinates in OpenCV's convention (pixel centres at integers).
- Masks are uint8 images of the same height and width as the image; 0 marks areas to ignore
  (the COLMAP and OpenCV convention), anything else is kept.
- Scores are "higher is more confident", for keypoints and for matches alike, so that
  thresholds read the same whatever the detector.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# eq=False: the generated __eq__ would compare numpy arrays element-wise and fail.
@dataclass(frozen=True, eq=False)
class Features:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 2:
            raise ValueError(f"keypoints must be (N, 2), got {self.keypoints.shape}")
        count = len(self.keypoints)
        if self.descriptors.ndim != 2 or len(self.descriptors) != count:
            raise ValueError(
                f"descriptors must be ({count}, D), got {self.descriptors.shape}"
            )
        if self.scores.shape != (count,):
            raise ValueError(f"scores must be ({count},), got {self.scores.shape}")

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass(frozen=True, eq=False)
class Matches:
    """Row k pairs keypoint `index_pairs[k, 0]` of image A with `index_pairs[k, 1]` of image B."""

    index_pairs: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.index_pairs.ndim != 2 or self.index_pairs.shape[1] != 2:
            raise ValueError(
                f"index_pairs must be (K, 2), got {self.index_pairs.shape}"
            )
        if not np.issubdtype(self.index_pairs.dtype, np.integer):
            raise ValueError(
                f"index_pairs must be integers, got {self.index_pairs.dtype}"
            )
        if self.scores.shape != (len(self.index_pairs),):
            raise ValueError(
                f"scores must be ({len(self.index_pairs)},), got {self.scores.shape}"
            )

    def __len__(self) -> int:
        return len(self.index_pairs)


class FeatureExtractor(Protocol):
    # Stored in the `features.detector` column, so it must be stable across runs. Read-only,
    # so frozen dataclasses (and plain class attributes) satisfy the protocol.
    @property
    def name(self) -> str: ...

    def extract(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> Features: ...


class FeatureMatcher(Protocol):
    @property
    def name(self) -> str: ...

    def match(self, features_a: Features, features_b: Features) -> Matches: ...
