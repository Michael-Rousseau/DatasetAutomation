from dataclasses import dataclass, field

import cv2
import numpy as np

from dataset_automation.features.interface import Features
from dataset_automation.features.preprocessing import check_scale, no_features, prepare


@dataclass(frozen=True)
class SiftExtractor:
    """SIFT keypoints, optionally detected on a downscaled copy of the image.

    Keypoints are always returned in full-resolution pixel coordinates, whatever `scale` is.
    `max_keypoints` = 0 keeps every keypoint SIFT finds.

    Defaults measured on 40 Mermaid images spread over the dive (docs/step0.md): OpenCV's
    contrast threshold (0.04) leaves low-contrast sand with ~16 keypoints and ~1 match between
    images 5 apart; 0.01 raises that to ~56 matches. Full resolution adds keypoints but almost
    no matches, and the 8000 cap keeps brute-force matching at ~0.08 s per pair.
    """

    max_keypoints: int = 8000
    scale: float = 0.5
    contrast_threshold: float = 0.01
    name: str = field(default="sift", init=False)

    def __post_init__(self) -> None:
        check_scale(self.scale)
        if self.max_keypoints < 0:
            raise ValueError(f"max_keypoints must be >= 0, got {self.max_keypoints}")
        if self.contrast_threshold <= 0:
            raise ValueError(
                f"contrast_threshold must be > 0, got {self.contrast_threshold}"
            )

    def extract(self, image: np.ndarray, mask: np.ndarray | None = None) -> Features:
        detection = prepare(image, mask, self.scale)
        # The default upscaling of the first octave biases every keypoint by ~+0.25 px;
        # precise upscaling removes it (measured on blobs with known centres).
        detector = cv2.SIFT.create(
            nfeatures=self.max_keypoints,
            contrastThreshold=self.contrast_threshold,
            enable_precise_upscale=True,
        )
        keypoints, descriptors = detector.detectAndCompute(
            detection.image, detection.mask
        )
        if not keypoints:
            return no_features()
        # `nfeatures` keeps ties at the cut-off, so it can return a few extra keypoints.
        if self.max_keypoints:
            strongest = np.argsort(
                [-keypoint.response for keypoint in keypoints], kind="stable"
            )[: self.max_keypoints]
            keypoints = [keypoints[i] for i in strongest]
            descriptors = descriptors[strongest]

        small_xy = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float64)
        return Features(
            keypoints=detection.to_full_resolution(small_xy),
            descriptors=descriptors,
            scores=np.array(
                [keypoint.response for keypoint in keypoints], dtype=np.float64
            ),
        )
