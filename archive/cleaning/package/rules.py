"""Hand-written baseline for the water surface, to compare against the zero-shot models.

Seen from below, the surface (and the sky through it) is a bright, smooth area at the top of
the image. The rule keeps pixels that are bright and locally flat, and only the connected
regions that touch the top border: a bright patch of sand in the middle of the image is not
the surface.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SurfaceRuleParameters:
    # The surface is only searched above this share of the image height.
    top_fraction: float = 0.5
    min_brightness: int = 200  # grey level, 0-255
    max_local_std: float = 8.0  # grey-level standard deviation over `window`
    window: int = 15  # pixels, odd

    def __post_init__(self) -> None:
        if not 0 < self.top_fraction <= 1:
            raise ValueError(f"top_fraction must be in (0, 1], got {self.top_fraction}")
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError(f"window must be a positive odd number, got {self.window}")


DEFAULT_PARAMETERS = SurfaceRuleParameters()


def surface_mask(
    image: np.ndarray, parameters: SurfaceRuleParameters = DEFAULT_PARAMETERS
) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    size = (parameters.window, parameters.window)
    mean = cv2.boxFilter(grey, -1, size)
    mean_of_squares = cv2.boxFilter(grey * grey, -1, size)
    # Rounding can make the variance slightly negative on flat areas.
    local_std = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0))

    bright = grey >= parameters.min_brightness
    bright[round(grey.shape[0] * parameters.top_fraction) :] = False
    candidate = bright & (local_std <= parameters.max_local_std)

    _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    touching_top = np.unique(labels[0][candidate[0]])
    surface = np.isin(labels, touching_top) & candidate

    # Near the surface's edge, the window straddles the seafloor and the local std is high:
    # grow back over bright pixels within half a window, or the mask stops short of the edge.
    kernel = np.ones((parameters.window, parameters.window), dtype=np.uint8)
    return cv2.dilate(surface.astype(np.uint8), kernel).astype(bool) & bright


class SurfaceRule:
    name = "rule-surface"
    classes = ("surface",)

    def __init__(self, parameters: SurfaceRuleParameters = DEFAULT_PARAMETERS) -> None:
        self.parameters = parameters

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return {"surface": surface_mask(image, self.parameters)}

    def run_parameters(self) -> dict[str, object]:
        return asdict(self.parameters)
