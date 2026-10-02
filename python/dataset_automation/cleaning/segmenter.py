"""Common shape of every mask producer: hand-written rule, zero-shot models, fine-tuned model.

Keeping them behind one protocol lets the pipeline, the inventory and the evaluation treat
them alike, so the zero-shot and fine-tuned approaches are compared on exactly the same code path.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

# Unwanted elements listed in docs/02_cleaning_tracking.md. From underwater, the sky is only
# seen through the surface, so a segmenter may merge the two under "surface".
CLASSES = ("fin", "diver", "robot_part", "surface", "sky")


class Segmenter(Protocol):
    # Read-only properties rather than plain attributes: a protocol attribute is mutable, hence
    # invariant, and would reject a class that declares `classes = ("fin", "diver")`.
    @property
    def name(self) -> str:
        """Stored in the `masks.method` column, so it must be stable across runs."""
        ...

    @property
    def classes(self) -> tuple[str, ...]: ...

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Return one boolean (H, W) mask per class in `classes`, True on the unwanted element.

        `image` is a BGR uint8 array as returned by `cv2.imread`. A class that is absent gets an
        all-False mask rather than no entry: the segmenter looked and found nothing.
        """
        ...

    def run_parameters(self) -> dict[str, object]:
        """JSON-serialisable settings recorded in the `runs` table (thresholds, model ids…)."""
        ...


METHODS = ("rule", "zero-shot", "segformer")


def load_segmenter(
    method: str, model_dir: Path | None = None, device: str | None = None
) -> Segmenter:
    """Build a segmenter by name, for the scripts. The model-based ones need the `cleaning` extra."""
    # Imported here so that the rule works without torch installed.
    if method == "rule":
        from dataset_automation.cleaning.rules import SurfaceRule

        return SurfaceRule()
    if method == "zero-shot":
        from dataset_automation.cleaning.zero_shot import ZeroShotSegmenter

        return ZeroShotSegmenter(device=device)
    if method == "segformer":
        if model_dir is None:
            raise ValueError("the segformer method needs a trained model directory")
        from dataset_automation.cleaning.segformer import SegformerSegmenter

        return SegformerSegmenter(model_dir, device=device)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")
