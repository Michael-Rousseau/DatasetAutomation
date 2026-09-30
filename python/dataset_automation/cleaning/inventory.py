"""Step 1 of the cleaning sheet: which unwanted elements actually appear, and how often.

Counts come from a segmenter run on a sample of images, so they are candidates, not truth:
the overlays written alongside are meant to be checked by eye before deciding which classes
are worth handling.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# BGR, one per class in order, readable on blue-green water.
OVERLAY_COLOURS = (
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 128, 255),
)


@dataclass(frozen=True)
class ClassPresence:
    class_name: str
    images_with_class: int
    images_seen: int
    # Mean share of the image covered, over the images where the class is present.
    mean_area_fraction: float | None

    @property
    def presence_rate(self) -> float:
        return self.images_with_class / self.images_seen if self.images_seen else 0.0


def presence(
    results: Iterable[Mapping[str, np.ndarray]],
    classes: tuple[str, ...],
    min_area_fraction: float = 0.001,
) -> dict[str, ClassPresence]:
    """Count, per class, the images where it covers at least `min_area_fraction` of the image.

    The threshold drops the few stray pixels a detector produces on almost every image.
    """
    seen = 0
    counts = dict.fromkeys(classes, 0)
    areas: dict[str, list[float]] = {name: [] for name in classes}
    for masks in results:
        seen += 1
        for name in classes:
            fraction = float(np.mean(masks[name]))
            if fraction >= min_area_fraction:
                counts[name] += 1
                areas[name].append(fraction)
    return {
        name: ClassPresence(
            class_name=name,
            images_with_class=counts[name],
            images_seen=seen,
            mean_area_fraction=float(np.mean(areas[name])) if areas[name] else None,
        )
        for name in classes
    }


def every_nth[T](items: Sequence[T], step: int) -> list[T]:
    """Evenly spaced sample: consecutive frames are near-identical, so spacing beats volume."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    return list(items[::step])


def overlay(
    image: np.ndarray, masks: Mapping[str, np.ndarray], opacity: float = 0.5
) -> np.ndarray:
    """Tint each class's pixels with its colour, for review by eye. The input is not modified."""
    if len(masks) > len(OVERLAY_COLOURS):
        raise ValueError(f"at most {len(OVERLAY_COLOURS)} classes can be overlaid")
    result = image.astype(np.float32)
    for colour, mask in zip(OVERLAY_COLOURS, masks.values(), strict=False):
        result[mask] = (1 - opacity) * result[mask] + opacity * np.asarray(colour)
    return np.round(result).astype(np.uint8)
