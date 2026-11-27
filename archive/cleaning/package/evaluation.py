"""IoU per class between predicted and hand-annotated masks (step 3 of the cleaning sheet)."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


def iou(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    """Intersection over union of two boolean masks; None when both are empty.

    Two empty masks agree perfectly but carry no information about segmentation quality,
    so they are left out of the average rather than counted as 1.
    """
    if predicted.shape != truth.shape:
        raise ValueError(f"shapes differ: {predicted.shape} vs {truth.shape}")
    union = np.count_nonzero(predicted | truth)
    if union == 0:
        return None
    return np.count_nonzero(predicted & truth) / union


@dataclass(frozen=True)
class ClassScore:
    """Scores of one class over a test set.

    `mean_iou` averages per-image IoU over images where the class is predicted or annotated;
    `pooled_iou` sums intersections and unions over all images, so large regions weigh more.
    Both are None when the class never appears, neither predicted nor annotated.
    """

    class_name: str
    mean_iou: float | None
    pooled_iou: float | None
    images_scored: int
    false_positive_images: int  # predicted, but not annotated
    missed_images: int  # annotated, but not predicted


def score_classes(
    samples: Iterable[tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
    classes: tuple[str, ...],
) -> dict[str, ClassScore]:
    """Score (predicted, truth) mask dictionaries, one pair per test image."""
    per_image: dict[str, list[float]] = {name: [] for name in classes}
    intersections = dict.fromkeys(classes, 0)
    unions = dict.fromkeys(classes, 0)
    false_positives = dict.fromkeys(classes, 0)
    misses = dict.fromkeys(classes, 0)

    for predicted, truth in samples:
        for name in classes:
            value = iou(predicted[name], truth[name])
            if value is None:
                continue
            per_image[name].append(value)
            intersections[name] += int(np.count_nonzero(predicted[name] & truth[name]))
            unions[name] += int(np.count_nonzero(predicted[name] | truth[name]))
            if not truth[name].any():
                false_positives[name] += 1
            elif not predicted[name].any():
                misses[name] += 1

    return {
        name: ClassScore(
            class_name=name,
            mean_iou=float(np.mean(per_image[name])) if per_image[name] else None,
            pooled_iou=intersections[name] / unions[name] if unions[name] else None,
            images_scored=len(per_image[name]),
            false_positive_images=false_positives[name],
            missed_images=misses[name],
        )
        for name in classes
    }
