"""Conversion between per-class boolean masks and a single label image (one class per pixel).

Semantic segmentation models such as SegFormer predict one label per pixel: 0 is background
(nothing unwanted), and label i + 1 is `classes[i]`.
"""

from collections.abc import Mapping

import numpy as np

BACKGROUND = 0


def label_image(
    masks: Mapping[str, np.ndarray], classes: tuple[str, ...]
) -> np.ndarray:
    """Merge boolean masks into a uint8 label image.

    Where annotations overlap, the class listed first in `classes` wins (e.g. a fin drawn over
    a diver stays a fin), so order `classes` from most to least specific.
    """
    if len(classes) > 254:
        raise ValueError("at most 254 classes fit in a uint8 label image")
    shape = next(iter(masks.values())).shape
    labels = np.full(shape, BACKGROUND, dtype=np.uint8)
    # Painted last to first so that earlier classes overwrite later ones.
    for index in reversed(range(len(classes))):
        labels[masks[classes[index]]] = index + 1
    return labels


def masks_from_labels(
    labels: np.ndarray, classes: tuple[str, ...]
) -> dict[str, np.ndarray]:
    return {name: labels == index + 1 for index, name in enumerate(classes)}
