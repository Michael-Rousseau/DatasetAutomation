"""Hand-made ground-truth masks, read from a COCO export (CVAT and Label Studio both offer it).

Every image listed in the export counts as annotated, even with no annotation: an image where
the annotator found nothing is exactly what measures false positives.
"""

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def parse_category_map(pairs: list[str]) -> dict[str, str]:
    """Parse `["Diver=diver", "Fins=fin"]` into `{"Diver": "diver", "Fins": "fin"}`."""
    mapping = {}
    for pair in pairs:
        category, separator, class_name = pair.partition("=")
        if not separator or not category or not class_name:
            raise ValueError(f"expected CATEGORY=CLASS, got {pair!r}")
        mapping[category] = class_name
    return mapping


def annotated_images(
    path: Path,
    image_root: Path,
    classes: tuple[str, ...],
    category_to_class: Mapping[str, str] | None = None,
) -> Iterator[tuple[str, np.ndarray, dict[str, np.ndarray]]]:
    """Yield (file_name, BGR image, truth masks) for every image of a COCO export.

    `file_name` is resolved against `image_root`; unreadable images are logged and skipped.
    """
    for file_name, masks in load_coco_masks(path, classes, category_to_class).items():
        image = cv2.imread(str(image_root / file_name))
        if image is None:
            logger.warning("cannot read %s, skipped", image_root / file_name)
            continue
        shape = next(iter(masks.values())).shape
        if image.shape[:2] != shape:
            raise ValueError(
                f"{file_name}: image is {image.shape[:2]}, annotations are {shape}"
            )
        yield file_name, image, masks


def load_coco_masks(
    path: Path,
    classes: tuple[str, ...],
    category_to_class: Mapping[str, str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Return, per image `file_name`, one boolean mask per class in `classes`.

    Category names map to classes through `category_to_class` (identity by default);
    categories that map to no class in `classes` are ignored.
    """
    coco = json.loads(path.read_text())
    mapping = category_to_class or {}
    class_by_category = {
        category["id"]: mapping.get(category["name"], category["name"])
        for category in coco["categories"]
    }

    images = {image["id"]: image for image in coco["images"]}
    masks = {
        image["file_name"]: {
            name: np.zeros((image["height"], image["width"]), dtype=bool)
            for name in classes
        }
        for image in images.values()
    }
    for annotation in coco["annotations"]:
        class_name = class_by_category[annotation["category_id"]]
        if class_name not in classes:
            continue
        image = images[annotation["image_id"]]
        shape = (image["height"], image["width"])
        masks[image["file_name"]][class_name] |= segmentation_mask(
            annotation["segmentation"], shape
        )
    return masks


def segmentation_mask(
    segmentation: list[list[float]] | dict, shape: tuple[int, int]
) -> np.ndarray:
    """Rasterise a COCO `segmentation`: polygons, or run-length encoding (RLE)."""
    if isinstance(segmentation, list):
        canvas = np.zeros(shape, dtype=np.uint8)
        polygons = [
            np.round(np.asarray(polygon, dtype=np.float64).reshape(-1, 2)).astype(
                np.int32
            )
            for polygon in segmentation
        ]
        cv2.fillPoly(canvas, polygons, 1)
        return canvas.astype(bool)

    counts = segmentation["counts"]
    if isinstance(counts, str):
        counts = decode_rle_string(counts)
    height, width = segmentation["size"]
    if (height, width) != shape:
        raise ValueError(f"RLE size {(height, width)} does not match image {shape}")
    return rle_to_mask(counts, shape)


def rle_to_mask(counts: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Runs alternate background / foreground, starting with background, in column-major order."""
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    if flat.size != shape[0] * shape[1]:
        raise ValueError(
            f"RLE covers {flat.size} pixels, expected {shape[0] * shape[1]}"
        )
    return flat.reshape(shape, order="F")


def decode_rle_string(encoded: str) -> list[int]:
    """Decode COCO's compressed RLE string (the format of pycocotools' `rleFrString`).

    Each count is written in 5-bit groups, as characters offset by 48; bit 0x20 means "another
    group follows" and bit 0x10 of the last group is the sign. From the third count on, counts
    are stored as the difference with the count two places before.
    """
    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        more = True
        while more:
            group = ord(encoded[position]) - 48
            value |= (group & 0x1F) << shift
            more = bool(group & 0x20)
            position += 1
            shift += 5
            if not more and group & 0x10:
                value |= -1 << shift
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts
