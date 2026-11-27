import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset_automation.cleaning.annotations import (
    annotated_images,
    decode_rle_string,
    load_coco_masks,
    parse_category_map,
    rle_to_mask,
    segmentation_mask,
)


def write_coco(path: Path, annotations: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.jpg", "height": 4, "width": 6},
                    {"id": 2, "file_name": "empty.jpg", "height": 4, "width": 6},
                ],
                "categories": [
                    {"id": 10, "name": "fin"},
                    {"id": 11, "name": "Diver"},
                    {"id": 12, "name": "fish"},
                ],
                "annotations": annotations,
            }
        )
    )
    return path


def test_rasterises_polygon_annotations(tmp_path: Path) -> None:
    square = [[1, 1, 3, 1, 3, 2, 1, 2]]
    coco = write_coco(
        tmp_path / "coco.json",
        [{"image_id": 1, "category_id": 10, "segmentation": square}],
    )

    masks = load_coco_masks(coco, ("fin",))

    assert masks["a.jpg"]["fin"][1:3, 1:4].all()
    assert np.count_nonzero(masks["a.jpg"]["fin"]) == 6


def test_images_without_annotation_get_empty_masks(tmp_path: Path) -> None:
    coco = write_coco(tmp_path / "coco.json", [])

    masks = load_coco_masks(coco, ("fin", "diver"))

    assert not masks["empty.jpg"]["fin"].any()
    assert masks["empty.jpg"]["diver"].shape == (4, 6)


def test_maps_category_names_and_ignores_unknown_ones(tmp_path: Path) -> None:
    coco = write_coco(
        tmp_path / "coco.json",
        [
            {"image_id": 1, "category_id": 11, "segmentation": [[0, 0, 2, 0, 2, 2]]},
            {"image_id": 1, "category_id": 12, "segmentation": [[0, 0, 5, 0, 5, 3]]},
        ],
    )

    masks = load_coco_masks(coco, ("diver",), category_to_class={"Diver": "diver"})

    assert masks["a.jpg"]["diver"].any()
    assert set(masks["a.jpg"]) == {"diver"}


def test_uncompressed_rle_is_column_major() -> None:
    # 2x3 image, columns read top to bottom: 0 0 | 1 1 | 0 1
    mask = rle_to_mask([2, 2, 1, 1], (2, 3))

    assert mask.tolist() == [[False, True, False], [False, True, True]]


def test_rle_covering_the_wrong_number_of_pixels_is_rejected() -> None:
    with pytest.raises(ValueError):
        rle_to_mask([2, 2], (2, 3))


def test_decodes_compressed_rle_strings() -> None:
    # Encoded with pycocotools' rleToString algorithm; the second one has multi-group and
    # negative (delta-coded) counts.
    assert decode_rle_string("221O") == [2, 2, 1, 1]
    assert decode_rle_string("5Xo0d0kPOl;") == [5, 1000, 20, 3, 400]


def test_compressed_rle_segmentation_matches_uncompressed() -> None:
    shape = (2, 3)
    compressed = segmentation_mask({"size": [2, 3], "counts": "221O"}, shape)
    uncompressed = segmentation_mask({"size": [2, 3], "counts": [2, 2, 1, 1]}, shape)

    assert (compressed == uncompressed).all()


def test_parses_category_to_class_pairs() -> None:
    assert parse_category_map(["Diver=diver", "Swim fin=fin"]) == {
        "Diver": "diver",
        "Swim fin": "fin",
    }


def test_rejects_category_pair_without_class() -> None:
    with pytest.raises(ValueError):
        parse_category_map(["Diver"])


def test_annotated_images_pairs_each_image_with_its_masks(tmp_path: Path) -> None:
    coco = write_coco(tmp_path / "coco.json", [])
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((4, 6, 3), dtype=np.uint8))

    found = list(annotated_images(coco, tmp_path, ("fin",)))

    # empty.jpg is listed in the export but missing on disk: skipped, not fatal.
    assert [name for name, _, _ in found] == ["a.jpg"]
    assert found[0][2]["fin"].shape == (4, 6)
