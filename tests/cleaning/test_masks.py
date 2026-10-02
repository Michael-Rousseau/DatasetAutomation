from pathlib import Path

import numpy as np
import pytest

from dataset_automation.cleaning.masks import (
    IGNORE,
    KEEP,
    colmap_mask_path,
    keep_mask,
    read_unwanted,
    write_mask,
)


def test_keep_mask_ignores_the_union_of_unwanted_pixels() -> None:
    first = np.array([[True, False, False]])
    second = np.array([[False, False, True]])

    mask = keep_mask([first, second], (1, 3))

    assert mask.tolist() == [[IGNORE, KEEP, IGNORE]]
    assert mask.dtype == np.uint8


def test_keep_mask_keeps_everything_when_nothing_is_unwanted() -> None:
    assert (keep_mask([], (2, 2)) == KEEP).all()


def test_keep_mask_rejects_mask_of_another_size() -> None:
    with pytest.raises(ValueError):
        keep_mask([np.zeros((2, 2), dtype=bool)], (3, 3))


def test_colmap_mask_path_appends_png_to_the_relative_image_path() -> None:
    path = colmap_mask_path(
        Path("masks"), Path("images/dive1/IMG_0001.JPG"), Path("images")
    )

    assert path == Path("masks/dive1/IMG_0001.JPG.png")


def test_written_mask_reads_back_identically(tmp_path: Path) -> None:
    unwanted = np.array([[True, False], [False, True]])
    path = tmp_path / "nested" / "a.jpg.png"

    write_mask(path, keep_mask([unwanted], (2, 2)))

    assert (read_unwanted(path) == unwanted).all()


def test_write_mask_refuses_colour_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_mask(tmp_path / "a.png", np.zeros((2, 2, 3), dtype=np.uint8))
