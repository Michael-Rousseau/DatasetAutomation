import numpy as np

from dataset_automation.cleaning.labels import (
    BACKGROUND,
    label_image,
    masks_from_labels,
)

CLASSES = ("fin", "diver")


def test_labels_start_at_one_after_background() -> None:
    fin = np.array([[True, False, False]])
    diver = np.array([[False, True, False]])

    labels = label_image({"fin": fin, "diver": diver}, CLASSES)

    assert labels.tolist() == [[1, 2, BACKGROUND]]


def test_first_listed_class_wins_where_masks_overlap() -> None:
    fin = np.array([[True, False]])
    diver = np.array([[True, True]])

    labels = label_image({"fin": fin, "diver": diver}, CLASSES)

    assert labels.tolist() == [[1, 2]]


def test_masks_round_trip_through_labels() -> None:
    fin = np.array([[True, False, False]])
    diver = np.array([[False, False, True]])

    masks = masks_from_labels(
        label_image({"fin": fin, "diver": diver}, CLASSES), CLASSES
    )

    assert (masks["fin"] == fin).all()
    assert (masks["diver"] == diver).all()
