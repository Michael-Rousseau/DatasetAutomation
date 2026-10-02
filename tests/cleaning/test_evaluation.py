import numpy as np
import pytest

from dataset_automation.cleaning.evaluation import iou, score_classes


def masks(*rows: str) -> np.ndarray:
    return np.array([[char == "#" for char in row] for row in rows])


def test_iou_of_partial_overlap() -> None:
    assert iou(masks("##.."), masks(".##.")) == pytest.approx(1 / 3)


def test_iou_is_undefined_when_both_masks_are_empty() -> None:
    assert iou(masks("...."), masks("....")) is None


def test_iou_rejects_masks_of_different_sizes() -> None:
    with pytest.raises(ValueError):
        iou(masks("##"), masks("###"))


def test_scores_count_false_positives_and_misses() -> None:
    samples = [
        ({"fin": masks("##..")}, {"fin": masks("##..")}),  # perfect
        ({"fin": masks("#...")}, {"fin": masks("....")}),  # false positive
        ({"fin": masks("....")}, {"fin": masks("..##")}),  # missed
        ({"fin": masks("....")}, {"fin": masks("....")}),  # nothing to score
    ]

    score = score_classes(samples, ("fin",))["fin"]

    assert score.images_scored == 3
    assert score.mean_iou == pytest.approx(1 / 3)
    assert score.pooled_iou == pytest.approx(2 / 5)
    assert score.false_positive_images == 1
    assert score.missed_images == 1


def test_class_never_seen_has_no_score() -> None:
    samples = [({"sky": masks("....")}, {"sky": masks("....")})]

    score = score_classes(samples, ("sky",))["sky"]

    assert score.mean_iou is None
    assert score.pooled_iou is None
    assert score.images_scored == 0
