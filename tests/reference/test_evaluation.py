import numpy as np
import pytest

from dataset_automation.reference.evaluation import evaluate_selection
from dataset_automation.reference.observability import grid_over

SEAFLOOR_HEIGHT = 0.0
SQUARE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])


def survey_line(camera_x: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Cameras 2 m above the seafloor along x, each seeing a 4 m square shifted with it."""
    footprints = np.stack([SQUARE + [x, 0.0] for x in camera_x])
    centres = np.array([[x + 2.0, 2.0, 2.0] for x in camera_x])
    return footprints, centres


def test_keeping_every_image_changes_nothing() -> None:
    footprints, centres = survey_line([0.0, 0.5, 1.0, 1.5, 2.0])
    grid = grid_over(footprints, cell_m=0.25)

    evaluation = evaluate_selection(
        grid, footprints, centres, SEAFLOOR_HEIGHT, range(5), range(5)
    )

    assert evaluation.reconstructible_m2 > 0
    assert evaluation.kept_share == 1.0
    assert evaluation.inside_kept_share == 1.0
    assert evaluation.median_depth_noise == pytest.approx(1.0)
    assert evaluation.weak_links == {0.5: (0, 0), 0.3: (0, 0)}


def test_removing_images_costs_area_on_the_rim_and_depth_precision() -> None:
    footprints, centres = survey_line([0.0, 0.5, 1.0, 1.5, 2.0])
    grid = grid_over(footprints, cell_m=0.25)

    evaluation = evaluate_selection(
        grid, footprints, centres, SEAFLOOR_HEIGHT, range(5), [0, 2, 4]
    )

    assert evaluation.kept_share < 1.0
    assert evaluation.rim_kept_share < evaluation.inside_kept_share
    assert evaluation.median_depth_noise > 1.0
    assert evaluation.median_views[1] < evaluation.median_views[0]


def test_counts_the_links_a_selection_breaks() -> None:
    # 4 m footprints stepping by 1 m overlap by 0.75; three steps apart, by 0.25.
    footprints, centres = survey_line([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    grid = grid_over(footprints, cell_m=0.25)

    evaluation = evaluate_selection(
        grid, footprints, centres, SEAFLOOR_HEIGHT, range(7), [0, 3, 6]
    )

    assert evaluation.weak_links == {0.5: (0, 2), 0.3: (0, 2)}
