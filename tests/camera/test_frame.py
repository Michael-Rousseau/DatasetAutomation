import numpy as np

from dataset_automation.camera.frame import image_border_pixels


def test_border_starts_at_top_left_corner_and_runs_clockwise() -> None:
    border = image_border_pixels(400, 300, samples_per_side=4)

    assert border.shape == (16, 2)
    np.testing.assert_allclose(border[0], [-0.5, -0.5])
    np.testing.assert_allclose(border[4], [399.5, -0.5])
    np.testing.assert_allclose(border[8], [399.5, 299.5])
    np.testing.assert_allclose(border[12], [-0.5, 299.5])
