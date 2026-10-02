import numpy as np


def image_border_pixels(
    width_px: int, height_px: int, samples_per_side: int
) -> np.ndarray:
    """Closed loop along the outer edge of the image, clockwise, as (4·samples_per_side, 2).

    Coordinates follow OpenCV (pixel centres at integers), so the outer edge runs from −0.5
    to size − 0.5. Each side starts at a corner and stops before the next one. Sampling the
    sides, not only the corners, matters once the border is undistorted: straight image edges
    become curves.
    """
    left, top = -0.5, -0.5
    right, bottom = width_px - 0.5, height_px - 0.5
    steps = np.linspace(0.0, 1.0, samples_per_side, endpoint=False)
    top_edge = np.column_stack(
        [left + steps * (right - left), np.full_like(steps, top)]
    )
    right_edge = np.column_stack(
        [np.full_like(steps, right), top + steps * (bottom - top)]
    )
    bottom_edge = np.column_stack(
        [right - steps * (right - left), np.full_like(steps, bottom)]
    )
    left_edge = np.column_stack(
        [np.full_like(steps, left), bottom - steps * (bottom - top)]
    )
    return np.concatenate([top_edge, right_edge, bottom_edge, left_edge])
