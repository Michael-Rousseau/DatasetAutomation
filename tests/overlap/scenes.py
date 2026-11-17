"""Synthetic image pairs whose overlap is known exactly."""

import cv2
import numpy as np

from dataset_automation.camera.intrinsics import Intrinsics
from dataset_automation.camera.undistortion import undistort_to_pixels
from dataset_automation.reference.intrinsics import MERMAID_INTRINSICS
from tests.features.textures import textured_image

WIDTH, HEIGHT = 640, 480
# Room around the views so that warped content never runs out of scene.
MARGIN = 400
# Mermaid's lens, scaled down to 640 × 480 (focal and offsets / 6, same distortion coefficients).
SMALL_GOPRO = Intrinsics(
    width_px=WIDTH,
    height_px=HEIGHT,
    focal_px=MERMAID_INTRINSICS.focal_px / 6,
    cx_offset_px=0.0,
    cy_offset_px=0.0,
    k1=MERMAID_INTRINSICS.k1,
    k2=MERMAID_INTRINSICS.k2,
    k3=MERMAID_INTRINSICS.k3,
    p1=MERMAID_INTRINSICS.p1,
    p2=MERMAID_INTRINSICS.p2,
)


def scene(seed: int = 0) -> np.ndarray:
    return textured_image(HEIGHT + 2 * MARGIN, WIDTH + 2 * MARGIN, seed)


def offset() -> np.ndarray:
    return np.array([[1.0, 0.0, MARGIN], [0.0, 1.0, MARGIN], [0.0, 0.0, 1.0]])


def view_pair(a_to_b: np.ndarray, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Image A is the centre of `world`; a point at pixel p of A appears at a_to_b(p) in B."""
    image_a = world[MARGIN : MARGIN + HEIGHT, MARGIN : MARGIN + WIDTH]
    # B(p) = A(a_to_b⁻¹ p), read from the scene: warpPerspective wants the dst → src map.
    b_to_world = offset() @ np.linalg.inv(a_to_b)
    image_b = cv2.warpPerspective(
        world, b_to_world, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    )
    return image_a, image_b


def distorted_view(world: np.ndarray, shift_x: float) -> np.ndarray:
    """What SMALL_GOPRO sees of the scene, translated by `shift_x` undistorted pixels."""
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    pixels = np.column_stack([columns.ravel(), rows.ravel()]).astype(np.float64)
    ideal = undistort_to_pixels(SMALL_GOPRO, pixels) + [MARGIN + shift_x, MARGIN]
    map_x = ideal[:, 0].reshape(HEIGHT, WIDTH).astype(np.float32)
    map_y = ideal[:, 1].reshape(HEIGHT, WIDTH).astype(np.float32)
    return cv2.remap(world, map_x, map_y, cv2.INTER_CUBIC)
