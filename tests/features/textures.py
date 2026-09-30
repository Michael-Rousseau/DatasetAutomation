import cv2
import numpy as np


def textured_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Blurred noise: blob-like texture on which SIFT finds many distinctive keypoints."""
    noise = np.random.default_rng(seed).integers(
        0, 256, size=(height, width), dtype=np.uint8
    )
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0)
    stretched = (blurred - blurred.min()) * (255.0 / (blurred.max() - blurred.min()))
    return stretched.astype(np.uint8)
