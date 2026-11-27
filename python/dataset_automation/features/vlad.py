"""One vector per image, to find the images that look alike without matching them.

Matching two images costs about 0.1 s; comparing two of these vectors is one dot product. They
propose the few far-apart images worth matching, among the hundreds of thousands of pairs of
a dive. VLAD (Jégou et al., 2010, "Aggregating local descriptors into a compact image
representation") summarises the local descriptors of an image against a small codebook; it
needs no training data other than the dive itself and runs on a CPU.

Changing this module affects persons A, B and C: it needs a review from all three.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WORD_COUNT = 64
KMEANS_ITERATIONS = 30
KMEANS_PRECISION = 1e-4
KMEANS_ATTEMPTS = 3


@dataclass(frozen=True, eq=False)
class Codebook:
    """`words` is (word count, descriptor size): the centres local descriptors are compared to.

    Vectors are only comparable when built with the same codebook.
    """

    words: np.ndarray


def save_codebook(path: Path, codebook: Codebook) -> None:
    """Kept next to a run: its image vectors cannot be compared or rebuilt without it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, codebook.words)


def load_codebook(path: Path) -> Codebook:
    return Codebook(words=np.load(path))


def root_sift(descriptors: np.ndarray) -> np.ndarray:
    """L1-normalise then square-root SIFT descriptors (Arandjelović & Zisserman, 2012).

    Euclidean distances between the results behave like the Hellinger distance between the
    original histograms, which stops a few large bins from dominating.
    """
    values = descriptors.astype(np.float32)
    totals = values.sum(axis=1, keepdims=True)
    return np.sqrt(values / np.maximum(totals, np.finfo(np.float32).tiny))


def train_codebook(
    descriptors: np.ndarray, word_count: int = DEFAULT_WORD_COUNT, seed: int = 0
) -> Codebook:
    """Cluster a sample of local descriptors, (M, D), taken across the whole dataset.

    The same descriptors and seed give the same codebook.
    """
    if len(descriptors) < word_count:
        raise ValueError(
            f"need at least {word_count} descriptors, got {len(descriptors)}"
        )
    # OpenCV's k-means draws its first centres from a global generator.
    cv2.setRNGSeed(seed)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_ITERATIONS,
        KMEANS_PRECISION,
    )
    samples = descriptors.astype(np.float32)
    # OpenCV wants an array to return the label of each sample in; they are not used here.
    labels = np.empty((len(samples), 1), dtype=np.int32)
    _, _, words = cv2.kmeans(
        samples,
        word_count,
        labels,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )
    return Codebook(words=words)


def vlad(descriptors: np.ndarray, codebook: Codebook) -> np.ndarray:
    """Unit vector of size word count × descriptor size; the dot product of two is their
    resemblance. An image without descriptors gives the zero vector, which resembles nothing.
    """
    words = codebook.words
    aggregated = np.zeros_like(words)
    if len(descriptors) == 0:
        return aggregated.ravel()

    values = descriptors.astype(np.float32)
    squared_distances = (
        (values**2).sum(axis=1)[:, None]
        - 2.0 * values @ words.T
        + (words**2).sum(axis=1)[None]
    )
    nearest = squared_distances.argmin(axis=1)
    np.add.at(aggregated, nearest, values - words[nearest])

    # Each word weighs the same, however many descriptors fell on it: a repeated texture
    # (sand, seagrass) would otherwise fill the vector on its own.
    aggregated /= np.maximum(
        np.linalg.norm(aggregated, axis=1, keepdims=True), np.finfo(np.float32).tiny
    )
    flat = aggregated.ravel()
    flat = np.sign(flat) * np.sqrt(np.abs(flat))
    return flat / max(float(np.linalg.norm(flat)), np.finfo(np.float32).tiny)
