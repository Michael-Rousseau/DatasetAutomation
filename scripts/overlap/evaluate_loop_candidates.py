"""How well image resemblance proposes the revisits of Mermaid, judged by the poses.

    uv run python scripts/overlap/evaluate_loop_candidates.py   # ~25 s once features are cached

One VLAD vector per image from its SIFT features, then for each image its most resembling
images that are not neighbours in time. No matching: this measures the proposals only. The
poses are used to judge, never to propose. SIFT features are cached in outputs/cache/.
"""

from pathlib import Path

import numpy as np

from dataset_automation.features.sift import SiftExtractor
from dataset_automation.features.store import load_or_extract
from dataset_automation.features.vlad import root_sift, train_codebook, vlad
from dataset_automation.overlap.loop_candidates import propose_far_pairs
from dataset_automation.reference.images import match_poses_to_images
from dataset_automation.reference.mermaid import load_mermaid_reference
from dataset_automation.reference.revisits import FarPairs, far_pairs

POSES_PATH = Path("data/mermaid/107177.xml")
IMAGES_DIRECTORY = Path("data/mermaid/images")
CACHE_DIRECTORY = Path("outputs/cache/sift_defaults")
# Images this close in time are neighbours: the sequential mode compares them.
MIN_GAP = 30
CODEBOOK_SAMPLES_PER_IMAGE = 120
CANDIDATE_COUNTS = (1, 5, 10, 20, 50)
ANGLE_BANDS_DEG = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 180))
MOST_CANDIDATES = max(CANDIDATE_COUNTS)


def main() -> None:
    reference = load_mermaid_reference(POSES_PATH)
    paths = images_in_capture_order(reference.poses.labels)
    vectors = image_vectors(paths)
    pairs = far_pairs(
        reference.footprints, reference.centres, reference.seafloor_height_m, MIN_GAP
    )
    is_revisit = pair_matrix(pairs, pairs.revisits(), len(paths))
    view_angle = np.full(is_revisit.shape, np.nan)
    view_angle[pairs.index_a, pairs.index_b] = pairs.view_angle_deg
    view_angle[pairs.index_b, pairs.index_a] = pairs.view_angle_deg
    ranking = ranked_far_images(vectors)

    far_pair_count = (len(paths) - MIN_GAP - 1) * (len(paths) - MIN_GAP) // 2
    print(
        f"{len(paths)} images, {int(pairs.revisits().sum())} revisits among "
        f"{far_pair_count} far pairs ({pairs.revisits().sum() / far_pair_count:.1%})"
    )
    print_by_candidate_count(ranking, is_revisit, view_angle)
    print_by_view_angle(ranking, pairs)


def images_in_capture_order(labels: list[str]) -> list[Path]:
    match = match_poses_to_images(labels, sorted(IMAGES_DIRECTORY.rglob("*")))
    if match.poses_without_image:
        raise SystemExit(
            f"{len(match.poses_without_image)} poses have no image: "
            "run scripts/fetch_mermaid.py --images"
        )
    return [match.matched[label] for label in labels]


def image_vectors(paths: list[Path]) -> np.ndarray:
    extractor = SiftExtractor()
    random = np.random.default_rng(0)
    descriptors = [
        root_sift(
            load_or_extract(path, CACHE_DIRECTORY, extractor).features.descriptors
        )
        for path in paths
    ]
    sample = np.vstack(
        [
            of_image[
                random.choice(
                    len(of_image),
                    min(CODEBOOK_SAMPLES_PER_IMAGE, len(of_image)),
                    replace=False,
                )
            ]
            for of_image in descriptors
        ]
    )
    codebook = train_codebook(sample)
    return np.stack([vlad(of_image, codebook) for of_image in descriptors])


def ranked_far_images(vectors: np.ndarray) -> np.ndarray:
    """Row i: the far images most resembling image i, most resembling first."""
    candidates = propose_far_pairs(vectors, MOST_CANDIDATES, MIN_GAP)
    similarity = np.full((len(vectors), len(vectors)), -np.inf)
    for candidate in candidates:
        similarity[candidate.index_a, candidate.index_b] = candidate.similarity
        similarity[candidate.index_b, candidate.index_a] = candidate.similarity
    return np.argsort(-similarity, axis=1, kind="stable")[:, :MOST_CANDIDATES]


def pair_matrix(pairs: FarPairs, selected: np.ndarray, image_count: int) -> np.ndarray:
    matrix = np.zeros((image_count, image_count), dtype=bool)
    matrix[pairs.index_a[selected], pairs.index_b[selected]] = True
    matrix[pairs.index_b[selected], pairs.index_a[selected]] = True
    return matrix


def print_by_candidate_count(
    ranking: np.ndarray, is_revisit: np.ndarray, view_angle: np.ndarray
) -> None:
    has_revisit = is_revisit.any(axis=1)
    rows = np.arange(len(ranking))[:, None]
    print("\ncandidates | images with a revisit among them | revisits | their angle")
    for count in CANDIDATE_COUNTS:
        proposed = ranking[:, :count]
        hits = is_revisit[rows, proposed]
        print(
            f"{count:10d} | {hits.any(axis=1)[has_revisit].mean():32.0%} "
            f"| {hits.mean():8.1%} "
            f"| {np.median(view_angle[rows, proposed][hits]):.0f} deg (median)"
        )


def print_by_view_angle(ranking: np.ndarray, pairs: FarPairs) -> None:
    image_count = len(ranking)
    rank = np.full((image_count, image_count), MOST_CANDIDATES)
    rank[np.arange(image_count)[:, None], ranking] = np.arange(MOST_CANDIDATES)
    revisits = pairs.revisits()
    index_a, index_b = pairs.index_a[revisits], pairs.index_b[revisits]
    angle = pairs.view_angle_deg[revisits]
    best_rank = np.minimum(rank[index_a, index_b], rank[index_b, index_a])
    print("\nangle between the views | revisits | in a top 10 | in a top 50")
    for low, high in ANGLE_BANDS_DEG:
        band = (angle >= low) & (angle < high)
        print(
            f"{low:14d}-{high:3d} deg | {band.sum():8d} "
            f"| {(best_rank[band] < 10).mean():11.0%} "
            f"| {(best_rank[band] < 50).mean():11.0%}"
        )


if __name__ == "__main__":
    main()
