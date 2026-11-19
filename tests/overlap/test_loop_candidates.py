import numpy as np
import pytest

from dataset_automation.overlap.loop_candidates import CandidatePair, propose_far_pairs


def unit(*angles_deg: float) -> np.ndarray:
    """One 2D unit vector per angle: the closer two angles, the more the images resemble."""
    angles = np.radians(angles_deg)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def pairs(candidates: list[CandidatePair]) -> list[tuple[int, int]]:
    return [(candidate.index_a, candidate.index_b) for candidate in candidates]


def test_proposes_the_most_resembling_far_image() -> None:
    # A dive going out and coming back: image 4 looks like image 0, image 3 like image 1.
    vectors = unit(0, 30, 90, 32, 2)

    candidates = propose_far_pairs(vectors, candidates_per_image=1, min_gap=1)

    assert (0, 4) in pairs(candidates)
    assert (1, 3) in pairs(candidates)


def test_never_proposes_neighbours_in_time() -> None:
    # Neighbours resemble most, as in a real sequence.
    vectors = unit(0, 1, 2, 3, 40, 41, 42, 43)

    candidates = propose_far_pairs(vectors, candidates_per_image=3, min_gap=2)

    assert all(c.index_b - c.index_a > 2 for c in candidates)


def test_a_pair_proposed_by_both_images_appears_once() -> None:
    vectors = unit(0, 90, 180, 1)

    candidates = propose_far_pairs(vectors, candidates_per_image=1, min_gap=1)

    assert pairs(candidates).count((0, 3)) == 1


def test_records_the_resemblance_and_the_best_rank_of_each_pair() -> None:
    vectors = unit(0, 90, 60, 10)

    candidates = propose_far_pairs(vectors, candidates_per_image=2, min_gap=0)
    by_pair = {(c.index_a, c.index_b): c for c in candidates}

    assert by_pair[0, 3].similarity == pytest.approx(np.cos(np.radians(10)))
    assert by_pair[0, 3].rank == 0
    # Image 2 is second for image 0, but first for image 1.
    assert by_pair[0, 2].rank == 1
    assert by_pair[1, 2].rank == 0


def test_pairs_are_ordered_by_position() -> None:
    vectors = unit(0, 50, 100, 5, 55, 105)

    candidates = propose_far_pairs(vectors, candidates_per_image=2, min_gap=1)

    assert pairs(candidates) == sorted(pairs(candidates))
    assert all(c.index_a < c.index_b for c in candidates)


def test_sequence_shorter_than_the_gap_has_no_candidate() -> None:
    assert propose_far_pairs(unit(0, 1, 2), candidates_per_image=5, min_gap=10) == []


def test_asking_for_more_candidates_than_far_images_returns_them_all() -> None:
    candidates = propose_far_pairs(unit(0, 10, 20, 30), 10, min_gap=1)

    assert pairs(candidates) == [(0, 2), (0, 3), (1, 3)]


@pytest.mark.parametrize("count, gap", [(0, 5), (3, -1)])
def test_rejects_meaningless_settings(count: int, gap: int) -> None:
    with pytest.raises(ValueError):
        propose_far_pairs(unit(0, 1), count, gap)
