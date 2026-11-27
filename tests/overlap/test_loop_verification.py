import numpy as np

from dataset_automation.features.matching import RatioTestMatcher
from dataset_automation.features.sift import SiftExtractor
from dataset_automation.overlap.loop_candidates import CandidatePair
from dataset_automation.overlap.loop_verification import verify_candidates
from dataset_automation.overlap.pairwise import UnknownReason
from dataset_automation.overlap.sequential import SequenceImage
from tests.overlap.scenes import HEIGHT, STEP_PX, WIDTH, scene, sliding_views

SIZE = (WIDTH, HEIGHT)
EXTRACTOR = SiftExtractor(scale=1.0)


class CountingLoader:
    def __init__(self, views: list[np.ndarray]) -> None:
        self.images = [SequenceImage(EXTRACTOR.extract(view), SIZE) for view in views]
        self.loaded: list[int] = []

    def __call__(self, position: int) -> SequenceImage:
        self.loaded.append(position)
        return self.images[position]


def candidate(index_a: int, index_b: int) -> CandidatePair:
    return CandidatePair(index_a, index_b, similarity=0.5, rank=0)


def test_candidate_seeing_the_same_place_gets_its_overlap() -> None:
    loader = CountingLoader(sliding_views(3, scene()))

    (measure,) = verify_candidates([candidate(0, 2)], loader, RatioTestMatcher())

    assert (measure.index_a, measure.index_b) == (0, 2)
    assert measure.estimate.unknown_reason is None
    assert measure.estimate.a_in_b is not None
    assert abs(measure.estimate.a_in_b - (1 - 2 * STEP_PX / WIDTH)) < 0.01


def test_candidate_seeing_another_place_is_unknown() -> None:
    (here,) = sliding_views(1, scene(seed=1))
    (elsewhere,) = sliding_views(1, scene(seed=2))
    loader = CountingLoader([here, elsewhere])

    (measure,) = verify_candidates([candidate(0, 1)], loader, RatioTestMatcher())

    assert measure.estimate.unknown_reason is UnknownReason.TOO_FEW_INLIERS
    assert measure.estimate.a_in_b is None


def test_first_image_is_loaded_once_for_consecutive_candidates() -> None:
    loader = CountingLoader(sliding_views(4, scene()))
    candidates = [candidate(0, 1), candidate(0, 2), candidate(0, 3), candidate(1, 3)]

    measures = list(verify_candidates(candidates, loader, RatioTestMatcher()))

    assert [(m.index_a, m.index_b) for m in measures] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
    ]
    assert loader.loaded == [0, 1, 2, 3, 1, 3]


def test_no_candidate_loads_nothing() -> None:
    loader = CountingLoader([])

    assert list(verify_candidates([], loader, RatioTestMatcher())) == []
    assert loader.loaded == []
