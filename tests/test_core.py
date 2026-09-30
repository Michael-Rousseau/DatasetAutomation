import pytest

from dataset_automation import _core


def test_add_returns_sum_computed_in_rust() -> None:
    assert _core.add(1.5, 2.0) == 3.5


def test_add_rejects_non_numeric_argument() -> None:
    with pytest.raises(TypeError):
        _core.add("a", 1.0)  # pyright: ignore[reportArgumentType] -- wrong type on purpose
