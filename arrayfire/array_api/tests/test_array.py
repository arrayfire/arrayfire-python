import pytest

from arrayfire.array_api import Array, float32


def test_empty_array() -> None:
    array = Array()

    assert array.dtype == float32
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_array_from_1d_list() -> None:
    array = Array([1, 2, 3])

    assert array.dtype == float32
    assert array.ndim == 1
    assert array.size == 3
    assert array.shape == (3,)
    assert len(array) == 3


def test_array_from_2d_list() -> None:
    with pytest.raises(TypeError):
        Array([[1, 2, 3], [1, 2, 3]])
