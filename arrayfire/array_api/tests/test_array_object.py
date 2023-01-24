import pytest

from arrayfire.array_api import Array, float32, int16
from arrayfire.array_api._dtypes import supported_dtypes


def test_empty_array() -> None:
    array = Array()

    assert array.dtype == float32
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_empty_array_with_nonempty_dtype() -> None:
    array = Array(dtype=int16)

    assert array.dtype == int16
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_empty_array_with_nonempty_shape() -> None:
    array = Array(shape=(2, 3))

    assert array.dtype == float32
    assert array.ndim == 2
    assert array.size == 6
    assert array.shape == (2, 3)
    assert len(array) == 2


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


def test_array_from_list_with_unsupported_dtype() -> None:
    for dtype in supported_dtypes:
        if dtype == float32:
            continue
        with pytest.raises(TypeError):
            Array([1, 2, 3], dtype=dtype)


def test_array_from_af_array() -> None:
    array1 = Array([1])
    array2 = Array(array1)

    assert array1.dtype == array2.dtype == float32
    assert array1.ndim == array2.ndim == 1
    assert array1.size == array2.size == 1
    assert array1.shape == array2.shape == (1,)
    assert len(array1) == len(array2) == 1


def test_array_from_int_without_shape() -> None:
    with pytest.raises(TypeError):
        Array(1)


def test_array_from_int_without_dtype() -> None:
    with pytest.raises(TypeError):
        Array(1, shape=(1,))

# def test_array_from_int_with_parameters() -> None:  # BUG seg fault
#     array = Array(1, shape=(1,), dtype=float32)

#     assert array.dtype == float32
#     assert array.ndim == 1
#     assert array.size == 1
#     assert array.shape == (1,)
#     assert len(array) == 1


def test_array_from_unsupported_type() -> None:
    with pytest.raises(TypeError):
        Array((5, 5))  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        Array({1: 2, 3: 4})  # type: ignore[arg-type]
