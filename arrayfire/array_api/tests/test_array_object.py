import array as pyarray

import pytest

from arrayfire.array_api import Array, float32, int16
from arrayfire.array_api._dtypes import supported_dtypes

# TODO change separated methods with setup and teardown to avoid code duplication
# TODO add tests for array arguments: device, offset, strides


def test_create_empty_array() -> None:
    array = Array()

    assert array.dtype == float32
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_create_empty_array_with_nonempty_dtype() -> None:
    array = Array(dtype=int16)

    assert array.dtype == int16
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_create_empty_array_with_str_dtype() -> None:
    array = Array(dtype="short int")

    assert array.dtype == int16
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_create_empty_array_with_literal_dtype() -> None:
    array = Array(dtype="h")

    assert array.dtype == int16
    assert array.ndim == 0
    assert array.size == 0
    assert array.shape == ()
    assert len(array) == 0


def test_create_empty_array_with_not_matching_str_dtype() -> None:
    with pytest.raises(TypeError):
        Array(dtype="hello world")


def test_create_empty_array_with_nonempty_shape() -> None:
    array = Array(shape=(2, 3))

    assert array.dtype == float32
    assert array.ndim == 2
    assert array.size == 6
    assert array.shape == (2, 3)
    assert len(array) == 2


def test_create_array_from_1d_list() -> None:
    array = Array([1, 2, 3])

    assert array.dtype == float32
    assert array.ndim == 1
    assert array.size == 3
    assert array.shape == (3,)
    assert len(array) == 3


def test_create_array_from_2d_list() -> None:
    with pytest.raises(TypeError):
        Array([[1, 2, 3], [1, 2, 3]])


def test_create_array_from_pyarray() -> None:
    py_array = pyarray.array("f", [1, 2, 3])
    array = Array(py_array)

    assert array.dtype == float32
    assert array.ndim == 1
    assert array.size == 3
    assert array.shape == (3,)
    assert len(array) == 3


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


def test_array_getitem() -> None:
    array = Array([1, 2, 3, 4, 5])

    int_item = array[2]
    assert array.dtype == int_item.dtype
    assert int_item.scalar() == 3

    # TODO add more tests for different dtypes


def test_scalar() -> None:
    array = Array([1, 2, 3])
    assert array[1].scalar() == 2


def test_scalar_is_empty() -> None:
    array = Array()
    assert array.scalar() is None


def test_array_to_list() -> None:
    array = Array([1, 2, 3])
    assert array.to_list() == [1, 2, 3]


def test_array_to_list_is_empty() -> None:
    array = Array()
    assert array.to_list() == []


def test_array_add() -> None:
    array = Array([1, 2, 3])
    res = array + 1
    assert res[0].scalar() == 2
    assert res[1].scalar() == 3
    assert res[2].scalar() == 4

    res = array + 1.5
    assert res[0].scalar() == 2.5
    assert res[1].scalar() == 3.5
    assert res[2].scalar() == 4.5

    res = array + Array([9, 9, 9])
    assert res[0].scalar() == 10
    assert res[1].scalar() == 11
    assert res[2].scalar() == 12


def test_array_add_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Array([1, 2, 3]) + "15"  # type: ignore[operator]


def test_array_sub() -> None:
    array = Array([1, 2, 3])
    res = array - 1
    assert res[0].scalar() == 0
    assert res[1].scalar() == 1
    assert res[2].scalar() == 2

    res = array - 1.5
    assert res[0].scalar() == -0.5
    assert res[1].scalar() == 0.5
    assert res[2].scalar() == 1.5

    res = array - Array([9, 9, 9])
    assert res[0].scalar() == -8
    assert res[1].scalar() == -7
    assert res[2].scalar() == -6


def test_array_mul() -> None:
    array = Array([1, 2, 3])
    res = array * 2
    assert res[0].scalar() == 2
    assert res[1].scalar() == 4
    assert res[2].scalar() == 6

    res = array * 1.5
    assert res[0].scalar() == 1.5
    assert res[1].scalar() == 3
    assert res[2].scalar() == 4.5

    res = array * Array([9, 9, 9])
    assert res[0].scalar() == 9
    assert res[1].scalar() == 18
    assert res[2].scalar() == 27


def test_array_truediv() -> None:
    array = Array([1, 2, 3])
    res = array / 2
    assert res[0].scalar() == 0.5
    assert res[1].scalar() == 1
    assert res[2].scalar() == 1.5

    res = array / 1.5
    assert round(res[0].scalar(), 5) == 0.66667  # type: ignore[arg-type]
    assert round(res[1].scalar(), 5) == 1.33333  # type: ignore[arg-type]
    assert res[2].scalar() == 2

    res = array / Array([2, 2, 2])
    assert res[0].scalar() == 0.5
    assert res[1].scalar() == 1
    assert res[2].scalar() == 1.5


def test_array_floordiv() -> None:
    # TODO add test after implementation of __floordiv__
    pass


def test_array_mod() -> None:
    array = Array([1, 2, 3])
    res = array % 2
    assert res[0].scalar() == 1
    assert res[1].scalar() == 0
    assert res[2].scalar() == 1

    res = array % 1.5
    assert res[0].scalar() == 1.0
    assert res[1].scalar() == 0.5
    assert res[2].scalar() == 0.0

    res = array % Array([9, 9, 9])
    assert res[0].scalar() == 1.0
    assert res[1].scalar() == 2.0
    assert res[2].scalar() == 3.0


def test_array_pow() -> None:
    array = Array([1, 2, 3])
    res = array ** 2
    assert res[0].scalar() == 1
    assert res[1].scalar() == 4
    assert res[2].scalar() == 9

    res = array ** 1.5
    assert res[0].scalar() == 1
    assert round(res[1].scalar(), 5) == 2.82843  # type: ignore[arg-type]
    assert round(res[2].scalar(), 5) == 5.19615  # type: ignore[arg-type]

    res = array ** Array([9, 9, 9])
    assert res[0].scalar() == 1
    assert res[1].scalar() == 512
    assert res[2].scalar() == 19683
