import array as pyarray
import math
from typing import Any, Optional, Tuple

import pytest

from arrayfire.array_api.array_object import Array
from arrayfire.array_api.dtypes import Dtype, float32, int16

# TODO add tests for array arguments: device, offset, strides
# TODO add tests for all supported dtypes on initialisation
# TODO add test generation


@pytest.mark.parametrize(
    "array, res_dtype, res_ndim, res_size, res_shape, res_len", [
        (Array(), float32, 0, 0, (), 0),
        (Array(dtype=int16), int16, 0, 0, (), 0),
        (Array(dtype="short int"), int16, 0, 0, (), 0),
        (Array(dtype="h"), int16, 0, 0, (), 0),
        (Array(shape=(2, 3)), float32, 2, 6, (2, 3), 2),
        (Array([1, 2, 3]), float32, 1, 3, (3,), 3),
        (Array(pyarray.array("f", [1, 2, 3])), float32, 1, 3, (3,), 3),
        (Array([1], shape=(1,), dtype=float32), float32, 1, 1, (1,), 1),  # BUG
        (Array(Array([1])), float32, 1, 1, (1,), 1)
    ])
def test_initialization_with_different_arguments(
        array: Array, res_dtype: Dtype, res_ndim: int, res_size: int, res_shape: Tuple[int, ...],
        res_len: int) -> None:
    assert array.dtype == res_dtype
    assert array.ndim == res_ndim
    assert array.size == res_size
    # NOTE math.prod from empty object returns 1, but it should work for other cases
    if res_size != 0:
        assert array.size == math.prod(res_shape)
    assert array.shape == res_shape
    assert len(array) == res_len


@pytest.mark.parametrize(
    "array_object, dtype, shape", [
        (None, "hello world", ()),
        ([[1, 2, 3], [1, 2, 3]], None, ()),
        (1, None, ()),
        (1, None, (1,)),
        ((5, 5), None, ()),
        ({1: 2, 3: 4}, None, ())
    ]
)
def test_initalization_with_unsupported_argument_types(
        array_object: Any, dtype: Optional[Dtype], shape: Tuple[int, ...]) -> None:
    with pytest.raises(TypeError):
        Array(x=array_object, dtype=dtype, shape=shape)
