import array as pyarray
from typing import Any

import pytest

from arrayfire.array_api import Array, float32, int16
from arrayfire.array_api._dtypes import supported_dtypes

# TODO change separated methods with setup and teardown to avoid code duplication
# TODO add tests for array arguments: device, offset, strides
# TODO add tests for all supported dtypes on initialisation


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


class TestArithmeticOperators:
    def setup_method(self, method: Any) -> None:
        self.list = [1, 2, 3]
        self.const_int = 2
        self.const_float = 1.5
        self.array = Array(self.list)
        self.array_other = Array([9, 9, 9])

        self.tuple = (1, 2, 3)
        self.const_str = "15"

    def teardown_method(self, method: Any) -> None:
        self.array = Array(self.list)

    def test_add_int(self) -> None:
        res = self.array + self.const_int
        assert res[0].scalar() == 3
        assert res[1].scalar() == 4
        assert res[2].scalar() == 5

    # Test __add__, __iadd__, __radd__

    def test_add_float(self) -> None:
        res = self.array + self.const_float
        assert res[0].scalar() == 2.5
        assert res[1].scalar() == 3.5
        assert res[2].scalar() == 4.5

    def test_add_array(self) -> None:
        res = self.array + self.array_other
        assert res[0].scalar() == 10
        assert res[1].scalar() == 11
        assert res[2].scalar() == 12

    def test_add_inplace_and_reflected(self) -> None:
        res = self.array + self.const_int
        ires = self.array
        ires += self.const_int
        rres = self.const_int + self.array  # type: ignore[operator]

        assert res[0].scalar() == ires[0].scalar() == rres[0].scalar() == 3
        assert res[1].scalar() == ires[1].scalar() == rres[1].scalar() == 4
        assert res[2].scalar() == ires[2].scalar() == rres[2].scalar() == 5

        assert res.dtype == ires.dtype == rres.dtype
        assert res.ndim == ires.ndim == rres.ndim
        assert res.size == ires.size == ires.size
        assert res.shape == ires.shape == rres.shape
        assert len(res) == len(ires) == len(rres)

    def test_add_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            Array([1, 2, 3]) + self.const_str  # type: ignore[operator]

        with pytest.raises(TypeError):
            Array([1, 2, 3]) + self.tuple  # type: ignore[operator]

    # Test __sub__, __isub__, __rsub__

    def test_sub_int(self) -> None:
        res = self.array - self.const_int
        assert res[0].scalar() == -1
        assert res[1].scalar() == 0
        assert res[2].scalar() == 1

    def test_sub_float(self) -> None:
        res = self.array - self.const_float
        assert res[0].scalar() == -0.5
        assert res[1].scalar() == 0.5
        assert res[2].scalar() == 1.5

    def test_sub_arr(self) -> None:
        res = self.array - self.array_other
        assert res[0].scalar() == -8
        assert res[1].scalar() == -7
        assert res[2].scalar() == -6

    def test_sub_inplace_and_reflected(self) -> None:
        res = self.array - self.const_int
        ires = self.array
        ires -= self.const_int
        rres = self.const_int - self.array  # type: ignore[operator]

        assert res[0].scalar() == ires[0].scalar() == rres[0].scalar() == -1
        assert res[1].scalar() == ires[1].scalar() == rres[1].scalar() == 0
        assert res[2].scalar() == ires[2].scalar() == rres[2].scalar() == 1

        assert res.dtype == ires.dtype == rres.dtype
        assert res.ndim == ires.ndim == rres.ndim
        assert res.size == ires.size == ires.size
        assert res.shape == ires.shape == rres.shape
        assert len(res) == len(ires) == len(rres)

    def test_sub_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            self.array - self.const_str  # type: ignore[operator]

        with pytest.raises(TypeError):
            self.array - self.tuple  # type: ignore[operator]

    # Test __mul__, __imul__, __rmul__

    def test_mul_int(self) -> None:
        res = self.array * self.const_int
        assert res[0].scalar() == 2
        assert res[1].scalar() == 4
        assert res[2].scalar() == 6

    def test_mul_float(self) -> None:
        res = self.array * self.const_float
        assert res[0].scalar() == 1.5
        assert res[1].scalar() == 3
        assert res[2].scalar() == 4.5

    def test_mul_array(self) -> None:
        res = self.array * self.array_other
        assert res[0].scalar() == 9
        assert res[1].scalar() == 18
        assert res[2].scalar() == 27

    def test_mul_inplace_and_reflected(self) -> None:
        res = self.array * self.const_int
        ires = self.array
        ires *= self.const_int
        rres = self.const_int * self.array  # type: ignore[operator]

        assert res[0].scalar() == ires[0].scalar() == rres[0].scalar() == 2
        assert res[1].scalar() == ires[1].scalar() == rres[1].scalar() == 4
        assert res[2].scalar() == ires[2].scalar() == rres[2].scalar() == 6

        assert res.dtype == ires.dtype == rres.dtype
        assert res.ndim == ires.ndim == rres.ndim
        assert res.size == ires.size == ires.size
        assert res.shape == ires.shape == rres.shape
        assert len(res) == len(ires) == len(rres)

    def test_mul_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            self.array * self.const_str  # type: ignore[operator]

        with pytest.raises(TypeError):
            self.array * self.tuple  # type: ignore[operator]

    # Test __truediv__, __itruediv__, __rtruediv__

    def test_truediv_int(self) -> None:
        res = self.array / self.const_int
        assert res[0].scalar() == 0.5
        assert res[1].scalar() == 1
        assert res[2].scalar() == 1.5

    def test_truediv_float(self) -> None:
        res = self.array / self.const_float
        assert round(res[0].scalar(), 5) == 0.66667  # type: ignore[arg-type]
        assert round(res[1].scalar(), 5) == 1.33333  # type: ignore[arg-type]
        assert res[2].scalar() == 2

    def test_truediv_array(self) -> None:
        res = self.array / self.array_other
        assert round(res[0].scalar(), 5) == 0.11111  # type: ignore[arg-type]
        assert round(res[1].scalar(), 5) == 0.22222  # type: ignore[arg-type]
        assert round(res[2].scalar(), 5) == 0.33333  # type: ignore[arg-type]

    def test_truediv_inplace_and_reflected(self) -> None:
        res = self.array / self.const_int
        ires = self.array
        ires /= self.const_int
        rres = self.const_int / self.array  # type: ignore[operator]

        assert res[0].scalar() == ires[0].scalar() == 0.5
        assert res[1].scalar() == ires[1].scalar() == 1
        assert res[2].scalar() == ires[2].scalar() == 1.5

        assert rres[0].scalar() == 2
        assert rres[1].scalar() == 1
        assert round(rres[2].scalar(), 5) == 0.66667

        assert res.dtype == ires.dtype == rres.dtype
        assert res.ndim == ires.ndim == rres.ndim
        assert res.size == ires.size == ires.size
        assert res.shape == ires.shape == rres.shape
        assert len(res) == len(ires) == len(rres)

    def test_truediv_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            self.array / self.const_str  # type: ignore[operator]

        with pytest.raises(TypeError):
            self.array / self.tuple  # type: ignore[operator]

    # TODO
    # Test __floordiv__, __ifloordiv__, __rfloordiv__

    # Test __mod__, __imod__, __rmod__

    def test_mod_int(self) -> None:
        res = self.array % self.const_int
        assert res[0].scalar() == 1
        assert res[1].scalar() == 0
        assert res[2].scalar() == 1

    def test_mod_float(self) -> None:
        res = self.array % self.const_float
        assert res[0].scalar() == 1.0
        assert res[1].scalar() == 0.5
        assert res[2].scalar() == 0.0

    def test_mod_array(self) -> None:
        res = self.array % self.array_other
        assert res[0].scalar() == 1.0
        assert res[1].scalar() == 2.0
        assert res[2].scalar() == 3.0

    def test_mod_inplace_and_reflected(self) -> None:
        res = self.array % self.const_int
        ires = self.array
        ires %= self.const_int
        rres = self.const_int % self.array  # type: ignore[operator]

        assert res[0].scalar() == ires[0].scalar() == 1
        assert res[1].scalar() == ires[1].scalar() == 0
        assert res[2].scalar() == ires[2].scalar() == 1

        assert rres[0].scalar() == 0
        assert rres[1].scalar() == 0
        assert rres[2].scalar() == 2

        assert res.dtype == ires.dtype == rres.dtype
        assert res.ndim == ires.ndim == rres.ndim
        assert res.size == ires.size == ires.size
        assert res.shape == ires.shape == rres.shape
        assert len(res) == len(ires) == len(rres)

    def test_mod_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            self.array % self.const_str  # type: ignore[operator]

        with pytest.raises(TypeError):
            self.array % self.tuple  # type: ignore[operator]

    # Test __pow__, __ipow__, __rpow__

    def test_pow_int(self) -> None:
        res = self.array ** self.const_int
        assert res[0].scalar() == 1
        assert res[1].scalar() == 4
        assert res[2].scalar() == 9

    def test_pow_float(self) -> None:
        res = self.array ** self.const_float
        assert res[0].scalar() == 1
        assert round(res[1].scalar(), 5) == 2.82843  # type: ignore[arg-type]
        assert round(res[2].scalar(), 5) == 5.19615  # type: ignore[arg-type]

    def test_pow_array(self) -> None:
        res = self.array ** self.array_other
        assert res[0].scalar() == 1
        assert res[1].scalar() == 512
        assert res[2].scalar() == 19683

    def test_pow_inplace_and_reflected(self) -> None:
        res = self.array ** self.const_int
        ires = self.array
        ires **= self.const_int
        rres = self.const_int ** self.array  # type: ignore[operator]

        assert res[0].scalar() == ires[0].scalar() == 1
        assert res[1].scalar() == ires[1].scalar() == 4
        assert res[2].scalar() == ires[2].scalar() == 9

        assert rres[0].scalar() == 2
        assert rres[1].scalar() == 4
        assert rres[2].scalar() == 8

        assert res.dtype == ires.dtype == rres.dtype
        assert res.ndim == ires.ndim == rres.ndim
        assert res.size == ires.size == ires.size
        assert res.shape == ires.shape == rres.shape
        assert len(res) == len(ires) == len(rres)

    def test_pow_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            self.array % self.const_str  # type: ignore[operator]

        with pytest.raises(TypeError):
            self.array % self.tuple  # type: ignore[operator]
