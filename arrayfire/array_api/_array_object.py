from __future__ import annotations

import array as py_array
import ctypes
from dataclasses import dataclass
from typing import Any

from arrayfire import backend, safe_call  # TODO refactor
from arrayfire.algorithm import count  # TODO refactor
from arrayfire.array import _get_indices, _in_display_dims_limit  # TODO refactor

from ._dtypes import CShape, Dtype
from ._dtypes import bool as af_bool
from ._dtypes import c_dim_t
from ._dtypes import complex64 as af_complex64
from ._dtypes import complex128 as af_complex128
from ._dtypes import float32 as af_float32
from ._dtypes import float64 as af_float64
from ._dtypes import int64 as af_int64
from ._dtypes import supported_dtypes
from ._dtypes import uint64 as af_uint64
from ._utils import PointerSource, is_number, to_str

ShapeType = tuple[int, ...]
_bcast_var = False  # HACK, TODO replace for actual bcast_var after refactoring


@dataclass
class _ArrayBuffer:
    address: int | None = None
    length: int = 0


class Array:
    # Numpy checks this attribute to know which class handles binary builtin operations, such as __add__.
    # Setting to such a high value should make sure that arrayfire has priority over
    # other classes, ensuring that e.g. numpy.float32(1)*arrayfire.randu(3) is handled by
    # arrayfire's __radd__() instead of numpy's __add__()
    __array_priority__ = 30

    def __init__(
            self, x: None | Array | py_array.array | int | ctypes.c_void_p | list = None, dtype: None | Dtype = None,
            pointer_source: PointerSource = PointerSource.host, shape: None | ShapeType = None,
            offset: None | ctypes._SimpleCData[int] = None, strides: None | ShapeType = None) -> None:
        _no_initial_dtype = False  # HACK, FIXME

        # Initialise array object
        self.arr = ctypes.c_void_p(0)

        if isinstance(dtype, str):
            dtype = _str_to_dtype(dtype)

        if dtype is None:
            _no_initial_dtype = True
            dtype = af_float32

        if x is None:
            if not shape:  # shape is None or empty tuple
                safe_call(backend.get().af_create_handle(
                    ctypes.pointer(self.arr), 0, ctypes.pointer(CShape().c_array), dtype.c_api_value))
                return

            # NOTE: applies inplace changes for self.arr
            safe_call(backend.get().af_create_handle(
                ctypes.pointer(self.arr), len(shape), ctypes.pointer(CShape(*shape).c_array), dtype.c_api_value))
            return

        if isinstance(x, Array):
            safe_call(backend.get().af_retain_array(ctypes.pointer(self.arr), x.arr))
            return

        if isinstance(x, py_array.array):
            _type_char = x.typecode
            _array_buffer = _ArrayBuffer(*x.buffer_info())

        elif isinstance(x, list):
            _array = py_array.array("f", x)  # BUG [True, False] -> dtype: f32   # TODO add int and float
            _type_char = _array.typecode
            _array_buffer = _ArrayBuffer(*_array.buffer_info())

        elif isinstance(x, int) or isinstance(x, ctypes.c_void_p):  # TODO
            _array_buffer = _ArrayBuffer(x if not isinstance(x, ctypes.c_void_p) else x.value)

            if not shape:
                raise TypeError("Expected to receive the initial shape due to the x being a data pointer.")

            if _no_initial_dtype:
                raise TypeError("Expected to receive the initial dtype due to the x being a data pointer.")

            _type_char = dtype.typecode

        else:
            raise TypeError("Passed object x is an object of unsupported class.")

        _cshape = _get_cshape(shape, _array_buffer.length)

        if not _no_initial_dtype and dtype.typecode != _type_char:
            raise TypeError("Can not create array of requested type from input data type")

        if not (offset or strides):
            if pointer_source == PointerSource.host:
                safe_call(backend.get().af_create_array(
                    ctypes.pointer(self.arr), ctypes.c_void_p(_array_buffer.address), _cshape.original_shape,
                    ctypes.pointer(_cshape.c_array), dtype.c_api_value))
                return

            safe_call(backend.get().af_device_array(
                ctypes.pointer(self.arr), ctypes.c_void_p(_array_buffer.address), _cshape.original_shape,
                ctypes.pointer(_cshape.c_array), dtype.c_api_value))
            return

        if offset is None:
            offset = c_dim_t(0)

        if strides is None:
            strides = (1, _cshape[0], _cshape[0]*_cshape[1], _cshape[0]*_cshape[1]*_cshape[2])

        if len(strides) < 4:
            strides += (strides[-1], ) * (4 - len(strides))
        strides_cshape = CShape(*strides).c_array

        safe_call(backend.get().af_create_strided_array(
            ctypes.pointer(self.arr), ctypes.c_void_p(_array_buffer.address), offset, _cshape.original_shape,
            ctypes.pointer(_cshape.c_array), ctypes.pointer(strides_cshape), dtype.c_api_value,
            pointer_source.value))

    def __str__(self) -> str:  # FIXME
        if not _in_display_dims_limit(self.shape):
            return _metadata_string(self.dtype, self.shape)

        return _metadata_string(self.dtype) + _array_as_str(self)

    def __repr__(self) -> str:  # FIXME
        return _metadata_string(self.dtype, self.shape)

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0  # type: ignore[return-value]

    def __pos__(self) -> Array:
        """
        Return +self
        """
        return self

    def __neg__(self) -> Array:
        """
        Return -self
        """
        return 0 - self

    def __add__(self, other: int | float | Array, /) -> Array:
        # TODO discuss either we need to support complex and bool as other input type
        """
        Return self + other.
        """
        return _process_c_function(self, other, backend.get().af_add)

    def __sub__(self, other: int | float | bool | complex | Array, /) -> Array:
        """
        Return self - other.
        """
        return _process_c_function(self, other, backend.get().af_sub)

    def __mul__(self, other: int | float | bool | complex | Array, /) -> Array:
        """
        Return self * other.
        """
        return _process_c_function(self, other, backend.get().af_mul)

    def __truediv__(self, other: int | float | bool | complex | Array, /) -> Array:
        """
        Return self / other.
        """
        return _process_c_function(self, other, backend.get().af_div)

    def __floordiv__(self, other: int | float | bool | complex | Array, /) -> Array:
        # TODO
        return NotImplemented

    def __mod__(self, other: int | float | bool | complex | Array, /) -> Array:
        """
        Return self % other.
        """
        return _process_c_function(self, other, backend.get().af_mod)

    def __pow__(self, other: int | float | bool | complex | Array, /) -> Array:
        """
        Return self ** other.
        """
        return _process_c_function(self, other, backend.get().af_pow)

    def __matmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    def __radd__(self, other: Array, /) -> Array:
        # TODO discuss either we need to support complex and bool as other input type
        """
        Return other + self.
        """
        return _process_c_function(other, self, backend.get().af_add)

    def __rsub__(self, other: Array, /) -> Array:
        """
        Return other - self.
        """
        return _process_c_function(other, self, backend.get().af_sub)

    def __rmul__(self, other: Array, /) -> Array:
        """
        Return other * self.
        """
        return _process_c_function(other, self, backend.get().af_mul)

    def __rtruediv__(self, other: Array, /) -> Array:
        """
        Return other / self.
        """
        return _process_c_function(other, self, backend.get().af_div)

    def __rfloordiv__(self, other:  Array, /) -> Array:
        # TODO
        return NotImplemented

    def __rmod__(self, other: Array, /) -> Array:
        """
        Return other / self.
        """
        return _process_c_function(other, self, backend.get().af_mod)

    def __rpow__(self, other: Array, /) -> Array:
        """
        Return other ** self.
        """
        return _process_c_function(other, self, backend.get().af_pow)

    def __rmatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    def __getitem__(self, key: int | slice | tuple[int | slice] | Array, /) -> Array:
        # TODO: API Specification - key: int | slice | ellipsis | tuple[int | slice] | Array
        # TODO: refactor
        out = Array()
        ndims = self.ndim

        if isinstance(key, Array) and key == af_bool.c_api_value:
            ndims = 1
            if count(key) == 0:
                return out

        safe_call(backend.get().af_index_gen(
            ctypes.pointer(out.arr), self.arr, c_dim_t(ndims), _get_indices(key).pointer))
        return out

    @property
    def dtype(self) -> Dtype:
        out = ctypes.c_int()
        safe_call(backend.get().af_get_type(ctypes.pointer(out), self.arr))
        return _c_api_value_to_dtype(out.value)

    @property
    def device(self) -> Any:
        raise NotImplementedError

    @property
    def mT(self) -> Array:
        # TODO
        raise NotImplementedError

    @property
    def T(self) -> Array:
        # TODO
        raise NotImplementedError

    @property
    def size(self) -> None | int:
        # NOTE previously - elements()
        out = c_dim_t(0)
        safe_call(backend.get().af_get_elements(ctypes.pointer(out), self.arr))
        return out.value

    @property
    def ndim(self) -> int:
        nd = ctypes.c_uint(0)
        safe_call(backend.get().af_get_numdims(ctypes.pointer(nd), self.arr))
        return nd.value

    @property
    def shape(self) -> ShapeType:
        """
        Return the shape of the array as a tuple.
        """
        # TODO refactor
        d0 = c_dim_t(0)
        d1 = c_dim_t(0)
        d2 = c_dim_t(0)
        d3 = c_dim_t(0)
        safe_call(backend.get().af_get_dims(
            ctypes.pointer(d0), ctypes.pointer(d1), ctypes.pointer(d2), ctypes.pointer(d3), self.arr))
        return (d0.value, d1.value, d2.value, d3.value)[:self.ndim]  # Skip passing None values

    def scalar(self) -> int | float | bool | complex:
        """
        Return the first element of the array
        """
        # BUG seg fault on empty array
        out = self.dtype.c_type()
        safe_call(backend.get().af_get_scalar(ctypes.pointer(out), self.arr))
        return out.value  # type: ignore[no-any-return]  # FIXME


def _array_as_str(array: Array) -> str:
    arr_str = ctypes.c_char_p(0)
    # FIXME add description to passed arguments
    safe_call(backend.get().af_array_to_string(ctypes.pointer(arr_str), "", array.arr, 4, True))
    py_str = to_str(arr_str)
    safe_call(backend.get().af_free_host(arr_str))
    return py_str


def _metadata_string(dtype: Dtype, dims: None | ShapeType = None) -> str:
    return (
        "arrayfire.Array()\n"
        f"Type: {dtype.typename}\n"
        f"Dims: {str(dims) if dims else ''}")


def _get_cshape(shape: None | ShapeType, buffer_length: int) -> CShape:
    if shape:
        return CShape(*shape)

    if buffer_length != 0:
        return CShape(buffer_length)

    raise RuntimeError("Shape and buffer length are size invalid.")


def _c_api_value_to_dtype(value: int) -> Dtype:
    for dtype in supported_dtypes:
        if value == dtype.c_api_value:
            return dtype

    raise TypeError("There is no supported dtype that matches passed dtype C API value.")


def _str_to_dtype(value: int) -> Dtype:
    for dtype in supported_dtypes:
        if value == dtype.typecode or value == dtype.typename:
            return dtype

    raise TypeError("There is no supported dtype that matches passed dtype typecode.")


def _process_c_function(
        target: Array, other: int | float | bool | complex | Array, c_function: Any) -> Array:
    out = Array()

    if isinstance(other, Array):
        safe_call(c_function(ctypes.pointer(out.arr), target.arr, other.arr, _bcast_var))
    elif is_number(other):
        other_dtype = _implicit_dtype(other, target.dtype)
        other_array = _constant_array(other, CShape(*target.shape), other_dtype)
        safe_call(c_function(ctypes.pointer(out.arr), target.arr, other_array.arr, _bcast_var))
    else:
        raise TypeError(f"{type(other)} is not supported and can not be passed to C binary function.")

    return out


def _implicit_dtype(value: int | float | bool | complex, array_dtype: Dtype) -> Dtype:
    if isinstance(value, bool):
        value_dtype = af_bool
    if isinstance(value, int):
        value_dtype = af_int64
    elif isinstance(value, float):
        value_dtype = af_float64
    elif isinstance(value, complex):
        value_dtype = af_complex128
    else:
        raise TypeError(f"{type(value)} is not supported and can not be converted to af.Dtype.")

    if not (array_dtype == af_float32 or array_dtype == af_complex64):
        return value_dtype

    if value_dtype == af_float64:
        return af_float32

    if value_dtype == af_complex128:
        return af_complex64

    return value_dtype


def _constant_array(value: int | float | bool | complex, shape: CShape, dtype: Dtype) -> Array:
    out = Array()

    if isinstance(value, complex):
        if dtype != af_complex64 and dtype != af_complex128:
            dtype = af_complex64

        safe_call(backend.get().af_constant_complex(
            ctypes.pointer(out.arr), ctypes.c_double(value.real), ctypes.c_double(value.imag), 4,
            ctypes.pointer(shape.c_array), dtype.c_api_value))
    elif dtype == af_int64:
        safe_call(backend.get().af_constant_long(
            ctypes.pointer(out.arr), ctypes.c_longlong(value.real), 4, ctypes.pointer(shape.c_array)))
    elif dtype == af_uint64:
        safe_call(backend.get().af_constant_ulong(
            ctypes.pointer(out.arr), ctypes.c_ulonglong(value.real), 4, ctypes.pointer(shape.c_array)))
    else:
        safe_call(backend.get().af_constant(
            ctypes.pointer(out.arr), ctypes.c_double(value), 4, ctypes.pointer(shape.c_array), dtype.c_api_value))

    return out
