from __future__ import annotations

import array as py_array
import ctypes
from dataclasses import dataclass

from arrayfire import backend, safe_call  # TODO refactoring
from arrayfire.array import _in_display_dims_limit  # TODO refactoring

from ._dtypes import CShape, Dtype, c_dim_t, float32, supported_dtypes
from ._utils import Device, PointerSource, to_str

ShapeType = tuple[int, ...]


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

    # Initialisation
    arr = ctypes.c_void_p(0)

    def __init__(
            self, x: None | Array | py_array.array | int | ctypes.c_void_p | list = None, dtype: None | Dtype = None,
            pointer_source: PointerSource = PointerSource.host, shape: None | ShapeType = None,
            offset: None | ctypes._SimpleCData[int] = None, strides: None | ShapeType = None) -> None:
        _no_initial_dtype = False  # HACK, FIXME

        if isinstance(dtype, str):
            dtype = _str_to_dtype(dtype)

        if dtype is None:
            _no_initial_dtype = True
            dtype = float32

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

        return _metadata_string(self.dtype) + self._as_str()

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
        # return 0 - self
        raise NotImplementedError

    def __add__(self, other: int | float | Array, /) -> Array:
        """
        Return self + other.
        """
        # return _binary_func(self, other, backend.get().af_add)  # TODO
        raise NotImplementedError

    @property
    def dtype(self) -> Dtype:
        out = ctypes.c_int()
        safe_call(backend.get().af_get_type(ctypes.pointer(out), self.arr))
        return _c_api_value_to_dtype(out.value)

    @property
    def device(self) -> Device:
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

    def _as_str(self) -> str:
        arr_str = ctypes.c_char_p(0)
        # FIXME add description to passed arguments
        safe_call(backend.get().af_array_to_string(ctypes.pointer(arr_str), "", self.arr, 4, True))
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

# TODO
# def _binary_func(lhs: int | float | Array, rhs: int | float | Array, c_func: Any) -> Array:  # TODO replace Any
#     out = Array()
#     other = rhs

#     if is_number(rhs):
#         ldims = _fill_dim4_tuple(lhs.shape)
#         rty = implicit_dtype(rhs, lhs.type())
#         other = Array()
#         other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], rty.value)
#     elif not isinstance(rhs, Array):
#         raise TypeError("Invalid parameter to binary function")

#     safe_call(c_func(c_pointer(out.arr), lhs.arr, other.arr, _bcast_var.get()))

#     return out


# TODO replace candidate below
# def dim4_to_tuple(shape: ShapeType, default: int=1) -> ShapeType:
#     assert(isinstance(dims, tuple))

#     if (default is not None):
#         assert(is_number(default))

#     out = [default]*4

#     for i, dim in enumerate(dims):
#         out[i] = dim

#     return tuple(out)

# def _fill_dim4_tuple(shape: ShapeType) -> tuple[int, ...]:
#     out = tuple([1 if value is None else value for value in shape])
#     if len(out) == 4:
#         return out

#     return out + (1,)*(4-len(out))