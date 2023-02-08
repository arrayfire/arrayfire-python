from __future__ import annotations

import array as py_array
import ctypes
import enum
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

# TODO replace imports from original lib with refactored ones
from arrayfire import backend, safe_call
from arrayfire.algorithm import count
from arrayfire.array import _get_indices, _in_display_dims_limit

from .device import PointerSource
from .dtypes import CShape, Dtype
from .dtypes import bool as af_bool
from .dtypes import c_dim_t
from .dtypes import complex64 as af_complex64
from .dtypes import complex128 as af_complex128
from .dtypes import float32 as af_float32
from .dtypes import float64 as af_float64
from .dtypes import int64 as af_int64
from .dtypes import supported_dtypes, to_str
from .dtypes import uint64 as af_uint64

ShapeType = Tuple[int, ...]
# HACK, TODO replace for actual bcast_var after refactoring ~ https://github.com/arrayfire/arrayfire/pull/2871
_bcast_var = False

# TODO use int | float in operators -> remove bool | complex support


@dataclass
class _ArrayBuffer:
    address: Optional[int] = None
    length: int = 0


class Array:
    def __init__(
            self, x: Union[None, Array, py_array.array, int, ctypes.c_void_p, List[Union[int, float]]] = None,
            dtype: Union[None, Dtype, str] = None, shape: Optional[ShapeType] = None,
            pointer_source: PointerSource = PointerSource.host, offset: Optional[ctypes._SimpleCData[int]] = None,
            strides: Optional[ShapeType] = None) -> None:
        _no_initial_dtype = False  # HACK, FIXME

        # Initialise array object
        self.arr = ctypes.c_void_p(0)

        if isinstance(dtype, str):
            dtype = _str_to_dtype(dtype)  # type: ignore[arg-type]

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

            _type_char = dtype.typecode  # type: ignore[assignment]  # FIXME

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

    # Arithmetic Operators

    def __pos__(self) -> Array:
        """
        Evaluates +self_i for each element of an array instance.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the evaluated result for each element. The returned array must have the same data type
            as self.
        """
        return self

    def __neg__(self) -> Array:
        """
        Evaluates +self_i for each element of an array instance.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the evaluated result for each element in self. The returned array must have a data type
            determined by Type Promotion Rules.

        """
        return 0 - self  # type: ignore[no-any-return, operator]  # FIXME

    def __add__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the sum for each element of an array instance with the respective element of the array other.

        Parameters
        ----------
        self : Array
            Array instance (augend array). Should have a numeric data type.
        other: Union[int, float, Array]
            Addend array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise sums. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_add)

    def __sub__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the difference for each element of an array instance with the respective element of the array other.

        The result of self_i - other_i must be the same as self_i + (-other_i) and must be governed by the same
        floating-point rules as addition (see array.__add__()).

        Parameters
        ----------
        self : Array
            Array instance (minuend array). Should have a numeric data type.
        other: Union[int, float, Array]
            Subtrahend array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise differences. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_sub)

    def __mul__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the product for each element of an array instance with the respective element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise products. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_mul)

    def __truediv__(self, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates self_i / other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array should have a floating-point data type
            determined by Type Promotion Rules.

        Note
        ----
        - If one or both of self and other have integer data types, the result is implementation-dependent, as type
        promotion between data type “kinds” (e.g., integer versus floating-point) is unspecified.
        Specification-compliant libraries may choose to raise an error or return an array containing the element-wise
        results. If an array is returned, the array must have a real-valued floating-point data type.
        """
        return _process_c_function(self, other, backend.get().af_div)

    def __floordiv__(self, other: Union[int, float, Array], /) -> Array:
        # TODO
        return NotImplemented

    def __mod__(self, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates self_i % other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a real-valued data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. Each element-wise result must have the same sign as the
            respective element other_i. The returned array must have a real-valued floating-point data type determined
            by Type Promotion Rules.

        Note
        ----
        - For input arrays which promote to an integer data type, the result of division by zero is unspecified and
        thus implementation-defined.
        """
        return _process_c_function(self, other, backend.get().af_mod)

    def __pow__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of
        an array instance to the power of other_i (the exponent), where other_i is the corresponding element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array whose elements correspond to the exponentiation exponent. Must be compatible with self
            (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_pow)

    # Array Operators

    def __matmul__(self, other: Array, /) -> Array:
        # TODO get from blas - make vanilla version and not copy af.matmul as is
        return NotImplemented

    # Bitwise Operators

    def __invert__(self) -> Array:
        """
        Evaluates ~self_i for each element of an array instance.

        Parameters
        ----------
        self : Array
            Array instance. Should have an integer or boolean data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        out = Array()
        safe_call(backend.get().af_bitnot(ctypes.pointer(out.arr), self.arr))
        return out

    def __and__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates self_i & other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_bitand)

    def __or__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates self_i | other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_bitor)

    def __xor__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates self_i ^ other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.get().af_bitxor)

    def __lshift__(self, other: Union[int, Array], /) -> Array:
        """
        Evaluates self_i << other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.
            Each element must be greater than or equal to 0.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return _process_c_function(self, other, backend.get().af_bitshiftl)

    def __rshift__(self, other: Union[int, Array], /) -> Array:
        """
        Evaluates self_i >> other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.
            Each element must be greater than or equal to 0.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return _process_c_function(self, other, backend.get().af_bitshiftr)

    # Comparison Operators

    def __lt__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i < other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.get().af_lt)

    def __le__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i <= other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.get().af_le)

    def __gt__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i > other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.get().af_gt)

    def __ge__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i >= other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.get().af_ge)

    def __eq__(self, other: Union[int, float, bool, Array], /) -> Array:  # type: ignore[override]  # FIXME
        """
        Computes the truth value of self_i == other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). May have any data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.get().af_eq)

    def __ne__(self, other: Union[int, float, bool, Array], /) -> Array:  # type: ignore[override]  # FIXME
        """
        Computes the truth value of self_i != other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). May have any data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.get().af_neq)

    # Reflected Arithmetic Operators

    def __radd__(self, other: Array, /) -> Array:
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
        Return other % self.
        """
        return _process_c_function(other, self, backend.get().af_mod)

    def __rpow__(self, other: Array, /) -> Array:
        """
        Return other ** self.
        """
        return _process_c_function(other, self, backend.get().af_pow)

    # Reflected Array Operators

    def __rmatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    # Reflected Bitwise Operators

    def __rand__(self, other: Array, /) -> Array:
        """
        Return other & self.
        """
        return _process_c_function(other, self, backend.get().af_bitand)

    def __ror__(self, other: Array, /) -> Array:
        """
        Return other | self.
        """
        return _process_c_function(other, self, backend.get().af_bitor)

    def __rxor__(self, other: Array, /) -> Array:
        """
        Return other ^ self.
        """
        return _process_c_function(other, self, backend.get().af_bitxor)

    def __rlshift__(self, other: Array, /) -> Array:
        """
        Return other << self.
        """
        return _process_c_function(other, self, backend.get().af_bitshiftl)

    def __rrshift__(self, other: Array, /) -> Array:
        """
        Return other >> self.
        """
        return _process_c_function(other, self, backend.get().af_bitshiftr)

    # In-place Arithmetic Operators

    def __iadd__(self, other: Union[int, float, Array], /) -> Array:
        # TODO discuss either we need to support complex and bool as other input type
        """
        Return self += other.
        """
        return _process_c_function(self, other, backend.get().af_add)

    def __isub__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self -= other.
        """
        return _process_c_function(self, other, backend.get().af_sub)

    def __imul__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self *= other.
        """
        return _process_c_function(self, other, backend.get().af_mul)

    def __itruediv__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self /= other.
        """
        return _process_c_function(self, other, backend.get().af_div)

    def __ifloordiv__(self, other: Union[int, float, Array], /) -> Array:
        # TODO
        return NotImplemented

    def __imod__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self %= other.
        """
        return _process_c_function(self, other, backend.get().af_mod)

    def __ipow__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self **= other.
        """
        return _process_c_function(self, other, backend.get().af_pow)

    # In-place Array Operators

    def __imatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    # In-place Bitwise Operators

    def __iand__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Return self &= other.
        """
        return _process_c_function(self, other, backend.get().af_bitand)

    def __ior__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Return self |= other.
        """
        return _process_c_function(self, other, backend.get().af_bitor)

    def __ixor__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Return self ^= other.
        """
        return _process_c_function(self, other, backend.get().af_bitxor)

    def __ilshift__(self, other: Union[int, Array], /) -> Array:
        """
        Return self <<= other.
        """
        return _process_c_function(self, other, backend.get().af_bitshiftl)

    def __irshift__(self, other: Union[int, Array], /) -> Array:
        """
        Return self >>= other.
        """
        return _process_c_function(self, other, backend.get().af_bitshiftr)

    # Methods

    def __abs__(self) -> Array:
        # TODO
        return NotImplemented

    def __array_namespace__(self, *, api_version: Optional[str] = None) -> Any:
        # TODO
        return NotImplemented

    def __bool__(self) -> bool:
        # TODO consider using scalar() and is_scalar()
        return NotImplemented

    def __complex__(self) -> complex:
        # TODO
        return NotImplemented

    def __dlpack__(self, *, stream: Union[None, int, Any] = None):  # type: ignore[no-untyped-def]
        # TODO implementation and expected return type -> PyCapsule
        return NotImplemented

    def __dlpack_device__(self) -> Tuple[enum.Enum, int]:
        # TODO
        return NotImplemented

    def __float__(self) -> float:
        # TODO
        return NotImplemented

    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice, ], ...], Array], /) -> Array:
        """
        Returns self[key].

        Parameters
        ----------
        self : Array
            Array instance.
        key : Union[int, slice, Tuple[Union[int, slice, ], ...], Array]
            Index key.

        Returns
        -------
        out : Array
            An array containing the accessed value(s). The returned array must have the same data type as self.
        """
        # TODO
        # API Specification - key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array].
        # consider using af.span to replace ellipsis during refactoring
        out = Array()
        ndims = self.ndim

        if isinstance(key, Array) and key == af_bool.c_api_value:
            ndims = 1
            if count(key) == 0:
                return out

        safe_call(backend.get().af_index_gen(
            ctypes.pointer(out.arr), self.arr, c_dim_t(ndims), _get_indices(key).pointer))
        return out

    def __index__(self) -> int:
        # TODO
        return NotImplemented

    def __int__(self) -> int:
        # TODO
        return NotImplemented

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __setitem__(
            self, key: Union[int, slice, Tuple[Union[int, slice, ], ...], Array],
            value: Union[int, float, bool, Array], /) -> None:
        # TODO
        return NotImplemented  # type: ignore[return-value]  # FIXME

    def __str__(self) -> str:
        # TODO change the look of array str. E.g., like np.array
        if not _in_display_dims_limit(self.shape):
            return _metadata_string(self.dtype, self.shape)

        return _metadata_string(self.dtype) + _array_as_str(self)

    def __repr__(self) -> str:
        # return _metadata_string(self.dtype, self.shape)
        # TODO change the look of array representation. E.g., like np.array
        return _array_as_str(self)

    def to_device(self, device: Any, /, *, stream: Union[int, Any] = None) -> Array:
        # TODO implementation and change device type from Any to Device
        return NotImplemented

    # Attributes

    @property
    def dtype(self) -> Dtype:
        """
        Data type of the array elements.

        Returns
        -------
        out : Dtype
            Array data type.
        """
        out = ctypes.c_int()
        safe_call(backend.get().af_get_type(ctypes.pointer(out), self.arr))
        return _c_api_value_to_dtype(out.value)

    @property
    def device(self) -> Any:
        # TODO
        return NotImplemented

    @property
    def mT(self) -> Array:
        # TODO
        return NotImplemented

    @property
    def T(self) -> Array:
        """
        Transpose of the array.

        Returns
        -------
        out : Array
            Two-dimensional array whose first and last dimensions (axes) are permuted in reverse order relative to
            original array. The returned array must have the same data type as the original array.

        Note
        ----
        - The array instance must be two-dimensional. If the array instance is not two-dimensional, an error
        should be raised.
        """
        if self.ndim < 2:
            raise TypeError(f"Array should be at least 2-dimensional. Got {self.ndim}-dimensional array")

        # TODO add check if out.dtype == self.dtype
        out = Array()
        safe_call(backend.get().af_transpose(ctypes.pointer(out.arr), self.arr, False))
        return out

    @property
    def size(self) -> int:
        """
        Number of elements in an array.

        Returns
        -------
        out : int
            Number of elements in an array

        Note
        ----
        - This must equal the product of the array's dimensions.
        """
        # NOTE previously - elements()
        out = c_dim_t(0)
        safe_call(backend.get().af_get_elements(ctypes.pointer(out), self.arr))
        return out.value

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).

        out : int
            Number of array dimensions (axes).
        """
        out = ctypes.c_uint(0)
        safe_call(backend.get().af_get_numdims(ctypes.pointer(out), self.arr))
        return out.value

    @property
    def shape(self) -> ShapeType:
        """
        Array dimensions.

        Returns
        -------
        out : tuple[int, ...]
            Array dimensions.
        """
        # TODO refactor
        d0 = c_dim_t(0)
        d1 = c_dim_t(0)
        d2 = c_dim_t(0)
        d3 = c_dim_t(0)
        safe_call(backend.get().af_get_dims(
            ctypes.pointer(d0), ctypes.pointer(d1), ctypes.pointer(d2), ctypes.pointer(d3), self.arr))
        return (d0.value, d1.value, d2.value, d3.value)[:self.ndim]  # Skip passing None values

    def scalar(self) -> Union[None, int, float, bool, complex]:
        """
        Return the first element of the array
        """
        # TODO change the logic of this method
        if self.is_empty():
            return None

        out = self.dtype.c_type()
        safe_call(backend.get().af_get_scalar(ctypes.pointer(out), self.arr))
        return out.value  # type: ignore[no-any-return]  # FIXME

    def is_empty(self) -> bool:
        """
        Check if the array is empty i.e. it has no elements.
        """
        out = ctypes.c_bool()
        safe_call(backend.get().af_is_empty(ctypes.pointer(out), self.arr))
        return out.value

    def to_list(self, row_major: bool = False) -> List[Union[None, int, float, bool, complex]]:
        if self.is_empty():
            return []

        array = _reorder(self) if row_major else self
        ctypes_array = _get_ctypes_array(array)

        if array.ndim == 1:
            return list(ctypes_array)

        out = []
        for i in range(array.size):
            idx = i
            sub_list = []
            for j in range(array.ndim):
                div = array.shape[j]
                sub_list.append(idx % div)
                idx //= div
            out.append(ctypes_array[sub_list[::-1]])  # type: ignore[call-overload]  # FIXME
        return out

    def to_ctype_array(self, row_major: bool = False) -> ctypes.Array:
        if self.is_empty():
            raise RuntimeError("Can not convert an empty array to ctype.")

        array = _reorder(self) if row_major else self
        return _get_ctypes_array(array)


def _get_ctypes_array(array: Array) -> ctypes.Array:
    c_shape = array.dtype.c_type * array.size
    ctypes_array = c_shape()
    safe_call(backend.get().af_get_data_ptr(ctypes.pointer(ctypes_array), array.arr))
    return ctypes_array


def _reorder(array: Array) -> Array:
    """
    Returns a reordered array to help interoperate with row major formats.
    """
    if array.ndim == 1:
        return array

    out = Array()
    c_shape = CShape(*(tuple(reversed(range(array.ndim))) + tuple(range(array.ndim, 4))))
    safe_call(backend.get().af_reorder(ctypes.pointer(out.arr), array.arr, *c_shape))
    return out


def _array_as_str(array: Array) -> str:
    arr_str = ctypes.c_char_p(0)
    # FIXME add description to passed arguments
    safe_call(backend.get().af_array_to_string(ctypes.pointer(arr_str), "", array.arr, 4, True))
    py_str = to_str(arr_str)
    safe_call(backend.get().af_free_host(arr_str))
    return py_str


def _metadata_string(dtype: Dtype, dims: Optional[ShapeType] = None) -> str:
    return (
        "arrayfire.Array()\n"
        f"Type: {dtype.typename}\n"
        f"Dims: {str(dims) if dims else ''}")


def _get_cshape(shape: Optional[ShapeType], buffer_length: int) -> CShape:
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
        lhs: Union[int, float, Array], rhs: Union[int, float, Array],
        c_function: Any) -> Array:
    out = Array()

    if isinstance(lhs, Array) and isinstance(rhs, Array):
        lhs_array = lhs.arr
        rhs_array = rhs.arr

    elif isinstance(lhs, Array) and isinstance(rhs, (int, float)):
        rhs_dtype = _implicit_dtype(rhs, lhs.dtype)
        rhs_constant_array = _constant_array(rhs, CShape(*lhs.shape), rhs_dtype)

        lhs_array = lhs.arr
        rhs_array = rhs_constant_array.arr

    elif isinstance(lhs, (int, float)) and isinstance(rhs, Array):
        lhs_dtype = _implicit_dtype(lhs, rhs.dtype)
        lhs_constant_array = _constant_array(lhs, CShape(*rhs.shape), lhs_dtype)

        lhs_array = lhs_constant_array.arr
        rhs_array = rhs.arr

    else:
        raise TypeError(f"{type(rhs)} is not supported and can not be passed to C binary function.")

    safe_call(c_function(ctypes.pointer(out.arr), lhs_array, rhs_array, _bcast_var))

    return out


def _implicit_dtype(value: Union[int, float], array_dtype: Dtype) -> Dtype:
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


def _constant_array(value: Union[int, float], cshape: CShape, dtype: Dtype) -> Array:
    out = Array()

    if isinstance(value, complex):
        if dtype != af_complex64 and dtype != af_complex128:
            dtype = af_complex64

        safe_call(backend.get().af_constant_complex(
            ctypes.pointer(out.arr), ctypes.c_double(value.real), ctypes.c_double(value.imag), 4,
            ctypes.pointer(cshape.c_array), dtype.c_api_value))
    elif dtype == af_int64:
        # TODO discuss workaround for passing float to ctypes
        safe_call(backend.get().af_constant_long(
            ctypes.pointer(out.arr), af_int64.c_type(value.real), 4, ctypes.pointer(cshape.c_array)))
    elif dtype == af_uint64:
        safe_call(backend.get().af_constant_ulong(
            ctypes.pointer(out.arr), af_uint64.c_type(value.real), 4, ctypes.pointer(cshape.c_array)))
    else:
        safe_call(backend.get().af_constant(
            ctypes.pointer(out.arr), ctypes.c_double(value), 4, ctypes.pointer(cshape.c_array), dtype.c_api_value))

    return out
