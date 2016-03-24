#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
arrayfire.Array class and helper functions.
"""

import inspect
from .library import *
from .util import *
from .util import _is_number
from .bcast import _bcast_var
from .base import *
from .index import *
from .index import _Index4

def _create_array(buf, numdims, idims, dtype, is_device):
    out_arr = ct.c_void_p(0)
    c_dims = dim4(idims[0], idims[1], idims[2], idims[3])
    if (not is_device):
        safe_call(backend.get().af_create_array(ct.pointer(out_arr), ct.c_void_p(buf),
                                                numdims, ct.pointer(c_dims), dtype.value))
    else:
        safe_call(backend.get().af_device_array(ct.pointer(out_arr), ct.c_void_p(buf),
                                                numdims, ct.pointer(c_dims), dtype.value))
    return out_arr

def _create_strided_array(buf, numdims, idims, dtype, is_device, offset, strides):
    out_arr = ct.c_void_p(0)
    c_dims = dim4(idims[0], idims[1], idims[2], idims[3])
    if offset is None:
        offset = 0
    offset = c_dim_t(offset)
    if strides is None:
        strides = (1, idims[0], idims[0]*idims[1], idims[0]*idims[1]*idims[2])
    while len(strides) < 4:
        strides = strides + (strides[-1],)
    strides = dim4(strides[0], strides[1], strides[2], strides[3])
    if is_device:
        location = Source.device
    else:
        location = Source.host
    safe_call(backend.get().af_create_strided_array(ct.pointer(out_arr), ct.c_void_p(buf),
                                                    offset, numdims, ct.pointer(c_dims),
                                                    ct.pointer(strides), dtype.value,
                                                    location.value))
    return out_arr

def _create_empty_array(numdims, idims, dtype):
    out_arr = ct.c_void_p(0)
    c_dims = dim4(idims[0], idims[1], idims[2], idims[3])
    safe_call(backend.get().af_create_handle(ct.pointer(out_arr),
                                             numdims, ct.pointer(c_dims), dtype.value))
    return out_arr

def constant_array(val, d0, d1=None, d2=None, d3=None, dtype=Dtype.f32):
    """
    Internal function to create a C array. Should not be used externall.
    """

    if not isinstance(dtype, ct.c_int):
        if isinstance(dtype, int):
            dtype = ct.c_int(dtype)
        elif isinstance(dtype, Dtype):
            dtype = ct.c_int(dtype.value)
        else:
            raise TypeError("Invalid dtype")

    out = ct.c_void_p(0)
    dims = dim4(d0, d1, d2, d3)

    if isinstance(val, complex):
        c_real = ct.c_double(val.real)
        c_imag = ct.c_double(val.imag)

        if (dtype.value != Dtype.c32.value and dtype.value != Dtype.c64.value):
            dtype = Dtype.c32.value

        safe_call(backend.get().af_constant_complex(ct.pointer(out), c_real, c_imag,
                                                    4, ct.pointer(dims), dtype))
    elif dtype.value == Dtype.s64.value:
        c_val = ct.c_longlong(val.real)
        safe_call(backend.get().af_constant_long(ct.pointer(out), c_val, 4, ct.pointer(dims)))
    elif dtype.value == Dtype.u64.value:
        c_val = ct.c_ulonglong(val.real)
        safe_call(backend.get().af_constant_ulong(ct.pointer(out), c_val, 4, ct.pointer(dims)))
    else:
        c_val = ct.c_double(val)
        safe_call(backend.get().af_constant(ct.pointer(out), c_val, 4, ct.pointer(dims), dtype))

    return out


def _binary_func(lhs, rhs, c_func):
    out = Array()
    other = rhs

    if (_is_number(rhs)):
        ldims = dim4_to_tuple(lhs.dims())
        rty = implicit_dtype(rhs, lhs.type())
        other = Array()
        other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], rty.value)
    elif not isinstance(rhs, Array):
        raise TypeError("Invalid parameter to binary function")

    safe_call(c_func(ct.pointer(out.arr), lhs.arr, other.arr, _bcast_var.get()))

    return out

def _binary_funcr(lhs, rhs, c_func):
    out = Array()
    other = lhs

    if (_is_number(lhs)):
        rdims = dim4_to_tuple(rhs.dims())
        lty = implicit_dtype(lhs, rhs.type())
        other = Array()
        other.arr = constant_array(lhs, rdims[0], rdims[1], rdims[2], rdims[3], lty.value)
    elif not isinstance(lhs, Array):
        raise TypeError("Invalid parameter to binary function")

    c_func(ct.pointer(out.arr), other.arr, rhs.arr, _bcast_var.get())

    return out

def _ctype_to_lists(ctype_arr, dim, shape, offset=0):
    if (dim == 0):
        return list(ctype_arr[offset : offset + shape[0]])
    else:
        dim_len = shape[dim]
        res = [[]] * dim_len
        for n in range(dim_len):
            res[n] = _ctype_to_lists(ctype_arr, dim - 1, shape, offset)
            offset += shape[0]
        return res

def _slice_to_length(key, dim):
    tkey = [key.start, key.stop, key.step]

    if tkey[0] is None:
        tkey[0] = 0
    elif tkey[0] < 0:
        tkey[0] = dim - tkey[0]

    if tkey[1] is None:
        tkey[1] = dim
    elif tkey[1] < 0:
        tkey[1] = dim - tkey[1]

    if tkey[2] is None:
        tkey[2] = 1

    return int(((tkey[1] - tkey[0] - 1) / tkey[2]) + 1)

def _get_info(dims, buf_len):
    elements = 1
    numdims = len(dims)
    idims = [1]*4

    for i in range(numdims):
        elements *= dims[i]
        idims[i] = dims[i]

    if (elements == 0):
        if (buf_len != 0):
            idims = [buf_len, 1, 1, 1]
            numdims = 1
        else:
            raise RuntimeError("Invalid size")

    return numdims, idims


def _get_indices(key):

    S = Index(slice(None))
    inds = _Index4(S, S, S, S)

    if isinstance(key, tuple):
        n_idx = len(key)
        for n in range(n_idx):
            inds[n] = Index(key[n])
    else:
        inds[0] = Index(key)

    return inds

def _get_assign_dims(key, idims):

    dims = [1]*4

    for n in range(len(idims)):
        dims[n] = idims[n]

    if _is_number(key):
        dims[0] = 1
        return dims
    elif isinstance(key, slice):
        dims[0] = _slice_to_length(key, idims[0])
        return dims
    elif isinstance(key, ParallelRange):
        dims[0] = _slice_to_length(key.S, idims[0])
        return dims
    elif isinstance(key, BaseArray):
        # If the array is boolean take only the number of nonzeros
        if(key.dtype() is Dtype.b8):
            dims[0] = int(sum(key))
        else:
            dims[0] = key.elements()
        return dims
    elif isinstance(key, tuple):
        n_inds = len(key)

        for n in range(n_inds):
            if (_is_number(key[n])):
                dims[n] = 1
            elif (isinstance(key[n], BaseArray)):
                # If the array is boolean take only the number of nonzeros
                if(key[n].dtype() is Dtype.b8):
                    dims[n] = int(sum(key[n]))
                else:
                    dims[n] = key[n].elements()
            elif (isinstance(key[n], slice)):
                dims[n] = _slice_to_length(key[n], idims[n])
            elif (isinstance(key[n], ParallelRange)):
                dims[n] = _slice_to_length(key[n].S, idims[n])
            else:
                raise IndexError("Invalid type while assigning to arrayfire.array")

        return dims
    else:
        raise IndexError("Invalid type while assigning to arrayfire.array")


def transpose(a, conj=False):
    """
    Perform the transpose on an input.

    Parameters
    -----------
    a : af.Array
        Multi dimensional arrayfire array.

    conj : optional: bool. default: False.
           Flag to specify if a complex conjugate needs to applied for complex inputs.

    Returns
    --------
    out : af.Array
          Containing the tranpose of `a` for all batches.

    """
    out = Array()
    safe_call(backend.get().af_transpose(ct.pointer(out.arr), a.arr, conj))
    return out

def transpose_inplace(a, conj=False):
    """
    Perform inplace transpose on an input.

    Parameters
    -----------
    a : af.Array
        - Multi dimensional arrayfire array.
        - Contains transposed values on exit.

    conj : optional: bool. default: False.
           Flag to specify if a complex conjugate needs to applied for complex inputs.

    Note
    -------
    Input `a` needs to be a square matrix or a batch of square matrices.

    """
    safe_call(backend.get().af_transpose_inplace(a.arr, conj))

class Array(BaseArray):

    """
    A multi dimensional array container.

    Parameters
    ----------
    src : optional: array.array, list or C buffer. default: None.
         - When `src` is `array.array` or `list`, the data is copied to create the Array()
         - When `src` is None, an empty buffer is created.

    dims : optional: tuple of ints. default: (0,)
         - When using the default values of `dims`, the dims are caclulated as `len(src)`

    dtype: optional: str or arrayfire.Dtype. default: None.
           - if str, must be one of the following:
               - 'f' for float
               - 'd' for double
               - 'b' for bool
               - 'B' for unsigned char
               - 'i' for signed 32 bit integer
               - 'I' for unsigned 32 bit integer
               - 'l' for signed 64 bit integer
               - 'L' for unsigned 64 bit integer
               - 'F' for 32 bit complex number
               - 'D' for 64 bit complex number
           - if arrayfire.Dtype, must be one of the following:
               - Dtype.f32 for float
               - Dtype.f64 for double
               - Dtype.b8  for bool
               - Dtype.u8  for unsigned char
               - Dtype.s32 for signed 32 bit integer
               - Dtype.u32 for unsigned 32 bit integer
               - Dtype.s64 for signed 64 bit integer
               - Dtype.u64 for unsigned 64 bit integer
               - Dtype.c32 for 32 bit complex number
               - Dtype.c64 for 64 bit complex number
            - if None, Dtype.f32 is assumed

    Attributes
    -----------
    arr: ctypes.c_void_p
         ctypes variable containing af_array from arrayfire library.

    Examples
    --------

    Creating an af.Array() from array.array()

    >>> import arrayfire as af
    >>> import array
    >>> a = array.array('f', (1, 2, 3, 4))
    >>> b = af.Array(a, (2,2))
    >>> af.display(b)
    [2 2 1 1]
        1.0000     3.0000
        2.0000     4.0000

    Creating an af.Array() from a list

    >>> import arrayfire as af
    >>> import array
    >>> a = [1, 2, 3, 4]
    >>> b = af.Array(a)
    >>> af.display(b)
    [4 1 1 1]
        1.0000
        2.0000
        3.0000
        4.0000

    Creating an af.Array() from numpy.array()

    >>> import numpy as np
    >>> import arrayfire as af
    >>> a = np.random.random((2,2))
    >>> a
    array([[ 0.33042524,  0.36135449],
           [ 0.86748649,  0.42199135]])
    >>> b = af.Array(a.ctypes.data, a.shape, a.dtype.char)
    >>> af.display(b)
    [2 2 1 1]
        0.3304     0.8675
        0.3614     0.4220

    Note
    -----
    - The class is currently limited to 4 dimensions.
    - arrayfire.Array() uses column major format.
    - numpy uses row major format by default which can cause issues during conversion

    """

    # Numpy checks this attribute to know which class handles binary builtin operations, such as __add__.
    # Setting to such a high value should make sure that arrayfire has priority over
    # other classes, ensuring that e.g. numpy.float32(1)*arrayfire.randu(3) is handled by
    # arrayfire's __radd__() instead of numpy's __add__()
    __array_priority__ = 30

    def __init__(self, src=None, dims=(0,), dtype=None, is_device=False, offset=None, strides=None):

        super(Array, self).__init__()

        buf=None
        buf_len=0

        if dtype is not None:
            if isinstance(dtype, str):
                type_char = dtype
            else:
                type_char = to_typecode[dtype.value]
        else:
            type_char = None

        _type_char='f'

        if src is not None:

            if (isinstance(src, Array)):
                safe_call(backend.get().af_retain_array(ct.pointer(self.arr), src.arr))
                return

            host = __import__("array")

            if isinstance(src, host.array):
                buf,buf_len = src.buffer_info()
                _type_char = src.typecode
                numdims, idims = _get_info(dims, buf_len)
            elif isinstance(src, list):
                tmp = host.array('f', src)
                buf,buf_len = tmp.buffer_info()
                _type_char = tmp.typecode
                numdims, idims = _get_info(dims, buf_len)
            elif isinstance(src, int) or isinstance(src, ct.c_void_p):
                buf = src if not isinstance(src, ct.c_void_p) else src.value

                numdims, idims = _get_info(dims, buf_len)

                elements = 1
                for dim in idims:
                    elements *= dim

                if (elements == 0):
                    raise RuntimeError("Expected dims when src is data pointer")

                if (type_char is None):
                    raise TypeError("Expected type_char when src is data pointer")

                _type_char = type_char

            else:
                raise TypeError("src is an object of unsupported class")

            if (type_char is not None and
                type_char != _type_char):
                raise TypeError("Can not create array of requested type from input data type")
            if(offset is None and strides is None):
                self.arr = _create_array(buf, numdims, idims, to_dtype[_type_char], is_device)
            else:
                self.arr = _create_strided_array(buf, numdims, idims, to_dtype[_type_char], is_device, offset, strides)

        else:

            if type_char is None:
                type_char = 'f'

            numdims = len(dims)
            idims = [1] * 4
            for n in range(numdims):
                idims[n] = dims[n]
            self.arr = _create_empty_array(numdims, idims, to_dtype[type_char])

    def as_type(self, ty):
        """
        Cast current array to a specified data type

        Parameters
        ----------
        ty : Return data type
        """
        return cast(self, ty)

    def copy(self):
        """
        Performs a deep copy of the array.

        Returns
        -------
        out: af.Array()
             An identical copy of self.
        """
        out = Array()
        safe_call(backend.get().af_copy_array(ct.pointer(out.arr), self.arr))
        return out

    def __del__(self):
        """
        Release the C array when going out of scope
        """
        if self.arr.value:
            backend.get().af_release_array(self.arr)
            self.arr.value = 0

    def device_ptr(self):
        """
        Return the device pointer exclusively held by the array.

        Returns
        ------
        ptr : int
              Contains location of the device pointer

        Note
        ----
        - This can be used to integrate with custom C code and / or PyCUDA or PyOpenCL.
        - No other arrays will share the same device pointer.
        - A copy of the memory is done if multiple arrays share the same memory or the array is not the owner of the memory.
        - In case of a copy the return value points to the newly allocated memory which is now exclusively owned by the array.
        """
        ptr = ct.c_void_p(0)
        backend.get().af_get_device_ptr(ct.pointer(ptr), self.arr)
        return ptr.value

    def raw_ptr(self):
        """
        Return the device pointer held by the array.

        Returns
        ------
        ptr : int
              Contains location of the device pointer

        Note
        ----
        - This can be used to integrate with custom C code and / or PyCUDA or PyOpenCL.
        - No mem copy is peformed, this function returns the raw device pointer.
        - This pointer may be shared with other arrays. Use this function with caution.
        - In particular the JIT compiler will not be aware of the shared arrays.
        - This results in JITed operations not being immediately visible through the other array.
        """
        ptr = ct.c_void_p(0)
        backend.get().af_get_raw_ptr(ct.pointer(ptr), self.arr)
        return ptr.value

    def offset(self):
        """
        Return the offset, of the first element relative to the raw pointer.

        Returns
        ------
        offset : int
                 The offset in number of elements
        """
        offset = c_dim_t(0)
        safe_call(backend.get().af_get_offset(ct.pointer(offset), self.arr))
        return offset.value

    def strides(self):
        """
        Return the distance in bytes between consecutive elements for each dimension.

        Returns
        ------
        strides : tuple
                  The strides for each dimension
        """
        s0 = c_dim_t(0)
        s1 = c_dim_t(0)
        s2 = c_dim_t(0)
        s3 = c_dim_t(0)
        safe_call(backend.get().af_get_strides(ct.pointer(s0), ct.pointer(s1),
                                   ct.pointer(s2), ct.pointer(s3), self.arr))
        strides = (s0.value,s1.value,s2.value,s3.value)
        return strides[:self.numdims()]

    def elements(self):
        """
        Return the number of elements in the array.
        """
        num = c_dim_t(0)
        safe_call(backend.get().af_get_elements(ct.pointer(num), self.arr))
        return num.value

    def dtype(self):
        """
        Return the data type as a arrayfire.Dtype enum value.
        """
        dty = ct.c_int(Dtype.f32.value)
        safe_call(backend.get().af_get_type(ct.pointer(dty), self.arr))
        return to_dtype[to_typecode[dty.value]]

    def type(self):
        """
        Return the data type as an int.
        """
        return self.dtype().value

    @property
    def T(self):
        """
        Return the transpose of the array
        """
        return transpose(self, False)

    @property
    def H(self):
        """
        Return the hermitian transpose of the array
        """
        return transpose(self, False)

    def dims(self):
        """
        Return the shape of the array as a tuple.
        """
        d0 = c_dim_t(0)
        d1 = c_dim_t(0)
        d2 = c_dim_t(0)
        d3 = c_dim_t(0)
        safe_call(backend.get().af_get_dims(ct.pointer(d0), ct.pointer(d1),
                                   ct.pointer(d2), ct.pointer(d3), self.arr))
        dims = (d0.value,d1.value,d2.value,d3.value)
        return dims[:self.numdims()]

    @property
    def shape(self):
        """
        The shape of the array
        """
        return self.dims()

    def numdims(self):
        """
        Return the number of dimensions of the array.
        """
        nd = ct.c_uint(0)
        safe_call(backend.get().af_get_numdims(ct.pointer(nd), self.arr))
        return nd.value

    def is_empty(self):
        """
        Check if the array is empty i.e. it has no elements.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_empty(ct.pointer(res), self.arr))
        return res.value

    def is_scalar(self):
        """
        Check if the array is scalar i.e. it has only one element.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_scalar(ct.pointer(res), self.arr))
        return res.value

    def is_row(self):
        """
        Check if the array is a row i.e. it has a shape of (1, cols).
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_row(ct.pointer(res), self.arr))
        return res.value

    def is_column(self):
        """
        Check if the array is a column i.e. it has a shape of (rows, 1).
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_column(ct.pointer(res), self.arr))
        return res.value

    def is_vector(self):
        """
        Check if the array is a vector i.e. it has a shape of one of the following:
        - (rows, 1)
        - (1, cols)
        - (1, 1, vols)
        - (1, 1, 1, batch)
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_vector(ct.pointer(res), self.arr))
        return res.value

    def is_complex(self):
        """
        Check if the array is of complex type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_complex(ct.pointer(res), self.arr))
        return res.value

    def is_real(self):
        """
        Check if the array is not of complex type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_real(ct.pointer(res), self.arr))
        return res.value

    def is_double(self):
        """
        Check if the array is of double precision floating point type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_double(ct.pointer(res), self.arr))
        return res.value

    def is_single(self):
        """
        Check if the array is of single precision floating point type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_single(ct.pointer(res), self.arr))
        return res.value

    def is_real_floating(self):
        """
        Check if the array is real and of floating point type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_realfloating(ct.pointer(res), self.arr))
        return res.value

    def is_floating(self):
        """
        Check if the array is of floating point type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_floating(ct.pointer(res), self.arr))
        return res.value

    def is_integer(self):
        """
        Check if the array is of integer type.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_integer(ct.pointer(res), self.arr))
        return res.value

    def is_bool(self):
        """
        Check if the array is of type b8.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_bool(ct.pointer(res), self.arr))
        return res.value

    def is_linear(self):
        """
        Check if all elements of the array are contiguous.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_linear(ct.pointer(res), self.arr))
        return res.value

    def is_owner(self):
        """
        Check if the array owns the raw pointer or is a derived array.
        """
        res = ct.c_bool(False)
        safe_call(backend.get().af_is_owner(ct.pointer(res), self.arr))
        return res.value

    def __add__(self, other):
        """
        Return self + other.
        """
        return _binary_func(self, other, backend.get().af_add)

    def __iadd__(self, other):
        """
        Perform self += other.
        """
        self = _binary_func(self, other, backend.get().af_add)
        return self

    def __radd__(self, other):
        """
        Return other + self.
        """
        return _binary_funcr(other, self, backend.get().af_add)

    def __sub__(self, other):
        """
        Return self - other.
        """
        return _binary_func(self, other, backend.get().af_sub)

    def __isub__(self, other):
        """
        Perform self -= other.
        """
        self = _binary_func(self, other, backend.get().af_sub)
        return self

    def __rsub__(self, other):
        """
        Return other - self.
        """
        return _binary_funcr(other, self, backend.get().af_sub)

    def __mul__(self, other):
        """
        Return self * other.
        """
        return _binary_func(self, other, backend.get().af_mul)

    def __imul__(self, other):
        """
        Perform self *= other.
        """
        self = _binary_func(self, other, backend.get().af_mul)
        return self

    def __rmul__(self, other):
        """
        Return other * self.
        """
        return _binary_funcr(other, self, backend.get().af_mul)

    def __truediv__(self, other):
        """
        Return self / other.
        """
        return _binary_func(self, other, backend.get().af_div)

    def __itruediv__(self, other):
        """
        Perform self /= other.
        """
        self =  _binary_func(self, other, backend.get().af_div)
        return self

    def __rtruediv__(self, other):
        """
        Return other / self.
        """
        return _binary_funcr(other, self, backend.get().af_div)

    def __div__(self, other):
        """
        Return self / other.
        """
        return _binary_func(self, other, backend.get().af_div)

    def __idiv__(self, other):
        """
        Perform other / self.
        """
        self =  _binary_func(self, other, backend.get().af_div)
        return self

    def __rdiv__(self, other):
        """
        Return other / self.
        """
        return _binary_funcr(other, self, backend.get().af_div)

    def __mod__(self, other):
        """
        Return self % other.
        """
        return _binary_func(self, other, backend.get().af_mod)

    def __imod__(self, other):
        """
        Perform self %= other.
        """
        self =  _binary_func(self, other, backend.get().af_mod)
        return self

    def __rmod__(self, other):
        """
        Return other % self.
        """
        return _binary_funcr(other, self, backend.get().af_mod)

    def __pow__(self, other):
        """
        Return self ** other.
        """
        return _binary_func(self, other, backend.get().af_pow)

    def __ipow__(self, other):
        """
        Perform self **= other.
        """
        self =  _binary_func(self, other, backend.get().af_pow)
        return self

    def __rpow__(self, other):
        """
        Return other ** self.
        """
        return _binary_funcr(other, self, backend.get().af_pow)

    def __lt__(self, other):
        """
        Return self < other.
        """
        return _binary_func(self, other, backend.get().af_lt)

    def __gt__(self, other):
        """
        Return self > other.
        """
        return _binary_func(self, other, backend.get().af_gt)

    def __le__(self, other):
        """
        Return self <= other.
        """
        return _binary_func(self, other, backend.get().af_le)

    def __ge__(self, other):
        """
        Return self >= other.
        """
        return _binary_func(self, other, backend.get().af_ge)

    def __eq__(self, other):
        """
        Return self == other.
        """
        return _binary_func(self, other, backend.get().af_eq)

    def __ne__(self, other):
        """
        Return self != other.
        """
        return _binary_func(self, other, backend.get().af_neq)

    def __and__(self, other):
        """
        Return self & other.
        """
        return _binary_func(self, other, backend.get().af_bitand)

    def __iand__(self, other):
        """
        Perform self &= other.
        """
        self = _binary_func(self, other, backend.get().af_bitand)
        return self

    def __or__(self, other):
        """
        Return self | other.
        """
        return _binary_func(self, other, backend.get().af_bitor)

    def __ior__(self, other):
        """
        Perform self |= other.
        """
        self = _binary_func(self, other, backend.get().af_bitor)
        return self

    def __xor__(self, other):
        """
        Return self ^ other.
        """
        return _binary_func(self, other, backend.get().af_bitxor)

    def __ixor__(self, other):
        """
        Perform self ^= other.
        """
        self = _binary_func(self, other, backend.get().af_bitxor)
        return self

    def __lshift__(self, other):
        """
        Return self << other.
        """
        return _binary_func(self, other, backend.get().af_bitshiftl)

    def __ilshift__(self, other):
        """
        Perform self <<= other.
        """
        self = _binary_func(self, other, backend.get().af_bitshiftl)
        return self

    def __rshift__(self, other):
        """
        Return self >> other.
        """
        return _binary_func(self, other, backend.get().af_bitshiftr)

    def __irshift__(self, other):
        """
        Perform self >>= other.
        """
        self = _binary_func(self, other, backend.get().af_bitshiftr)
        return self

    def __neg__(self):
        """
        Return -self
        """
        return 0 - self

    def __pos__(self):
        """
        Return +self
        """
        return self

    def __invert__(self):
        """
        Return ~self
        """
        return self == 0

    def __nonzero__(self):
        return self != 0

    # TODO:
    # def __abs__(self):
    #     return self

    def __getitem__(self, key):
        """
        Return self[key]

        Note
        ----
        Ellipsis not supported as key
        """
        try:
            out = Array()
            n_dims = self.numdims()

            if (isinstance(key, Array) and key.type() == Dtype.b8.value):
                n_dims = 1
                if (count(key) == 0):
                    return out

            inds = _get_indices(key)

            safe_call(backend.get().af_index_gen(ct.pointer(out.arr),
                                    self.arr, c_dim_t(n_dims), inds.pointer))
            return out
        except RuntimeError as e:
            raise IndexError(str(e))


    def __setitem__(self, key, val):
        """
        Perform self[key] = val

        Note
        ----
        Ellipsis not supported as key
        """
        try:
            n_dims = self.numdims()

            is_boolean_idx = isinstance(key, Array) and key.type() == Dtype.b8.value

            if (is_boolean_idx):
                n_dims = 1
                num = count(key)
                if (num == 0):
                    return

            if (_is_number(val)):
                tdims = _get_assign_dims(key, self.dims())
                if (is_boolean_idx):
                    n_dims = 1
                    other_arr = constant_array(val, int(num), dtype=self.type())
                else:
                    other_arr = constant_array(val, tdims[0] , tdims[1], tdims[2], tdims[3], self.type())
                del_other = True
            else:
                other_arr = val.arr
                del_other = False

            out_arr = ct.c_void_p(0)
            inds  = _get_indices(key)

            safe_call(backend.get().af_assign_gen(ct.pointer(out_arr),
                                                  self.arr, c_dim_t(n_dims), inds.pointer,
                                                  other_arr))
            safe_call(backend.get().af_release_array(self.arr))
            if del_other:
                safe_call(backend.get().af_release_array(other_arr))
            self.arr = out_arr

        except RuntimeError as e:
            raise IndexError(str(e))

    def to_ctype(self, row_major=False, return_shape=False):
        """
        Return the data as a ctype C array after copying to host memory

        Parameters
        ---------

        row_major: optional: bool. default: False.
            Specifies if a transpose needs to occur before copying to host memory.

        return_shape: optional: bool. default: False.
            Specifies if the shape of the array needs to be returned.

        Returns
        -------

        If return_shape is False:
            res: The ctypes array of the appropriate type and length.
        else :
            (res, dims): tuple of the ctypes array and the shape of the array

        """
        if (self.arr.value == 0):
            raise RuntimeError("Can not call to_ctype on empty array")

        tmp = transpose(self) if row_major else self
        ctype_type = to_c_type[self.type()] * self.elements()
        res = ctype_type()
        safe_call(backend.get().af_get_data_ptr(ct.pointer(res), self.arr))
        if (return_shape):
            return res, self.dims()
        else:
            return res

    def to_array(self, row_major=False, return_shape=False):
        """
        Return the data as array.array

        Parameters
        ---------

        row_major: optional: bool. default: False.
            Specifies if a transpose needs to occur before copying to host memory.

        return_shape: optional: bool. default: False.
            Specifies if the shape of the array needs to be returned.

        Returns
        -------

        If return_shape is False:
            res: array.array of the appropriate type and length.
        else :
            (res, dims): array.array and the shape of the array

        """
        if (self.arr.value == 0):
            raise RuntimeError("Can not call to_array on empty array")

        res = self.to_ctype(row_major, return_shape)

        host = __import__("array")
        h_type = to_typecode[self.type()]

        if (return_shape):
            return host.array(h_type, res[0]), res[1]
        else:
            return host.array(h_type, res)

    def to_list(self, row_major=False):
        """
        Return the data as list

        Parameters
        ---------

        row_major: optional: bool. default: False.
            Specifies if a transpose needs to occur before copying to host memory.

        return_shape: optional: bool. default: False.
            Specifies if the shape of the array needs to be returned.

        Returns
        -------

        If return_shape is False:
            res: list of the appropriate type and length.
        else :
            (res, dims): list and the shape of the array

        """
        ct_array, shape = self.to_ctype(row_major, True)
        return _ctype_to_lists(ct_array, len(shape) - 1, shape)

    def __repr__(self):
        """
        Displays the meta data and contents of  the arrayfire array.

        Note
        ----
        You can also use af.display(a, pres) to display the contents of the array with better precision.
        """

        arr_str = ct.c_char_p(0)
        safe_call(backend.get().af_array_to_string(ct.pointer(arr_str), "", self.arr, 4, True))

        return 'arrayfire.Array()\nType: %s' % \
            (to_typename[self.type()]) + to_str(arr_str)

    def __array__(self):
        """
        Constructs a numpy.array from arrayfire.Array
        """
        import numpy as np
        res = np.empty(self.dims(), dtype=np.dtype(to_typecode[self.type()]), order='F')
        safe_call(backend.get().af_get_data_ptr(ct.c_void_p(res.ctypes.data), self.arr))
        return res

def display(a, precision=4):
    """
    Displays the contents of an array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array
    precision: int. optional.
        Specifies the number of precision bits to display
    """
    expr = inspect.stack()[1][-2]
    name = ""

    try:
        if (expr is not None):
            st = expr[0].find('(') + 1
            en = expr[0].rfind(')')
            name = expr[0][st:en]
    except:
        pass

    safe_call(backend.get().af_print_array_gen(name.encode('utf-8'),
                                               a.arr, ct.c_int(precision)))

def save_array(key, a, filename, append=False):
    """
    Save an array to disk.

    Parameters
    ----------
    key     : str
            A name / key associated with the array

    a       : af.Array
            The array to be stored to disk

    filename : str
             Location of the data file.

    append   : Boolean. optional. default: False.
             If the file already exists, specifies if the data should be appended or overwritten.

    Returns
    ---------
    index   : int
            The index of the array stored in the file.
    """
    index = ct.c_int(-1)
    safe_call(backend.get().af_save_array(ct.pointer(index),
                                          key.encode('utf-8'),
                                          a.arr,
                                          filename.encode('utf-8'),
                                          append))
    return index.value

def read_array(filename, index=None, key=None):
    """
    Read an array from disk.

    Parameters
    ----------

    filename : str
             Location of the data file.

    index   : int. Optional. Default: None.
            - The index of the array stored in the file.
            - If None, key is used.

    key     : str. Optional. Default: None.
            - A name / key associated with the array
            - If None, index is used.

    Returns
    ---------
    """
    assert((index is not None) or (key is not None))
    out = Array()
    if (index is not None):
        safe_call(backend.get().af_read_array_index(ct.pointer(out.arr),
                                                    filename.encode('utf-8'),
                                                    index))
    elif (key is not None):
        safe_call(backend.get().af_read_array_key(ct.pointer(out.arr),
                                                  filename.encode('utf-8'),
                                                  key.encode('utf-8')))

    return out

from .algorithm import (sum, count)
from .arith import cast
