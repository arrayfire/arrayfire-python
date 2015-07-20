#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import inspect
from .library import *
from .util import *

def create_array(buf, numdims, idims, dtype):
    out_arr = ct.c_longlong(0)
    ct.c_dims = dim4(idims[0], idims[1], idims[2], idims[3])
    safe_call(clib.af_create_array(ct.pointer(out_arr), ct.c_longlong(buf),\
                                   numdims, ct.pointer(ct.c_dims), dtype))
    return out_arr

def constant_array(val, d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, ct.c_int):
        if isinstance(dtype, int):
            dtype = ct.c_int(dtype)
        else:
            raise TypeError("Invalid dtype")

    out = ct.c_longlong(0)
    dims = dim4(d0, d1, d2, d3)

    if isinstance(val, complex):
        c_real = ct.c_double(val.real)
        c_imag = ct.c_double(val.imag)

        if (dtype != c32 and dtype != c64):
            dtype = c32

        safe_call(clib.af_constant_complex(ct.pointer(out), c_real, c_imag,\
                                           4, ct.pointer(dims), dtype))
    elif dtype == s64:
        c_val = ct.c_longlong(val.real)
        safe_call(clib.af_constant_long(ct.pointer(out), c_val, 4, ct.pointer(dims)))
    elif dtype == u64:
        c_val = ct.c_ulonglong(val.real)
        safe_call(clib.af_constant_ulong(ct.pointer(out), c_val, 4, ct.pointer(dims)))
    else:
        c_val = ct.c_double(val)
        safe_call(clib.af_constant(ct.pointer(out), c_val, 4, ct.pointer(dims), dtype))

    return out


def binary_func(lhs, rhs, c_func):
    out = array()
    other = rhs

    if (is_number(rhs)):
        ldims = dim4_tuple(lhs.dims())
        lty = lhs.type()
        other = array()
        other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], lty)
    elif not isinstance(rhs, array):
        raise TypeError("Invalid parameter to binary function")

    safe_call(c_func(ct.pointer(out.arr), lhs.arr, other.arr, False))

    return out

def binary_funcr(lhs, rhs, c_func):
    out = array()
    other = lhs

    if (is_number(lhs)):
        rdims = dim4_tuple(rhs.dims())
        rty = rhs.type()
        other = array()
        other.arr = constant_array(lhs, rdims[0], rdims[1], rdims[2], rdims[3], rty)
    elif not isinstance(lhs, array):
        raise TypeError("Invalid parameter to binary function")

    c_func(ct.pointer(out.arr), other.arr, rhs.arr, False)

    return out

def transpose(a, conj=False):
    out = array()
    safe_call(clib.af_transpose(ct.pointer(out.arr), a.arr, conj))
    return out

def transpose_inplace(a, conj=False):
    safe_call(clib.af_transpose_inplace(a.arr, conj))

class seq(ct.Structure):
    _fields_ = [("begin", ct.c_double),
                ("end"  , ct.c_double),
                ("step" , ct.c_double)]

    def __init__ (self, S):
        num = __import__("numbers")

        self.begin = ct.c_double( 0)
        self.end   = ct.c_double(-1)
        self.step  = ct.c_double( 1)

        if is_number(S):
            self.begin = ct.c_double(S)
            self.end   = ct.c_double(S)
        elif isinstance(S, slice):
            if (S.start is not None):
                self.begin = ct.c_double(S.start)
            if (S.stop is not None):
                self.end   = ct.c_double(S.stop - 1)
            if (S.step is not None):
                self.step  = ct.c_double(S.step)
        else:
            raise IndexError("Invalid type while indexing arrayfire.array")

class uidx(ct.Union):
    _fields_ = [("arr", ct.c_longlong),
                ("seq", seq)]

class index(ct.Structure):
    _fields_ = [("idx", uidx),
                ("isSeq", ct.c_bool),
                ("isBatch", ct.c_bool)]

    def __init__ (self, idx):

        self.idx     = uidx()
        self.isBatch = False
        self.isSeq   = True

        if isinstance(idx, array):
            self.idx.arr = idx.arr
            self.isSeq   = False
        else:
            self.idx.seq = seq(idx)

def get_indices(key, n_dims):
    index_vec = index * n_dims
    inds = index_vec()

    for n in range(n_dims):
        inds[n] = index(slice(None))

    if isinstance(key, tuple):
        n_idx = len(key)
        for n in range(n_idx):
            inds[n] = index(key[n])
    else:
        inds[0] = index(key)

    return inds

def slice_to_length(key, dim):
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

def ctype_to_lists(ctype_arr, dim, shape, offset=0):
    if (dim == 0):
        return list(ctype_arr[offset : offset + shape[0]])
    else:
        dim_len = shape[dim]
        res = [[]] * dim_len
        for n in range(dim_len):
            res[n] = ctype_to_lists(ctype_arr, dim - 1, shape, offset)
            offset += shape[0]
        return res

def get_assign_dims(key, idims):
    dims = [1]*4

    for n in range(len(idims)):
        dims[n] = idims[n]

    if is_number(key):
        dims[0] = 1
        return dims
    elif isinstance(key, slice):
        dims[0] = slice_to_length(key, idims[0])
        return dims
    elif isinstance(key, array):
        dims[0] = key.elements()
        return dims
    elif isinstance(key, tuple):
        n_inds = len(key)

        if (n_inds > len(idims)):
            raise IndexError("Number of indices greater than array dimensions")

        for n in range(n_inds):
            if (is_number(key[n])):
                dims[n] = 1
            elif (isinstance(key[n], array)):
                dims[n] = key[n].elements()
            elif (isinstance(key[n], slice)):
                dims[n] = slice_to_length(key[n], idims[n])
            else:
                raise IndexError("Invalid type while assigning to arrayfire.array")

        return dims
    else:
        raise IndexError("Invalid type while assigning to arrayfire.array")

class array(object):

    def __init__(self, src=None, dims=(0,)):

        self.arr = ct.c_longlong(0)

        buf=None
        buf_len=0
        type_char='f'
        dtype = f32

        if src is not None:

            if (isinstance(src, array)):
                safe_call(clib.af_retain_array(ct.pointer(self.arr), src.arr))
                return

            host = __import__("array")

            if isinstance(src, host.array):
                buf,buf_len = src.buffer_info()
                type_char = src.typecode
            elif isinstance(src, list):
                tmp = host.array('f', src)
                buf,buf_len = tmp.buffer_info()
                type_char = tmp.typecode
            else:
                raise TypeError("src is an object of unsupported class")

            elements = 1
            numdims = len(dims)
            idims = [1]*4

            for i in range(numdims):
                elements *= dims[i]
                idims[i] = dims[i]

            if (elements == 0):
                idims = [buf_len, 1, 1, 1]
                numdims = 1

            dtype = to_dtype[type_char]

            self.arr = create_array(buf, numdims, idims, dtype)

    def copy(self):
        out = array()
        safe_call(clib.af_retain_array(ct.pointer(out.arr), self.arr))
        return out

    def __del__(self):
        if (self.arr.value != 0):
            clib.af_release_array(self.arr)

    def elements(self):
        num = ct.c_ulonglong(0)
        safe_call(clib.af_get_elements(ct.pointer(num), self.arr))
        return num.value

    def type(self):
        dty = ct.c_int(f32.value)
        safe_call(clib.af_get_type(ct.pointer(dty), self.arr))
        return dty.value

    def dims(self):
        d0 = ct.c_longlong(0)
        d1 = ct.c_longlong(0)
        d2 = ct.c_longlong(0)
        d3 = ct.c_longlong(0)
        safe_call(clib.af_get_dims(ct.pointer(d0), ct.pointer(d1),\
                                   ct.pointer(d2), ct.pointer(d3), self.arr))
        dims = (d0.value,d1.value,d2.value,d3.value)
        return dims[:self.numdims()]

    def numdims(self):
        nd = ct.c_uint(0)
        safe_call(clib.af_get_numdims(ct.pointer(nd), self.arr))
        return nd.value

    def is_empty(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_empty(ct.pointer(res), self.arr))
        return res.value

    def is_scalar(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_scalar(ct.pointer(res), self.arr))
        return res.value

    def is_row(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_row(ct.pointer(res), self.arr))
        return res.value

    def is_column(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_column(ct.pointer(res), self.arr))
        return res.value

    def is_vector(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_vector(ct.pointer(res), self.arr))
        return res.value

    def is_complex(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_complex(ct.pointer(res), self.arr))
        return res.value

    def is_real(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_real(ct.pointer(res), self.arr))
        return res.value

    def is_double(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_double(ct.pointer(res), self.arr))
        return res.value

    def is_single(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_single(ct.pointer(res), self.arr))
        return res.value

    def is_real_floating(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_realfloating(ct.pointer(res), self.arr))
        return res.value

    def is_floating(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_floating(ct.pointer(res), self.arr))
        return res.value

    def is_integer(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_integer(ct.pointer(res), self.arr))
        return res.value

    def is_bool(self):
        res = ct.c_bool(False)
        safe_call(clib.af_is_bool(ct.pointer(res), self.arr))
        return res.value

    def __add__(self, other):
        return binary_func(self, other, clib.af_add)

    def __iadd__(self, other):
        self = binary_func(self, other, clib.af_add)
        return self

    def __radd__(self, other):
        return binary_funcr(other, self, clib.af_add)

    def __sub__(self, other):
        return binary_func(self, other, clib.af_sub)

    def __isub__(self, other):
        self = binary_func(self, other, clib.af_sub)
        return self

    def __rsub__(self, other):
        return binary_funcr(other, self, clib.af_sub)

    def __mul__(self, other):
        return binary_func(self, other, clib.af_mul)

    def __imul__(self, other):
        self = binary_func(self, other, clib.af_mul)
        return self

    def __rmul__(self, other):
        return binary_funcr(other, self, clib.af_mul)

    # Necessary for python3
    def __truediv__(self, other):
        return binary_func(self, other, clib.af_div)

    def __itruediv__(self, other):
        self =  binary_func(self, other, clib.af_div)
        return self

    def __rtruediv__(self, other):
        return binary_funcr(other, self, clib.af_div)

    # Necessary for python2
    def __div__(self, other):
        return binary_func(self, other, clib.af_div)

    def __idiv__(self, other):
        self =  binary_func(self, other, clib.af_div)
        return self

    def __rdiv__(self, other):
        return binary_funcr(other, self, clib.af_div)

    def __mod__(self, other):
        return binary_func(self, other, clib.af_mod)

    def __imod__(self, other):
        self =  binary_func(self, other, clib.af_mod)
        return self

    def __rmod__(self, other):
        return binary_funcr(other, self, clib.af_mod)

    def __pow__(self, other):
        return binary_func(self, other, clib.af_pow)

    def __ipow__(self, other):
        self =  binary_func(self, other, clib.af_pow)
        return self

    def __rpow__(self, other):
        return binary_funcr(other, self, clib.af_pow)

    def __lt__(self, other):
        return binary_func(self, other, clib.af_lt)

    def __gt__(self, other):
        return binary_func(self, other, clib.af_gt)

    def __le__(self, other):
        return binary_func(self, other, clib.af_le)

    def __ge__(self, other):
        return binary_func(self, other, clib.af_ge)

    def __eq__(self, other):
        return binary_func(self, other, clib.af_eq)

    def __ne__(self, other):
        return binary_func(self, other, clib.af_neq)

    def __and__(self, other):
        return binary_func(self, other, clib.af_bitand)

    def __iand__(self, other):
        self = binary_func(self, other, clib.af_bitand)
        return self

    def __or__(self, other):
        return binary_func(self, other, clib.af_bitor)

    def __ior__(self, other):
        self = binary_func(self, other, clib.af_bitor)
        return self

    def __xor__(self, other):
        return binary_func(self, other, clib.af_bitxor)

    def __ixor__(self, other):
        self = binary_func(self, other, clib.af_bitxor)
        return self

    def __lshift__(self, other):
        return binary_func(self, other, clib.af_bitshiftl)

    def __ilshift__(self, other):
        self = binary_func(self, other, clib.af_bitshiftl)
        return self

    def __rshift__(self, other):
        return binary_func(self, other, clib.af_bitshiftr)

    def __irshift__(self, other):
        self = binary_func(self, other, clib.af_bitshiftr)
        return self

    def __neg__(self):
        return 0 - self

    def __pos__(self):
        return self

    def __invert__(self):
        return self == 0

    def __nonzero__(self):
        return self != 0

    # TODO:
    # def __abs__(self):
    #     return self

    def __getitem__(self, key):
        try:
            out = array()
            n_dims = self.numdims()
            inds = get_indices(key, n_dims)

            safe_call(clib.af_index_gen(ct.pointer(out.arr),\
                                        self.arr, ct.c_longlong(n_dims), ct.pointer(inds)))
            return out
        except RuntimeError as e:
            raise IndexError(str(e))


    def __setitem__(self, key, val):
        try:
            n_dims = self.numdims()

            if (is_number(val)):
                tdims = get_assign_dims(key, self.dims())
                other_arr = constant_array(val, tdims[0], tdims[1], tdims[2], tdims[3])
            else:
                other_arr = val.arr

            out_arr = ct.c_longlong(0)
            inds  = get_indices(key, n_dims)

            safe_call(clib.af_assign_gen(ct.pointer(out_arr),\
                                         self.arr, ct.c_longlong(n_dims), ct.pointer(inds),\
                                         other_arr))
            safe_call(clib.af_release_array(self.arr))
            self.arr = out_arr

        except RuntimeError as e:
            raise IndexError(str(e))

    def to_ctype(self, row_major=False, return_shape=False):
        tmp = transpose(self) if row_major else self
        ctype_type = to_c_type[self.type()] * self.elements()
        res = ctype_type()
        safe_call(clib.af_get_data_ptr(ct.pointer(res), self.arr))
        if (return_shape):
            return res, self.dims()
        else:
            return res

    def to_array(self, row_major=False, return_shape=False):
        res = self.to_ctype(row_major, return_shape)

        host = __import__("array")
        h_type = to_typecode[self.type()]

        if (return_shape):
            return host.array(h_type, res[0]), res[1]
        else:
            return host.array(h_type, res)

    def to_list(self, row_major=False):
        ct_array, shape = self.to_ctype(row_major, True)
        return ctype_to_lists(ct_array, len(shape) - 1, shape)

def display(a):
    expr = inspect.stack()[1][-2]
    if (expr is not None):
        print('%s' % expr[0].split('display(')[1][:-2])
    safe_call(clib.af_print_array(a.arr))
