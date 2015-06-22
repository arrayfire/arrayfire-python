import array as host
import inspect
from .library import *
from .util import *
from .data import *

def create_array(buf, numdims, idims, dtype):
    out_arr = c_longlong(0)
    c_dims = dim4(idims[0], idims[1], idims[2], idims[3])
    safe_call(clib.af_create_array(pointer(out_arr), c_longlong(buf), numdims, pointer(c_dims), dtype))
    return out_arr

def constant_array(val, d0, d1=None, d2=None, d3=None, dtype=f32):
    if not isinstance(dtype, c_int):
        raise TypeError("Invalid dtype")

    out = c_longlong(0)
    dims = dim4(d0, d1, d2, d3)

    if isinstance(val, complex):
        c_real = c_double(val.real)
        c_imag = c_double(val.imag)

        if (dtype != c32 and dtype != c64):
            dtype = c32

        safe_call(clib.af_constant_complex(pointer(out), c_real, c_imag, 4, pointer(dims), dtype))
    elif dtype == s64:
        c_val = c_longlong(val.real)
        safe_call(clib.af_constant_long(pointer(out), c_val, 4, pointer(dims)))
    elif dtype == u64:
        c_val = c_ulonglong(val.real)
        safe_call(clib.af_constant_ulong(pointer(out), c_val, 4, pointer(dims)))
    else:
        c_val = c_double(val)
        safe_call(clib.af_constant(pointer(out), c_val, 4, pointer(dims), dtype))

    return out


def binary_func(lhs, rhs, c_func):
    out = array()
    other = rhs

    if (is_valid_scalar(rhs)):
        ldims = dim4_tuple(lhs.dims())
        lty = lhs.type()
        other = array()
        other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], lty)
    elif not isinstance(rhs, array):
        TypeError("Invalid parameter to binary function")

    safe_call(c_func(pointer(out.arr), lhs.arr, other.arr, False))

    return out

def binary_funcr(lhs, rhs, c_func):
    out = array()
    other = lhs

    if (is_valid_scalar(lhs)):
        rdims = dim4_tuple(rhs.dims())
        rty = rhs.type()
        other = array()
        other.arr = constant_array(lhs, rdims[0], rdims[1], rdims[2], rdims[3], rty)
    elif not isinstance(lhs, array):
        TypeError("Invalid parameter to binary function")

    c_func(pointer(out.arr), other.arr, rhs.arr, False)

    return out

class array(object):

    def __init__(self, src=None, dims=(0,)):

        self.arr = c_longlong(0)

        buf=None
        buf_len=0
        type_char='f'
        dtype = f32

        if src is not None:

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

            if type_char == 'f':
                dtype = f32
            elif type_char == 'd':
                dtype = f64
            elif type_char == 'b':
                dtype = b8
            elif type_char == 'B':
                dtype = u8
            elif type_char == 'i':
                dtype = s32
            elif type_char == 'I':
                dtype = u32
            elif type_char == 'l':
                dtype = s64
            elif type_char == 'L':
                dtype = u64

            self.arr = create_array(buf, numdims, idims, dtype)

    def __del__(self):
        if (self.arr.value != 0):
            clib.af_release_array(self.arr)

    def numdims(self):
        nd = c_uint(0)
        safe_call(clib.af_get_numdims(pointer(nd), self.arr))
        return nd.value

    def dims(self):
        d0 = c_longlong(0)
        d1 = c_longlong(0)
        d2 = c_longlong(0)
        d3 = c_longlong(0)
        safe_call(clib.af_get_dims(pointer(d0), pointer(d1), pointer(d2), pointer(d3), self.arr))
        dims = (d0.value,d1.value,d2.value,d3.value)
        return dims[:self.numdims()]

    def type(self):
        dty = c_int(f32.value)
        safe_call(clib.af_get_type(pointer(dty), self.arr))
        return dty

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

def print_array(a):
    expr = inspect.stack()[1][-2]
    if (expr is not None):
        print('%s' % expr[0].split('print_array(')[1][:-2])
    safe_call(clib.af_print_array(a.arr))
