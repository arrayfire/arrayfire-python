import array as host
from .library import *
from .util import dim4

def create_array(buf, numdims, idims, dtype):
    out_arr = c_longlong(0)
    c_dims = dim4(idims[0], idims[1], idims[2], idims[3])
    clib.af_create_array(pointer(out_arr), c_longlong(buf), numdims, pointer(c_dims), dtype)
    return out_arr

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
