#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from .library import *
from .array import *

def approx1(signal, pos0, method=INTERP.LINEAR, off_grid=0.0):
    output = Array()
    safe_call(backend.get().af_approx1(ct.pointer(output.arr), signal.arr, pos0.arr,
                                       method.value, ct.c_double(off_grid)))
    return output

def approx2(signal, pos0, pos1, method=INTERP.LINEAR, off_grid=0.0):
    output = Array()
    safe_call(backend.get().af_approx2(ct.pointer(output.arr), signal.arr,
                                       pos0.arr, pos1.arr, method.value, ct.c_double(off_grid)))
    return output

def fft(signal, dim0 = None , scale = None):

    if dim0 is None:
        dim0 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft(ct.pointer(output.arr), signal.arr, ct.c_double(scale), ct.c_longlong(dim0)))
    return output

def fft2(signal, dim0 = None, dim1 = None , scale = None):

    if dim0 is None:
        dim0 = 0

    if dim1 is None:
        dim1 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft2(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                    ct.c_longlong(dim0), ct.c_longlong(dim1)))
    return output

def fft3(signal, dim0 = None, dim1 = None , dim2 = None, scale = None):

    if dim0 is None:
        dim0 = 0

    if dim1 is None:
        dim1 = 0

    if dim2 is None:
        dim2 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft3(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                    ct.c_longlong(dim0), ct.c_longlong(dim1), ct.c_longlong(dim2)))
    return output

def ifft(signal, dim0 = None , scale = None):

    if dim0 is None:
        dim0 = signal.dims()[0]

    if scale is None:
        scale = 1.0/float(dim0)

    output = Array()
    safe_call(backend.get().af_ifft(ct.pointer(output.arr), signal.arr, ct.c_double(scale), ct.c_longlong(dim0)))
    return output

def ifft2(signal, dim0 = None, dim1 = None , scale = None):

    dims = signal.dims()

    if (len(dims) < 2):
        return ifft(signal)

    if dim0 is None:
        dim0 = dims[0]

    if dim1 is None:
        dim1 = dims[1]

    if scale is None:
        scale = 1.0/float(dim0 * dim1)

    output = Array()
    safe_call(backend.get().af_ifft2(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                     ct.c_longlong(dim0), ct.c_longlong(dim1)))
    return output

def ifft3(signal, dim0 = None, dim1 = None , dim2 = None, scale = None):

    dims = signal.dims()

    if (len(dims) < 3):
        return ifft2(signal)

    if dim0 is None:
        dim0 = dims[0]

    if dim1 is None:
        dim1 = dims[1]

    if dim2 is None:
        dim2 = dims[2]

    if scale is None:
        scale = 1.0 / float(dim0 * dim1 * dim2)

    output = Array()
    safe_call(backend.get().af_ifft3(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                     ct.c_longlong(dim0), ct.c_longlong(dim1), ct.c_longlong(dim2)))
    return output

def dft(signal, scale = None, odims=(None, None, None, None)):

    odims4 = dim4_to_tuple(odims, default=None)

    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return fft(signal, scale, dims[0])
    elif (ndims == 2):
        return fft2(signal, scale, dims[0], dims[1])
    else:
        return fft3(signal, scale, dims[0], dims[1], dims[2])

def idft(signal, scale = None, odims=(None, None, None, None)):

    odims4 = dim4_to_tuple(odims, default=None)

    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return ifft(signal, scale, dims[0])
    elif (ndims == 2):
        return ifft2(signal, scale, dims[0], dims[1])
    else:
        return ifft3(signal, scale, dims[0], dims[1], dims[2])

def convolve1(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    output = Array()
    safe_call(backend.get().af_convolve1(ct.pointer(output.arr), signal.arr, kernel.arr,
                                         conv_mode.value, conv_domain.value))
    return output

def convolve2(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    output = Array()
    safe_call(backend.get().af_convolve2(ct.pointer(output.arr), signal.arr, kernel.arr,
                                         conv_mode.value, conv_domain.value))
    return output

def convolve3(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    output = Array()
    safe_call(backend.get().af_convolve3(ct.pointer(output.arr), signal.arr, kernel.arr,
                                         conv_mode.value, conv_domain.value))
    return output

def convolve(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return convolve1(signal, kernel, conv_mode, conv_domain)
    elif (ndims == 2):
        return convolve2(signal, kernel, conv_mode, conv_domain)
    else:
        return convolve3(signal, kernel, conv_mode, conv_domain)

def fft_convolve1(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    output = Array()
    safe_call(backend.get().af_fft_convolve1(ct.pointer(output.arr), signal.arr, kernel.arr,
                                             conv_mode.value))
    return output

def fft_convolve2(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    output = Array()
    safe_call(backend.get().af_fft_convolve2(ct.pointer(output.arr), signal.arr, kernel.arr,
                                             conv_mode.value))
    return output

def fft_convolve3(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    output = Array()
    safe_call(backend.get().af_fft_convolve3(ct.pointer(output.arr), signal.arr, kernel.arr,
                                             conv_mode.value))
    return output

def fft_convolve(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return fft_convolve1(signal, kernel, conv_mode)
    elif (ndims == 2):
        return fft_convolve2(signal, kernel, conv_mode)
    else:
        return fft_convolve3(signal, kernel, conv_mode)

def fir(B, X):
    Y = Array()
    safe_call(backend.get().af_fir(ct.pointer(Y.arr), B.arr, X.arr))
    return Y

def iir(B, A, X):
    Y = Array()
    safe_call(backend.get().af_iir(ct.pointer(Y.arr), B.arr, A.arr, X.arr))
    return Y
