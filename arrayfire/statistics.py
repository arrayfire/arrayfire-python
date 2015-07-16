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

def mean(a, weights=None, dim=None):
    if dim is not None:
        out = array()

        if weights is None:
            safe_call(clib.af_mean(ct.pointer(out.arr), a.arr, ct.c_int(dim)))
        else:
            safe_call(clib.af_mean_weighted(ct.pointer(out.arr), a.arr, weights.arr, ct.c_int(dim)))

        return out
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)

        if weights is None:
            safe_call(clib.af_mean_all(ct.pointer(real), ct.pointer(imag), a.arr))
        else:
            safe_call(clib.af_mean_all_weighted(ct.pointer(real), ct.pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def var(a, isbiased=False, weights=None, dim=None):
    if dim is not None:
        out = array()

        if weights is None:
            safe_call(clib.af_var(ct.pointer(out.arr), a.arr, isbiased, ct.c_int(dim)))
        else:
            safe_call(clib.af_var_weighted(ct.pointer(out.arr), a.arr, weights.arr, ct.c_int(dim)))

        return out
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)

        if weights is None:
            safe_call(clib.af_var_all(ct.pointer(real), ct.pointer(imag), a.arr, isbiased))
        else:
            safe_call(clib.af_var_all_weighted(ct.pointer(real), ct.pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def stdev(a, dim=None):
    if dim is not None:
        out = array()
        safe_call(clib.af_stdev(ct.pointer(out.arr), a.arr, ct.c_int(dim)))
        return out
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)
        safe_call(clib.af_stdev_all(ct.pointer(real), ct.pointer(imag), a.arr))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def cov(a, isbiased=False, dim=None):
    if dim is not None:
        out = array()
        safe_call(clib.af_cov(ct.pointer(out.arr), a.arr, isbiased, ct.c_int(dim)))
        return out
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)
        safe_call(clib.af_cov_all(ct.pointer(real), ct.pointer(imag), a.arr, isbiased))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def median(a, dim=None):
    if dim is not None:
        out = array()
        safe_call(clib.af_median(ct.pointer(out.arr), a.arr, ct.c_int(dim)))
        return out
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)
        safe_call(clib.af_median_all(ct.pointer(real), ct.pointer(imag), a.arr))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def corrcoef(x, y):
    real = ct.c_double(0)
    imag = ct.c_double(0)
    safe_call(clib.af_corrcoef(ct.pointer(real), ct.pointer(imag), x.arr, y.arr))
    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j
