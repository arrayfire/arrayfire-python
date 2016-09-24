#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Statistical algorithms (mean, var, stdev, etc).
"""

from .library import *
from .array import *

def mean(a, weights=None, dim=None):
    if dim is not None:
        out = Array()

        if weights is None:
            safe_call(backend.get().af_mean(c_pointer(out.arr), a.arr, c_int_t(dim)))
        else:
            safe_call(backend.get().af_mean_weighted(c_pointer(out.arr), a.arr, weights.arr, c_int_t(dim)))

        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)

        if weights is None:
            safe_call(backend.get().af_mean_all(c_pointer(real), c_pointer(imag), a.arr))
        else:
            safe_call(backend.get().af_mean_all_weighted(c_pointer(real), c_pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def var(a, isbiased=False, weights=None, dim=None):
    if dim is not None:
        out = Array()

        if weights is None:
            safe_call(backend.get().af_var(c_pointer(out.arr), a.arr, isbiased, c_int_t(dim)))
        else:
            safe_call(backend.get().af_var_weighted(c_pointer(out.arr), a.arr, weights.arr, c_int_t(dim)))

        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)

        if weights is None:
            safe_call(backend.get().af_var_all(c_pointer(real), c_pointer(imag), a.arr, isbiased))
        else:
            safe_call(backend.get().af_var_all_weighted(c_pointer(real), c_pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def stdev(a, dim=None):
    if dim is not None:
        out = Array()
        safe_call(backend.get().af_stdev(c_pointer(out.arr), a.arr, c_int_t(dim)))
        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)
        safe_call(backend.get().af_stdev_all(c_pointer(real), c_pointer(imag), a.arr))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def cov(a, isbiased=False, dim=None):
    if dim is not None:
        out = Array()
        safe_call(backend.get().af_cov(c_pointer(out.arr), a.arr, isbiased, c_int_t(dim)))
        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)
        safe_call(backend.get().af_cov_all(c_pointer(real), c_pointer(imag), a.arr, isbiased))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def median(a, dim=None):
    if dim is not None:
        out = Array()
        safe_call(backend.get().af_median(c_pointer(out.arr), a.arr, c_int_t(dim)))
        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)
        safe_call(backend.get().af_median_all(c_pointer(real), c_pointer(imag), a.arr))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def corrcoef(x, y):
    real = c_double_t(0)
    imag = c_double_t(0)
    safe_call(backend.get().af_corrcoef(c_pointer(real), c_pointer(imag), x.arr, y.arr))
    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j
