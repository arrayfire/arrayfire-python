from .library import *
from .array import *

def mean(a, weights=None, dim=None):
    if dim is not None:
        out = array()

        if weights is None:
            safe_call(clib.af_mean(pointer(out.arr), a.arr, c_int(dim)))
        else:
            safe_call(clib.af_mean_weighted(pointer(out.arr), a.arr, weights.arr, c_int(dim)))

        return out
    else:
        real = c_double(0)
        imag = c_double(0)

        if weights is None:
            safe_call(clib.af_mean_all(pointer(real), pointer(imag), a.arr))
        else:
            safe_call(clib.af_mean_all_weighted(pointer(real), pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def var(a, isbiased=False, weights=None, dim=None):
    if dim is not None:
        out = array()

        if weights is None:
            safe_call(clib.af_var(pointer(out.arr), a.arr, isbiased, c_int(dim)))
        else:
            safe_call(clib.af_var_weighted(pointer(out.arr), a.arr, weights.arr, c_int(dim)))

        return out
    else:
        real = c_double(0)
        imag = c_double(0)

        if weights is None:
            safe_call(clib.af_var_all(pointer(real), pointer(imag), a.arr, isbiased))
        else:
            safe_call(clib.af_var_all_weighted(pointer(real), pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def stdev(a, dim=None):
    if dim is not None:
        out = array()
        safe_call(clib.af_stdev(pointer(out.arr), a.arr, c_int(dim)))
        return out
    else:
        real = c_double(0)
        imag = c_double(0)
        safe_call(clib.af_stdev_all(pointer(real), pointer(imag), a.arr))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def cov(a, isbiased=False, dim=None):
    if dim is not None:
        out = array()
        safe_call(clib.af_cov(pointer(out.arr), a.arr, isbiased, c_int(dim)))
        return out
    else:
        real = c_double(0)
        imag = c_double(0)
        safe_call(clib.af_cov_all(pointer(real), pointer(imag), a.arr, isbiased))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def median(a, dim=None):
    if dim is not None:
        out = array()
        safe_call(clib.af_median(pointer(out.arr), a.arr, c_int(dim)))
        return out
    else:
        real = c_double(0)
        imag = c_double(0)
        safe_call(clib.af_median_all(pointer(real), pointer(imag), a.arr))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def corrcoef(x, y):
    real = c_double(0)
    imag = c_double(0)
    safe_call(clib.af_corrcoef(pointer(real), pointer(imag), x.arr, y.arr))
    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j
