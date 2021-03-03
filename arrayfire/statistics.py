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
    """
    Calculate mean along a given dimension.

    Parameters
    ----------
    a: af.Array
        The input array.

    weights: optional: af.Array. default: None.
        Array to calculate the weighted mean. Must match size of the
        input array.

    dim: optional: int. default: None.
        The dimension for which to obtain the mean from input data.

    Returns
    -------
    output: af.Array
        Array containing the mean of the input array along a given
        dimension.
    """
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

def var(a, bias=VARIANCE.DEFAULT, weights=None, dim=None):
    """
    Calculate variance along a given dimension.

    Parameters
    ----------
    a: af.Array
        The input array.

    bias: optional: af.VARIANCE. default: DEFAULT.
        population variance(VARIANCE.POPULATION) or sample variance(VARIANCE.SAMPLE).
        This is ignored if weights are provided.

    weights: optional: af.Array. default: None.
        Array to calculate for the weighted mean. Must match size of
        the input array.

    dim: optional: int. default: None.
        The dimension for which to obtain the variance from input data.

    Returns
    -------
    output: af.Array
        Array containing the variance of the input array along a given
        dimension.
    """
    if dim is not None:
        out = Array()

        if weights is None:
            safe_call(backend.get().af_var_v2(c_pointer(out.arr), a.arr, bias.value, c_int_t(dim)))
        else:
            safe_call(backend.get().af_var_weighted(c_pointer(out.arr), a.arr, weights.arr, c_int_t(dim)))

        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)

        if weights is None:
            safe_call(backend.get().af_var_all_v2(c_pointer(real), c_pointer(imag), a.arr, bias.value))
        else:
            safe_call(backend.get().af_var_all_weighted(c_pointer(real), c_pointer(imag), a.arr, weights.arr))

        real = real.value
        imag = imag.value

        return real if imag == 0 else real + imag * 1j

def meanvar(a, weights=None, bias=VARIANCE.DEFAULT, dim=-1):
    """
    Calculate mean and variance along a given dimension.

    Parameters
    ----------
    a: af.Array
        The input array.

    weights: optional: af.Array. default: None.
        Array to calculate for the weighted mean. Must match size of
        the input array.

    bias: optional: af.VARIANCE. default: DEFAULT.
        population variance(VARIANCE.POPULATION) or
        sample variance(VARIANCE.SAMPLE).

    dim: optional: int. default: -1.
        The dimension for which to obtain the variance from input data.

    Returns
    -------
    mean: af.Array
        Array containing the mean of the input array along a given
        dimension.
    variance: af.Array
        Array containing the variance of the input array along a given
        dimension.
    """

    mean_out = Array()
    var_out  = Array()

    if weights is None:
        weights  = Array()

    safe_call(backend.get().af_meanvar(c_pointer(mean_out.arr), c_pointer(var_out.arr),
                                       a.arr, weights.arr, bias.value, c_int_t(dim)))

    return mean_out, var_out


def stdev(a, bias=VARIANCE.DEFAULT, dim=None):
    """
    Calculate standard deviation along a given dimension.

    Parameters
    ----------
    a: af.Array
        The input array.

    bias: optional: af.VARIANCE. default: DEFAULT.
        population variance(VARIANCE.POPULATION) or sample variance(VARIANCE.SAMPLE).
        This is ignored if weights are provided.

    dim: optional: int. default: None.
        The dimension for which to obtain the standard deviation from
        input data.

    Returns
    -------
    output: af.Array
        Array containing the standard deviation of the input array
        along a given dimension.
    """
    if dim is not None:
        out = Array()
        safe_call(backend.get().af_stdev_v2(c_pointer(out.arr), a.arr, bias.value,
                                            c_int_t(dim)))
        return out
    else:
        real = c_double_t(0)
        imag = c_double_t(0)
        safe_call(backend.get().af_stdev_all_v2(c_pointer(real), c_pointer(imag), a.arr,
                                                bias.value))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j

def cov(a, b, bias=VARIANCE.DEFAULT):
    """
    Calculate covariance along a given dimension.

    Parameters
    ----------
    a: af.Array
        Input array.

    b: af.Array
        Input array.

    bias: optional: af.VARIANCE. default: DEFAULT.
        population variance(VARIANCE.POPULATION) or sample variance(VARIANCE.SAMPLE).

    Returns
    -------
    output: af.Array
        Array containing the covariance of the input array along a given dimension.
    """
    out = Array()
    safe_call(backend.get().af_cov_v2(c_pointer(out.arr), a.arr, b.arr, bias.value))
    return out

def median(a, dim=None):
    """
    Calculate median along a given dimension.

    Parameters
    ----------
    a: af.Array
        The input array.

    dim: optional: int. default: None.
        The dimension for which to obtain the median from input data.

    Returns
    -------
    output: af.Array
        Array containing the median of the input array along a
        given dimension.
    """
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
    """
    Calculate the correlation coefficient of the input arrays.

    Parameters
    ----------
    x: af.Array
        The first input array.

    y: af.Array
        The second input array.

    Returns
    -------
    output: af.Array
        Array containing the correlation coefficient of the input arrays.
    """
    real = c_double_t(0)
    imag = c_double_t(0)
    safe_call(backend.get().af_corrcoef(c_pointer(real), c_pointer(imag), x.arr, y.arr))
    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j

def topk(data, k, dim=0, order=TOPK.DEFAULT):
    """
    Return top k elements along a single dimension.

    Parameters
    ----------

    data: af.Array
          Input array to return k elements from.

    k: scalar. default: 0
       The number of elements to return from input array.

    dim: optional: scalar. default: 0
         The dimension along which the top k elements are
         extracted. Note: at the moment, topk() only supports the
         extraction of values along the first dimension.

    order: optional: af.TOPK. default: af.TOPK.DEFAULT
           The ordering of k extracted elements. Defaults to top k max values.

    Returns
    -------

    values: af.Array
            Top k elements from input array.
    indices: af.Array
             Corresponding index array to top k elements.
    """

    values = Array()
    indices = Array()

    safe_call(backend.get().af_topk(c_pointer(values.arr), c_pointer(indices.arr), data.arr, k, c_int_t(dim), order.value))

    return values,indices
