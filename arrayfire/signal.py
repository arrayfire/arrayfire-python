#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
signal processing functions for arrayfire.
"""

from .library import *
from .array import *

def approx1(signal, pos0, method=INTERP.LINEAR, off_grid=0.0):
    """
    Interpolate along a single dimension.

    Parameters
    ----------

    signal: af.Array
            A 1 dimensional signal or batch of 1 dimensional signals.

    pos0  : af.Array
            Locations of the interpolation points.

    method: optional: af.INTERP. default: af.INTERP.LINEAR.
            Interpolation method.

    off_grid: optional: scalar. default: 0.0.
            The value used for positions outside the range.

    Returns
    -------

    output: af.Array
            Values calculated at interpolation points.

    Note
    -----

    The initial measurements are assumed to have taken place at equal steps between [0, N - 1],
    where N is the length of the first dimension of `signal`.


    """
    output = Array()
    safe_call(backend.get().af_approx1(ct.pointer(output.arr), signal.arr, pos0.arr,
                                       method.value, ct.c_double(off_grid)))
    return output

def approx2(signal, pos0, pos1, method=INTERP.LINEAR, off_grid=0.0):
    """
    Interpolate along a two dimension.

    Parameters
    ----------

    signal: af.Array
            A 2 dimensional signal or batch of 2 dimensional signals.

    pos0  : af.Array
            Locations of the interpolation points along the first dimension.

    pos1  : af.Array
            Locations of the interpolation points along the second dimension.

    method: optional: af.INTERP. default: af.INTERP.LINEAR.
            Interpolation method.

    off_grid: optional: scalar. default: 0.0.
            The value used for positions outside the range.

    Returns
    -------

    output: af.Array
            Values calculated at interpolation points.

    Note
    -----

    The initial measurements are assumed to have taken place at equal steps between [(0,0) - [M - 1, N - 1]]
    where M is the length of the first dimension of `signal`,
    and N is the length of the second dimension of `signal`.


    """
    output = Array()
    safe_call(backend.get().af_approx2(ct.pointer(output.arr), signal.arr,
                                       pos0.arr, pos1.arr, method.value, ct.c_double(off_grid)))
    return output

def fft(signal, dim0 = None , scale = None):
    """
    Fast Fourier Transform: 1D

    Parameters
    ----------

    signal: af.Array
           A 1 dimensional signal or a batch of 1 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    Returns
    -------

    output: af.Array
            A complex af.Array containing the full output of the fft.

    """

    if dim0 is None:
        dim0 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft(ct.pointer(output.arr), signal.arr, ct.c_double(scale), c_dim_t(dim0)))
    return output

def fft2(signal, dim0 = None, dim1 = None , scale = None):
    """
    Fast Fourier Transform: 2D

    Parameters
    ----------

    signal: af.Array
           A 2 dimensional signal or a batch of 2 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    dim1: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim1 is calculated to be the second dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    Returns
    -------

    output: af.Array
            A complex af.Array containing the full output of the fft.

    """
    if dim0 is None:
        dim0 = 0

    if dim1 is None:
        dim1 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft2(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                    c_dim_t(dim0), c_dim_t(dim1)))
    return output

def fft3(signal, dim0 = None, dim1 = None , dim2 = None, scale = None):
    """
    Fast Fourier Transform: 3D

    Parameters
    ----------

    signal: af.Array
           A 3 dimensional signal or a batch of 3 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    dim1: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim1 is calculated to be the second dimension of `signal`.

    dim2: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim2 is calculated to be the third dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    Returns
    -------

    output: af.Array
            A complex af.Array containing the full output of the fft.

    """
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
                                    c_dim_t(dim0), c_dim_t(dim1), c_dim_t(dim2)))
    return output

def ifft(signal, dim0 = None , scale = None):
    """
    Inverse Fast Fourier Transform: 1D

    Parameters
    ----------

    signal: af.Array
           A 1 dimensional signal or a batch of 1 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.0 / (dim0)

    Returns
    -------

    output: af.Array
            A complex af.Array containing the full output of the inverse fft.

    Note
    ----

    The output is always complex.

    """

    if dim0 is None:
        dim0 = signal.dims()[0]

    if scale is None:
        scale = 1.0/float(dim0)

    output = Array()
    safe_call(backend.get().af_ifft(ct.pointer(output.arr), signal.arr, ct.c_double(scale), c_dim_t(dim0)))
    return output

def ifft2(signal, dim0 = None, dim1 = None , scale = None):
    """
    Inverse Fast Fourier Transform: 2D

    Parameters
    ----------

    signal: af.Array
           A 2 dimensional signal or a batch of 2 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    dim1: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim1 is calculated to be the second dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.0 / (dim0 * dim1)

    Returns
    -------

    output: af.Array
            A complex af.Array containing the full output of the inverse fft.

    Note
    ----

    The output is always complex.

    """

    dims = signal.dims()

    if dim0 is None:
        dim0 = dims[0]

    if dim1 is None:
        dim1 = dims[1]

    if scale is None:
        scale = 1.0/float(dim0 * dim1)

    output = Array()
    safe_call(backend.get().af_ifft2(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                     c_dim_t(dim0), c_dim_t(dim1)))
    return output

def ifft3(signal, dim0 = None, dim1 = None , dim2 = None, scale = None):
    """
    Inverse Fast Fourier Transform: 3D

    Parameters
    ----------

    signal: af.Array
           A 3 dimensional signal or a batch of 3 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    dim1: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim1 is calculated to be the second dimension of `signal`.

    dim2: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim2 is calculated to be the third dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.0 / (dim0 * dim1 * dim2).

    Returns
    -------

    output: af.Array
            A complex af.Array containing the full output of the inverse fft.

    Note
    ----

    The output is always complex.

    """

    dims = signal.dims()

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
                                     c_dim_t(dim0), c_dim_t(dim1), c_dim_t(dim2)))
    return output

def fft_inplace(signal, scale = None):
    """
    In-place Fast Fourier Transform: 1D

    Parameters
    ----------

    signal: af.Array
           A 1 dimensional signal or a batch of 1 dimensional signals.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    """

    if scale is None:
        scale = 1.0

    safe_call(backend.get().af_fft_inplace(signal.arr, ct.c_double(scale)))

def fft2_inplace(signal, scale = None):
    """
    In-place Fast Fourier Transform: 2D

    Parameters
    ----------

    signal: af.Array
           A 2 dimensional signal or a batch of 2 dimensional signals.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    """

    if scale is None:
        scale = 1.0

    safe_call(backend.get().af_fft2_inplace(signal.arr, ct.c_double(scale)))

def fft3_inplace(signal, scale = None):
    """
    In-place Fast Fourier Transform: 3D

    Parameters
    ----------

    signal: af.Array
           A 3 dimensional signal or a batch of 3 dimensional signals.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.
    """

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft3_inplace(signal.arr, ct.c_double(scale)))

def ifft_inplace(signal, scale = None):
    """
    Inverse In-place Fast Fourier Transform: 1D

    Parameters
    ----------

    signal: af.Array
           A 1 dimensional signal or a batch of 1 dimensional signals.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.0 / (signal.dims()[0])
    """

    if scale is None:
        dim0 = signal.dims()[0]
        scale = 1.0/float(dim0)

    safe_call(backend.get().af_ifft_inplace(signal.arr, ct.c_double(scale)))

def ifft2_inplace(signal, scale = None):
    """
    Inverse In-place Fast Fourier Transform: 2D

    Parameters
    ----------

    signal: af.Array
           A 2 dimensional signal or a batch of 2 dimensional signals.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.0 / (signal.dims()[0] * signal.dims()[1])
    """

    dims = signal.dims()

    if scale is None:
        dim0 = dims[0]
        dim1 = dims[1]
        scale = 1.0/float(dim0 * dim1)

    safe_call(backend.get().af_ifft2_inplace(signal.arr, ct.c_double(scale)))

def ifft3_inplace(signal, scale = None):
    """
    Inverse In-place Fast Fourier Transform: 3D

    Parameters
    ----------

    signal: af.Array
           A 3 dimensional signal or a batch of 3 dimensional signals.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.0 / (signal.dims()[0] * signal.dims()[1] * signal.dims()[2]).
    """

    dims = signal.dims()

    if scale is None:
        dim0 = dims[0]
        dim1 = dims[1]
        dim2 = dims[2]
        scale = 1.0 / float(dim0 * dim1 * dim2)

    safe_call(backend.get().af_ifft3_inplace(signal.arr, ct.c_double(scale)))

def fft_r2c(signal, dim0 = None , scale = None):
    """
    Real to Complex Fast Fourier Transform: 1D

    Parameters
    ----------

    signal: af.Array
           A 1 dimensional signal or a batch of 1 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    Returns
    -------

    output: af.Array
            A complex af.Array containing the non-redundant parts of the full FFT.

    """

    if dim0 is None:
        dim0 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft_r2c(ct.pointer(output.arr), signal.arr, ct.c_double(scale), c_dim_t(dim0)))
    return output

def fft2_r2c(signal, dim0 = None, dim1 = None , scale = None):
    """
    Real to Complex Fast Fourier Transform: 2D

    Parameters
    ----------

    signal: af.Array
           A 2 dimensional signal or a batch of 2 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    dim1: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim1 is calculated to be the second dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    Returns
    -------

    output: af.Array
            A complex af.Array containing the non-redundant parts of the full FFT.

    """
    if dim0 is None:
        dim0 = 0

    if dim1 is None:
        dim1 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft2_r2c(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                        c_dim_t(dim0), c_dim_t(dim1)))
    return output

def fft3_r2c(signal, dim0 = None, dim1 = None , dim2 = None, scale = None):
    """
    Real to Complex Fast Fourier Transform: 3D

    Parameters
    ----------

    signal: af.Array
           A 3 dimensional signal or a batch of 3 dimensional signals.

    dim0: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim0 is calculated to be the first dimension of `signal`.

    dim1: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim1 is calculated to be the second dimension of `signal`.

    dim2: optional: int. default: None.
          - Specifies the size of the output.
          - If None, dim2 is calculated to be the third dimension of `signal`.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1.

    Returns
    -------

    output: af.Array
            A complex af.Array containing the non-redundant parts of the full FFT.

    """
    if dim0 is None:
        dim0 = 0

    if dim1 is None:
        dim1 = 0

    if dim2 is None:
        dim2 = 0

    if scale is None:
        scale = 1.0

    output = Array()
    safe_call(backend.get().af_fft3_r2c(ct.pointer(output.arr), signal.arr, ct.c_double(scale),
                                        c_dim_t(dim0), c_dim_t(dim1), c_dim_t(dim2)))
    return output

def _get_c2r_dim(dim, is_odd):
    return 2 *(dim - 1) + int(is_odd)

def fft_c2r(signal, is_odd = False, scale = None):
    """
    Real to Complex Fast Fourier Transform: 1D

    Parameters
    ----------

    signal: af.Array
           A 1 dimensional signal or a batch of 1 dimensional signals.

    is_odd: optional: Boolean. default: False.
          - Specifies if the first dimension of output should be even or odd.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1 / (signal.dims()[0]).

    Returns
    -------

    output: af.Array
            A real af.Array containing the full output of the fft.

    """


    if scale is None:
        dim0 = _get_c2r_dim(signal.dims()[0], is_odd)
        scale = 1.0/float(dim0)

    output = Array()
    safe_call(backend.get().af_fft_c2r(ct.pointer(output.arr), signal.arr, ct.c_double(scale), is_odd))
    return output

def fft2_c2r(signal, is_odd = False, scale = None):
    """
    Real to Complex Fast Fourier Transform: 2D

    Parameters
    ----------

    signal: af.Array
           A 2 dimensional signal or a batch of 2 dimensional signals.

    is_odd: optional: Boolean. default: False.
          - Specifies if the first dimension of output should be even or odd.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1 / (signal.dims()[0] * signal.dims()[1]).

    Returns
    -------

    output: af.Array
            A real af.Array containing the full output of the fft.

    """
    dims = signal.dims()

    if scale is None:
        dim0 = _get_c2r_dim(dims[0], is_odd)
        dim1 = dims[1]
        scale = 1.0/float(dim0 * dim1)

    output = Array()
    safe_call(backend.get().af_fft2_c2r(ct.pointer(output.arr), signal.arr, ct.c_double(scale), is_odd))
    return output

def fft3_c2r(signal, is_odd = False, scale = None):
    """
    Real to Complex Fast Fourier Transform: 3D

    Parameters
    ----------

    signal: af.Array
           A 3 dimensional signal or a batch of 3 dimensional signals.

    is_odd: optional: Boolean. default: False.
          - Specifies if the first dimension of output should be even or odd.

    scale: optional: scalar. default: None.
          - Specifies the scaling factor.
          - If None, scale is set to 1 / (signal.dims()[0] * signal.dims()[1] * signal.dims()[2]).

    Returns
    -------

    output: af.Array
            A real af.Array containing the full output of the fft.

    """
    dims = signal.dims()

    if scale is None:
        dim0 = _get_c2r_dim(dims[0], is_odd)
        dim1 = dims[1]
        dim2 = dims[2]
        scale = 1.0/float(dim0 * dim1 * dim2)

    output = Array()
    safe_call(backend.get().af_fft3_c2r(ct.pointer(output.arr), signal.arr, ct.c_double(scale), is_odd))
    return output


def dft(signal, odims=(None, None, None, None), scale = None):

    """
    Non batched Fourier transform.

    This function performs n-dimensional fourier transform depending on the input dimensions.

    Parameters
    ----------

    signal: af.Array
          - A multi dimensional arrayfire array.

    odims: optional: tuple of ints. default: (None, None, None, None).
          - If None, calculated to be `signal.dims()`

    scale: optional: scalar. default: None.
           - Scale factor for the fourier transform.
           - If none, calculated to be 1.0.

    Returns
    -------
    output: af.Array
           - A complex array that is the ouput of n-dimensional fourier transform.

    """

    odims4 = dim4_to_tuple(odims, default=None)

    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return fft(signal, dims[0], scale)
    elif (ndims == 2):
        return fft2(signal, dims[0], dims[1], scale)
    else:
        return fft3(signal, dims[0], dims[1], dims[2], scale)

def idft(signal, scale = None, odims=(None, None, None, None)):
    """
    Non batched Inverse Fourier transform.

    This function performs n-dimensional inverse fourier transform depending on the input dimensions.

    Parameters
    ----------

    signal: af.Array
          - A multi dimensional arrayfire array.

    odims: optional: tuple of ints. default: (None, None, None, None).
          - If None, calculated to be `signal.dims()`

    scale: optional: scalar. default: None.
           - Scale factor for the fourier transform.
           - If none, calculated to be 1.0 / signal.elements()

    Returns
    -------
    output: af.Array
           - A complex array that is the ouput of n-dimensional inverse fourier transform.

    Note
    ----

    the output is always complex.

    """

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
    """
    Convolution: 1D

    Parameters
    -----------

    signal: af.Array
            - A 1 dimensional signal or batch of 1 dimensional signals.

    kernel: af.Array
            - A 1 dimensional kernel or batch of 1 dimensional kernels.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    conv_domain: optional: af.CONV_DOMAIN. default: af.CONV_DOMAIN.AUTO.
            - Specifies the domain in which convolution is performed.
            - af.CONV_DOMAIN.SPATIAL: Performs convolution in spatial domain.
            - af.CONV_DOMAIN.FREQ: Performs convolution in frequency domain.
            - af.CONV_DOMAIN.AUTO: Switches between spatial and frequency based on input size.

    Returns
    --------

    output: af.Array
          - Output of 1D convolution.

    Note
    -----

    Supported batch combinations:

    | Signal    | Kernel    | output    |
    |:---------:|:---------:|:---------:|
    | [m 1 1 1] | [m 1 1 1] | [m 1 1 1] |
    | [m n 1 1] | [m n 1 1] | [m n 1 1] |
    | [m n p 1] | [m n 1 1] | [m n p 1] |
    | [m n p 1] | [m n p 1] | [m n p 1] |
    | [m n p 1] | [m n 1 q] | [m n p q] |
    | [m n 1 p] | [m n q 1] | [m n q p] |

    """
    output = Array()
    safe_call(backend.get().af_convolve1(ct.pointer(output.arr), signal.arr, kernel.arr,
                                         conv_mode.value, conv_domain.value))
    return output

def convolve2(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    """
    Convolution: 2D

    Parameters
    -----------

    signal: af.Array
            - A 2 dimensional signal or batch of 2 dimensional signals.

    kernel: af.Array
            - A 2 dimensional kernel or batch of 2 dimensional kernels.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    conv_domain: optional: af.CONV_DOMAIN. default: af.CONV_DOMAIN.AUTO.
            - Specifies the domain in which convolution is performed.
            - af.CONV_DOMAIN.SPATIAL: Performs convolution in spatial domain.
            - af.CONV_DOMAIN.FREQ: Performs convolution in frequency domain.
            - af.CONV_DOMAIN.AUTO: Switches between spatial and frequency based on input size.

    Returns
    --------

    output: af.Array
          - Output of 2D convolution.

    Note
    -----

    Supported batch combinations:

    | Signal    | Kernel    | output    |
    |:---------:|:---------:|:---------:|
    | [m n 1 1] | [m n 1 1] | [m n 1 1] |
    | [m n p 1] | [m n 1 1] | [m n p 1] |
    | [m n p 1] | [m n p 1] | [m n p 1] |
    | [m n p 1] | [m n 1 q] | [m n p q] |
    | [m n 1 p] | [m n q 1] | [m n q p] |

    """
    output = Array()
    safe_call(backend.get().af_convolve2(ct.pointer(output.arr), signal.arr, kernel.arr,
                                         conv_mode.value, conv_domain.value))
    return output

def convolve2_separable(col_kernel, row_kernel, signal, conv_mode = CONV_MODE.DEFAULT):
    """
    Convolution: 2D separable convolution

    Parameters
    -----------

    col_kernel: af.Array
            - A column vector to be applied along each column of `signal`

    row_kernel: af.Array
            - A row vector to be applied along each row of `signal`

    signal: af.Array
            - A 2 dimensional signal or batch of 2 dimensional signals.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).
    Returns
    --------

    output: af.Array
          - Output of 2D sepearable convolution.
    """
    output = Array()
    safe_call(backend.get().af_convolve2_sep(ct.pointer(output.arr),
                                             col_kernel.arr, row_kernel.arr,signal.arr,
                                             conv_mode.value))
    return output

def convolve3(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    """
    Convolution: 3D

    Parameters
    -----------

    signal: af.Array
            - A 3 dimensional signal or batch of 3 dimensional signals.

    kernel: af.Array
            - A 3 dimensional kernel or batch of 3 dimensional kernels.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    conv_domain: optional: af.CONV_DOMAIN. default: af.CONV_DOMAIN.AUTO.
            - Specifies the domain in which convolution is performed.
            - af.CONV_DOMAIN.SPATIAL: Performs convolution in spatial domain.
            - af.CONV_DOMAIN.FREQ: Performs convolution in frequency domain.
            - af.CONV_DOMAIN.AUTO: Switches between spatial and frequency based on input size.

    Returns
    --------

    output: af.Array
          - Output of 3D convolution.

    Note
    -----

    Supported batch combinations:

    | Signal    | Kernel    | output    |
    |:---------:|:---------:|:---------:|
    | [m n p 1] | [m n p 1] | [m n p 1] |
    | [m n p 1] | [m n p q] | [m n p q] |
    | [m n q p] | [m n q p] | [m n q p] |

    """
    output = Array()
    safe_call(backend.get().af_convolve3(ct.pointer(output.arr), signal.arr, kernel.arr,
                                         conv_mode.value, conv_domain.value))
    return output

def convolve(signal, kernel, conv_mode = CONV_MODE.DEFAULT, conv_domain = CONV_DOMAIN.AUTO):
    """
    Non batched Convolution.

    This function performs n-dimensional convolution based on input dimensionality.

    Parameters
    -----------

    signal: af.Array
            - An n-dimensional array.

    kernel: af.Array
            - A n-dimensional kernel.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    conv_domain: optional: af.CONV_DOMAIN. default: af.CONV_DOMAIN.AUTO.
            - Specifies the domain in which convolution is performed.
            - af.CONV_DOMAIN.SPATIAL: Performs convolution in spatial domain.
            - af.CONV_DOMAIN.FREQ: Performs convolution in frequency domain.
            - af.CONV_DOMAIN.AUTO: Switches between spatial and frequency based on input size.

    Returns
    --------

    output: af.Array
          - Output of n-dimensional convolution.
    """

    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return convolve1(signal, kernel, conv_mode, conv_domain)
    elif (ndims == 2):
        return convolve2(signal, kernel, conv_mode, conv_domain)
    else:
        return convolve3(signal, kernel, conv_mode, conv_domain)

def fft_convolve1(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    """
    FFT based Convolution: 1D

    Parameters
    -----------

    signal: af.Array
            - A 1 dimensional signal or batch of 1 dimensional signals.

    kernel: af.Array
            - A 1 dimensional kernel or batch of 1 dimensional kernels.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    Returns
    --------

    output: af.Array
          - Output of 1D convolution.

    Note
    -----

    This is same as convolve1(..., conv_mode=af.CONV_MODE.FREQ)

    Supported batch combinations:

    | Signal    | Kernel    | output    |
    |:---------:|:---------:|:---------:|
    | [m 1 1 1] | [m 1 1 1] | [m 1 1 1] |
    | [m n 1 1] | [m n 1 1] | [m n 1 1] |
    | [m n p 1] | [m n 1 1] | [m n p 1] |
    | [m n p 1] | [m n p 1] | [m n p 1] |
    | [m n p 1] | [m n 1 q] | [m n p q] |
    | [m n 1 p] | [m n q 1] | [m n q p] |

    """
    output = Array()
    safe_call(backend.get().af_fft_convolve1(ct.pointer(output.arr), signal.arr, kernel.arr,
                                             conv_mode.value))
    return output

def fft_convolve2(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    """
    FFT based Convolution: 2D

    Parameters
    -----------

    signal: af.Array
            - A 2 dimensional signal or batch of 2 dimensional signals.

    kernel: af.Array
            - A 2 dimensional kernel or batch of 2 dimensional kernels.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    Returns
    --------

    output: af.Array
          - Output of 2D convolution.

    Note
    -----

    This is same as convolve2(..., conv_mode=af.CONV_MODE.FREQ)

    Supported batch combinations:

    | Signal    | Kernel    | output    |
    |:---------:|:---------:|:---------:|
    | [m n 1 1] | [m n 1 1] | [m n 1 1] |
    | [m n p 1] | [m n 1 1] | [m n p 1] |
    | [m n p 1] | [m n p 1] | [m n p 1] |
    | [m n p 1] | [m n 1 q] | [m n p q] |
    | [m n 1 p] | [m n q 1] | [m n q p] |

    """
    output = Array()
    safe_call(backend.get().af_fft_convolve2(ct.pointer(output.arr), signal.arr, kernel.arr,
                                             conv_mode.value))
    return output

def fft_convolve3(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    """
    FFT based Convolution: 3D

    Parameters
    -----------

    signal: af.Array
            - A 3 dimensional signal or batch of 3 dimensional signals.

    kernel: af.Array
            - A 3 dimensional kernel or batch of 3 dimensional kernels.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    Returns
    --------

    output: af.Array
          - Output of 3D convolution.

    Note
    -----

    This is same as convolve3(..., conv_mode=af.CONV_MODE.FREQ)

    Supported batch combinations:

    | Signal    | Kernel    | output    |
    |:---------:|:---------:|:---------:|
    | [m n p 1] | [m n p 1] | [m n p 1] |
    | [m n p 1] | [m n p q] | [m n p q] |
    | [m n q p] | [m n q p] | [m n q p] |

    """
    output = Array()
    safe_call(backend.get().af_fft_convolve3(ct.pointer(output.arr), signal.arr, kernel.arr,
                                             conv_mode.value))
    return output

def fft_convolve(signal, kernel, conv_mode = CONV_MODE.DEFAULT):
    """
    Non batched FFT Convolution.

    This function performs n-dimensional convolution based on input dimensionality.

    Parameters
    -----------

    signal: af.Array
            - An n-dimensional array.

    kernel: af.Array
            - A n-dimensional kernel.

    conv_mode: optional: af.CONV_MODE. default: af.CONV_MODE.DEFAULT.
            - Specifies if the output does full convolution (af.CONV_MODE.EXPAND) or
              maintains the same size as input (af.CONV_MODE.DEFAULT).

    Returns
    --------

    output: af.Array
          - Output of n-dimensional convolution.

    Note
    -----

    This is same as convolve(..., conv_mode=af.CONV_MODE.FREQ)

    """
    dims = signal.dims()
    ndims = len(dims)

    if (ndims == 1):
        return fft_convolve1(signal, kernel, conv_mode)
    elif (ndims == 2):
        return fft_convolve2(signal, kernel, conv_mode)
    else:
        return fft_convolve3(signal, kernel, conv_mode)

def fir(B, X):
    """
    Finite impulse response filter.

    Parameters
    ----------

    B : af.Array
        A 1 dimensional array containing the coefficients of the filter.

    X : af.Array
        A 1 dimensional array containing the signal.

    Returns
    -------

    Y : af.Array
        The output of the filter.

    """
    Y = Array()
    safe_call(backend.get().af_fir(ct.pointer(Y.arr), B.arr, X.arr))
    return Y

def iir(B, A, X):
    """
    Infinite impulse response filter.

    Parameters
    ----------

    B : af.Array
        A 1 dimensional array containing the feed forward coefficients of the filter.

    A : af.Array
        A 1 dimensional array containing the feed back coefficients of the filter.

    X : af.Array
        A 1 dimensional array containing the signal.

    Returns
    -------

    Y : af.Array
        The output of the filter.

    """
    Y = Array()
    safe_call(backend.get().af_iir(ct.pointer(Y.arr), B.arr, A.arr, X.arr))
    return Y
