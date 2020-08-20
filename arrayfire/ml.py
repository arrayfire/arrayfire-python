#######################################################
# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Machine learning functions
    - Pool 2D, ND, maxpooling, minpooling, meanpooling
    - Forward and backward convolution passes
"""

from .library import *
from .array import *

def convolve2GradientNN(incoming_gradient, original_signal, original_kernel, convolved_output, stride = (1, 1), padding = (0, 0), dilation = (1, 1), gradType = CONV_GRADIENT.DEFAULT):
    """
    Function for calculating backward pass gradient of 2D convolution.

    This function calculates the gradient with respect to the output of the
    \ref convolve2NN() function that uses the machine learning formulation
    for the dimensions of the signals and filters

    Multiple signals and filters can be batched against each other, however
    their dimensions must match.

    Example:
        Signals with dimensions: d0 x d1 x d2 x Ns
        Filters with dimensions: d0 x d1 x d2 x Nf

        Resulting Convolution:   d0 x d1 x Nf x Ns

    Parameters
    -----------

    incoming_gradient: af.Array
            - Gradients to be distributed in backwards pass

    original_signal: af.Array
            - A 2 dimensional signal or batch of 2 dimensional signals.

    original_kernel: af.Array
            - A 2 dimensional kernel or batch of 2 dimensional kernels.

    convolved_output: af.Array
            - output of forward pass of convolution

    stride: tuple of ints. default: (1, 1).
            - Specifies how much to stride along each dimension

    padding: tuple of ints. default: (0, 0).
            - Specifies signal padding along each dimension

    dilation: tuple of ints. default: (1, 1).
            - Specifies how much to dilate kernel along each dimension before convolution

    Returns
    --------

    output: af.Array
          - Gradient wrt/requested gradient type

    """
    output = Array()
    stride_dim   = dim4(stride[0],   stride[1])
    padding_dim  = dim4(padding[0],  padding[1])
    dilation_dim = dim4(dilation[0], dilation[1])

    safe_call(backend.get().af_convolve2_gradient_nn(
                                            c_pointer(output.arr),
                                            incoming_gradient.arr,
                                            original_signal.arr,
                                            original_kernel.arr,
                                            convolved_output.arr,
                                            2, c_pointer(stride_dim),
                                            2, c_pointer(padding_dim),
                                            2, c_pointer(dilation_dim),
                                            gradType.value))
    return output

