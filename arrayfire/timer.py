#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
"""
Functions to time arrayfire functions
"""

from .library import *
from .device import (sync, eval)
from time import time

def timeit(af_func, *args, min_iters = 10):
    """
    Function to time arrayfire functions.

    Parameters
    ----------

    af_func    : arrayfire function

    *args      : arguments to `af_func`

    min_iters  : Minimum number of iterations to be run for `af_func`

    Returns
    --------

    t   : Time in seconds
    """

    res = af_func(*args)
    eval(res)
    sync()

    start = time()
    elapsed = 0
    num_iters = 0
    while elapsed < 1:
        for n in range(min_iters):
            res = af_func(*args)
            eval(res)
        sync()
        elapsed += time() - start
        num_iters += min_iters

    return elapsed / num_iters
