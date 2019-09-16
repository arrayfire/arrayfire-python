#!/usr/bin/env python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from ._util import tests
from .algorithm import simple_algorithm
from .arith import simple_arith
from .array_test import simple_array
from .blas import simple_blas
from .data import simple_data
from .device import simple_device
from .image import simple_image
from .index import simple_index
from .interop import simple_interop
from .lapack import simple_lapack
from .random import simple_random
from .signal import simple_signal
from .sparse import simple_sparse
from .statistics import simple_statistics

__all__ = [
    "tests",
    "simple_algorithm",
    "simple_arith",
    "simple_array",
    "simple_blas",
    "simple_data",
    "simple_device",
    "simple_image",
    "simple_index",
    "simple_interop",
    "simple_lapack",
    "simple_random",
    "simple_signal",
    "simple_sparse",
    "simple_statistics"
]
