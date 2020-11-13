#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Implementation of BaseArray class.
"""

from .library import c_void_ptr_t


class BaseArray:
    """
    Base array class for arrayfire. For internal use only.
    """

    def __init__(self):
        self.arr = c_void_ptr_t(0)
