#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Implementation of BaseArray class.
"""

from .library import *
from .util import *

class BaseArray(object):
    """
    Base array class for arrayfire. For internal use only.
    """
    def __init__(self):
        self.arr = ct.c_void_p(0)
