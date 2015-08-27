#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
from .library import *
from .util import *

class base_array(object):
    def __init__(self):
        self.arr = ct.c_void_p(0)
