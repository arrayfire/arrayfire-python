#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
from .library import *
from .array import *
import numbers

class features(object):

    def __init__(self, num=None):
        self.feat = c_longlong(0)
        if num is not None:
            assert(isinstance(num, numbers.Number))
            safe_call(clib.af_create_features(pointer(self.feat), c_longlong(num)))

    def num_features():
        num = c_longlong(0)
        safe_call(clib.af_get_features_num(pointer(num), self.feat))
        return num

    def get_xpos():
        out = array()
        safe_call(clib.af_get_features_xpos(pointer(out.arr), self.feat))
        return out

    def get_ypos():
        out = array()
        safe_call(clib.af_get_features_ypos(pointer(out.arr), self.feat))
        return out

    def get_score():
        out = array()
        safe_call(clib.af_get_features_score(pointer(out.arr), self.feat))
        return out

    def get_orientation():
        out = array()
        safe_call(clib.af_get_features_orientation(pointer(out.arr), self.feat))
        return out

    def get_size():
        out = array()
        safe_call(clib.af_get_features_size(pointer(out.arr), self.feat))
        return out
