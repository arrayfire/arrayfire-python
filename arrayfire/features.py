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

class Features(object):

    def __init__(self, num=None):
        self.feat = ct.c_void_p(0)
        if num is not None:
            assert(isinstance(num, numbers.Number))
            safe_call(clib.af_create_features(ct.pointer(self.feat), ct.c_longlong(num)))

    def num_features():
        num = ct.c_longlong(0)
        safe_call(clib.af_get_features_num(ct.pointer(num), self.feat))
        return num

    def get_xpos():
        out = Array()
        safe_call(clib.af_get_features_xpos(ct.pointer(out.arr), self.feat))
        return out

    def get_ypos():
        out = Array()
        safe_call(clib.af_get_features_ypos(ct.pointer(out.arr), self.feat))
        return out

    def get_score():
        out = Array()
        safe_call(clib.af_get_features_score(ct.pointer(out.arr), self.feat))
        return out

    def get_orientation():
        out = Array()
        safe_call(clib.af_get_features_orientation(ct.pointer(out.arr), self.feat))
        return out

    def get_size():
        out = Array()
        safe_call(clib.af_get_features_size(ct.pointer(out.arr), self.feat))
        return out
