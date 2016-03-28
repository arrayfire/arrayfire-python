#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
"""
arrayfire.Features class
"""
from .library import *
from .array import *
import numbers

class Features(object):
    """
    A container class used for various feature detectors.

    Parameters
    ----------

    num: optional: int. default: 0.
         Specifies the number of features.
    """

    def __init__(self, num=0):
        self.feat = ct.c_void_p(0)
        if num is not None:
            assert(isinstance(num, numbers.Number))
            safe_call(backend.get().af_create_features(ct.pointer(self.feat), c_dim_t(num)))

    def num_features():
        """
        Returns the number of features detected.
        """
        num = c_dim_t(0)
        safe_call(backend.get().af_get_features_num(ct.pointer(num), self.feat))
        return num

    def get_xpos():
        """
        Returns the x-positions of the features detected.
        """
        out = Array()
        safe_call(backend.get().af_get_features_xpos(ct.pointer(out.arr), self.feat))
        return out

    def get_ypos():
        """
        Returns the y-positions of the features detected.
        """
        out = Array()
        safe_call(backend.get().af_get_features_ypos(ct.pointer(out.arr), self.feat))
        return out

    def get_score():
        """
        Returns the scores of the features detected.
        """
        out = Array()
        safe_call(backend.get().af_get_features_score(ct.pointer(out.arr), self.feat))
        return out

    def get_orientation():
        """
        Returns the orientations of the features detected.
        """
        out = Array()
        safe_call(backend.get().af_get_features_orientation(ct.pointer(out.arr), self.feat))
        return out

    def get_size():
        """
        Returns the sizes of the features detected.
        """
        out = Array()
        safe_call(backend.get().af_get_features_size(ct.pointer(out.arr), self.feat))
        return out
