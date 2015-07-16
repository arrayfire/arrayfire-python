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
from .features import *

def fast(image, threshold=20.0, arc_length=9, non_max=True, feature_ratio=0.05, edge=3):
    out = features()
    safe_call(clib.af_fast(ct.pointer(out.feat),\
                           image.arr, ct.c_float(threshold), ct.c_uint(arc_length), non_max, \
                           ct.c_float(feature_ratio), ct.c_uint(edge)))
    return out

def orb(image, threshold=20.0, max_features=400, scale = 1.5, num_levels = 4, blur_image = False):
    feat = features()
    desc = array()
    safe_call(clib.af_orb(ct.pointer(feat.feat), ct.pointer(desc.arr),\
                          ct.c_float(threshold), ct.c_uint(max_features),\
                          ct.c_float(scale), ct.c_uint(num_levels), blur_image))
    return feat, desc

def hamming_matcher(query, database, dim = 0, num_nearest = 1):
    index = array()
    dist = array()
    safe_call(clib.af_hamming_matcher(ct.pointer(idx.arr), ct.pointer(dist.arr),\
                                      query.arr, database.arr, \
                                      ct.c_longlong(dim), ct.c_longlong(num_nearest)))
    return index, dist

def match_template(image, template, match_type = AF_SAD):
    out = array()
    safe_call(clib.af_match_template(ct.pointer(out.arr), image.arr, template.arr, match_type))
    return out
