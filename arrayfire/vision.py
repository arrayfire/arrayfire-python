#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Computer vision functions for arrayfire.
"""

from .library import *
from .array import *
from .features import *

def fast(image, threshold=20.0, arc_length=9, non_max=True, feature_ratio=0.05, edge=3):
    """
    FAST feature detector.

    Parameters
    ----------

    image         : af.Array
                  A 2D array representing an image.

    threshold     : scalar. optional. default: 20.0.
                  FAST threshold for which a pixel of the circle around a central pixel is consdered.

    arc_length    : scalar. optional. default: 9
                  The minimum length of arc length to be considered. Max length should be 16.

    non_max       : Boolean. optional. default: True
                  A boolean flag specifying if non max suppression has to be performed.

    feature_ratio : scalar. optional. default: 0.05 (5%)
                  Specifies the maximum ratio of features to pixels in the image.

    edge          : scalar. optional. default: 3.
                  Specifies the number of edge rows and columns to be ignored.

    Returns
    ---------
    features     : af.Features()
                 - x, y, and score are calculated
                 - orientation is 0 because FAST does not compute orientation
                 - size is 1 because FAST does not compute multiple scales

    """
    out = Features()
    safe_call(backend.get().af_fast(ct.pointer(out.feat),
                                    image.arr, ct.c_float(threshold), ct.c_uint(arc_length), non_max,
                                    ct.c_float(feature_ratio), ct.c_uint(edge)))
    return out

def orb(image, threshold=20.0, max_features=400, scale = 1.5, num_levels = 4, blur_image = False):
    """
    ORB Feature descriptor.

    Parameters
    ----------

    image         : af.Array
                  A 2D array representing an image.

    threshold     : scalar. optional. default: 20.0.
                  FAST threshold for which a pixel of the circle around a central pixel is consdered.

    max_features  : scalar. optional. default: 400.
                  Specifies the maximum number of features to be considered.

    scale         : scalar. optional. default: 1.5.
                  Specifies the factor by which images are down scaled at each level.

    num_levles    : scalar. optional. default: 4.
                  Specifies the number of levels used in the image pyramid.

    blur_image    : Boolean. optional. default: False.
                  Flag specifying if the input has to be blurred before computing descriptors.
                  A gaussian filter with sigma = 2 is applied if True.


    Returns
    ---------
    (features, descriptor)     : tuple of (af.Features(), af.Array)

    """
    feat = Features()
    desc = Array()
    safe_call(backend.get().af_orb(ct.pointer(feat.feat), ct.pointer(desc.arr),
                                   ct.c_float(threshold), ct.c_uint(max_features),
                                   ct.c_float(scale), ct.c_uint(num_levels), blur_image))
    return feat, desc

def hamming_matcher(query, database, dim = 0, num_nearest = 1):
    """
    Hamming distance matcher.

    Parameters
    -----------

    query    : af.Array
             A query feature descriptor

    database : af.Array
             A multi dimensional array containing the feature descriptor database.

    dim      : scalar. optional. default: 0.
             Specifies the dimension along which feature descriptor lies.

    num_neaarest: scalar. optional. default: 1.
             Specifies the number of nearest neighbors to find.

    Returns
    ---------

    (location, distance): tuple of af.Array
                          location and distances of closest matches.

    """
    index = Array()
    dist = Array()
    safe_call(backend.get().af_hamming_matcher(ct.pointer(idx.arr), ct.pointer(dist.arr),
                                               query.arr, database.arr,
                                               ct.c_longlong(dim), ct.c_longlong(num_nearest)))
    return index, dist

def match_template(image, template, match_type = MATCH.SAD):
    """
    Find the closest match of a template in an image.

    Parameters
    ----------

    image    : af.Array
             A multi dimensional array specifying an image or batch of images.

    template : af.Array
             A multi dimensional array specifying a template or batch of templates.

    match_type: optional: af.MATCH. default: af.MATCH.SAD
             Specifies the match function metric.

    Returns
    --------
    out     : af.Array
            An array containing the score of the match at each pixel.

    """
    out = Array()
    safe_call(backend.get().af_match_template(ct.pointer(out.arr), image.arr, template.arr, match_type))
    return out
