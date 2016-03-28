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
                 Contains the location and score. Orientation and size are not computed.

    """
    out = Features()
    safe_call(backend.get().af_fast(ct.pointer(out.feat),
                                    image.arr, ct.c_float(threshold), ct.c_uint(arc_length), non_max,
                                    ct.c_float(feature_ratio), ct.c_uint(edge)))
    return out

def harris(image, max_corners=500, min_response=1E5, sigma=1.0, block_size=0, k_thr=0.04):
    """
    Harris corner detector.

    Parameters
    ----------
    image         : af.Array
                  A 2D array specifying an image.

    max_corners   : scalar. optional. default: 500.
                  Specifies the maximum number of corners to be calculated.

    min_response  : scalar. optional. default: 1E5
                  Specifies the cutoff score for a corner to be considered

    sigma         : scalar. optional. default: 1.0
                  - Specifies the standard deviation of a circular window.
                  - Only used when block_size == 0. Must be >= 0.5 and <= 5.0.

    block_size    : scalar. optional. default: 0
                  Specifies the window size.

    k_thr         : scalar. optional. default: 0.04
                  Harris constant. must be >= 0.01

    Returns
    ---------

    features     : af.Features()
                 Contains the location and score. Orientation and size are not computed.

    Note
    ------

    The covariation matrix will be square when `block_size` is used and circular when `sigma` is used.


    """
    out = Features()
    safe_call(backend.get().af_harris(ct.pointer(out.feat),
                                      image.arr, ct.c_uint(max_corners), ct.c_float(min_response),
                                      ct.c_float(sigma), ct.c_uint(block_size), ct.c_float(k_thr)))
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
                               - descriptor is an af.Array of size N x 8

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

    num_nearest: scalar. optional. default: 1.
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
                                               c_dim_t(dim), c_dim_t(num_nearest)))
    return index, dist

def nearest_neighbour(query, database, dim = 0, num_nearest = 1, match_type=MATCH.SSD):
    """
    Nearest Neighbour matcher.

    Parameters
    -----------

    query    : af.Array
             A query feature descriptor

    database : af.Array
             A multi dimensional array containing the feature descriptor database.

    dim      : scalar. optional. default: 0.
             Specifies the dimension along which feature descriptor lies.

    num_nearest: scalar. optional. default: 1.
             Specifies the number of nearest neighbors to find.

    match_type: optional: af.MATCH. default: af.MATCH.SSD
             Specifies the match function metric.

    Returns
    ---------

    (location, distance): tuple of af.Array
                          location and distances of closest matches.

    """
    index = Array()
    dist = Array()
    safe_call(backend.get().af_nearest_neighbour(ct.pointer(idx.arr), ct.pointer(dist.arr),
                                                 query.arr, database.arr,
                                                 c_dim_t(dim), c_dim_t(num_nearest),
                                                 match_type.value))
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
    safe_call(backend.get().af_match_template(ct.pointer(out.arr),
                                              image.arr, template.arr,
                                              match_type.value))
    return out

def susan(image, radius=3, diff_thr=32, geom_thr=10, feature_ratio=0.05, edge=3):
    """
    SUSAN corner detector.

    Parameters
    ----------
    image         : af.Array
                  A 2D array specifying an image.

    radius        : scalar. optional. default: 500.
                  Specifies the radius of each pixel neighborhood.

    diff_thr      : scalar. optional. default: 1E5
                  Specifies the intensity difference threshold.

    geom_thr      : scalar. optional. default: 1.0
                  Specifies the geometric threshold.

    feature_ratio : scalar. optional. default: 0.05 (5%)
                  Specifies the ratio of corners found to number of pixels.

    edge         : scalar. optional. default: 3
                  Specifies the number of edge rows and columns that are ignored.

    Returns
    ---------

    features     : af.Features()
                 Contains the location and score. Orientation and size are not computed.

    """
    out = Features()
    safe_call(backend.get().af_susan(ct.pointer(out.feat),
                                     image.arr, ct.c_uint(radius), ct.c_float(diff_thr),
                                     ct.c_float(geom_thr), ct.c_float(feature_ratio),
                                     ct.c_uint(edge)))
    return out

def dog(image, radius1, radius2):
    """
    Difference of gaussians.

    Parameters
    ----------
    image    : af.Array
             A 2D array specifying an image.

    radius1  : scalar.
             The radius of first gaussian kernel.

    radius2  : scalar.
             The radius of second gaussian kernel.


    Returns
    --------

    out      : af.Array
             A multi dimensional array containing the difference of gaussians.

    Note
    ------

    The sigma values are calculated to be 0.25 * radius.
    """

    out = Array()
    safe_call(backend.get().af_dog(ct.pointer(out.arr),
                                   image.arr, radius1, radius2))
    return out

def sift(image, num_layers=3, contrast_threshold=0.04, edge_threshold=10.0, initial_sigma = 1.6,
         double_input = True, intensity_scale = 0.00390625, feature_ratio = 0.05):
    """
    SIFT feature detector and descriptor.

    Parameters
    ----------
    image              : af.Array
                       A 2D array representing an image

    num_layers         : optional: integer. Default: 3
                       Number of layers per octave. The number of octaves is calculated internally.

    contrast_threshold : optional: float. Default: 0.04
                       Threshold used to filter out features that have low contrast.

    edge_threshold     : optional: float. Default: 10.0
                       Threshold used to filter out features that are too edge-like.

    initial_sigma      : optional: float. Default: 1.6
                       The sigma value used to filter the input image at the first octave.

    double_input       : optional: bool. Default: True
                       If True, the input image will be scaled to double the size for the first octave.

    intensity_scale    : optional: float. Default: 1.0/255
                       The inverse of the difference between maximum and minimum intensity values.

    feature_ratio      : optional: float. Default: 0.05
                       Specifies the maximum number of features to detect as a ratio of image pixels.

    Returns
    --------
    (features, descriptor)     : tuple of (af.Features(), af.Array)
                               - descriptor is an af.Array of size N x 128

    """

    feat = Features()
    desc = Array()
    safe_call(af_sift(ct.pointer(feat), ct.pointer(desc),
                      image.arr, num_layers, contrast_threshold, edge_threshold,
                      initial_sigma, double_input, intensity_scale, feature_ratio))

    return (feat, desc)

def gloh(image, num_layers=3, contrast_threshold=0.04, edge_threshold=10.0, initial_sigma = 1.6,
         double_input = True, intensity_scale = 0.00390625, feature_ratio = 0.05):
    """
    GLOH feature detector and descriptor.

    Parameters
    ----------
    image              : af.Array
                       A 2D array representing an image

    num_layers         : optional: integer. Default: 3
                       Number of layers per octave. The number of octaves is calculated internally.

    contrast_threshold : optional: float. Default: 0.04
                       Threshold used to filter out features that have low contrast.

    edge_threshold     : optional: float. Default: 10.0
                       Threshold used to filter out features that are too edge-like.

    initial_sigma      : optional: float. Default: 1.6
                       The sigma value used to filter the input image at the first octave.

    double_input       : optional: bool. Default: True
                       If True, the input image will be scaled to double the size for the first octave.

    intensity_scale    : optional: float. Default: 1.0/255
                       The inverse of the difference between maximum and minimum intensity values.

    feature_ratio      : optional: float. Default: 0.05
                       Specifies the maximum number of features to detect as a ratio of image pixels.

    Returns
    --------
    (features, descriptor)     : tuple of (af.Features(), af.Array)
                               - descriptor is an af.Array of size N x 272

    """

    feat = Features()
    desc = Array()
    safe_call(af_gloh(ct.pointer(feat), ct.pointer(desc),
                      image.arr, num_layers, contrast_threshold, edge_threshold,
                      initial_sigma, double_input, intensity_scale, feature_ratio))

    return (feat, desc)

def homography(x_src, y_src, x_dst, y_dst, htype = HOMOGRAPHY.RANSAC,
               ransac_threshold = 3.0, iters = 1000, out_type = Dtype.f32):
    """
    Homography estimation

    Parameters
    ----------
    x_src            :  af.Array
                     A list of x co-ordinates of the source points.

    y_src            :  af.Array
                     A list of y co-ordinates of the source points.

    x_dst            :  af.Array
                     A list of x co-ordinates of the destination points.

    y_dst            :  af.Array
                     A list of y co-ordinates of the destination points.

    htype            : optional: af.HOMOGRAPHY. Default: HOMOGRAPHY.RANSAC
                     htype can be one of
                         - HOMOGRAPHY.RANSAC: RANdom SAmple Consensus will be used to evaluate quality.
                         - HOMOGRAPHY.LMEDS : Least MEDian of Squares is used to evaluate quality.

    ransac_threshold : optional: scalar. Default: 3.0
                     If `htype` is HOMOGRAPHY.RANSAC, it specifies the L2-distance threshold for inliers.

    out_type         : optional. af.Dtype. Default: Dtype.f32.
                     Specifies the output data type.

    Returns
    -------
    (H, inliers)     : A tuple of (af.Array, integer)
    """

    H = Array()
    inliers = ct.c_int(0)
    safe_call(backend.get().af_homography(ct.pointer(H), ct.pointer(inliers),
                                          x_src.arr, y_src.arr, x_dst.arr, y_dst.arr,
                                          htype.value, ransac_threshold, iters, out_type.value))
    return (H, inliers)
