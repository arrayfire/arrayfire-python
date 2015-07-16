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
from .data import constant
import os

def gradient(image):
    dx = array()
    dy = array()
    safe_call(clib.af_gradient(ct.pointer(dx.arr), ct.pointer(dy.arr), image.arr))
    return dx, dy

def load_image(file_name, is_color=False):
    assert(os.path.isfile(file_name))
    image = array()
    safe_call(clib.af_load_image(ct.pointer(image.arr), ct.c_char_p(file_name.encode('ascii')), is_color))
    return image

def save_image(image, file_name):
    assert(isinstance(file_name, str))
    safe_call(clib.af_save_image(ct.c_char_p(file_name.encode('ascii')), image.arr))
    return image

def resize(image, scale=None, odim0=None, odim1=None, method=AF_INTERP_NEAREST):

    if (scale is None):
        assert(odim0 is not None)
        assert(odim1 is not None)
    else:
        idims = image.dims()
        odim0 = int(scale * idims[0])
        odim1 = int(scale * idims[1])

    output = array()
    safe_call(clib.af_resize(ct.pointer(output.arr),\
                             image.arr, ct.c_longlong(odim0), ct.c_longlong(odim1), method))

    return output

def transform(image, transform, odim0 = 0, odim1 = 0, method=AF_INTERP_NEAREST, is_inverse=True):
    output = array()
    safe_call(clib.af_transform(ct.pointer(output.arr),\
                                image.arr, transform.arr,\
                                ct.c_longlong(odim0), ct.c_longlong(odim1), method, is_inverse))
    return output

def rotate(image, theta, is_crop = True, method = AF_INTERP_NEAREST):
    output = array()
    safe_call(clib.af_rotate(ct.pointer(output.arr), image.arr, ct.c_double(theta), is_crop, method))
    return output

def translate(image, trans0, trans1, odim0 = 0, odim1 = 0, method = AF_INTERP_NEAREST):
    output = array()
    safe_call(clib.af_translate(ct.pointer(output.arr), \
                                image.arr, trans0, trans1, ct.c_longlong(odim0), ct.c_longlong(odim1), method))
    return output

def scale(image, scale0, scale1, odim0 = 0, odim1 = 0, method = AF_INTERP_NEAREST):
    output = array()
    safe_call(clib.af_scale(ct.pointer(output.arr),\
                            image.arr, ct.c_double(scale0), ct.c_double(scale1),\
                            ct.c_longlong(odim0), ct.c_longlong(odim1), method))
    return output

def skew(image, skew0, skew1, odim0 = 0, odim1 = 0, method = AF_INTERP_NEAREST, is_inverse=True):
    output = array()
    safe_call(clib.af_skew(ct.pointer(output.arr),\
                           image.arr, ct.c_double(skew0), ct.c_double(skew1), \
                           ct.c_longlong(odim0), ct.c_longlong(odim1), method, is_inverse))

    return output

def histogram(image, nbins, min_val = None, max_val = None):
    from .algorithm import min as af_min
    from .algorithm import max as af_max

    if min_val is None:
        min_val = af_min(image)

    if max_val is None:
        max_val = af_max(image)

    output = array()
    safe_call(clib.af_histogram(ct.pointer(output.arr),\
                                image.arr, ct.c_uint(nbins), ct.c_double(min_val), ct.c_double(max_val)))
    return output

def hist_equal(image, hist):
    output = array()
    safe_call(clib.af_hist_equal(ct.pointer(output.arr), image.arr, hist.arr))
    return output

def dilate(image, mask = None):

    if mask is None:
        mask = constant(1, 3, 3, dtype=f32)

    output = array()
    safe_call(clib.af_dilate(ct.pointer(output.arr), image.arr, mask.arr))

    return output

def dilate3(image, mask = None):

    if mask is None:
        mask = constant(1, 3, 3, 3, dtype=f32)

    output = array()
    safe_call(clib.af_dilate3(ct.pointer(output.arr), image.arr, mask.arr))

    return output

def erode(image, mask = None):

    if mask is None:
        mask = constant(1, 3, 3, dtype=f32)

    output = array()
    safe_call(clib.af_erode(ct.pointer(output.arr), image.arr, mask.arr))

    return output

def erode3(image, mask = None):

    if mask is None:
        mask = constant(1, 3, 3, 3, dtype=f32)

    output = array()
    safe_call(clib.af_erode3(ct.pointer(output.arr), image.arr, mask.arr))

    return output

def bilateral(image, s_sigma, c_sigma, is_color = False):
    output = array()
    safe_call(clib.af_bilateral(ct.pointer(output.arr),\
                                image.arr, ct.c_double(s_sigma), ct.c_double(c_sigma), is_color))
    return output

def mean_shift(image, s_sigma, c_sigma, n_iter, is_color = False):
    output = array()
    safe_call(clib.af_mean_shift(ct.pointer(output.arr),\
                                 image.arr, ct.c_double(s_sigma), ct.c_double(c_sigma),\
                                 ct.c_uint(n_iter), is_color))
    return output

def medfilt(image, w_len = 3, w_wid = 3, edge_pad = AF_PAD_ZERO):
    output = array()
    safe_call(clib.af_medfilt(ct.pointer(output.arr), \
                              image.arr, ct.c_longlong(w_len), ct.c_longlong(w_wid), edge_pad))
    return output

def minfilt(image, w_len = 3, w_wid = 3, edge_pad = AF_PAD_ZERO):
    output = array()
    safe_call(clib.af_minfilt(ct.pointer(output.arr), \
                              image.arr, ct.c_longlong(w_len), ct.c_longlong(w_wid), edge_pad))
    return output

def maxfilt(image, w_len = 3, w_wid = 3, edge_pad = AF_PAD_ZERO):
    output = array()
    safe_call(clib.af_maxfilt(ct.pointer(output.arr), \
                              image.arr, ct.c_longlong(w_len), ct.c_longlong(w_wid), edge_pad))
    return output

def regions(image, connectivity = AF_CONNECTIVITY_4, out_type = f32):
    output = array()
    safe_call(clib.af_regions(ct.pointer(output.arr), image.arr, connectivity, out_type))
    return output

def sobel_derivatives(image, w_len=3):
    dx = array()
    dy = array()
    safe_call(clib.af_sobel_operator(ct.pointer(dx.arr), ct.pointer(dy.arr),\
                                     image.arr, ct.c_uint(w_len)))
    return dx,dy

def sobel_filter(image, w_len = 3, is_fast = False):
    from .arith import abs as af_abs
    from .arith import hypot as af_hypot

    dx,dy = sobel_derivatives(image, w_len)
    if (is_fast):
        return af_abs(dx) + af_abs(dy)
    else:
        return af_hypot(dx, dy)

def rgb2gray(image, r_factor = 0.2126, g_factor = 0.7152, b_factor = 0.0722):
    output=array()
    safe_call(clib.af_rgb2gray(ct.pointer(output.arr), \
                               image.arr, ct.c_float(r_factor), ct.c_float(g_factor), ct.c_float(b_factor)))
    return output

def gray2rgb(image, r_factor = 1.0, g_factor = 1.0, b_factor = 1.0):
    output=array()
    safe_call(clib.af_gray2rgb(ct.pointer(output.arr), \
                               image.arr, ct.c_float(r_factor), ct.c_float(g_factor), ct.c_float(b_factor)))
    return output

def hsv2rgb(image):
    output = array()
    safe_call(clib.af_hsv2rgb(ct.pointer(output.arr), image.arr))
    return output

def rgb2hsv(image):
    output = array()
    safe_call(clib.af_rgb2hsv(ct.pointer(output.arr), image.arr))
    return output

def color_space(image, to_type, from_type):
    output = array()
    safe_call(clib.af_color_space(ct.pointer(output.arr), image.arr, to_type, from_type))
    return output
