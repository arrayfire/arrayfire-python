#######################################################
# Copyright (c) 2014, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import platform
from ctypes import *

def load_backend(name):
    platform_name = platform.system()

    libname = 'libaf' + name
    if platform_name == 'Linux':
        libname += '.so'
    elif platform_name == 'Darwin':
        libname += '.dylib'
    else:
        raise OSError(platform_name + ' not supported')

    cdll.LoadLibrary(libname)
    clib = CDLL(libname)
    print("Using %s backend" % name)
    return clib

try:
    clib = load_backend('cuda')
except:
    try:
        clib = load_backend('opencl')
    except:
        clib = load_backend('cpu')


AF_SUCCESS            =   c_int(0)

#100-199 Errors in environment
AF_ERR_NO_MEM         = c_int(101)
AF_ERR_DRIVER         = c_int(102)
AF_ERR_RUNTIME        = c_int(103)

# 200-299 Errors in input parameters
AF_ERR_INVALID_ARRAY  = c_int(201)
AF_ERR_ARG            = c_int(202)
AF_ERR_SIZE           = c_int(203)
AF_ERR_TYPE           = c_int(204)
AF_ERR_DIFF_TYPE      = c_int(205)
AF_ERR_BATCH          = c_int(207)

# 300-399 Errors for missing software features
AF_ERR_NOT_SUPPORTED  = c_int(301)
AF_ERR_NOT_CONFIGURED = c_int(302)

# 400-499 Errors for missing hardware features
AF_ERR_NO_DBL         = c_int(401)
AF_ERR_NO_GFX         = c_int(402)

# 900-999 Errors from upstream libraries and runtimes
AF_ERR_INTERNAL       = c_int(998)
AF_ERR_UNKNOWN        = c_int(999)

f32 = c_int(0)
c32 = c_int(1)
f64 = c_int(2)
c64 = c_int(3)
b8  = c_int(4)
s32 = c_int(5)
u32 = c_int(6)
u8  = c_int(7)
s64 = c_int(8)
u64 = c_int(9)

afDevice = c_int(0)
afHost   = c_int(1)

AF_INTERP_NEAREST   = c_int(0)
AF_INTERP_LINEAR    = c_int(1)
AF_INTERP_BILINEAR  = c_int(2)
AF_INTERP_CUBIC     = c_int(3)

AF_PAD_ZERO = c_int(0)
AF_PAD_SYM  = c_int(1)

AF_CONNECTIVITY_4 = c_int(4)
AF_CONNECTIVITY_8 = c_int(8)

AF_CONV_DEFAULT = c_int(0)
AF_CONV_EXPAND  = c_int(1)

AF_CONV_AUTO    = c_int(0)
AF_CONV_SPATIAL = c_int(1)
AF_CONV_FREQ    = c_int(2)

AF_SAD  = c_int(0)
AF_ZSAD = c_int(1)
AF_LSAD = c_int(2)
AF_SSD  = c_int(3)
AF_ZSSD = c_int(4)
AF_LSSD = c_int(5)
AF_NCC  = c_int(6)
AF_ZNCC = c_int(7)
AF_SHD  = c_int(8)

AF_GRAY = c_int(0)
AF_RGB  = c_int(1)
AF_HSV  = c_int(2)

AF_MAT_NONE       = c_int(0)
AF_MAT_TRANS      = c_int(1)
AF_MAT_CTRANS     = c_int(2)
AF_MAT_UPPER      = c_int(32)
AF_MAT_LOWER      = c_int(64)
AF_MAT_DIAG_UNIT  = c_int(128)
AF_MAT_SYM        = c_int(512)
AF_MAT_POSDEF     = c_int(1024)
AF_MAT_ORTHOG     = c_int(2048)
AF_MAT_TRI_DIAG   = c_int(4096)
AF_MAT_BLOCK_DIAG = c_int(8192)

AF_NORM_VECTOR_1    = c_int(0)
AF_NORM_VECTOR_INF  = c_int(1)
AF_NORM_VECTOR_2    = c_int(2)
AF_NORM_VECTOR_P    = c_int(3)
AF_NORM_MATRIX_1    = c_int(4)
AF_NORM_MATRIX_INF  = c_int(5)
AF_NORM_MATRIX_2    = c_int(6)
AF_NORM_MATRIX_L_PQ = c_int(7)
AF_NORM_EUCLID      = AF_NORM_VECTOR_2

AF_COLORMAP_DEFAULT  = c_int(0)
AF_COLORMAP_SPECTRUM = c_int(1)
AF_COLORMAP_COLORS   = c_int(2)
AF_COLORMAP_RED      = c_int(3)
AF_COLORMAP_MOOD     = c_int(4)
AF_COLORMAP_HEAT     = c_int(5)
AF_COLORMAP_BLUE     = c_int(6)
