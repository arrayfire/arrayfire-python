#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import platform
import ctypes as ct

def load_backend(name):
    platform_name = platform.system()
    assert(len(platform_name) >= 3)

    libname = 'libaf' + name
    if platform_name == 'Linux':
        libname += '.so'
    elif platform_name == 'Darwin':
        libname += '.dylib'
    elif platform_name == "Windows" or platform_name[:3] == "CYG":
        libname += '.dll'
        libname = libname[3:] # remove 'lib'
        if platform_name == "Windows":
            '''
            Supressing crashes caused by missing dlls
            http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            https://msdn.microsoft.com/en-us/library/windows/desktop/ms680621.aspx
            '''
            ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002);
    else:
        raise OSError(platform_name + ' not supported')

    ct.cdll.LoadLibrary(libname)
    clib = ct.CDLL(libname)
    return clib, name

try:
    clib, backend = load_backend('cuda')
except:
    try:
        clib, backend = load_backend('opencl')
    except:
        clib, backend = load_backend('cpu')


AF_SUCCESS            =   ct.c_int(0)

#100-199 Errors in environment
AF_ERR_NO_MEM         = ct.c_int(101)
AF_ERR_DRIVER         = ct.c_int(102)
AF_ERR_RUNTIME        = ct.c_int(103)

# 200-299 Errors in input parameters
AF_ERR_INVALID_ARRAY  = ct.c_int(201)
AF_ERR_ARG            = ct.c_int(202)
AF_ERR_SIZE           = ct.c_int(203)
AF_ERR_TYPE           = ct.c_int(204)
AF_ERR_DIFF_TYPE      = ct.c_int(205)
AF_ERR_BATCH          = ct.c_int(207)

# 300-399 Errors for missing software features
AF_ERR_NOT_SUPPORTED  = ct.c_int(301)
AF_ERR_NOT_CONFIGURED = ct.c_int(302)

# 400-499 Errors for missing hardware features
AF_ERR_NO_DBL         = ct.c_int(401)
AF_ERR_NO_GFX         = ct.c_int(402)

# 900-999 Errors from upstream libraries and runtimes
AF_ERR_INTERNAL       = ct.c_int(998)
AF_ERR_UNKNOWN        = ct.c_int(999)

f32 = ct.c_int(0)
c32 = ct.c_int(1)
f64 = ct.c_int(2)
c64 = ct.c_int(3)
b8  = ct.c_int(4)
s32 = ct.c_int(5)
u32 = ct.c_int(6)
u8  = ct.c_int(7)
s64 = ct.c_int(8)
u64 = ct.c_int(9)

afDevice = ct.c_int(0)
afHost   = ct.c_int(1)

AF_INTERP_NEAREST   = ct.c_int(0)
AF_INTERP_LINEAR    = ct.c_int(1)
AF_INTERP_BILINEAR  = ct.c_int(2)
AF_INTERP_CUBIC     = ct.c_int(3)

AF_PAD_ZERO = ct.c_int(0)
AF_PAD_SYM  = ct.c_int(1)

AF_CONNECTIVITY_4 = ct.c_int(4)
AF_CONNECTIVITY_8 = ct.c_int(8)

AF_CONV_DEFAULT = ct.c_int(0)
AF_CONV_EXPAND  = ct.c_int(1)

AF_CONV_AUTO    = ct.c_int(0)
AF_CONV_SPATIAL = ct.c_int(1)
AF_CONV_FREQ    = ct.c_int(2)

AF_SAD  = ct.c_int(0)
AF_ZSAD = ct.c_int(1)
AF_LSAD = ct.c_int(2)
AF_SSD  = ct.c_int(3)
AF_ZSSD = ct.c_int(4)
AF_LSSD = ct.c_int(5)
AF_NCC  = ct.c_int(6)
AF_ZNCC = ct.c_int(7)
AF_SHD  = ct.c_int(8)

AF_GRAY = ct.c_int(0)
AF_RGB  = ct.c_int(1)
AF_HSV  = ct.c_int(2)

AF_MAT_NONE       = ct.c_int(0)
AF_MAT_TRANS      = ct.c_int(1)
AF_MAT_CTRANS     = ct.c_int(2)
AF_MAT_UPPER      = ct.c_int(32)
AF_MAT_LOWER      = ct.c_int(64)
AF_MAT_DIAG_UNIT  = ct.c_int(128)
AF_MAT_SYM        = ct.c_int(512)
AF_MAT_POSDEF     = ct.c_int(1024)
AF_MAT_ORTHOG     = ct.c_int(2048)
AF_MAT_TRI_DIAG   = ct.c_int(4096)
AF_MAT_BLOCK_DIAG = ct.c_int(8192)

AF_NORM_VECTOR_1    = ct.c_int(0)
AF_NORM_VECTOR_INF  = ct.c_int(1)
AF_NORM_VECTOR_2    = ct.c_int(2)
AF_NORM_VECTOR_P    = ct.c_int(3)
AF_NORM_MATRIX_1    = ct.c_int(4)
AF_NORM_MATRIX_INF  = ct.c_int(5)
AF_NORM_MATRIX_2    = ct.c_int(6)
AF_NORM_MATRIX_L_PQ = ct.c_int(7)
AF_NORM_EUCLID      = AF_NORM_VECTOR_2

AF_COLORMAP_DEFAULT  = ct.c_int(0)
AF_COLORMAP_SPECTRUM = ct.c_int(1)
AF_COLORMAP_COLORS   = ct.c_int(2)
AF_COLORMAP_RED      = ct.c_int(3)
AF_COLORMAP_MOOD     = ct.c_int(4)
AF_COLORMAP_HEAT     = ct.c_int(5)
AF_COLORMAP_BLUE     = ct.c_int(6)
