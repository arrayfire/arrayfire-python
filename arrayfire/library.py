#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
module containing enums and other constants from arrayfire library
"""

import platform
import ctypes as ct

try:
    from enum import Enum as _Enum
    def _Enum_Type(v):
        return v
except:
    class _Enum(object):
        pass

    class _Enum_Type(object):
        def __init__(self, v):
            self.value = v

class _clibrary(object):

    def __libname(self, name):
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

        return libname

    def set(self, name, unsafe=False):
        if (not unsafe and self.__lock):
            raise RuntimeError("Can not change backend after creating an Array")
        if (self.clibs[name] is None):
            raise RuntimeError("Could not load any ArrayFire %s backend" % name)
        self.name = name
        return

    def __init__(self):
        self.clibs = {}
        self.name = None
        self.__lock = False
        # Iterate in reverse order of preference
        for name in ('cpu', 'opencl', 'cuda'):
            try:
                libname = self.__libname(name)
                ct.cdll.LoadLibrary(libname)
                self.clibs[name] = ct.CDLL(libname)
                self.name = name
            except:
                self.clibs[name] = None

        if (self.name is None):
            raise RuntimeError("Could not load any ArrayFire libraries")

    def get(self):
        return self.clibs[self.name]

    def lock(self):
        self.__lock = True

backend = _clibrary()

class ERR(_Enum):
    """
    Error values. For internal use only.
    """

    NONE            =   _Enum_Type(0)

    #100-199 Errors in environment
    NO_MEM         = _Enum_Type(101)
    DRIVER         = _Enum_Type(102)
    RUNTIME        = _Enum_Type(103)

    # 200-299 Errors in input parameters
    INVALID_ARRAY  = _Enum_Type(201)
    ARG            = _Enum_Type(202)
    SIZE           = _Enum_Type(203)
    TYPE           = _Enum_Type(204)
    DIFF_TYPE      = _Enum_Type(205)
    BATCH          = _Enum_Type(207)

    # 300-399 Errors for missing software features
    NOT_SUPPORTED  = _Enum_Type(301)
    NOT_CONFIGURED = _Enum_Type(302)

    # 400-499 Errors for missing hardware features
    NO_DBL         = _Enum_Type(401)
    NO_GFX         = _Enum_Type(402)

    # 900-999 Errors from upstream libraries and runtimes
    INTERNAL       = _Enum_Type(998)
    UNKNOWN        = _Enum_Type(999)

class Dtype(_Enum):
    """
    Error values. For internal use only.
    """
    f32 = _Enum_Type(0)
    c32 = _Enum_Type(1)
    f64 = _Enum_Type(2)
    c64 = _Enum_Type(3)
    b8  = _Enum_Type(4)
    s32 = _Enum_Type(5)
    u32 = _Enum_Type(6)
    u8  = _Enum_Type(7)
    s64 = _Enum_Type(8)
    u64 = _Enum_Type(9)

class Source(_Enum):
    """
    Source of the pointer
    """
    device = _Enum_Type(0)
    host   = _Enum_Type(1)

class INTERP(_Enum):
    """
    Interpolation method
    """
    NEAREST   = _Enum_Type(0)
    LINEAR    = _Enum_Type(1)
    BILINEAR  = _Enum_Type(2)
    CUBIC     = _Enum_Type(3)

class PAD(_Enum):
    """
    Edge padding types
    """
    ZERO = _Enum_Type(0)
    SYM  = _Enum_Type(1)

class CONNECTIVITY(_Enum):
    """
    Neighborhood connectivity
    """
    FOUR  = _Enum_Type(4)
    EIGHT = _Enum_Type(8)

class CONV_MODE(_Enum):
    """
    Convolution mode
    """
    DEFAULT = _Enum_Type(0)
    EXPAND  = _Enum_Type(1)

class CONV_DOMAIN(_Enum):
    """
    Convolution domain
    """
    AUTO    = _Enum_Type(0)
    SPATIAL = _Enum_Type(1)
    FREQ    = _Enum_Type(2)

class MATCH(_Enum):
    """
    Match type
    """

    """
    Sum of absolute differences
    """
    SAD  = _Enum_Type(0)

    """
    Zero mean SAD
    """
    ZSAD = _Enum_Type(1)

    """
    Locally scaled SAD
    """
    LSAD = _Enum_Type(2)

    """
    Sum of squared differences
    """
    SSD  = _Enum_Type(3)

    """
    Zero mean SSD
    """
    ZSSD = _Enum_Type(4)

    """
    Locally scaled SSD
    """
    LSSD = _Enum_Type(5)

    """
    Normalized cross correlation
    """
    NCC  = _Enum_Type(6)

    """
    Zero mean NCC
    """
    ZNCC = _Enum_Type(7)

    """
    Sum of hamming distances
    """
    SHD  = _Enum_Type(8)

class CSPACE(_Enum):
    """
    Colorspace formats
    """
    GRAY = _Enum_Type(0)
    RGB  = _Enum_Type(1)
    HSV  = _Enum_Type(2)

class MATPROP(_Enum):
    """
    Matrix properties
    """

    """
    None, general.
    """
    NONE       = _Enum_Type(0)

    """
    Transposed.
    """
    TRANS      = _Enum_Type(1)

    """
    Conjugate transposed.
    """
    CTRANS     = _Enum_Type(2)

    """
    Upper triangular matrix.
    """
    UPPER      = _Enum_Type(32)

    """
    Lower triangular matrix.
    """
    LOWER      = _Enum_Type(64)

    """
    Treat diagonal as units.
    """
    DIAG_UNIT  = _Enum_Type(128)

    """
    Symmetric matrix.
    """
    SYM        = _Enum_Type(512)

    """
    Positive definite matrix.
    """
    POSDEF     = _Enum_Type(1024)

    """
    Orthogonal matrix.
    """
    ORTHOG     = _Enum_Type(2048)

    """
    Tri diagonal matrix.
    """
    TRI_DIAG   = _Enum_Type(4096)

    """
    Block diagonal matrix.
    """
    BLOCK_DIAG = _Enum_Type(8192)

class NORM(_Enum):
    """
    Norm types
    """
    VECTOR_1    = _Enum_Type(0)
    VECTOR_INF  = _Enum_Type(1)
    VECTOR_2    = _Enum_Type(2)
    VECTOR_P    = _Enum_Type(3)
    MATRIX_1    = _Enum_Type(4)
    MATRIX_INF  = _Enum_Type(5)
    MATRIX_2    = _Enum_Type(6)
    MATRIX_L_PQ = _Enum_Type(7)
    EUCLID      = VECTOR_2

class COLORMAP(_Enum):
    """
    Colormaps
    """
    DEFAULT  = _Enum_Type(0)
    SPECTRUM = _Enum_Type(1)
    COLORS   = _Enum_Type(2)
    RED      = _Enum_Type(3)
    MOOD     = _Enum_Type(4)
    HEAT     = _Enum_Type(5)
    BLUE     = _Enum_Type(6)
