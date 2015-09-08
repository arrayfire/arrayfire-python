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
    from enum import Enum

    def Enum_value(val):
        return val.value

except:
    class Enum(object):
        pass

    def Enum_value(val):
        return val

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
del _clibrary


class ERR(Enum):
    """
    Error values. For internal use only.
    """

    NONE            =   (0)

    #100-199 Errors in environment
    NO_MEM         = (101)
    DRIVER         = (102)
    RUNTIME        = (103)

    # 200-299 Errors in input parameters
    INVALID_ARRAY  = (201)
    ARG            = (202)
    SIZE           = (203)
    TYPE           = (204)
    DIFF_TYPE      = (205)
    BATCH          = (207)

    # 300-399 Errors for missing software features
    NOT_SUPPORTED  = (301)
    NOT_CONFIGURED = (302)

    # 400-499 Errors for missing hardware features
    NO_DBL         = (401)
    NO_GFX         = (402)

    # 900-999 Errors from upstream libraries and runtimes
    INTERNAL       = (998)
    UNKNOWN        = (999)

class Dtype(Enum):
    """
    Error values. For internal use only.
    """
    f32 = (0)
    c32 = (1)
    f64 = (2)
    c64 = (3)
    b8  = (4)
    s32 = (5)
    u32 = (6)
    u8  = (7)
    s64 = (8)
    u64 = (9)

class Source(Enum):
    """
    Source of the pointer
    """
    device = (0)
    host   = (1)

class INTERP(Enum):
    """
    Interpolation method
    """
    NEAREST   = (0)
    LINEAR    = (1)
    BILINEAR  = (2)
    CUBIC     = (3)

class PAD(Enum):
    """
    Edge padding types
    """
    ZERO = (0)
    SYM  = (1)

class CONNECTIVITY(Enum):
    """
    Neighborhood connectivity
    """
    FOUR  = (4)
    EIGHT = (8)

class CONV_MODE(Enum):
    """
    Convolution mode
    """
    DEFAULT = (0)
    EXPAND  = (1)

class CONV_DOMAIN(Enum):
    """
    Convolution domain
    """
    AUTO    = (0)
    SPATIAL = (1)
    FREQ    = (2)

class MATCH(Enum):
    """
    Match type
    """

    """
    Sum of absolute differences
    """
    SAD  = (0)

    """
    Zero mean SAD
    """
    ZSAD = (1)

    """
    Locally scaled SAD
    """
    LSAD = (2)

    """
    Sum of squared differences
    """
    SSD  = (3)

    """
    Zero mean SSD
    """
    ZSSD = (4)

    """
    Locally scaled SSD
    """
    LSSD = (5)

    """
    Normalized cross correlation
    """
    NCC  = (6)

    """
    Zero mean NCC
    """
    ZNCC = (7)

    """
    Sum of hamming distances
    """
    SHD  = (8)

class CSPACE(Enum):
    """
    Colorspace formats
    """
    GRAY = (0)
    RGB  = (1)
    HSV  = (2)

class MATPROP(Enum):
    """
    Matrix properties
    """

    """
    None, general.
    """
    NONE       = (0)

    """
    Transposed.
    """
    TRANS      = (1)

    """
    Conjugate transposed.
    """
    CTRANS     = (2)

    """
    Upper triangular matrix.
    """
    UPPER      = (32)

    """
    Lower triangular matrix.
    """
    LOWER      = (64)

    """
    Treat diagonal as units.
    """
    DIAG_UNIT  = (128)

    """
    Symmetric matrix.
    """
    SYM        = (512)

    """
    Positive definite matrix.
    """
    POSDEF     = (1024)

    """
    Orthogonal matrix.
    """
    ORTHOG     = (2048)

    """
    Tri diagonal matrix.
    """
    TRI_DIAG   = (4096)

    """
    Block diagonal matrix.
    """
    BLOCK_DIAG = (8192)

class NORM(Enum):
    """
    Norm types
    """
    VECTOR_1    = (0)
    VECTOR_INF  = (1)
    VECTOR_2    = (2)
    VECTOR_P    = (3)
    MATRIX_1    = (4)
    MATRIX_INF  = (5)
    MATRIX_2    = (6)
    MATRIX_L_PQ = (7)
    EUCLID      = VECTOR_2

class COLORMAP(Enum):
    """
    Colormaps
    """
    DEFAULT  = (0)
    SPECTRUM = (1)
    COLORS   = (2)
    RED      = (3)
    MOOD     = (4)
    HEAT     = (5)
    BLUE     = (6)

del Enum
