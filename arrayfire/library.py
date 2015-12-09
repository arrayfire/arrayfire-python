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
    NONFREE        = _Enum_Type(303)

    # 400-499 Errors for missing hardware features
    NO_DBL         = _Enum_Type(401)
    NO_GFX         = _Enum_Type(402)

    # 500-599 Errors specific to the heterogeneous API
    LOAD_LIB       = _Enum_Type(501)
    LOAD_SYM       = _Enum_Type(502)
    ARR_BKND_MISMATCH = _Enum_Type(503)

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
    s16 = _Enum_Type(10)
    u16 = _Enum_Type(11)

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
    LOWER     = _Enum_Type(4)

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


class YCC_STD(_Enum):
    """
    YCC Standard formats
    """
    BT_601   = _Enum_Type(601)
    BT_709   = _Enum_Type(709)
    BT_2020  = _Enum_Type(2020)

class CSPACE(_Enum):
    """
    Colorspace formats
    """
    GRAY = _Enum_Type(0)
    RGB  = _Enum_Type(1)
    HSV  = _Enum_Type(2)
    YCbCr= _Enum_Type(3)

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

class IMAGE_FORMAT(_Enum):
    """
    Image Formats
    """
    BMP      = _Enum_Type(0)
    ICO      = _Enum_Type(1)
    JPEG     = _Enum_Type(2)
    JNG      = _Enum_Type(3)
    PNG      = _Enum_Type(13)
    PPM      = _Enum_Type(14)
    PPMRAW   = _Enum_Type(15)
    TIFF     = _Enum_Type(18)
    PSD      = _Enum_Type(20)
    HDR      = _Enum_Type(26)
    EXR      = _Enum_Type(29)
    JP2      = _Enum_Type(31)
    RAW      = _Enum_Type(34)

class HOMOGRAPHY(_Enum):
    """
    Homography Types
    """
    RANSAC   = _Enum_Type(0)
    LMEDS    = _Enum_Type(1)

class BACKEND(_Enum):
    """
    Backend libraries
    """
    DEFAULT = _Enum_Type(0)
    CPU     = _Enum_Type(1)
    CUDA    = _Enum_Type(2)
    OPENCL  = _Enum_Type(4)

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

    def set_unsafe(self, name):
        lib = self.__clibs[name]
        if (lib is None):
            raise RuntimeError("Backend not found")
        self.__name = name

    def __init__(self):
        self.__name = None

        self.__clibs = {'cuda'   : None,
                        'opencl' : None,
                        'cpu'    : None,
                        ''       : None}

        self.__backend_map = {0 : 'default',
                              1 : 'cpu'    ,
                              2 : 'cuda'   ,
                              4 : 'opencl' }

        self.__backend_name_map = {'default' : 0,
                                   'cpu'     : 1,
                                   'cuda'    : 2,
                                   'opencl'  : 4}

        # Iterate in reverse order of preference
        for name in ('cpu', 'opencl', 'cuda', ''):
            try:
                libname = self.__libname(name)
                ct.cdll.LoadLibrary(libname)
                self.__clibs[name] = ct.CDLL(libname)
                self.__name = name
            except:
                pass

        if (self.__name is None):
            raise RuntimeError("Could not load any ArrayFire libraries")

    def get_id(self, name):
        return self.__backend_name_map[name]

    def get_name(self, bk_id):
        return self.__backend_map[bk_id]

    def get(self):
        return self.__clibs[self.__name]

    def name(self):
        return self.__name

    def is_unified(self):
        return self.__name == ''

    def parse(self, res):
        lst = []
        for key,value in self.__backend_name_map.items():
            if (value & res):
                lst.append(key)
        return tuple(lst)

backend = _clibrary()

def set_backend(name, unsafe=False):
    """
    Set a specific backend by name

    Parameters
    ----------

    name : str.

    unsafe : optional: bool. Default: False.
           If False, does not switch backend if current backend is not unified backend.
    """
    if (backend.is_unified() == False and unsanfe == False):
        raise RuntimeError("Can not change backend after loading %s" % name)

    if (backend.is_unified()):
        safe_call(backend.get().af_set_backend(backend.get_id(name)))
    else:
        backend.set_unsafe(name)
    return

def get_backend_id(A):
    """
    Get backend name of an array

    Parameters
    ----------
    A    : af.Array

    Returns
    ----------

    name : str.
         Backend name
    """
    if (backend.is_unified()):
        backend_id = ct.c_int(BACKEND.DEFAULT.value)
        safe_call(backend.get().af_get_backend_id(ct.pointer(backend_id), A.arr))
        return backend.get_name(backend_id.value)
    else:
        return backend.name()

def get_backend_count():
    """
    Get number of available backends

    Returns
    ----------

    count : int
          Number of available backends
    """
    if (backend.is_unified()):
        count = ct.c_int(0)
        safe_call(backend.get().af_get_backend_count(ct.pointer(count)))
        return count.value
    else:
        return 1

def get_available_backends():
    """
    Get names of available backends

    Returns
    ----------

    names : tuple of strings
          Names of available backends
    """
    if (backend.is_unified()):
        available = ct.c_int(0)
        safe_call(backend.get().af_get_available_backends(ct.pointer(available)))
        return backend.parse(int(available.value))
    else:
        return (backend.name(),)

from .util import safe_call
