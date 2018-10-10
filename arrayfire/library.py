#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Module containing enums and other constants.
"""

import platform
import ctypes as ct
import traceback
import os

c_float_t     = ct.c_float
c_double_t    = ct.c_double
c_int_t       = ct.c_int
c_uint_t      = ct.c_uint
c_longlong_t  = ct.c_longlong
c_ulonglong_t = ct.c_ulonglong
c_char_t      = ct.c_char
c_bool_t      = ct.c_bool
c_uchar_t     = ct.c_ubyte
c_short_t     = ct.c_short
c_ushort_t    = ct.c_ushort
c_pointer     = ct.pointer
c_void_ptr_t  = ct.c_void_p
c_char_ptr_t  = ct.c_char_p
c_size_t      = ct.c_size_t


AF_VER_MAJOR = '3'
FORGE_VER_MAJOR = '1'

# Work around for unexpected architectures
if 'c_dim_t_forced' in globals():
    global c_dim_t_forced
    c_dim_t = c_dim_t_forced
else:
    # dim_t is long long by default
    c_dim_t = c_longlong_t
    # Change to int for 32 bit x86 and amr architectures
    if (platform.architecture()[0][0:2] == '32' and
        (platform.machine()[-2:] == '86' or
         platform.machine()[0:3] == 'arm')):
        c_dim_t = c_int_t

try:
    from enum import Enum as _Enum
    def _Enum_Type(v):
        return v
except ImportError:
    class _MetaEnum(type):
        def __init__(cls, name, bases, attrs):
            for attrname, attrvalue in attrs.iteritems():
                if name != '_Enum' and isinstance(attrvalue, _Enum_Type):
                    attrvalue.__class__ = cls
                    attrs[attrname] = attrvalue

    class _Enum(object):
        __metaclass__ = _MetaEnum

    class _Enum_Type(object):
        def __init__(self, v):
            self.value = v

class ERR(_Enum):
    """
    Error values. For internal use only.
    """

    NONE            = _Enum_Type(0)

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
    DEVICE         = _Enum_Type(208)

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
    NEAREST         = _Enum_Type(0)
    LINEAR          = _Enum_Type(1)
    BILINEAR        = _Enum_Type(2)
    CUBIC           = _Enum_Type(3)
    LOWER           = _Enum_Type(4)
    LINEAR_COSINE   = _Enum_Type(5)
    BILINEAR_COSINE = _Enum_Type(6)
    BICUBIC         = _Enum_Type(7)
    CUBIC_SPLINE    = _Enum_Type(8)
    BICUBIC_SPLINE  = _Enum_Type(9)

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

class MARKER(_Enum):
    """
    Markers used for different points in graphics plots
    """
    NONE       = _Enum_Type(0)
    POINT      = _Enum_Type(1)
    CIRCLE     = _Enum_Type(2)
    SQUARE     = _Enum_Type(3)
    TRIANGE    = _Enum_Type(4)
    CROSS      = _Enum_Type(5)
    PLUS       = _Enum_Type(6)
    STAR       = _Enum_Type(7)

class MOMENT(_Enum):
    """
    Image Moments types
    """
    M00         = _Enum_Type(1)
    M01         = _Enum_Type(2)
    M10         = _Enum_Type(4)
    M11         = _Enum_Type(8)
    FIRST_ORDER = _Enum_Type(15)

class BINARYOP(_Enum):
    """
    Binary Operators
    """
    ADD  = _Enum_Type(0)
    MUL  = _Enum_Type(1)
    MIN  = _Enum_Type(2)
    MAX  = _Enum_Type(3)

class RANDOM_ENGINE(_Enum):
    """
    Random engine types
    """
    PHILOX_4X32_10   = _Enum_Type(100)
    THREEFRY_2X32_16 = _Enum_Type(200)
    MERSENNE_GP11213 = _Enum_Type(300)
    PHILOX           = PHILOX_4X32_10
    THREEFRY         = THREEFRY_2X32_16
    DEFAULT          = PHILOX

class STORAGE(_Enum):
    """
    Matrix Storage types
    """
    DENSE = _Enum_Type(0)
    CSR   = _Enum_Type(1)
    CSC   = _Enum_Type(2)
    COO   = _Enum_Type(3)

class CANNY_THRESHOLD(_Enum):
    """
    Canny Edge Threshold types
    """
    MANUAL = _Enum_Type(0)
    AUTO_OTSU = _Enum_Type(1)

class FLUX(_Enum):
    """
    Flux functions
    """
    DEFAULT     = _Enum_Type(0)
    QUADRATIC   = _Enum_Type(1)
    EXPONENTIAL = _Enum_Type(2)

class DIFFUSION(_Enum):
    """
    Diffusion equations
    """
    DEFAULT = _Enum_Type(0)
    GRAD    = _Enum_Type(1)
    MCDE    = _Enum_Type(2)

class TOPK(_Enum):
    """
    Top-K ordering
    """
    DEFAULT = _Enum_Type(0)
    MIN     = _Enum_Type(1)
    MAX     = _Enum_Type(2)

_VER_MAJOR_PLACEHOLDER = "__VER_MAJOR__"

def _setup():
    import platform

    platform_name = platform.system()

    try:
        AF_PATH = os.environ['AF_PATH']
    except KeyError:
        AF_PATH = None
        pass

    AF_SEARCH_PATH = AF_PATH

    try:
        CUDA_PATH = os.environ['CUDA_PATH']
    except KeyError:
        CUDA_PATH= None
        pass

    CUDA_FOUND = False

    assert(len(platform_name) >= 3)
    if platform_name == 'Windows' or platform_name[:3] == 'CYG':

        ## Windows specific setup
        pre = ''
        post = '.dll'
        if platform_name == "Windows":
            '''
            Supressing crashes caused by missing dlls
            http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            https://msdn.microsoft.com/en-us/library/windows/desktop/ms680621.aspx
            '''
            ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH="C:/Program Files/ArrayFire/v" + AF_VER_MAJOR +"/"

        if CUDA_PATH is not None:
            CUDA_FOUND = os.path.isdir(CUDA_PATH + '/bin') and os.path.isdir(CUDA_PATH + '/nvvm/bin/')

    elif platform_name == 'Darwin':

        ## OSX specific setup
        pre = 'lib'
        post = '.' + _VER_MAJOR_PLACEHOLDER + '.dylib'

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH='/usr/local/'

        if CUDA_PATH is None:
            CUDA_PATH='/usr/local/cuda/'

        CUDA_FOUND = os.path.isdir(CUDA_PATH + '/lib') and os.path.isdir(CUDA_PATH + '/nvvm/lib')

    elif platform_name == 'Linux':
        pre = 'lib'
        post = '.so.' + _VER_MAJOR_PLACEHOLDER

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH='/opt/arrayfire-' + AF_VER_MAJOR + '/'

        if CUDA_PATH is None:
            CUDA_PATH='/usr/local/cuda/'

        if platform.architecture()[0][:2] == '64':
            CUDA_FOUND = os.path.isdir(CUDA_PATH + '/lib64') and os.path.isdir(CUDA_PATH + '/nvvm/lib64')
        else:
            CUDA_FOUND = os.path.isdir(CUDA_PATH + '/lib') and os.path.isdir(CUDA_PATH + '/nvvm/lib')
    else:
        raise OSError(platform_name + ' not supported')

    if AF_PATH is None:
        os.environ['AF_PATH'] = AF_SEARCH_PATH

    return pre, post, AF_SEARCH_PATH, CUDA_FOUND

class _clibrary(object):

    def __libname(self, name, head='af', ver_major=AF_VER_MAJOR):
        post = self.__post.replace(_VER_MAJOR_PLACEHOLDER, ver_major)
        libname = self.__pre + head + name + post
        libname_full = self.AF_PATH + '/lib/' + libname
        return (libname, libname_full)

    def set_unsafe(self, name):
        lib = self.__clibs[name]
        if (lib is None):
            raise RuntimeError("Backend not found")
        self.__name = name

    def __init__(self):

        more_info_str = "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."

        pre, post, AF_PATH, CUDA_FOUND = _setup()

        self.__pre = pre
        self.__post = post
        self.AF_PATH = AF_PATH
        self.CUDA_FOUND = CUDA_FOUND

        self.__name = None

        self.__clibs = {'cuda'    : None,
                        'opencl'  : None,
                        'cpu'     : None,
                        'unified' : None}

        self.__backend_map = {0 : 'unified',
                              1 : 'cpu'    ,
                              2 : 'cuda'   ,
                              4 : 'opencl' }

        self.__backend_name_map = {'default' : 0,
                                   'unified' : 0,
                                   'cpu'     : 1,
                                   'cuda'    : 2,
                                   'opencl'  : 4}

        # Try to pre-load forge library if it exists
        libnames = self.__libname('forge', head='', ver_major=FORGE_VER_MAJOR)

        try:
            VERBOSE_LOADS = os.environ['AF_VERBOSE_LOADS'] == '1'
        except KeyError:
            VERBOSE_LOADS = False
            pass

        for libname in libnames:
            try:
                ct.cdll.LoadLibrary(libname)
                if VERBOSE_LOADS:
                    print('Loaded ' + libname)
                break
            except OSError:
                if VERBOSE_LOADS:
                    traceback.print_exc()
                    print('Unable to load ' + libname)
                pass

        c_dim4 = c_dim_t*4
        out = c_void_ptr_t(0)
        dims = c_dim4(10, 10, 1, 1)

        # Iterate in reverse order of preference
        for name in ('cpu', 'opencl', 'cuda', ''):
            libnames = self.__libname(name)
            for libname in libnames:
                try:
                    ct.cdll.LoadLibrary(libname)
                    __name = 'unified' if name == '' else name
                    clib = ct.CDLL(libname)
                    self.__clibs[__name] = clib
                    err = clib.af_randu(c_pointer(out), 4, c_pointer(dims), Dtype.f32.value)
                    if (err == ERR.NONE.value):
                        self.__name = __name
                        clib.af_release_array(out)
                        if VERBOSE_LOADS:
                            print('Loaded ' + libname)
                        break;
                except OSError:
                    if VERBOSE_LOADS:
                        traceback.print_exc()
                        print('Unable to load ' + libname)
                    pass

        if (self.__name is None):
            raise RuntimeError("Could not load any ArrayFire libraries.\n" + more_info_str)

    def get_id(self, name):
        return self.__backend_name_map[name]

    def get_name(self, bk_id):
        return self.__backend_map[bk_id]

    def get(self):
        return self.__clibs[self.__name]

    def name(self):
        return self.__name

    def is_unified(self):
        return self.__name == 'unified'

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
    if (backend.is_unified() == False and unsafe == False):
        raise RuntimeError("Can not change backend to %s after loading %s" % (name, backend.name()))

    if (backend.is_unified()):
        safe_call(backend.get().af_set_backend(backend.get_id(name)))
    else:
        backend.set_unsafe(name)
    return

def get_backend():
    """
    Return the name of the backend
    """
    return backend.name()

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
    backend_id = c_int_t(BACKEND.CPU.value)
    safe_call(backend.get().af_get_backend_id(c_pointer(backend_id), A.arr))
    return backend.get_name(backend_id.value)

def get_backend_count():
    """
    Get number of available backends

    Returns
    ----------

    count : int
          Number of available backends
    """
    count = c_int_t(0)
    safe_call(backend.get().af_get_backend_count(c_pointer(count)))
    return count.value

def get_available_backends():
    """
    Get names of available backends

    Returns
    ----------

    names : tuple of strings
          Names of available backends
    """
    available = c_int_t(0)
    safe_call(backend.get().af_get_available_backends(c_pointer(available)))
    return backend.parse(int(available.value))

def get_active_backend():
    """
    Get the current active backend

    name : str.
         Backend name
    """
    backend_id = c_int_t(BACKEND.CPU.value)
    safe_call(backend.get().af_get_active_backend(c_pointer(backend_id)))
    return backend.get_name(backend_id.value)

def get_device_id(A):
    """
    Get the device id of the array

    Parameters
    ----------
    A    : af.Array

    Returns
    ----------

    dev : Integer
         id of the device array was created on
    """
    device_id = c_int_t(0)
    safe_call(backend.get().af_get_device_id(c_pointer(device_id), A.arr))
    return device_id

def get_size_of(dtype):
    """
    Get the size of the type represented by arrayfire.Dtype
    """
    size = c_size_t(0)
    safe_call(backend.get().af_get_size_of(c_pointer(size), dtype.value))
    return size.value

from .util import safe_call
