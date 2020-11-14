#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Module containing enums and other constants.
"""

import ctypes as ct
import os
import platform
import traceback
from enum import Enum

c_float_t = ct.c_float
c_double_t = ct.c_double
c_int_t = ct.c_int
c_uint_t = ct.c_uint
c_longlong_t = ct.c_longlong
c_ulonglong_t = ct.c_ulonglong
c_char_t = ct.c_char
c_bool_t = ct.c_bool
c_uchar_t = ct.c_ubyte
c_short_t = ct.c_short
c_ushort_t = ct.c_ushort
c_pointer = ct.pointer
c_void_ptr_t = ct.c_void_p
c_char_ptr_t = ct.c_char_p
c_size_t = ct.c_size_t
c_cast = ct.cast


class af_cfloat_t(ct.Structure):
    _fields_ = [("real", ct.c_float), ("imag", ct.c_float)]


class af_cdouble_t(ct.Structure):
    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]


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


class ERR(Enum):
    """
    Error values. For internal use only.
    """

    NONE = 0

    # 100-199 Errors in environment
    NO_MEM = 101
    DRIVER = 102
    RUNTIME = 103

    # 200-299 Errors in input parameters
    INVALID_ARRAY = 201
    ARG = 202
    SIZE = 203
    TYPE = 204
    DIFF_TYPE = 205
    BATCH = 207
    DEVICE = 208

    # 300-399 Errors for missing software features
    NOT_SUPPORTED = 301
    NOT_CONFIGURED = 302
    NONFREE = 303

    # 400-499 Errors for missing hardware features
    NO_DBL = 401
    NO_GFX = 402
    NO_HALF = 403

    # 500-599 Errors specific to the heterogeneous API
    LOAD_LIB = 501
    LOAD_SYM = 502
    ARR_BKND_MISMATCH = 503

    # 900-999 Errors from upstream libraries and runtimes
    INTERNAL = 998
    UNKNOWN = 999


class Dtype(Enum):
    """
    Error values. For internal use only.
    """
    f32 = 0
    c32 = 1
    f64 = 2
    c64 = 3
    b8 = 4
    s32 = 5
    u32 = 6
    u8 = 7
    s64 = 8
    u64 = 9
    s16 = 10
    u16 = 11
    f16 = 12


class Source(Enum):
    """
    Source of the pointer
    """
    device = 0
    host = 1


class INTERP(Enum):
    """
    Interpolation method
    """
    NEAREST = 0
    LINEAR = 1
    BILINEAR = 2
    CUBIC = 3
    LOWER = 4
    LINEAR_COSINE = 5
    BILINEAR_COSINE = 6
    BICUBIC = 7
    CUBIC_SPLINE = 8
    BICUBIC_SPLINE = 9


class PAD(Enum):
    """
    Edge padding types
    """
    ZERO = 0
    SYM = 1
    CLAMP_TO_EDGE = 2
    PERIODIC = 3


class CONNECTIVITY(Enum):
    """
    Neighborhood connectivity
    """
    FOUR = 4
    EIGHT = 8


class CONV_MODE(Enum):
    """
    Convolution mode
    """
    DEFAULT = 0
    EXPAND = 1


class CONV_DOMAIN(Enum):
    """
    Convolution domain
    """
    AUTO = 0
    SPATIAL = 1
    FREQ = 2


class CONV_GRADIENT(Enum):
    """
    Convolution gradient type
    """
    DEFAULT = 0
    FILTER = 1
    DATA = 2
    BIAS = 3


class MATCH(Enum):
    """
    Match type
    """

    """
    Sum of absolute differences
    """
    SAD = 0

    """
    Zero mean SAD
    """
    ZSAD = 1

    """
    Locally scaled SAD
    """
    LSAD = 2

    """
    Sum of squared differences
    """
    SSD = 3

    """
    Zero mean SSD
    """
    ZSSD = 4

    """
    Locally scaled SSD
    """
    LSSD = 5

    """
    Normalized cross correlation
    """
    NCC = 6

    """
    Zero mean NCC
    """
    ZNCC = 7

    """
    Sum of hamming distances
    """
    SHD = 8


class YCC_STD(Enum):
    """
    YCC Standard formats
    """
    BT_601 = 601
    BT_709 = 709
    BT_2020 = 2020


class CSPACE(Enum):
    """
    Colorspace formats
    """
    GRAY = 0
    RGB = 1
    HSV = 2
    YCbCr = 3


class MATPROP(Enum):
    """
    Matrix properties
    """

    """
    None, general.
    """
    NONE = 0

    """
    Transposed.
    """
    TRANS = 1

    """
    Conjugate transposed.
    """
    CTRANS = 2

    """
    Upper triangular matrix.
    """
    UPPER = 32

    """
    Lower triangular matrix.
    """
    LOWER = 64

    """
    Treat diagonal as units.
    """
    DIAG_UNIT = 128

    """
    Symmetric matrix.
    """
    SYM = 512

    """
    Positive definite matrix.
    """
    POSDEF = 1024

    """
    Orthogonal matrix.
    """
    ORTHOG = 2048

    """
    Tri diagonal matrix.
    """
    TRI_DIAG = 4096

    """
    Block diagonal matrix.
    """
    BLOCK_DIAG = 8192


class NORM(Enum):
    """
    Norm types
    """
    VECTOR_1 = 0
    VECTOR_INF = 1
    VECTOR_2 = 2
    VECTOR_P = 3
    MATRIX_1 = 4
    MATRIX_INF = 5
    MATRIX_2 = 6
    MATRIX_L_PQ = 7
    EUCLID = VECTOR_2


class COLORMAP(Enum):
    """
    Colormaps
    """
    DEFAULT = 0
    SPECTRUM = 1
    COLORS = 2
    RED = 3
    MOOD = 4
    HEAT = 5
    BLUE = 6


class IMAGE_FORMAT(Enum):
    """
    Image Formats
    """
    BMP = 0
    ICO = 1
    JPEG = 2
    JNG = 3
    PNG = 13
    PPM = 14
    PPMRAW = 15
    TIFF = 18
    PSD = 20
    HDR = 26
    EXR = 29
    JP2 = 31
    RAW = 34


class HOMOGRAPHY(Enum):
    """
    Homography Types
    """
    RANSAC = 0
    LMEDS = 1


class BACKEND(Enum):
    """
    Backend libraries
    """
    DEFAULT = 0
    CPU = 1
    CUDA = 2
    OPENCL = 4


class MARKER(Enum):
    """
    Markers used for different points in graphics plots
    """
    NONE = 0
    POINT = 1
    CIRCLE = 2
    SQUARE = 3
    TRIANGE = 4
    CROSS = 5
    PLUS = 6
    STAR = 7


class MOMENT(Enum):
    """
    Image Moments types
    """
    M00 = 1
    M01 = 2
    M10 = 4
    M11 = 8
    FIRST_ORDER = 15


class BINARYOP(Enum):
    """
    Binary Operators
    """
    ADD = 0
    MUL = 1
    MIN = 2
    MAX = 3


class RANDOM_ENGINE(Enum):
    """
    Random engine types
    """
    PHILOX_4X32_10 = 100
    THREEFRY_2X32_16 = 200
    MERSENNE_GP11213 = 300
    PHILOX = PHILOX_4X32_10
    THREEFRY = THREEFRY_2X32_16
    DEFAULT = PHILOX


class STORAGE(Enum):
    """
    Matrix Storage types
    """
    DENSE = 0
    CSR = 1
    CSC = 2
    COO = 3


class CANNY_THRESHOLD(Enum):
    """
    Canny Edge Threshold types
    """
    MANUAL = 0
    AUTO_OTSU = 1


class FLUX(Enum):
    """
    Flux functions
    """
    DEFAULT = 0
    QUADRATIC = 1
    EXPONENTIAL = 2


class DIFFUSION(Enum):
    """
    Diffusion equations
    """
    DEFAULT = 0
    GRAD = 1
    MCDE = 2


class TOPK(Enum):
    """
    Top-K ordering
    """
    DEFAULT = 0
    MIN = 1
    MAX = 2


class ITERATIVE_DECONV(Enum):
    """
    Iterative deconvolution algorithm
    """
    DEFAULT = 0
    LANDWEBER = 1
    RICHARDSONLUCY = 2


class INVERSE_DECONV(Enum):
    """
    Inverse deconvolution algorithm
    """
    DEFAULT = 0
    TIKHONOV = 1


class VARIANCE(Enum):
    """
    Variance bias type
    """
    DEFAULT = 0
    SAMPLE = 1
    POPULATION = 2


AF_VER_MAJOR = "3"
FORGE_VER_MAJOR = "1"
_VER_MAJOR_PLACEHOLDER = "__VER_MAJOR__"


def _setup():
    import platform

    platform_name = platform.system()

    try:
        AF_PATH = os.environ["AF_PATH"]
    except KeyError:
        AF_PATH = None

    AF_SEARCH_PATH = AF_PATH

    try:
        CUDA_PATH = os.environ["CUDA_PATH"]
    except KeyError:
        CUDA_PATH = None

    CUDA_FOUND = False

    assert len(platform_name) >= 3
    if platform_name == "Windows" or platform_name[:3] == "CYG":
        # Windows specific setup
        pre = ""
        post = ".dll"
        if platform_name == "Windows":
            # Supressing crashes caused by missing dlls
            # http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            # https://msdn.microsoft.com/en-us/library/windows/desktop/ms680621.aspx
            ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH = "C:/Program Files/ArrayFire/v" + AF_VER_MAJOR + "/"

        if CUDA_PATH is not None:
            CUDA_FOUND = os.path.isdir(CUDA_PATH + "/bin") and os.path.isdir(CUDA_PATH + "/nvvm/bin/")

    elif platform_name == "Darwin":
        # OSX specific setup
        pre = "lib"
        post = "." + _VER_MAJOR_PLACEHOLDER + ".dylib"

        if AF_SEARCH_PATH is None:
            if os.path.exists('/opt/arrayfire'):
                AF_SEARCH_PATH = '/opt/arrayfire/'
            else:
                AF_SEARCH_PATH = '/usr/local/'

        if CUDA_PATH is None:
            CUDA_PATH = "/usr/local/cuda/"

        CUDA_FOUND = os.path.isdir(CUDA_PATH + "/lib") and os.path.isdir(CUDA_PATH + "/nvvm/lib")

    elif platform_name == "Linux":
        pre = "lib"
        post = ".so." + _VER_MAJOR_PLACEHOLDER

        if AF_SEARCH_PATH is None:
            AF_SEARCH_PATH = "/opt/arrayfire-" + AF_VER_MAJOR + "/"

        if CUDA_PATH is None:
            CUDA_PATH = "/usr/local/cuda/"

        if platform.architecture()[0][:2] == "64":
            CUDA_FOUND = os.path.isdir(CUDA_PATH + "/lib64") and os.path.isdir(CUDA_PATH + "/nvvm/lib64")
        else:
            CUDA_FOUND = os.path.isdir(CUDA_PATH + "/lib") and os.path.isdir(CUDA_PATH + "/nvvm/lib")
    else:
        raise OSError(platform_name + " not supported")

    if AF_PATH is None:
        os.environ["AF_PATH"] = AF_SEARCH_PATH

    return pre, post, AF_SEARCH_PATH, CUDA_FOUND


class _clibrary(object):

    def __libname(self, name, head="af", ver_major=AF_VER_MAJOR):
        post = self.__post.replace(_VER_MAJOR_PLACEHOLDER, ver_major)
        libname = self.__pre + head + name + post
        if os.path.isdir(self.AF_PATH + '/lib64'):
            libname_full = self.AF_PATH + '/lib64/' + libname
        else:
            libname_full = self.AF_PATH + '/lib/' + libname
        return (libname, libname_full)

    def set_unsafe(self, name):
        lib = self.__clibs[name]
        if lib is None:
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

        self.__clibs = {
            "cuda": None,
            "opencl": None,
            "cpu": None,
            "unified": None}

        self.__backend_map = {
            0: "unified",
            1: "cpu",
            2: "cuda",
            4: "opencl"}

        self.__backend_name_map = {
            "default": 0,
            "unified": 0,
            "cpu": 1,
            "cuda": 2,
            "opencl": 4}

        # Try to pre-load forge library if it exists
        libnames = self.__libname("forge", head="", ver_major=FORGE_VER_MAJOR)

        try:
            VERBOSE_LOADS = os.environ["AF_VERBOSE_LOADS"] == "1"
        except KeyError:
            VERBOSE_LOADS = False

        for libname in libnames:
            try:
                ct.cdll.LoadLibrary(libname)
                if VERBOSE_LOADS:
                    print("Loaded " + libname)
                break
            except OSError:
                if VERBOSE_LOADS:
                    traceback.print_exc()
                    print("Unable to load " + libname)

        c_dim4 = c_dim_t*4
        out = c_void_ptr_t(0)
        dims = c_dim4(10, 10, 1, 1)

        # Iterate in reverse order of preference
        for name in {"cpu", "opencl", "cuda", ""}:
            libnames = self.__libname(name)
            for libname in libnames:
                try:
                    ct.cdll.LoadLibrary(libname)
                    __name = "unified" if name == "" else name
                    clib = ct.CDLL(libname)
                    self.__clibs[__name] = clib
                    err = clib.af_randu(c_pointer(out), 4, c_pointer(dims), Dtype.f32.value)
                    if err != ERR.NONE.value:
                        return
                    self.__name = __name
                    clib.af_release_array(out)
                    if VERBOSE_LOADS:
                        print("Loaded " + libname)
                    break
                except OSError:
                    if VERBOSE_LOADS:
                        traceback.print_exc()
                        print("Unable to load " + libname)

        if self.__name is None:
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
        return self.__name == "unified"

    def parse(self, res):
        lst = []
        for key, value in self.__backend_name_map.items():
            if value & res:
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
    if not (backend.is_unified() or unsafe):
        raise RuntimeError("Can not change backend to %s after loading %s" % (name, backend.name()))

    if backend.is_unified():
        safe_call(backend.get().af_set_backend(backend.get_id(name)))

    backend.set_unsafe(name)


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


def safe_call(af_error):
    if af_error == ERR.NONE.value:
        return
    err_str = c_char_ptr_t(0)
    err_len = c_dim_t(0)
    backend.get().af_get_last_error(c_pointer(err_str), c_pointer(err_len))
    raise RuntimeError(to_str(err_str))


def to_str(c_str):
    return str(c_str.value.decode('utf-8'))
