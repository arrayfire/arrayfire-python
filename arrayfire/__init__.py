#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
ArrayFire is a high performance scientific computing library with an easy to use API.


    >>> # Monte Carlo estimation of pi
    >>> def calc_pi_device(samples):
            # Simple, array based API
            # Generate uniformly distributed random numers
            x = af.randu(samples)
            y = af.randu(samples)
            # Supports Just In Time Compilation
            # The following line generates a single kernel
            within_unit_circle = (x * x + y * y) < 1
            # Intuitive function names
            return 4 * af.count(within_unit_circle) / samples

Programs written using ArrayFire are portable across CUDA, OpenCL and CPU devices.

The default backend is chosen in the following order of preference based on the available libraries:

    1. CUDA
    2. OpenCL
    3. CPU

The backend can be chosen at the beginning of the program by using the following function

    >>> af.set_backend(name)

where name is one of 'cuda', 'opencl' or 'cpu'.

The functionality provided by ArrayFire spans the following domains:

    1. Vector Algorithms
    2. Image Processing
    3. Signal Processing
    4. Computer Vision
    5. Linear Algebra
    6. Statistics

"""

from .algorithm import (
    accum, all_true, allTrueByKey, any_true, anyTrueByKey, count, countByKey, diff1, diff2, imax, imin, max, maxByKey, min, minByKey, product, productByKey, scan, scan_by_key, set_intersect,
    set_union, set_unique, sort, sort_by_key, sort_index, sum, sumByKey, where)
from .arith import (
    abs, acos, acosh, arg, asin, asinh, atan, atan2, atanh, cast, cbrt, ceil, clamp, conjg, cos, cosh, cplx, erf, erfc, exp,
    expm1, factorial, floor, hypot, imag, isinf, isnan, iszero, lgamma, log, log1p, log2, log10, maxof, minof,
    mod, pow,
    pow2, real, rem, root, round, rsqrt, sigmoid, sign, sin, sinh, sqrt, tan, tanh, tgamma, trunc)
from .array import (
    Array, constant_array, display, get_display_dims_limit, read_array, save_array,
    set_display_dims_limit, transpose, transpose_inplace)
from .base import BaseArray
from .bcast import broadcast
from .blas import dot, matmul, matmulNT, matmulTN, matmulTT, gemm
from .cuda import get_native_id, get_stream, set_native_id
from .data import (
    constant, diag, flat, flip, identity, iota, join, lookup, lower, moddims, pad, range, reorder, replace, select, shift,
    tile, upper)
from .library import (
    BACKEND, BINARYOP, CANNY_THRESHOLD, COLORMAP, CONNECTIVITY, CONV_DOMAIN, CONV_GRADIENT, CONV_MODE, CSPACE, DIFFUSION, ERR, FLUX,
    HOMOGRAPHY, IMAGE_FORMAT, INTERP, ITERATIVE_DECONV, INVERSE_DECONV, MARKER, MATCH, MATPROP, MOMENT, NORM, PAD, RANDOM_ENGINE, STORAGE, TOPK, VARIANCE, YCC_STD, Dtype, Source, AF_VER_MAJOR, FORGE_VER_MAJOR)
from .device import (
    alloc_device, alloc_host, alloc_pinned, device_gc, device_info, device_mem_info, eval, free_device, free_host,
    free_pinned, get_device, get_device_count, get_device_ptr, get_manual_eval_flag, info,
    info_str, init, is_dbl_supported, is_half_supported, is_locked_array, lock_array, lock_device_ptr, print_mem_info, set_device,
    set_manual_eval_flag, sync, unlock_array, unlock_device_ptr)
from .graphics import Window
from .image import (
    anisotropic_diffusion, bilateral, canny, color_space, confidenceCC, dilate, dilate3, erode, erode3, gaussian_kernel, gradient,
    gray2rgb, hist_equal, histogram, hsv2rgb, is_image_io_available, iterativeDeconv, inverseDeconv, load_image, load_image_native, maxfilt,
    mean_shift, minfilt, moments, regions, resize, rgb2gray, rgb2hsv, rgb2ycbcr, rotate, sat, save_image,
    save_image_native, scale, skew, sobel_derivatives, sobel_filter, transform, translate, unwrap, wrap, ycbcr2rgb)
from .index import Index, ParallelRange, Seq
from .interop import AF_NUMBA_FOUND, AF_NUMPY_FOUND, AF_PYCUDA_FOUND, AF_PYOPENCL_FOUND, to_array
from .lapack import (
    cholesky, cholesky_inplace, det, inverse, is_lapack_available, lu, lu_inplace, norm, pinverse, qr, qr_inplace, rank, solve,
    solve_lu, svd, svd_inplace)
from .library import (
    get_active_backend, get_available_backends, get_backend, get_backend_count, get_backend_id, get_device_id,
    get_size_of, safe_call, set_backend)
from .ml import convolve2GradientNN
from .random import (
    Random_Engine, get_default_random_engine, get_seed, randn, randu, set_default_random_engine_type,
    set_seed)
from .signal import (
    approx1, approx1_uniform, approx2, approx2_uniform, convolve, convolve1, convolve2, convolve2NN, convolve2_separable, convolve3, dft, fft, fft2, fft2_c2r,
    fft2_inplace, fft2_r2c, fft3, fft3_c2r, fft3_inplace, fft3_r2c, fft_c2r, fft_convolve, fft_convolve1,
    fft_convolve2, fft_convolve3, fft_inplace, fft_r2c, fir, idft, ifft, ifft2, ifft2_inplace, ifft3, ifft3_inplace,
    ifft_inplace, iir, medfilt, medfilt1, medfilt2, set_fft_plan_cache_size)
from .sparse import (
    convert_sparse, convert_sparse_to_dense, create_sparse, create_sparse_from_dense, create_sparse_from_host,
    sparse_get_col_idx, sparse_get_info, sparse_get_nnz, sparse_get_row_idx, sparse_get_storage, sparse_get_values)
from .statistics import corrcoef, cov, mean, meanvar, median, stdev, topk, var
from .timer import timeit
from .util import dim4, dim4_to_tuple, implicit_dtype, number_dtype, to_str, get_reversion, get_version, to_dtype, to_typecode, to_c_type

try:
    # FIXME: pycuda imported but unused
    import pycuda.autoinit
except ImportError:
    pass


__all__ = [
    # algorithm
    "accum", "all_true", "allTrueByKey", "any_true", "anyTrueByKey", "count", "countByKey",
    "diff1", "diff2", "imax", "imin", "max", "maxByKey", "min", "minByKey", "product",
    "productByKey", "scan", "scan_by_key", "set_intersect", "set_union", "set_unique",
    "sort", "sort_by_key", "sort_index", "sum", "sumByKey", "where",
    # arith
    "abs", "acos", "acosh", "arg", "asin", "asinh", "atan", "atan2", "atanh",
    "cast", "cbrt", "ceil", "clamp", "conjg", "cos", "cosh", "cplx", "erf",
    "erfc", "exp", "expm1", "factorial", "floor", "hypot", "imag", "isinf",
    "isnan", "iszero", "lgamma", "log", "log1p", "log2", "log10", "maxof",
    "minof", "mod", "pow", "pow2", "real", "rem", "root", "round", "rsqrt",
    "sigmoid", "sign", "sin", "sinh", "sqrt", "tan", "tanh", "tgamma", "trunc",
    # array
    "Array", "constant_array", "display", "get_display_dims_limit", "read_array",
    "save_array", "set_display_dims_limit", "transpose", "transpose_inplace",
    # base
    "BaseArray",
    # bcast
    "broadcast",
    # blas
    "dot", "matmul", "matmulNT", "matmulTN", "matmulTT", "gemm",
    #cuda
    "get_native_id", "get_stream", "set_native_id",
    # data
    "constant", "diag", "flat", "flip", "identity", "iota", "join", "lookup",
    "lower", "moddims", "pad", "range", "reorder", "replace", "select",
    "shift", "tile", "upper",
    # library
    "BACKEND", "BINARYOP", "CANNY_THRESHOLD", "COLORMAP", "CONNECTIVITY", "CONV_DOMAIN",
    "CONV_GRADIENT", "CONV_MODE", "CSPACE", "DIFFUSION", "ERR", "FLUX", "HOMOGRAPHY",
    "IMAGE_FORMAT", "INTERP", "ITERATIVE_DECONV", "INVERSE_DECONV", "MARKER", "MATCH",
    "MATPROP", "MOMENT", "NORM", "PAD", "RANDOM_ENGINE", "STORAGE", "TOPK", "VARIANCE",
    "YCC_STD", "Dtype", "Source", "AF_VER_MAJOR", "FORGE_VER_MAJOR",
    # device
    "alloc_device", "alloc_host", "alloc_pinned", "device_gc", "device_info", "device_mem_info",
    "eval", "free_device", "free_host", "free_pinned", "get_device", "get_device_count",
    "get_device_ptr", "get_manual_eval_flag", "info", "info_str", "init", "is_dbl_supported",
    "is_half_supported", "is_locked_array", "lock_array", "lock_device_ptr", "print_mem_info",
    "set_device", "set_manual_eval_flag", "sync", "unlock_array", "unlock_device_ptr",
    # graphics
    "Window",
    # image
    "anisotropic_diffusion", "bilateral", "canny", "color_space", "confidenceCC", "dilate", "dilate3",
    "erode", "erode3", "gaussian_kernel", "gradient", "gray2rgb", "hist_equal", "histogram", "hsv2rgb",
    "is_image_io_available", "iterativeDeconv", "inverseDeconv", "load_image", "load_image_native",
    "maxfilt", "mean_shift", "minfilt", "moments", "regions", "resize", "rgb2gray", "rgb2hsv", "rgb2ycbcr",
    "rotate", "sat", "save_image", "save_image_native", "scale", "skew", "sobel_derivatives", "sobel_filter",
    "transform", "translate", "unwrap", "wrap", "ycbcr2rgb",
    # index
    "Index", "ParallelRange", "Seq",
    # interop
    "AF_NUMBA_FOUND", "AF_NUMPY_FOUND", "AF_PYCUDA_FOUND", "AF_PYOPENCL_FOUND", "to_array",
    # lapack
    "cholesky", "cholesky_inplace", "det", "inverse", "is_lapack_available", "lu", "lu_inplace",
    "norm", "pinverse", "qr", "qr_inplace", "rank", "solve", "solve_lu", "svd", "svd_inplace",
    # library
    "get_active_backend", "get_available_backends", "get_backend", "get_backend_count",
    "get_backend_id", "get_device_id", "get_size_of", "safe_call", "set_backend",
    # ml
    "convolve2GradientNN",
    # random
    "Random_Engine", "get_default_random_engine", "get_seed", "randn", "randu",
    "set_default_random_engine_type", "set_seed",
    # signal
    "approx1", "approx1_uniform", "approx2", "approx2_uniform", "convolve", "convolve1",
    "convolve2", "convolve2NN", "convolve2_separable", "convolve3", "dft", "fft",
    "fft2", "fft2_c2r", "fft2_inplace", "fft2_r2c", "fft3", "fft3_c2r", "fft3_inplace",
    "fft3_r2c", "fft_c2r", "fft_convolve", "fft_convolve1", "fft_convolve2", "fft_convolve3",
    "fft_inplace", "fft_r2c", "fir", "idft", "ifft", "ifft2", "ifft2_inplace", "ifft3",
    "ifft3_inplace", "ifft_inplace", "iir", "medfilt", "medfilt1", "medfilt2",
    "set_fft_plan_cache_size",
    # sparse
    "convert_sparse", "convert_sparse_to_dense", "create_sparse", "create_sparse_from_dense",
    "create_sparse_from_host", "sparse_get_col_idx", "sparse_get_info", "sparse_get_nnz",
    "sparse_get_row_idx", "sparse_get_storage", "sparse_get_values",
    # statistics
    "corrcoef", "cov", "mean", "meanvar", "median", "stdev", "topk", "var",
    # timer
    "timeit",
    # util
    "dim4", "dim4_to_tuple", "implicit_dtype", "number_dtype", "to_str", "get_reversion",
    "get_version", "to_dtype", "to_typecode", "to_c_type"
]
