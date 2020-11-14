#######################################################
# Copyright (c) 2020, ArrayFire
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

# =============================================================================
# Algorithm module
# =============================================================================
from .algorithm import accum  # noqa : E401
from .algorithm import all_true  # noqa : E401
from .algorithm import allTrueByKey  # noqa : E401
from .algorithm import any_true  # noqa : E401
from .algorithm import anyTrueByKey  # noqa : E401
from .algorithm import count  # noqa : E401
from .algorithm import countByKey  # noqa : E401
from .algorithm import diff1  # noqa : E401
from .algorithm import diff2  # noqa : E401
from .algorithm import imax  # noqa : E401
from .algorithm import imin  # noqa : E401
from .algorithm import max  # noqa : E401  # FIXME: do not use python reserved variable names
from .algorithm import maxByKey  # noqa : E401
from .algorithm import min  # noqa : E401  # FIXME: do not use python reserved variable names
from .algorithm import minByKey  # noqa : E401
from .algorithm import product  # noqa : E401
from .algorithm import productByKey  # noqa : E401
from .algorithm import scan  # noqa : E401
from .algorithm import scan_by_key  # noqa : E401
from .algorithm import set_intersect  # noqa : E401
from .algorithm import set_union  # noqa : E401
from .algorithm import set_unique  # noqa : E401
from .algorithm import sort  # noqa : E401
from .algorithm import sort_by_key  # noqa : E401
from .algorithm import sort_index  # noqa : E401
from .algorithm import sum  # noqa : E401  # FIXME: do not use python reserved variable names
from .algorithm import sumByKey  # noqa : E401
from .algorithm import where  # noqa : E401
# =============================================================================
# Arith module
# =============================================================================
from .arith import abs  # noqa : E401  # FIXME: do not use python reserved variable names
from .arith import acos  # noqa : E401
from .arith import acosh  # noqa : E401
from .arith import arg  # noqa : E401
from .arith import asin  # noqa : E401
from .arith import asinh  # noqa : E401
from .arith import atan  # noqa : E401
from .arith import atan2  # noqa : E401
from .arith import atanh  # noqa : E401
from .arith import cast  # noqa : E401
from .arith import cbrt  # noqa : E401
from .arith import ceil  # noqa : E401
from .arith import clamp  # noqa : E401
from .arith import conjg  # noqa : E401
from .arith import cos  # noqa : E401
from .arith import cosh  # noqa : E401
from .arith import cplx  # noqa : E401
from .arith import erf  # noqa : E401
from .arith import erfc  # noqa : E401
from .arith import exp  # noqa : E401
from .arith import expm1  # noqa : E401
from .arith import factorial  # noqa : E401
from .arith import floor  # noqa : E401
from .arith import hypot  # noqa : E401
from .arith import imag  # noqa : E401
from .arith import isinf  # noqa : E401
from .arith import isnan  # noqa : E401
from .arith import iszero  # noqa : E401
from .arith import lgamma  # noqa : E401
from .arith import log  # noqa : E401
from .arith import log1p  # noqa : E401
from .arith import log2  # noqa : E401
from .arith import log10  # noqa : E401
from .arith import maxof  # noqa : E401
from .arith import minof  # noqa : E401
from .arith import mod  # noqa : E401
from .arith import pow  # noqa : E401  # FIXME: do not use python reserved variable names
from .arith import pow2  # noqa : E401
from .arith import real  # noqa : E401
from .arith import rem  # noqa : E401
from .arith import root  # noqa : E401
from .arith import round  # noqa : E401  # FIXME: do not use python reserved variable names
from .arith import rsqrt  # noqa : E401
from .arith import sigmoid  # noqa : E401
from .arith import sign  # noqa : E401
from .arith import sin  # noqa : E401
from .arith import sinh  # noqa : E401
from .arith import sqrt  # noqa : E401
from .arith import tan  # noqa : E401
from .arith import tanh  # noqa : E401
from .arith import tgamma  # noqa : E401
from .arith import trunc  # noqa : E401
# =============================================================================
# Array module
# =============================================================================
from .array import Array  # noqa : E401
from .array import constant_array  # noqa : E401
from .array import display  # noqa : E401
from .array import get_display_dims_limit  # noqa : E401
from .array import read_array  # noqa : E401
from .array import save_array  # noqa : E401
from .array import set_display_dims_limit  # noqa : E401
from .array import transpose  # noqa : E401
from .array import transpose_inplace  # noqa : E401
# =============================================================================
# Base module
# =============================================================================
from .base import BaseArray  # noqa : E401
# =============================================================================
# Bcast module
# =============================================================================
from .bcast import broadcast  # noqa : E401
# =============================================================================
# Blas module
# =============================================================================
from .blas import dot  # noqa : E401
from .blas import gemm  # noqa : E401
from .blas import matmul  # noqa : E401
from .blas import matmulNT  # noqa : E401
from .blas import matmulTN  # noqa : E401
from .blas import matmulTT  # noqa : E401
# =============================================================================
# Cuda module
# =============================================================================
from .cuda import get_native_id  # noqa : E401
from .cuda import get_stream  # noqa : E401
from .cuda import set_native_id  # noqa : E401
# =============================================================================
# Data module
# =============================================================================
from .data import constant  # noqa : E401
from .data import diag  # noqa : E401
from .data import flat  # noqa : E401
from .data import flip  # noqa : E401
from .data import identity  # noqa : E401
from .data import iota  # noqa : E401
from .data import join  # noqa : E401
from .data import lookup  # noqa : E401
from .data import lower  # noqa : E401
from .data import moddims  # noqa : E401
from .data import pad  # noqa : E401
from .data import range  # noqa : E401  # FIXME: do not use python reserved variable names
from .data import reorder  # noqa : E401
from .data import replace  # noqa : E401
from .data import select  # noqa : E401
from .data import shift  # noqa : E401
from .data import tile  # noqa : E401
from .data import upper  # noqa : E401
# =============================================================================
# Device module
# =============================================================================
from .device import alloc_device  # noqa : E401
from .device import alloc_host  # noqa : E401
from .device import alloc_pinned  # noqa : E401
from .device import device_gc  # noqa : E401
from .device import device_info  # noqa : E401
from .device import device_mem_info  # noqa : E401
from .device import eval  # noqa : E401  # FIXME: do not use python reserved variable names
from .device import free_device  # noqa : E401
from .device import free_host  # noqa : E401
from .device import free_pinned  # noqa : E401
from .device import get_device  # noqa : E401
from .device import get_device_count  # noqa : E401
from .device import get_device_ptr  # noqa : E401
from .device import get_manual_eval_flag  # noqa : E401
from .device import info  # noqa : E401
from .device import info_str  # noqa : E401
from .device import init  # noqa : E401
from .device import is_dbl_supported  # noqa : E401
from .device import is_half_supported  # noqa : E401
from .device import is_locked_array  # noqa : E401
from .device import lock_array  # noqa : E401
from .device import lock_device_ptr  # noqa : E401
from .device import print_mem_info  # noqa : E401
from .device import set_device  # noqa : E401
from .device import set_manual_eval_flag  # noqa : E401
from .device import sync  # noqa : E401
from .device import unlock_array  # noqa : E401
from .device import unlock_device_ptr  # noqa : E401
# =============================================================================
# Graphics module
# =============================================================================
from .graphics import Window  # noqa : E401
# =============================================================================
# Image module
# =============================================================================
from .image import ITERATIVE_DECONV  # noqa : E401
from .image import anisotropic_diffusion  # noqa : E401
from .image import bilateral  # noqa : E401
from .image import canny  # noqa : E401
from .image import color_space  # noqa : E401
from .image import confidenceCC  # noqa : E401
from .image import dilate  # noqa : E401
from .image import dilate3  # noqa : E401
from .image import erode  # noqa : E401
from .image import erode3  # noqa : E401
from .image import gaussian_kernel  # noqa : E401
from .image import gradient  # noqa : E401
from .image import gray2rgb  # noqa : E401
from .image import hist_equal  # noqa : E401
from .image import histogram  # noqa : E401
from .image import hsv2rgb  # noqa : E401
from .image import is_image_io_available  # noqa : E401
from .image import inverseDeconv  # noqa : E401
from .image import iterativeDeconv  # noqa : E401
from .image import load_image  # noqa : E401
from .image import load_image_native  # noqa : E401
from .image import maxfilt  # noqa : E401
from .image import mean_shift  # noqa : E401
from .image import minfilt  # noqa : E401
from .image import moments  # noqa : E401
from .image import regions  # noqa : E401
from .image import resize  # noqa : E401
from .image import rgb2gray  # noqa : E401
from .image import rgb2hsv  # noqa : E401
from .image import rgb2ycbcr  # noqa : E401
from .image import rotate  # noqa : E401
from .image import sat  # noqa : E401
from .image import save_image  # noqa : E401
from .image import save_image_native  # noqa : E401
from .image import scale  # noqa : E401
from .image import skew  # noqa : E401
from .image import sobel_derivatives  # noqa : E401
from .image import sobel_filter  # noqa : E401
from .image import transform  # noqa : E401
from .image import translate  # noqa : E401
from .image import unwrap  # noqa : E401
from .image import wrap  # noqa : E401
from .image import ycbcr2rgb  # noqa : E401
# =============================================================================
# Index module
# =============================================================================
from .index import Index  # noqa : E401
from .index import ParallelRange  # noqa : E401
from .index import Seq  # noqa : E401
# =============================================================================
# Interop module
# =============================================================================
from .interop import AF_NUMBA_FOUND  # noqa : E401
from .interop import AF_NUMPY_FOUND  # noqa : E401
from .interop import AF_PYCUDA_FOUND  # noqa : E401
from .interop import AF_PYOPENCL_FOUND  # noqa : E401
from .interop import to_array  # noqa : E401
# =============================================================================
# Lapack module
# =============================================================================
from .lapack import cholesky  # noqa : E401
from .lapack import cholesky_inplace  # noqa : E401
from .lapack import det  # noqa : E401
from .lapack import inverse  # noqa : E401
from .lapack import is_lapack_available  # noqa : E401
from .lapack import lu  # noqa : E401
from .lapack import lu_inplace  # noqa : E401
from .lapack import norm  # noqa : E401
from .lapack import pinverse  # noqa : E401
from .lapack import qr  # noqa : E401
from .lapack import qr_inplace  # noqa : E401
from .lapack import rank  # noqa : E401
from .lapack import solve  # noqa : E401
from .lapack import solve_lu  # noqa : E401
from .lapack import svd  # noqa : E401
from .lapack import svd_inplace  # noqa : E401
# =============================================================================
# Library module
# =============================================================================
from .library import AF_VER_MAJOR  # noqa : E401
from .library import BACKEND  # noqa : E401
from .library import BINARYOP  # noqa : E401
from .library import CANNY_THRESHOLD  # noqa : E401
from .library import COLORMAP  # noqa : E401
from .library import CONNECTIVITY  # noqa : E401
from .library import CONV_DOMAIN  # noqa : E401
from .library import CONV_GRADIENT  # noqa : E401
from .library import CONV_MODE  # noqa : E401
from .library import CSPACE  # noqa : E401
from .library import DIFFUSION  # noqa : E401
from .library import ERR  # noqa : E401
from .library import FLUX  # noqa : E401
from .library import FORGE_VER_MAJOR  # noqa : E401
from .library import HOMOGRAPHY  # noqa : E401
from .library import IMAGE_FORMAT  # noqa : E401
from .library import INTERP  # noqa : E401
from .library import INVERSE_DECONV  # noqa : E401
from .library import ITERATIVE_DECONV  # noqa : E401
from .library import MARKER  # noqa : E401
from .library import MATCH  # noqa : E401
from .library import MATPROP  # noqa : E401
from .library import MOMENT  # noqa : E401
from .library import NORM  # noqa : E401
from .library import PAD  # noqa : E401
from .library import RANDOM_ENGINE  # noqa : E401
from .library import STORAGE  # noqa : E401
from .library import TOPK  # noqa : E401
from .library import VARIANCE  # noqa : E401
from .library import YCC_STD  # noqa : E401
from .library import Dtype  # noqa : E401
from .library import Source  # noqa : E401
from .library import get_active_backend  # noqa : E401
from .library import get_available_backends  # noqa : E401
from .library import get_backend  # noqa : E401
from .library import get_backend_count  # noqa : E401
from .library import get_backend_id  # noqa : E401
from .library import get_device_id  # noqa : E401
from .library import get_size_of  # noqa : E401
from .library import safe_call  # noqa : E401
from .library import set_backend  # noqa : E401
from .library import to_str  # noqa : E401
# =============================================================================
# Machine Learning (ML) module
# =============================================================================
from .ml import convolve2GradientNN  # noqa : E401
# =============================================================================
# Random module
# =============================================================================
from .random import Random_Engine  # noqa : E401
from .random import get_default_random_engine  # noqa : E401
from .random import get_seed  # noqa : E401
from .random import randn  # noqa : E401
from .random import randu  # noqa : E401
from .random import set_default_random_engine_type  # noqa : E401
from .random import set_seed  # noqa : E401
# =============================================================================
# Signal module
# =============================================================================
from .signal import approx1  # noqa : E401
from .signal import approx1_uniform  # noqa : E401
from .signal import approx2  # noqa : E401
from .signal import approx2_uniform  # noqa : E401
from .signal import convolve  # noqa : E401
from .signal import convolve1  # noqa : E401
from .signal import convolve2  # noqa : E401
from .signal import convolve2_separable  # noqa : E401
from .signal import convolve2NN  # noqa : E401
from .signal import convolve3  # noqa : E401
from .signal import dft  # noqa : E401
from .signal import fft  # noqa : E401
from .signal import fft2  # noqa : E401
from .signal import fft2_c2r  # noqa : E401
from .signal import fft2_inplace  # noqa : E401
from .signal import fft2_r2c  # noqa : E401
from .signal import fft3  # noqa : E401
from .signal import fft3_c2r  # noqa : E401
from .signal import fft3_inplace  # noqa : E401
from .signal import fft3_r2c  # noqa : E401
from .signal import fft_c2r  # noqa : E401
from .signal import fft_convolve  # noqa : E401
from .signal import fft_convolve1  # noqa : E401
from .signal import fft_convolve2  # noqa : E401
from .signal import fft_convolve3  # noqa : E401
from .signal import fft_inplace  # noqa : E401
from .signal import fft_r2c  # noqa : E401
from .signal import fir  # noqa : E401
from .signal import idft  # noqa : E401
from .signal import ifft  # noqa : E401
from .signal import ifft2  # noqa : E401
from .signal import ifft2_inplace  # noqa : E401
from .signal import ifft3  # noqa : E401
from .signal import ifft3_inplace  # noqa : E401
from .signal import ifft_inplace  # noqa : E401
from .signal import iir  # noqa : E401
from .signal import medfilt  # noqa : E401
from .signal import medfilt1  # noqa : E401
from .signal import medfilt2  # noqa : E401
from .signal import set_fft_plan_cache_size  # noqa : E401
# =============================================================================
# Sparse module
# =============================================================================
from .sparse import convert_sparse  # noqa : E401
from .sparse import convert_sparse_to_dense  # noqa : E401
from .sparse import create_sparse  # noqa : E401
from .sparse import create_sparse_from_dense  # noqa : E401
from .sparse import create_sparse_from_host  # noqa : E401
from .sparse import sparse_get_col_idx  # noqa : E401
from .sparse import sparse_get_info  # noqa : E401
from .sparse import sparse_get_nnz  # noqa : E401
from .sparse import sparse_get_row_idx  # noqa : E401
from .sparse import sparse_get_storage  # noqa : E401
from .sparse import sparse_get_values  # noqa : E401
# =============================================================================
# Statistics module
# =============================================================================
from .statistics import corrcoef  # noqa : E401
from .statistics import cov  # noqa : E401
from .statistics import mean  # noqa : E401
from .statistics import meanvar  # noqa : E401
from .statistics import median  # noqa : E401
from .statistics import stdev  # noqa : E401
from .statistics import topk  # noqa : E401
from .statistics import var  # noqa : E401
# =============================================================================
# Timer module
# =============================================================================
from .timer import timeit  # noqa : E401
# =============================================================================
# Utils module
# =============================================================================
from .util import dim4  # noqa : E401
from .util import dim4_to_tuple  # noqa : E401
from .util import get_reversion  # noqa : E401
from .util import get_version  # noqa : E401
from .util import implicit_dtype  # noqa : E401
from .util import number_dtype  # noqa : E401
from .util import to_c_type  # noqa : E401
from .util import to_dtype  # noqa : E401
from .util import to_typecode  # noqa : E401

try:
    # FIXME: pycuda imported but unused
    import pycuda.autoinit  # noqa : E401
except ImportError:
    pass
