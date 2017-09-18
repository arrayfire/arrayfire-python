#!/usr/bin/python
#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af
from . import _util

def simple_interop(verbose = False):
    if af.AF_NUMPY_FOUND:
        import numpy as np
        n = np.random.random((5,))
        a = af.to_array(n)
        n2 = np.array(a)
        assert((n==n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n==n2).all())

        n = np.random.random((5,3))
        a = af.to_array(n)
        n2 = np.array(a)
        assert((n==n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n==n2).all())

        n = np.random.random((5,3,2))
        a = af.to_array(n)
        n2 = np.array(a)
        assert((n==n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n==n2).all())

        n = np.random.random((5,3,2,2))
        a = af.to_array(n)
        n2 = np.array(a)
        assert((n==n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n==n2).all())

    if af.AF_PYCUDA_FOUND and af.get_active_backend() == 'cuda':
        import pycuda.autoinit
        import pycuda.gpuarray as cudaArray
        n = np.random.random((5,))
        c = cudaArray.to_gpu(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

        n = np.random.random((5,3))
        c = cudaArray.to_gpu(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

        n = np.random.random((5,3,2))
        c = cudaArray.to_gpu(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

        n = np.random.random((5,3,2,2))
        c = cudaArray.to_gpu(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

    if af.AF_PYOPENCL_FOUND and af.backend.name() == 'opencl':
        # This needs fixing upstream
        # https://github.com/arrayfire/arrayfire/issues/1728

        # import pyopencl as cl
        # import pyopencl.array as clArray
        # ctx = cl.create_some_context()
        # queue = cl.CommandQueue(ctx)

        # n = np.random.random((5,))
        # c = cl.array.to_device(queue, n)
        # a = af.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())

        # n = np.random.random((5,3))
        # c = cl.array.to_device(queue, n)
        # a = af.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())

        # n = np.random.random((5,3,2))
        # c = cl.array.to_device(queue, n)
        # a = af.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())

        # n = np.random.random((5,3,2,2))
        # c = cl.array.to_device(queue, n)
        # a = af.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())
        pass

    if af.AF_NUMBA_FOUND and af.get_active_backend() == 'cuda':

        import numba
        from numba import cuda

        n = np.random.random((5,))
        c = cuda.to_device(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

        n = np.random.random((5,3))
        c = cuda.to_device(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

        n = np.random.random((5,3,2))
        c = cuda.to_device(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

        n = np.random.random((5,3,2,2))
        c = cuda.to_device(n)
        a = af.to_array(c)
        n2 = np.array(a)
        assert((n==n2).all())

_util.tests['interop'] = simple_interop
