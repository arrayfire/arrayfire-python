### v3.3.20160222
- Bugfix: Fixes typo in `approx1`.
- Bugfix: Fixes typo in `hamming_matcher` and `nearest_neighbour`.
- Bugfix: Added necessary copy and lock mechanisms in interop.py.
- Example / Benchmark: New conjugate gradient benchmark.
- Feature: Added support to create arrayfire arrays from numba.
- Behavior change: af.print() only prints full arrays for smaller sizes.

### v3.3.20161126
- Fixing memory leak in array creation.
- Supporting 16 bit integer types in interop.

### v3.4.20160925
- Feature parity with ArrayFire 3.4 libs

    - [Sparse matrix support](http://arrayfire.org/arrayfire-python/arrayfire.sparse.html#module-arrayfire.sparse)
        - `create_sparse`
        - `create_sparse_from_dense`
        - `create_sparse_from_host`
        - `convert_sparse_to_dense`
        - `convert_sparse`
        - `sparse_get_info`
        - `sparse_get_nnz`
        - `sparse_get_values`
        - `sparse_get_row_idx`
        - `sparse_get_col_idx`
        - `sparse_get_storage`

    - [Random Engine support](http://arrayfire.org/arrayfire-python/arrayfire.random.html#module-arrayfire.random)
        - Three new random engines, `RANDOM_ENGINE.PHILOX`, `RANDOM_ENGINE.THREEFRY`, and `RANDOM_ENGINE.MERSENNE`.
        - `randu` and `randn` now accept an additional engine parameter.
        - `set_default_random_engine_type`
        - `get_default_random_engine`

    - New functions
        - [`scan`](http://arrayfire.org/arrayfire-python/arrayfire.algorithm.html?arrayfire.algorithm.scan#arrayfire.algorithm.scan)
        - [`scan_by_key`](http://arrayfire.org/arrayfire-python/arrayfire.algorithm.html?arrayfire.algorithm.scan#arrayfire.algorithm.scan_by_key)
        - [`clamp`](http://arrayfire.org/arrayfire-python/arrayfire.arith.html?arrayfire.arith.clamp#arrayfire.arith.clamp)
        - [`medfilt1`](http://arrayfire.org/arrayfire-python/arrayfire.signal.html#arrayfire.signal.medfilt1)
        - [`medfilt2`](http://arrayfire.org/arrayfire-python/arrayfire.signal.html#arrayfire.signal.medfilt2)
        - [`moments`](http://arrayfire.org/arrayfire-python/arrayfire.image.html#arrayfire.image.moments)
        - [`get_size_of`](http://arrayfire.org/arrayfire-python/arrayfire.library.html#arrayfire.library.get_size_of)
        - [`get_manual_eval_flag`](http://arrayfire.org/arrayfire-python/arrayfire.device.html#arrayfire.device.get_manual_eval_flag)
        - [`set_manual_eval_flag`](http://arrayfire.org/arrayfire-python/arrayfire.device.html#arrayfire.device.set_manual_eval_flag)

    - Behavior changes
        - [`eval`](http://arrayfire.org/arrayfire-python/arrayfire.device.html#arrayfire.device.eval) now supports fusing kernels.

    - Graphics updates
       - [`plot`](http://arrayfire.org/arrayfire-python/arrayfire.graphics.html#arrayfire.graphics.Window.plot) updated to take new parameters.
       - [`plot2`](http://arrayfire.org/arrayfire-python/arrayfire.graphics.html#arrayfire.graphics.Window.plot2) added.
       - [`scatter`](http://arrayfire.org/arrayfire-python/arrayfire.graphics.html#arrayfire.graphics.Window.scatter) updated to take new parameters.
       - [`scatter2`](http://arrayfire.org/arrayfire-python/arrayfire.graphics.html#arrayfire.graphics.Window.scatter2) added.
       - [`vector_field`](http://arrayfire.org/arrayfire-python/arrayfire.graphics.html#arrayfire.graphics.Window.vector_field) added.
       - [`set_axes_limits`](http://arrayfire.org/arrayfire-python/arrayfire.graphics.html#arrayfire.graphics.Window.set_axes_limits) added.

- Bug fixes

  - ArrayFire now has higher priority when numpy for mixed operations. <sup>[1](https://github.com/arrayfire/arrayfire-python/issues/69) [2](https://github.com/arrayfire/arrayfire-python/pull/71) </sup>
  - Numpy interoperability issues on Widnows. <sup>[1](https://github.com/arrayfire/arrayfire-python/issues/92)</sup>
  - Switch to a working backend by default. <sup>[1](https://github.com/arrayfire/arrayfire-python/issues/90)</sup>
  - Fixed incorrect behavior for Hermitian transpose and QR. <sup>[1](https://github.com/arrayfire/arrayfire-python/issues/91)</sup>
  - `array[0:0]` now returns empty arrays. <sup>[1](https://github.com/arrayfire/arrayfire-python/issues/26)</sup>

- Further Improvements from upstream can be read in the [arrayfire release notes](https://github.com/arrayfire/arrayfire/blob/master/docs/pages/release_notes.md).

### v3.3.20160624
- Adding 16 bit integer support
- Adding support for sphinx documentation

### v3.3.20160516
- Bugfix: Increase arrayfire's priority over numpy for mixed operations

- Added new library functions
   - `get_backend` returns backend name

### v3.3.20160510
- Bugfix to `af.histogram`

- Added missing functions / methods
   - `gaussian_kernel`

- Added new array properties
   - `Array.T` now returns transpose
   - `Array.H` now returns hermitian transpose
   - `Array.shape` now allows easier access individual dimensions

### v3.3.20160427
- Fixes to numpy interop on Windows
- Fixes issues with occasional double free
- Fixes to graphics examples

### v3.3.20160328
- Fixes to make arrayfire-python to work on 32 bit systems

### v3.3.20160320
- Feature parity with Arrayfire 3.3 libs
    - Functions to interact with arryafire's internal data structures.
        - `Array.offset`
        - `Array.strides`
        - `Array.is_owner`
        - `Array.is_linear`
        - `Array.raw_ptr`
    - Array constructor now takes `offset` and `strides` as optional parameters.
    - New visualization functions: `scatter` and `scatter3`
    - OpenCL backend specific functions:
        - `get_device_type`
        - `get_platform`
        - `add_device_context`
        - `delete_device_context`
        - `set_device_context`
    - Functions to allocate and free memory on host and device
        - `alloc_host` and `free_host`
        - `alloc_pinned` and `free_pinned`
        - `alloc_device` and `free_device`
    - Function to query which device and backend an array was created on
        - `get_device_id`
        - `get_backend_id`
    - Miscellaneous functions
        - `is_lapack_available`
        - `is_image_io_available`

- Interopability
    - Transfer PyCUDA GPUArrays using `af.pycuda_to_af_array`
    - Transfer PyOpenCL Arrays using `af.pyopencl_to_af_array`
    - New helper function `af.to_array` added to convert a different `array` to arrayfire Array.
        - This function can be used in place of `af.xyz_to_af_array` functions mentioned above.

- Deprecated functions list
    - `lock_device_ptr` is deprecated. Use `lock_array` instead.
    - `unlock_device_ptr` is deprecated. Use `unlock_array` instead.

- Bug Fixes:
    - [Boolean indexing giving faulty results](https://github.com/arrayfire/arrayfire-python/issues/68) for multi dimensional arrays.
    - [Enum types comparision failures](https://github.com/arrayfire/arrayfire-python/issues/65) in Python 2.x
    - [Support loading SO versioned libraries](https://github.com/arrayfire/arrayfire-python/issues/64) in Linux and OSX.
    - Fixed typo that prevented changing backend
    - Fixed image processing functions that accepted floating point scalar paramters.
        - Affected functions include: `translate`, `scale`, `skew`, `histogram`, `bilateral`, `mean_shift`.
### v3.2.20151224
- Bug fixes:
    - A default `AF_PATH` is set if none is found as an environment variable.

- Examples:
    - Heston model example uses a smaller data set to help run on low end GPUs.

### v3.2.20151214
- Bug fixes:
    - `get_version()` now returns ints instead of `c_int`
    - Fixed bug in `tests/simple/device.py`

- The module now looks at additional paths when loading ArrayFire libraries.
    - Link to the wiki is provided when `ctypes.cdll.LoadLibrary` fails.

- New function:
    - `info_str()` returns information similar to `info()` as a string.

- Updated README.md with latest instructions

### v3.2.20151211
- Feature parity with ArrayFire 3.2 libs
    - New computer vision functions: `sift`, `gloh`, `homography`
    - New graphics functions: `plot3`, `surface`
    - Functions to load and save native images: `load_image_native`, `save_image_native`
    - Use `unified` backend when possible

- Added missing functions
    - `eval`, `init`, `convolve2_separable`, `as_type` method
    - `cuda` backend specific functions
    - `opencl` backend specific functions
    - `timeit` function to benchmark arrayfire functions

- Added new examples
    - getting_started: `intro`, `convolve`
    - benchmarks: `bench_blas`, `bench_fft`
    - financial: `monte_carlo_options`, `black_scholes`, `heston_model`
    - graphics: `fractal`, `histogram`, `plot3d`, `conway`, `surface`

- Bug fixes
    - Fixed bug when array types were being reported incorrectly
    - Fixed various bugs in graphics functions

### v3.1.20151111
- Feature parity with ArrayFire 3.1 libs
- Ability to interop with other python libs
- Ability to extract raw device pointers
- Load and Save arrays from disk
- Improved `__repr__` support

### v3.0.20150914
- Feature parity with ArrayFire 3.0 libs
- Ability to switch all backends
- Supports both python2 and python3
