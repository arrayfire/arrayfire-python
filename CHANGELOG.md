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
