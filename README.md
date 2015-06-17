# ArrayFire Python Bindings

[ArrayFire](https://github.com/arrayfire/arrayfire) is a high performance library for parallel computing wih an easy-to-use API. This project provides Python bindings for the ArrayFire library. It enables the users to write scientific computing code that is portable across CUDA, OpenCL and CPU devices.

## Example

```
import arrayfire as af

# Display backend information
af.info()

# Generate a uniform random array with a size of 5 elements
a = af.randu(5, 1)

# Get the minimum value of a
a_min = af.min(a)

# Print a and its minimum value
af.print_array(a)
af.print_array(a_min)
```

## Sample outputs

On an AMD GPU:

```
Using opencl backend
ArrayFire v3.0.1 (OpenCL, 64-bit Linux, build 17db1c9)
[0] AMD     : Spectre
-1- AMD     : AMD A10-7850K Radeon R7, 12 Compute Cores 4C+8G

[5 1 1 1]
0.4107
0.8224
0.9518
0.1794
0.4198


Min value of a
[1 1 1 1]
0.1794
```

On an NVIDIA GPU:

```
Using cuda backend
ArrayFire v3.0.0 (CUDA, 64-bit Linux, build 86426db)
Platform: CUDA Toolkit 7, Driver: 346.46
[0] Tesla K40c, 12288 MB, CUDA Compute 3.5
-1- GeForce GTX 750, 1024 MB, CUDA Compute 5.0

Generate a random matrix a:
[5 1 1 1]
0.7402
0.9210
0.0390
0.9690
0.9251


Min value of a
[1 1 1 1]
0.0390

Max value of a
[1 1 1 1]
0.9690
```

Fallback to CPU when CUDA and OpenCL are not availabe:

```
Using cpu backend
ArrayFire v3.0.0 (CPU, 64-bit Linux, build 86426db)

Generate a random matrix a:
[5 1 1 1]
0.0000
0.1315
0.7556
0.4587
0.5328


Min value of a
[1 1 1 1]
0.0000

Max value of a
[1 1 1 1]
```

The backend selection is automated currently. Choosing a particular backend will be made available in the future.

## Requirements

Currently, this project is tested only on Linux and OSX. You also need to have the ArrayFire C/C++ library installed on your machine. You can get it from the following sources.

- [Download and install binaries](https://arrayfire.com/download)
- [Build and install from source](https://github.com/arrayfire/arrayfire)

Please check the following links for dependencies.

- [Linux dependencies](http://www.arrayfire.com/docs/using_on_linux.htm)
- [OSX dependencies](http://www.arrayfire.com/docs/using_on_osx.htm)

## Getting started

If you have not installed the ArrayFire library in your system paths, please make sure the following environment variables are exported.

**On Linux**

```
export LD_LIBRARY_PATH=/path/to/arrayfire/lib:$LD_LIBRARY_PATH
```

**On OSX**

```
export DYLD_LIBRARY_PATH=/path/to/arrayfire/lib:$DYLD_LIBRARY_PATH
```

On both systems, to run the example, you will need to add the python bindings to your `PYTHONPATH`

```
export PYTHONPATH=/path/to/arrayfire_python/:$PYTHONPATH
```

You are now good to go!

## Note

This is a work in progress and is not intended for production use.
