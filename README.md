# ArrayFire Python Bindings

[ArrayFire](https://github.com/arrayfire/arrayfire) is a high performance library for parallel computing with an easy-to-use API. It enables users to write scientific computing code that is portable across CUDA, OpenCL and CPU devices. This project provides Python bindings for the ArrayFire library.

## Status
|  OS     | Tests   |
|:-------:|:-------:|
| Linux   | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-linux)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-linux/)      |
| Windows | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-windows)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-windows/)  |
| OSX     | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-osx)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-osx/)          |

## Example

```python
import arrayfire as af

# Display backend information
af.info()

# Generate a uniform random array with a size of 5 elements
a = af.randu(5, 1)

# Print a and its minimum value
af.display(a)

# Print min and max values of a
print("Minimum, Maximum: ", af.min(a), af.max(a))
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

Minimum, Maximum:  0.17936542630195618 0.9517996311187744
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

Minimum, Maximum:  0.039020489901304245 0.9689629077911377
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

Minimum, Maximum:  7.825903594493866e-06 0.7556053400039673
```

Choosing a particular backend can be done using `af.backend.set( backend_name )`  where backend_name can be one of: "_cuda_", "_opencl_", or "_cpu_". The default device is chosen in the same order of preference.

## Requirements

Currently, this project is tested only on Linux and OSX. You also need to have the ArrayFire C/C++ library installed on your machine. You can get it from the following sources.

- [Download and install binaries](https://arrayfire.com/download)
- [Build and install from source](https://github.com/arrayfire/arrayfire)

Please check the following links for dependencies.

- [Linux dependencies](http://www.arrayfire.com/docs/using_on_linux.htm)
- [OSX dependencies](http://www.arrayfire.com/docs/using_on_osx.htm)

## Getting started

**Install the last stable version:**

```
pip install arrayfire
```

**Install the development version:**

```
pip install git+git://github.com/arrayfire/arrayfire-python.git@devel
```

**Installing offline**

```
cd path/to/arrayfire-python
python setup.py install
```

**Post Installation**

Please follow [these instructions](https://github.com/arrayfire/arrayfire-python/wiki) to ensure the arrayfire-python can find the arrayfire libraries.

## Acknowledgements

The ArrayFire library is written by developers at [ArrayFire](http://arrayfire.com) LLC
with [contributions from several individuals](https://github.com/arrayfire/arrayfire_python/graphs/contributors).

The developers at ArrayFire LLC have received partial financial support
from several grants and institutions. Those that wish to receive public
acknowledgement are listed below:

<!--
The following section contains acknowledgements for grant funding. In most
circumstances, the specific phrasing of the text is mandated by the grant
provider. Thus these acknowledgements must remain intact without modification.
-->

### Grants

This material is based upon work supported by the DARPA SBIR Program Office
under Contract Numbers W31P4Q-14-C-0012 and W31P4Q-15-C-0008.
Any opinions, findings and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect the views of
the DARPA SBIR Program Office.
