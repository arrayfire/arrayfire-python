# ArrayFire Python Bindings

[ArrayFire](https://github.com/arrayfire/arrayfire) is a high performance library for parallel computing with an easy-to-use API. It enables users to write scientific computing code that is portable across CUDA, OpenCL and CPU devices. This project provides Python bindings for the ArrayFire library.

## Status
|  OS     | Tests   |
|:-------:|:-------:|
| Linux   | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-linux)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-linux/)      |
| Windows | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-windows)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-windows/)  |
| OSX     | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-osx)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-osx/)          |
| Linux on ARM | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-wrappers/python-tegrax1)](http://ci.arrayfire.org/view/All/job/arrayfire-wrappers/job/python-tegrak1/)|

## Documentation

Documentation for this project can be found [over here](http://arrayfire.org/arrayfire-python/).

## Example

```python
# Monte Carlo estimation of pi
def calc_pi_device(samples):
    # Simple, array based API
    # Generate uniformly distributed random numers
    x = af.randu(samples)
    y = af.randu(samples)
    # Supports Just In Time Compilation
    # The following line generates a single kernel
    within_unit_circle = (x * x + y * y) < 1
    # Intuitive function names
    return 4 * af.count(within_unit_circle) / samples
```


Choosing a particular backend can be done using `af.set_backend(name)`  where name is either "_cuda_", "_opencl_", or "_cpu_". The default device is chosen in the same order of preference.

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

To run arrayfire tests, you can run the following command from command line.

```
python -m arrayfire.tests
```

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
