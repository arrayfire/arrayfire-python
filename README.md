# ArrayFire Python Bindings

[ArrayFire](https://github.com/arrayfire/arrayfire) is a high performance library for parallel computing with an easy-to-use API. It enables users to write scientific computing code that is portable across CUDA, OpenCL and CPU devices. This project provides Python bindings for the ArrayFire library.

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

## Getting started
ArrayFire can be installed from a variety of sources. [Pre-built wheels](https://repo.arrayfire.com/python/wheels/3.8.0/) are available for a number of systems and toolkits. These will include a distribution of the ArrayFire libraries. Currently, only the python wrapper is available on PyPI. Wrapper-only installations will require a separate installation of the ArrayFire C/C++ libraries. 
You can get the ArrayFire C/C++ library from the following sources:

- [Download and install binaries](https://arrayfire.com/download)
- [Build and install from source](https://github.com/arrayfire/arrayfire)


**Install the last stable version of python wrapper:**  
```
pip install arrayfire
```

**Install a pre-built wheel for a specific CUDA toolkit version:**
```
pip install arrayfire==3.8.0+cu112 -f https://repo.arrayfire.com/python/wheels/3.8.0/
# Replace the +cu112 local version with the desired toolkit
```

**Install the development source distribution:**

```
pip install git+git://github.com/arrayfire/arrayfire-python.git@master
```

**Installing offline:**

```
cd path/to/arrayfire-python
python setup.py install
```
Rather than installing and building ArrayFire elsewhere in the system, you can also build directly through python by first setting the `AF_BUILD_LOCAL_LIBS=1` environment variable. Additional setup will be required to build ArrayFire, including satisfying dependencies and further CMake configuration. Details on how to pass additional arguments to the build systems can be found in the [scikit-build documentation.](https://scikit-build.readthedocs.io/en/latest/)

**Post Installation:**

If you are not using one of the pre-built wheels, you may need to ensure arrayfire-python can find the installed arrayfire libraries. Please follow [these instructions](https://github.com/arrayfire/arrayfire-python/wiki) to ensure that arrayfire-python can find the arrayfire libraries.

To run arrayfire tests, you can run the following command from command line.

```
python -m arrayfire.tests
```

## Communication

* [Slack Chat](https://join.slack.com/t/arrayfire-org/shared_invite/MjI4MjIzMDMzMTczLTE1MDI5ODg4NzYtN2QwNGE3ODA5OQ)
* [Google Groups](https://groups.google.com/forum/#!forum/arrayfire-users)

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
