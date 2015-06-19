#!/usr/bin/python
from arrayfire import (array, randu)
import arrayfire as af
import random

def calc_pi_device(samples):
    x = randu(samples)
    y = randu(samples)
    return 4 * af.sum((x * x  + y * y) < 1) / samples

if __name__ == "__main__":
    samples=1000000
    print("Monte carlo estimate of pi with %d samples: %f" % (samples, calc_pi_device(samples)))
