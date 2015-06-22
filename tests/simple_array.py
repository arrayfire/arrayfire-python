#!/usr/bin/python
import arrayfire as af
import array as host

a = af.array([1, 2, 3])
af.print_array(a)
print(a.numdims(), a.dims(), a.type())

a = af.array(host.array('d', [4, 5, 6]))
af.print_array(a)
print(a.numdims(), a.dims(), a.type())

a = af.array(host.array('l', [7, 8, 9] * 4), (2, 5))
af.print_array(a)
print(a.numdims(), a.dims(), a.type())
