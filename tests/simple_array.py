#!/usr/bin/python
import arrayfire as af
import array as host

a = af.array([1, 2, 3])
af.print_array(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())


a = af.array(host.array('d', [4, 5, 6]))
af.print_array(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

a = af.array(host.array('l', [7, 8, 9] * 4), (2, 5))
af.print_array(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())
