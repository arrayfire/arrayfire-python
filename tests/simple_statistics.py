#!/usr/bin/python
import arrayfire as af

a = af.randu(5, 5)
b = af.randu(5, 5)
w = af.randu(5, 1)

af.print_array(af.mean(a, dim=0))
af.print_array(af.mean(a, weights=w, dim=0))
print(af.mean(a))
print(af.mean(a, weights=w))

af.print_array(af.var(a, dim=0))
af.print_array(af.var(a, isbiased=True, dim=0))
af.print_array(af.var(a, weights=w, dim=0))
print(af.var(a))
print(af.var(a, isbiased=True))
print(af.var(a, weights=w))

af.print_array(af.stdev(a, dim=0))
print(af.stdev(a))

af.print_array(af.var(a, dim=0))
af.print_array(af.var(a, isbiased=True, dim=0))
print(af.var(a))
print(af.var(a, isbiased=True))

af.print_array(af.median(a, dim=0))
print(af.median(w))

print(af.corrcoef(a, b))
