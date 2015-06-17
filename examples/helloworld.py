#!/usr/bin/python
import arrayfire as af

af.info()

print('\nGenerate a random matrix a:')
a = af.randu(5, 1)
af.print_array(a)

print('\nMin value of a')
a_min = af.min(a)
af.print_array(a_min)

print('\nMax value of a')
a_max = af.max(a)
af.print_array(a_max)
