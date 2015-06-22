#!/usr/bin/python
import arrayfire as af

a = af.randu(5,5)
b = af.randu(5,5)

af.print_array(af.matmul(a,b))
af.print_array(af.matmul(a,b,af.AF_MAT_TRANS))
af.print_array(af.matmul(a,b,af.AF_MAT_NONE, af.AF_MAT_TRANS))

b = af.randu(5,1)
af.print_array(af.dot(a,b))

af.print_array(af.transpose(a))

af.transpose_inplace(a)
af.print_array(a)
