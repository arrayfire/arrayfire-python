#!/usr/bin/python
import arrayfire as af

a = af.randu(3,3,dtype=af.u32)
b = af.constant(4, 3, 3, dtype=af.u32)
af.print_array(a)
af.print_array(b)

c = a + b
d = a
d += b

af.print_array(c)
af.print_array(d)
af.print_array(a + 2)
af.print_array(3 + a)


c = a - b
d = a
d -= b

af.print_array(c)
af.print_array(d)
af.print_array(a - 2)
af.print_array(3 - a)

c = a * b
d = a
d *= b

af.print_array(c * 2)
af.print_array(3 * d)
af.print_array(a * 2)
af.print_array(3 * a)

c = a / b
d = a
d /= b

af.print_array(c / 2.0)
af.print_array(3.0 / d)
af.print_array(a / 2)
af.print_array(3 / a)

c = a % b
d = a
d %= b

af.print_array(c % 2.0)
af.print_array(3.0 % d)
af.print_array(a % 2)
af.print_array(3 % a)

c = a ** b
d = a
d **= b

af.print_array(c ** 2.0)
af.print_array(3.0 ** d)
af.print_array(a ** 2)
af.print_array(3 ** a)

af.print_array(a < b)
af.print_array(a < 0.5)
af.print_array(0.5 < a)

af.print_array(a <= b)
af.print_array(a <= 0.5)
af.print_array(0.5 <= a)

af.print_array(a > b)
af.print_array(a > 0.5)
af.print_array(0.5 > a)

af.print_array(a >= b)
af.print_array(a >= 0.5)
af.print_array(0.5 >= a)

af.print_array(a != b)
af.print_array(a != 0.5)
af.print_array(0.5 != a)

af.print_array(a == b)
af.print_array(a == 0.5)
af.print_array(0.5 == a)

af.print_array(a & b)
af.print_array(a & 2)
c = a
c &= 2
af.print_array(c)

af.print_array(a | b)
af.print_array(a | 2)
c = a
c |= 2
af.print_array(c)

af.print_array(a >> b)
af.print_array(a >> 2)
c = a
c >>= 2
af.print_array(c)

af.print_array(a << b)
af.print_array(a << 2)
c = a
c <<= 2
af.print_array(c)

af.print_array(-a)
af.print_array(+a)
af.print_array(~a)
af.print_array(a)
