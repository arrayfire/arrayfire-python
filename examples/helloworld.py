#!/usr/bin/python
import arrayfire as af

# Display backend information
af.info()

# Generate a uniform random array with a size of 5 elements
a = af.randu(5, 1)

# Print a and its minimum value
af.display(a)

# Print min and max values of a
print("Minimum, Maximum: ", af.min(a), af.max(a))
