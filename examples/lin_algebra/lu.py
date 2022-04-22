#!/usr/bin/env python
import arrayfire as af
def main():
    try:
        device = 0
        #af.setDevice(device)
        af.info()

        inn = af.randu(5,8)
        print(inn)

        lin = inn

        print("Running LU InPlace\n")
        # Ask if this is correct.
        pivot = af.lu_inplace(lin)
        print(lin)
        print(pivot)

        print("Running LU with Upper Lower Factorization\n")
        lower, upper, pivot = af.lu(inn)
        print(lower)
        print(upper)
        print(pivot)
    except Exception as e:
        print('Error: ', str(e))

if __name__ == '__main__':
    main()

