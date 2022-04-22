#!/usr/bin/env python
import arrayfire as af
def main():
    try:
        #Skip device=argc....
        device = 0
        af.info()

        print("Running QR InPlace\n")
        inn = af.randu(5,8)
        print(inn)

        qin = inn
        tau = af.qr_inplace(qin)

        print(qin)
        print(tau)

        print("Running QR with Q and R factorization\n")
        q,r,tau = af.qr(inn)

        print(q)
        print(r)
        print(tau)

    except Exception as e:
        print("Error: ",str(e))



if __name__ == '__main__':
    main()
