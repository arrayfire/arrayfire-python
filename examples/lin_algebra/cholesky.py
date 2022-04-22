import arrayfire as af

def main():
    try:
        device = 0
        af.info()
    
        n = 5
        t = af.randu(n,n)
        inn = af.matmulNT(t,t) + af.identity(n,n)*n

        print("Running Cholesky InPlace\n")
        cin_upper = inn
        cin_lower = inn
        
        af.cholesky_inplace(cin_upper, True)
        af.cholesky_inplace(cin_lower, False)

        print(cin_upper)
        print(cin_lower)

        print("Running Cholesky Out of place\n")

        out_upper = af.cholesky(inn, True)
        out_lower = af.cholesky(inn, False)
        
        # Do we want to print the array like above? If yes this is correct.
        print(out_upper[0])
        print(out_lower[0])


    except Exception as e:
        print('Error: ',str(e))

if __name__ == '__main__':
    main()
