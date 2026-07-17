import matrix_utils as mu
import numpy as np

def erdos_renyi(n, must_be_irreducible=False):
    # I belive that the irreducibility is independent of the diagonal, but I am not sure.
    while True:
        A = np.zeros((n,n))
        for i in range(n):
            while not np.isclose(sum(A[i]), 1.0):
                for j in np.random.permutation(n):
                    if j == i:
                        continue
                    A[i][j] = np.random.uniform()
                    if sum(A[i]) > 1:
                        A[i] = A[i] / sum(A[i])
                        break

        if must_be_irreducible:
            if mu.irreducible(np.abs(A)):
                return A
            else:
                continue

        return A 
