import numpy as np

def irreducible(B):
    '''
    Checks irriducibility of nonnegative matrices.
    If sum of M^k, k < n-1 is positive, the whole sum
    must be positive.
    '''
    n = B.shape[0]

    S = np.zeros((n, n), dtype=float)
    P = np.eye(n, dtype=float)

    # sum_{k = 0}^{n-1} 
    S += P
    for _ in range(1, B.shape[0]):
        P = P @ B
        S += P 
        if (S > 0).all():
            return True

    return False
