import matrix_utils as mu
import numpy as np
from adoptionopinions_model import Networks

def row_stochatsic_test(func):
    def assert_rs(*args, **kwargs):
        n = args[0]
        A = func(*args, **kwargs)
        assert np.allclose(A @ np.ones(n), np.ones(n)), f"Not row stochastic"
        return A
    return assert_rs

@row_stochatsic_test
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

@row_stochatsic_test
def influencer_network(n, influencers):
    A = np.zeros((n,n))
    n_influencers = len(influencers)

    for influencer in influencers:
        A.T[influencer] = 1/n_influencers
    
    return A


@row_stochatsic_test
def random_complete(n):
    A = np.zeros((n,n))
    for r in range(n):
        while not np.isclose(sum(A[r]), 1.0):
            A[r] = np.random.random(n)
            A[r][r] = 0
            if sum(A[r]) > 1:
                A[r] = A[r] / sum(A[r])
                break
    return A


@row_stochatsic_test
def ring(n):
    '''
    Returns undirected ring topology of 'n' vertices.
    '''
    A = np.zeros((n,n))
    for i in range(n):
        A[i][(i+1)%n] = A[(i+1)%n][i] = 1/2
    A[0][n-1] = A[n-1][0] = 1/2
    assert np.allclose(A @ np.ones(n), np.ones(n)), "Not row stochastic"
    return A


@row_stochatsic_test
def regular_lattice(n, neighbors, self_loop=False):
    '''
    Adds one level of connectivity. TODO: define what a level is.
    '''
    assert neighbors % 2 == 0 or neighbors == n, f"{neighbors}"
    A = np.zeros((n,n))

    for _ in range(0, neighbors, 2):
        for i in range(n):
            for j in range(i+1, n+i):
                # find first zero element
                if A[i][j%n] != 0:
                    continue
                A[i][j%n] = 1/n if self_loop else 1
                break
            for j in range(i-1, -n+i, -1):
                if A[i][j] != 0:
                    continue
                A[i][j%n] = 1/n if self_loop else 1
                break

    if self_loop:
        np.fill_diagonal(A, 1 - np.sum(A[0]))
    else:
        A[A != 0] = 1/(np.count_nonzero(A)/n)
    return A

