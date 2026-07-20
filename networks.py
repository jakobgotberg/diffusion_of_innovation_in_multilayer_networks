import matrix_utils as mu
import numpy as np
from adoptionopinions_model import Networks

def network_factory(n, topology):

    if topology == "random":
        return Networks(erdos_renyi(n, True), erdos_renyi(n, True))

    elif topology == "default":
        return Networks(W=ring(n), V=influencer_network(n))

    else:
        raise ValueError(f"Unknown argument: {topology}")

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

def influencer_network(n):
    A = np.zeros((n,n))

    n_influencers = int(np.log2(n))
    influencers = np.random.choice(range(n), size=n_influencers, replace=False)

    for influencer in influencers:
        n_fans = np.random.randint(int(np.log2(n)), n)
        fans = np.random.choice([i for i in range(n) if i != influencer], size=n_fans, replace=False)
        for fan in fans:
            A[fan][influencer] = 1
    
    for i in range(n):
        if sum(A[i]) == 0:
            A[i][np.random.choice([j for j in influencers if j != i])] = 1
        else:
            A[i] = A[i]/max(1, sum(A[i]))

    assert np.allclose(A @ np.ones(n), np.ones(n)), "Not row stochastic"
    return A


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


def ring(n):
    '''
    Returns undirected ring topology of 'n' vertices.
    '''
    A = np.zeros((n,n))
    for i in range(n):
        A[i][(i+1)%n] = A[(i+1)%n][i] = 1/2
    A[0][n-1] = A[n-1][0] = 1/2
    return A


def regular_lattice(n, neighbors):
    '''
    Adds one level of connectivity. TODO: define what a level is.
    '''
    assert 1 < neighbors
    assert neighbors % 2 == 0 or neighbors == n-1
    A = np.zeros((n,n))

    for _ in range(0, neighbors, 2):
        for i in range(n):
            for j in range(i+1, n+i):
                # find first zero element
                if A[i][j%n] != 0:
                    continue
                A[i][j%n] = 1
                break
            for j in range(i-1, -n+i, -1):
                if A[i][j] != 0:
                    continue
                A[i][j%n] = 1
                break

    degree = np.count_nonzero(A[0])
    assert (A @ np.ones(n) == np.array([degree] * n)).all()
    A[A != 0] = 1/degree
    return A

