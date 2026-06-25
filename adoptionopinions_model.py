import random, argparse, time, csv, os, statistics, sys, itertools
from dataclasses import dataclass
import numpy as np
import matrix_utils as mu
import random_graphs
import sys, tty, termios



@dataclass
class State:
    n:  int
    K:  int
    W_p : np.ndarray # physical layer (adaption)
    W_np: np.ndarray # non-physical layer (opinion)
    s: np.ndarray # n-vector of floats
    # All arrays below are K times n floats
    a: np.ndarray
    d: np.ndarray
    x: np.ndarray
    x0: np.ndarray
    '''
    homogenous system, all scalars are identical, i.e., vectors below are of K dim.
    '''
    lambd : np.ndarray  # (non-physical) opinion influence
    xi    : np.ndarray  # (non-physical) adoption influence
    beta  : np.ndarray  # (pysical) adoption influence from susceptible
    gamma : np.ndarray  # (pysical) adoption influence from dissatisfied
    delta : np.ndarray  # (pysical) dissatisfaction rate
    t:  int = 0

    def adoptions(self):
        def to_list(c):
            return [ (c[k] @ np.ones(self.n)) / self.n for k in range(self.K) ]
        return to_list(self.a), to_list(self.d)

    def tick(self, test=False):

        # TODO: improve/remove these functions
        def B(k):
            # diagonal matrix of the k:th row of the beta matrix
            return np.diag(self.beta[k])

        def X(k):
            return np.diag(self.x[k])
        def D(k):
            return np.diag(self.delta[k])
        def T(k):
            return np.diag(self.gamma[k])
        def L(k):
            return np.diag(self.lambd[k])
        def Xi(k):
            return np.diag(self.xi[k])

        def opinion_inf(k):
            return self.W_np @ self.x[k]

        def adoption_inf(k):
            return self.W_p @ self.a[k]

        def s_next():
            #TODO: list decomp
            ret = 0
            for k in range(self.K):
                ret += np.diag(self.s) @ B(k) @ X(k) @ adoption_inf(k)
            return self.s - ret

        def a_next(k):
            # TODO: make 'dissat' a list decomp
            dissat = np.zeros_like(self.d[k])
            for i in range(self.K):
                if i == k:
                    continue
                dissat += self.d[i]

            return self.a[k] - (D(k) @ self.a[k]) + T(k) @ X(k) @ dissat + B(k) @ X(k) @ np.diag(self.s) @ adoption_inf(k)

        def d_next(k):
            return self.d[k] -(sum(T(i) @ X(i) @ self.d[k] for i in range(self.K) if i != k)) + D(k) @ self.a[k]

        def x_next(k):
            return (np.eye(self.n) - L(k) - Xi(k)) @ self.x0[k] + L(k) @ opinion_inf(k) + Xi(k) @ adoption_inf(k)

        def check_nonnegative(v):
            if (v < 0).any():
                raise Exception(f"Value underflow: {v}")
            return v


        self.s = check_nonnegative(s_next())
        for k in range(self.K):
            self.a[k] = check_nonnegative(a_next(k))
            try:
                self.d[k] = check_nonnegative(d_next(k))
            except Exception:
                adoption_v = D(k) @ self.a[k]
                dissatisfaction_v = sum(T(i) @ X(i) @ self.d[k] for i in range(self.K) if i != k)
                for j in range(self.n):
                    print(f"{self.d[k][j] + adoption_v[j]} >= {dissatisfaction_v[j]} -> {self.d[k][j] + adoption_v[j] >= dissatisfaction_v[j]}")
                sys.exit(-1)
            self.x[k] = check_nonnegative(x_next(k))

        self.t += 1


def show_state(state):
    print(f't: {state.t}')
    a, d = state.adoptions()
    for k in range(state.K):
        print(f'a[{k}]: {a[k]:.3f}')
        print(f'd[{k}]: {d[k]:.3f}')
    print()

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def random_state(n, k):
    min_float = np.nextafter(0,1)

    def susceptible_and_adopters():
        a = np.zeros((k,n))
        s = np.ones(n)
        # Go column by column, each sum(col) in (0,1), subtract the sum from s[k]
        for j in range(n):
            while(True):
                a[:,j] = 1/10 * np.random.standard_exponential(k)
                if (0 < sum(a[:,j]) <= 1):
                    s[j] = s[j] - sum(a[:,j])
                    break
        return s, a

    def opinion_rates():
        '''
        Beta in k x n, the col sum must be in (0,1)
        '''
        B = np.zeros((k,n))
        for j in range(n):
            while(True):
                B[:,j] = (1/n) * np.random.rand(k)
                if (0 < sum(B[:,j]) < 1):
                    break
        return B

    def opinion_scalers():
        '''
        lambd_i, xi_i >= 0, lambd_i + xi_i < 1
        '''
        l = np.random.rand(k,n)
        xi = np.random.rand(k,n)
        for e in itertools.product(range(k), range(n)):
            # 'e' is the Cartesian product of k and n, i.e., all indexes of 
            # the matries
            while (l[e[0]][e[1]] + xi[e[0]][e[1]] >= 1):
                l[e[0]][e[1]], xi[e[0]][e[1]] = np.random.rand(2)
        return l, xi


    lambd, xi = opinion_scalers()
    susceptible, adopters = susceptible_and_adopters()


    state = State(
    n=n,
    K=k,
    W_p=random_graphs.erdos_renyi(n, must_be_irreducible=True),
    W_np=random_graphs.erdos_renyi(n, must_be_irreducible=True),
    s=susceptible,
    a=adopters,
    d=np.zeros((k,n)),
    x=np.zeros((k,n)),
    x0=np.random.uniform(min_float, 1, size=(k, n)),
    lambd = lambd,
    xi = xi,
    beta  = opinion_rates(),
    gamma = np.random.uniform(min_float, 0.1, (k,n)),

    delta = np.random.uniform(0,1,(k,n))
            )

    assert ((state.beta.T @ np.ones(k) > 0).all()) and \
        ((state.beta.T @ np.ones(k) < 1).all()), "Beta is not in allowed range"
    assert ((state.delta >= 0).all() and (state.delta <= 1).all())
    assert ((state.s >= 0).all()     and (state.s <= 1).all())
    assert ((state.a >= 0).all()     and (state.a <= 1).all())
    assert state.lambd.shape == (k,n) and state.xi.shape == (k,n)
    for i in range(k):
        assert (state.lambd >= 0).all() and (state.xi >= 0).all() and (state.lambd + state.xi < 1).all()
    assert np.allclose(state.W_p @ np.ones(n),  np.ones(n)), "W_p is not row-stoc"
    assert np.allclose(state.W_np @ np.ones(n), np.ones(n)), "W_np is not row-stoc"
    assert mu.irreducible(state.W_p), "W_p not strongly connected"
    assert mu.irreducible(state.W_np), "W_np not strongly connected"
    return state


def main(pid):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--scenario", type=int, default=0)
    p.add_argument("--test",action="store_true")
    a = p.parse_args()

    state = random_state(a.n, a.K)
    if a.test:
        state.tick(a.test)

    show_state(state)
    while(getch() != 'q'):
        state.tick()
        show_state(state)


if __name__ == "__main__":
    t_program_start = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
