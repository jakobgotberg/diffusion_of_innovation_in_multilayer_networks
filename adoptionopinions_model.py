import random, argparse, time, csv, os, statistics
from dataclasses import dataclass
import numpy as np

import sys, tty, termios

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
            return [ (c[k] @ np.ones(self.n)) / self.n for k in range(self.K)]
        return to_list(self.a), to_list(self.d)

    def tick(self):
        # TODO: improve/remove these functions
        def B(k):
            ret = np.diag([self.beta[k]] * self.n)
            return ret

        def X(k):
            return np.diag([self.x[k]] * self.n)
        def D(k):
            return np.diag([self.delta[k]] * self.n)
        def T(k):
            return np.diag([self.gamma[k]] * self.n)
        def L(k):
            ret = np.diag([self.lambd[k]] * self.n)
            return ret

        def Xi(k):
            return np.diag([self.xi[k]] * self.n)

        def opinion_inf(k):
            return self.W_np @ self.x[k]
        def adoption_inf(k):
            return self.W_p @ self.a[k]

        def s_next():
            #TODO: list decomp
            ret = 0
            for k in range(self.K):
                ret += np.diag(self.s) @ B(k) @ X(k) @ adoption_inf(k)
            return -ret

        def a_next(k):
            # TODO: make 'dissat' a list decomp
            dissat = np.zeros_like(self.d[k])
            for i in range(self.K):
                if i == k:
                    continue
                dissat += self.d[i]

            return -(D(k) @ self.a[k]) + T(k) @ X(k) @ dissat + B(k) @ X(k) @ np.diag(self.s) @ adoption_inf(k)

        def d_next(k):
            # TODO: list decomp
            to_adopt = np.zeros_like(T(k))
            for i in range(self.K):
                if i == k:
                    continue
                to_adopt += T(i) @ X(i)
            return -(to_adopt @ self.d[k]) + D(k) @ self.a[k]

        def x_next(k):
            return L(k) @ opinion_inf(k) + Xi(k) @ adoption_inf(k)

        self.s = self.s + s_next()
        for k in range(self.K):
            self.a[k] = self.a[k] + a_next(k)
            self.d[k] = self.d[k] + d_next(k)
            self.x[k] = (np.eye(self.n) - L(k) - Xi(k)) @ self.x0[k] + x_next(k)

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

def random_state(k):
    min_float = np.nextafter(0,1)
    n = 3

    def susceptible_and_adopters():
        a = np.zeros((k,n))
        s = np.ones(n)
        for i in range(a.shape[1]):
            while((r := np.random.standard_exponential()) > 0.1):
                pass
            a[np.random.randint(k)][i] = r
            s[i] -= r
        return s, a

    def opinion_scalers():
        '''
        lambd_i, xi_i >= 0, lambd_i + xi_i < 1
        '''
        v = np.random.rand(2, k)
        for i in range(v.shape[1]):
            a,b = v[:,i]
            while (a + b >= 1):
                a,b = np.random.rand(2)
            v[:,i] = (a,b)
        lambd = v[0]
        xi = v[1]
        return lambd, xi


    lambd, xi = opinion_scalers()
    susceptible, adopters = susceptible_and_adopters()


    state = State(
    n=n,
    K=k,
    W_p=np.array([
        [0,1,0],
        [0.5,0,0.5],
        [0,1,0]]),

    W_np=np.array([
        [0,0,1],
        [0,0,1],
        [0.5,0.5,0]]),
    s=susceptible,
    a=adopters,
    d=np.zeros((k,n)),
    x=np.zeros((k,n)),
    x0=np.random.uniform(min_float, 1, size=(k, n)),
    lambd = lambd,
    xi = xi,
    beta  = np.random.uniform(min_float, 1, k),
    gamma = np.random.uniform(min_float, 1, k),

    delta = np.random.uniform(0,1,k)
            )

    assert ((state.delta >= 0).all() and (state.delta <= 1).all())
    assert ((state.s >= 0).all()     and (state.s <= 1).all())
    assert ((state.a >= 0).all()     and (state.a <= 1).all())
    assert state.lambd.shape == (k,) and state.xi.shape == (k,)
    for i in range(k):
        assert (state.lambd[i] >= 0) and (state.xi[i] >= 0) and (state.lambd[i] + state.xi[i] < 1)
    assert (state.W_p @ np.ones(n) == np.ones(n)).all(), "W_p not row-stoc"
    assert (state.W_np @ np.ones(n) == np.ones(n)).all(), "W_np not row-stoc"
    assert irreducible(state.W_p), "W_p not strongly connected"
    return state


def main(pid):
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--V", type=int, default=2)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--file-name", default="oriented_hypergraph_data")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--no-output",action="store_true")
    a = p.parse_args()

    state = random_state(a.K)
    show_state(state)
    while(getch() != 'q'):
        state.tick()
        show_state(state)


if __name__ == "__main__":
    t_program_start = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
