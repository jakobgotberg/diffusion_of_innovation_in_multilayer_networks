import random, argparse, time, csv, os, statistics, sys, itertools, copy
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import numpy as np
import matrix_utils as mu
import random_graphs
import sys, tty, termios


@dataclass(frozen=True)
class Simulation_constants:
    n:  int
    K:  int
    W: np.ndarray # physical layer (adaption)
    V: np.ndarray # non-physical layer (opinion)
    lambd : np.ndarray  # (non-physical) opinion influence
    xi    : np.ndarray  # (non-physical) adoption influence
    beta  : np.ndarray  # (pysical) adoption influence from susceptible
    gamma : np.ndarray  # (pysical) adoption influence from dissatisfied
    delta : np.ndarray  # (pysical) dissatisfaction rate
    x0    : np.ndarray  # initial opinions
    alfa  : np.ndarray = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "alfa", np.ones((self.K, self.n)) - self.lambd -self.xi)

@dataclass(eq=False)
class State:
    s : np.ndarray
    a : np.ndarray
    d : np.ndarray
    x : np.ndarray

    def __repr__(self):
        return f"susceptible: {self.s}\nadopters:{self.a}\ndissatisfied:{self.d}\nopinions:{self.x}"

    def __call__(self, consts):
        next_s = np.zeros(self.s.shape[0])
        next_a = np.zeros((self.a.shape[0], self.a.shape[1]))
        next_d = np.zeros((self.d.shape[0], self.d.shape[1]))
        next_x = np.zeros((self.x.shape[0], self.x.shape[1]))

        # equation (1a)
        next_s = self.s - sum(np.diag(self.s) @ np.diag(consts.beta[k]) @ np.diag(self.x[k]) @ consts.W @ self.a[k] for k in range(consts.K))
        if (next_s < 0).any():
            raise Exception("s", next_s)

        for k in range(consts.K):
            # equation (1b)
            next_a[k] = self.a[k] \
                    + np.diag(consts.beta[k]) @ np.diag(self.x[k]) @ np.diag(self.s) @ consts.W @ self.a[k] \
                    - np.diag(consts.delta[k]) @ self.a[k] \
                    + np.diag(consts.gamma[k]) @ np.diag(self.x[k]) @ sum(self.d[j] for j in range(consts.K) if j != k)
            if (next_a[k] < 0).any():
                raise Exception("a", next_a[k])

            # equation (1c)
            next_d[k] = self.d[k] \
                    - sum(np.diag(consts.gamma[j]) @ np.diag(self.x[j]) for j in range(consts.K) if j != k) @ self.d[k] \
                    + np.diag(consts.delta[k]) @ self.a[k]
            if (next_d[k] < 0).any():
                raise Exception("d", next_d[k])

            # equation (1d)
            next_x[k] = consts.alfa[k] @ consts.x0[k] \
                    + np.diag(consts.lambd[k]) @ consts.V @ self.x[k] \
                    + np.diag(consts.xi[k]) @ consts.W @ self.a[k]
            if (next_x[k] < 0).any():
                raise Exception("x", next_x[k])

        return State(next_s, next_a, next_d, next_x)
        
        

class Simulation:

    constants : Simulation_constants
    states : list[State] = field(default_factory=list)
    max_states : int = 2048
    min_converge_check : int = 128
    converge_check_window_size : int = 32
    converge_check_atol = 1e-2

    def __init__(self, n, k, W, V, lambd, xi, beta, gamma, delta, s, a, d, x):
        self.constants = Simulation_constants(n=n, K=k, W=W, V=V, lambd=lambd, xi=xi, beta=beta, gamma=gamma, delta=delta, x0=x)
        self.states = [(State(s,a,d,x))]

    def __call__(self, plot=False):
        if len(self.states) <= self.max_states:
            self.states.append( self.states[-1](self.constants) )
        else:
            return False

        if plot:
            clear_output(wait=True)
            plt.figure()
            t = range(len(self.states))
            for k in range(self.constants.K):
                plt.plot(t, [state.a[k].mean(axis=0) for state in self.states], label=f"a{k}")
               
            plt.xlabel("Time")
            plt.ylabel("Ratio")
            plt.legend()
            plt.show()
            #clear_output(wait=True)

        if (len(self.states) > self.min_converge_check and \
                np.isclose(np.sum([state.a for state in self.states[-self.converge_check_window_size:]], axis=0), \
                self.converge_check_window_size * self.states[-1].a, atol=self.converge_check_atol).all()):
            return True
        return False

    def __repr__(self):
        dominant_index, dominant_ratio = max(((k, self.states[-1].a[k].mean(axis=0)) for k in range(self.constants.K)), key=lambda x: x[1])
        return f"Technology nr {dominant_index} dominates, {dominant_ratio * 100:.3f}% of the population is using the technology."

def simulation_factory(n, k):
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
    beta  = opinion_rates()

    W=random_graphs.erdos_renyi(n, must_be_irreducible=True)
    V=random_graphs.erdos_renyi(n, must_be_irreducible=True)
    d=np.zeros((k,n))
    x= np.random.uniform(min_float, 1, size=(k, n))
    gamma = np.random.uniform(min_float, 0.1, (k,n))
    delta = np.random.rand(k,n)


    assert ((beta.T @ np.ones(k) > 0).all()) and \
        ((beta.T @ np.ones(k) < 1).all()), "Beta is not in allowed range"
    assert ((delta >= 0).all() and (delta <= 1).all())
    assert ((susceptible >= 0).all()     and (susceptible <= 1).all())
    assert ((adopters >= 0).all()     and (adopters <= 1).all())
    assert lambd.shape == (k,n) and xi.shape == (k,n)
    for i in range(k):
        assert (lambd >= 0).all() and (xi >= 0).all() and (lambd + xi < 1).all()
    assert np.allclose(W @ np.ones(n),  np.ones(n)), "W is not row-stoc"
    assert np.allclose(V @ np.ones(n), np.ones(n)), "V is not row-stoc"
    assert mu.irreducible(W), "W not strongly connected"
    assert mu.irreducible(V), "V not strongly connected"

    return Simulation(
    n=n,
    k=k,
    W=W,
    V=V,
    s=susceptible,
    a=adopters,
    d=d,
    x= x,
    lambd = lambd,
    xi = xi,
    beta  = beta,
    gamma = gamma,

    delta = np.random.rand(k,n)
            )

