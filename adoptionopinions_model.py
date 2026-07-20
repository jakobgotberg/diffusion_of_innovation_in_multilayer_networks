import itertools
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields, field


@dataclass(frozen=True)
class Simulation_constants:
    n:  int
    k:  int
    lambd : np.ndarray  # (non-physical) opinion influence
    xi    : np.ndarray  # (non-physical) adoption influence
    beta  : np.ndarray  # (pysical) adoption influence from susceptible
    gamma : np.ndarray  # (pysical) adoption influence from dissatisfied
    delta : np.ndarray  # (pysical) dissatisfaction rate
    x0    : np.ndarray  # initial opinions
    alfa  : np.ndarray = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "alfa", np.ones((self.k, self.n)) - self.lambd -self.xi)

    def set_beta(self, beta):
        assert (beta.sum(axis=0) > np.zeros(self.n)).all(), f"beta out of range"
        assert (beta.sum(axis=0) < np.ones(self.n)).all(), f"beta out of range"
        object.__setattr__(self, "beta", beta)


@dataclass
class Networks:
    W : np.ndarray # pysical network
    V : np.ndarray # non-physical network


@dataclass(frozen=True)
class Initial_state:
    s : np.ndarray
    a : np.ndarray
    d : np.ndarray
    x : np.ndarray

    def __post_init__(self):
        for f in fields(self):
             m = getattr(self, f.name)
             m.setflags(write=False)

    def __call__(self):
        return [self.s, self.a, self.d, self.x]

@dataclass
class State:
    s : np.ndarray
    a : np.ndarray
    d : np.ndarray
    x : np.ndarray

    def __repr__(self):
        return f"susceptible: {self.s}\nadopters:{self.a}\ndissatisfied:{self.d}\nopinions:{self.x}"

    def __call__(self, consts, net):
        next_s = np.zeros(self.s.shape[0])
        next_a = np.zeros((self.a.shape[0], self.a.shape[1]))
        next_d = np.zeros((self.d.shape[0], self.d.shape[1]))
        next_x = np.zeros((self.x.shape[0], self.x.shape[1]))

        def ass(vector):
            eps = 1e-12
            assert (vector >= -eps).all(), f"Negative values: {vector[vector < -eps]}"
            assert (vector <= 1 + eps).all(), f"Values above 1: {vector[vector > 1 + eps]}"

        # equation (1a)
        next_s = self.s - sum(np.diag(self.s) @ np.diag(consts.beta[k]) @ np.diag(self.x[k]) @ net.W @ self.a[k] for k in range(consts.k))
        ass(next_s)

        for k in range(consts.k):
            # equation (1b)
            next_a[k] = self.a[k] \
                    + np.diag(consts.beta[k]) @ np.diag(self.x[k]) @ np.diag(self.s) @ net.W @ self.a[k] \
                    - np.diag(consts.delta[k]) @ self.a[k] \
                    + np.diag(consts.gamma[k]) @ np.diag(self.x[k]) @ sum(self.d[j] for j in range(consts.k) if j != k)
            ass(next_a[k])

            # equation (1c)
            next_d[k] = self.d[k] \
                    - sum(np.diag(consts.gamma[j]) @ np.diag(self.x[j]) for j in range(consts.k) if j != k) @ self.d[k] \
                    + np.diag(consts.delta[k]) @ self.a[k]
            ass(next_d[k])

            # equation (1d)
            next_x[k] = np.diag(consts.alfa[k]) @ consts.x0[k] \
                    + np.diag(consts.lambd[k]) @ net.V @ self.x[k] \
                    + np.diag(consts.xi[k]) @ net.W @ self.a[k]

            ass(next_x[k])

        return State(next_s, next_a, next_d, next_x)
        
        

class Simulation:

    constants : Simulation_constants
    states : list[State] = field(default_factory=list)
    net : Networks

    max_states : int = 2**18
    min_converge_check : int = 128
    converge_check_window_size : int = 32
    converge_check_atol = 1e-2

    def __init__(self, simulation_constants, initial_states, net):
        self.constants = simulation_constants
        self.states = [State(initial_states.s, initial_states.a, initial_states.d, initial_states.x)]
        self.net = net

    def __call__(self, plot=False):
        def atexit():
            if plot:
                #clear_output(wait=True)
                plt.figure()
                t = range(len(self.states))
                for k in range(self.constants.k):
                    plt.plot(t, [state.a[k].mean(axis=0) for state in self.states], label=f"a{k}")
                    plt.plot(t, [state.d[k].mean(axis=0) for state in self.states], label=f"d{k}")
                    plt.plot(t, [state.x[k].mean(axis=0) for state in self.states], label=f"x{k}")
                plt.plot(t, [state.s.mean(axis=0) for state in self.states], label=f"s")
                   
                plt.xlabel("Time")
                plt.ylabel("Ratio")
                plt.legend()
                plt.show()

        if len(self.states) <= self.max_states:
            self.states.append( self.states[-1](self.constants, self.net) )
        else:
            #print("\t\t --- Max states reached ---")
            atexit()
            return True


        if (len(self.states) > self.min_converge_check and \
                np.isclose(np.sum([state.a for state in self.states[-self.converge_check_window_size:]], axis=0), \
                self.converge_check_window_size * self.states[-1].a, atol=self.converge_check_atol).all()):
            atexit()
            return True
        return False

    def __repr__(self):
        dominant_index, dominant_ratio = max(((k, self.states[-1].a[k].mean(axis=0)) for k in range(self.constants.k)), key=lambda x: x[1])
        min_index, min_ratio = min(((k, self.states[-1].a[k].mean(axis=0)) for k in range(self.constants.k)), key=lambda x: x[1])
        return f"{dominant_index} dominates {dominant_ratio * 100:.3f}% of the population. {min_index} at {min_ratio * 100:.3f}%."



def simulation_constants_factory(n, k, x0):
    '''
    Returns the arguments needed to construct a Simulation class instance.
    If k == 2, technology 0 will be the dominant technology with (delta[0] < delta[1]).all().
    If k != 2, delta is homogenous.
    '''
    min_float = np.nextafter(0,1)

    lambd = np.random.rand(k,n)
    xi = np.random.rand(k,n)

    if k == 2:
        delta = np.zeros((k,n))
        for e in itertools.product(range(k), range(n)):
            # 'e' is the Cartesian product of k and n, i.e., all indexes of 
            # the matries
            while lambd[e[0]][e[1]] + xi[e[0]][e[1]] >= 1:
                lambd[e[0]][e[1]], xi[e[0]][e[1]] = np.random.rand(k)

        for j in range(n):
            while(True):
                delta[:,j] = np.random.rand(k)
                if (delta[0][j] < delta[1][j]):
                    break
    else:
        # hetrogeneous dissatisfaction with technologies, i.e., each community is equally likely to dislike tech i, for all i.
        dislike_range = np.linspace(0.5, 0.9, k)
        delta = np.zeros((k,n))
        for i in range(k):
            delta[i] = dislike_range[i]
        for i in range(k):
            while True:
                lambd[i] = np.random.beta(1, 10)
                xi[i]    = np.random.beta(1, 10)
                if (lambd[i] + xi[i] < 1).all() and \
                        ((max(xi[i]) / 1 - max(lambd[i])) * (1/dislike_range[i]) * (1 + 1/dislike_range[-1]) < 1):
                    break

    # The col sum of the beta matrix must be in (0,1)
    beta = np.zeros((k,n))
    for j in range(n):
        while(True):
            beta[:,j] = np.random.rand()
            if (0 < sum(beta[:,j]) < 1):
                break


    # The col sum of the gamma matrix must be in (0,1)
    gamma = np.random.rand(k,n)
    for j in range(n):
        gamma[:,j] /= max(1, gamma[:,j].sum())
    gamma = np.maximum(gamma, min_float)


    assert ((beta.T @ np.ones(k) > 0).all()) and \
        ((beta.T @ np.ones(k) < 1).all()), "Beta is not in allowed range"
    assert ((delta >= 0).all() and (delta <= 1).all())



    assert lambd.shape == (k,n) and xi.shape == (k,n)
    for i in range(k):
        assert (lambd >= 0).all() and (xi >= 0).all() and (lambd + xi < 1).all()
    if k == 2:
        assert (delta[1] > delta[0]).all()

    return Simulation_constants(
            n=n,
            k=k,
            lambd = lambd,
            xi = xi,
            beta  = beta,
            gamma = gamma,
            delta = delta,
            x0 = x0
            )

def initial_state_factory(n, k):
    min_float = np.nextafter(0,1)
    s = np.ones(n)
    a = np.zeros((k,n))
    d=np.zeros((k,n))
    x= np.random.uniform(min_float, 1, size=(k, n))

    # Go column by column, each sum(col) in (0,1), subtract the sum from s[k]
    for z in zip(range(k), np.linspace(0,n,k, endpoint=False)):
        tech = z[0]
        community = int(z[1])
        a[tech][community] = np.random.beta(10, 1)
        s[community] = 1 - a[tech][community]

    assert ((s >= 0).all() and (s <= 1).all())
    assert ((a >= 0).all() and (a <= 1).all())

    assert np.isclose(s + sum(a[j] + d[j] for j in range(k)), np.ones(n)).all(), \
            f"{s + sum(a[j] + d[j] for j in range(k))}"

    return Initial_state(s, a, d, x)
