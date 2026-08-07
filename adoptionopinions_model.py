import itertools, copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from dataclasses import dataclass, fields, field
from collections import deque

import numpy as np

TESTING = True

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

    def set_x0(self, x0):
        object.__setattr__(self, "x0", x0)


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

    def __call__(self, consts, net):
        next_s = np.empty_like(self.s)
        next_a = np.empty_like(self.a)
        next_d = np.empty_like(self.d)
        next_x = np.empty_like(self.x)

        def ass(vector):
            if TESTING:
                eps = 1e-12
                assert (vector >= -eps).all(), f"Negative values: {vector[vector < -eps]}"
                assert (vector <= 1 + eps).all(), f"Values above 1: {vector[vector > 1 + eps]}"

        Wa = [net.W @ self.a[k] for k in range(consts.k)]
        Vx = [net.V @ self.x[k] for k in range(consts.k)]
        total_d = self.d.sum(axis=0)
        gamma_x = consts.gamma * self.x
        total_gamma_x = gamma_x.sum(axis=0)

        # equation (1a)
        next_s = self.s - sum(self.s * consts.beta[k] * self.x[k] * Wa[k] for k in range(consts.k))
        ass(next_s)

        for k in range(consts.k):
            other_d = total_d - self.d[k]
            other_gamma_x = total_gamma_x - gamma_x[k]

            # equation (1b)
            next_a[k] = self.a[k] \
                    + consts.beta[k] * self.x[k] * self.s * Wa[k] \
                    - consts.delta[k] * self.a[k] \
                    + consts.gamma[k] * self.x[k] * other_d
            ass(next_a[k])

            # equation (1c)
            next_d[k] = self.d[k] \
                    - other_gamma_x * self.d[k] \
                    + consts.delta[k] * self.a[k]
            ass(next_d[k])

            # equation (1d)
            next_x[k] = consts.alfa[k] * consts.x0[k] \
                    + consts.lambd[k] * Vx[k] \
                    + consts.xi[k] * Wa[k]

            ass(next_x[k])

        return State(next_s, next_a, next_d, next_x)


    def update(self, consts, net):
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
    max_states : int = 2**20
    min_converge_check : int = 2**10
    converge_check_atol = 1e-8
    n_states = 1

    def __init__(self, simulation_constants, initial_states, net, window_size=2**5):
        self.constants = simulation_constants
        self.net = net
        self.states = deque(maxlen=window_size)
        self.states.append(State(initial_states.s, initial_states.a, initial_states.d, initial_states.x))
        self.converge_check_window_size = window_size
        self.converge_check_freq = window_size

    def __call__(self, plot=False):
        def atexit():
            if plot:
                #clear_output(wait=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

                t = range(len(self.states))
                for k in range(self.constants.k):
                    for n in range(self.constants.n):
                        ax1.plot(t, [state.x[k][n] for state in self.states], color="green" if k == 0 else "grey")
                ax1.set_ylabel("Ratio")
                ax1.set_title("Opinion Model")
                #ax1.legend()
                for k in range(self.constants.k):
                    #ax2.plot(t, [state.a[k].mean(axis=0) for state in self.states], label=f"a{k}" if self.constants.k < 5 or k == 0 else None)
                    #ax2.plot(t, [state.d[k].mean(axis=0) for state in self.states], label=f"d{k}" if self.constants.k < 5 or k == 0 else None)
                    for n in range(self.constants.n):
                        ax2.plot(t, [state.a[k][n] for state in self.states], color="green" if k == 0 else "grey")
                ax2.plot(t, [state.s.mean(axis=0) for state in self.states], label="Susceptible", linewidth=2, color="black")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Ratio")
                ax2.set_title("Adoption model")
                ax2.legend()
                plt.tight_layout()
                plt.show()

        if self.n_states <= self.max_states:
            self.states.append( self.states[-1](self.constants, self.net) )
            self.n_states += 1
        else:
            print("\t\t --- Max states reached ---")
            atexit()
            return True

        if (self.n_states > self.min_converge_check and \
                self.n_states % self.converge_check_freq == 0 and \
                np.isclose(np.sum([state.a for state in list(self.states)], axis=0), \
                self.converge_check_window_size * self.states[-1].a, atol=self.converge_check_atol).all()):
            atexit()
            return True
        return False


    def get_adoption_states(self):
        return [state.a for state in self.states]

    def get_adoption_steady_state(self):
        return self.states[-1].a

    def get_opinion_steady_state(self):
        return self.states[-1].x

    def get_adoption_convergence_point(self):
        return self.n_states

    def get_opinion_convergence_point(self):
        return self.n_states

    def get_physical_network(self):
        return self.net.W

    def get_virtual_network(self):
        return self.net.V


def _random_simulation_constants_factory(n, k, x0):
    '''
    Returns the arguments needed to construct a Simulation class instance.
    If k == 2, technology 0 will be the dominant technology with (delta[0] < delta[1]).all().
    If k != 2, delta is homogenous.
    '''
    min_float = np.nextafter(0,1)

    def equal_across():
        scalers = np.random.rand(k)
        return np.array( [[scalers[i]] * n for i in range(k)] )

    if k == 2:
        lambd = np.random.rand(k,n)
        xi    = np.random.rand(k,n)
        delta = np.zeros((k,n))
        for e in itertools.product(range(k), range(n)):
            # 'e' is the Cartesian product of k and n, i.e., all indexes of the matries
            while lambd[e[0]][e[1]] + xi[e[0]][e[1]] >= 1:
                lambd[e[0]][e[1]], xi[e[0]][e[1]] = np.random.rand(k)

        for j in range(n):
            while(True):
                delta[:,j] = np.random.rand(k)
                if (delta[0][j] < delta[1][j]):
                    break
    else:

        # The deltas are uniform: each community has the same dissatisfaction rate for each
        # technology, this makes the random generation much more likely to satisfy the inequality in 
        # equaiton 18.
        limits = sorted(np.random.random(2))
        delta_range = np.linspace(limits[0], limits[1], k)
        delta = np.array( [[delta_range[i]] * n for i in range(k)] )

        upper_limit = 1
        for i in range(k):
            while True:
                while True:
                    xi    = upper_limit * equal_across()
                    lambd = upper_limit * equal_across()
                    if (xi + lambd < 1).all():
                        break

                # Equation 18
                if (max(xi[i]) / 1 - max(lambd[i])) * (1/delta_range[-1]) * (1 + 1/delta_range[0]) < 1:
                    break

                # if the inequality is false, we limit how large xi and lambd can be and try again.
                # Obvioulsy cannot be a negative number though.
                upper_limit -= 0.01
                upper_limit = max(upper_limit, min_float)

    assert (lambd + xi < 1).all()


    def col_sum_control(A):
        col_sum_max = A.sum(axis=0).max()
        if  col_sum_max >= 1:
            A *= 0.99 / col_sum_max
        return A

    beta  = col_sum_control(equal_across())
    gamma = col_sum_control(equal_across())

    assert ((beta.T @ np.ones(k) > 0).all()) and \
            ((beta.T @ np.ones(k) < 1).all()), f"Beta is not in allowed range: {beta.T @ np.ones(k)}"
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

def eq18(k, n, delta):
    '''
    Special case when eq18 should not be true.
    '''
    LOOP_LIMIT = (2 * n) // k
    lambd = np.empty((k,n))
    xi    = np.empty((k,n))
    for i in range(k):
        for _ in range(LOOP_LIMIT):
            lambd[i] = np.random.random(n)
            xi[i]    = np.random.random(n)
            for j in range(n):
                while lambd[i][j] + xi[i][j] >= 1:
                    lambd[i][j] = np.random.rand()
                    xi[i][j]   = np.random.rand()

            #print((max(xi[i]) / 1 - max(lambd[i])) * (1/max(delta[i])) * (1 + 1/min(delta[:,i])))
            if (max(xi[i]) / 1 - max(lambd[i])) * (1/max(delta[i])) * (1 + 1/min(delta[:,i])) >= 1:
                break
        else:
            return False
    return (lambd, delta, xi)


def random_simulation_constants_factory(n, k, x0, obej_eq_18=True):
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
            # 'e' is the Cartesian product of k and n, i.e., all indexes of the matries
            while lambd[e[0]][e[1]] + xi[e[0]][e[1]] >= 1:
                lambd[e[0]][e[1]], xi[e[0]][e[1]] = np.random.rand(k)

        for j in range(n):
            while(True):
                delta[:,j] = np.random.rand(k)
                if (delta[0][j] < delta[1][j]):
                    break
    else:

        # The deltas are uniform: each community has the same dissatisfaction rate for each
        # technology, this makes the random generation much more likely to satisfy the inequality in 
        # equaiton 18.

        if obej_eq_18:
            limits = sorted(np.random.random(2))
            delta_range = np.linspace(limits[0], limits[1], k)
            delta = np.array( [[delta_range[i]] * n for i in range(k)] )

            upper_limit = 1
            for i in range(k):
                while True:
                    lambd[i] = upper_limit * np.random.random(n)
                    xi[i]    = upper_limit * np.random.random(n)
                    for j in range(n):
                        while lambd[i][j] + xi[i][j] >= 1:
                            lambd[i][j] = np.random.rand()
                            xi[i][j]   = np.random.rand()

                    # Equation 18
                    if (max(xi[i]) / 1 - max(lambd[i])) * (1/max(delta[i])) * (1 + 1/min(delta[:,i])) < 1:
                        break

                    # if the inequality is false, we limit how large xi and lambd can be and try again.
                    # Obvioulsy cannot be a negative number though.
                    upper_limit -= 0.01
                    upper_limit = max(upper_limit, min_float)
        else:
            delta = np.random.rand(k,n)
            for i in range(1, n):
                if (tup := eq18(k=k, n=n, delta=delta)):
                    lambd, delta, xi = tup
                    break
                else:
                    delta = np.random.beta(1, i, size=(k,n))
            else:
                raise Exception("Unable to generate simulation constants")

    a = lambd + xi
    assert (lambd + xi < 1).all(), f"{a[a > 1]}"
    # The col sum of the beta matrix must be in (0,1)
    def col_sum_control(A):
        col_sum_max = A.sum(axis=0).max()
        if  col_sum_max >= 1:
            A *= 0.99 / col_sum_max
        return A

    beta  = col_sum_control(np.random.rand(k,n))
    gamma = col_sum_control(np.random.rand(k,n))

    assert ((beta.T @ np.ones(k) > 0).all()) and \
            ((beta.T @ np.ones(k) < 1).all()), f"Beta is not in allowed range: {beta.T @ np.ones(k)}"
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


def initial_state_factory(n, k, adopters=None, influencers=None):
    a = np.zeros((k,n))
    s = np.ones(n)
    d= np.zeros((k,n))
    x = np.random.rand(k,n)

    if adopters is not None:
        for community in adopters:
            prev = a[0][community]
            a[0][community] = min(a[0][community] + np.random.beta(10,1), s[community])
            s[community] = s[community] - a[0][community] + prev
    else:
        for i in range(k, 100 * k):
            a = np.random.beta(1, i, size=(k,n))
            s = np.ones(n) - np.sum(a, axis=0)
            if (s >= 0).all() and (s <= 1).all():
                break
        else:
            raise Exception("Unable to generate initial state")

    if influencers is not None:
        for community in influencers:
            x[0][community] = np.random.beta(100, 1)

    assert ((s >= 0).all() and (s <= 1).all()), f"{s}"
    assert ((a >= 0).all() and (a <= 1).all())

    assert np.isclose(s + sum(a[j] + d[j] for j in range(k)), np.ones(n)).all(), \
            f"{s + sum(a[j] + d[j] for j in range(k))}"

    return Initial_state(s, a, d, x)
