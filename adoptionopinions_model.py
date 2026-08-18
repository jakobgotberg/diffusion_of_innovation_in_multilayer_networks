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


class Simulation:
    max_states : int = 2**20
    min_converge_check : int = 2**10
    converge_check_atol = 1e-8
    n_states = 1

    def __init__(self, simulation_constants, initial_states, net, window_size=2**5):
        self.constants = simulation_constants
        self.net = net
        self.states = deque(maxlen=window_size)
        self.states = [State(initial_states.s, initial_states.a, initial_states.d, initial_states.x)]
        #self.states.append(State(initial_states.s, initial_states.a, initial_states.d, initial_states.x))
        self.converge_check_window_size = window_size
        self.converge_check_freq = window_size

    def __call__(self):

        if self.n_states <= self.max_states:
            self.states.append( self.states[-1](self.constants, self.net) )
            self.n_states += 1
        else:
            print("\t\t --- Max states reached ---")
            return True

        if (self.n_states > self.min_converge_check and \
                self.n_states % self.converge_check_freq == 0 and \
                np.isclose(np.sum([state.a for state in list(self.states)], axis=0), \
                self.converge_check_window_size * self.states[-1].a, atol=self.converge_check_atol).all()):
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



def sat_eq18(k, n, delta, node_independent=False, influencers=None):
    '''
    Special case when eq18 should be true.
    '''
    min_float = np.nextafter(0,1)
    LOOP_LIMIT = 2 * n
    lambd = np.empty((k,n))
    xi    = np.empty((k,n))
    upper_limit = 1
    for i in range(k):
        for _ in range(LOOP_LIMIT):
            if not node_independent:
                lambd[i] = upper_limit * np.random.random(n)
                xi[i]    = upper_limit * np.random.random(n)
                for j in range(n):
                    while lambd[i][j] + xi[i][j] >= 1:
                        lambd[i][j] = np.random.rand()
                        xi[i][j]   = np.random.rand()
            else:
                while True:
                    xi    = equal_across(n,k)
                    lambd = equal_across(n,k)
                    if (xi + lambd < 1).all():
                        break

            if i == 0:
                set_influencer(influencers, delta, lambd, xi)
            if (max(xi[i]) / (1 - max(lambd[i]))) * (1/max(delta[i])) * (1 + 1/min(delta[:,i])) < 1:
                break
            # if the inequality is false, we limit how large xi and lambd can be and try again.
            # Obvioulsy cannot be a negative number though.
            upper_limit -= 0.01
            upper_limit = max(upper_limit, min_float)
        else:
            return False
    return (lambd, delta, xi)

def violate_eq18(k, n, delta, node_independent=False, influencers=None):
    '''
    Special case when eq18 should NOT be true.
    '''
    LOOP_LIMIT = (2 * n) // k
    lambd = np.empty((k,n))
    xi    = np.empty((k,n))
    for i in range(k):
        for _ in range(LOOP_LIMIT):
            if not node_independent:
                lambd[i] = np.random.random(n)
                xi[i]    = np.random.random(n)
                for j in range(n):
                    while lambd[i][j] + xi[i][j] >= 1:
                        lambd[i][j] = np.random.rand()
                        xi[i][j]   = np.random.rand()
            else:
                while True:
                    xi    = equal_across(n,k)
                    lambd = equal_across(n,k)
                    if (xi + lambd < 1).all():
                        break

            if i == 0:
                set_influencer(influencers, delta, lambd, xi)
            if (max(xi[i]) / (1 - max(lambd[i]))) * (1/max(delta[i])) * (1 + 1/min(delta[:,i])) >= 1:
                break
        else:
            return False
    return (lambd, delta, xi)

def set_influencer(influencers, delta, lambd, xi):
    if influencers:
        opinion_param = 1/200
        dislike_param = 1/100
        for influencer in influencers:
            lambd[0][influencer] = xi[0][influencer] = opinion_param
            delta[0][influencer] = dislike_param

def equal_across(n,k):
    scalers = np.random.rand(k)
    return np.array( [[scalers[i]] * n for i in range(k)] )

def random_simulation_constants_factory(n, k, x0, obej_eq_18=True, unique_pref=False, influencers=None):
    '''
    Returns the arguments needed to construct a Simulation class instance.
    If k == 2, technology 0 will be the dominant technology with (delta[0] < delta[1]).all().
    If k != 2, delta is homogenous.
    '''
    min_float = np.nextafter(0,1)
    lambd = np.random.rand(k,n) if unique_pref else equal_across(n,k)
    xi    = np.random.rand(k,n) if unique_pref else equal_across(n,k)

    if k == 2:
        assert False, "Not implemented"

    # The deltas are uniform: each community has the same dissatisfaction rate for each
    # technology, this makes the random generation much more likely to satisfy the inequality in 
    # equaiton 18.

    #if obej_eq_18:
    #    limits = sorted(np.random.random(2))
    #    delta_range = np.linspace(limits[0], limits[1], k)
    #    delta = np.array( [[delta_range[i]] * n for i in range(k)] )

    #    upper_limit = 1
    #    for i in range(k):
    #        for i in range(1,n): # amount of tries to generate one rows before aborting entirely
    #            if unique_pref:
    #                lambd[i] = upper_limit * np.random.random(n)
    #                xi[i]    = upper_limit * np.random.random(n)
    #                for j in range(n):
    #                    while lambd[i][j] + xi[i][j] >= 1:
    #                        lambd[i][j] = np.random.rand()
    #                        xi[i][j]   = np.random.rand()
    #            else:
    #                while True:
    #                    xi    = upper_limit * equal_across(n,k)
    #                    lambd = upper_limit * equal_across(n,k)
    #                    if (xi + lambd < 1).all():
    #                        break

    #            if i == 0:
    #                set_influencer(influencers, delta, lambd, xi)
    #            # Equation 18
    #            E = (max(xi[i]) / (1 - max(lambd[i]))) * (1/max(delta[i])) * (1 + 1/min(delta[:,i]))
    #            if E < 1:
    #                break

    #            # if the inequality is false, we limit how large xi and lambd can be and try again.
    #            # Obvioulsy cannot be a negative number though.
    #            upper_limit -= 0.01
    #            upper_limit = max(upper_limit, min_float)
    #        else:
    #            raise Exception("Unable to generate simulation constants: EQ18 satisfied branch")
    #else:
    eq18 = sat_eq18 if obej_eq_18 else violate_eq18

    if unique_pref:
        delta = np.random.rand(k,n)
    else:
        limits = sorted(np.random.random(2))
        delta_range = np.linspace(limits[0], limits[1], k)
        delta = np.array( [[delta_range[i]] * n for i in range(k)] )
    for i in range(1, n):
        if (tup := eq18(k=k, n=n, delta=delta, node_independent=unique_pref, influencers=influencers)):
            lambd, delta, xi = tup
            break
        else:
            (a,b) = (i,1) if obej_eq_18 else (1,i)
            if unique_pref:
                delta = np.random.beta(a, b, size=(k,n))
            else:
                limits = sorted(np.random.beta(a, b, size=2))
                delta_range = np.linspace(limits[0], limits[1], k)
                delta = np.array( [[delta_range[i]] * n for i in range(k)] )
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

    beta  = col_sum_control(np.random.rand(k,n)) if unique_pref else col_sum_control(equal_across(n,k))
    gamma = col_sum_control(np.random.rand(k,n)) if unique_pref else col_sum_control(equal_across(n,k))

    assert ((beta.T @ np.ones(k) > 0).all()) and \
            ((beta.T @ np.ones(k) < 1).all()), f"Beta is not in allowed range: {beta.T @ np.ones(k)}"
    assert ((delta >= 0).all() and (delta <= 1).all())
    assert lambd.shape == (k,n) and xi.shape == (k,n)
    for i in range(k):
        assert (lambd >= 0).all() and (xi >= 0).all() and (lambd + xi < 1).all()
    if k == 2:
        assert (delta[1] > delta[0]).all()

    for i in range(k):
        E = (max(xi[i]) / (1 - max(lambd[i]))) * (1/max(delta[i])) * (1 + 1/min(delta[:,i]))
        assert E >= 0

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
