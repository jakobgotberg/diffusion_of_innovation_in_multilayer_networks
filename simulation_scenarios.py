import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output, display
import adoptionopinions_model as aom

class Simulation:

    def __init__(self, n=2, k=2, scenario=0, history_len=50):
        def dominant_technology():
            '''
            delta^2 > delta^1 for all communities
            For K = 2 only
            '''
            d = np.zeros((2,n))
            for j in range(n):
                while(True):
                    d[:,j] = np.random.rand(2)
                    if (d[0][j] < d[1][j]):
                        break
            return d
        self.history_len = history_len
        self.history = deque(maxlen=history_len)
        self.n = n
        self.k = k
        self.scenario = scenario
        self.state = aom.random_state(n,k)
        if scenario > 0:
            self.state.delta = dominant_technology()

        self.a0 = copy.deepcopy(self.state.a)
        self.d0 = copy.deepcopy(self.state.d)
        self.s0 = copy.deepcopy(self.state.s)

        self.t_series = []
        self.a_series = [None] * self.state.K
        self.d_series = [None] * self.state.K
        for k in range(self.state.K):
            self.a_series[k] = []
            self.d_series[k] = []

    def check_adoption_dominance(self):
        return self.state.a[0] > self.state.a[1]

    def reset_state(self):
        self.state.t = 0
        self.state.a = self.a0
        self.state.d = self.d0
        self.state.s = self.s0
        self.t_series = []
        self.a_series = [[] for _ in range(self.state.K)]
        self.d_series = [[] for _ in range(self.state.K)]

    def set_partial_dominance(self, I=0):
        '''
        Scenario 2, delta^k_i < delta^k_j -> delta^k_i > delta^k_j for 
        'I' random communities.
        '''
        if I == 0:
            return
        assert self.scenario == 2 and self.k == 2

        def flip(j):
            self.state.delta[0, j], self.state.delta[1, j] = self.state.delta[1, j], self.state.delta[0, j]
        for j in np.random.choice(self.n, size=I, replace=False):
            flip(j)

    def __call__(self, plot=True):

        if len(self.history) > self.history_len and np.linalg.norm(self.history[0] - self.history[-1], ord=np.inf) < 1e-3:
            return True

        self.t_series.append(self.state.t)
        a_list, d_list = self.state.adoptions()
        for k in range(self.state.K):
            self.a_series[k].append(a_list[k])
            self.d_series[k].append(d_list[k])
        self.state.tick()

        if plot:
            clear_output(wait=True)
            plt.figure()
            for k in range(self.state.K):
                plt.plot(self.t_series, self.a_series[k], label=f"a{k}")
                if self.scenario == 0:
                    plt.plot(self.t_series, self.d_series[k], label=f"d{k}")
                
            plt.xlabel("Time")
            plt.ylabel("Ratio")
            plt.legend()
            plt.show()
            clear_output(wait=True)

    

    
