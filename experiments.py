import argparse, os, time, math, itertools
from dataclasses import dataclass, field, asdict
from adoptionopinions_model import simulation_factory, Simulation, Simulation_constants, Initial_state, homogenous_simulation_factory
import random_graphs as rg
import matrix_utils as mu
import numpy as np
import json

@dataclass
class Conn_and_dom():
    degree : int
    connetivity : float
    adoption_ratio : float

@dataclass()
class Connectiviy_experiment_data():
    inial_state : Initial_state
    constants : Simulation_constants
    social_network : np.ndarray
    data : list[Conn_and_dom] = field(default_factory=list, init=False)


def connectivity_and_dominance_exp(n,k,constants, initial_states, V):

    
    def opinion_scalers():

        '''
        lambd_i, xi_i >= 0, lambd_i + xi_i < 1
        '''
        l = np.random.rand(k,n)
        xi = np.random.rand(k,n)
        for e in itertools.product(range(k), range(n)):
            # 'e' is the Cartesian product of k and n, i.e., all indexes of 
            # the matries
            while l[e[0]][e[1]] + xi[e[0]][e[1]] >= 1:
                l[e[0]][e[1]], xi[e[0]][e[1]] = np.random.rand(2)

        return l, xi

    lambd, xi = opinion_scalers()
    experiment_results = Connectiviy_experiment_data(initial_states, constants, V)

    number_of_neighbors = [2**i for i in range(1, int(np.log2(n))+1)]
    for degree in number_of_neighbors:
        W = rg.regular_lattice(n, degree)
        conn = mu.algebraic_connectivity(W)
        assert mu.irreducible(W) and mu.irreducible(V)
        assert mu.row_stochastic(W) and mu.row_stochastic(V)
        
        constants.set_beta( np.array([[np.log2(degree)/int(np.log2(number_of_neighbors[-1]))] * n for _ in range(k)]))
        
        sim = Simulation(constants, W, V, *initial_states())
        while (not sim()):
            pass
        experiment_results.data.append(Conn_and_dom(degree, conn,sim.states[-1].a[0].mean(axis=0)))


    #print(f"Constants: {experiment_results.constants}")
    for d in experiment_results.data:
        print(f"{d.adoption_ratio:.5f}")

    return experiment_results


@dataclass()
class Exp():
    n : int
    inial_state : Initial_state
    constants : Simulation_constants
    social_network : np.ndarray
    physical_network : np.ndarray
    NMAE : float

def fitting():
    import os, pickle
    import pandas as pd
    def NMAE(y, y_hat):
        return np.mean(np.abs(y - y_hat)) / np.mean(y)
    
    df = pd.read_csv("../os_combined-ww-monthly-201407-202605.csv")
    df = df['Windows'] / 100
    experiments = []
    rows = len(df)
    for i in range(1_000):
        n = np.random.randint(3,20)
        constants, IS = simulation_factory(n, 2, dislike_tech_2=True)
        W, V = rg.erdos_renyi(n, True), rg.erdos_renyi(n, True)
        sim = Simulation(constants, W, V, *IS())
        sim.max_states = rows
        sim.min_converge_check = rows + 1
        while (not sim()):
            pass
        ss = [state.a[0].mean() for state in sim.states[:rows]]
        nmae = NMAE(df, ss)
        exp = Exp(n, IS, constants, V, W, nmae)
        experiments.append(exp)

    experiments = sorted(experiments, key=lambda x: x.NMAE)
    filename = "best.pkl"
    if os.path.isfile(filename):
        with open(filename, "rb") as file_desc:
            if pickle.load(file_desc).NMAE < experiments[0].NMAE:
                print("sorry")
                return
        with open(filename, "wb") as file_desc:
            print(f"New best model: {experiments[0].NMAE}")
            pickle.dump(experiments[0], file_desc)
    else:
        with open(filename, "wb") as file_desc:
            print("first time buyer")
            pickle.dump(experiments[0], file_desc)


    

def main(pid):
    p = argparse.ArgumentParser()
    p.add_argument("--file-name", default="oriented_hypergraph_data")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--fitting", action="store_true")
    a = p.parse_args()

    n = 129
    k = 2
        
    if a.fitting:
        fitting()
        return

    constants, initial_states = homogenous_simulation_factory(n,k, val=0.1)
    V = rg.regular_lattice(n, n-1)

    experiments = []
    for i in range(a.rounds):
        experiments.append(connectivity_and_dominance_exp(n, k, constants, initial_states, V))
        print()
    df = pd.DataFrame([asdict(p) for p in experiments])

    #df.to_csv("people.csv", index=False)

if __name__ == "__main__":
    t_program_start = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
