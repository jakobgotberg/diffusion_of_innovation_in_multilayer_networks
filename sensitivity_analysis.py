import argparse, os, time, pickle, copy, sys
from pathlib import Path

import numpy as np
import pandas as pd

from adoptionopinions_model import initial_state_factory, random_simulation_constants_factory, Simulation, Simulation_constants, Initial_state, Networks

from real_data import get_initial_state, vendors, vendor_map
from networks import complete



def run(SC, IS, net, rounds, rows):

    data_1 = []
    data_2 = []
    data_3 = []
    for i in range(rounds):
        beta = np.array([[np.random.random()] * SC.n] * SC.k)
        gamma = np.array([[np.random.random()] * SC.n] * SC.k)
        SC = Simulation_constants(SC.n, SC.k, SC.lambd, SC.xi, SC.beta, gamma, SC.delta, SC.x0)
        sim = Simulation(
            simulation_constants = SC,
            initial_states = IS,
            net = net,
            window_size=2 * rows
            )
        for _ in range(rows):
            sim()
        states = sim.get_adoption_states()
        market_share_prediction = [None] * SC.k
        for vendor in vendors:
            index = vendor_map[vendor]
            market_share_prediction[index] = np.array([state[index].mean() for state in states])

        data_1.append(market_share_prediction[0])
        data_2.append(market_share_prediction[1])
        data_3.append(market_share_prediction[2])
        print(f"{i+1} of {rounds}")
    return data_1, data_2, data_3

def main():

    n = 1000
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--filename", required=True)
    p.add_argument("--rounds", type=int, default=128)
    a = p.parse_args()

    with open(a.model_path, "rb") as fd:
        SC = pickle.load(fd)

    g = a.rounds
    df = pd.read_csv(a.filename)
    df = df[vendors] / 100
    s, a = get_initial_state(n, df.iloc[0].tolist(), SC.k)
    IS = Initial_state(s, a, np.zeros((SC.k,n)), SC.x0)
    net = Networks(W=complete(n), V = complete(n))
    data_1, data_2, data_3 = run(SC, IS, net, g, len(df))
    with open("sensitive_analysis.pkl", "wb") as fd:
        pickle.dump((data_1, data_2, data_3), fd)

if __name__ == "__main__":
    t_program_start = time.perf_counter()
    main()
    pid = str(os.getpid())
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
