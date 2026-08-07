import argparse, os, time, pickle, copy, sys
from pathlib import Path

import numpy as np
import pandas as pd

from adoptionopinions_model import initial_state_factory, random_simulation_constants_factory, Simulation, Simulation_constants, Initial_state, Networks
from networks import complete


def NMAE(y, y_pred):
    ret = np.mean(np.abs(y - y_pred)) / np.max(y)
    return ret


vendors = ["Samsung", "Apple", "Unknown", "Xiaomi", "Huawei"]#, "Oppo", "Nokia", "LG", "Motorola"]
vendor_map = {vendor: i for i, vendor in enumerate(vendors)}

def get_initial_state(n, initial_adopters, k):
    assert sum(initial_adopters) < 1
    susceptible = np.array([1 - sum(initial_adopters)] * n)
    adopters = np.empty(shape=(k,n))
    opinions = np.empty(shape=(k,n))
    m = max(initial_adopters)
    for vendor in vendors:
        index = vendor_map[vendor]
        adopters[index] = initial_adopters[index]
        opinions[index] = initial_adopters[index] * m

    for g in [adopters, susceptible, opinions]:
        assert ((g >= 0).all() and (g <= 1).all())

    return Initial_state(susceptible, adopters, np.zeros((k,n)), opinions)

def tune(max_error, df, rounds, n):
    '''
    Returns a new model and its simulation error if the sum of 
        its error is smaller than the previous.
    If not, retuns None and the error untoched.
    '''
    k = len(vendors)
    best = None
    rows = len(df)
    IS = get_initial_state(n, df.iloc[0].tolist(), k)
    net = Networks(W=complete(n), V = complete(n))

    for i in range(rounds):
        SC = random_simulation_constants_factory(n, k, x0=IS.x)
        sim = Simulation(
                simulation_constants = SC,
                initial_states = IS,
                net = net,
                window_size=rows
                )

        for _ in range(rows):
            sim()

        # the shape of the object is: rows * k * n
        states = sim.get_adoption_states()
        nmae = np.empty((k))
        for vendor in vendors:
            index = vendor_map[vendor]
            # market_share_prediction is a row long list of the mean of the k:th vendor.
            # rows * 1 * n -> rows * 1
            market_share_prediction = np.array([state[index].mean() for state in states])
            nmae[index] = NMAE(df[vendor], market_share_prediction)

        s = ""
        if max(nmae) < max_error:
            # creates a new, local object 'max_error'
            max_error = max(nmae)
            s = f"\tNew low: {max_error}"
            best = SC
        print(f"{i+1} of {rounds}" + s)
    return best, max_error

def main():

    n = 1000
    p = argparse.ArgumentParser()
    p.add_argument("--filename", required=True)
    p.add_argument("--rounds", type=int, default=100)
    a = p.parse_args()


    max_error = sys.float_info.max
    path = Path("best_model/model_error.pkl")
    if path.exists():
        with open(path, "rb") as fd:
            max_error = pickle.load(fd)
    print(f"Model error's inf norm error:\t\t{max_error}")

    df = pd.read_csv(a.filename)
    df = df[vendors] / 100

    model, max_error = tune(max_error, df, a.rounds, n)

    if model:
        print(f"New lowest max:\t\t{max_error}")
        model_file = f"best_model/best_model_{n}.pkl"
        error_file = "best_model/model_error.pkl"
        with open(model_file, "wb") as fd:
            pickle.dump((model), fd)
        with open(error_file, "wb") as fd:
            pickle.dump((max_error), fd)

        
if __name__ == "__main__":
    t_program_start = time.perf_counter()
    main()
    pid = str(os.getpid())
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
