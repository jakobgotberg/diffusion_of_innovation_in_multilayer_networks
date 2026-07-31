import argparse, os, time, pickle
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import SimpleQueue, Process

import numpy as np
import pandas as pd

from adoptionopinions_model import initial_state_factory, random_simulation_constants_factory, Simulation, Simulation_constants, Initial_state, Networks
from networks import regular_lattice, random_complete, ring, influencer_network


T = 12


@dataclass()
class Experiment:
    experiment_id : str
    n : int
    k : int
    trials : list[Trial] = field(default_factory=list, init=False)


@dataclass(frozen=True)
class Trial:
    variable_value : int
    a_converged_at : int
    x_converged_at : int
    a_steady_state : np.ndarray
    x_steady_state : np.ndarray


@dataclass(frozen=True)
class Connectivity_experiment_data_point:
    network : str
    self_loops : bool
    relative_degree : float
    relative_a_convergence : float
    relative_x_convergence : float
    a_steady_state_distribution : np.ndarray
    x_steady_state_distribution : np.ndarray

@dataclass(frozen=True)
class Influencer_experiment_data_point:
    network : str
    relative_n_influencers : float
    relative_a_convergence : float
    relative_x_convergence : float
    a_steady_state_distribution : np.ndarray
    x_steady_state_distribution : np.ndarray

def degrees(n):
    return np.unique(sorted([g + 1 if g % 2 != 0 else g for g in [int(np.sqrt(2)**i) for i in range(T)] if 1<g<n]) + [n])


def printing(ix, trials) -> None:
    def preamble(p,c,typ):
        if p:
            p = p.a_converged_at if typ == "a" else p.x_converged_at
            c = c.a_converged_at if typ == "a" else c.x_converged_at
            return "\033[33m[increase] " if p < c else "\033[34m[decrease] " if p > c else "\033[37m"
        return "\033[37m"

    a = trials[-1].a_steady_state.mean(axis=1)
    x = trials[-1].x_steady_state.mean(axis=1)
    prev, current = (trials[-2], trials[-1]) if len(trials) > 1 else (trials[-1], trials[-1]) 
    print(f"{ix:2} - " + \
            preamble(prev, current, "a") + \
            f"Tech conv. at {current.a_converged_at}" + \
            f"\33[32m Tech {np.argmax(a)} at {a[np.argmax(a)]*100:.2f}%" + \
            "\33[0m" + " | " + \
            preamble(prev, current, "x") + \
            f"Ops conv. at {current.x_converged_at}" + \
            f"\33[32m Op {np.argmax(x)} at {x[np.argmax(x)]*100:.2f}%" + \
            "\33[0m"
          )


def run_influencer_experiment(pid:int, queue:SimpleQueue, n:int, k:int, rounds:int, network:str, self_loops:bool):
    data = []
    for i in range(rounds):

        experiment = Experiment(
                experiment_id=f"{pid}{time.time_ns()}",
                n=n, 
                k=k)
        SC = random_simulation_constants_factory(n, k, x0=np.array([[0.1] * n] * k))
        print(f"({i+1} of {rounds})" + "INFLUENCER" + "-" * 10 + f" n:{n},k:{k} " + "-" * 10)
        W = random_complete(n)
        influencers = np.random.permutation(n)
        for n_influencers in range(1, n+5, 4):
            IS = initial_state_factory(
                                    n, 
                                    k, 
                                    adopters=influencers[:n_influencers],
                                    influencers=influencers[:n_influencers]
                                    )
            SC.set_x0(IS.x)
            V = influencer_network(n, influencers[:n_influencers])
            sim = Simulation(
                    simulation_constants = SC,
                    initial_states = IS,
                    net = Networks(W=W, V=V)
                    )
            while not sim():
                pass
            trial = Trial(
                        variable_value = n_influencers,
                        a_converged_at = sim.get_adoption_convergence_point(),
                        x_converged_at = sim.get_opinion_convergence_point(),
                        a_steady_state = sim.get_adoption_steady_state(),
                        x_steady_state = sim.get_opinion_steady_state()
                        )
            experiment.trials.append(trial)
            printing(n_influencers, experiment.trials)

        max_inf = max([trial.variable_value for trial in experiment.trials])
        max_a_convergence = max([trial.a_converged_at for trial in experiment.trials])
        max_x_convergence = max([trial.x_converged_at for trial in experiment.trials])

        for trial in experiment.trials:
            data.append(
                    dict(
                        network=network,
                        self_loops=True,
                        relative_value=trial.variable_value/max_inf,
                        relative_a_convergence=trial.a_converged_at / max_a_convergence,
                        relative_x_convergence=trial.x_converged_at / max_x_convergence,
                        a_steady_state_distribution=trial.a_steady_state.mean(axis=1),
                        x_steady_state_distribution=trial.x_steady_state.mean(axis=1)
                    )
                    )
    queue.put(data)

def run_experiment(pid:int, queue:SimpleQueue, n:int, k:int, rounds:int, network:str, self_loops:bool):
    data = []

    for i in range(rounds):

        t0 = time.perf_counter()
        experiment = Experiment(
                experiment_id=f"{pid}{time.time_ns()}",
                n=n,
                k=k)
        start = np.random.randint(0,n)
        IS = initial_state_factory(n, k, adopters=[start], influencers=[start])
        SC = random_simulation_constants_factory(n, k, IS.x)
        gen_t = time.perf_counter() - t0
        print(f"({i+1} of {rounds}) -- (network: {network}) " + f"self-loops: {self_loops} " + \
                "-" * 10 + f" n:{n},k:{k} " + "-" * 10 + f" (Generation: {gen_t:.3f} s)")
        RC = random_complete(n)

        for degree in degrees(n):
            RL = regular_lattice(n, degree, self_loop=self_loops)
            W, V = (RL, RC) if network == "physical" else (RC, RL) if network == "virtual" else (None, None)

            sim = Simulation(
                    simulation_constants = SC,
                    initial_states = IS,
                    net = Networks(W=W, V=V)
                    )
            while not sim():
                pass
            trial = Trial(
                        variable_value = np.count_nonzero(W if network == "physical" else V) / n,
                        a_converged_at = sim.get_adoption_convergence_point(),
                        x_converged_at = sim.get_opinion_convergence_point(),
                        a_steady_state = sim.get_adoption_steady_state(),
                        x_steady_state = sim.get_opinion_steady_state()
                        )
            experiment.trials.append(trial)
            printing(degree, experiment.trials)

        max_degree = max([trial.variable_value for trial in experiment.trials])
        max_a_convergence = max([trial.a_converged_at for trial in experiment.trials])
        max_x_convergence = max([trial.x_converged_at for trial in experiment.trials])

        for trial in experiment.trials:
            data.append(
                    dict(
                        network=network,
                        self_loops=self_loops,
                        relative_value=trial.variable_value/max_degree,
                        relative_a_convergence=trial.a_converged_at / max_a_convergence,
                        relative_x_convergence=trial.x_converged_at / max_x_convergence,
                        a_steady_state_distribution=trial.a_steady_state.mean(axis=1),
                        x_steady_state_distribution=trial.x_steady_state.mean(axis=1)
                    )
                    )
    queue.put(data)

def main(pid):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--network", choices=["physical", "virtual", "influencer"], required=True)
    p.add_argument("--self-loops",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args()
    s = "_self_loops" if a.self_loops else ""
    filename = "data_" + a.network + f"_{pid}_" + datetime.now().strftime("%B_%d__%H_%M") + s + ".csv"

    data_list = []
    procs_done = 0
    queue = SimpleQueue()
    work = run_experiment if not a.network == "influencer" else run_influencer_experiment
    args = (pid, queue, a.n, a.k, a.rounds, a.network, a.self_loops)

    for _ in range(a.procs):
        Process(target=work, args=args).start() 
    while procs_done < a.procs:
        data_list.append(queue.get())
        procs_done += 1
        


    df = pd.DataFrame([data_point for data in data_list for data_point in data])

    df.to_csv(filename, index=False)

if __name__ == "__main__":
    t_program_start = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
