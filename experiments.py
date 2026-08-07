import argparse, os, time, pickle
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import SimpleQueue, Process

import numpy as np
import pandas as pd

from adoptionopinions_model import initial_state_factory, random_simulation_constants_factory, Simulation, Simulation_constants, Initial_state, Networks
from networks import regular_lattice, random_complete, ring, influencer_network, complete, directed_lattice


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


def printing(pid, ix, trials) -> None:
    def preamble(p,c,typ):
        if p:
            p = p.a_converged_at if typ == "a" else p.x_converged_at
            c = c.a_converged_at if typ == "a" else c.x_converged_at
            return "\033[33m[increase] " if p < c else "\033[34m[decrease] " if p > c else "\033[37m"
        return "\033[37m"

    a = trials[-1].a_steady_state.mean(axis=1)
    x = trials[-1].x_steady_state.mean(axis=1)
    prev, current = (trials[-2], trials[-1]) if len(trials) > 1 else (trials[-1], trials[-1]) 
    print(f"({pid}) " + f"{ix:2} - " + \
            preamble(prev, current, "a") + \
            f"Tech conv. at {current.a_converged_at}" + \
            f"\33[32m Tech {np.argmax(a)} at {a[np.argmax(a)]*100:.2f}%" + \
            "\33[0m" + " | " + \
            preamble(prev, current, "x") + \
            f"Ops conv. at {current.x_converged_at}" + \
            f"\33[32m Op {np.argmax(x)} at {x[np.argmax(x)]*100:.2f}%" + \
            "\33[0m"
          )


def run_experiment(queue:SimpleQueue, n:int, k:int, rounds:int, network:str, self_loops:bool, verbose:bool, disobej_eq18:bool):
    np.random.seed(None)
    increments = [2**i for i in range(1,11)]
    def lattice(experiment):
        t0 = time.perf_counter()
        IS = initial_state_factory(n, k, adopters=None, influencers=None)
        SC = random_simulation_constants_factory(n, k, IS.x, obej_eq_18=not disobej_eq18)
        RC = random_complete(n)
        print(f"\tgen: {abs(t0-time.perf_counter()):.3f} s")

        for degree in increments:
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
            if verbose:
                printing(pid, degree, experiment.trials)

        max_degree = max([trial.variable_value for trial in experiment.trials])
        return max_degree

    def influencer(experiment):
        SC = random_simulation_constants_factory(n, k, x0=np.random.rand(k,n), obej_eq_18=not disobej_eq18)
        W = random_complete(n)
        influencers = [i for i in range(n)]
        for n_influencers in [1] + increments:
            IS = initial_state_factory(
                                    n, 
                                    k, 
                                    adopters=[0], #influencers[:n_influencers],
                                    influencers=[0],#influencers[:n_influencers]
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
            if verbose:
                printing(pid, n_influencers, experiment.trials)

        max_inf = max([trial.variable_value for trial in experiment.trials])
        return max_inf

    data = []
    pid = os.getpid()
    for i in range(rounds):
        s = f" self-loops: {self_loops} " if network != "influencer" else ""
        print(f"({pid})" + f" ({i+1} of {rounds}) -- (network: {network}) " + \
                "-" * 2 + f" n:{n},k:{k} " + "-" * 2 + s, end="")

        experiment = Experiment(
                experiment_id=f"{pid}{time.time_ns()}",
                n=n,
                k=k)

        # The inner functions have side effects to update the 'experiment' object
        max_value = influencer(experiment) if network == "influencer" else lattice(experiment)
        max_a_convergence = max([trial.a_converged_at for trial in experiment.trials])
        max_x_convergence = max([trial.x_converged_at for trial in experiment.trials])

        for trial in experiment.trials:
            data.append(
                    dict(
                        network=network,
                        self_loops=self_loops,
                        relative_value=trial.variable_value/max_value,
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
    p.add_argument("--disobej-eq18",action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()
    s = "_self_loops" if a.self_loops else ""
    filename = "data_" + a.network + f"_{pid}_" + datetime.now().strftime("%B_%d__%H_%M") + s + ".csv"

    data_list = []
    queue = SimpleQueue()
    work = run_experiment
    args = (queue, a.n, a.k, a.rounds, a.network, a.self_loops, True if a.procs < 3 else False, a.disobej_eq18)

    procs = []
    for _ in range(a.procs):
        p = Process(target=work, args=args)
        p.start()
        procs.append(p)
    procs_done = 0
    while procs_done < a.procs:
        data_list.append(queue.get())
        procs_done += 1
    for p in procs:
        p.join()
        
    # The list decomp flattens the list
    df = pd.DataFrame([data_point for data in data_list for data_point in data])
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    t_program_start = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
