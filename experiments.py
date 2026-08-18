import argparse, os, time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import SimpleQueue, Process

import numpy as np
import pandas as pd

from adoptionopinions_model import initial_state_factory, random_simulation_constants_factory, Simulation, Simulation_constants, Initial_state, Networks
from networks import regular_lattice, random_complete, influencer_network, ring, binary_tree, random_complete_self_loops, directed_lattice, directed_binary_tree, complete, directed_star, T_directed_star, I


@dataclass()
class Experiment:
    experiment_id : str
    n : int
    k : int
    trials : list[Trial] = field(default_factory=list, init=False)


@dataclass(frozen=True)
class Trial:
    variable_value : str
    a_converged_at : int
    x_converged_at : int
    a_steady_state : np.ndarray
    x_steady_state : np.ndarray
    W : np.ndarray


def w(A):
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.argmax(np.abs(eigenvalues))
    lambda_dominant = eigenvalues[idx]
    v_left = eigenvectors[:, idx]
    return np.unique(v_left)

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


def run_topo_experiment(queue:SimpleQueue, n:int, k:int, rounds:int, unique_pref:bool):
    pid = os.getpid()
    np.random.seed(None)
    adopters = [n-1]
    network_influencers = [n-1]
    W = ring(n)
    data = []
    for _ in range(rounds):
        experiment = Experiment(
                experiment_id=f"{pid}{time.time_ns()}",
                n=n,
                k=k)
        IS = initial_state_factory(n, k, adopters=adopters, influencers=network_influencers)
        SC = random_simulation_constants_factory(n, k, IS.x, unique_pref=unique_pref, influencers=network_influencers)
        for topo in [I, complete, directed_star]:
            V = topo(n)
            sim = Simulation(
                    simulation_constants = SC,
                    initial_states = IS,
                    net = Networks(W=W, V=V)
                    )
            while not sim():
                pass
            topo_str = ""
            if topo == I:
                topo_str = "Identity"
            elif topo == complete: 
                topo_str = "Complete"
            elif topo == directed_star:
                topo_str = "Star"
            else:
                raise Exception("topo")

            trial = Trial(
                    variable_value = topo_str,
                        a_converged_at = sim.get_adoption_convergence_point(),
                        x_converged_at = sim.get_opinion_convergence_point(),
                        a_steady_state = sim.get_adoption_steady_state(),
                        x_steady_state = sim.get_opinion_steady_state(),
                        W = W
                        )
            experiment.trials.append(trial)

        for trial in experiment.trials:
            data.append(
                    dict(
                        experiment_id=experiment.experiment_id,
                        #network=None
                        #self_loops=None,
                        #eq18_violated=None,
                        n=n,
                        k=k,
                        relative_value=trial.variable_value,
                        relative_a_convergence=trial.a_converged_at,
                        a_steady_state_distribution=trial.a_steady_state.mean(axis=1),
                        #x_steady_state_distribution=trial.x_steady_state.mean(axis=1),
                        #influencers=network_influencers,
                        #adopters=adopters,
                        #W=V
                    )
                    )
    queue.put(data)


def run_experiment(queue:SimpleQueue, n:int, k:int, rounds:int, network:str, self_loops:bool, verbose:bool, disobej_eq18:bool, unique_pref:bool):
    np.random.seed(None)
    increments = [2, 16, 64, 128, 256, 512, 768, 1024]
    adopters = [i for i in range(8)]
    network_influencers = [i for i in range(8)]
    def lattice(experiment):
        t0 = time.perf_counter()
        IS = initial_state_factory(n, k, adopters=adopters, influencers=network_influencers)
        SC = random_simulation_constants_factory(n, k, IS.x, obej_eq_18=not disobej_eq18, unique_pref=unique_pref)
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
        IS = initial_state_factory( n, k, adopters=adopters, influencers=network_influencers)
        SC = random_simulation_constants_factory(n, k, x0=IS.x, obej_eq_18=not disobej_eq18)
        W = random_complete(n)
        influencers = [i for i in range(n)]
        for n_influencers in increments:
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
        #max_x_convergence = max([trial.x_converged_at for trial in experiment.trials])

        for trial in experiment.trials:
            data.append(
                    dict(
                        experiment_id=experiment.experiment_id,
                        network=network,
                        self_loops=self_loops,
                        eq18_violated=disobej_eq18,
                        n=n,
                        k=k,
                        relative_value=trial.variable_value/max_value,
                        relative_a_convergence=trial.a_converged_at / max_a_convergence,
                        a_steady_state_distribution=trial.a_steady_state.mean(axis=1),
                        x_steady_state_distribution=trial.x_steady_state.mean(axis=1),
                        influencers=network_influencers,
                        adopters=adopters,
                        W=trial.W
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
    p.add_argument("--topo-experiment",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--disobej-eq18",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--unique-pref",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--filename-trailer", default="")
    a = p.parse_args()

    data_list = []
    queue = SimpleQueue()
    work = run_topo_experiment if a.topo_experiment else run_experiment
    kwargs = {'queue':queue,
            'n':a.n, 
            'k':a.k, 
            'rounds':a.rounds, 
            'network':a.network, 
            'self_loops':a.self_loops, 
            'verbose':True if a.procs < 3 else False, 
            'disobej_eq18':a.disobej_eq18, 
            'unique_pref':a.unique_pref
            }

    if a.topo_experiment:
        kwargs = {'queue':queue,
                'n':a.n, 
                'k':a.k, 
                'rounds':a.rounds, 
                'unique_pref':a.unique_pref
                }

    procs = []
    for _ in range(a.procs):
        p = Process(target=work, kwargs=kwargs)
        p.start()
        procs.append(p)
    procs_done = 0
    while procs_done < a.procs:
        data_list.append(queue.get())
        procs_done += 1
    for p in procs:
        p.join()
        
    s = "_self_loops" if a.self_loops else ""
    filename = "data_" + a.network + f"_{pid}_" + datetime.now().strftime("%B_%d__%H_%M") + s + a.filename_trailer + ".csv"
    # The list decomp flattens the list
    df = pd.DataFrame([data_point for data in data_list for data_point in data])
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    t_program_start = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t_program_start)/60:.1f} min")
