import umm as umm
import pandas as pd
import numpy as np
import time
from timer import Timer

def get_problem(instance_name):
    instance_name = instance_name.lower()
    if "qap" in instance_name:
        from qap import QAP
        return QAP
    elif "lop" in instance_name:
        from lop import LOP
        return LOP
    elif "pfsp_cmax" in instance_name:
        from pfsp import PFSP_Cmax
        return PFSP_Cmax
    elif "pfsp_csum" in instance_name:
        from pfsp import PFSP_Csum
        return PFSP_Csum
    elif "arp" in instance_name:
        from arp import AsteroidRoutingProblem
        return AsteroidRoutingProblem
    raise ValueError("Unknown problem: " + instance_name)


def run_once(algo_name, instance_name, seed, out_filename = None, **algo_params):
    if algo_name == "UMM":
        from umm import UMM
        algo = UMM
    elif algo_name == "CEGO":
        from cego import cego
        algo = cego
    elif algo_name == "GreedyNN":
        from greedy_nn import GreedyNN
        algo = GreedyNN
    elif algo_name == "RandomSearch":
        from random_search import RandomSearch
        algo = RandomSearch
    else:
        raise ValueError("Unknown algo: " + algo_name)

    problem = get_problem(instance_name)
    instance = problem.read_instance(instance_name)

    timer = Timer()
    df = algo(instance, seed, **algo_params)
    if instance.best_fitness is not None and instance.worst_fitness is not None:
        df['Fitness_norm'] = (df.Fitness - instance.best_fitness) / (instance.worst_fitness - instance.best_fitness)
    df['Function evaluations'] = np.arange(1, len(df['Fitness'])+1)
    df['run_time'] = timer.elapsed()
    df['Problem'] = instance.problem_name
    df['instance'] = instance.instance_name
    df['Solver'] = algo_name
    if out_filename is not None:
        df.to_csv(out_filename + '.csv.xz', index=False, compression = "xz")
        # df.to_pickle(out_filename + '.pkl.xz', compression = "xz")
    return df
