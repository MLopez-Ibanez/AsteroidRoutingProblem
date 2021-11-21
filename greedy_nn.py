import numpy as np
import pandas as pd

def GreedyNN(instance, seed = None, distance=None):
    n = instance.n
    x = np.full(n, -1, dtype=int)
    x, f = instance.nearest_neighbor(x, distance=distance)
    df = pd.DataFrame(dict(Fitness=[f], x=[ ' '.join(map(str,x)) ],
                           eval_ranks = [0], budget = [1], seed = [0],
                           distance = [distance]))
    return df
