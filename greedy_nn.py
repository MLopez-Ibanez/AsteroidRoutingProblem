import numpy as np
import pandas as pd

def GreedyNN(instance, seed, distance="energy"):
    n = instance.n
    print(distance)
    x = np.full(n, -1, dtype=int)
    f = instance.nearest_neighbor(x, distance)
    df = pd.DataFrame(dict(Fitness=[f], x=[ ' '.join(map(str,x)) ],
                           eval_ranks = [0], budget = [1], seed = [0]))
    return df
