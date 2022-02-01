import numpy as np
import pandas as pd

def RandomSearch(instance, seed, budget):
    np.random.seed(seed)
    n = instance.n
    sample = []
    fitnesses = []
    for m in range(budget):
        p = np.random.permutation(n)
        f = instance.fitness(p)
        sample.append(p)
        fitnesses.append(f)
    df = pd.DataFrame()
    df['Fitness'] = fitnesses
    df['x'] = [ ' '.join(map(str,s)) for s in sample ]
    df['seed'] = seed
    df['budget'] = budget
    return df
