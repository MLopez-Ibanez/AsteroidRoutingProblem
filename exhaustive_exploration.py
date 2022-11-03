import numpy as np
import pandas as pd
import itertools as it

def ExhaustiveExploration(instance, seed, budget):
    n = instance.n
    sample = []
    fitnesses = []
    for c,p in it.takewhile(lambda x: x[0] < seed+budget, it.dropwhile((lambda x: x[0] < seed), enumerate(it.permutations(range(n))))):
        p = np.array(p)
        f = instance.fitness(p)
        sample.append(p)
        fitnesses.append(f)
    df = pd.DataFrame()
    df['Fitness'] = fitnesses
    df['x'] = [ ' '.join(map(str,s)) for s in sample ]
    df['seed'] = seed
    df['budget'] = budget
    return df

