import numpy as np

class Problem:
    def __init__(self, best_sol = None, worst_sol = None, instance_name = None,
                 best_fitness = None, worst_fitness=None):
        self.best_sol = best_sol
        self.worst_sol = worst_sol
        self.instance_name = instance_name
        if best_sol is None:
            self.best_fitness = best_fitness
        else:
            self.best_fitness = self.fitness_nosave(best_sol)
            if best_fitness != None:
                assert self.best_fitness == best_fitness
        if worst_sol is None:
            self.worst_fitness = None
        else:
            self.worst_fitness = self.fitness_nosave(worst_sol)
            if worst_fitness != None:
                assert self.worst_fitness == worst_fitness
        self.reset()

    def check_permutation(self, x):
        # Assumes numpy array
        return ((x >= 0) & (x < self.n)).all() and np.unique(x).shape[0] == x.shape[0]

    def reset(self):
        self.evaluations = []
        self.solutions = []

    def fitness_nosave(self, x):
        raise NotImplementedError("virtual method")

    def fitness(self, x):
      f = self.fitness_nosave(x)
      self.solutions.append(x)
      self.evaluations.append(f)
      return f

    def distance_to_best(self, perm, distance):
        if self.best_sol is None:
            return np.nan
        # MANUEL: Why divide by n * (n -1) / 2 ?
        # if kendall: return kendallTau(perm, self.best_sol) / (self.n * (self.n - 1) * 0.5)
        # MANUEL: Why distance to identity?
        # return self.n - (perm==np.arange(self.n)).sum()
        return distance(perm, self.best_sol)
    
    # # Returns a closure function that can be called from R.
    # # WARNING: this function minimizes for CEGO
    # def make_r_fitness(self):
    #     @ri.rternalize
    #     def r_fitness(x):
    #         xpy = np.asarray(x) - 1 # R vectors are 1-indexed
    #         y = self.fitness(xpy)
    #         return FloatVector(np.asarray(y))
    #     return r_fitness
