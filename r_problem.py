# FIXME: Separate this to a different file so it is optional
from rpy2.robjects import FloatVector
import rpy2.rinterface as ri
from numpy import asarray
# Returns a closure function that can be called from R.
# WARNING: this function minimizes for CEGO
def make_r_fitness(self):
    @ri.rternalize
    def r_fitness(x):
        xpy = asarray(x) - 1 # R vectors are 1-indexed
        y = self.fitness(xpy)
        return FloatVector(asarray(y))
    return r_fitness
