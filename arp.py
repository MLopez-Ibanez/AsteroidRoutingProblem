import numpy as np
from space_util import (
    Asteroids,
    to_timedelta,
    transfer_from_Earth,
    two_shot_transfer,
    START_EPOCH,
    Earth,
    MU
)

from scipy.optimize import minimize,Bounds
from scipy.spatial import distance

def assert_bounds(x, bounds):
    bounds = np.asarray(bounds)
    assert (x >= bounds[:,0]).all(), f'{x} >= {bounds[:,0]}'
    assert (x <= bounds[:,1]).all(), f'{x} <= {bounds[:,1]}'

def get_default_opts(method, tol = 1e-4, adaptive = True, eps = 1.4901161193847656e-08,
                     rhobeg = 1.0, maxls = 20, maxcor = 10, jac = "2-point", maxiter = 1000):
    options = { 'Nelder-Mead' : dict(tol = tol, options = dict(fatol=0.0001, adaptive = adaptive)),
                'COBYLA' : dict(tol = tol, options=dict(rhobeg = rhobeg)),
                'L-BFGS-B' : dict(tol = tol, jac = jac, options = dict(eps = eps, maxls = maxls, maxcor = maxcor)),
                'SLSQP' : dict(tol = tol, jac = jac, options = dict(maxiter = maxiter, eps = eps)), }
    return options[method]

class CommonProblem:
    TRANSFER_BOUNDS = [(0., 730.)] # (0 days, 2 years)
    VISIT_BOUNDS = [(1., 730.)] # (1 day, 2 years)
    #
    cost_time_tradeoff = 2 / 30 # 2 km/s ~ 30 days

    def __init__(self):
        self.best_x = np.empty(len(self.x0))
        self.best_f = np.inf
        self.best_man = None
        self.lower = np.array(self.bounds)[:,0]
        self.upper = np.array(self.bounds)[:,1]
        # print(f'lower: {self.lower}\tupper: {self.upper}')

    def to_Bounds(self):
        return Bounds(lb = self.lower, ub = self.upper)

    @classmethod
    def f(self, cost, time):
        return cost + self.cost_time_tradeoff * time 

    def update_best(self, x, cost, time, man):
        f = self.f(cost, time)
        if f < self.best_f:
            self.best_x[:] = x[:]
            self.best_f = f
            self.best_man = man
            if self.print_best:
                print(f'New best:{f}:{cost}:{time}:{x}')
        elif self.print_all:
            print(f'{f}:{cost}:{time}:{x}')
        return f

class VisitProblem(CommonProblem):
    bounds = CommonProblem.TRANSFER_BOUNDS + CommonProblem.VISIT_BOUNDS
    x0 = np.array([1., 30.])
    assert_bounds(x0, bounds)
    print_best = False
    print_all = print_best and False
    
    def __init__(self, from_orbit, to_orbit):
        self.from_orbit = from_orbit
        self.to_orbit = to_orbit
        super().__init__()
        
    def __call__(self, x):
        man, to_orbit = two_shot_transfer(self.from_orbit, self.to_orbit, t0=x[0], t1=x[1])
        cost = man.get_total_cost().value
        time = x.sum()
        f = self.update_best(x, cost, time, man)
        return f


def optimize_problem(problem, method = 'SLSQP', **kwargs):
    options = get_default_opts(method, **kwargs)
    result = minimize(problem, x0 = problem.x0, bounds = problem.bounds,
                      method=method, **options)
    return result

class Spaceship:

    def __init__(self, asteroids):
        self.get_ast_orbit = asteroids.get_orbit
        self.ast_list = []
        self.maneuvers = []
        self.orbit = Earth.propagate(START_EPOCH)
        self.x = np.array([])
        self.f = np.inf

    def add_ast(self, ast_id, x, f, maneuvers):
        self.ast_list.append(ast_id)
        self.orbit = self.get_ast_orbit(ast_id)
        self.x = np.append(self.x, x)
        self.f += f
        self.maneuvers.append(maneuvers)

    def optimize(self, ast_id, instance, **kwargs):
        optimize_problem(instance, **kwargs)
        self.add_ast(ast_id, x = instance.best_x, f = instance.best_f, maneuvers = instance.best_man)
        
    def launch(self, ast_id, **kwargs):
        self.f = 0.0
        return self.visit(ast_id, **kwargs)

    def visit(self, ast_id, **kwargs):
        epoch = START_EPOCH + to_timedelta(self.x.sum())
        from_orbit = self.orbit.propagate(epoch)
        to_orbit = self.get_ast_orbit(ast_id)
        self.optimize(ast_id, VisitProblem(from_orbit, to_orbit), **kwargs)
        return self

    def get_energy_nearest(self, asteroids):
        epoch = START_EPOCH + to_timedelta(self.x.sum())
        ship = self.orbit.propagate(epoch)
        ship_r = ship.r.to_value()[None, :] # Convert it to 1-row 3-cols matrix
        ship_v = ship.v.to_value()[None, :]
        ast_orbits = [ self.get_ast_orbit(ast_id).propagate(epoch) for ast_id in asteroids ]
        ast_r = np.array([ orbit.r.to_value() for orbit in ast_orbits ])
        ast_v = np.array([ orbit.v.to_value() for orbit in ast_orbits ])
        ast_energy = (ast_v**2).sum(axis=1)/2 - MU / np.linalg.norm(ast_r, axis=1)
        ship_energy = (ship_v**2).sum(axis=1) / 2 - MU / np.linalg.norm(ship_r, axis=1)
        energy_difference = np.abs(ast_energy - ship_energy)
        ast_dist = distance.cdist(ship_r, ast_r, 'euclidean')
        print(f'diff_r[0]={ast_dist[0]}, energy_diff[0]={energy_difference[0]}')
        ast_dist /= 1.5e+8
        ast_dist += 0.1 * energy_difference
        return asteroids[np.argmin(ast_dist)]

    def get_euclidean_nearest(self, asteroids):
        epoch = START_EPOCH + to_timedelta(self.x.sum())
        ship = self.orbit.propagate(epoch)
        ship_r = ship.r.to_value()[None,:] # Convert it to 1-row 3-cols matrix
        ast_r = np.array([ self.get_ast_orbit(ast_id).propagate(epoch).r.to_value() for ast_id in asteroids ])
        ast_dist = distance.cdist(ship_r, ast_r, 'euclidean')
        return asteroids[np.argmin(ast_dist)]

    
from problem import Problem
class AsteroidRoutingProblem(Problem):
    # Class attributes
    problem_name = "ARP"

    class _Solution:
        def __init__(self, instance):
            self.instance = instance
            self.ship = Spaceship(instance.asteroids)
            self._x = []

        def step(self, k):
            assert k not in self._x 
            assert len(self._x) < self.instance.n
            if len(self._x) == 0:
                self.ship.launch(k)
            else:
                self.ship.visit(k)
            self._x.append(k)
            return self._x, self.ship.f
        
        @property
        def x(self):
            return np.asarray(self._x, dtype=int)

        @property
        def f(self):
            return self.ship.f

        def get_cost(self):
            cost = 0.0
            for man in self.ship.maneuvers:
                cost += man.get_total_cost().value
            return cost

        def get_time(self):
            return self.ship.x.sum()


    def EmptySolution(self):
        return self._Solution(self)

    def CompleteSolution(self, x):
        self.check_permutation(x)
        sol = self._Solution(self)
        for k in x:
            sol.step(k)
        return sol
            
    def PartialSolution(self, x):
        sol = self._Solution(self)
        for k in x:
            if k < 0:
                break
            sol.step(k)
        return sol

    @classmethod
    def read_instance(cls, instance_name):
        *_, n, seed = instance_name.split("_")
        return cls(int(n), int(seed))
    
    def __init__(self, n, seed):
        self.asteroids = Asteroids(n, seed=seed)
        self.get_ast_orbit = self.asteroids.get_orbit
        self.n = n
        self.seed = seed
        super().__init__(instance_name = str(n) + "_" + str(seed))

    def nearest_neighbor(self, x, distance):
        # This could be optimized to avoid re-evaluating
        sol = self.PartialSolution(x)
        if distance == "euclidean":
            get_next = sol.ship.get_euclidean_nearest
        elif distance == "energy":
            get_next = sol.ship.get_energy_nearest
        else:
            raise ValueError("Unknown distance " + distance)
        
        ast_list = list(set(range(self.n)) - set(sol.x))
        while ast_list:
            k = get_next(ast_list)
            ast_list.remove(k)
            sol.step(k)

        return sol.x, sol.f

    def get_euclidean_distance(self, from_id, to_id, time):
        """Return euclidean distance from one asteroid to a list of asteroids at a particular time:

        from_id : asteroid ID
        
        to_id : List of asteroid IDs

        time : time (relative to START_EPOCH).
        """
        epoch = START_EPOCH + to_timedelta(time)
        from_r = self.get_ast_orbit(from_id).propagate(epoch).r.to_value()
        ast_r = np.array([ self.get_ast_orbit(ast_id).propagate(epoch).r.to_value() for ast_id in to_id ])
        return distance.cdist(from_r, ast_r, 'euclidean')

    def evaluate_transfer(self, from_id, to_id, t0, t1):
        """Calculate objective function value of going from one asteroid to another departing at t0 and arriving at t1. An asteroid ID of -1 denotes Earth."""
        from_orbit = Earth if from_id == -1 else self.get_ast_orbit(from_id)
        to_orbit = Earth if to_id == -1 else self.get_ast_orbit(to_id)
        man, to_orbit = two_shot_transfer(from_orbit, to_orbit, t0=t0, t1=t1-t0)
        cost = man.get_total_cost().value
        return CommonProblem.f(cost, t1)

    def fitness_nosave(self, x):
        return self.CompleteSolution(x).f
