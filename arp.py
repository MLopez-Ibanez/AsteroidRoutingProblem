import numpy as np
from space_util import (
    Asteroids,
    transfer_from_Earth,
    two_shot_transfer,
    START_EPOCH
)

from scipy.optimize import minimize
from scipy.optimize import Bounds

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
    # The Mother Ship's escaping velocity from the earth cannot exceed 6 km/s:
    MAX_VELOCITY=6. # km / s

    LAUNCH_BOUNDS = [(0., 365.),
                     (0.01, MAX_VELOCITY),
                     (0., np.pi),
                     (0., 2 * np.pi)]
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

class LaunchProblem(CommonProblem):
    bounds =  CommonProblem.LAUNCH_BOUNDS + CommonProblem.TRANSFER_BOUNDS + CommonProblem.VISIT_BOUNDS
    # Initial solution
    x0 = np.array([0, 3., np.pi/2, np.pi, 60., 60.])
    assert_bounds(x0, bounds)
    print_best = False
    print_all = print_best and True
        
    def __init__(self, ast_orbit):
        self.ast_orbit = ast_orbit
        super().__init__()
        
    def __call__(self, x):
        man, to_orbit = transfer_from_Earth(self.ast_orbit,
                                            t0 = x[0], t1 = x[4], t2 = x[5],
                                            v_spherical = x[1:4])
        cost = man.get_total_cost().value
        time = x[[0,4,5]].sum()
        f = self.update_best(x, cost, time, man)
        return f

class VisitProblem(CommonProblem):
    bounds = CommonProblem.TRANSFER_BOUNDS + CommonProblem.VISIT_BOUNDS
    x0 = np.array([0., 30.])
    assert_bounds(x0, bounds)
    print_best = False
    print_all = print_best and True
    
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
        self.x = np.array([])
        self.f = np.inf
        self.maneuvers = []

    def add_ast(self, ast_id, x, f, maneuvers):
        self.ast_list.append(ast_id)
        self.x = np.append(self.x, x)
        self.f += f
        self.maneuvers.append(maneuvers)

    def optimize(self, ast_id, instance, **kwargs):
        optimize_problem(instance, **kwargs)
        self.add_ast(ast_id, x = instance.best_x, f = instance.best_f, maneuvers = instance.best_man)
        
    def launch(self, ast_id, **kwargs):
        self.f = 0.0
        self.optimize(ast_id, LaunchProblem(self.get_ast_orbit(ast_id)), **kwargs)
        return self

    def visit(self, ast_id, **kwargs):
        epoch = START_EPOCH + (self.x[[0,4,5]].sum() + self.x[6:].sum())
        from_orbit = self.get_ast_orbit(self.ast_list[-1]).propagate(epoch)
        to_orbit = self.get_ast_orbit(ast_id)
        self.optimize(ast_id, VisitProblem(from_orbit, to_orbit), **kwargs)
        return self

from problem import Problem
class AsteroidRouting(Problem):
    # Class attributes
    problem_name = "ARP"

    @classmethod
    def read_instance(self, instance_name):
        *_, n, seed = instance_name.split("_")
        return AsteroidRouting(int(n), int(seed))
    
    def __init__(self, n, seed):
        self.asteroids = Asteroids(n, seed=seed)
        self.n = n
        self.seed = seed
        super().__init__(instance_name = str(n) + "_" + str(seed))

    def fitness_nosave(self, x):
        ship = Spaceship(self.asteroids)
        ship.launch(x[0])
        for ast in x[1:]:
            ship.visit(ast)
        return ship.f

