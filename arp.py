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

def get_default_opts(method, tol = 1e-6, adaptive = True, eps = 1.4901161193847656e-08,
                     rhobeg = 1.0, maxls = 20, maxcor = 10, jac = "2-point", maxiter = 1000):
    options = { 'Nelder-Mead' : dict(tol = tol, options = dict(fatol=0.0001, adaptive = adaptive)),
                'COBYLA' : dict(tol = tol, options=dict(rhobeg = rhobeg)),
                'L-BFGS-B' : dict(tol = tol, jac = jac, options = dict(eps = eps, maxls = maxls, maxcor = maxcor)),
                'SLSQP' : dict(tol = tol, jac = jac, options = dict(maxiter = maxiter, eps = eps)), }
    return options[method]

class CommonProblem:
    TRANSFER_BOUNDS = (0., 730.) # (0 days, 2 years)
    VISIT_BOUNDS = (1., 730.) # (1 day, 2 years)
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
    bounds = [CommonProblem.TRANSFER_BOUNDS, CommonProblem.VISIT_BOUNDS]
    x0 = np.array([1., 30.]) # FIXME: perhaps it should be [0., 30.] to match the optimize_* functions below.
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


def inner_minimize_multistart(fun, multi, bounds, method = 'SLSQP', constraints = (), **kwargs):
    options = get_default_opts(method, **kwargs)
    best_f = np.inf
    best_t0 = None
    best_t1 = None
    deltas = [ .0, .98, .5, .25, .75, .125, .375, .625, .875]
    for d in deltas[:multi]:
        x0 = (bounds[0][0] + d * (bounds[0][1] - bounds[0][0]), min(30, bounds[1][1]))
        print(f"t0_bounds = {bounds[0]}, t1_bounds = {bounds[1]}, x0 = {x0}")
        res = minimize(fun, x0 = x0, bounds = bounds, method = method, constraints = constraints, **options)
        if res.fun < best_f:
            best_f, best_t0, best_t1 = res.fun, res.x[0], res.x[1] 

    return (best_f, best_t0, best_t1)

def inner_minimize(fun, x0, bounds, method = 'SLSQP', constraints = (), **kwargs):
    options = get_default_opts(method, **kwargs)
    res = minimize(fun, x0 = x0, bounds = bounds, method = method, constraints = constraints, **options)
    return (res.fun, res.x[0], res.x[1])


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
        #print(f"f = {self.f}")
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
        self.get_ast_orbit = lambda x: Earth if x == -1 else self.asteroids.get_orbit(x)
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

    def _evaluate_transfer_orbit(self, from_orbit, to_orbit, current_time, t0, t1, only_cost, free_wait):
        """Here t0 is relative to current_time and t1 is relative to current_time + t0"""
        man, _ = two_shot_transfer(from_orbit, to_orbit, t0 = current_time + t0, t1=t1)
        cost = man.get_total_cost().value
        assert not (only_cost and free_wait)
        if only_cost:
            return cost
        if free_wait:
            t0 = 0
        f = CommonProblem.f(cost, t0+t1)
        # if f < self.best_f:
        #     self.best_f = f
        #     print(f'New best:{f}:{cost}:{t0+t1}:[{t0}, {t1}]')
        return f

    def evaluate_transfer(self, from_id, to_id, current_time, t0, t1, only_cost = False, free_wait = False):
        """Calculate objective function value of going from one asteroid to another departing at current_time + t0 and flying for a duration of t1. An asteroid ID of -1 denotes Earth."""
        from_orbit = self.get_ast_orbit(from_id)
        to_orbit = self.get_ast_orbit(to_id)
        return self._evaluate_transfer_orbit(from_orbit, to_orbit, current_time, t0, t1, only_cost = only_cost, free_wait = free_wait)

    def optimize_transfer_orbit_total_time(self, from_orbit, to_orbit, current_time, total_time_bounds,
                                           only_cost = False, free_wait = False):
        """ total_time_bounds are relative to current_time."""
        t0_s, t0_f = CommonProblem.TRANSFER_BOUNDS
        t1_s, t1_f = CommonProblem.VISIT_BOUNDS
        assert total_time_bounds[1] >= total_time_bounds[0]
        # We cannot do less than t0_bounds[0], but we could do more (by arriving later if needed).
        t0_s = max(t0_s, total_time_bounds[0] - t1_f)
        t1_f = min(t1_f, total_time_bounds[1] - t0_s)
        t0_f = max(t0_s, total_time_bounds[1] - t1_s)
        t0_bounds = (t0_s, t0_f)
        t1_bounds = (t1_s, t1_f)
        starting_guess = (t0_s, 30)
        print(f"t0_bounds = {t0_bounds}, t1_bounds = {t1_bounds}, x0 = {starting_guess}")
        cons = ({'type': 'ineq', 'fun': lambda x: total_time_bounds[1] - (x[0] + x[1]) },
                {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - total_time_bounds[0]})
        res = inner_minimize(lambda x: self._evaluate_transfer_orbit(from_orbit, to_orbit, current_time, x[0], x[1],
                                                                     only_cost = only_cost, free_wait = free_wait),
                             x0 = starting_guess, bounds = (t0_bounds, t1_bounds), constraints = cons)
        return res
        
    def optimize_transfer_total_time(self, from_id, to_id, current_time, total_time_bounds,
                                     only_cost = False, free_wait = False):
        """ total_time_bounds are relative to current_time."""
        from_orbit = self.get_ast_orbit(from_id)
        to_orbit = self.get_ast_orbit(to_id)
        return self.optimize_transfer_orbit_total_time(from_orbit, to_orbit, current_time, total_time_bounds,
                                                       only_cost = only_cost, free_wait = free_wait)
            
    def optimize_transfer_orbit(self, from_orbit, to_orbit, current_time, t0_bounds, t1_bounds,
                                only_cost = False, free_wait = False, multi = 1):
        """Here t0_bounds are relative to current_time and t1_bounds are relative to current_time + t0"""
        #self.best_f = np.inf
        res = inner_minimize_multistart(lambda x: self._evaluate_transfer_orbit(from_orbit, to_orbit, current_time, x[0], x[1],
                                                                                only_cost = only_cost, free_wait = free_wait),
                             multi = multi, bounds = (t0_bounds, t1_bounds))
        return res

    def optimize_transfer(self, from_id, to_id, current_time, t0_bounds, t1_bounds,
                          only_cost = False, free_wait = False, multi = 1):
        from_orbit = self.get_ast_orbit(from_id)
        to_orbit = self.get_ast_orbit(to_id)
        return self.optimize_transfer_orbit(from_orbit, to_orbit, current_time, t0_bounds, t1_bounds,
                                            only_cost = only_cost, free_wait = free_wait, multi = multi)
    
    def get_nearest_neighbor_euclidean(self, from_id, unvisited_ids, current_time):
        epoch = START_EPOCH + to_timedelta(current_time)
        from_r = self.get_ast_orbit(from_id).propagate(epoch).r.to_value()[None,:] # Convert it to 1-row 3-cols matrix
        ast_r = np.array([ self.get_ast_orbit(ast_id).propagate(epoch).r.to_value() for ast_id in unvisited_ids ])
        ast_dist = distance.cdist(from_r, ast_r, 'euclidean')
        return unvisited_ids[np.argmin(ast_dist)]

    def build_nearest_neighbor(self, current_time):
        from_id = -1 # From Earth
        unvisited_ids = np.arange(self.n)
        f_total = 0.0
        x = []
        s = [ from_id ]
        while len(unvisited_ids) > 0:
            to_id = self.get_nearest_neighbor_euclidean(from_id = from_id, unvisited_ids = unvisited_ids, current_time = current_time)
            f, t0, t1 = self.optimize_transfer(from_id, to_id, current_time, t0_bounds = CommonProblem.TRANSFER_BOUNDS, t1_bounds = CommonProblem.VISIT_BOUNDS)
            unvisited_ids = np.setdiff1d(unvisited_ids, to_id)
            f_total += f
            print(f'Departs from {from_id} at time {current_time + t0} after waiting {t0} days and arrives at {to_id} at time {current_time + t0 + t1} after travelling {t1} days, total cost = {f_total}')
            from_id = to_id
            x += [t0, t1]
            s += [ to_id ]
            current_time += t0 + t1
        return (f_total, s, x)

    def evaluate_sequence(self, sequence, current_time):
        seq_orbits = [ self.get_ast_orbit(i) for i in sequence ]
        f_total = 0.0
        x = []
        for i in range(1, len(seq_orbits)):
            from_orbit = seq_orbits[i-1]
            to_orbit = seq_orbits[i]
            f, t0, t1 = self.optimize_transfer_orbit(from_orbit, to_orbit, current_time, t0_bounds = CommonProblem.TRANSFER_BOUNDS, t1_bounds = CommonProblem.VISIT_BOUNDS)
            f_total += f
            print(f'Departs from {sequence[i-1]} at time {current_time + t0} after waiting {t0} days and arrives at {sequence[i]} at time {current_time + t0 + t1} after travelling {t1} days, total cost = {f_total}')
            x += [t0, t1]
            current_time += t0 + t1
        return (f_total, x)

    def fitness_nosave(self, x):
        return self.CompleteSolution(x).f


    
