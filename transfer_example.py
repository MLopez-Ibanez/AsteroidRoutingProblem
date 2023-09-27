from arp import AsteroidRoutingProblem
from space_util import (
    two_shot_transfer,
)

arp_instance = AsteroidRoutingProblem(10, 42)

from_id = -1 # From Earth
to_id = 1
t0 = 1
t1 = 10
result = arp_instance.evaluate_transfer(from_id, to_id, t0, t1)
print (result)

# Brute-force
from_id = 1
to_id = 2
t0 = 1
t1 = 10
best = 10e6
best_t1=-1
for t1 in range(5, 730, 1):
    # man, _ = two_shot_transfer(arp_instance.get_ast_orbit(from_id), arp_instance.get_ast_orbit(to_id), t0=t0, t1=t1-t0)
    # print(f"{t1}:{man.get_total_cost().value}")
    result = arp_instance.evaluate_transfer(from_id, to_id, t0, t1)
    if result < best:
        best = result
        best_t1 = t1
    print(f"{t1}:{result}")


# Sequential one dimensional optimization
t0 = 1
from scipy.optimize import minimize_scalar, minimize
best = minimize_scalar(lambda x: arp_instance.evaluate_transfer(from_id, to_id, t0, t0+x),
                       bounds = (1,730), method = 'bounded', options = dict(xatol=1))
print(best)
t1 = int(best.x)
best = minimize_scalar(lambda x: arp_instance.evaluate_transfer(from_id, to_id, x, x + t1),
                       bounds = (t0,730), method = 'bounded', options = dict(xatol=1))
print(best)

# SLSQP to optimize both.
res = minimize(lambda x: arp_instance.evaluate_transfer(from_id, to_id, x[0], x[0] + x[1]),
               x0 = (0,30),
               bounds = ((0, 730), (1, 730)), method='SLSQP', options=dict(maxiter=50))
print(res)
best = arp_instance.evaluate_transfer(from_id, to_id, int(res.x[0]), int(res.x[1]))
print(best)

# Simpler:
f, t0, t1 = arp_instance.optimize_transfer(from_id, to_id, bounds_t0 = (0,730), bounds_t1 = (1,730))
print(f"t0={t0}, t1={t1}, f={f}")

