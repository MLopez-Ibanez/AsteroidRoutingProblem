from arp import AsteroidRoutingProblem
from arp_vis import plot_solution
from space_util import (
    two_shot_transfer,
    START_EPOCH,
)
import numpy as np

arp_instance = AsteroidRoutingProblem(10, 42)

x = [9,3,8,7,2,6,1,5,4,0]
res1 = arp_instance.CompleteSolution(np.asarray(x))
res2 = arp_instance.evaluate_sequence([-1] + x, current_time=0)

#res = plot_solution(arp_instance, [8,3,0,6,7,9,2,4,1,5])
res1 = arp_instance.CompleteSolution(np.asarray([8,3,0,6,7,9,2,4,1,5]))

res2 = arp_instance.evaluate_sequence([-1,8,3,0,6,7,9,2,4,1,5], current_time=0)


res = arp_instance.optimize_transfer(0, 8, current_time = 0, t0_bounds = (0, 5110), t1_bounds = (1, 730), free_wait = True, multi = 3)
print(res)        

res = arp_instance.optimize_transfer(0, 8, current_time = 126.96358, t0_bounds = (0,4983.0366), t1_bounds = (1, 730), free_wait = True, multi = 3)
print(res)        

res = arp_instance.optimize_transfer(0, 3, current_time = 0, t0_bounds = (0, 13870), t1_bounds = (1, 730), free_wait = True, multi=3)
print(res)        

res = arp_instance.optimize_transfer(0, 3, current_time = 69.746826, t0_bounds = (0,2120.2532), t1_bounds = (1, 730), free_wait = True, multi=4)
print(res)        
 
# Build nearest neighbor solution
f, s, x = arp_instance.build_nearest_neighbor(current_time = 0)
print(f"*** sequence = {s}, t = {x}, cost = {f}\n\n")
f, x = arp_instance.evaluate_sequence(s, current_time = 0)
print(f"*** t = {x}, cost = {f}\n\n")

res = arp_instance.optimize_transfer(1, 2, current_time = 574, t0_bounds = (0, 1), t1_bounds = (240, 260))
print(res)        

res = arp_instance.optimize_transfer_total_time(1, 2, current_time = 574, total_time_bounds = (200,260))
print(res)        

from_id = -1 # From Earth
to_id = 1
t0 = 1 # relative to initial epoch
t1 = 10 # relative to t1
result = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1)
print (result)

result = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1, only_cost = True)
print (result)

result = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1, free_wait = True)
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
best = minimize_scalar(lambda x: arp_instance.evaluate_transfer(from_id, to_id, t0, x),
                       bounds = (1,730), method = 'bounded', options = dict(xatol=1))
print(best)
t1 = int(best.x)
best = minimize_scalar(lambda x: arp_instance.evaluate_transfer(from_id, to_id, x, t1),
                       bounds = (t0,730), method = 'bounded', options = dict(xatol=1))
print(best)

# SLSQP to optimize both.
res = minimize(lambda x: arp_instance.evaluate_transfer(from_id, to_id, x[0], x[1]),
               x0 = (0,30),
               bounds = ((0, 730), (1, 730)), method='SLSQP', options=dict(maxiter=50))
print(res)
best = arp_instance.evaluate_transfer(from_id, to_id, int(res.x[0]), int(res.x[1]))
print(best)

# Simpler:
f, t0, t1 = arp_instance.optimize_transfer(from_id, to_id, t0_bounds = (0,730), t1_bounds = (1,730))
print(f"t0={t0}, t1={t1}, f={f}")

f, t0, t1 = arp_instance.optimize_transfer(from_id, to_id, t0_bounds = (0,730), t1_bounds = (1,730), free_wait = True)
print(f"t0={t0}, t1={t1}, f={f}")

f, t0, t1 = arp_instance.optimize_transfer(-1, 9, t0_bounds = (730,1000), t1_bounds = (1,730))
print(f"t0={t0}, t1={t1}, f={f}")

f, t0, t1 = arp_instance.optimize_transfer(8, 1, (0,730), (0.01,730))


