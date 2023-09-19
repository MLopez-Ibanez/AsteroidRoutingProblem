from arp import AsteroidRoutingProblem

arp_instance = AsteroidRoutingProblem(10, 42)
from_id = 1 # your value
to_id = 2 # your value
# t0 = astropy_timedelta(0, format="jd") # your value
# t1 = astropy_timedelta(4, format="jd") # your value
t0 = 1
t1 = 10
result = arp_instance.evaluate_transfer(from_id, to_id, t0, t1)

# x,f = instance.nearest_neighbor([], "euclidean")
# print(x)
# print(f)

# plot_solution(instance, x)
