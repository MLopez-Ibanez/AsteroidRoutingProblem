import numpy as np
import matplotlib.pyplot as plt

from poliastro.plotting import StaticOrbitPlotter
from poliastro.plotting.util import generate_label
from poliastro.twobody.propagation import propagate
from astropy import units as u
from astropy.time import TimeDelta
from arp import AsteroidRoutingProblem
from space_util import (
    START_EPOCH,
    Earth,
)

def plot_solution(self, x):
    sol = self.CompleteSolution(x)
    t = sol.ship.x
    print(t)
    frame = StaticOrbitPlotter(plane=Earth.plane)
    epoch = START_EPOCH
    ship = Earth.propagate(epoch)
    frame.plot(ship, label="Earth")
        
    for k, (ast, man) in enumerate(zip(x,sol.ship.maneuvers)):
        print(k)
        print(ast)
        print(man)
        epoch += t[2*k]
        ship = ship.propagate(epoch)
        frame.plot_maneuver(ship, man, color = f'C{k+1}', label=generate_label(epoch, f'Impulse {k}'))
        ship = ship.apply_maneuver(man)
        epoch += t[2*k + 1]
        if 2*(k+1) >= len(t):
            tofs = TimeDelta(0 * u.day)
        else:
            tofs = TimeDelta(np.linspace(0,  t[2*(k+1)] * u.day, num=100))
        rr = propagate(ship, tofs)
        frame.plot_trajectory(rr, color = f'C{k+1}',label=generate_label(epoch, f'Asteroid {ast}'))
          
    return frame



# instance = AsteroidRoutingProblem(10, 42)
# x,f = instance.nearest_neighbor([], "euclidean")
# print(x)
# print(f)

# plot_solution(instance, x)
