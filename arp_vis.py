from astropy import units as u
from astropy.time import Time
import numpy as np

from poliastro.plotting import StaticOrbitPlotter
import matplotlib.pyplot as plt

from arp import AsteroidRoutingProblem
from space_util import (
    Asteroids,
    transfer_from_Earth,
    two_shot_transfer,
    START_EPOCH,
    Earth,
    MU
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
        frame.plot_maneuver(ship, man, label=f'Impulse {ast}')
        ship = ship.apply_maneuver(man)
        # frame.plot_trajectory(
        #    ship.sample(max_anomaly=180 * u.deg), label=f'Asteroid {ast}')
        #frame.plot(ship, label=f'Asteroid {ast}')
        
    plt.show()


instance = AsteroidRoutingProblem(4, 42)
x,f = instance.nearest_neighbor([], "euclidean")
print(x)
print(f)

plot_solution(instance, x)
