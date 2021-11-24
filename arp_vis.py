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

# From https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def get_fig_size(width, fraction=1, subplots=(1, 1)):
    """Get figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'lncs':
        width_pt = 347.12354
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def plot_solution(self, x, ax = None):
    sol = self.CompleteSolution(x)
    t = sol.ship.x
    print(t)
    frame = StaticOrbitPlotter(ax = ax, plane=Earth.plane)
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
    
    return ax, frame



# instance = AsteroidRoutingProblem(10, 42)
# x,f = instance.nearest_neighbor([], "euclidean")
# print(x)
# print(f)

# plot_solution(instance, x)
