import numpy as np
import matplotlib.pyplot as plt

from poliastro.plotting import StaticOrbitPlotter
from poliastro.plotting.util import generate_label
from poliastro.twobody.propagation import propagate
from poliastro.ephem import Ephem
from astropy import units as u
from astropy.time import TimeDelta
from arp import AsteroidRoutingProblem
from space_util import (
    START_EPOCH,
    Earth,
)

# From https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def get_fig_size(width, fraction=1, subplots=(1, 1), ratio = (5**.5 - 1) / 2):
    """Get figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    ratio: height = ratio * width, optional.
           By default, the golden ratio. https://disq.us/p/2940ij3
           
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
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def plot_solution(self, x, ax = None):
    x = np.asarray(x)
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
        frame.plot_maneuver(ship, man, color = f'C{k}', label=generate_label(epoch, f'Transfer {k+1}'))
        ship = ship.apply_maneuver(man)
        epoch += t[2*k + 1]
        if 2*(k+1) >= len(t):
#            tofs = TimeDelta(np.array([0]) * u.day)
            #rr = propagate(ship, tofs)
            frame.plot_ephem(Ephem.from_orbit(ship, ship.epoch, plane=Earth.plane), epoch = ship.epoch, color = f'C{k+1}',label=generate_label(epoch, f'Asteroid {ast}'))
        else:
            tofs = TimeDelta(np.linspace(0,  t[2*(k+1)] * u.day, num=100))
            rr = propagate(ship, tofs)
            frame.plot_trajectory(rr, color = f'C{k+1}',label=generate_label(epoch, f'Asteroid {ast}'))

    return ax, frame, sol.f, sol.get_cost(), sol.get_time()

def plot_solution_to_pdf(instance, sol, pdf_file, title = None, figsize = "lncs"):
    fig, ax = plt.subplots(figsize=get_fig_size(figsize, fraction=1))
    ax, _, f, cost, time = plot_solution(instance, sol, ax = ax)
    if title is not None:
        fig.suptitle(title + f'$\Delta V$={cost:.1f} km/s, $T$={time:.1f} days, $f(\pi)$={f:.1f}', x=.58, y=0.94)
    fig.savefig(pdf_file, bbox_inches="tight")

instance = AsteroidRoutingProblem(10, 8)

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Use Type 1 fonts in plots.
    "pdf.fonttype": 42,
}
plt.rcParams.update(tex_fonts)
plot_solution_to_pdf(instance, [5,3,2,8,1,9,6,7,0,4], pdf_file = "sol_dd_10_8.pdf", figsize="thesis")

# x,f = instance.nearest_neighbor([], "euclidean")
# print(x)
# print(f)

# plot_solution(instance, x)
