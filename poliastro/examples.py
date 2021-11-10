"""Example data.

"""
from astropy import time, units as u

from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit

iss = Orbit.from_vectors(
    Earth,
    [8.59072560e2, -4.13720368e3, 5.29556871e3] * u.km,
    [7.37289205, 2.08223573, 4.39999794e-1] * u.km / u.s,
    time.Time("2013-03-18 12:00", scale="utc"),
)
"""ISS orbit example

Taken from Plyades (c) 2012 Helge Eichhorn (MIT License)

"""

molniya = Orbit.from_classical(
    attractor=Earth,
    a=26600 * u.km,
    ecc=0.75 * u.one,
    inc=63.4 * u.deg,
    raan=0 * u.deg,
    argp=270 * u.deg,
    nu=80 * u.deg,
)
"""Molniya orbit example"""

_r_a = Earth.R + 35950 * u.km
_r_p = Earth.R + 250 * u.km
_a = (_r_a + _r_p) / 2
soyuz_gto = Orbit.from_classical(
    attractor=Earth,
    a=_a,
    ecc=_r_a / _a - 1,
    inc=6 * u.deg,
    raan=188.5 * u.deg,
    argp=178 * u.deg,
    nu=0 * u.deg,
)
"""Soyuz geostationary transfer orbit (GTO) example

Taken from Soyuz User's Manual, issue 2 revision 0

"""

churi = Orbit.from_classical(
    attractor=Sun,
    a=3.46250 * u.AU,
    ecc=0.64 * u.one,
    inc=7.04 * u.deg,
    raan=50.1350 * u.deg,
    argp=12.8007 * u.deg,
    nu=63.89 * u.deg,
    epoch=time.Time("2015-11-05 12:00", scale="utc"),
)
"""Comet 67P/Churyumov–Gerasimenko orbit example"""
