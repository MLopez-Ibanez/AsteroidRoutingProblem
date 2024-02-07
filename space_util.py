from numpy import deg2rad
from astropy import units as u
from astropy.time import Time,TimeDelta
from poliastro.core.util import spherical_to_cartesian
from poliastro.core.angles import M_to_E, E_to_nu
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from poliastro.frames import Planes
from poliastro.maneuver import Maneuver

def to_timedelta(x):
    return x * u.day

def MJD_to_Time(x):
    return Time(x, scale='ut1', format='mjd')

# The start time of earth and asteroids orbit.
EARTH_START_EPOCH = MJD_to_Time(59396)
# The mission is limited within 20 years.
# 00:00:00 1st January 2121 (MJD) (first launch)
START_EPOCH = MJD_to_Time(95739)
# 00:00:00 1st January 2141 (MJD) (last launch)
LAST_EPOCH = MJD_to_Time(103044)

class Asteroids:
    def __init__(self, size, seed):
        from pandas import read_pickle
        ast_orbits = read_pickle("ast_orbits.pkl.gz")
        from numpy.random import default_rng
        rng = default_rng(seed)
        ids = rng.integers(ast_orbits.index.min(), ast_orbits.index.max() + 1, size = size)
        self.ast_orbits = ast_orbits[ids].reset_index(drop=True)

    def get_orbit(self, ast_id):
        return self.ast_orbits.loc[ast_id].propagate(START_EPOCH)

# Table 2 Constants and unit conversions
AU = 1.49597870691e8 # km
MU = 1.32712440018e11 # km^3/s^2
SEC_PER_DAY = 86400 # s
DAYS_PER_YEAR = 365.25 # days

def M_to_nu(M, ecc):
    return E_to_nu(M_to_E(M = M, ecc = ecc), ecc = ecc)
    
class OrbitBuilder:
    # Create my own Sun
    Sun = Body(
        parent=None,
        k = MU * (u.km ** 3 / u.s ** 2),
        name="Sun")

    @classmethod
    def from_vectors(cls, r, v, epoch):
        return Orbit.from_vectors(cls.Sun, r, v, epoch = epoch,
                                  plane = Planes.EARTH_ECLIPTIC)
    
    @classmethod
    def eliptic(cls, a, e, i, raan, w, nu, epoch):
        return Orbit.from_classical(cls.Sun, a * AU * u.km,
                                    e * u.one,
                                    i * u.rad,
                                    raan * u.rad,
                                    w * u.rad,
                                    nu = nu * u.rad,
                                    epoch = epoch, # MJD
                                    # Same as HeliocentricEclipticJ2000
                                    # https://github.com/poliastro/poliastro/blob/main/src/poliastro/frames/util.py
                                    plane = Planes.EARTH_ECLIPTIC)
    @classmethod
    def circular(cls, a, i, raan, arglat, epoch):
        return Orbit.circular(cls.Sun, alt = a * AU * u.km,
                              inc = i * u.rad,
                              raan = raan * u.rad,
                              # Argument of latitude or phase.
                              arglat = arglat * u.rad,
                              epoch = epoch,
                              # Same as HeliocentricEclipticJ2000
                              # https://github.com/poliastro/poliastro/blob/main/src/poliastro/frames/util.py
                              plane = Planes.EARTH_ECLIPTIC)

Earth = OrbitBuilder.eliptic(
    # Table 1 Earth’s orbital elements in the J2000 heliocentric ecliptic reference frame
    a = 9.998012770769207e-1 # AU
    , e = 1.693309475505424e-2
    , i = deg2rad(3.049485258137714e-3) # deg
    , raan = deg2rad(1.662869706216879e2) # deg
    , w = deg2rad(2.978214889887391e2) # omega deg
    , nu = M_to_nu(M = deg2rad(1.757352290983351e2), # deg
                   ecc = 1.693309475505424e-2) # theta, rad
    , epoch = EARTH_START_EPOCH).propagate(START_EPOCH) # MJD


def apply_impulse(orbit, dt = 0, dv = None):
    tmp = orbit
    if dt > 0:
        tmp = tmp.propagate(dt)
    r, v = tmp.rv()
    return OrbitBuilder.from_vectors(r, v + dv, tmp.epoch)

def launch_from_Earth(launch_epoch, launch_v):
    intermediate_orbit = apply_impulse(Earth.propagate(launch_epoch),
                                       dt = 0,
                                       dv = launch_v * (u.km / u.s))
    return intermediate_orbit


def transfer_from_Earth(to_orbit, t0, t1, t2,
                      # (v_norm, v_colat, v_long)
                      v_cartesian = None, v_spherical = None):
    v = v_cartesian
    if v is None:
        v = spherical_to_cartesian(v_spherical)
        
    launch_epoch = START_EPOCH + to_time_delta(t0)
    intermediate_orbit = launch_from_Earth(launch_epoch, launch_v = v)
    intermediate_orbit = intermediate_orbit.propagate(launch_epoch + to_timedelta(t1))
    epoch = START_EPOCH + to_timedelta(t0 + t1 + t2)
    assert epoch.value < LAST_EPOCH.value
    to_orbit = to_orbit.propagate(epoch)
    mann = Maneuver.lambert(intermediate_orbit, to_orbit)
    return mann, to_orbit
    

def two_shot_transfer(from_orbit, to_orbit, t0, t1):
    assert t0 >= 0 and t1 > 0,f'It must be true that t0={t0} >= 0 and t1={t1} > 0'
    from_orbit = from_orbit.propagate(from_orbit.epoch + to_timedelta(t0))
    #print(f'from_orbit.epoch: {from_orbit.epoch}')
    epoch = from_orbit.epoch + to_timedelta(t1)
    to_orbit = to_orbit.propagate(epoch)
    #assert epoch.value < LAST_EPOCH.value
    try:
        man = Maneuver.lambert(from_orbit, to_orbit)
    except Exception as e:
        e.args = (e.args if e.args else tuple())
        print(f'two_shot_transfer failed: {type(e)} {str(e.args)}: {from_orbit.rv()} {from_orbit.epoch} {to_orbit.rv()} {to_orbit.epoch} {t0} {t1}')
        # import dump
        # dump.to_pickle(prefix='two_shot_transfer_proposal', from_orbit=from_orbit, to_orbit=to_orbit, e=e)
        # Return a very bad solution
        return (Maneuver((0 * u.s, [1e6, 1e6, 1e6] * u.km / u.s),
                         (LAST_EPOCH.value * SEC_PER_DAY * u.s, [1e6, 1e6, 1e6] * u.km / u.s)),
                to_orbit)
    return man, to_orbit

