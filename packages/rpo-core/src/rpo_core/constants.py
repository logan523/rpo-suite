"""Physical constants and derived orbital quantities.

Units are SI throughout and are carried in the *names*, never as attached unit objects:
metres, seconds, radians, kilograms. Unit conversion happens only at the I/O boundary.

Sources
-------
WGS-84 / EGM-96 defining and derived parameters, as published in NIMA TR8350.2
(Department of Defense World Geodetic System 1984, 3rd edition, amendment 1) and
reproduced in Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
Appendix D.
"""

from __future__ import annotations

import math

#: Earth gravitational parameter GM (WGS-84), m^3 / s^2.
MU_EARTH_M3_S2: float = 3.986004418e14

#: Earth equatorial radius (WGS-84 semi-major axis of the reference ellipsoid), m.
R_EARTH_EQUATORIAL_M: float = 6378137.0

#: Earth second zonal harmonic, dimensionless (unnormalised J2, EGM-96).
J2_EARTH: float = 1.08262668e-3

#: Earth flattening (WGS-84), dimensionless.
FLATTENING_EARTH: float = 1.0 / 298.257223563


def mean_motion_rad_s(semi_major_axis_m: float, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return the Keplerian mean motion ``n = sqrt(mu / a**3)``.

    Parameters
    ----------
    semi_major_axis_m
        Semi-major axis, metres. Must be strictly positive (elliptical/circular orbits only).
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Defaults to Earth.

    Returns
    -------
    float
        Mean motion in radians per second.

    Raises
    ------
    ValueError
        If ``semi_major_axis_m`` or ``mu_m3_s2`` is not strictly positive.

    """
    if semi_major_axis_m <= 0.0:
        raise ValueError(f"semi_major_axis_m must be > 0, got {semi_major_axis_m!r} m")
    if mu_m3_s2 <= 0.0:
        raise ValueError(f"mu_m3_s2 must be > 0, got {mu_m3_s2!r} m^3/s^2")
    return math.sqrt(mu_m3_s2 / semi_major_axis_m**3)


def orbital_period_s(semi_major_axis_m: float, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return the Keplerian orbital period ``T = 2*pi / n``, seconds.

    Raises
    ------
    ValueError
        If ``semi_major_axis_m`` or ``mu_m3_s2`` is not strictly positive.

    """
    return 2.0 * math.pi / mean_motion_rad_s(semi_major_axis_m, mu_m3_s2)
