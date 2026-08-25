r"""Non-spherical gravity (J2) and atmospheric drag, and a propagator that composes them.

This is model M6 of ``docs/project1/math-model.md`` plus the drag term that M6 does not
cover. Everything here is an *addition* to the point-mass acceleration of
:mod:`rpo_core.propagate`; that module is untouched and remains the pure two-body oracle.

The equations
-------------
**J2 — the second zonal harmonic.** Earth is not a point mass. Expanding its gravitational
potential in spherical harmonics and keeping the first zonal term beyond the monopole gives
the potential energy per unit mass

.. math::

    V(\mathbf{r}) = -\frac{\mu}{r}
        \left[ 1 - J_2 \left(\frac{R_e}{r}\right)^{2} P_2(z/r) \right],
    \qquad P_2(u) = \tfrac{1}{2}\left(3u^{2} - 1\right),

with :math:`r = \lVert\mathbf{r}\rVert` and :math:`z = r_z` in the Earth-centred inertial
frame, so that :math:`z/r = \sin\varphi` is the sine of geocentric latitude. The
perturbing part is

.. math::

    V_{J_2} = \frac{\mu J_2 R_e^{2}}{2} \, \frac{3 z^{2} - r^{2}}{r^{5}} .

Taking :math:`\mathbf{a} = -\nabla V_{J_2}` and writing
:math:`k = -\tfrac{3}{2} J_2 \frac{\mu}{r^{2}} \left(\frac{R_e}{r}\right)^{2}`:

.. math::

    a_x &= k \left(1 - 5 (z/r)^{2}\right) \frac{x}{r} \\
    a_y &= k \left(1 - 5 (z/r)^{2}\right) \frac{y}{r} \\
    a_z &= k \left(3 - 5 (z/r)^{2}\right) \frac{z}{r}

The asymmetry between the in-plane and the :math:`z` bracket (1 versus 3) is the whole
physics: it is the term that torques the orbit plane, and dropping it leaves a perturbation
that is merely a radial correction and produces no nodal regression at all.

Two closed-form consequences are used as the tests of this module, because they are
independent of the implementation:

.. math::

    \dot{\Omega} &= -\tfrac{3}{2}\, n J_2 \left(\frac{R_e}{p}\right)^{2} \cos i \\
    \dot{\omega} &= +\tfrac{3}{4}\, n J_2 \left(\frac{R_e}{p}\right)^{2}
                     \left(5\cos^{2} i - 1\right)

with :math:`p = a(1 - e^{2})`. Both are first-order secular rates in the mean elements.
The second vanishes at the *critical inclination* :math:`\cos^{2} i = 1/5`,
:math:`i = 63.435^{\circ}` -- the Molniya inclination -- which is the cheapest way to check
that the coefficient is :math:`(5\cos^{2} i - 1)` and not one of the several near-misses in
circulation. See :func:`secular_arg_periapsis_rate_rad_s`.

Because :math:`V_{J_2}` is a potential, **J2 is conservative**: the total specific energy
:math:`v^{2}/2 + V` is invariant, while the *Keplerian* energy :math:`v^{2}/2 - \mu/r` is
not. :func:`specific_energy_with_j2_j_kg` computes the invariant one, and the test suite
asserts the first is conserved and the second is not.

**Atmospheric drag.** With :math:`\boldsymbol{\omega}_\oplus` the Earth rotation vector and
a co-rotating atmosphere,

.. math::

    \mathbf{v}_{\mathrm{rel}} = \mathbf{v} - \boldsymbol{\omega}_\oplus \times \mathbf{r},
    \qquad
    \mathbf{a}_{\mathrm{drag}} = -\tfrac{1}{2}\, \rho(h) \,
        \frac{C_D A}{m} \, \lVert\mathbf{v}_{\mathrm{rel}}\rVert \,
        \mathbf{v}_{\mathrm{rel}} .

Drag is *not* conservative and *not* a function of position alone: it removes energy
monotonically, and the semi-major axis falls. That contrast with J2 is the sharpest
available test that the two force models have not been swapped or mis-signed.

The atmosphere model, and why you should not trust it
-----------------------------------------------------
:func:`exponential_density_kg_m3` is the standard **piecewise-exponential** fit: within each
altitude band, :math:`\rho(h) = \rho_0 \exp\!\left[-(h - h_0)/H\right]`. The
:math:`(h_0, \rho_0, H)` table is Vallado, *Fundamentals of Astrodynamics and Applications*,
4th ed., Table 8-4, which is itself a band-by-band fit to CIRA-72 at an exospheric
temperature near 1000 K, i.e. **one particular moderate level of solar activity**.

What that means in practice, stated plainly because a drag number is worthless without it:

* **It carries no solar activity.** Thermospheric density at 400 km varies by more than an
  order of magnitude over a solar cycle. This model returns the same number in 2008 and in
  2014. A decay estimate from it can be wrong by 10x, and it will be wrong in the
  optimistic direction exactly when it matters -- at solar maximum.
* **It carries no diurnal bulge.** Real density on the sunlit side peaks near 14:00 local
  solar time and is roughly 2-3x the pre-dawn minimum at the same altitude. This model is
  spherically symmetric.
* **It carries no geomagnetic storms.** A severe storm can double density at 400 km within
  hours; nothing here can represent that.
* **It carries no seasonal or latitudinal structure**, and no helium bulge.
* **Altitude is geocentric, not geodetic**: :math:`h = r - R_e` with the *equatorial*
  radius. Earth's polar radius is 21.4 km smaller, so over the poles this understates
  altitude by up to 21 km and therefore overstates density by roughly 40 % at 400 km. For a
  polar orbit that bias is systematic, not random.

Use it for order-of-magnitude lifetime and station-keeping budgets, for relative comparisons
at fixed epoch, and for verifying that a propagator's drag term is wired up correctly. Do
not use it to predict a reentry date, and do not report a decay rate from it without the
solar-activity caveat attached.

The ballistic coefficient :math:`C_D A / m` is a **required argument with no default**. There
is no such thing as "the" drag on a spacecraft: a 3U cubesat and a rocket body at the same
altitude differ by two orders of magnitude in this number. A default would let a caller
obtain a drag figure without ever having said which vehicle they meant.

Validity
--------
Relative to :mod:`rpo_core.propagate` this module adds J2 and drag and nothing else. Still
neglected: all zonal harmonics above :math:`J_2` and every tesseral term, lunisolar third
bodies, solar radiation pressure, Earth albedo, tides, relativity, and attitude-dependent
drag area (the ballistic coefficient is a scalar constant, so a tumbling or articulated
vehicle is not represented). The inertial frame is the pseudo-inertial GCRF approximation of
``docs/conventions.md``, and the Earth rotation vector is taken as constant along
:math:`+\hat{z}` with no precession, nutation, or polar motion.

Units are SI: metres, seconds, radians, kilograms.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from ._validate import as_vector, validate_positive
from .constants import J2_EARTH, MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from .exceptions import DegenerateGeometryError, PropagationError, RpoCoreError
from .propagate import DEFAULT_ATOL, DEFAULT_RTOL, two_body_derivative

#: Earth mean angular velocity (WGS-84 nominal), rad/s. Used for the co-rotating
#: atmosphere. Length-of-day variation and polar motion are neglected, consistent with the
#: pseudo-inertial frame assumption in ``docs/conventions.md``.
OMEGA_EARTH_RAD_S: float = 7.292115e-5

#: Mean rate of the Earth's orbit about the Sun, rad/s -- 360 degrees per mean tropical-ish
#: year of 365.25 days. This is the RAAN drift rate a sun-synchronous orbit must match, and
#: it is quoted here rather than derived so the sun-synchronous check has an external number
#: to hit.
SUN_SYNCHRONOUS_RAAN_RATE_RAD_S: float = 2.0 * math.pi / (365.25 * 86400.0)

#: Highest altitude in the density table (m). Above this the topmost band is extrapolated;
#: see :func:`exponential_density_kg_m3`.
MAX_TABULATED_ALTITUDE_M: float = 1000.0e3

# Vallado 4th ed., Table 8-4 (piecewise-exponential fit to CIRA-72, exospheric temperature
# ~1000 K). Columns: base altitude (km), base density (kg/m^3), scale height (km).
#
# The table is self-consistent by construction: evaluating band k at the base altitude of
# band k+1 reproduces band k+1's base density to the printed precision. That property is
# asserted in the test suite, which is what makes a transcription typo in any of the 84
# numbers below detectable rather than merely plausible.
_DENSITY_TABLE_KM_KG_M3_KM: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
)

_BASE_ALTITUDE_M: npt.NDArray[np.float64] = (
    np.array([row[0] for row in _DENSITY_TABLE_KM_KG_M3_KM], dtype=np.float64) * 1.0e3
)
_BASE_DENSITY_KG_M3: npt.NDArray[np.float64] = np.array(
    [row[1] for row in _DENSITY_TABLE_KM_KG_M3_KM], dtype=np.float64
)
_SCALE_HEIGHT_M: npt.NDArray[np.float64] = (
    np.array([row[2] for row in _DENSITY_TABLE_KM_KG_M3_KM], dtype=np.float64) * 1.0e3
)


class AtmosphericModelError(RpoCoreError, ValueError):
    """Raised when the density model is asked for an altitude it cannot represent.

    In practice this means an altitude below the reference ellipsoid's equatorial radius:
    the trajectory has intersected the Earth. Returning the sea-level density there -- or
    worse, extrapolating the surface band downwards -- would let a propagation continue
    happily underground and deliver a decay profile for a vehicle that has already impacted.
    """


def _validate_finite(value: float, name: str) -> float:
    """Return ``value`` as a validated finite float of any sign."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


# --------------------------------------------------------------------------------------
# J2
# --------------------------------------------------------------------------------------


def _j2_acceleration_unchecked(
    r_eci_m: npt.NDArray[np.float64], mu_m3_s2: float, j2: float, r_body_m: float
) -> npt.NDArray[np.float64]:
    """J2 acceleration with no input validation -- the hot path inside the derivative.

    The public wrapper validates; this one is called once per integrator stage and must not
    pay for that. It assumes ``|r| > 0``, which the two-body term has already enforced.
    """
    r_norm = float(np.linalg.norm(r_eci_m))
    z_over_r = float(r_eci_m[2]) / r_norm
    factor = -1.5 * j2 * (mu_m3_s2 / r_norm**2) * (r_body_m / r_norm) ** 2
    in_plane_bracket = 1.0 - 5.0 * z_over_r**2
    return np.array(
        (
            factor * in_plane_bracket * (float(r_eci_m[0]) / r_norm),
            factor * in_plane_bracket * (float(r_eci_m[1]) / r_norm),
            factor * (3.0 - 5.0 * z_over_r**2) * z_over_r,
        ),
        dtype=np.float64,
    )


def j2_acceleration_m_s2(
    r_eci_m: npt.ArrayLike,
    *,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
) -> npt.NDArray[np.float64]:
    r"""Return the J2 perturbing acceleration in the Earth-centred inertial frame, m/s^2.

    This is the gradient of :math:`V_{J_2}` only -- the point-mass term is *not* included,
    so the result is a small correction (order :math:`10^{-2}\ \mathrm{m/s^2}` in LEO,
    about :math:`10^{-3}` of central gravity) to be added to
    :func:`rpo_core.propagate.two_body_derivative`.

    Parameters
    ----------
    r_eci_m
        Inertial position, metres, shape (3,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Must be finite and positive.
    j2
        Second zonal harmonic, dimensionless and unnormalised. Any finite value is accepted
        including zero (a spherically symmetric body, for which the result is exactly zero)
        and negative (a prolate body).
    r_body_m
        Reference equatorial radius the harmonic is normalised to, metres. Must match the
        radius used to derive ``j2``; pairing EGM-96's ``J2`` with a different radius is a
        silent scale error of order the ratio squared.

    Returns
    -------
    numpy.ndarray
        Shape (3,), m/s^2.

    Raises
    ------
    DegenerateGeometryError
        If ``|r| = 0``, where the harmonic expansion does not converge and every term is
        singular.
    ValueError
        Wrong shape, non-finite input, non-positive ``mu_m3_s2`` or ``r_body_m``, or
        non-finite ``j2``.

    Examples
    --------
    Over the north pole the perturbation is purely radial and points *outward*: an oblate
    Earth's potential well is shallower over the poles than a sphere's.

    >>> import numpy as np
    >>> a = j2_acceleration_m_s2([0.0, 0.0, 7.0e6])
    >>> bool(a[2] > 0.0), bool(np.allclose(a[:2], 0.0))
    (True, True)

    """
    r = as_vector(r_eci_m, "r_eci_m")
    mu = validate_positive(mu_m3_s2, "mu_m3_s2")
    radius = validate_positive(r_body_m, "r_body_m")
    coefficient = _validate_finite(j2, "j2")
    if float(np.linalg.norm(r)) == 0.0:
        raise DegenerateGeometryError(
            "J2 acceleration is undefined at |r| = 0: the spherical-harmonic expansion is "
            "singular at the origin and every term diverges"
        )
    return _j2_acceleration_unchecked(r, mu, coefficient, radius)


def j2_potential_j_kg(
    r_eci_m: npt.ArrayLike,
    *,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
) -> float:
    r"""Return the J2 part of the specific potential energy, J/kg.

    :math:`V_{J_2} = \tfrac{1}{2}\mu J_2 R_e^{2} (3z^{2} - r^{2}) / r^{5}`, whose negative
    gradient is :func:`j2_acceleration_m_s2`. Exposed because it is what makes the
    conservation test possible: J2 is a conservative force, so
    :math:`v^{2}/2 - \mu/r + V_{J_2}` is invariant along a J2-perturbed trajectory even
    though the Keplerian energy alone is not.

    Raises
    ------
    DegenerateGeometryError
        If ``|r| = 0``.
    ValueError
        On malformed, non-finite, or non-positive input.

    """
    r = as_vector(r_eci_m, "r_eci_m")
    mu = validate_positive(mu_m3_s2, "mu_m3_s2")
    radius = validate_positive(r_body_m, "r_body_m")
    coefficient = _validate_finite(j2, "j2")
    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise DegenerateGeometryError("J2 potential is undefined at |r| = 0")
    z = float(r[2])
    return 0.5 * mu * coefficient * radius**2 * (3.0 * z**2 - r_norm**2) / r_norm**5


def specific_energy_with_j2_j_kg(
    state_eci: npt.ArrayLike,
    *,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
) -> float:
    r"""Return :math:`v^{2}/2 - \mu/r + V_{J_2}`, J/kg -- the invariant under J2.

    Compare :func:`rpo_core.propagate.specific_energy_j_kg`, which omits :math:`V_{J_2}`
    and therefore oscillates at roughly the :math:`J_2` level along a J2-perturbed
    trajectory. Both are useful: the difference between them is exactly the diagnostic that
    distinguishes "my integrator is drifting" from "my force model has a potential I forgot
    to account for".

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, shape (6,), metres and m/s.
    mu_m3_s2, j2, r_body_m
        As for :func:`j2_acceleration_m_s2`.

    Raises
    ------
    DegenerateGeometryError
        If ``|r| = 0``.
    ValueError
        On malformed, non-finite, or non-positive input.

    """
    state = np.asarray(state_eci, dtype=np.float64)
    if state.shape != (6,):
        raise ValueError(f"state_eci must have shape (6,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"state_eci must be finite, got {state!r}")
    mu = validate_positive(mu_m3_s2, "mu_m3_s2")
    r_norm = float(np.linalg.norm(state[:3]))
    if r_norm == 0.0:
        raise DegenerateGeometryError("specific energy is undefined at |r| = 0")
    v_norm = float(np.linalg.norm(state[3:]))
    potential = j2_potential_j_kg(state[:3], mu_m3_s2=mu, j2=j2, r_body_m=r_body_m)
    return 0.5 * v_norm**2 - mu / r_norm + potential


# --------------------------------------------------------------------------------------
# Secular first-order rates -- the closed forms the propagator is checked against
# --------------------------------------------------------------------------------------


def _semi_latus_rectum_m(semi_major_axis_m: float, eccentricity: float) -> float:
    """Return ``p = a(1 - e**2)`` with the closed-orbit preconditions enforced."""
    a = validate_positive(semi_major_axis_m, "semi_major_axis_m")
    e = _validate_finite(eccentricity, "eccentricity")
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1) for a closed orbit, got {e!r}")
    return a * (1.0 - e**2)


def secular_raan_rate_rad_s(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_rad: float,
    *,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
) -> float:
    r"""Return the first-order secular nodal regression rate, rad/s.

    .. math:: \dot{\Omega} = -\tfrac{3}{2}\, n J_2 (R_e/p)^{2} \cos i,
              \qquad p = a(1 - e^{2}),\ n = \sqrt{\mu/a^{3}}

    Negative (westward regression) for prograde orbits, zero at :math:`i = 90^{\circ}`, and
    positive for retrograde ones -- which is what makes a sun-synchronous orbit necessarily
    retrograde. This is *mean*-element theory: an osculating RAAN read off a propagated
    state also carries short-period oscillations of order :math:`J_2` that average out over
    an orbit but do not vanish pointwise.

    Raises
    ------
    ValueError
        On non-positive ``a``, ``mu_m3_s2`` or ``r_body_m``, eccentricity outside
        ``[0, 1)``, or non-finite input.

    """
    p = _semi_latus_rectum_m(semi_major_axis_m, eccentricity)
    mu = validate_positive(mu_m3_s2, "mu_m3_s2")
    radius = validate_positive(r_body_m, "r_body_m")
    coefficient = _validate_finite(j2, "j2")
    inclination = _validate_finite(inclination_rad, "inclination_rad")
    n = math.sqrt(mu / float(semi_major_axis_m) ** 3)
    return -1.5 * n * coefficient * (radius / p) ** 2 * math.cos(inclination)


def secular_arg_periapsis_rate_rad_s(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_rad: float,
    *,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
) -> float:
    r"""Return the first-order secular apsidal rotation rate, rad/s.

    .. math:: \dot{\omega} = \tfrac{3}{4}\, n J_2 (R_e/p)^{2} (5\cos^{2} i - 1)

    equivalently :math:`\tfrac{3}{4} n J_2 (R_e/p)^2 (4 - 5\sin^2 i)`.

    The bracket vanishes at :math:`\cos^{2} i = 1/5`, i.e. :math:`i = 63.4349^{\circ}` --
    the critical inclination that Molniya orbits are flown at precisely so that apogee stays
    over the northern hemisphere. Any variant of this formula that does not put its zero at
    63.43 degrees is wrong, and that is the cheapest available check on it; see
    ``test_critical_inclination_is_where_apsidal_drift_vanishes``.

    Raises
    ------
    ValueError
        Same conditions as :func:`secular_raan_rate_rad_s`.

    """
    p = _semi_latus_rectum_m(semi_major_axis_m, eccentricity)
    mu = validate_positive(mu_m3_s2, "mu_m3_s2")
    radius = validate_positive(r_body_m, "r_body_m")
    coefficient = _validate_finite(j2, "j2")
    inclination = _validate_finite(inclination_rad, "inclination_rad")
    n = math.sqrt(mu / float(semi_major_axis_m) ** 3)
    return 0.75 * n * coefficient * (radius / p) ** 2 * (5.0 * math.cos(inclination) ** 2 - 1.0)


def sun_synchronous_inclination_rad(
    semi_major_axis_m: float,
    eccentricity: float = 0.0,
    *,
    target_raan_rate_rad_s: float = SUN_SYNCHRONOUS_RAAN_RATE_RAD_S,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
) -> float:
    r"""Return the inclination whose J2 nodal drift equals ``target_raan_rate_rad_s``.

    Inverting :func:`secular_raan_rate_rad_s` for :math:`i`:

    .. math:: \cos i = -\frac{\dot{\Omega}_{\mathrm{target}}}
                             {\tfrac{3}{2} n J_2 (R_e/p)^{2}}

    The default target is 360 degrees per 365.25 days, which makes the orbit plane hold a
    fixed angle to the Sun line. Because the required :math:`\cos i` is negative, every
    sun-synchronous orbit is retrograde -- around 98 degrees in LEO, rising with altitude.

    Raises
    ------
    ValueError
        If no inclination satisfies the request, i.e. :math:`|\cos i| > 1`. This happens for
        orbits too high for J2 to drive the node fast enough; the message reports the
        required cosine and the fastest achievable rate so the caller can see how far out
        the request was.

    """
    # Rate at i = 0, i.e. the fastest nodal drift J2 can supply at this a and e.
    max_rate = -secular_raan_rate_rad_s(
        semi_major_axis_m,
        eccentricity,
        0.0,
        mu_m3_s2=mu_m3_s2,
        j2=j2,
        r_body_m=r_body_m,
    )
    target = _validate_finite(target_raan_rate_rad_s, "target_raan_rate_rad_s")
    if max_rate == 0.0:
        raise ValueError(
            "J2 supplies no nodal drift at all (j2 = 0), so no inclination is "
            f"sun-synchronous; target_raan_rate_rad_s={target!r} rad/s is unreachable"
        )
    cos_i = -target / max_rate
    if abs(cos_i) > 1.0:
        raise ValueError(
            f"no inclination gives a nodal rate of {target:.6e} rad/s at "
            f"semi_major_axis_m={float(semi_major_axis_m):.6e}, eccentricity="
            f"{float(eccentricity):.6g}: it would need cos(i) = {cos_i:.6f}, and the "
            f"fastest rate J2 can supply here is {max_rate:.6e} rad/s at i = 0"
        )
    return math.acos(cos_i)


# --------------------------------------------------------------------------------------
# Atmosphere and drag
# --------------------------------------------------------------------------------------


def exponential_density_kg_m3(altitude_m: float) -> float:
    r"""Return atmospheric mass density from the piecewise-exponential model, kg/m^3.

    :math:`\rho(h) = \rho_0 \exp[-(h - h_0)/H]` using the band containing ``h``. See the
    module docstring for provenance (Vallado Table 8-4 / CIRA-72 at ~1000 K exospheric
    temperature) and, more importantly, for the list of things this model does not know
    about -- solar activity, the diurnal bulge, and geomagnetic storms among them. It can be
    wrong by an order of magnitude and it is wrong in a *biased*, not random, way.

    Parameters
    ----------
    altitude_m
        Geometric altitude above the WGS-84 equatorial radius, metres. Must be finite and
        non-negative.

    Returns
    -------
    float
        Density, kg/m^3. Above :data:`MAX_TABULATED_ALTITUDE_M` the topmost band is
        extrapolated; that is a formal extrapolation with no data behind it, but the value
        it produces is already below 3e-15 kg/m^3 and falls exponentially, so drag there is
        far below the perturbations this module does not model at all.

    Raises
    ------
    AtmosphericModelError
        If ``altitude_m`` is negative -- the trajectory is inside the Earth.
    ValueError
        If ``altitude_m`` is not finite.

    Examples
    --------
    >>> f"{exponential_density_kg_m3(400.0e3):.3e}"
    '3.725e-12'
    >>> exponential_density_kg_m3(800.0e3) < exponential_density_kg_m3(400.0e3)
    True

    """
    altitude = float(altitude_m)
    if not math.isfinite(altitude):
        raise ValueError(f"altitude_m must be finite, got {altitude_m!r}")
    if altitude < 0.0:
        raise AtmosphericModelError(
            f"altitude_m = {altitude:.6g} m is below the reference ellipsoid: the "
            "trajectory has intersected the Earth, and there is no atmospheric density to "
            "report for a vehicle that has already impacted"
        )
    return _density_unchecked(altitude)


def _density_unchecked(altitude_m: float) -> float:
    """Density lookup with no validation -- the hot path inside the derivative."""
    index = int(np.searchsorted(_BASE_ALTITUDE_M, altitude_m, side="right")) - 1
    if index < 0:
        index = 0
    return float(
        _BASE_DENSITY_KG_M3[index]
        * math.exp(-(altitude_m - _BASE_ALTITUDE_M[index]) / _SCALE_HEIGHT_M[index])
    )


def _drag_acceleration_unchecked(
    r_eci_m: npt.NDArray[np.float64],
    v_eci_m_s: npt.NDArray[np.float64],
    ballistic_coefficient_m2_kg: float,
    r_body_m: float,
    omega_earth_rad_s: float,
) -> npt.NDArray[np.float64]:
    """Drag acceleration with no input validation -- the hot path inside the derivative.

    Raises :class:`AtmosphericModelError` through :func:`_density_unchecked`'s caller
    contract: altitude is checked here because a decaying orbit reaching the ground is a
    physical event the propagation must stop on, not a validation concern.
    """
    r_norm = float(np.linalg.norm(r_eci_m))
    altitude = r_norm - r_body_m
    if altitude < 0.0:
        raise AtmosphericModelError(
            f"trajectory reached |r| = {r_norm:.6g} m, i.e. {altitude:.6g} m altitude "
            f"above r_body_m = {r_body_m:.6g} m: the vehicle has impacted and the density "
            "model has nothing to say below the surface"
        )
    density = _density_unchecked(altitude)
    # Co-rotating atmosphere: omega x r with omega = [0, 0, w].
    v_rel = np.array(
        (
            float(v_eci_m_s[0]) + omega_earth_rad_s * float(r_eci_m[1]),
            float(v_eci_m_s[1]) - omega_earth_rad_s * float(r_eci_m[0]),
            float(v_eci_m_s[2]),
        ),
        dtype=np.float64,
    )
    speed = float(np.linalg.norm(v_rel))
    return -0.5 * density * ballistic_coefficient_m2_kg * speed * v_rel


def drag_acceleration_m_s2(
    r_eci_m: npt.ArrayLike,
    v_eci_m_s: npt.ArrayLike,
    ballistic_coefficient_m2_kg: float,
    *,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
    omega_earth_rad_s: float = OMEGA_EARTH_RAD_S,
) -> npt.NDArray[np.float64]:
    r"""Return the atmospheric drag acceleration in the inertial frame, m/s^2.

    .. math::

        \mathbf{v}_{\mathrm{rel}} = \mathbf{v} - \boldsymbol{\omega}_\oplus \times \mathbf{r},
        \qquad
        \mathbf{a} = -\tfrac{1}{2} \rho(h) \frac{C_D A}{m}
                     \lVert \mathbf{v}_{\mathrm{rel}} \rVert \mathbf{v}_{\mathrm{rel}}

    Parameters
    ----------
    r_eci_m, v_eci_m_s
        Inertial position (m) and velocity (m/s), shape (3,) each.
    ballistic_coefficient_m2_kg
        :math:`C_D A / m` in m^2/kg. **Required, with no default.** Note this is the
        *inverse* of the quantity some references call the ballistic coefficient
        (:math:`m / C_D A`, kg/m^2); the unit in the name is the disambiguator. Typical
        values: ~0.01 for a large LEO platform (:math:`C_D = 2.2`, 10 m^2, 2200 kg), ~0.02
        for a 3U cubesat, ~0.2 for a deployed drag sail. Must be finite and non-negative;
        zero is accepted and means "no drag", which is the limiting case the tests use.
    r_body_m
        Radius the altitude is measured from, metres. Geocentric, not geodetic -- see the
        module docstring for the ~40 % polar density bias this introduces.
    omega_earth_rad_s
        Rotation rate of the co-rotating atmosphere about ``+z``, rad/s. Zero models a
        non-rotating atmosphere, which is wrong by up to 465 m/s in relative speed at the
        equator (about 6 % of orbital velocity, so ~12 % in drag magnitude).

    Returns
    -------
    numpy.ndarray
        Shape (3,), m/s^2. Anti-parallel to ``v_rel`` by construction, so it can only remove
        energy.

    Raises
    ------
    AtmosphericModelError
        If the position is below ``r_body_m``.
    ValueError
        Wrong shape, non-finite input, negative ``ballistic_coefficient_m2_kg`` (which would
        make drag a thruster), or non-positive ``r_body_m``.

    """
    r = as_vector(r_eci_m, "r_eci_m")
    v = as_vector(v_eci_m_s, "v_eci_m_s")
    radius = validate_positive(r_body_m, "r_body_m")
    omega = _validate_finite(omega_earth_rad_s, "omega_earth_rad_s")
    bc = _validate_finite(ballistic_coefficient_m2_kg, "ballistic_coefficient_m2_kg")
    if bc < 0.0:
        raise ValueError(
            "ballistic_coefficient_m2_kg must be >= 0, got "
            f"{ballistic_coefficient_m2_kg!r}; a negative value would make drag accelerate "
            "the spacecraft along its velocity, which is a thruster, not an atmosphere"
        )
    return _drag_acceleration_unchecked(r, v, bc, radius, omega)


# --------------------------------------------------------------------------------------
# Perturbed propagation
# --------------------------------------------------------------------------------------


def perturbed_derivative(
    _t: float,
    state: npt.NDArray[np.float64],
    mu_m3_s2: float,
    j2: float | None,
    r_body_m: float,
    ballistic_coefficient_m2_kg: float | None,
    omega_earth_rad_s: float,
) -> npt.NDArray[np.float64]:
    """Return ``d/dt [r, v]`` for two-body motion plus the enabled perturbations.

    ``None`` disables a term, rather than a boolean flag beside a value: it makes it
    impossible to express "drag is off but here is the ballistic coefficient anyway", the
    state in which a caller believes they are modelling drag and are not.

    Signature matches what ``solve_ivp`` expects. ``_t`` is unused -- the dynamics are
    autonomous, because the atmosphere model has no epoch (see the module docstring).
    """
    derivative = two_body_derivative(_t, state, mu_m3_s2)
    if j2 is not None:
        derivative[3:] += _j2_acceleration_unchecked(state[:3], mu_m3_s2, j2, r_body_m)
    if ballistic_coefficient_m2_kg is not None:
        derivative[3:] += _drag_acceleration_unchecked(
            state[:3],
            state[3:],
            ballistic_coefficient_m2_kg,
            r_body_m,
            omega_earth_rad_s,
        )
    return derivative


def propagate_perturbed(
    state0_eci: npt.ArrayLike,
    times_s: npt.ArrayLike,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    enable_j2: bool = False,
    enable_drag: bool = False,
    ballistic_coefficient_m2_kg: float | None = None,
    j2: float = J2_EARTH,
    r_body_m: float = R_EARTH_EQUATORIAL_M,
    omega_earth_rad_s: float = OMEGA_EARTH_RAD_S,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> npt.NDArray[np.float64]:
    """Propagate an inertial state under two-body motion plus selectable perturbations.

    Same interface shape as :func:`rpo_core.propagate.propagate_two_body`, and with both
    switches off it integrates exactly the same right-hand side -- which is the limiting
    case the test suite asserts.

    Parameters
    ----------
    state0_eci
        Initial state ``[r(3), v(3)]``, metres and m/s, shape (6,).
    times_s
        Output times, seconds from the epoch of ``state0_eci``. Non-decreasing, starting
        at 0.0.
    mu_m3_s2
        Gravitational parameter, m^3/s^2.
    enable_j2
        Include the J2 zonal harmonic.
    enable_drag
        Include atmospheric drag. Requires ``ballistic_coefficient_m2_kg``.
    ballistic_coefficient_m2_kg
        :math:`C_D A / m`, m^2/kg. Required when ``enable_drag`` is true and rejected when
        it is false, so the two can never disagree about whether drag is being modelled.
    j2, r_body_m, omega_earth_rad_s
        Model parameters; see :func:`j2_acceleration_m_s2` and
        :func:`drag_acceleration_m_s2`.
    rtol, atol
        Integrator tolerances. Note that a perturbed trajectory is *not* converged simply
        because the two-body one was at the same setting: the perturbing accelerations are
        3-8 orders of magnitude smaller than the central term, so their contribution is the
        part most easily lost to truncation. Sweep before quoting.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times_s), 6)``.

    Raises
    ------
    PropagationError
        If the integrator fails, or returns fewer states than were requested. Never a
        truncated trajectory.
    AtmosphericModelError
        If a drag-enabled trajectory decays below ``r_body_m``. Surfaced rather than
        clamped: a propagation that continued underground would report a decay profile for
        a vehicle that had already impacted.
    ValueError
        Malformed or non-finite state, malformed time schedule, or a ``enable_drag`` /
        ``ballistic_coefficient_m2_kg`` combination that contradicts itself.

    Examples
    --------
    >>> import numpy as np
    >>> from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
    >>> a = R_EARTH_EQUATORIAL_M + 700.0e3
    >>> v = np.sqrt(MU_EARTH_M3_S2 / a)
    >>> state = [a, 0.0, 0.0, 0.0, v * np.cos(0.9), v * np.sin(0.9)]
    >>> out = propagate_perturbed(state, [0.0, 600.0], enable_j2=True)
    >>> out.shape
    (2, 6)

    """
    state0 = np.asarray(state0_eci, dtype=np.float64)
    if state0.shape != (6,):
        raise ValueError(f"state0_eci must have shape (6,), got {state0.shape}")
    if not np.all(np.isfinite(state0)):
        raise ValueError(f"state0_eci must be finite, got {state0!r}")

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"times_s must be a non-empty 1-D array, got shape {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_s must be finite")
    if times[0] != 0.0:
        raise ValueError(f"times_s must start at 0.0, got {times[0]!r}")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times_s must be non-decreasing")

    mu = validate_positive(mu_m3_s2, "mu_m3_s2")
    radius = validate_positive(r_body_m, "r_body_m")
    omega = _validate_finite(omega_earth_rad_s, "omega_earth_rad_s")

    j2_arg: float | None = None
    if enable_j2:
        j2_arg = _validate_finite(j2, "j2")

    bc_arg: float | None = None
    if enable_drag:
        if ballistic_coefficient_m2_kg is None:
            raise ValueError(
                "enable_drag=True requires an explicit ballistic_coefficient_m2_kg "
                "(C_D * A / m, m^2/kg). There is no default: a 3U cubesat and a spent "
                "upper stage differ by two orders of magnitude in this number, so a drag "
                "result without one names no spacecraft and means nothing"
            )
        bc_arg = _validate_finite(ballistic_coefficient_m2_kg, "ballistic_coefficient_m2_kg")
        if bc_arg < 0.0:
            raise ValueError(
                "ballistic_coefficient_m2_kg must be >= 0, got "
                f"{ballistic_coefficient_m2_kg!r}; a negative value would make drag "
                "accelerate the spacecraft"
            )
    elif ballistic_coefficient_m2_kg is not None:
        raise ValueError(
            f"ballistic_coefficient_m2_kg={ballistic_coefficient_m2_kg!r} was given but "
            "enable_drag is False, so it would be silently ignored. Set enable_drag=True "
            "or drop the argument"
        )

    if times.size == 1:
        return state0.reshape(1, 6).copy()

    solution = solve_ivp(
        perturbed_derivative,
        (0.0, float(times[-1])),
        state0,
        method="DOP853",
        t_eval=times,
        args=(mu, j2_arg, radius, bc_arg, omega),
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        enabled = (
            ", ".join([name for name, on in (("J2", enable_j2), ("drag", enable_drag)) if on])
            or "none"
        )
        raise PropagationError(
            "perturbed propagation failed at t = "
            f"{solution.t[-1] if solution.t.size else 0.0:.6g} s of "
            f"{float(times[-1]):.6g} s requested (perturbations enabled: {enabled}): "
            f"{solution.message}"
        )
    if solution.y.shape[1] != times.size:
        raise PropagationError(
            f"integrator returned {solution.y.shape[1]} states for {times.size} requested "
            "times; the trajectory is incomplete"
        )
    return np.ascontiguousarray(solution.y.T)
