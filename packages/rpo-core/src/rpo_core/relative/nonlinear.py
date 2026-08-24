r"""Nonlinear relative motion, by differencing two absolutely-propagated orbits.

Why this module exists
----------------------
Clohessy-Wiltshire is a *linearisation* of the difference between two two-body orbits,
about a circular reference. This module computes that difference without linearising:
propagate the target and the chaser independently under full two-body dynamics, then
express the difference in the target's rotating Hill frame at every output epoch.

That makes it the **external oracle** for :mod:`rpo_core.relative.cw`. Every test in
``test_cw.py`` validates CW against closed-form properties of CW itself; the drift-free and
2:1-ellipse checks are genuine independent physics, but none of them answers the question
that actually matters -- *how wrong is the linear model?* This module answers it by
measurement.

What this isolates, and what it does not
---------------------------------------
CW makes two separate approximations, and they must not be confused:

1. **Linearisation** in the separation :math:`\rho`, with leading neglected terms of order
   :math:`(\rho/r)`.
2. **A circular reference orbit.**

Driving this module with an *exactly circular* target isolates error source (1) cleanly:
any disagreement with CW is linearisation error and nothing else. Eccentricity is a
separate study with a separate test, and conflating the two produces an error budget that
cannot be attributed.

This module is still a two-body model. It carries no J2, no drag, no third-body
perturbation. It is the right oracle for CW and the wrong oracle for anything else.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.typing as npt

from ..constants import MU_EARTH_M3_S2
from ..frames import relative_state_eci_to_hill, relative_state_hill_to_eci
from ..propagate import DEFAULT_ATOL, DEFAULT_RTOL, propagate_two_body


def propagate_relative_nonlinear(
    r_target0_eci_m: npt.ArrayLike,
    v_target0_eci_m_s: npt.ArrayLike,
    relative_state0_hill: npt.ArrayLike,
    times_s: npt.ArrayLike,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> npt.NDArray[np.float64]:
    """Propagate a relative state without linearising the dynamics.

    The chaser's absolute state is reconstructed from the initial relative state, both
    vehicles are propagated independently under two-body dynamics, and the difference is
    re-expressed in the target's Hill frame at each output time. The Hill frame is
    recomputed from the *propagated* target state at every epoch, so the frame rotation
    and its angular velocity stay consistent with the trajectory rather than being frozen
    at the epoch.

    Parameters
    ----------
    r_target0_eci_m, v_target0_eci_m_s
        Target inertial state at the epoch, shape (3,) each.
    relative_state0_hill
        Initial relative state ``[x, y, z, xdot, ydot, zdot]`` in the target Hill frame.
    times_s
        Output times, seconds from the epoch. Must be non-decreasing and start at 0.0.
    mu_m3_s2
        Gravitational parameter, m^3/s^2.
    rtol, atol
        Integrator tolerances passed to :func:`rpo_core.propagate.propagate_two_body`.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times_s), 6)`` -- Hill-frame relative states, metres and m/s.

    Raises
    ------
    PropagationError
        If either propagation fails.

    """
    r_t0 = np.asarray(r_target0_eci_m, dtype=np.float64)
    v_t0 = np.asarray(v_target0_eci_m_s, dtype=np.float64)
    r_c0, v_c0 = relative_state_hill_to_eci(r_t0, v_t0, relative_state0_hill)

    times = np.asarray(times_s, dtype=np.float64)
    target = propagate_two_body(np.concatenate((r_t0, v_t0)), times, mu_m3_s2, rtol=rtol, atol=atol)
    chaser = propagate_two_body(np.concatenate((r_c0, v_c0)), times, mu_m3_s2, rtol=rtol, atol=atol)

    relative = np.empty((times.size, 6), dtype=np.float64)
    for index in range(times.size):
        relative[index] = relative_state_eci_to_hill(
            target[index, :3], target[index, 3:], chaser[index, :3], chaser[index, 3:]
        )
    return relative


def cw_position_error_m(
    r_target0_eci_m: npt.ArrayLike,
    v_target0_eci_m_s: npt.ArrayLike,
    relative_state0_hill: npt.ArrayLike,
    times_s: npt.ArrayLike,
    n_rad_s: float,
    mu_m3_s2: float = MU_EARTH_M3_S2,
) -> npt.NDArray[np.float64]:
    """Return the CW position error against the nonlinear reference at each output time.

    Positive by construction: this is the magnitude of the difference between the linear
    and nonlinear relative position vectors, in metres.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times_s),)``, metres.

    """
    from .cw import propagate_cw  # local import: avoids a cycle at module import time

    times = np.asarray(times_s, dtype=np.float64)
    truth = propagate_relative_nonlinear(
        r_target0_eci_m, v_target0_eci_m_s, relative_state0_hill, times, mu_m3_s2
    )
    linear = np.array([propagate_cw(n_rad_s, relative_state0_hill, float(t)) for t in times])
    error = np.linalg.norm(truth[:, :3] - linear[:, :3], axis=1)
    return np.asarray(error, dtype=np.float64)


#: Measured coefficient of the one-orbit CW linearisation error law, dimensionless.
#:
#: Determined by direct measurement against nonlinear two-body motion, not assumed:
#: ``err_1_orbit = CW_ERROR_COEFFICIENT * rho**2 / r`` reproduces to six significant
#: figures across 400/800/1500 km altitudes and 1 km/10 km separations, with the
#: coefficient landing on exactly ``6*pi``. Conditions: circular target orbit, chaser at a
#: pure along-track offset with zero initial Hill-frame relative velocity.
#:
#: The law is empirical. An analytic derivation from the secular along-track drift of the
#: induced semi-major-axis difference is a worthwhile follow-up and is **not** claimed here.
CW_ERROR_COEFFICIENT: float = 6.0 * math.pi


#: Safety factor converting the central estimate into a conservative upper bound.
#:
#: Set from measurement, not judgement. Scanning measured-error / linear-estimate across
#: 400/800/1500 km altitudes, 100 m/1 km/5 km separations and 0.1-3.0 orbits, the linear
#: law under-predicts by at most a factor of **1.2253** (worst case: 800 km, 100 m,
#: 0.7 orbits). 1.5 clears that worst case with 1.22x headroom.
#:
#: The under-prediction is real and matters: growth in time is NOT linear. It is
#: super-linear between roughly 0.4 and 1.0 orbits and sub-linear below that, so the plain
#: linear law is optimistic in exactly the regime where a half-orbit V-bar hop operates.
CW_ERROR_SAFETY_FACTOR: float = 1.5


def estimated_cw_error_m(
    separation_m: float, orbit_radius_m: float, n_orbits: float = 1.0
) -> float:
    """Return the *central estimate* of CW linearisation position error, metres.

    Uses the measured law ``6*pi * rho**2 / r`` scaled linearly in elapsed orbits.

    **This is an estimate, not a bound.** Measured accuracy over 0.1-3.0 orbits: it
    over-predicts by up to 4.7x below ~0.4 orbits and under-predicts by up to 1.23x
    between ~0.4 and 1.0 orbits. For any guard or go/no-go decision use
    :func:`conservative_cw_error_bound_m` instead -- a validity check that under-warns is
    worse than no check at all.

    Parameters
    ----------
    separation_m
        Chaser-target separation, metres.
    orbit_radius_m
        Target orbit radius, metres.
    n_orbits
        Elapsed time in orbital periods.

    Returns
    -------
    float
        Estimated position error, metres.

    """
    if separation_m < 0.0:
        raise ValueError(f"separation_m must be >= 0, got {separation_m!r}")
    if orbit_radius_m <= 0.0:
        raise ValueError(f"orbit_radius_m must be > 0, got {orbit_radius_m!r}")
    if n_orbits < 0.0:
        raise ValueError(f"n_orbits must be >= 0, got {n_orbits!r}")
    return CW_ERROR_COEFFICIENT * separation_m**2 / orbit_radius_m * n_orbits


def conservative_cw_error_bound_m(
    separation_m: float, orbit_radius_m: float, n_orbits: float = 1.0
) -> float:
    """Return a conservative upper bound on CW linearisation position error, metres.

    The central estimate scaled by :data:`CW_ERROR_SAFETY_FACTOR`. Verified against direct
    measurement to over-predict everywhere on a grid spanning three altitudes, three
    separations and 0.1-3.0 orbits. Use this, not :func:`estimated_cw_error_m`, for any
    decision about whether CW is admissible for a scenario.
    """
    return CW_ERROR_SAFETY_FACTOR * estimated_cw_error_m(separation_m, orbit_radius_m, n_orbits)


def check_cw_validity(
    separation_m: float,
    n_rad_s: float,
    n_orbits: float = 1.0,
    *,
    tolerance_m: float,
    mu_m3_s2: float = MU_EARTH_M3_S2,
) -> None:
    """Warn if CW linearisation error is likely to exceed ``tolerance_m``.

    Call this **once per run**, at the scenario boundary -- not inside a propagation or
    Monte Carlo loop. The check is a modelling decision, not a per-step guard, and issuing
    a warning 100 000 times would train the user to ignore it.

    Parameters
    ----------
    separation_m
        Largest chaser-target separation the scenario will reach.
    n_rad_s
        Target mean motion, rad/s. The orbit radius is recovered as ``(mu/n**2)**(1/3)``.
    n_orbits
        Scenario duration in orbital periods.
    mu_m3_s2
        Gravitational parameter, m^3/s^2, used to recover the orbit radius from ``n``.
    tolerance_m
        Position error the scenario can tolerate. A sensible default is a small fraction
        of the tightest safety distance in play, e.g. 1 % of the keep-out-zone radius.

    Warns
    -----
    UserWarning
        If the estimated error exceeds ``tolerance_m``, with the numbers that motivated it.

    """
    if tolerance_m <= 0.0:
        raise ValueError(f"tolerance_m must be > 0, got {tolerance_m!r}")
    orbit_radius_m = (mu_m3_s2 / n_rad_s**2) ** (1.0 / 3.0)
    # Guard on the conservative bound, never the central estimate: the plain linear law is
    # optimistic between ~0.4 and 1.0 orbits, which is precisely the regime a half-orbit
    # V-bar hop lives in.
    estimate = conservative_cw_error_bound_m(separation_m, orbit_radius_m, n_orbits)
    if estimate > tolerance_m:
        warnings.warn(
            f"Clohessy-Wiltshire is likely outside its useful envelope for this scenario: "
            f"at {separation_m:,.0f} m separation over {n_orbits:.2f} orbits the estimated "
            f"linearisation error is {estimate:,.1f} m, exceeding the {tolerance_m:,.1f} m "
            "tolerance. Use nonlinear relative propagation "
            "(rpo_core.relative.nonlinear) for this regime.",
            UserWarning,
            stacklevel=2,
        )
