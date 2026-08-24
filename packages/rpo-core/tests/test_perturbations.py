"""J2 and drag: closed-form secular rates, conservation laws, limiting cases, raise paths.

The headline is ``test_j2_raan_drift_matches_the_closed_form_within_one_percent``. Nodal
regression is the cheapest externally-known consequence of J2 that a wrong implementation
cannot accidentally reproduce: it depends on the sign, the magnitude, the ``(Re/r)**2``
scaling, *and* the 1-versus-3 asymmetry between the in-plane and z brackets. Drop any one of
them and the measured rate moves by far more than the 1 % bound.

Every numerical bound below was set by running the thing first and writing the measurement
into the comment beside it.
"""

import math
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from rpo_core.constants import (
    J2_EARTH,
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    orbital_period_s,
)
from rpo_core.elements import (
    ClassicalElements,
    cartesian_to_classical,
    classical_to_cartesian,
)
from rpo_core.exceptions import DegenerateGeometryError, PropagationError
from rpo_core.perturbations import (
    _DENSITY_TABLE_KM_KG_M3_KM,
    MAX_TABULATED_ALTITUDE_M,
    OMEGA_EARTH_RAD_S,
    SUN_SYNCHRONOUS_RAAN_RATE_RAD_S,
    AtmosphericModelError,
    drag_acceleration_m_s2,
    exponential_density_kg_m3,
    j2_acceleration_m_s2,
    j2_potential_j_kg,
    perturbed_derivative,
    propagate_perturbed,
    secular_arg_periapsis_rate_rad_s,
    secular_raan_rate_rad_s,
    specific_energy_with_j2_j_kg,
    sun_synchronous_inclination_rad,
)
from rpo_core.propagate import propagate_two_body, specific_energy_j_kg

A_700_M = R_EARTH_EQUATORIAL_M + 700.0e3
A_300_M = R_EARTH_EQUATORIAL_M + 300.0e3


def _circular_state(a_m: float, inclination_rad: float) -> np.ndarray:
    """Exactly circular state at ascending-node crossing, so `a` has no eccentric wobble."""
    speed = math.sqrt(MU_EARTH_M3_S2 / a_m)
    return np.array(
        [
            a_m,
            0.0,
            0.0,
            0.0,
            speed * math.cos(inclination_rad),
            speed * math.sin(inclination_rad),
        ]
    )


def _fit_secular_rate(times_s: np.ndarray, angles_rad: np.ndarray) -> float:
    """Least-squares slope of an unwrapped angle history, rad/s.

    Osculating elements carry short-period oscillations of order J2 on top of the secular
    trend. Fitting over a whole number of orbits averages them out; taking an endpoint
    difference instead would inherit the full oscillation amplitude, which for RAAN here is
    0.049 deg against a 5.9 deg secular signal -- 0.8 %, i.e. it would eat most of the 1 %
    budget before any real error had a chance to show up.
    """
    unwrapped = np.unwrap(angles_rad)
    design = np.vstack([times_s, np.ones_like(times_s)]).T
    slope, _ = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
    return float(slope)


def _propagate_and_fit(
    a_m: float,
    eccentricity: float,
    inclination_rad: float,
    n_orbits: int = 20,
    samples_per_orbit: int = 20,
    **kwargs: float,
) -> tuple[float, float]:
    """Propagate with J2 and return the fitted (RAAN rate, argument-of-perigee rate)."""
    state0 = classical_to_cartesian(
        ClassicalElements(a_m, eccentricity, inclination_rad, 0.4, 0.7, 0.0), MU_EARTH_M3_S2
    )
    period = orbital_period_s(a_m)
    times = np.linspace(0.0, n_orbits * period, n_orbits * samples_per_orbit + 1)
    states = propagate_perturbed(state0, times, enable_j2=True, **kwargs)
    elements = [cartesian_to_classical(state) for state in states]
    return (
        _fit_secular_rate(times, np.array([e.raan_rad for e in elements])),
        _fit_secular_rate(times, np.array([e.arg_periapsis_rad for e in elements])),
    )


# ======================================================================================
# THE HEADLINE: secular RAAN drift against the closed form
# ======================================================================================


@pytest.mark.integration
def test_j2_raan_drift_matches_the_closed_form_within_one_percent():
    r"""Nodal regression must match ``-3/2 n J2 (Re/p)^2 cos i`` to 1 %.

    Model M6's stated independent check, and the single most valuable physics validation in
    the project. The propagator shares no code with
    :func:`secular_raan_rate_rad_s`: one integrates a Cartesian acceleration with DOP853,
    the other evaluates a mean-element formula, so agreement is evidence about the physics
    rather than about a shared helper.

    Measured: -8.696603e-07 rad/s against a closed form of -8.685443e-07 rad/s, a
    disagreement of **0.1285 %** over 20 orbits of a 700 km, e = 0.01, i = 51.6 deg orbit.
    The 1 % bound therefore carries roughly 8x headroom.

    That residual 0.13 % is not integrator error -- it is unchanged from rtol 1e-9 to 1e-12
    (see ``test_raan_drift_is_not_an_integrator_artefact``) -- it is the second-order term
    the first-order theory omits. It tracks ``J2 (Re/p)**2`` across altitude, which is what
    ``test_raan_drift_agreement_improves_at_higher_altitude`` asserts.
    """
    eccentricity, inclination = 0.01, math.radians(51.6)
    measured, _ = _propagate_and_fit(A_700_M, eccentricity, inclination)
    closed_form = secular_raan_rate_rad_s(A_700_M, eccentricity, inclination)

    disagreement = abs(measured / closed_form - 1.0)
    assert disagreement < 0.01, (
        f"measured RAAN rate {measured:.8e} rad/s vs closed form {closed_form:.8e} rad/s: "
        f"{disagreement * 100:.4f} % disagreement"
    )
    # Sign matters independently of magnitude: a prograde orbit regresses westward.
    assert measured < 0.0


@pytest.mark.slow
@pytest.mark.integration
def test_raan_drift_is_not_an_integrator_artefact():
    """The 0.13 % residual must be insensitive to integrator tolerance.

    A number that moves when rtol moves is a numerical setting, not physics. Measured
    across rtol = atol = 1e-9, 1e-11 and 1e-12 the fitted rate agrees to 8 significant
    figures (-8.6966034e-07 in every case), so the bound below is 1e-6 relative -- two
    decades above the observed 1e-9 spread.
    """
    eccentricity, inclination = 0.01, math.radians(51.6)
    rates = [
        _propagate_and_fit(A_700_M, eccentricity, inclination, n_orbits=10, rtol=tol, atol=tol)[0]
        for tol in (1e-9, 1e-11, 1e-12)
    ]
    spread = (max(rates) - min(rates)) / abs(rates[-1])
    assert spread < 1e-6, f"RAAN rate moved with integrator tolerance: {rates}"


@pytest.mark.slow
@pytest.mark.integration
def test_raan_drift_agreement_improves_at_higher_altitude():
    """Convergence behaviour, not a single threshold: the residual is a truncated series.

    First-order secular theory drops terms of order ``J2 * (Re/p)**2`` relative to the terms
    it keeps, so the disagreement must shrink as the orbit rises. Measured disagreement at
    400 / 700 / 1500 / 3000 km altitude: 0.1401 / 0.1285 / 0.1039 / 0.0735 %, against
    ``J2 (Re/p)**2`` of 0.0959 / 0.0879 / 0.0710 / 0.0501 % -- the same factor of ~1.46
    throughout. That constant ratio is much stronger evidence of a correct implementation
    than any single bound: a scaling error in the ``(Re/r)**2`` factor would break the trend
    rather than merely shift it.
    """
    eccentricity, inclination = 0.01, math.radians(51.6)
    disagreements = []
    for altitude_m in (400.0e3, 700.0e3, 1500.0e3, 3000.0e3):
        a = R_EARTH_EQUATORIAL_M + altitude_m
        measured, _ = _propagate_and_fit(a, eccentricity, inclination)
        closed_form = secular_raan_rate_rad_s(a, eccentricity, inclination)
        disagreements.append(abs(measured / closed_form - 1.0))

    assert all(earlier > later for earlier, later in pairwise(disagreements)), (
        f"disagreement did not shrink with altitude: {[f'{d * 100:.4f} %' for d in disagreements]}"
    )
    assert max(disagreements) < 0.01


@pytest.mark.integration
def test_sun_synchronous_orbit_drifts_at_the_earths_orbital_rate():
    """The independent cross-check: 98.60 deg at 800 km must give 360 deg/year of nodal drift.

    Sun-synchrony is a physically meaningful, externally verifiable target that has nothing
    to do with how this module is written. The closed form is inverted for the inclination,
    that inclination is fed to the *numerical* propagator, and the fitted drift is compared
    against 360 deg / 365.25 days.

    Measured at 800 km, e = 1e-3: required inclination 98.6029 deg, fitted drift
    1.991296e-07 rad/s versus a target of 1.991021e-07 rad/s -- 360.0496 deg/year against
    360, i.e. **0.0138 %**. Bound set at 1 % (70x headroom) so that the test measures the
    force model rather than the fit residual.

    Note for anyone matching this against a textbook figure: 97.79 deg is the sun-synchronous
    inclination at *600* km, not 800 km. See
    ``test_sun_synchronous_inclination_rises_with_altitude``.
    """
    a = R_EARTH_EQUATORIAL_M + 800.0e3
    eccentricity = 1.0e-3  # elements are undefined below e = 1e-8; 1e-3 is safely above it
    inclination = sun_synchronous_inclination_rad(a, eccentricity)
    assert math.degrees(inclination) == pytest.approx(98.6029, abs=1e-3)

    measured, _ = _propagate_and_fit(a, eccentricity, inclination)
    disagreement = abs(measured / SUN_SYNCHRONOUS_RAAN_RATE_RAD_S - 1.0)
    assert disagreement < 0.01, (
        f"sun-synchronous drift {math.degrees(measured) * 86400.0 * 365.25:.4f} deg/year "
        f"instead of 360 ({disagreement * 100:.4f} % off)"
    )
    # Retrograde: eastward nodal drift is only reachable with cos(i) < 0.
    assert inclination > math.pi / 2.0


@pytest.mark.slow
@pytest.mark.integration
def test_sun_synchronous_check_holds_at_600_km_too():
    """Same check at the altitude where the familiar 97.79 deg figure actually applies.

    Measured: inclination 97.7875 deg, fitted drift 1.991283e-07 rad/s, 0.0131 % from the
    360 deg/year target.
    """
    a = R_EARTH_EQUATORIAL_M + 600.0e3
    inclination = sun_synchronous_inclination_rad(a, 1.0e-3)
    assert math.degrees(inclination) == pytest.approx(97.7875, abs=1e-3)
    measured, _ = _propagate_and_fit(a, 1.0e-3, inclination)
    assert abs(measured / SUN_SYNCHRONOUS_RAAN_RATE_RAD_S - 1.0) < 0.01


@pytest.mark.integration
def test_raan_does_not_drift_without_j2():
    """Complement to the headline: the same fit on a two-body arc must return ~zero.

    Otherwise the RAAN test would be measuring the element extraction, not the force model.
    Measured two-body drift is 1e-9 of the J2 rate, so the bound is 1e-3 of it.
    """
    eccentricity, inclination = 0.01, math.radians(51.6)
    state0 = classical_to_cartesian(
        ClassicalElements(A_700_M, eccentricity, inclination, 0.4, 0.7, 0.0), MU_EARTH_M3_S2
    )
    times = np.linspace(0.0, 20.0 * orbital_period_s(A_700_M), 401)
    states = propagate_two_body(state0, times)
    measured = _fit_secular_rate(
        times, np.array([cartesian_to_classical(s).raan_rad for s in states])
    )
    j2_rate = abs(secular_raan_rate_rad_s(A_700_M, eccentricity, inclination))
    assert abs(measured) < 1e-3 * j2_rate, f"two-body RAAN drifted at {measured:.3e} rad/s"


# ======================================================================================
# Argument of perigee
# ======================================================================================


@pytest.mark.integration
def test_j2_argument_of_perigee_drift_matches_the_closed_form():
    r"""Apsidal rotation must match ``+3/4 n J2 (Re/p)^2 (5 cos^2 i - 1)``.

    Eccentricity is 0.15 rather than the RAAN test's 0.01 for a concrete reason: the
    short-period variation of osculating omega scales as 1/e (it is the argument of a vector
    whose length is e), so at e = 0.01 the oscillation swamps the secular trend. Measured
    disagreement at e = 0.01 / 0.05 / 0.15 over 20 orbits: 0.79 / 0.26 / 0.078 %. At
    e = 0.15 the measured rate is 1.09818673e-06 against a closed form of 1.09733168e-06 --
    **0.078 %**. Bound set at 1 %, ~13x headroom, chosen to also cover the 0.235 % seen at a
    coarser 12-samples-per-orbit schedule.
    """
    eccentricity, inclination = 0.15, math.radians(45.0)
    _, measured = _propagate_and_fit(A_700_M, eccentricity, inclination)
    closed_form = secular_arg_periapsis_rate_rad_s(A_700_M, eccentricity, inclination)

    disagreement = abs(measured / closed_form - 1.0)
    assert disagreement < 0.01, (
        f"measured apsidal rate {measured:.8e} rad/s vs closed form {closed_form:.8e} "
        f"rad/s: {disagreement * 100:.4f} % disagreement"
    )
    # Below the critical inclination the bracket is positive, so perigee advances.
    assert measured > 0.0


@pytest.mark.integration
def test_measured_apsidal_drift_rules_out_the_five_cos_squared_minus_three_variant():
    """Knife edge on the bracket: ``(5cos^2 i - 3)`` is not merely imprecise, it is backwards.

    ``docs/project1/math-model.md`` does not state the apsidal rate, and the variant
    ``(3/4) n J2 (Re/p)^2 (5 cos^2 i - 3)`` circulates in places. At i = 45 deg it evaluates
    to -3.657772e-07 rad/s while the measured rate is +1.098187e-06 rad/s: wrong sign and
    400 % out. Its zero would sit at 39.2 deg, whereas the physically-established critical
    inclination -- the one Molniya orbits are flown at so apogee stays put -- is 63.43 deg.
    """
    eccentricity, inclination = 0.15, math.radians(45.0)
    _, measured = _propagate_and_fit(A_700_M, eccentricity, inclination)

    n = math.sqrt(MU_EARTH_M3_S2 / A_700_M**3)
    p = A_700_M * (1.0 - eccentricity**2)
    rejected = (
        0.75
        * n
        * J2_EARTH
        * (R_EARTH_EQUATORIAL_M / p) ** 2
        * (5.0 * math.cos(inclination) ** 2 - 3.0)
    )
    assert rejected < 0.0 < measured, "the rejected variant should have the opposite sign"
    assert abs(measured / rejected - 1.0) > 1.0


@pytest.mark.unit
def test_critical_inclination_is_where_apsidal_drift_vanishes():
    """``5 cos^2 i - 1 = 0`` puts the zero at 63.4349 deg -- the Molniya inclination.

    Measured by bisection on the closed form: 63.434949 deg, matching
    ``arccos(1/sqrt(5)) = 63.434949 deg`` to the 1e-9 deg the bisection was run to.
    """
    expected = math.degrees(math.acos(1.0 / math.sqrt(5.0)))
    assert expected == pytest.approx(63.434949, abs=1e-5)

    a, e = R_EARTH_EQUATORIAL_M + 1000.0e3, 0.2
    lo, hi = math.radians(50.0), math.radians(75.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (
            secular_arg_periapsis_rate_rad_s(a, e, lo) * secular_arg_periapsis_rate_rad_s(a, e, mid)
            <= 0.0
        ):
            hi = mid
        else:
            lo = mid
    assert math.degrees(0.5 * (lo + hi)) == pytest.approx(expected, abs=1e-9)


# ======================================================================================
# J2 acceleration: closed form, geometry, gradient
# ======================================================================================


@pytest.mark.unit
def test_j2_at_the_equator_is_purely_radial_and_inward():
    """On the equator ``z = 0``: the bracket is 1, so a = -3/2 J2 (mu/r^2)(Re/r)^2 r_hat.

    Inward because the equatorial bulge deepens the potential well in its own plane.
    """
    r = 7.0e6
    acceleration = j2_acceleration_m_s2([r, 0.0, 0.0])
    expected = -1.5 * J2_EARTH * (MU_EARTH_M3_S2 / r**2) * (R_EARTH_EQUATORIAL_M / r) ** 2
    assert float(acceleration[0]) == pytest.approx(expected, rel=1e-14)
    assert acceleration[0] < 0.0
    np.testing.assert_array_equal(acceleration[1:], [0.0, 0.0])


@pytest.mark.unit
def test_j2_over_the_pole_is_outward_and_exactly_twice_the_equatorial_magnitude():
    """The 1-versus-3 bracket asymmetry, tested where it is unambiguous.

    At ``z = r`` the z bracket is ``3 - 5 = -2`` against the equator's ``1``, so the polar
    acceleration is exactly twice the equatorial one and points the *other* way: an oblate
    body's potential well is shallower over the poles. Measured ratio 2.0 exactly (both
    branches reduce to the same factor times a small integer, so this is exact in floating
    point, not merely close).

    This is the pair a plausible-looking sign error cannot survive.
    """
    r = 7.0e6
    equator = j2_acceleration_m_s2([r, 0.0, 0.0])
    pole = j2_acceleration_m_s2([0.0, 0.0, r])

    expected_pole = 3.0 * J2_EARTH * (MU_EARTH_M3_S2 / r**2) * (R_EARTH_EQUATORIAL_M / r) ** 2
    assert float(pole[2]) == pytest.approx(expected_pole, rel=1e-14)
    assert pole[2] > 0.0 > equator[0]
    np.testing.assert_array_equal(pole[:2], [0.0, 0.0])
    assert float(np.linalg.norm(pole) / np.linalg.norm(equator)) == pytest.approx(2.0, rel=1e-14)


@pytest.mark.unit
def test_j2_acceleration_is_minus_the_gradient_of_the_j2_potential():
    """Conservation test's precondition: the force really is derived from the potential.

    Central differences with a 1 m step over 20 random LEO positions give a worst relative
    error of 9.76e-10, which is the step's own truncation error. Bound 1e-7 -- two decades of
    headroom -- because tightening it would only measure the finite-difference scheme.
    """
    rng = np.random.default_rng(20260824)
    worst = 0.0
    for _ in range(20):
        direction = rng.normal(size=3)
        position = direction / np.linalg.norm(direction) * (6.8e6 + rng.uniform(0.0, 1.0e6))
        analytic = j2_acceleration_m_s2(position)
        numeric = np.empty(3)
        for axis in range(3):
            plus, minus = position.copy(), position.copy()
            plus[axis] += 1.0
            minus[axis] -= 1.0
            numeric[axis] = -(j2_potential_j_kg(plus) - j2_potential_j_kg(minus)) / 2.0
        worst = max(worst, float(np.linalg.norm(numeric - analytic) / np.linalg.norm(analytic)))
    assert worst < 1e-7, f"a != -grad V, worst relative error {worst:.3e}"


@pytest.mark.unit
def test_j2_acceleration_falls_off_as_one_over_r_to_the_fourth():
    """``|a| ~ mu J2 Re^2 / r^4``: doubling the radius must divide it by exactly 16."""
    near = j2_acceleration_m_s2([7.0e6, 0.0, 0.0])
    far = j2_acceleration_m_s2([14.0e6, 0.0, 0.0])
    ratio = float(np.linalg.norm(near) / np.linalg.norm(far))
    assert ratio == pytest.approx(16.0, rel=1e-13)


@pytest.mark.unit
def test_j2_is_about_a_thousandth_of_central_gravity_in_leo():
    """Order-of-magnitude sanity: a J2 term at 1e-1 or 1e-6 of gravity is a units bug.

    Measured at r = 7000 km: 1.3482e-03 in the equatorial plane and 2.6964e-03 over the pole.
    """
    r = 7.0e6
    central = MU_EARTH_M3_S2 / r**2
    equator_ratio = float(np.linalg.norm(j2_acceleration_m_s2([r, 0.0, 0.0])) / central)
    assert 1e-4 < equator_ratio < 1e-2
    assert equator_ratio == pytest.approx(1.3482e-3, rel=1e-3)


@pytest.mark.unit
def test_a_spherical_body_produces_no_j2_acceleration():
    """Limiting case: ``j2 = 0`` gives exactly zero, not merely small."""
    np.testing.assert_array_equal(j2_acceleration_m_s2([4.0e6, 5.0e6, 3.0e6], j2=0.0), np.zeros(3))


@pytest.mark.unit
def test_flipping_the_sign_of_j2_flips_the_acceleration():
    """Complement: the result is strictly linear in ``j2``, so a sign slip is detectable."""
    position = [4.0e6, 5.0e6, 3.0e6]
    np.testing.assert_allclose(
        j2_acceleration_m_s2(position, j2=-J2_EARTH),
        -j2_acceleration_m_s2(position, j2=J2_EARTH),
        rtol=1e-15,
    )


@pytest.mark.unit
def test_j2_vanishes_where_the_brackets_do():
    """Both brackets have a real zero, and the acceleration components follow them.

    In-plane vanishes at ``(z/r)^2 = 1/5`` (i.e. latitude 26.57 deg) and the z component at
    ``(z/r)^2 = 3/5`` (50.77 deg). Two independent nulls, which together pin both brackets.

    The nulls are asserted *relative to the surviving component*, because absolute
    cancellation lands at machine precision, not at zero: measured 2.220e-16 and 2.719e-16
    respectively, both one ulp of the bracket subtraction. Bound 1e-13 relative -- ~400x
    headroom -- while still being three decades tighter than any bracket error could hide in.
    """
    r = 7.0e6
    z_in_plane = r / math.sqrt(5.0)
    xy = math.sqrt(r**2 - z_in_plane**2)
    a = j2_acceleration_m_s2([xy, 0.0, z_in_plane])
    assert abs(float(a[0]) / float(a[2])) < 1e-13
    assert abs(float(a[2])) > 1e-3

    z_cross = r * math.sqrt(0.6)
    xy2 = math.sqrt(r**2 - z_cross**2)
    b = j2_acceleration_m_s2([xy2, 0.0, z_cross])
    assert abs(float(b[2]) / float(b[0])) < 1e-13
    assert abs(float(b[0])) > 1e-3


@pytest.mark.unit
def test_j2_potential_is_negative_at_the_equator_and_positive_over_the_pole():
    """Sign of ``V_J2 ~ (3z^2 - r^2)``: the bulge deepens the well in its own plane."""
    r = 7.0e6
    assert j2_potential_j_kg([r, 0.0, 0.0]) < 0.0
    assert j2_potential_j_kg([0.0, 0.0, r]) > 0.0
    # The 54.7356 deg "magic latitude" where P2 vanishes.
    z = r / math.sqrt(3.0)
    assert abs(j2_potential_j_kg([math.sqrt(r**2 - z**2), 0.0, z])) < 1e-12


# ======================================================================================
# Conservation: J2 conserves total energy, drag does not
# ======================================================================================


@pytest.mark.integration
def test_j2_conserves_the_total_energy_including_its_potential():
    """J2 is conservative, so ``v^2/2 - mu/r + V_J2`` is invariant.

    Measured relative drift over 10 orbits: 2.760e-12, which is the DOP853 noise floor at
    rtol = 1e-12 (the two-body run shows the same order). Bound 1e-9 -- ~360x headroom --
    tight enough that adding a non-conservative term of any physical size would break it.
    """
    state0 = classical_to_cartesian(
        ClassicalElements(A_700_M, 0.01, math.radians(51.6), 0.4, 0.7, 0.0), MU_EARTH_M3_S2
    )
    times = np.linspace(0.0, 10.0 * orbital_period_s(A_700_M), 501)
    states = propagate_perturbed(state0, times, enable_j2=True)
    energies = np.array([specific_energy_with_j2_j_kg(s) for s in states])
    drift = float(np.abs(energies - energies[0]).max() / abs(energies[0]))
    assert drift < 1e-9, f"J2 total energy drifted by {drift:.3e} relative"


@pytest.mark.integration
def test_keplerian_energy_alone_is_not_conserved_under_j2():
    """The complement, and the point of having two energy functions.

    ``v^2/2 - mu/r`` omits ``V_J2`` and therefore oscillates: measured 9.661e-04 relative
    against the invariant's 2.760e-12, eight orders of magnitude apart. If this test ever
    starts passing as a conservation check, the J2 term is not reaching the integrator.
    """
    state0 = classical_to_cartesian(
        ClassicalElements(A_700_M, 0.01, math.radians(51.6), 0.4, 0.7, 0.0), MU_EARTH_M3_S2
    )
    times = np.linspace(0.0, 10.0 * orbital_period_s(A_700_M), 501)
    states = propagate_perturbed(state0, times, enable_j2=True)
    keplerian = np.array([specific_energy_j_kg(s) for s in states])
    variation = float(np.abs(keplerian - keplerian[0]).max() / abs(keplerian[0]))
    assert variation > 1e-5, f"Keplerian energy barely moved ({variation:.3e}) under J2"


@pytest.mark.integration
def test_drag_removes_energy_monotonically():
    """Drag is dissipative: energy must fall at every step, never rise.

    Measured over 5 orbits at 300 km with C_D A/m = 0.02 m^2/kg: -29843685.58 to
    -29846502.31 J/kg, strictly decreasing at all 500 steps.
    """
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    states = propagate_perturbed(
        _circular_state(A_300_M, 0.9),
        times,
        enable_drag=True,
        ballistic_coefficient_m2_kg=0.02,
    )
    energies = np.array([specific_energy_j_kg(s) for s in states])
    steps = np.diff(energies)
    assert np.all(steps < 0.0), f"energy rose under drag: max step {steps.max():.6e} J/kg"
    assert energies[-1] < energies[0]


@pytest.mark.integration
@pytest.mark.parametrize("inclination_rad", [0.0, 0.9, 1.2])
def test_drag_decays_the_semi_major_axis_monotonically(inclination_rad):
    """The orbit shrinks, always. Semi-major axis must never grow under drag.

    Taken from the energy rather than from ``cartesian_to_classical`` so an exactly circular
    orbit can be used, which removes the eccentric oscillation that would otherwise make
    "monotone" a statement about sampling rather than about physics.

    Measured over 5 orbits at 300 km, C_D A/m = 0.02: -630 m (equatorial) to -652 m
    (i = 68.75 deg), with every one of the 500 steps negative and the largest single step
    -1.25 m.
    """
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    states = propagate_perturbed(
        _circular_state(A_300_M, inclination_rad),
        times,
        enable_drag=True,
        ballistic_coefficient_m2_kg=0.02,
    )
    sma = np.array([-MU_EARTH_M3_S2 / (2.0 * specific_energy_j_kg(s)) for s in states])
    steps = np.diff(sma)
    assert np.all(steps < 0.0), f"semi-major axis grew under drag: max step {steps.max():.6e} m"
    # Measured decay is 630-652 m; require at least 100 m so a near-zero drag term fails.
    assert sma[0] - sma[-1] > 100.0


# ======================================================================================
# Density model
# ======================================================================================


@pytest.mark.unit
def test_density_table_bands_join_continuously():
    """Transcription guard on 84 hand-entered numbers.

    The published table is constructed so that band k evaluated at band k+1's base altitude
    reproduces band k+1's base density. Measured worst relative mismatch above 25 km:
    9.5877e-05, consistent with the four-significant-figure printing. The 0-25 km band is the
    coarsest fit in the table and misses by 1.36e-03, so it is bounded separately rather than
    hidden by a loose global tolerance.

    A typo in any base density or scale height breaks this; a plain "is it a float" check
    would not.
    """
    mismatches = []
    for lower, upper in pairwise(_DENSITY_TABLE_KM_KG_M3_KM):
        h0, rho0, scale_height = lower
        h1, rho1, _ = upper
        predicted = rho0 * math.exp(-(h1 - h0) / scale_height)
        mismatches.append(abs(predicted / rho1 - 1.0))

    assert mismatches[0] < 2e-3, f"surface band mismatch {mismatches[0]:.3e}"
    assert max(mismatches[1:]) < 5e-4, f"worst mismatch above 25 km {max(mismatches[1:]):.3e}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("altitude_m", "expected_kg_m3"),
    [
        (0.0, 1.225),
        (400.0e3, 3.725e-12),
        (500.0e3, 6.967e-13),
        (800.0e3, 1.170e-14),
    ],
)
def test_density_reproduces_the_published_base_values(altitude_m, expected_kg_m3):
    """At a band base the exponential is exactly the tabulated density.

    Also an order-of-magnitude anchor against the literature: 3.7e-12 kg/m^3 at 400 km is the
    standard moderate-solar-activity figure, and the real value ranges over roughly
    1e-12..1e-11 across a solar cycle -- which is the model limitation, not a test failure.
    """
    assert exponential_density_kg_m3(altitude_m) == pytest.approx(expected_kg_m3, rel=1e-12)


@pytest.mark.unit
def test_density_decreases_monotonically_with_altitude():
    """No band boundary may produce an upward step. Checked on 4001 samples to 1200 km."""
    altitudes = np.linspace(0.0, 1200.0e3, 4001)
    densities = np.array([exponential_density_kg_m3(h) for h in altitudes])
    assert np.all(np.diff(densities) < 0.0)


@pytest.mark.unit
def test_density_extrapolates_above_the_table_without_blowing_up():
    """Above 1000 km the top band is extrapolated; it must decay, not diverge.

    Measured: 3.019e-15 at 1000 km, 7.234e-17 at 2000 km, 5.785e-72 at GEO. Documented as an
    extrapolation with no data behind it, but harmless -- drag there is far below the
    perturbations this module does not model at all.
    """
    assert exponential_density_kg_m3(MAX_TABULATED_ALTITUDE_M) == pytest.approx(3.019e-15, rel=1e-9)
    assert 0.0 < exponential_density_kg_m3(2000.0e3) < 1e-16
    assert 0.0 <= exponential_density_kg_m3(36000.0e3) < 1e-60


@pytest.mark.unit
def test_density_below_the_reference_ellipsoid_raises():
    with pytest.raises(AtmosphericModelError, match="below the reference ellipsoid"):
        exponential_density_kg_m3(-1.0)


@pytest.mark.unit
def test_density_rejects_non_finite_altitude():
    with pytest.raises(ValueError, match="altitude_m must be finite"):
        exponential_density_kg_m3(float("nan"))


# ======================================================================================
# Drag acceleration
# ======================================================================================


@pytest.mark.unit
def test_drag_matches_the_closed_form_with_a_corotating_relative_velocity():
    """Full closed-form reconstruction, including ``omega x r``.

    Independently forms ``v_rel = v - omega x r`` with ``numpy.cross`` and evaluates
    ``-1/2 rho B |v_rel| v_rel``. Measured difference: exactly 0.0 in every component, so
    the assertion is exact equality rather than a tolerance. Dropping the rotation term or
    getting its sign wrong fails here immediately.
    """
    position = np.array([5.0e6, 3.0e6, 3.0e6])  # 179.3 km altitude
    velocity = np.array([-2.0e3, 6.0e3, 1.0e3])
    ballistic_coefficient = 0.03

    omega = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])
    v_rel = velocity - np.cross(omega, position)
    density = exponential_density_kg_m3(float(np.linalg.norm(position)) - R_EARTH_EQUATORIAL_M)
    expected = -0.5 * density * ballistic_coefficient * float(np.linalg.norm(v_rel)) * v_rel

    np.testing.assert_array_equal(
        drag_acceleration_m_s2(position, velocity, ballistic_coefficient), expected
    )


@pytest.mark.unit
def test_drag_is_antiparallel_to_the_relative_velocity():
    """Direction test independent of magnitude: cos(angle) must be exactly -1.

    Measured -1.0000000000000002 (one ulp past -1 from the normalisation), so the bound is
    stated on the residual rather than as an equality.
    """
    position = np.array([5.0e6, 3.0e6, 3.0e6])
    velocity = np.array([-2.0e3, 6.0e3, 1.0e3])
    omega = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])
    v_rel = velocity - np.cross(omega, position)
    acceleration = drag_acceleration_m_s2(position, velocity, 0.03)
    cosine = float(
        np.dot(acceleration, v_rel) / (np.linalg.norm(acceleration) * np.linalg.norm(v_rel))
    )
    assert cosine == pytest.approx(-1.0, abs=1e-14)


@pytest.mark.unit
def test_ignoring_earth_rotation_changes_drag_by_a_measurable_amount():
    """Complement for the ``omega x r`` term: it is not a rounding-level correction.

    A non-rotating atmosphere overstates the drag magnitude by a measured factor of 1.1411
    at this geometry (~14 %), and by ~12 % for an equatorial LEO orbit where the surface
    moves at 465 m/s against 7.7 km/s of orbital speed. If a test suite cannot tell the two
    apart, dropping the transport term is free.
    """
    position = np.array([5.0e6, 3.0e6, 3.0e6])
    velocity = np.array([-2.0e3, 6.0e3, 1.0e3])
    with_rotation = drag_acceleration_m_s2(position, velocity, 0.03)
    without = drag_acceleration_m_s2(position, velocity, 0.03, omega_earth_rad_s=0.0)
    ratio = float(np.linalg.norm(without) / np.linalg.norm(with_rotation))
    assert ratio == pytest.approx(1.1411, rel=1e-3)
    assert ratio > 1.05


@pytest.mark.unit
def test_zero_ballistic_coefficient_gives_exactly_zero_drag():
    """Limiting case. Zero, not small: the whole term is linear in ``C_D A / m``."""
    acceleration = drag_acceleration_m_s2([5.0e6, 3.0e6, 3.0e6], [-2.0e3, 6.0e3, 1.0e3], 0.0)
    np.testing.assert_array_equal(np.abs(acceleration), np.zeros(3))


@pytest.mark.unit
def test_drag_grows_linearly_with_the_ballistic_coefficient():
    """The spacecraft-specific factor must be exactly a scale factor, not a shape change."""
    args = ([5.0e6, 3.0e6, 3.0e6], [-2.0e3, 6.0e3, 1.0e3])
    np.testing.assert_allclose(
        drag_acceleration_m_s2(*args, 0.06), 2.0 * drag_acceleration_m_s2(*args, 0.03), rtol=1e-15
    )


@pytest.mark.unit
def test_negative_ballistic_coefficient_raises():
    """A negative C_D A/m turns the atmosphere into a thruster. Refuse it."""
    with pytest.raises(ValueError, match="thruster"):
        drag_acceleration_m_s2([5.0e6, 3.0e6, 3.0e6], [-2.0e3, 6.0e3, 1.0e3], -0.01)


@pytest.mark.unit
def test_drag_below_the_surface_raises():
    with pytest.raises(AtmosphericModelError, match="impacted"):
        drag_acceleration_m_s2([1.0e6, 0.0, 0.0], [0.0, 1.0e3, 0.0], 0.01)


# ======================================================================================
# Input validation on the acceleration models
# ======================================================================================


@pytest.mark.unit
def test_j2_acceleration_at_the_origin_raises():
    with pytest.raises(DegenerateGeometryError, match="singular at the origin"):
        j2_acceleration_m_s2([0.0, 0.0, 0.0])


@pytest.mark.unit
def test_j2_potential_at_the_origin_raises():
    with pytest.raises(DegenerateGeometryError, match=r"undefined at \|r\| = 0"):
        j2_potential_j_kg([0.0, 0.0, 0.0])


@pytest.mark.unit
def test_specific_energy_with_j2_at_the_origin_raises():
    with pytest.raises(DegenerateGeometryError, match=r"undefined at \|r\| = 0"):
        specific_energy_with_j2_j_kg(np.zeros(6))


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros(4), np.zeros((3, 3))])
def test_j2_acceleration_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        j2_acceleration_m_s2(bad)


@pytest.mark.unit
def test_j2_acceleration_rejects_non_finite_position():
    with pytest.raises(ValueError, match="must be finite"):
        j2_acceleration_m_s2([1.0e7, np.nan, 0.0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mu_m3_s2": 0.0}, "mu_m3_s2"),
        ({"mu_m3_s2": -1.0}, "mu_m3_s2"),
        ({"r_body_m": 0.0}, "r_body_m"),
        ({"r_body_m": float("inf")}, "r_body_m"),
        ({"j2": float("nan")}, "j2"),
    ],
)
def test_j2_acceleration_rejects_bad_model_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        j2_acceleration_m_s2([7.0e6, 0.0, 0.0], **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"r_body_m": -1.0}, "r_body_m"),
        ({"omega_earth_rad_s": float("nan")}, "omega_earth_rad_s"),
    ],
)
def test_drag_rejects_bad_model_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        drag_acceleration_m_s2([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 0.01, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros((2, 3))])
def test_drag_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        drag_acceleration_m_s2(bad, [0.0, 7.5e3, 0.0], 0.01)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        drag_acceleration_m_s2([7.0e6, 0.0, 0.0], bad, 0.01)


@pytest.mark.unit
def test_drag_rejects_non_finite_ballistic_coefficient():
    with pytest.raises(ValueError, match="ballistic_coefficient_m2_kg must be finite"):
        drag_acceleration_m_s2([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], float("inf"))


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros(7)])
def test_specific_energy_with_j2_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        specific_energy_with_j2_j_kg(bad)


@pytest.mark.unit
def test_specific_energy_with_j2_rejects_non_finite_state():
    with pytest.raises(ValueError, match="must be finite"):
        specific_energy_with_j2_j_kg([7.0e6, 0.0, 0.0, 0.0, np.inf, 0.0])


# ======================================================================================
# Secular-rate helpers: validation and inversion
# ======================================================================================


@pytest.mark.unit
def test_secular_raan_rate_matches_a_literal_transcription_of_the_formula():
    """Guards the helper against a typo without asserting a golden number."""
    a, e, inclination = A_700_M, 0.02, math.radians(63.0)
    n = math.sqrt(MU_EARTH_M3_S2 / a**3)
    p = a * (1.0 - e**2)
    expected = -1.5 * n * J2_EARTH * (R_EARTH_EQUATORIAL_M / p) ** 2 * math.cos(inclination)
    assert secular_raan_rate_rad_s(a, e, inclination) == pytest.approx(expected, rel=1e-15)


@pytest.mark.unit
def test_nodal_regression_is_westward_prograde_zero_polar_eastward_retrograde():
    """Sign structure follows ``cos i`` exactly, and the polar case is an exact zero."""
    assert secular_raan_rate_rad_s(A_700_M, 0.0, math.radians(30.0)) < 0.0
    assert abs(secular_raan_rate_rad_s(A_700_M, 0.0, math.pi / 2.0)) < 1e-22
    assert secular_raan_rate_rad_s(A_700_M, 0.0, math.radians(150.0)) > 0.0


@pytest.mark.unit
def test_sun_synchronous_inclination_rises_with_altitude_and_is_retrograde():
    """Measured 97.7875 deg at 600 km and 98.6029 deg at 800 km.

    Worth recording because the commonly-quoted "97.8 deg" belongs to ~600 km, not 800 km;
    a check that expected 97.8 deg at 800 km would fail against a correct implementation.
    """
    low = sun_synchronous_inclination_rad(R_EARTH_EQUATORIAL_M + 600.0e3)
    high = sun_synchronous_inclination_rad(R_EARTH_EQUATORIAL_M + 800.0e3)
    assert math.degrees(low) == pytest.approx(97.7875, abs=1e-3)
    assert math.degrees(high) == pytest.approx(98.6029, abs=1e-3)
    assert math.pi / 2.0 < low < high


@pytest.mark.unit
def test_sun_synchronous_inclination_round_trips_through_the_rate():
    """Inversion consistency: feeding the answer back gives the requested rate."""
    a = R_EARTH_EQUATORIAL_M + 800.0e3
    inclination = sun_synchronous_inclination_rad(a, 1.0e-3)
    assert secular_raan_rate_rad_s(a, 1.0e-3, inclination) == pytest.approx(
        SUN_SYNCHRONOUS_RAAN_RATE_RAD_S, rel=1e-12
    )


@pytest.mark.unit
def test_sun_synchronous_inclination_raises_when_j2_cannot_supply_the_rate():
    """Far above LEO the required ``cos i`` exceeds 1: refuse rather than clamp."""
    with pytest.raises(ValueError, match="no inclination gives a nodal rate"):
        sun_synchronous_inclination_rad(R_EARTH_EQUATORIAL_M + 40000.0e3)


@pytest.mark.unit
def test_sun_synchronous_inclination_raises_for_a_spherical_body():
    with pytest.raises(ValueError, match="no nodal drift at all"):
        sun_synchronous_inclination_rad(A_700_M, 0.0, j2=0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "e", "match"),
    [
        (-1.0, 0.0, "semi_major_axis_m"),
        (0.0, 0.0, "semi_major_axis_m"),
        (A_700_M, 1.0, r"eccentricity must be in \[0, 1\)"),
        (A_700_M, -0.1, r"eccentricity must be in \[0, 1\)"),
        (A_700_M, float("nan"), "eccentricity must be finite"),
    ],
)
def test_secular_rates_reject_non_closed_orbits(a, e, match):
    with pytest.raises(ValueError, match=match):
        secular_raan_rate_rad_s(a, e, 0.5)
    with pytest.raises(ValueError, match=match):
        secular_arg_periapsis_rate_rad_s(a, e, 0.5)


@pytest.mark.unit
def test_secular_rates_reject_non_finite_inclination():
    with pytest.raises(ValueError, match="inclination_rad"):
        secular_raan_rate_rad_s(A_700_M, 0.0, float("nan"))
    with pytest.raises(ValueError, match="inclination_rad"):
        secular_arg_periapsis_rate_rad_s(A_700_M, 0.0, float("inf"))


# ======================================================================================
# propagate_perturbed: limiting cases, composition, failure behaviour
# ======================================================================================


@pytest.mark.integration
def test_with_every_perturbation_off_it_reproduces_propagate_two_body_exactly():
    """The knife-edge complement's other half: no perturbation means no difference at all.

    Measured max absolute difference over 5 orbits: 0.0 in all six components, bit for bit.
    The two functions integrate the same right-hand side with the same method and
    tolerances, so anything other than exact equality would mean an unintended term.
    """
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    state0 = _circular_state(A_300_M, 0.9)
    np.testing.assert_array_equal(
        propagate_perturbed(state0, times), propagate_two_body(state0, times)
    )


@pytest.mark.integration
def test_a_zero_j2_coefficient_reproduces_two_body_through_the_live_code_path():
    """Stronger than the switch test: J2 is computed, evaluates to zero, and is added.

    ``enable_j2=False`` proves only that the branch is skipped. This exercises the branch and
    still lands on the two-body answer, measured bit-identical over 5 orbits.
    """
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    state0 = _circular_state(A_300_M, 0.9)
    np.testing.assert_array_equal(
        propagate_perturbed(state0, times, enable_j2=True, j2=0.0),
        propagate_two_body(state0, times),
    )


@pytest.mark.integration
def test_enabling_j2_changes_the_trajectory_by_hundreds_of_kilometres():
    """Complement: the limiting-case tests above would pass if J2 were never applied.

    Measured separation after 5 orbits at 300 km, i = 51.6 deg: 3.569e+05 m. The bound is
    1e4 m -- 35x below the measurement -- so it fails loudly if the switch is ignored but
    does not encode the exact geometry.
    """
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    state0 = _circular_state(A_300_M, 0.9)
    difference = np.abs(
        propagate_perturbed(state0, times, enable_j2=True)[:, :3]
        - propagate_two_body(state0, times)[:, :3]
    ).max()
    assert difference > 1.0e4, f"J2 moved the trajectory by only {difference:.3e} m"


@pytest.mark.integration
def test_zero_ballistic_coefficient_reproduces_the_undragged_trajectory():
    """Drag limiting case, measured bit-identical over 5 orbits."""
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    state0 = _circular_state(A_300_M, 0.9)
    np.testing.assert_array_equal(
        propagate_perturbed(state0, times, enable_drag=True, ballistic_coefficient_m2_kg=0.0),
        propagate_two_body(state0, times),
    )


@pytest.mark.integration
def test_negligible_density_reproduces_the_undragged_trajectory():
    """The other drag limiting case: zero density instead of zero area.

    At 5000 km the model returns 9.95e-22 kg/m^3, and the measured position difference over
    two orbits with C_D A/m = 0.02 is 8.497e-06 m. Bound 1e-3 m: two decades of headroom,
    still far below anything a real drag term could hide behind.
    """
    a = R_EARTH_EQUATORIAL_M + 5000.0e3
    times = np.linspace(0.0, 2.0 * orbital_period_s(a), 101)
    state0 = _circular_state(a, 0.0)
    difference = np.abs(
        propagate_perturbed(state0, times, enable_drag=True, ballistic_coefficient_m2_kg=0.02)
        - propagate_two_body(state0, times)
    ).max()
    assert difference < 1e-3, f"drag at 5000 km moved the state by {difference:.3e}"


@pytest.mark.integration
def test_j2_and_drag_switch_independently_and_compose():
    """Each switch must contribute on its own and both must contribute together.

    Measured over 5 orbits at 300 km, i = 68.75 deg: ``both`` differs from J2-only by
    1.615e+04 m and from drag-only by 2.413e+05 m, so neither term is being dropped when the
    other is present.
    """
    times = np.linspace(0.0, 5.0 * orbital_period_s(A_300_M), 501)
    state0 = _circular_state(A_300_M, 1.2)
    kwargs = {"ballistic_coefficient_m2_kg": 0.02}
    both = propagate_perturbed(state0, times, enable_j2=True, enable_drag=True, **kwargs)
    j2_only = propagate_perturbed(state0, times, enable_j2=True)
    drag_only = propagate_perturbed(state0, times, enable_drag=True, **kwargs)

    assert np.abs(both[:, :3] - j2_only[:, :3]).max() > 1.0e3
    assert np.abs(both[:, :3] - drag_only[:, :3]).max() > 1.0e4


@pytest.mark.integration
def test_perturbed_solution_converges_as_the_tolerance_tightens():
    """Convergence behaviour rather than one hand-picked bound.

    A perturbing acceleration three orders of magnitude below the central term is the part
    most easily lost to truncation, so a J2 run being converged is a separate claim from a
    two-body run being converged at the same setting.
    """
    times = np.array([0.0, 3.0 * orbital_period_s(A_700_M)])
    state0 = _circular_state(A_700_M, 0.9)
    reference = propagate_perturbed(state0, times, enable_j2=True, rtol=1e-13, atol=1e-13)[-1]
    deviations = [
        float(
            np.linalg.norm(
                propagate_perturbed(state0, times, enable_j2=True, rtol=tol, atol=tol)[-1][:3]
                - reference[:3]
            )
        )
        for tol in (1e-9, 1e-10, 1e-12)
    ]
    assert deviations[0] > deviations[1] > deviations[2], (
        f"deviation did not shrink monotonically with tolerance: {deviations}"
    )
    assert deviations[-1] < 1e-3


@pytest.mark.unit
def test_drag_without_a_ballistic_coefficient_raises():
    """No default means no accidental drag number for an unnamed spacecraft."""
    with pytest.raises(ValueError, match="requires an explicit ballistic_coefficient"):
        propagate_perturbed(_circular_state(A_300_M, 0.0), [0.0, 100.0], enable_drag=True)


@pytest.mark.unit
def test_a_ballistic_coefficient_without_enable_drag_raises():
    """The other half: a value that would be silently ignored is a trap, not a default."""
    with pytest.raises(ValueError, match="but enable_drag is False"):
        propagate_perturbed(
            _circular_state(A_300_M, 0.0), [0.0, 100.0], ballistic_coefficient_m2_kg=0.01
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.01, float("nan")])
def test_propagate_perturbed_rejects_a_bad_ballistic_coefficient(bad):
    with pytest.raises(ValueError, match="ballistic_coefficient_m2_kg"):
        propagate_perturbed(
            _circular_state(A_300_M, 0.0),
            [0.0, 100.0],
            enable_drag=True,
            ballistic_coefficient_m2_kg=bad,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("times", "match"),
    [
        (np.array([10.0, 20.0]), "must start at 0.0"),
        (np.array([0.0, 20.0, 10.0]), "non-decreasing"),
        (np.array([]), "non-empty"),
        (np.array([0.0, np.nan]), "finite"),
        (np.zeros((2, 2)), "non-empty 1-D"),
    ],
)
def test_propagate_perturbed_rejects_a_malformed_time_schedule(times, match):
    with pytest.raises(ValueError, match=match):
        propagate_perturbed(_circular_state(A_700_M, 0.0), times, enable_j2=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "match"),
    [
        (np.zeros(3), r"shape \(6,\)"),
        (np.full(6, np.nan), "must be finite"),
    ],
)
def test_propagate_perturbed_rejects_a_malformed_state(state, match):
    with pytest.raises(ValueError, match=match):
        propagate_perturbed(state, np.array([0.0, 100.0]), enable_j2=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mu_m3_s2": -1.0}, "mu_m3_s2"),
        ({"r_body_m": 0.0}, "r_body_m"),
        ({"omega_earth_rad_s": float("nan")}, "omega_earth_rad_s"),
        ({"j2": float("inf")}, "j2"),
    ],
)
def test_propagate_perturbed_rejects_bad_model_parameters(kwargs, match):
    mu = kwargs.pop("mu_m3_s2", MU_EARTH_M3_S2)
    with pytest.raises(ValueError, match=match):
        propagate_perturbed(
            _circular_state(A_700_M, 0.0), np.array([0.0, 100.0]), mu, enable_j2=True, **kwargs
        )


@pytest.mark.unit
def test_a_single_output_time_returns_the_initial_state_unchanged():
    state0 = _circular_state(A_700_M, 0.9)
    result = propagate_perturbed(state0, np.array([0.0]), enable_j2=True)
    assert result.shape == (1, 6)
    np.testing.assert_array_equal(result[0], state0)


@pytest.mark.unit
def test_integrator_failure_raises_rather_than_returning_a_short_trajectory():
    """An integrator that gives up must surface, never hand back what it managed.

    Stubbed rather than provoked: a real failure needs a pathological state whose runtime is
    unbounded, and the branch under test is the error handling, not the dynamics.
    """
    import rpo_core.perturbations as module

    def failing(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            message="required step size is less than spacing between numbers",
            t=np.array([0.0, 12.5]),
            y=np.zeros((6, 2)),
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "solve_ivp", failing)
        with pytest.raises(PropagationError, match=r"perturbed propagation failed at t = 12\.5"):
            propagate_perturbed(
                _circular_state(A_700_M, 0.9),
                np.array([0.0, 100.0, 200.0]),
                enable_j2=True,
                enable_drag=True,
                ballistic_coefficient_m2_kg=0.01,
            )


@pytest.mark.unit
def test_a_successful_but_incomplete_integration_still_raises():
    """``success=True`` with fewer states than requested is a truncated trajectory."""
    import rpo_core.perturbations as module

    def short(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=True, message="ok", t=np.array([0.0]), y=np.zeros((6, 1)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "solve_ivp", short)
        with pytest.raises(PropagationError, match="trajectory is incomplete"):
            propagate_perturbed(
                _circular_state(A_700_M, 0.9), np.array([0.0, 100.0, 200.0]), enable_j2=True
            )


@pytest.mark.unit
def test_the_failure_message_names_the_enabled_perturbations():
    """Error messages carry the numbers and switches that motivated them."""
    import rpo_core.perturbations as module

    def failing(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, message="boom", t=np.array([]), y=np.zeros((6, 0)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "solve_ivp", failing)
        with pytest.raises(PropagationError, match="perturbations enabled: none"):
            propagate_perturbed(_circular_state(A_700_M, 0.9), np.array([0.0, 100.0]))


@pytest.mark.integration
def test_a_trajectory_that_reaches_the_ground_raises_instead_of_continuing_underground():
    """Decay past the surface is a physical event, surfaced as a typed error.

    A steep sub-orbital arc from 150 km reaches the surface in about 90 s; the propagation
    raises there rather than reporting a decay profile for a vehicle that has impacted.
    """
    state0 = np.array([R_EARTH_EQUATORIAL_M + 150.0e3, 0.0, 0.0, -2000.0, 3000.0, 0.0])
    with pytest.raises(AtmosphericModelError, match="impacted"):
        propagate_perturbed(
            state0,
            np.linspace(0.0, 200.0, 21),
            enable_drag=True,
            ballistic_coefficient_m2_kg=1e-3,
        )


# ======================================================================================
# perturbed_derivative
# ======================================================================================


@pytest.mark.unit
def test_perturbed_derivative_returns_velocity_then_total_acceleration():
    """Composition check: the RHS is exactly two-body + J2 + drag, term by term."""
    state = np.array([5.0e6, 3.0e6, 3.0e6, -2.0e3, 6.0e3, 1.0e3])
    derivative = perturbed_derivative(
        0.0, state, MU_EARTH_M3_S2, J2_EARTH, R_EARTH_EQUATORIAL_M, 0.03, OMEGA_EARTH_RAD_S
    )
    r_norm = float(np.linalg.norm(state[:3]))
    expected = (
        -MU_EARTH_M3_S2 * state[:3] / r_norm**3
        + j2_acceleration_m_s2(state[:3])
        + drag_acceleration_m_s2(state[:3], state[3:], 0.03)
    )
    np.testing.assert_array_equal(derivative[:3], state[3:])
    np.testing.assert_allclose(derivative[3:], expected, rtol=1e-14)


@pytest.mark.unit
def test_perturbed_derivative_with_both_terms_disabled_is_the_two_body_derivative():
    """``None`` means off, and off means bit-identical to the unperturbed right-hand side."""
    from rpo_core.propagate import two_body_derivative

    state = np.array([5.0e6, 3.0e6, 3.0e6, -2.0e3, 6.0e3, 1.0e3])
    np.testing.assert_array_equal(
        perturbed_derivative(
            0.0, state, MU_EARTH_M3_S2, None, R_EARTH_EQUATORIAL_M, None, OMEGA_EARTH_RAD_S
        ),
        two_body_derivative(0.0, state, MU_EARTH_M3_S2),
    )


@pytest.mark.unit
def test_perturbed_derivative_at_the_central_body_singularity_raises():
    with pytest.raises(PropagationError, match="singularity"):
        perturbed_derivative(
            0.0, np.zeros(6), MU_EARTH_M3_S2, J2_EARTH, R_EARTH_EQUATORIAL_M, None, 0.0
        )
