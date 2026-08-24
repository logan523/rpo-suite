"""Finite burns: convergence to the impulsive limit, Tsiolkovsky, policies, raise paths.

The headline is ``test_terminal_offset_converges_first_order_toward_the_impulsive_limit``.
Model M8's stated independent check is that the finite-burn trajectory converges to the
impulsive one as thrust grows at fixed total impulse, and that *the rate is the test*. A
suite that only asserted the limit would pass for an implementation that never integrated
anything -- one that simply applied the delta-v instantaneously would sail through it at
every thrust level. Measuring the order forces the implementation to be wrong by the right
amount at every intermediate thrust, which an accidentally-impulsive one cannot be.

Two orders are measured and they are different, which is the point:

======================================  ===============  ==========================
Reference impulse placed at             Measured slope   Terminal offset at F = 1 N
======================================  ===============  ==========================
ignition (``ImpulseEpoch.IGNITION``)    -1.000           3.998 m
delta-v centroid (``ImpulseEpoch.CENTROID``)  -3.001     3.384e-04 m
======================================  ===============  ==========================

Every numerical bound below was set by running the thing first and writing the measurement
into the comment beside it. Reference case throughout: a 200 kg servicer, Isp 220 s
(hydrazine monopropellant), 0.2 m/s commanded delta-v, 420 km circular orbit (period
5578.223 s, n = 1.126378e-03 rad/s). Exhaust velocity c = 2157.4630 m/s, so the burn
duration is 40.0 / F seconds.
"""

import math
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M, orbital_period_s
from rpo_core.exceptions import DegenerateGeometryError, PropagationError
from rpo_core.finite_burn import (
    STANDARD_GRAVITY_M_S2,
    FiniteBurn,
    FiniteBurnLoss,
    ImpulseEpoch,
    PropellantExhaustedError,
    ThrustDirection,
    equivalent_impulsive_delta_v,
    finite_burn_derivative,
    finite_burn_loss,
    propagate_with_finite_burn,
    thrust_unit_eci,
)
from rpo_core.propagate import (
    propagate_two_body,
    specific_energy_j_kg,
    two_body_derivative,
)

# ---------------------------------------------------------------------------------------
# The reference vehicle and orbit. 420 km matches the flagship RPO scenario used elsewhere
# in the suite (see rpo_core.targeting); 200 kg / 220 s / 22 N is a small servicer with a
# hydrazine monopropellant thruster, and 0.2 m/s is the order of a terminal-approach burn.
# ---------------------------------------------------------------------------------------
ALTITUDE_M = 420.0e3
A_M = R_EARTH_EQUATORIAL_M + ALTITUDE_M
CIRCULAR_SPEED_M_S = math.sqrt(MU_EARTH_M3_S2 / A_M)
STATE0 = np.array([A_M, 0.0, 0.0, 0.0, CIRCULAR_SPEED_M_S, 0.0])
PERIOD_S = orbital_period_s(A_M)

# STATE0 sits where the ECI-to-Hill rotation is exactly the identity, which makes it a
# useless place to test anything about the *sense* of that rotation. These three do not.
QUARTER_ORBIT_STATE = np.array([0.0, A_M, 0.0, -CIRCULAR_SPEED_M_S, 0.0, 0.0])
HALF_ORBIT_STATE = np.array([-A_M, 0.0, 0.0, 0.0, -CIRCULAR_SPEED_M_S, 0.0])
INCLINED_STATE = np.array(
    [
        A_M * math.cos(0.7),
        A_M * math.sin(0.7) * math.cos(0.9),
        A_M * math.sin(0.7) * math.sin(0.9),
        -CIRCULAR_SPEED_M_S * math.sin(0.7),
        CIRCULAR_SPEED_M_S * math.cos(0.7) * math.cos(0.9),
        CIRCULAR_SPEED_M_S * math.cos(0.7) * math.sin(0.9),
    ]
)

MASS_KG = 200.0
ISP_S = 220.0
DELTA_V_M_S = 0.2
V_BAR = (0.0, 1.0, 0.0)

# g0 written as a literal, deliberately: every expected value in this file must be
# computable without asking the module under test what it thinks standard gravity is.
G0_LITERAL = 9.80665


def _burn(
    thrust_n: float,
    *,
    policy: ThrustDirection = ThrustDirection.INERTIAL_FIXED,
    delta_v_m_s: float = DELTA_V_M_S,
    start_time_s: float = 0.0,
    direction: tuple[float, float, float] = V_BAR,
) -> FiniteBurn:
    return FiniteBurn(
        thrust_n,
        ISP_S,
        MASS_KG,
        direction,
        start_time_s=start_time_s,
        commanded_delta_v_m_s=delta_v_m_s,
        direction_policy=policy,
    )


def _thrust_for_duration_n(duration_s: float, delta_v_m_s: float = DELTA_V_M_S) -> float:
    """Thrust that makes a ``delta_v_m_s`` burn last exactly ``duration_s``."""
    exhaust_velocity_m_s = ISP_S * G0_LITERAL
    propellant_kg = MASS_KG * -math.expm1(-delta_v_m_s / exhaust_velocity_m_s)
    return exhaust_velocity_m_s * propellant_kg / duration_s


def _log_log_slope(x: list[float], y: list[float]) -> list[float]:
    """Pairwise log-log slopes ``d ln y / d ln x``."""
    return [math.log(y[i + 1] / y[i]) / math.log(x[i + 1] / x[i]) for i in range(len(x) - 1)]


# =======================================================================================
# THE HEADLINE: convergence toward the impulsive limit, and its rate
# =======================================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_terminal_offset_converges_first_order_toward_the_impulsive_limit():
    r"""Terminal offset must fall as ``1/F`` over five decades of thrust.

    Model M8's independent check. At fixed total impulse the burn duration is
    ``t_b = 40.0 / F`` seconds, and a constant acceleration delivers only *half* the
    position gain of an equal impulse applied at the start of the same interval, so the
    leading term is ``0.5 * delta_v * t_b``, first order in ``t_b`` and hence in ``1/F``.

    Measured, 1 N to 1e5 N::

        F (N)        t_b (s)     |dr| (m)      slope
        1            39.998      3.997850e+00
        10            3.9998     3.999856e-01  -0.9998
        100           0.39998    3.999876e-02  -1.0000
        1000          0.039998   3.999876e-03  -1.0000
        1e4           0.0039998  3.999876e-04  -1.0000
        1e5           0.00039998 3.999876e-05  -1.0000

    An implementation that applied the delta-v impulsively and merely *reported* a burn
    duration would give an offset of zero at every thrust, so this fails for it. One that
    integrated the thrust but forgot the mass flow, or froze a rotating direction, would
    still converge -- but its slope and its 0.5*delta_v*t_b coefficient would not both come
    out right, which is why the coefficient is asserted too.
    """
    thrusts_n = [1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5]
    offsets_m = [
        finite_burn_loss(STATE0, _burn(f), MU_EARTH_M3_S2).terminal_position_offset_m
        for f in thrusts_n
    ]

    # Monotone decrease is the weak statement; the slope is the real one.
    assert offsets_m == sorted(offsets_m, reverse=True)

    slopes = _log_log_slope(thrusts_n, offsets_m)
    # Measured: -0.9998, -1.0000, -1.0000, -1.0000, -1.0000. Bound +-0.02 is 100x the
    # worst measured deviation from -1 (2e-4), which is ample headroom for integrator noise
    # while still rejecting order 0 (impulsive) and order 2.
    for slope in slopes:
        assert -1.02 < slope < -0.98, f"slopes were {slopes}"

    # The limit itself. Measured 3.999876e-05 m at 1e5 N; bound 1e-4 m.
    assert offsets_m[-1] < 1.0e-4

    # The leading-order coefficient, independently predicted. Measured ratio of the
    # observed offset to 0.5*delta_v*t_b: 0.99951 at 1 N, 1.00000 at 1e5 N. Bound 0.3 %.
    for thrust_n, offset_m in zip(thrusts_n, offsets_m, strict=True):
        predicted_m = 0.5 * DELTA_V_M_S * _burn(thrust_n).burn_duration_s
        assert abs(offset_m / predicted_m - 1.0) < 3.0e-3


@pytest.mark.slow
@pytest.mark.integration
def test_centroid_aligned_reference_converges_third_order():
    r"""Placing the reference impulse at the burn centroid raises the order from 1 to 3.

    The complement to the test above, and the module's most actionable result. Matching the
    burn's zeroth moment (total delta-v) and its first moment (the delta-v-weighted centroid
    time) kills both the constant and the linear term of the position error, leaving the
    second-moment mismatch at third order.

    Measured::

        F (N)   t_b (s)     |dr| centroid (m)   slope    |dr| ignition (m)
        0.1     399.98      3.505394e-01                 3.833190e+01
        1        39.998     3.384022e-04        -3.015   3.997850e+00
        10        3.9998    3.382890e-07        -3.000   3.999856e-01
        100       0.39998   3.374225e-10        -3.001   3.999876e-02

    Below 1e-12 m the offset reaches the integrator's own noise floor (measured 3.98e-13 m
    at 1000 N, where the apparent slope degrades to -2.93), so the sweep stops at 100 N.
    """
    thrusts_n = [0.1, 1.0, 10.0, 100.0]
    offsets_m = [
        finite_burn_loss(
            STATE0, _burn(f), MU_EARTH_M3_S2, impulse_epoch=ImpulseEpoch.CENTROID
        ).terminal_position_offset_m
        for f in thrusts_n
    ]

    slopes = _log_log_slope(thrusts_n, offsets_m)
    # Measured: -3.0153, -3.0001, -3.0011. Bound +-0.15 covers the worst deviation (0.015)
    # with 10x headroom and still separates order 3 from order 2 unambiguously.
    for slope in slopes:
        assert -3.15 < slope < -2.85, f"slopes were {slopes}"

    # And the practical consequence: the same physical burn, judged against a better-placed
    # reference impulse, is orders of magnitude closer to impulsive. Measured ratios
    # ignition/centroid: 109 at 0.1 N, 11814 at 1 N, 1.18e6 at 10 N, 1.19e8 at 100 N.
    for thrust_n, centroid_offset_m in zip(thrusts_n[1:], offsets_m[1:], strict=True):
        ignition_offset_m = finite_burn_loss(
            STATE0, _burn(thrust_n), MU_EARTH_M3_S2
        ).terminal_position_offset_m
        assert ignition_offset_m / centroid_offset_m > 1.0e4


@pytest.mark.slow
@pytest.mark.integration
def test_hill_fixed_policy_also_converges_to_its_own_impulsive_limit():
    """Convergence is a property of the dynamics, not of the inertial direction policy.

    Complement to the headline: if only the inertially-fixed burn converged, the Hill branch
    could be silently broken. A Hill-fixed burn's impulsive limit is the same impulse (the
    frame has not had time to rotate), so it must converge at the same first order.

    Measured |dr| (m): 3.998980e+00, 3.999867e-01, 3.999876e-02, 3.999876e-03, 3.999876e-04
    at 1, 10, 100, 1e3, 1e4 N -- slopes -0.9999, -1.0000, -1.0000, -1.0000.
    """
    thrusts_n = [1.0, 10.0, 100.0, 1.0e3, 1.0e4]
    offsets_m = [
        finite_burn_loss(
            STATE0, _burn(f, policy=ThrustDirection.HILL_FIXED), MU_EARTH_M3_S2
        ).terminal_position_offset_m
        for f in thrusts_n
    ]
    for slope in _log_log_slope(thrusts_n, offsets_m):
        assert -1.02 < slope < -0.98
    assert offsets_m[-1] < 1.0e-3  # measured 3.999876e-04 m


# =======================================================================================
# Mass flow against Tsiolkovsky
# =======================================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("thrust_n", "delta_v_m_s"),
    [(1.0, 0.2), (22.0, 0.2), (5.0, 50.0), (100.0, 500.0)],
)
def test_propagated_mass_matches_tsiolkovsky_in_free_space(thrust_n, delta_v_m_s):
    """The seventh state must reproduce ``m0 * exp(-dv / (Isp * g0))`` with g0 = 9.80665.

    Run with ``mu_m3_s2 = 0`` so there is no gravity term to confound either check. Two
    independent things are asserted: the propagated mass against the closed form, and the
    achieved velocity change against the same closed form read the other way round. The
    expected values are built from the literal 9.80665 rather than from the module's
    constant, so a wrong g0 in the implementation cannot cancel out of both sides.

    Measured relative errors: mass 1.42e-16 (one ulp -- mass is linear in time, so DOP853
    integrates it exactly), achieved delta-v 9.10e-13 at 0.2 m/s and 1.21e-12 at 500 m/s.
    """
    burn = _burn(thrust_n, delta_v_m_s=delta_v_m_s)
    states = propagate_with_finite_burn(STATE0, (0.0, burn.end_time_s), burn, 0.0)

    expected_mass_kg = MASS_KG * math.exp(-delta_v_m_s / (ISP_S * G0_LITERAL))
    # Bound 1e-13: 700x the measured 1.42e-16, and still far tighter than any plausible
    # "mass not actually propagated" defect (which lands at 9.3e-5 relative here).
    assert abs(states[-1, 6] / expected_mass_kg - 1.0) < 1.0e-13

    achieved_delta_v_m_s = float(np.linalg.norm(states[-1, 3:6] - STATE0[3:]))
    # Bound 1e-10: 80x the worst measured 1.21e-12.
    assert abs(achieved_delta_v_m_s / delta_v_m_s - 1.0) < 1.0e-10


@pytest.mark.unit
def test_duration_specified_burn_reproduces_its_own_tsiolkovsky_delta_v():
    """A burn sized by duration must agree with one sized by the delta-v it delivers.

    The two specification branches meet here: ``ideal_delta_v_m_s`` is derived from the mass
    ratio, the propagated mass is derived from the ODE, and the free-space velocity change
    is derived from the momentum. All three must be the same number.

    Measured: spec 0.200009270720326, from propagated mass 0.200009270719847, achieved
    0.200009270720329 -- agreement to 2.4e-12 relative.
    """
    burn = FiniteBurn(1.0, ISP_S, MASS_KG, V_BAR, duration_s=40.0)
    states = propagate_with_finite_burn(STATE0, (0.0, 40.0), burn, 0.0)

    delta_v_from_mass_m_s = ISP_S * G0_LITERAL * math.log(MASS_KG / states[-1, 6])
    achieved_m_s = float(np.linalg.norm(states[-1, 3:6] - STATE0[3:]))
    # Bound 1e-10: 40x the measured 2.4e-12.
    assert abs(delta_v_from_mass_m_s / burn.ideal_delta_v_m_s - 1.0) < 1.0e-10
    assert abs(achieved_m_s / burn.ideal_delta_v_m_s - 1.0) < 1.0e-10


@pytest.mark.unit
def test_mass_falls_only_while_the_thruster_is_lit():
    """Mass is constant before ignition, strictly falling during, constant after cut-off.

    Measured: mass is bit-identical across every sample after cut-off (peak-to-peak 0.0) and
    matches the analytic final mass exactly. A defect that applied thrust after burnout, or
    that ignored ``start_time_s``, changes the mass history here even when the terminal
    position happens to look plausible.
    """
    burn = _burn(1.0, delta_v_m_s=1.0, start_time_s=60.0)
    times_s = np.linspace(0.0, burn.end_time_s + 60.0, 241)
    mass_kg = propagate_with_finite_burn(STATE0, times_s, burn, MU_EARTH_M3_S2)[:, 6]

    before = times_s <= burn.start_time_s
    during = (times_s > burn.start_time_s) & (times_s < burn.end_time_s)
    after = times_s >= burn.end_time_s

    assert np.ptp(mass_kg[before]) == 0.0
    assert mass_kg[before][0] == MASS_KG
    assert np.all(np.diff(mass_kg[during]) < 0.0)
    assert np.ptp(mass_kg[after]) == 0.0
    assert mass_kg[after][0] == pytest.approx(burn.final_mass_kg, rel=1.0e-14)

    # The mass flow itself, read off the slope. Analytic mdot = F/(g0*Isp).
    expected_mdot_kg_s = burn.thrust_n / (G0_LITERAL * ISP_S)
    measured_mdot_kg_s = -np.diff(mass_kg[during]) / np.diff(times_s[during])
    assert np.allclose(measured_mdot_kg_s, expected_mdot_kg_s, rtol=1.0e-10)


# =======================================================================================
# The zero-thrust limiting case
# =======================================================================================


@pytest.mark.unit
def test_burn_outside_the_window_reproduces_two_body_bit_for_bit():
    """A propagation with no powered arc must be byte-identical to ``propagate_two_body``.

    Not "close to" -- identical. Coast segments are handed to ``propagate_two_body`` with
    the caller's own time array, so the two calls make the same ``solve_ivp`` call with the
    same arguments. Measured: ``np.array_equal`` is True and the max absolute difference is
    exactly 0.0 over half an orbit and 17 samples.

    This is the strongest available statement that the finite-burn machinery adds nothing
    when it should not, and it would fail immediately for an implementation that carried a
    seventh state through the coast (the extra component changes DOP853's error norm and
    therefore its step selection).
    """
    times_s = np.linspace(0.0, 0.5 * PERIOD_S, 17)
    burn = _burn(22.0, start_time_s=float(times_s[-1]) + 1.0)

    got = propagate_with_finite_burn(STATE0, times_s, burn, MU_EARTH_M3_S2)
    expected = propagate_two_body(STATE0, times_s, MU_EARTH_M3_S2)

    assert np.array_equal(got[:, :6], expected)
    assert np.array_equal(got[:, 6], np.full(times_s.size, MASS_KG))


@pytest.mark.unit
def test_zero_thrust_derivative_is_exactly_the_two_body_derivative():
    """With ``F = 0`` the seven-state right-hand side must reduce exactly to the six-state one.

    The limiting case at the level of the equations rather than the trajectory, and the
    complement that stops the test above from being satisfiable by a special case: here the
    powered code path is executed, it just has nothing to add.
    """
    state7 = np.concatenate((STATE0, (MASS_KG,)))
    derivative = finite_burn_derivative(
        0.0,
        state7,
        MU_EARTH_M3_S2,
        0.0,
        0.0,
        np.array([0.0, 1.0, 0.0]),
        ThrustDirection.INERTIAL_FIXED,
    )
    assert np.array_equal(derivative[:6], two_body_derivative(0.0, STATE0, MU_EARTH_M3_S2))
    assert derivative[6] == 0.0


@pytest.mark.unit
def test_thrust_enters_the_derivative_as_f_over_m():
    """Complement to the zero-thrust case: with thrust on, the added term is exactly ``F/m``.

    Proves the zero-thrust agreement above is a knife edge and not a plateau -- if the
    thrust term were dropped entirely, the test above would still pass and this one would
    not.
    """
    state7 = np.concatenate((STATE0, (MASS_KG,)))
    thrust_n = 22.0
    derivative = finite_burn_derivative(
        0.0,
        state7,
        MU_EARTH_M3_S2,
        thrust_n,
        thrust_n / (G0_LITERAL * ISP_S),
        np.array([0.0, 1.0, 0.0]),
        ThrustDirection.INERTIAL_FIXED,
    )
    baseline = two_body_derivative(0.0, STATE0, MU_EARTH_M3_S2)
    added = derivative[3:6] - baseline[3:]
    assert added == pytest.approx([0.0, thrust_n / MASS_KG, 0.0], abs=1.0e-15)
    assert derivative[6] == pytest.approx(-thrust_n / (G0_LITERAL * ISP_S), rel=1.0e-15)


# =======================================================================================
# The two direction policies
# =======================================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    ("orbit_fraction", "min_position_offset_m", "min_velocity_offset_m_s"),
    # Measured |dr| / |dv| between the two policies at cut-off:
    #   10 % of an orbit: 1.200443e+01 m / 6.567362e-02 m/s
    #   25 % of an orbit: 7.769680e+01 m / 1.685368e-01 m/s
    #   50 % of an orbit: 2.012581e+02 m / 1.123616e-01 m/s
    # Bounds set at roughly half the measured value.
    [(0.10, 6.0, 0.03), (0.25, 40.0, 0.08), (0.50, 100.0, 0.05)],
)
def test_direction_policies_differ_materially_for_a_long_burn(
    orbit_fraction, min_position_offset_m, min_velocity_offset_m_s
):
    """Inertially-fixed and Hill-fixed thrust must give materially different answers.

    This is the test that makes the policy switch load-bearing. If ``HILL_FIXED`` silently
    behaved like ``INERTIAL_FIXED`` -- a frozen direction where a rotating one was
    commanded, which is the classic finite-burn implementation bug -- the two would agree to
    round-off here and this fails.

    Note the scale: for a burn spanning a quarter of an orbit the *velocity* disagreement is
    0.169 m/s against a commanded 0.2 m/s. The choice of policy is not a refinement, it is
    most of the answer.
    """
    duration_s = orbit_fraction * PERIOD_S
    thrust_n = _thrust_for_duration_n(duration_s)
    inertial = _burn(thrust_n, policy=ThrustDirection.INERTIAL_FIXED)
    hill = _burn(thrust_n, policy=ThrustDirection.HILL_FIXED)
    cutoff_s = inertial.end_time_s

    state_inertial = propagate_with_finite_burn(STATE0, (0.0, cutoff_s), inertial, MU_EARTH_M3_S2)[
        -1
    ]
    state_hill = propagate_with_finite_burn(STATE0, (0.0, cutoff_s), hill, MU_EARTH_M3_S2)[-1]

    assert float(np.linalg.norm(state_inertial[:3] - state_hill[:3])) > min_position_offset_m
    assert float(np.linalg.norm(state_inertial[3:6] - state_hill[3:6])) > min_velocity_offset_m_s


@pytest.mark.slow
@pytest.mark.integration
def test_direction_policies_converge_second_order_for_a_short_burn():
    """The two policies must agree as the burn shortens, at second order in duration.

    The complement to the test above: a difference that never shrinks would mean one of the
    two policies is simply wrong rather than differently right. The rotation angle over the
    burn is ``n * t_b``, and the resulting position discrepancy goes as ``delta_v * t_b *
    (n * t_b)``, i.e. second order.

    Measured::

        F (N)   t_b (s)   n*t_b       |dr| (m)
        0.4     100       1.126e-01   3.758119e-01
        4        10       1.126e-02   3.754307e-03   slope -2.0004
        40        1       1.126e-03   3.754348e-05   slope -2.0000
        400       0.1     1.126e-04   3.762543e-07   slope -1.9991
    """
    thrusts_n = [0.4, 4.0, 40.0, 400.0]
    offsets_m = []
    for thrust_n in thrusts_n:
        inertial = _burn(thrust_n, policy=ThrustDirection.INERTIAL_FIXED)
        hill = _burn(thrust_n, policy=ThrustDirection.HILL_FIXED)
        cutoff_s = inertial.end_time_s
        a = propagate_with_finite_burn(STATE0, (0.0, cutoff_s), inertial, MU_EARTH_M3_S2)[-1]
        b = propagate_with_finite_burn(STATE0, (0.0, cutoff_s), hill, MU_EARTH_M3_S2)[-1]
        offsets_m.append(float(np.linalg.norm(a[:3] - b[:3])))

    # Measured slopes -2.0004, -2.0000, -1.9991; bound +-0.1 is 250x the worst deviation
    # and still cleanly separates order 2 from orders 1 and 3.
    for slope in _log_log_slope(thrusts_n, offsets_m):
        assert -2.1 < slope < -1.9, f"slopes were {_log_log_slope(thrusts_n, offsets_m)}"
    # A one-second burn: measured 3.754348e-05 m. Bound 1e-4 m.
    assert offsets_m[2] < 1.0e-4


@pytest.mark.unit
def test_hill_radial_thrust_stays_exactly_in_the_orbit_plane():
    """A Hill-frame radial burn on a planar orbit must not leave the plane, ever.

    Radial is ``r_hat``, which lies in the orbit plane by construction, so a correct
    rotation from Hill to ECI produces no ``z`` component at any point of the burn. Measured
    max |z| and max |vz|: exactly 0.0. A transposed rotation matrix, or one built with the
    wrong axis order, leaks cross-track immediately.
    """
    burn = FiniteBurn(
        10.0,
        ISP_S,
        MASS_KG,
        (1.0, 0.0, 0.0),
        commanded_delta_v_m_s=1.0,
        direction_policy=ThrustDirection.HILL_FIXED,
    )
    states = propagate_with_finite_burn(
        STATE0, np.linspace(0.0, burn.end_time_s, 20), burn, MU_EARTH_M3_S2
    )
    assert np.max(np.abs(states[:, 2])) == 0.0
    assert np.max(np.abs(states[:, 5])) == 0.0


@pytest.mark.unit
def test_hill_cross_track_thrust_produces_cross_track_velocity_of_the_commanded_size():
    """A Hill-frame cross-track burn must put essentially all of its delta-v into ``vz``.

    ``z_hat`` is the positive orbit normal (``docs/conventions.md``), which for this planar
    orbit is inertially fixed, so a 1 m/s commanded burn lands 1 m/s of ``vz``. Measured
    ``vz`` at cut-off: 0.9999154665 m/s, i.e. 8.5e-5 low -- that is the cross-track restoring
    acceleration acting over the 21.6 s burn, not an implementation error. In-plane residual
    speed change: 6.53e-05 m/s.

    Complement to the radial test: together they pin down the sign and the ordering of the
    Hill triad, not merely its orthonormality.
    """
    burn = FiniteBurn(
        10.0,
        ISP_S,
        MASS_KG,
        (0.0, 0.0, 1.0),
        commanded_delta_v_m_s=1.0,
        direction_policy=ThrustDirection.HILL_FIXED,
    )
    powered = propagate_with_finite_burn(STATE0, (0.0, burn.end_time_s), burn, MU_EARTH_M3_S2)
    coasting = propagate_two_body(STATE0, (0.0, burn.end_time_s), MU_EARTH_M3_S2)

    # Bound 1e-3: 12x the measured 8.5e-4 shortfall of the 1 m/s command.
    assert powered[-1, 5] == pytest.approx(1.0, abs=1.0e-3)
    in_plane_residual_m_s = float(np.linalg.norm(powered[-1, 3:5] - coasting[-1, 3:5]))
    # Bound 1e-3 m/s: 15x the measured 6.53e-5.
    assert in_plane_residual_m_s < 1.0e-3


@pytest.mark.unit
def test_the_two_policies_agree_at_the_instant_of_ignition():
    """At ignition the Hill-fixed direction, expressed in ECI, is a fixed vector.

    Both policies must therefore start pointing the same way when the commanded Hill vector
    is converted through the ignition-epoch frame. This is what makes the comparison in
    ``finite_burn_loss`` fair: the two policies differ because of what happens *after*
    ignition, not because they were aimed differently.

    Evaluated at a quarter-orbit state rather than at ``STATE0``. At ``STATE0`` the
    ECI-to-Hill rotation is the identity, so this assertion would hold for a transposed,
    mirrored, or otherwise wrong rotation -- a test with no bite. See
    ``test_hill_axes_are_the_radial_along_track_and_normal_directions``.
    """
    hill_unit = np.array([0.3, 0.9, -0.2])
    hill_unit /= np.linalg.norm(hill_unit)
    eci_unit = thrust_unit_eci(QUARTER_ORBIT_STATE, hill_unit, ThrustDirection.HILL_FIXED)
    assert float(np.linalg.norm(eci_unit)) == pytest.approx(1.0, abs=1.0e-15)
    # Same vector rebuilt as an inertially-fixed command must be returned unchanged.
    assert np.array_equal(
        thrust_unit_eci(QUARTER_ORBIT_STATE, eci_unit, ThrustDirection.INERTIAL_FIXED),
        eci_unit,
    )


@pytest.mark.unit
def test_hill_axes_are_the_radial_along_track_and_normal_directions():
    """A Hill-frame command must be rotated *into* ECI, not through the inverse rotation.

    ``docs/conventions.md`` fixes the triad: for a circular orbit the Hill x axis is
    ``r_hat``, the y axis is ``v_hat``, and the z axis is ``h_hat``. Asserting that
    directly is an external statement about the frame, not a restatement of the code.

    This test exists because it was *missing*: a mutation replacing ``rotation.T @ u`` with
    ``rotation @ u`` survived the entire rest of the suite. Every other Hill test used a
    reference state at which the ECI-to-Hill rotation happens to be the identity, where the
    two are indistinguishable. A quarter of an orbit later they are exact opposites -- the
    correct V-bar direction there is ``[-1, 0, 0]`` and the transposed one is ``[+1, 0, 0]``
    -- so a V-bar burn planned with the wrong sense would brake instead of accelerating.
    """
    for state in (STATE0, QUARTER_ORBIT_STATE, HALF_ORBIT_STATE, INCLINED_STATE):
        r_hat = state[:3] / np.linalg.norm(state[:3])
        v_hat = state[3:] / np.linalg.norm(state[3:])
        h = np.cross(state[:3], state[3:])
        h_hat = h / np.linalg.norm(h)

        assert thrust_unit_eci(
            state, np.array([1.0, 0.0, 0.0]), ThrustDirection.HILL_FIXED
        ) == pytest.approx(r_hat, abs=1.0e-15)
        assert thrust_unit_eci(
            state, np.array([0.0, 1.0, 0.0]), ThrustDirection.HILL_FIXED
        ) == pytest.approx(v_hat, abs=1.0e-15)
        assert thrust_unit_eci(
            state, np.array([0.0, 0.0, 1.0]), ThrustDirection.HILL_FIXED
        ) == pytest.approx(h_hat, abs=1.0e-15)


@pytest.mark.integration
def test_hill_fixed_v_bar_burn_adds_energy_wherever_it_ignites():
    """A V-bar burn is prograde by definition, so it must raise the orbit from any epoch.

    The trajectory-level consequence of the axis test above, and the one a mission designer
    would notice: a Hill-fixed V-bar burn ignited a quarter of an orbit after epoch must
    still speed the vehicle up. Under a transposed rotation it points anti-velocity there
    and removes energy instead -- a sign error that turns a raise into a lower.

    Specific orbital energy is the right diagnostic because it is frame-independent and
    monotone in the work done by the thruster.
    """
    for start_fraction in (0.0, 0.25, 0.5, 0.75):
        burn = _burn(
            10.0,
            policy=ThrustDirection.HILL_FIXED,
            delta_v_m_s=1.0,
            start_time_s=start_fraction * PERIOD_S,
        )
        final = propagate_with_finite_burn(STATE0, (0.0, burn.end_time_s), burn, MU_EARTH_M3_S2)[-1]
        energy_before = specific_energy_j_kg(STATE0)
        energy_after = specific_energy_j_kg(final[:6])
        # A 1 m/s prograde burn on a 7657 m/s orbit adds about v*dv = 7.66e3 J/kg.
        assert energy_after - energy_before > 1.0e3


# =======================================================================================
# Finite-burn loss
# =======================================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_finite_burn_loss_grows_as_thrust_falls():
    """The gravity/steering loss must be monotone in falling thrust, and grow as ``t_b**2``.

    Measured for the reference case (200 kg, Isp 220 s, 0.2 m/s commanded, 420 km)::

        F (N)   t_b (s)   extra delta-v (m/s)   fraction of command   slope
        100       0.4     6.766024e-09          3.38e-08
        22        1.8     1.397887e-07          6.99e-07              -2.0000
        5         8.0     2.705898e-06          1.35e-05              -1.9999
        1        40.0     6.740350e-05          3.37e-04              -1.9978
        0.5      80.0     2.665647e-04          1.33e-03              -1.9836
        0.2     200.0     1.532555e-03          7.66e-03              -1.9089

    The sweep stops at 0.2 N (3.6 % of an orbit) because the metric's own range of validity
    ends near 9 %; see ``test_extra_delta_v_stops_meaning_loss_for_a_burn_spanning_a_large_arc``.
    """
    thrusts_n = [100.0, 22.0, 5.0, 1.0, 0.5, 0.2]
    losses = [finite_burn_loss(STATE0, _burn(f), MU_EARTH_M3_S2) for f in thrusts_n]
    extra_delta_v_m_s = [loss.extra_delta_v_m_s for loss in losses]

    assert all(value > 0.0 for value in extra_delta_v_m_s)
    for lower, higher in pairwise(extra_delta_v_m_s):
        assert higher > lower, f"not monotone: {extra_delta_v_m_s}"

    slopes = _log_log_slope(thrusts_n, extra_delta_v_m_s)
    # Measured -2.0000 down to -1.9089; the drift away from exactly -2 is the burn taking
    # up a growing fraction of an orbit, not noise. Bound [-2.05, -1.85].
    for slope in slopes:
        assert -2.05 < slope < -1.85, f"slopes were {slopes}"


@pytest.mark.unit
def test_finite_burn_loss_is_negligible_at_realistic_rpo_thrust_levels():
    """For this mission class the finite-burn correction does not matter, and here is why.

    A 200 kg servicer with a 22 N hydrazine thruster executing a 0.2 m/s terminal-approach
    burn lights for 1.82 s -- 0.03 % of an orbit. Measured at cut-off:

    * extra delta-v (gravity + steering loss): **1.398e-07 m/s**, i.e. **7.0e-07 of the
      commanded 0.2 m/s**. Below any plausible thruster calibration or execution error
      (model M9 uses magnitude errors of order 1e-2 relative), so it is not merely small,
      it is unobservable.
    * terminal position offset against a centroid-placed impulse: **3.18e-08 m**.
    * terminal position offset against an ignition-placed impulse: **0.182 m**.

    The honest conclusion is that **at realistic RPO thrust levels the finite-burn
    correction is negligible, and the entire visible effect is where the reference impulse
    was placed**. The one caveat worth carrying: at 1 N the ignition-placed offset is
    3.998 m, which exceeds the 2 m CW-linearisation budget quoted in ``docs/cw_validity.md``
    -- so a low-thrust burn planned as an impulse "at" its start time can dominate the
    modelling error the suite otherwise works hard to bound, while the same burn judged
    against a centroid-placed impulse is 0.34 mm off. That is a free correction.
    """
    burn = _burn(22.0)
    ignition = finite_burn_loss(STATE0, burn, MU_EARTH_M3_S2)
    centroid = finite_burn_loss(STATE0, burn, MU_EARTH_M3_S2, impulse_epoch=ImpulseEpoch.CENTROID)

    assert burn.burn_duration_s / PERIOD_S < 1.0e-3  # measured 3.26e-4
    # Bound 1e-6 m/s: 7x the measured 1.398e-07.
    assert 0.0 < ignition.extra_delta_v_m_s < 1.0e-6
    # Bound 1e-5 of the command: 14x the measured 6.99e-07.
    assert ignition.extra_delta_v_m_s / ignition.ideal_delta_v_m_s < 1.0e-5
    # Bound 1e-6 m: 30x the measured 3.18e-08 m.
    assert centroid.terminal_position_offset_m < 1.0e-6
    # And the complement -- the ignition-placed reference is *not* negligible at 0.182 m,
    # which is the whole reason the two epochs exist as separate options.
    assert ignition.terminal_position_offset_m > 0.1


@pytest.mark.slow
@pytest.mark.integration
def test_extra_delta_v_stops_meaning_loss_for_a_burn_spanning_a_large_arc():
    """``extra_delta_v_m_s`` changes sign once the burn spans a large fraction of an orbit.

    Documenting a limitation as a test rather than as a hope. The metric compares a powered
    arc against an unpowered one, which isolates the propulsive loss only while the two arcs
    stay close. Measured::

        t_b / period   extra delta-v (m/s)
        2.39 %          7.203738e-04
        3.59 %          1.532555e-03
        7.17 %          4.230621e-03
        8.96 %          4.422993e-03   <- peak
        11.95 %        -1.188111e-04   <- sign change
        14.34 %        -1.051694e-02

    A future change that clamped the value at zero would hide this and make the metric look
    trustworthy where it is not, so the sign change is asserted, not tolerated.
    """
    short = finite_burn_loss(STATE0, _burn(_thrust_for_duration_n(0.09 * PERIOD_S)), MU_EARTH_M3_S2)
    long = finite_burn_loss(STATE0, _burn(_thrust_for_duration_n(0.14 * PERIOD_S)), MU_EARTH_M3_S2)
    assert short.extra_delta_v_m_s > 0.0
    assert long.extra_delta_v_m_s < 0.0


@pytest.mark.unit
def test_finite_burn_loss_fields_are_self_consistent():
    """Reported fields must agree with each other and with the burn they describe."""
    burn = _burn(1.0)
    loss = finite_burn_loss(STATE0, burn, MU_EARTH_M3_S2)

    assert isinstance(loss, FiniteBurnLoss)
    assert loss.extra_delta_v_m_s == loss.ideal_delta_v_m_s - loss.effective_delta_v_m_s
    assert loss.ideal_delta_v_m_s == burn.ideal_delta_v_m_s
    assert loss.burn_duration_s == burn.burn_duration_s
    assert loss.propellant_mass_kg == burn.propellant_mass_kg
    assert loss.comparison_time_s == burn.end_time_s
    assert loss.impulse_time_s == burn.start_time_s

    centroid = finite_burn_loss(STATE0, burn, MU_EARTH_M3_S2, impulse_epoch=ImpulseEpoch.CENTROID)
    # The centroid sits just past the midpoint: measured fraction 0.500007725 of the burn.
    assert burn.start_time_s + 0.5 * burn.burn_duration_s < centroid.impulse_time_s
    assert centroid.impulse_time_s < burn.end_time_s


@pytest.mark.slow
@pytest.mark.integration
def test_loss_reported_for_a_finite_burn_is_never_identically_zero():
    """A real burn must produce a non-zero offset, at every thrust level tested.

    Guards the failure mode where the loss is computed but reported as zero -- the most
    dangerous possible defect in this module, because a mission designer reads "no
    correction needed" and moves on.
    """
    for thrust_n in (0.5, 1.0, 22.0, 100.0, 1000.0):
        loss = finite_burn_loss(STATE0, _burn(thrust_n), MU_EARTH_M3_S2)
        assert loss.terminal_position_offset_m > 0.0
        assert loss.terminal_velocity_offset_m_s > 0.0
        assert loss.extra_delta_v_m_s != 0.0


# =======================================================================================
# Numerical discipline: tolerance sweep, schedule independence, composition
# =======================================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_headline_offset_survives_a_tolerance_sweep():
    """The quoted offset must be physics, not an integrator setting.

    Measured terminal offset at 1 N across rtol = atol from 1e-9 to 1e-13::

        1e-09  3.997850228053 m
        1e-10  3.997850227646 m
        1e-11  3.997850228053 m
        1e-12  3.997850227937 m
        1e-13  3.997850227821 m

    Spread 4.1e-10 m on 4.0 m, i.e. 1.0e-10 relative -- the result is converged four decades
    before the default tolerance is reached.
    """
    burn = _burn(1.0)
    offsets_m = [
        finite_burn_loss(
            STATE0, burn, MU_EARTH_M3_S2, rtol=tol, atol=tol
        ).terminal_position_offset_m
        for tol in (1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13)
    ]
    # Bound 1e-8 relative: 100x the measured 1.0e-10 spread.
    assert (max(offsets_m) - min(offsets_m)) / offsets_m[-1] < 1.0e-8


@pytest.mark.unit
def test_output_schedule_does_not_change_the_trajectory():
    """Asking for more output times must not change the answer at the times they share.

    The interval is cut into segments at the burn boundaries and each is integrated
    separately; if the segment partition depended on the requested times, a densely sampled
    run and a sparsely sampled one would disagree. Measured max difference over the shared
    times: exactly 0.0 (``t_eval`` does not influence DOP853's step selection).
    """
    burn = _burn(1.0)
    dense_s = np.linspace(0.0, 200.0, 401)
    sparse_s = np.array([0.0, 10.0, 40.0, 100.0, 200.0])
    dense = propagate_with_finite_burn(STATE0, dense_s, burn, MU_EARTH_M3_S2)
    sparse = propagate_with_finite_burn(STATE0, sparse_s, burn, MU_EARTH_M3_S2)
    shared = [int(np.argmin(np.abs(dense_s - t))) for t in sparse_s]
    # Bound 1e-9: measured 0.0 exactly, but tied to scipy's dense-output behaviour, so a
    # metre-scale bound would be too loose and exact equality too brittle across versions.
    assert float(np.max(np.abs(dense[shared] - sparse))) < 1.0e-9


@pytest.mark.integration
def test_coast_burn_coast_composes_with_three_separate_propagations():
    """One call across a delayed burn must equal coast, then burn, then coast, done by hand.

    The semigroup property of the flow. It is what proves ``start_time_s`` is honoured and
    that the state is handed across segment boundaries intact rather than restarted.
    Measured over 900 s with a 40 s burn beginning at t = 300 s: position 4.66e-10 m,
    velocity 4.55e-13 m/s, mass exactly 0.
    """
    delayed = _burn(1.0, start_time_s=300.0)
    horizon_s = 900.0
    combined = propagate_with_finite_burn(STATE0, (0.0, horizon_s), delayed, MU_EARTH_M3_S2)[-1]

    after_coast = propagate_two_body(STATE0, (0.0, 300.0), MU_EARTH_M3_S2)[-1]
    prompt = _burn(1.0)
    after_burn = propagate_with_finite_burn(
        after_coast, (0.0, prompt.burn_duration_s), prompt, MU_EARTH_M3_S2
    )[-1]
    after_second_coast = propagate_two_body(
        after_burn[:6], (0.0, horizon_s - 300.0 - prompt.burn_duration_s), MU_EARTH_M3_S2
    )[-1]

    # Bounds 1e-8 m and 1e-11 m/s: ~20x the measured 4.66e-10 m and 4.55e-13 m/s.
    assert float(np.linalg.norm(combined[:3] - after_second_coast[:3])) < 1.0e-8
    assert float(np.linalg.norm(combined[3:6] - after_second_coast[3:])) < 1.0e-11
    assert combined[6] == after_burn[6]


@pytest.mark.unit
def test_a_burn_straddling_the_horizon_is_cut_off_at_the_horizon():
    """A burn still running at the last output time must have burned exactly that long.

    Measured: a 1 N burn starting at t = 10 s, sampled at t = 20 s, leaves
    199.99536492630463 kg -- exactly ``m0 - mdot * 10``.
    """
    burn = _burn(1.0, start_time_s=10.0)
    mass_kg = propagate_with_finite_burn(STATE0, (0.0, 20.0), burn, MU_EARTH_M3_S2)[-1, 6]
    expected_kg = MASS_KG - 10.0 * 1.0 / (G0_LITERAL * ISP_S)
    assert mass_kg == pytest.approx(expected_kg, rel=1.0e-14)


@pytest.mark.unit
def test_single_output_time_returns_the_initial_state_and_mass():
    states = propagate_with_finite_burn(STATE0, [0.0], _burn(1.0), MU_EARTH_M3_S2)
    assert states.shape == (1, 7)
    assert np.array_equal(states[0, :6], STATE0)
    assert states[0, 6] == MASS_KG


# =======================================================================================
# Tsiolkovsky helpers and the defined constant
# =======================================================================================


@pytest.mark.unit
def test_standard_gravity_is_the_defined_value():
    """g0 is fixed by definition (3rd CGPM 1901), so it is asserted exactly, not approximately.

    A measured quantity would deserve a tolerance. This one does not: 9.80665 m/s^2 is a
    convention for converting specific impulse in seconds to exhaust velocity, and 9.81
    would be a 0.034 % error in every mass flow the module computes.
    """
    assert STANDARD_GRAVITY_M_S2 == 9.80665


@pytest.mark.unit
def test_equivalent_impulsive_delta_v_matches_the_literal_rocket_equation():
    """Tsiolkovsky against the formula written out longhand with a literal g0."""
    assert equivalent_impulsive_delta_v(200.0, 199.0, 220.0) == pytest.approx(
        220.0 * 9.80665 * math.log(200.0 / 199.0), rel=1.0e-15
    )
    # Equal masses spend no propellant and therefore buy no delta-v.
    assert equivalent_impulsive_delta_v(200.0, 200.0, 220.0) == 0.0
    # And it is not symmetric: halving the mass is worth c * ln 2, not something else.
    assert equivalent_impulsive_delta_v(200.0, 100.0, 220.0) == pytest.approx(
        220.0 * 9.80665 * math.log(2.0), rel=1.0e-15
    )


@pytest.mark.unit
def test_burn_derived_quantities_round_trip_through_tsiolkovsky():
    """A burn sized by commanded delta-v must report that delta-v back.

    Duration is inverted from Tsiolkovsky and the ideal delta-v is then recomputed from the
    resulting mass ratio, so agreement is a round trip through ``expm1``/``log1p``.
    Measured relative error: 5.56e-13.
    """
    for thrust_n in (0.1, 1.0, 22.0, 1000.0):
        burn = _burn(thrust_n)
        assert burn.ideal_delta_v_m_s == pytest.approx(DELTA_V_M_S, rel=1.0e-11)
        assert burn.exhaust_velocity_m_s == pytest.approx(220.0 * 9.80665, rel=1.0e-15)
        assert burn.mass_flow_rate_kg_s == pytest.approx(thrust_n / (220.0 * 9.80665), rel=1.0e-15)
        assert burn.final_mass_kg == pytest.approx(
            MASS_KG * math.exp(-DELTA_V_M_S / (220.0 * 9.80665)), rel=1.0e-12
        )
        assert burn.end_time_s == burn.start_time_s + burn.burn_duration_s


@pytest.mark.unit
def test_delta_v_centroid_matches_its_series_expansion():
    """The centroid fraction must equal ``1/2 + x/12 + x**2/24`` for small burnt fraction.

    Derived independently of the implementation, which evaluates the closed form (with a
    series branch only to dodge cancellation). Measured for the reference burn:
    x = 9.269718e-05, fraction = 0.5000077251228244, and ``1/2 + x/12 + x**2/24`` =
    0.5000077251228034. The 2.1e-14 residual is the truncated ``O(x**3)`` term of the
    expansion (``x**3`` itself is 8.0e-13 here), not an implementation error -- the ``x**2``
    term alone accounts for 3.5805e-10 of the 3.5805e-10 gap from ``1/2 + x/12``.

    Note the fraction exceeds 1/2: the acceleration ``F/m`` rises through the burn, so the
    delta-v centroid sits *after* the midpoint. An implementation that used the midpoint
    instead would land 7.7e-06 low here -- 3.7e08 times the bound below.
    """
    burn = _burn(1.0)
    x = burn.propellant_mass_kg / burn.initial_mass_kg
    fraction = (burn.delta_v_centroid_time_s - burn.start_time_s) / burn.burn_duration_s
    # Bound 1e-13: 5x the measured 2.1e-14 series-truncation residual, and eight orders of
    # magnitude below the 7.7e-06 error a midpoint implementation would make.
    assert fraction == pytest.approx(0.5 + x / 12.0 + x * x / 24.0, abs=1.0e-13)
    assert fraction > 0.5

    # A large burn, where the closed form is used rather than the series branch.
    big = _burn(1000.0, delta_v_m_s=1500.0)
    x_big = big.propellant_mass_kg / big.initial_mass_kg
    fraction_big = (big.delta_v_centroid_time_s - big.start_time_s) / big.burn_duration_s
    expected = (-math.log1p(-x_big) - x_big) / (x_big * -math.log1p(-x_big))
    assert fraction_big == pytest.approx(expected, rel=1.0e-14)


# =======================================================================================
# Raise paths -- every one, matched on message content
# =======================================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thrust_n": 0.0}, r"thrust_n must be finite and > 0, got 0\.0"),
        ({"thrust_n": -1.0}, r"thrust_n must be finite and > 0, got -1\.0"),
        ({"thrust_n": float("inf")}, r"thrust_n must be finite and > 0"),
        ({"specific_impulse_s": 0.0}, r"specific_impulse_s must be finite and > 0, got 0\.0"),
        ({"specific_impulse_s": -220.0}, r"specific_impulse_s must be finite and > 0"),
        ({"initial_mass_kg": 0.0}, r"initial_mass_kg must be finite and > 0, got 0\.0"),
        ({"initial_mass_kg": -200.0}, r"initial_mass_kg must be finite and > 0"),
        ({"initial_mass_kg": float("nan")}, r"initial_mass_kg must be finite and > 0"),
        ({"start_time_s": -1.0}, r"start_time_s must be finite and >= 0, got -1\.0"),
        ({"start_time_s": float("nan")}, r"start_time_s must be finite and >= 0"),
    ],
)
def test_non_positive_thrust_isp_or_mass_raises(kwargs: dict[str, Any], match: str):
    defaults: dict[str, Any] = {
        "thrust_n": 1.0,
        "specific_impulse_s": ISP_S,
        "initial_mass_kg": MASS_KG,
        "direction_unit": V_BAR,
        "duration_s": 1.0,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FiniteBurn(**defaults)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("direction", "match"),
    [
        ((0.0, 0.0, 0.0), r"direction_unit has zero magnitude"),
        ((0.0, 1.0), r"direction_unit must have shape \(3,\), got \(2,\)"),
        ((0.0, 0.0, 0.0, 1.0), r"direction_unit must have shape \(3,\), got \(4,\)"),
        ((0.0, float("nan"), 0.0), r"direction_unit must be finite"),
        ((float("inf"), 0.0, 0.0), r"direction_unit must be finite"),
    ],
)
def test_malformed_thrust_direction_raises(direction, match: str):
    with pytest.raises(ValueError, match=match):
        FiniteBurn(1.0, ISP_S, MASS_KG, direction, duration_s=1.0)


@pytest.mark.unit
def test_specifying_both_or_neither_burn_length_raises():
    with pytest.raises(ValueError, match=r"specify exactly one of duration_s and"):
        FiniteBurn(1.0, ISP_S, MASS_KG, V_BAR, duration_s=1.0, commanded_delta_v_m_s=DELTA_V_M_S)
    with pytest.raises(ValueError, match=r"specify exactly one of duration_s and"):
        FiniteBurn(1.0, ISP_S, MASS_KG, V_BAR)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration_s": 0.0}, r"duration_s must be finite and > 0, got 0\.0"),
        ({"duration_s": -1.0}, r"duration_s must be finite and > 0, got -1\.0"),
        ({"commanded_delta_v_m_s": 0.0}, r"commanded_delta_v_m_s must be finite and > 0"),
        ({"commanded_delta_v_m_s": -0.2}, r"commanded_delta_v_m_s must be finite and > 0"),
    ],
)
def test_non_positive_burn_length_raises(kwargs: dict[str, Any], match: str):
    with pytest.raises(ValueError, match=match):
        FiniteBurn(1.0, ISP_S, MASS_KG, V_BAR, **kwargs)


@pytest.mark.unit
def test_non_enum_direction_policy_raises():
    with pytest.raises(ValueError, match=r"direction_policy must be a ThrustDirection"):
        FiniteBurn(1.0, ISP_S, MASS_KG, V_BAR, duration_s=1.0, direction_policy="hill_fixed")


@pytest.mark.unit
def test_a_burn_that_would_consume_the_whole_vehicle_raises():
    """Mass reaching zero raises at construction, with the numbers that motivated it.

    A 1000 N burn for 1000 s at Isp 220 s wants 463.507 kg from a 200 kg vehicle, and the
    message says so along with the 431.5 s at which the vehicle runs out.
    """
    with pytest.raises(PropellantExhaustedError, match=r"consumes 463\.507 kg"):
        FiniteBurn(1000.0, ISP_S, MASS_KG, V_BAR, duration_s=1000.0)
    # The exact boundary: consuming precisely the whole mass is rejected too, because the
    # final acceleration F/m is unbounded there.
    with pytest.raises(PropellantExhaustedError, match=r"runs out of vehicle after"):
        FiniteBurn(220.0 * 9.80665, 220.0, 100.0, V_BAR, duration_s=100.0)


@pytest.mark.unit
def test_a_burn_sized_by_delta_v_can_never_exhaust_the_vehicle():
    """Complement: the Tsiolkovsky branch has no exhaustion path, however large the delta-v.

    ``m_f = m_0 exp(-dv/c)`` is positive for every finite ``dv``, so the guard above is
    specific to the duration branch. Asserting it stops a future change from "fixing" a
    non-problem by clamping the delta-v branch.
    """
    extreme = FiniteBurn(1000.0, ISP_S, MASS_KG, V_BAR, commanded_delta_v_m_s=20_000.0)
    assert extreme.final_mass_kg > 0.0
    assert extreme.final_mass_kg < 0.01 * MASS_KG  # measured 1.9e-02 kg


@pytest.mark.unit
def test_mass_reaching_zero_inside_the_derivative_raises():
    """The integrator-level backstop, reachable only by driving the derivative directly.

    ``FiniteBurn`` proves exhaustion cannot happen over the commanded duration, so this
    branch exists for the case where the right-hand side is used outside that contract. It
    must still refuse rather than return an ``F/m`` with negative ``m``.
    """
    state7 = np.concatenate((STATE0, (0.0,)))
    with pytest.raises(PropellantExhaustedError, match=r"vehicle mass reached 0 kg"):
        finite_burn_derivative(
            0.0,
            state7,
            MU_EARTH_M3_S2,
            1.0,
            1.0e-4,
            np.array([0.0, 1.0, 0.0]),
            ThrustDirection.INERTIAL_FIXED,
        )
    negative = np.concatenate((STATE0, (-1.0,)))
    with pytest.raises(PropellantExhaustedError, match=r"vehicle mass reached -1 kg"):
        finite_burn_derivative(
            0.0,
            negative,
            MU_EARTH_M3_S2,
            1.0,
            1.0e-4,
            np.array([0.0, 1.0, 0.0]),
            ThrustDirection.INERTIAL_FIXED,
        )


@pytest.mark.unit
def test_hill_policy_on_a_degenerate_state_raises():
    """A radial trajectory has no LVLH frame, so a Hill-fixed direction is undefined there.

    Surfaced from ``rpo_core.frames.hill_basis`` rather than reimplemented, which is the
    point of calling it from the hot path.
    """
    radial = np.array([A_M, 0.0, 0.0, 1.0, 0.0, 0.0, MASS_KG])
    with pytest.raises(DegenerateGeometryError, match=r"parallel to within"):
        finite_burn_derivative(
            0.0,
            radial,
            MU_EARTH_M3_S2,
            1.0,
            1.0e-4,
            np.array([0.0, 1.0, 0.0]),
            ThrustDirection.HILL_FIXED,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("times_s", "match"),
    [
        ([1.0, 2.0], r"times_s must start at 0\.0"),
        ([], r"times_s must be a non-empty 1-D array"),
        ([[0.0, 1.0]], r"times_s must be a non-empty 1-D array"),
        ([0.0, float("nan")], r"times_s must be finite"),
        ([0.0, 2.0, 2.0], r"times_s must be strictly increasing after the first entry"),
        ([0.0, 2.0, 1.0], r"times_s must be strictly increasing after the first entry"),
        ([0.0, 0.0], r"times_s must be strictly increasing after the first entry"),
    ],
)
def test_malformed_time_schedule_raises(times_s, match: str):
    with pytest.raises(ValueError, match=match):
        propagate_with_finite_burn(STATE0, times_s, _burn(1.0), MU_EARTH_M3_S2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "match"),
    [
        ([1.0, 2.0, 3.0], r"state0_eci must have shape \(6,\), got \(3,\)"),
        (np.zeros((2, 6)), r"state0_eci must have shape \(6,\)"),
        ([A_M, 0.0, 0.0, 0.0, float("nan"), 0.0], r"state0_eci must be finite"),
    ],
)
def test_malformed_initial_state_raises(state, match: str):
    with pytest.raises(ValueError, match=match):
        propagate_with_finite_burn(state, (0.0, 10.0), _burn(1.0), MU_EARTH_M3_S2)


@pytest.mark.unit
@pytest.mark.parametrize("mu_m3_s2", [-1.0, float("nan"), float("-inf")])
def test_negative_or_non_finite_mu_raises(mu_m3_s2):
    with pytest.raises(ValueError, match=r"mu_m3_s2 must be finite and >= 0"):
        propagate_with_finite_burn(STATE0, (0.0, 10.0), _burn(1.0), mu_m3_s2)
    with pytest.raises(ValueError, match=r"mu_m3_s2 must be finite and >= 0"):
        finite_burn_loss(STATE0, _burn(1.0), mu_m3_s2)


@pytest.mark.unit
def test_zero_mu_is_accepted_as_field_free_space():
    """Complement to the check above: zero is a legitimate value, not an omission.

    Field-free space is the configuration the Tsiolkovsky tests need, so rejecting mu = 0
    alongside negative mu would remove the module's only gravity-free oracle.
    """
    burn = _burn(1.0)
    states = propagate_with_finite_burn(STATE0, (0.0, burn.end_time_s), burn, 0.0)
    assert np.all(np.isfinite(states))
    # Straight-line motion: the velocity change is exactly along the commanded direction.
    delta_v = states[-1, 3:6] - STATE0[3:]
    assert delta_v[0] == 0.0
    assert delta_v[2] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("bad_burn", [None, "burn", 22.0])
def test_non_finiteburn_argument_raises(bad_burn):
    with pytest.raises(ValueError, match=r"burn must be a FiniteBurn"):
        propagate_with_finite_burn(STATE0, (0.0, 10.0), bad_burn, MU_EARTH_M3_S2)
    with pytest.raises(ValueError, match=r"burn must be a FiniteBurn"):
        finite_burn_loss(STATE0, bad_burn, MU_EARTH_M3_S2)


@pytest.mark.unit
@pytest.mark.parametrize("total_time_s", [0.0, 1.0, -5.0, float("nan")])
def test_comparison_epoch_before_cut_off_raises(total_time_s):
    with pytest.raises(ValueError, match=r"total_time_s must be finite and >= the burn"):
        finite_burn_loss(STATE0, _burn(1.0), MU_EARTH_M3_S2, total_time_s=total_time_s)


@pytest.mark.unit
def test_non_enum_impulse_epoch_raises():
    with pytest.raises(ValueError, match=r"impulse_epoch must be an ImpulseEpoch"):
        finite_burn_loss(STATE0, _burn(1.0), MU_EARTH_M3_S2, impulse_epoch="midpoint")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("masses", "match"),
    [
        ((0.0, 100.0), r"initial_mass_kg must be finite and > 0"),
        ((-200.0, 100.0), r"initial_mass_kg must be finite and > 0"),
        ((200.0, 0.0), r"final_mass_kg must be finite and > 0"),
        ((200.0, -1.0), r"final_mass_kg must be finite and > 0"),
        ((100.0, 200.0), r"exceeds initial_mass_kg"),
    ],
)
def test_equivalent_impulsive_delta_v_rejects_impossible_mass_pairs(masses, match: str):
    with pytest.raises(ValueError, match=match):
        equivalent_impulsive_delta_v(masses[0], masses[1], 220.0)


@pytest.mark.unit
def test_equivalent_impulsive_delta_v_rejects_non_positive_isp():
    with pytest.raises(ValueError, match=r"specific_impulse_s must be finite and > 0"):
        equivalent_impulsive_delta_v(200.0, 199.0, 0.0)


@pytest.mark.unit
def test_integrator_failure_is_surfaced_not_swallowed(monkeypatch):
    """A failed or short integration must raise, never return a truncated trajectory.

    ``solve_ivp`` is monkeypatched because a genuine DOP853 failure on a well-posed powered
    arc is hard to provoke on purpose -- and a raise path that is never exercised is a raise
    path that has never been shown to work.
    """
    import rpo_core.finite_burn as module

    monkeypatch.setattr(
        module,
        "solve_ivp",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, t=np.array([0.0]), y=np.zeros((7, 1)), message="step size underflow"
        ),
    )
    with pytest.raises(PropagationError, match=r"finite-burn propagation failed at t ="):
        propagate_with_finite_burn(STATE0, (0.0, 10.0), _burn(1.0), MU_EARTH_M3_S2)

    monkeypatch.setattr(
        module,
        "solve_ivp",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, t=np.array([0.0]), y=np.zeros((7, 1)), message=""
        ),
    )
    with pytest.raises(PropagationError, match=r"the trajectory is incomplete"):
        propagate_with_finite_burn(STATE0, (0.0, 10.0), _burn(1.0), MU_EARTH_M3_S2)


@pytest.mark.unit
def test_burn_specification_is_immutable():
    """``FiniteBurn`` is frozen: every derived property is a pure function of its fields."""
    burn = _burn(1.0)
    with pytest.raises(AttributeError):
        burn.thrust_n = 2.0  # type: ignore[misc]


@pytest.mark.unit
def test_direction_is_normalised_at_construction():
    """Magnitude in the commanded direction is discarded, so it cannot scale the burn."""
    scaled = FiniteBurn(1.0, ISP_S, MASS_KG, (0.0, 17.0, 0.0), duration_s=10.0)
    assert np.allclose(scaled.direction_unit, [0.0, 1.0, 0.0], atol=0.0, rtol=0.0)
    plain = FiniteBurn(1.0, ISP_S, MASS_KG, (0.0, 1.0, 0.0), duration_s=10.0)
    a = propagate_with_finite_burn(STATE0, (0.0, 10.0), scaled, MU_EARTH_M3_S2)
    b = propagate_with_finite_burn(STATE0, (0.0, 10.0), plain, MU_EARTH_M3_S2)
    assert np.array_equal(a, b)
