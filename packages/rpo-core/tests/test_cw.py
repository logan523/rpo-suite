"""Clohessy-Wiltshire state transition matrix and two-impulse targeting.

The assertions here are limiting cases and closed-form solutions, not golden numbers
copied from a previous run of this same code. A regression suite that only compares
against its own past output cannot detect a sign error that was present from the start.
"""

import math

import numpy as np
import pytest
from rpo_core.constants import R_EARTH_EQUATORIAL_M, mean_motion_rad_s, orbital_period_s
from rpo_core.exceptions import InfeasibleTransferError, SingularTransferTimeError
from rpo_core.relative.cw import (
    cw_dynamics_matrix,
    cw_stm,
    propagate_cw,
    two_impulse_transfer,
)
from scipy.integrate import solve_ivp

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
N = mean_motion_rad_s(A_ISS_M)
PERIOD_S = orbital_period_s(A_ISS_M)


# --------------------------------------------------------------------------------------
# State transition matrix
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_stm_at_zero_time_is_identity():
    np.testing.assert_array_equal(cw_stm(N, 0.0), np.eye(6))


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.3])
def test_forward_then_backward_propagation_recovers_initial_state(fraction):
    """Phi(-t) @ Phi(t) == I, checked on a physically scaled state.

    Checked via a state round trip rather than a matrix norm because the STM mixes units
    (the position-from-velocity block is in seconds, the velocity-from-position block in
    1/s), so ``norm(Phi @ Phi_inv - I)`` has no meaningful scale.
    """
    dt = fraction * PERIOD_S
    state0 = np.array([120.0, -1000.0, 45.0, 0.05, -0.2, 0.01])
    recovered = propagate_cw(N, propagate_cw(N, state0, dt), -dt)
    np.testing.assert_allclose(recovered[:3], state0[:3], atol=1e-9)
    np.testing.assert_allclose(recovered[3:], state0[3:], atol=1e-12)


@pytest.mark.unit
def test_stm_composes_over_split_intervals():
    """Phi(t1 + t2) == Phi(t2) @ Phi(t1): the STM must be a proper flow map."""
    t1, t2 = 0.31 * PERIOD_S, 0.47 * PERIOD_S
    state0 = np.array([250.0, -800.0, -30.0, 0.1, 0.02, -0.03])
    direct = propagate_cw(N, state0, t1 + t2)
    stepwise = propagate_cw(N, propagate_cw(N, state0, t1), t2)
    np.testing.assert_allclose(direct[:3], stepwise[:3], atol=1e-9)
    np.testing.assert_allclose(direct[3:], stepwise[3:], atol=1e-12)


@pytest.mark.unit
def test_closed_form_stm_agrees_with_numerical_integration():
    """Independent check: integrate xdot = A x and compare against the closed form.

    Two separate derivations of the same dynamics. Agreement to sub-micrometre over half
    an orbit means neither the matrix entries nor the plant matrix has a transposed or
    sign-flipped term.
    """
    a_matrix = cw_dynamics_matrix(N)
    state0 = np.array([200.0, -1500.0, 80.0, 0.1, -0.3, 0.02])
    tof = 0.5 * PERIOD_S

    solution = solve_ivp(
        lambda _t, y: a_matrix @ y,
        (0.0, tof),
        state0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )
    assert solution.success, f"reference integration failed: {solution.message}"

    analytic = propagate_cw(N, state0, tof)
    np.testing.assert_allclose(analytic[:3], solution.y[:3, -1], atol=1e-6)
    np.testing.assert_allclose(analytic[3:], solution.y[3:, -1], atol=1e-9)


@pytest.mark.unit
def test_cross_track_is_simple_harmonic_at_the_orbital_frequency():
    z0, zdot0 = 150.0, 0.4
    times = np.linspace(0.0, 2.0 * PERIOD_S, 401)
    for t in times:
        state = propagate_cw(N, np.array([0.0, 0.0, z0, 0.0, 0.0, zdot0]), t)
        expected_z = z0 * math.cos(N * t) + (zdot0 / N) * math.sin(N * t)
        expected_zdot = -z0 * N * math.sin(N * t) + zdot0 * math.cos(N * t)
        assert state[2] == pytest.approx(expected_z, abs=1e-9)
        assert state[5] == pytest.approx(expected_zdot, abs=1e-12)


@pytest.mark.unit
def test_cross_track_does_not_couple_into_in_plane_motion():
    """Pure cross-track initial conditions must never produce in-plane motion."""
    state = propagate_cw(N, np.array([0.0, 0.0, 500.0, 0.0, 0.0, 0.7]), 0.37 * PERIOD_S)
    np.testing.assert_allclose(state[[0, 1, 3, 4]], np.zeros(4), atol=1e-12)


# --------------------------------------------------------------------------------------
# Known analytic behaviours -- the tests that actually catch sign errors
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_drift_free_condition_produces_a_closed_relative_orbit():
    """With ydot0 = -2*n*x0 the secular along-track drift cancels and the state repeats.

    This is the sharpest single check on a Hill-frame implementation. The cancellation is
    between a -12*pi*x0 term and a +12*pi*x0 term; any sign error anywhere in the in-plane
    block breaks it by hundreds of metres, not by rounding.
    """
    x0 = 100.0
    state0 = np.array([x0, -500.0, 0.0, 0.0, -2.0 * N * x0, 0.0])
    after_one_period = propagate_cw(N, state0, PERIOD_S)
    np.testing.assert_allclose(after_one_period[:3], state0[:3], atol=1e-9)
    np.testing.assert_allclose(after_one_period[3:], state0[3:], atol=1e-12)


@pytest.mark.unit
def test_violating_the_drift_free_condition_produces_secular_drift():
    """The complement of the previous test: confirm it is a knife edge, not a plateau.

    A test that only shows the drift-free case closing would also pass if the along-track
    secular term had been dropped entirely.
    """
    x0 = 100.0
    state0 = np.array([x0, -500.0, 0.0, 0.0, 0.0, 0.0])  # ydot0 = 0, not -2*n*x0
    drift_m = propagate_cw(N, state0, PERIOD_S)[1] - state0[1]
    # Closed form: along-track secular drift is -3*t*(2*n*x0 + ydot0). Over exactly one
    # period with ydot0 = 0 that collapses to -12*pi*x0, independent of n.
    assert drift_m == pytest.approx(-3.0 * PERIOD_S * (2.0 * N * x0), rel=1e-9)
    assert drift_m == pytest.approx(-12.0 * math.pi * x0, rel=1e-9)
    assert abs(drift_m) > 1.0e3  # kilometre-scale: unmistakably not a rounding artefact


@pytest.mark.unit
def test_radial_impulse_traces_a_closed_two_to_one_ellipse():
    """From rest at the origin, a radial impulse gives a 2:1 ellipse closing after one period.

    x(t) = (dv/n) sin(nt), y(t) = -(2 dv/n)(1 - cos(nt)): along-track extent is exactly
    twice the radial extent, and the chaser returns to the origin at t = T.
    """
    dv = 0.1
    times = np.linspace(0.0, PERIOD_S, 721)
    track = np.array([propagate_cw(N, np.array([0.0, 0, 0, dv, 0, 0]), t) for t in times])

    radial_half_extent = 0.5 * (track[:, 0].max() - track[:, 0].min())
    along_track_half_extent = 0.5 * (track[:, 1].max() - track[:, 1].min())
    assert radial_half_extent == pytest.approx(dv / N, rel=1e-4)
    assert along_track_half_extent == pytest.approx(2.0 * dv / N, rel=1e-4)
    assert along_track_half_extent / radial_half_extent == pytest.approx(2.0, rel=1e-4)

    np.testing.assert_allclose(track[-1][:3], np.zeros(3), atol=1e-9)


@pytest.mark.unit
def test_along_track_impulse_causes_secular_drift_not_a_closed_orbit():
    """The counterpart to the radial impulse: along-track burns do not close."""
    dv = 0.1
    final = propagate_cw(N, np.array([0.0, 0, 0, 0, dv, 0]), PERIOD_S)
    assert final[1] == pytest.approx(-3.0 * dv * PERIOD_S, rel=1e-9)


@pytest.mark.unit
def test_radially_inward_burn_moves_the_chaser_forward():
    """Physical direction check: burn toward Earth, drop altitude, gain along-track rate.

    If this comes out backwards, the sign convention on the radial axis is inverted, and
    every approach trajectory in the suite would fly the wrong way.
    """
    inward = propagate_cw(N, np.array([0.0, 0, 0, -0.05, 0, 0]), 0.25 * PERIOD_S)
    assert inward[1] > 0.0


# --------------------------------------------------------------------------------------
# Two-impulse targeting
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.35, 0.75, 1.4])
def test_two_impulse_solution_reaches_the_commanded_terminal_state(fraction):
    """Round trip: solve for the impulses, fly them, confirm the terminal state.

    Half-period multiples are deliberately excluded from this *out-of-plane* case: the
    cross-track subproblem is exactly rank-deficient there, so a z-changing request is
    infeasible rather than merely hard. Those times of flight are covered by the dedicated
    planar and infeasibility tests below.
    """
    r0 = np.array([50.0, -1000.0, 20.0])
    v0 = np.array([0.01, -0.02, 0.005])
    rf = np.array([0.0, -250.0, 0.0])
    vf = np.zeros(3)
    tof = fraction * PERIOD_S

    dv1, dv2 = two_impulse_transfer(N, r0, v0, rf, vf, tof)

    arrival = propagate_cw(N, np.concatenate((r0, v0 + dv1)), tof)
    np.testing.assert_allclose(arrival[:3], rf, atol=1e-9)
    np.testing.assert_allclose(arrival[3:] + dv2, vf, atol=1e-12)


@pytest.mark.unit
def test_half_period_v_bar_hop_matches_the_closed_form_solution():
    """The baseline manoeuvre, against an analytic result rather than a stored number.

    For a coplanar V-bar hop of length dy over exactly half a period starting and ending
    at rest, the CW solution collapses to two equal, purely radial impulses of magnitude
    n*dy/4 -- so the total delta-v is n*dy/2, independent of everything else.
    """
    dy = 750.0  # -1000 m -> -250 m
    tof = 0.5 * PERIOD_S
    dv1, dv2 = two_impulse_transfer(
        N, [0.0, -1000.0, 0.0], np.zeros(3), [0.0, -250.0, 0.0], np.zeros(3), tof
    )

    expected_magnitude = N * dy / 4.0
    np.testing.assert_allclose(dv1, [-expected_magnitude, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dv2, [-expected_magnitude, 0.0, 0.0], atol=1e-12)

    total_dv = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
    assert total_dv == pytest.approx(N * dy / 2.0, rel=1e-12)
    # Sanity on magnitude: a sub-kilometre hop over ~46 min costs tens of cm/s, not m/s.
    assert 0.1 < total_dv < 1.0


@pytest.mark.unit
def test_transfer_at_a_full_orbital_period_raises_singular():
    """In-plane Phi_rv loses rank at integer multiples of the period."""
    with pytest.raises(SingularTransferTimeError, match="in-plane"):
        two_impulse_transfer(
            N, [0.0, -1000.0, 0.0], np.zeros(3), [0.0, -250.0, 0.0], np.zeros(3), PERIOD_S
        )


@pytest.mark.unit
def test_singular_error_reports_the_condition_number_and_period_count():
    with pytest.raises(SingularTransferTimeError) as excinfo:
        two_impulse_transfer(
            N,
            [10.0, -1000.0, 0.0],
            np.zeros(3),
            [0.0, -250.0, 0.0],
            np.zeros(3),
            2.0 * PERIOD_S,
        )
    message = str(excinfo.value)
    assert "condition number" in message
    assert "2.000000 orbital periods" in message


@pytest.mark.unit
def test_cross_track_change_at_half_period_is_infeasible():
    """Cross-track position cannot be changed at integer multiples of the half period."""
    with pytest.raises(InfeasibleTransferError, match="cross-track"):
        two_impulse_transfer(
            N,
            [0.0, -1000.0, 0.0],
            np.zeros(3),
            [0.0, -250.0, 300.0],
            np.zeros(3),
            0.5 * PERIOD_S,
        )


@pytest.mark.unit
def test_planar_transfer_at_half_period_is_accepted():
    """Regression guard against over-eager conditioning checks.

    A single 3x3 conditioning test on Phi_rv would reject this transfer, because the
    cross-track block is exactly singular at half a period. The in-plane problem is
    perfectly well conditioned, and this is the suite's baseline manoeuvre.
    """
    dv1, dv2 = two_impulse_transfer(
        N,
        [0.0, -1000.0, 0.0],
        np.zeros(3),
        [0.0, -250.0, 0.0],
        np.zeros(3),
        0.5 * PERIOD_S,
    )
    assert np.all(np.isfinite(dv1)) and np.all(np.isfinite(dv2))


@pytest.mark.unit
def test_cross_track_preserved_through_half_period_needs_no_cross_track_impulse():
    """z(T/2) is pinned to -z0; requesting exactly that is feasible and costs no dv_z."""
    z0 = 300.0
    dv1, _ = two_impulse_transfer(
        N,
        [0.0, -1000.0, z0],
        np.zeros(3),
        [0.0, -250.0, -z0],
        np.zeros(3),
        0.5 * PERIOD_S,
    )
    assert dv1[2] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_zero_length_transfer_costs_nothing():
    """Holding station: same state in, same state out, no impulse."""
    r0, v0 = np.array([0.0, -250.0, 0.0]), np.array([0.0, 0.0, 0.0])
    state = propagate_cw(N, np.concatenate((r0, v0)), 0.3 * PERIOD_S)
    dv1, dv2 = two_impulse_transfer(N, r0, v0, state[:3], state[3:], 0.3 * PERIOD_S)
    np.testing.assert_allclose(dv1, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(dv2, np.zeros(3), atol=1e-14)


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("bad_n", [0.0, -1e-3, float("nan"), float("inf")])
def test_invalid_mean_motion_raises(bad_n):
    with pytest.raises(ValueError, match="n_rad_s"):
        cw_stm(bad_n, 100.0)


@pytest.mark.unit
@pytest.mark.parametrize("bad_tof", [0.0, -100.0, float("nan")])
def test_invalid_time_of_flight_raises(bad_tof):
    with pytest.raises(ValueError, match="tof_s"):
        two_impulse_transfer(N, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), bad_tof)


@pytest.mark.unit
def test_wrong_vector_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        two_impulse_transfer(N, np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3), 100.0)


@pytest.mark.unit
def test_non_finite_state_raises():
    with pytest.raises(ValueError, match="finite"):
        propagate_cw(N, np.array([np.nan, 0, 0, 0, 0, 0]), 100.0)
