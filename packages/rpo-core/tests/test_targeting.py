"""Differential correction verified against the dynamics it claims to correct onto.

The load-bearing test in this file is
:func:`test_correction_beats_raw_cw_across_separations`. Everything else supports it. The
question this module exists to answer is *does correcting actually beat CW, and by how
much*, and the only honest way to answer it is to fly both impulses through the nonlinear
oracle and compare the terminal misses. That comparison is the entire justification for the
module; if it failed, the right response would be to delete the module, not to loosen the
bound.

Two properties make the checks here genuine rather than self-referential:

* The oracle, :func:`rpo_core.relative.nonlinear.propagate_relative_nonlinear`, shares no
  code with the Newton loop. Tests re-propagate the *returned* impulse independently rather
  than trusting the residual the solver reported about itself -- a solver that returned a
  good-looking residual alongside a delta-v that does not produce it would pass the first
  check and fail the second.
* The finite-difference Jacobian is checked against the closed-form CW ``Phi_rv``, which is
  an analytic matrix from a different module. That is what pins the finite-difference step
  to something other than taste.

Numbers quoted in tolerance comments were measured on this machine before the bound was
chosen; the headroom is stated in each case.
"""

import itertools
import math

import numpy as np
import pytest
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from rpo_core.exceptions import (
    InfeasibleTransferError,
    PropagationError,
    SingularTransferTimeError,
)
from rpo_core.relative.cw import cw_stm
from rpo_core.relative.nonlinear import propagate_relative_nonlinear
from rpo_core.targeting import (
    DEFAULT_FD_STEP_M_S,
    IllConditionedJacobianError,
    TargetingConvergenceError,
    correct_two_impulse_transfer,
    raw_cw_terminal_miss_m,
)

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
V_CIRCULAR = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
N = mean_motion_rad_s(A_ISS_M)
PERIOD_S = orbital_period_s(A_ISS_M)

# Exactly circular target, inclined 51.6 deg -- the same reference the CW validity study
# uses, so linearisation error is isolated from eccentricity error.
_INC = math.radians(51.6)
R_TARGET = np.array([A_ISS_M, 0.0, 0.0])
V_TARGET = V_CIRCULAR * np.array([0.0, math.cos(_INC), math.sin(_INC)])
AT_REST = np.zeros(3)

#: A transfer time away from both singular families: not a whole period (in-plane rank
#: loss) and not a half period (cross-track rank loss).
GENERIC_TOF_S = 0.4 * PERIOD_S


def _hop(separation_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r0, rf)`` for a V-bar hop from ``-rho`` to ``-rho/4``, both on V-bar."""
    return (
        np.array([0.0, -separation_m, 0.0]),
        np.array([0.0, -0.25 * separation_m, 0.0]),
    )


def _correct(r0, rf, tof_s=GENERIC_TOF_S, **kwargs):
    """Run the corrector on a hop, with the standard circular target."""
    return correct_two_impulse_transfer(
        R_TARGET, V_TARGET, r0, AT_REST, rf, AT_REST, tof_s, **kwargs
    )


def _fly(r0: np.ndarray, dv1: np.ndarray, tof_s: float) -> np.ndarray:
    """Independently propagate a departure impulse through the nonlinear oracle.

    Deliberately does not go through :mod:`rpo_core.targeting`: this is how a test checks
    the *returned* impulse rather than the solver's own account of it.
    """
    initial = np.concatenate((r0, AT_REST + dv1))
    trajectory = propagate_relative_nonlinear(
        R_TARGET, V_TARGET, initial, np.array([0.0, tof_s]), MU_EARTH_M3_S2
    )
    return np.asarray(trajectory[-1])


# ---------------------------------------------------------------------------------------
# The headline: does correction measurably beat raw CW?
# ---------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_correction_beats_raw_cw_across_separations():
    """The number that justifies this module: terminal miss, raw CW versus corrected.

    Half-period V-bar hop, the suite's baseline manoeuvre. Measured on this machine:

    ==========  ===============  ==============  ============
    Separation  Raw CW miss (m)  Corrected (m)   Improvement
    ==========  ===============  ==============  ============
    100 m       1.225271e-02     1.41e-08        8.7e+05x
    1 km        1.225271e+00     5.52e-07        2.2e+06x
    10 km       1.225388e+02     7.48e-04        1.6e+05x
    ==========  ===============  ==============  ============

    The raw miss scales as ``rho**2`` to four figures (1.2253e-2, 1.2253e0, 1.2254e2 --
    exactly 100x per decade of separation), which is the ``6*pi*rho**2/r`` law of
    ``docs/cw_validity.md`` showing up in a targeting problem rather than a propagation one.

    Bounds below are set from those measurements, and the corrected miss is checked by
    **independent re-propagation** of the returned impulse, not by reading back the
    residual the solver reported about itself.
    """
    tolerance_m = 1.0e-3
    measured = []
    for separation_m in (100.0, 1_000.0, 10_000.0):
        r0, rf = _hop(separation_m)
        raw_miss_m = raw_cw_terminal_miss_m(
            R_TARGET, V_TARGET, r0, AT_REST, rf, AT_REST, 0.5 * PERIOD_S
        )
        result = _correct(r0, rf, tof_s=0.5 * PERIOD_S, tolerance_m=tolerance_m)

        # Independent check: re-fly the returned dv1 and measure where it actually arrives.
        arrival = _fly(r0, result.dv1_hill_m_s, 0.5 * PERIOD_S)
        corrected_miss_m = float(np.linalg.norm(arrival[:3] - rf))
        measured.append((separation_m, raw_miss_m, corrected_miss_m))

        assert corrected_miss_m <= tolerance_m, (
            f"at {separation_m:g} m the corrected impulse missed by {corrected_miss_m:.3e} m, "
            f"above the {tolerance_m:g} m tolerance it claimed to meet"
        )
        # The solver's self-reported residual must agree with the independent flight.
        # Measured agreement: exact to the last bit, since both call the same propagator
        # on the same state. Bound at 1e-12 m to allow for accumulation elsewhere.
        assert abs(corrected_miss_m - result.final_residual_m) < 1e-12

    print("\nseparation |    raw CW miss |  corrected miss |   improvement")
    for separation_m, raw_miss_m, corrected_miss_m in measured:
        print(
            f"{separation_m:9.0f}m | {raw_miss_m:14.6e} | {corrected_miss_m:15.3e} | "
            f"{raw_miss_m / corrected_miss_m:12.2e}x"
        )

    # The complement, and the whole point: at the larger separations the *uncorrected*
    # solution must be above tolerance. Without this the test would still pass if
    # correction did nothing, because CW is already good enough at 100 m.
    _, raw_1km, _ = measured[1]
    _, raw_10km, _ = measured[2]
    assert raw_1km > tolerance_m, (
        f"raw CW at 1 km missed by only {raw_1km:.3e} m, at or below the {tolerance_m:g} m "
        "tolerance -- there would be nothing for this module to correct"
    )
    assert raw_10km > tolerance_m
    # Measured 1.2253 m and 122.54 m; bound with ~2x headroom so the assertion states the
    # order of magnitude rather than pinning a golden number.
    assert raw_1km > 0.5, f"raw CW miss at 1 km was {raw_1km:.3e} m, expected ~1.23 m"
    assert raw_10km > 50.0, f"raw CW miss at 10 km was {raw_10km:.3e} m, expected ~122.5 m"


@pytest.mark.integration
def test_correction_beats_raw_cw_at_one_kilometre():
    """A fast single-separation version of the headline, so the fast job still checks it."""
    r0, rf = _hop(1_000.0)
    raw_miss_m = raw_cw_terminal_miss_m(R_TARGET, V_TARGET, r0, AT_REST, rf, AT_REST, GENERIC_TOF_S)
    result = _correct(r0, rf, tolerance_m=1.0e-3)
    arrival = _fly(r0, result.dv1_hill_m_s, GENERIC_TOF_S)
    corrected_miss_m = float(np.linalg.norm(arrival[:3] - rf))
    # Measured: raw 7.337731e-01 m, corrected 8.02e-08 m.
    assert raw_miss_m > 0.3
    assert corrected_miss_m < 1.0e-3
    assert corrected_miss_m < raw_miss_m / 1_000.0


@pytest.mark.integration
def test_raw_cw_baseline_matches_the_solvers_initial_residual():
    """The baseline helper and the solver's first history entry must be the same number.

    If they disagreed, the improvement ratio reported by the headline test would be
    comparing two different problems.
    """
    r0, rf = _hop(1_000.0)
    raw_miss_m = raw_cw_terminal_miss_m(R_TARGET, V_TARGET, r0, AT_REST, rf, AT_REST, GENERIC_TOF_S)
    result = _correct(r0, rf)
    # Both propagate the same CW impulse through the same oracle: measured identical to
    # the last bit. Bound at 1e-12 m.
    assert abs(raw_miss_m - result.initial_residual_m) < 1e-12


# ---------------------------------------------------------------------------------------
# Limiting case: at 1 m separation CW is already right, so correction must be a near no-op
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_correction_is_almost_a_no_op_at_one_metre_separation():
    """CW is essentially exact at 1 m, so the corrector must barely move the impulse.

    This is the limiting case that proves the module corrects a real error rather than
    injecting one: a sign error in the Jacobian, or a residual defined against the wrong
    target, would still converge at larger separations but would have to *move* the impulse
    here, where there is nothing to fix.

    The tolerance is deliberately tightened to 1e-7 m so the solver actually iterates. At
    the 1e-3 m default the CW guess already misses by only 8.41e-07 m and the loop exits
    immediately, which would make a "the correction is tiny" assertion true by construction
    rather than by measurement.
    """
    r0, rf = _hop(1.0)
    result = _correct(r0, rf, tolerance_m=1.0e-7)

    assert result.iterations >= 1, "tolerance was not tight enough to force an iteration"
    cw_magnitude = float(np.linalg.norm(result.cw_dv1_hill_m_s))
    relative = result.dv1_correction_m_s / cw_magnitude
    print(
        f"\n1 m separation: |dv1_cw| = {cw_magnitude:.6e} m/s, "
        f"correction = {result.dv1_correction_m_s:.6e} m/s ({relative:.3e} relative)"
    )
    # Measured: |dv1_cw| = 3.084085e-04 m/s, correction = 1.357027e-10 m/s, relative
    # 4.400e-07. Bound at 1e-5 relative gives ~23x headroom.
    assert relative < 1.0e-5, (
        f"correction at 1 m separation is {relative:.3e} of the CW impulse, "
        "which is not the near-no-op CW's accuracy there demands"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_correction_magnitude_grows_as_separation_squared():
    """Complement to the no-op test: prove the correction is real, not uniformly tiny.

    A corrector that always returned the CW impulse unchanged would pass the 1 m no-op
    assertion perfectly. It fails here. The correction must not merely be nonzero but must
    grow with separation the way the error it removes does -- quadratically, mirroring
    ``6*pi*rho**2/r``.
    """
    corrections = []
    for separation_m in (100.0, 1_000.0, 10_000.0):
        r0, rf = _hop(separation_m)
        result = _correct(r0, rf, tolerance_m=1.0e-6)
        corrections.append(result.dv1_correction_m_s)
    print(f"\ncorrection magnitudes (m/s) at 100 m / 1 km / 10 km: {corrections}")

    # Measured: 1.152167e-06, 1.152207e-04, 1.152249e-02 m/s -- exactly 100x per decade.
    assert all(c > 0.0 for c in corrections), "correction was identically zero"
    for smaller, larger in itertools.pairwise(corrections):
        ratio = larger / smaller
        # Bracket 100x with generous headroom; the claim is the quadratic scaling law
        # holds, not that it hits an exact power.
        assert 50.0 < ratio < 200.0, f"correction scaled {ratio:.1f}x per decade, expected ~100x"

    # And the knife edge against the no-op test: the 10 km correction is seven orders
    # larger than the 1 m one, so "tiny at 1 m" is a measurement, not a plateau.
    r0, rf = _hop(1.0)
    tiny = _correct(r0, rf, tolerance_m=1.0e-7).dv1_correction_m_s
    assert corrections[-1] > 1.0e5 * tiny


# ---------------------------------------------------------------------------------------
# The arrival impulse must be recomputed against nonlinear arrival, not carried over
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_arrival_impulse_nulls_the_achieved_terminal_velocity():
    """``dv2`` must zero the velocity error against the state the coast actually delivers."""
    r0, rf = _hop(10_000.0)
    result = _correct(r0, rf, tolerance_m=1.0e-3)
    residual_velocity = result.arrival_state_hill[3:] + result.dv2_hill_m_s - AT_REST
    # Measured exactly 0.0 m/s; bound at 1e-12 m/s for float-arithmetic headroom.
    assert float(np.linalg.norm(residual_velocity)) < 1.0e-12
    assert float(np.linalg.norm(result.terminal_state_hill[3:] - AT_REST)) < 1.0e-12


@pytest.mark.integration
def test_carrying_over_the_cw_arrival_impulse_would_leave_residual_drift():
    """Complement: prove recomputing ``dv2`` is load-bearing, not decoration.

    If the CW arrival impulse were reused, the chaser would be left drifting. Measured
    leftover velocity at 10 km separation: 4.205435e-03 m/s out of a 3.088 m/s burn.
    """
    r0, rf = _hop(10_000.0)
    result = _correct(r0, rf, tolerance_m=1.0e-3)
    leftover = float(np.linalg.norm(result.arrival_state_hill[3:] + result.cw_dv2_hill_m_s))
    print(f"\nleftover drift if the CW dv2 were reused: {leftover:.6e} m/s")
    # Measured 4.205435e-03 m/s. Bound at 1e-3 m/s gives 4.2x headroom, and is four
    # orders above the 1e-12 m/s the recomputed dv2 achieves.
    assert leftover > 1.0e-3, (
        f"the CW arrival impulse left only {leftover:.3e} m/s of drift, so this test could "
        "not tell a recomputed dv2 from a carried-over one"
    )
    assert float(np.linalg.norm(result.dv2_hill_m_s - result.cw_dv2_hill_m_s)) > 1.0e-3


# ---------------------------------------------------------------------------------------
# Convergence behaviour
# ---------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_residual_history_is_monotone_and_short():
    """Damped Newton must reduce the residual every accepted step, and do it quickly.

    Monotonicity is a *guarantee* with ``damping=True``, not an observation: a step is only
    accepted if it strictly reduces the residual norm. The iteration-count bound is the
    check that keeps the finite-difference step honest -- the answer is right for a wide
    range of steps because the residual is evaluated exactly, but a badly chosen step shows
    up immediately as extra iterations. Measured at 10 km over 0.4 periods: 2 iterations at
    the default 1e-4 m/s step, 4 at 1e+1, 9 at 1e-10, and 14 at 1e+3.
    """
    for separation_m in (100.0, 1_000.0, 10_000.0, 100_000.0):
        r0, rf = _hop(separation_m)
        result = _correct(r0, rf, tof_s=0.5 * PERIOD_S, tolerance_m=1.0e-3)
        history = result.residual_history_m
        assert all(later < earlier for earlier, later in itertools.pairwise(history)), (
            f"residual history at {separation_m:g} m was not monotone: {history}"
        )
        assert result.iterations == len(history) - 1
        # Measured: 1, 1, 1, 2 iterations respectively. 3 leaves headroom while still
        # failing for a finite-difference step wrong by three or more orders.
        assert result.iterations <= 3, (
            f"took {result.iterations} iterations at {separation_m:g} m; the Newton step is "
            "converging far more slowly than a well-scaled Jacobian should allow"
        )


@pytest.mark.integration
def test_reported_iteration_count_matches_the_history():
    r0, rf = _hop(10_000.0)
    result = _correct(r0, rf, tolerance_m=1.0e-6)
    assert result.iterations == len(result.residual_history_m) - 1
    assert result.final_residual_m == result.residual_history_m[-1]
    assert result.initial_residual_m == result.residual_history_m[0]


@pytest.mark.integration
def test_solver_is_deterministic():
    """Same inputs, bitwise identical output. No RNG, no iteration-order dependence."""
    r0, rf = _hop(1_000.0)
    first = _correct(r0, rf, tolerance_m=1.0e-6)
    second = _correct(r0, rf, tolerance_m=1.0e-6)
    assert first.dv1_hill_m_s.tobytes() == second.dv1_hill_m_s.tobytes()
    assert first.dv2_hill_m_s.tobytes() == second.dv2_hill_m_s.tobytes()
    assert first.arrival_state_hill.tobytes() == second.arrival_state_hill.tobytes()
    assert first.terminal_state_hill.tobytes() == second.terminal_state_hill.tobytes()
    assert first.residual_history_m == second.residual_history_m
    assert first.iterations == second.iterations


@pytest.mark.slow
@pytest.mark.integration
def test_damping_rescues_a_separation_where_plain_newton_diverges():
    """Measured justification for the line search, and for defaulting it on.

    At 5000 km separation -- comparable to the orbit radius itself, far outside anything
    this module is for, but the regime that decides whether damping is needed at all --
    undamped Newton wanders (residual reaching 2.6e+09 m) and never converges in 40
    iterations. The damped iteration converges monotonically in 8.

    Inside the real envelope damping costs one extra propagation per iteration and changes
    nothing else: full step length is accepted on the first trial at every separation up to
    2000 km, so the damped and undamped residual histories are identical there.
    """
    r0, rf = _hop(5.0e6)
    damped = _correct(r0, rf, tof_s=0.5 * PERIOD_S, tolerance_m=1.0e-3, max_iterations=40)
    history = damped.residual_history_m
    assert all(later < earlier for earlier, later in itertools.pairwise(history))
    print(f"\ndamped at 5000 km: {damped.iterations} iterations, history {history}")

    with pytest.raises(TargetingConvergenceError, match="did not converge"):
        _correct(
            r0,
            rf,
            tof_s=0.5 * PERIOD_S,
            tolerance_m=1.0e-3,
            max_iterations=40,
            damping=False,
        )


@pytest.mark.slow
@pytest.mark.integration
def test_damping_is_inert_inside_the_operating_envelope():
    """Complement to the rescue test: damping must not perturb the ordinary case.

    If the line search were rejecting good full steps, every result in this module would
    be slower and the histories would differ. Measured: identical to the last bit at 1 km.
    """
    r0, rf = _hop(1_000.0)
    damped = _correct(r0, rf, tolerance_m=1.0e-6)
    undamped = _correct(r0, rf, tolerance_m=1.0e-6, damping=False)
    assert damped.dv1_hill_m_s.tobytes() == undamped.dv1_hill_m_s.tobytes()
    assert damped.iterations == undamped.iterations


# ---------------------------------------------------------------------------------------
# The finite-difference step, checked against an analytic matrix from another module
# ---------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_default_finite_difference_step_recovers_the_analytic_cw_jacobian():
    """The shooting Jacobian must reproduce the closed-form CW ``Phi_rv``, and a bad step must not.

    ``Phi_rv`` is the analytic sensitivity of terminal position to departure velocity under
    the linear model. The nonlinear sensitivity differs from it by the linearisation error
    and nothing else, so at small separation the finite-difference Jacobian must land on it.
    This is what fixes the step to a measurement rather than a preference: an independent
    matrix, from a different module, with a closed form.

    Measured relative error at ``DEFAULT_FD_STEP_M_S``: 7.21e-05 at a quarter period,
    1.09e-04 at 0.4 periods. Those residuals are the physics, not the arithmetic -- they
    scale with separation as a linearisation error should.
    """
    r0, rf = _hop(1_000.0)
    tof_s = GENERIC_TOF_S
    dv1 = _correct(r0, rf, tof_s=tof_s).dv1_hill_m_s

    def jacobian_at(step_m_s: float) -> np.ndarray:
        """Forward-difference the terminal position w.r.t. the departure impulse."""
        base = _fly(r0, dv1, tof_s)
        columns = np.empty((3, 3))
        for axis in range(3):
            perturbed = dv1.copy()
            perturbed[axis] += step_m_s
            columns[:, axis] = (_fly(r0, perturbed, tof_s)[:3] - base[:3]) / step_m_s
        return columns

    phi_rv = cw_stm(N, tof_s)[:3, 3:]
    good = np.linalg.norm(jacobian_at(DEFAULT_FD_STEP_M_S) - phi_rv) / np.linalg.norm(phi_rv)
    print(f"\nJacobian relative error at h={DEFAULT_FD_STEP_M_S:g}: {good:.4e}")
    # Measured 1.0861e-04; bound at 1e-3 gives ~9x headroom.
    assert good < 1.0e-3, f"default step recovered Phi_rv to only {good:.3e} relative"

    # Complement: a step chosen orders of magnitude too small drowns in integrator noise.
    # Measured at h=1e-12: relative error 1.6e+00 -- the Jacobian is destroyed. Without
    # this the test above would pass for almost any step and would not justify the choice.
    drowned = np.linalg.norm(jacobian_at(1.0e-12) - phi_rv) / np.linalg.norm(phi_rv)
    print(f"Jacobian relative error at h=1e-12: {drowned:.4e}")
    assert drowned > 100.0 * good, (
        f"a 1e-12 m/s step gave relative error {drowned:.3e} against {good:.3e} for the "
        "default -- the noise floor is not where the step choice assumed it is"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_a_finite_difference_step_far_below_the_noise_floor_is_rejected():
    """A step small enough to destroy the Jacobian must raise, not quietly limp along."""
    r0, rf = _hop(10_000.0)
    with pytest.raises(IllConditionedJacobianError, match="singular or ill-conditioned"):
        _correct(r0, rf, tolerance_m=1.0e-6, fd_step_m_s=1.0e-12)


# ---------------------------------------------------------------------------------------
# Raise paths
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_unreachable_terminal_state_raises_rather_than_returning_best_effort():
    """A cross-track change at a half-period transfer time is genuinely impossible.

    ``z(t_f)`` is pinned to ``cos(tau) * z_0`` no matter what impulse is applied, so this
    is not an accuracy problem the solver could grind away at. It must refuse.
    """
    r0, _ = _hop(1_000.0)
    unreachable_rf = np.array([0.0, -250.0, 500.0])
    with pytest.raises(InfeasibleTransferError, match="cross-track targeting is rank-deficient"):
        _correct(r0, unreachable_rf, tof_s=0.5 * PERIOD_S)


@pytest.mark.integration
def test_unreachable_cross_track_is_caught_by_the_shooting_rank_guard():
    """The same impossibility, reached past the CW seed via an explicit initial guess.

    This exercises the module's *own* rank guard rather than the one it inherits from
    :func:`~rpo_core.relative.cw.two_impulse_transfer`. It is the guard that matters: at
    this transfer time the full 3x3 Jacobian has condition 1.0e6 -- below the 1e8
    singularity limit -- and the step it produces asks for a 52 km/s cross-track impulse.
    A single 3x3 conditioning test would accept that.
    """
    r0, _ = _hop(1_000.0)
    unreachable_rf = np.array([0.0, -250.0, 500.0])
    with pytest.raises(InfeasibleTransferError, match="cross-track shooting is rank-deficient"):
        _correct(
            r0,
            unreachable_rf,
            tof_s=0.5 * PERIOD_S,
            dv1_guess_m_s=[-0.0211, 0.0, 0.5],
        )


@pytest.mark.integration
def test_tolerance_below_the_noise_floor_raises_instead_of_claiming_success():
    """Asking for 1e-12 m must fail loudly. The oracle's floor is ~5e-09 m."""
    r0, rf = _hop(1_000.0)
    with pytest.raises(TargetingConvergenceError, match="stalled"):
        _correct(r0, rf, tolerance_m=1.0e-12)


@pytest.mark.integration
def test_iteration_cap_is_honoured_and_raises():
    """A cap too small to converge must raise, not return the last iterate."""
    r0, rf = _hop(500_000.0)
    with pytest.raises(TargetingConvergenceError, match="did not converge"):
        _correct(r0, rf, tof_s=0.5 * PERIOD_S, tolerance_m=1.0e-3, max_iterations=1)


@pytest.mark.integration
def test_convergence_error_carries_the_diagnosis():
    """The exception must carry enough to tell a stall from a divergence without a debugger."""
    r0, rf = _hop(500_000.0)
    with pytest.raises(TargetingConvergenceError) as excinfo:
        _correct(r0, rf, tof_s=0.5 * PERIOD_S, tolerance_m=1.0e-3, max_iterations=1)
    error = excinfo.value
    assert error.iterations == 1
    assert error.residual_m > 1.0e-3
    assert len(error.residual_history_m) == 2
    assert error.residual_history_m[-1] == error.residual_m
    # The history must show progress was being made -- that is what distinguishes "needs
    # more iterations" from "diverging".
    assert error.residual_history_m[1] < error.residual_history_m[0]


@pytest.mark.integration
def test_whole_period_transfer_time_raises_from_the_cw_seed():
    """In-plane targeting loses rank at whole periods; the seed refuses first."""
    r0, rf = _hop(1_000.0)
    with pytest.raises(SingularTransferTimeError, match="singular or ill-conditioned"):
        _correct(r0, rf, tof_s=PERIOD_S)


@pytest.mark.integration
def test_whole_period_transfer_time_raises_from_the_shooting_jacobian():
    """Past the seed, the module's own in-plane conditioning guard must catch it."""
    r0, rf = _hop(1_000.0)

    # The contract is REFUSAL WITH A DIAGNOSTIC, not a particular subclass. Which guard trips
    # first is a property of the LAPACK backend, and both outcomes are correct:
    #   macOS / Accelerate  -> IllConditionedJacobianError (conditioning guard catches it)
    #   Linux / OpenBLAS    -> TargetingConvergenceError  (the damped line search stalls first)
    # An earlier version asserted the subclass and then the exact iteration count. Both passed
    # locally and failed on CI's first ever run, on a different architecture. What must never
    # happen is that an uncertifiable answer is returned.
    with pytest.raises((IllConditionedJacobianError, TargetingConvergenceError)) as excinfo:
        _correct(r0, rf, tof_s=PERIOD_S, dv1_guess_m_s=[0.0, 0.0, 0.0])

    message = str(excinfo.value)
    assert "1.000000" in message, "the error must name the offending transfer time in periods"
    assert any(word in message for word in ("singular", "ill-conditioned", "stalled")), message
    if isinstance(excinfo.value, IllConditionedJacobianError):
        assert excinfo.value.condition_number > 1.0e8


@pytest.mark.integration
def test_propagation_failure_is_not_swallowed():
    """An integrator failure must surface, not become a partial answer.

    A chaser placed one target-radius inward on R-bar sits exactly at the central body, so
    the propagation is singular at the first evaluation.
    """
    with pytest.raises(PropagationError, match="central body singularity"):
        _correct(np.array([-A_ISS_M, 0.0, 0.0]), np.array([0.0, -250.0, 0.0]))


# ---------------------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("r_target0_eci_m", {"r_target0_eci_m": [1.0, 2.0]}),
        ("v_target0_eci_m_s", {"v_target0_eci_m_s": np.zeros((3, 3))}),
        ("r0_hill_m", {"r0_hill_m": [0.0, 0.0]}),
        ("v0_hill_m_s", {"v0_hill_m_s": [0.0, 0.0, 0.0, 0.0]}),
        ("rf_hill_m", {"rf_hill_m": [1.0]}),
        ("vf_hill_m_s", {"vf_hill_m_s": []}),
    ],
)
def test_wrong_shape_inputs_raise(name, kwargs):
    base = {
        "r_target0_eci_m": R_TARGET,
        "v_target0_eci_m_s": V_TARGET,
        "r0_hill_m": np.array([0.0, -1_000.0, 0.0]),
        "v0_hill_m_s": AT_REST,
        "rf_hill_m": np.array([0.0, -250.0, 0.0]),
        "vf_hill_m_s": AT_REST,
        "tof_s": GENERIC_TOF_S,
    }
    with pytest.raises(ValueError, match=f"{name} must have shape"):
        correct_two_impulse_transfer(**{**base, **kwargs})


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_inputs_raise(bad):
    r0 = np.array([0.0, -1_000.0, bad])
    with pytest.raises(ValueError, match="r0_hill_m must be finite"):
        _correct(r0, np.array([0.0, -250.0, 0.0]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tof_s": 0.0}, "tof_s must be finite and strictly positive"),
        ({"tof_s": -100.0}, "tof_s must be finite and strictly positive"),
        ({"tof_s": math.nan}, "tof_s must be finite and strictly positive"),
        ({"tolerance_m": 0.0}, "tolerance_m must be finite and strictly positive"),
        ({"tolerance_m": -1.0}, "tolerance_m must be finite and strictly positive"),
        ({"fd_step_m_s": 0.0}, "fd_step_m_s must be finite and strictly positive"),
        ({"fd_step_m_s": -1e-4}, "fd_step_m_s must be finite and strictly positive"),
        ({"mu_m3_s2": 0.0}, "mu_m3_s2 must be finite and strictly positive"),
        ({"n_rad_s": 0.0}, "n_rad_s must be finite and strictly positive"),
        ({"cross_track_rank_tol": 0.0}, "cross_track_rank_tol must be finite"),
        ({"max_iterations": 0}, "max_iterations must be >= 1"),
        ({"max_iterations": -3}, "max_iterations must be >= 1"),
    ],
)
def test_invalid_scalar_arguments_raise(kwargs, match):
    r0, rf = _hop(1_000.0)
    tof_s = kwargs.pop("tof_s", GENERIC_TOF_S)
    with pytest.raises(ValueError, match=match):
        _correct(r0, rf, tof_s=tof_s, **kwargs)


@pytest.mark.unit
def test_zero_target_radius_raises():
    with pytest.raises(ValueError, match="r_target0_eci_m must be a nonzero position vector"):
        correct_two_impulse_transfer(
            np.zeros(3),
            V_TARGET,
            np.array([0.0, -1_000.0, 0.0]),
            AT_REST,
            np.array([0.0, -250.0, 0.0]),
            AT_REST,
            GENERIC_TOF_S,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tof_s": -1.0}, "tof_s must be finite and strictly positive"),
        ({"mu_m3_s2": -1.0}, "mu_m3_s2 must be finite and strictly positive"),
        ({"r0_hill_m": [0.0, 0.0]}, "r0_hill_m must have shape"),
        ({"r0_hill_m": [0.0, 0.0, np.nan]}, "r0_hill_m must be finite"),
    ],
)
def test_baseline_helper_validates_its_inputs(kwargs, match):
    base = {
        "r_target0_eci_m": R_TARGET,
        "v_target0_eci_m_s": V_TARGET,
        "r0_hill_m": np.array([0.0, -1_000.0, 0.0]),
        "v0_hill_m_s": AT_REST,
        "rf_hill_m": np.array([0.0, -250.0, 0.0]),
        "vf_hill_m_s": AT_REST,
        "tof_s": GENERIC_TOF_S,
    }
    with pytest.raises(ValueError, match=match):
        raw_cw_terminal_miss_m(**{**base, **kwargs})


@pytest.mark.unit
def test_baseline_helper_rejects_a_zero_target_radius():
    with pytest.raises(ValueError, match="r_target0_eci_m must be a nonzero position vector"):
        raw_cw_terminal_miss_m(
            np.zeros(3),
            V_TARGET,
            np.array([0.0, -1_000.0, 0.0]),
            AT_REST,
            np.array([0.0, -250.0, 0.0]),
            AT_REST,
            GENERIC_TOF_S,
        )


# ---------------------------------------------------------------------------------------
# Warm starting
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_an_explicit_guess_reaches_the_same_solution_as_the_cw_seed():
    """The converged answer is a property of the problem, not of where the iteration began.

    Newton from a different starting point must land on the same root. If it did not, the
    residual would not be defining a unique transfer and the whole method would be
    reporting whichever iterate it happened to stop on.
    """
    r0, rf = _hop(10_000.0)
    from_cw = _correct(r0, rf, tolerance_m=1.0e-6)
    nudged = from_cw.cw_dv1_hill_m_s + np.array([0.05, -0.05, 0.02])
    from_guess = _correct(r0, rf, tolerance_m=1.0e-6, dv1_guess_m_s=nudged)

    # Both converge to a terminal miss under 1e-6 m; with a sensitivity of ~3.5e3 m per
    # (m/s), two impulses that both arrive within 1e-6 m must agree to ~1e-9 m/s. Measured
    # agreement is better than that; bound at 1e-7 m/s for headroom.
    assert float(np.linalg.norm(from_cw.dv1_hill_m_s - from_guess.dv1_hill_m_s)) < 1.0e-7
    assert from_guess.initial_residual_m > from_cw.initial_residual_m


@pytest.mark.integration
def test_a_guess_that_already_arrives_needs_no_iterations():
    """Seeding with the converged answer must be recognised, not re-derived."""
    r0, rf = _hop(1_000.0)
    converged = _correct(r0, rf, tolerance_m=1.0e-6)
    again = _correct(r0, rf, tolerance_m=1.0e-3, dv1_guess_m_s=converged.dv1_hill_m_s)
    assert again.iterations == 0
    assert len(again.residual_history_m) == 1
    assert again.dv1_hill_m_s.tobytes() == converged.dv1_hill_m_s.tobytes()


@pytest.mark.integration
def test_a_stalled_correction_names_the_transfer_time_it_stalled_on():
    """Every refusal must say WHICH transfer time it refused.

    Found by CI: `IllConditionedJacobianError` named the offending time in orbital periods
    but `TargetingConvergenceError` did not, so the diagnostic a caller needs depended on
    which guard happened to trip first -- and which one trips is a property of the LAPACK
    backend. docs/CONTRIBUTING.md requires error messages to carry the numbers that
    motivated them, so a user can act without reaching for a debugger.
    """
    r0, rf = _hop(1_000.0)
    with pytest.raises(TargetingConvergenceError) as excinfo:
        # A tolerance below the nonlinear oracle's own noise floor cannot be delivered.
        _correct(r0, rf, tof_s=0.5 * PERIOD_S, tolerance_m=1.0e-15)
    message = str(excinfo.value)
    assert "0.500000" in message, message
    assert "orbital periods" in message, message
