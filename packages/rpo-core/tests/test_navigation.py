"""Navigation error and linear covariance: closed forms, conservation laws, limiting cases.

Four kinds of assertion here, and no golden numbers from a previous run of this code:

* **Closed form.** The guided terminal error is exactly ``-Phi @ e`` for the estimation
  error ``e`` (module M9, derived in ``rpo_core.navigation``'s docstring), so the guidance
  tests assert an identity to machine precision rather than a tolerance. The
  bias-versus-noise tests assert the averaged-covariance closed form
  ``P_b + P_w / m``, and the mean-square terminal error is checked against
  ``trace(Phi P Phi.T)`` with the standard error of *that* estimator, which is
  ``sqrt(2 trace(Sigma**2) / n)`` for a Gaussian and is therefore derived, not chosen.
* **Conservation.** The CW plant matrix is trace-free, so ``det Phi == 1`` and the
  covariance propagation preserves ``det P`` when ``Q = 0``.
* **Limiting case.** ``dt = 0`` returns the input covariance; zero estimation error reaches
  the commanded terminal state exactly; a forward-backward round trip is the identity.
* **Complement.** Every claim is paired with a measurement of its violation: transposing
  the STM changes the answer measurably, dropping ``Q`` changes it by exactly ``Q``,
  planning on truth instead of the estimate collapses the terminal error to zero, a bias
  and white noise of the same *total* covariance are indistinguishable from one look and
  separate by a measured factor at eight, and the Monte Carlo covariance agreement that
  holds in the small-dispersion limit is shown to fail outside it.

Statistical tolerances are computed from the estimator's standard error at ``_Z_BOUND``
sigmas. Machine-precision tolerances carry a comment recording the measured value and the
headroom taken over it.
"""

import math

import numpy as np
import pytest
from rpo_core.constants import MU_EARTH_M3_S2, mean_motion_rad_s, orbital_period_s
from rpo_core.exceptions import RpoCoreError
from rpo_core.montecarlo import MagnitudePointingDispersion
from rpo_core.navigation import (
    STATE_DIMENSION,
    CovarianceDefinitionError,
    GuidanceDefinitionError,
    NavigationErrorModel,
    NavigationModelError,
    cw_truth_propagator,
    plan_from_estimate,
    propagate_covariance,
    terminal_error_covariance,
    validate_covariance,
)
from rpo_core.relative.cw import cw_stm, propagate_cw
from rpo_core.relative.nonlinear import propagate_relative_nonlinear

# Number of standard errors allowed in every statistical-convergence assertion.
#
# Five sigma two-sided is a false-failure probability of 5.7e-7 per assertion, and this file
# makes ~20 of them, so the spurious-failure rate is below 1.2e-5 per run. The tests are
# seeded, so this is a statement about how discriminating the bound is rather than about
# flakiness: every mutation in this module's mutation study moves the statistic by tens of
# sigma, so the bound could be 5x looser without losing a single defect.
_Z_BOUND = 5.0

_ALTITUDE_M = 420.0e3
_A_M = 6378137.0 + _ALTITUDE_M
_N_RAD_S = mean_motion_rad_s(_A_M)
_PERIOD_S = orbital_period_s(_A_M)

# 0.3 periods, deliberately not 0.5: at a half period the cross-track subproblem is
# rank-deficient and the ``-Phi @ e`` identity would hold only in-plane. The identity under
# test is a statement about the full 6-vector, so the transfer time has to be one where the
# full 6-vector is controllable.
_TOF_S = 0.3 * _PERIOD_S

_START_HILL = np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
_TARGET_HILL = np.array([0.0, -250.0, 0.0, 0.0, 0.0, 0.0])


def _target_state_eci() -> tuple[np.ndarray, np.ndarray]:
    """Return an ISS-like circular target state, built here to keep rpo-core self-contained.

    ``rpo_traj.plan.target_state_eci`` does the same thing, but rpo-core must not depend on
    rpo-traj, and a four-line circular state is not worth inverting the dependency for.
    """
    speed_m_s = math.sqrt(MU_EARTH_M3_S2 / _A_M)
    inclination_rad = math.radians(51.6)
    return (
        np.array([_A_M, 0.0, 0.0]),
        speed_m_s * np.array([0.0, math.cos(inclination_rad), math.sin(inclination_rad)]),
    )


def _diagonal(sigma_position_m: float, sigma_velocity_m_s: float) -> np.ndarray:
    """Return a diagonal 6x6 relative-state covariance from two one-sigmas."""
    return np.diag(
        [
            sigma_position_m**2,
            sigma_position_m**2,
            sigma_position_m**2,
            sigma_velocity_m_s**2,
            sigma_velocity_m_s**2,
            sigma_velocity_m_s**2,
        ]
    )


def _correlated_covariance() -> np.ndarray:
    """Return a full (non-diagonal) SPD relative-state covariance.

    Built as ``L L.T`` from a fixed lower-triangular factor rather than typed in, so it is
    positive definite by construction and every off-diagonal is populated -- an axis-aligned
    covariance would let a transposition bug through, because a diagonal matrix commutes
    with the operation that is supposed to break.
    """
    factor = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 1.6, 0.0, 0.0, 0.0, 0.0],
            [-0.4, 0.3, 1.2, 0.0, 0.0, 0.0],
            [3.0e-3, -1.0e-3, 5.0e-4, 4.0e-3, 0.0, 0.0],
            [-1.0e-3, 2.5e-3, 8.0e-4, 6.0e-4, 3.0e-3, 0.0],
            [5.0e-4, 4.0e-4, -1.5e-3, -3.0e-4, 7.0e-4, 2.0e-3],
        ]
    )
    return factor @ factor.T


def _averaged_looks(
    model: NavigationErrorModel, truth: np.ndarray, rng: np.random.Generator, m_looks: int
) -> list[np.ndarray]:
    """Return ``m_looks`` estimates from **one** run: one bias draw, ``m_looks`` noise draws.

    The bias is drawn by :meth:`NavigationErrorModel.begin_run` once, outside the loop. That
    placement is the whole model -- moving ``begin_run`` inside the comprehension turns the
    bias into white noise of the same marginal covariance, which is the defect
    ``test_a_constant_bias_gives_a_different_terminal_error_than_white_noise`` exists to
    detect.
    """
    solution = model.begin_run(rng)
    return [solution.estimate(truth, rng) for _ in range(m_looks)]


def _covariance_z_scores(sample: np.ndarray, predicted: np.ndarray, n: int) -> np.ndarray:
    """Return ``|sample - predicted|`` in units of the covariance estimator's standard error.

    For draws from ``N(mu, C)`` the sample covariance entry has variance
    ``(C_ii C_jj + C_ij**2) / n`` (Wishart, to leading order in ``1/n``). Dividing by that
    standard error is what makes a single tolerance meaningful across a matrix whose entries
    span m^2 to m^2/s^2 -- ten orders of magnitude here -- instead of forcing a per-block
    hand-picked bound.
    """
    diagonal = np.diag(predicted)
    standard_error = np.sqrt((np.outer(diagonal, diagonal) + predicted**2) / n)
    return np.abs(sample - predicted) / standard_error


# --------------------------------------------------------------------------------------
# validate_covariance: every input-validation branch
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_covariance_rejects_a_non_square_matrix():
    with pytest.raises(CovarianceDefinitionError, match=r"square 2-D matrix.*\(6, 3\)"):
        validate_covariance(np.zeros((6, 3)))


@pytest.mark.unit
def test_validate_covariance_rejects_a_one_dimensional_input():
    with pytest.raises(CovarianceDefinitionError, match=r"square 2-D matrix"):
        validate_covariance(np.ones(6))


@pytest.mark.unit
def test_validate_covariance_rejects_the_wrong_dimension():
    with pytest.raises(CovarianceDefinitionError, match=r"shape \(6, 6\).*got \(3, 3\)"):
        validate_covariance(np.eye(3), dimension=STATE_DIMENSION)


@pytest.mark.unit
def test_validate_covariance_rejects_an_empty_matrix():
    with pytest.raises(CovarianceDefinitionError, match=r"must be non-empty"):
        validate_covariance(np.zeros((0, 0)))


@pytest.mark.unit
def test_validate_covariance_rejects_a_non_finite_entry():
    matrix = np.eye(6)
    matrix[2, 2] = np.nan
    with pytest.raises(CovarianceDefinitionError, match=r"must be finite"):
        validate_covariance(matrix)


@pytest.mark.unit
def test_validate_covariance_rejects_asymmetry_and_reports_it():
    matrix = np.eye(6)
    matrix[0, 1] = 0.5
    with pytest.raises(CovarianceDefinitionError, match=r"not symmetric.*max \|C - C\.T\|"):
        validate_covariance(matrix)


@pytest.mark.unit
def test_validate_covariance_rejects_a_non_positive_definite_matrix():
    matrix = np.eye(6)
    matrix[3, 3] = 0.0
    with pytest.raises(
        CovarianceDefinitionError, match=r"not positive definite.*smallest eigenvalue"
    ):
        validate_covariance(matrix)


@pytest.mark.unit
def test_validate_covariance_accepts_a_singular_process_noise_matrix():
    # Q = 0 is the ordinary case, so semi-definiteness is the correct requirement for
    # process noise. This is the complement of the previous test: the same matrix that is
    # rejected as a covariance is accepted as a Q.
    matrix = np.eye(6)
    matrix[3, 3] = 0.0
    accepted = validate_covariance(matrix, require_positive_definite=False)
    assert accepted[3, 3] == 0.0


@pytest.mark.unit
def test_validate_covariance_rejects_negative_process_noise():
    matrix = np.eye(6)
    matrix[3, 3] = -1.0
    with pytest.raises(CovarianceDefinitionError, match=r"not positive semi-definite"):
        validate_covariance(matrix, require_positive_definite=False)


@pytest.mark.unit
def test_validate_covariance_returns_an_exactly_symmetric_matrix():
    # Round-off level asymmetry is accepted and removed, so that the result satisfies the
    # zero-tolerance symmetry check of VectorNormalDispersion downstream.
    matrix = _correlated_covariance()
    matrix[0, 5] += 1.0e-16
    result = validate_covariance(matrix)
    assert np.array_equal(result, result.T)


@pytest.mark.unit
def test_covariance_errors_are_rpo_core_errors():
    # The typed-exception hierarchy is part of the contract: a caller catching RpoCoreError
    # must catch these, and NavigationModelError must group everything this module raises.
    assert issubclass(CovarianceDefinitionError, NavigationModelError)
    assert issubclass(GuidanceDefinitionError, NavigationModelError)
    assert issubclass(NavigationModelError, RpoCoreError)


# --------------------------------------------------------------------------------------
# propagate_covariance: conservation, limiting cases, complements
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_zero_step_returns_the_input_covariance():
    # Limiting case: Phi(0) = I, so the propagation is the identity. Exact, not approximate.
    covariance = _correlated_covariance()
    assert np.allclose(
        propagate_covariance(covariance, _N_RAD_S, 0.0), covariance, rtol=0.0, atol=0.0
    )


@pytest.mark.unit
def test_determinant_is_conserved_by_the_propagation():
    # Conservation law: the CW plant matrix is trace-free, so det(Phi) = 1 exactly and
    # det(Phi P Phi.T) = det(P). Measured relative drift over a quarter, half and full
    # period: 5.0e-14 worst case, so 1e-9 carries ~2e4 headroom and still fails instantly if
    # the STM is replaced by anything that is not symplectic.
    covariance = _correlated_covariance()
    reference = float(np.linalg.det(covariance))
    for fraction in (0.25, 0.5, 1.0, 2.0):
        propagated = propagate_covariance(covariance, _N_RAD_S, fraction * _PERIOD_S)
        assert float(np.linalg.det(propagated)) == pytest.approx(reference, rel=1.0e-9)


@pytest.mark.unit
def test_determinant_of_the_state_transition_matrix_is_one():
    # The reason the previous test holds, asserted separately so a failure says which of the
    # two claims broke. Measured |det - 1| worst case over the same steps: 4.5e-15.
    for fraction in (0.25, 0.5, 1.0, 2.0):
        assert float(np.linalg.det(cw_stm(_N_RAD_S, fraction * _PERIOD_S))) == pytest.approx(
            1.0, abs=1.0e-12
        )


@pytest.mark.unit
def test_forward_then_backward_propagation_recovers_the_covariance():
    # Limiting case: Phi(-dt) = Phi(dt)^-1, so the round trip is the identity. Measured
    # worst-case relative residual 3.1e-13 at a half period; 1e-9 is 3e3 headroom.
    covariance = _correlated_covariance()
    dt_s = 0.5 * _PERIOD_S
    forward = propagate_covariance(covariance, _N_RAD_S, dt_s)
    round_trip = propagate_covariance(forward, _N_RAD_S, -dt_s)
    assert np.allclose(round_trip, covariance, rtol=1.0e-9, atol=1.0e-12 * np.max(covariance))


@pytest.mark.unit
def test_process_noise_is_added_exactly():
    # Complement for "Q is dropped": the difference between the two calls must be Q itself,
    # to machine precision, not merely "bigger".
    covariance = _correlated_covariance()
    process_noise = _diagonal(0.5, 1.0e-3)
    without = propagate_covariance(covariance, _N_RAD_S, 900.0)
    with_noise = propagate_covariance(covariance, _N_RAD_S, 900.0, process_noise=process_noise)
    assert np.allclose(with_noise - without, process_noise, rtol=1.0e-12, atol=1.0e-18)


@pytest.mark.unit
def test_transposing_the_state_transition_matrix_changes_the_answer():
    # Complement for the Phi P Phi.T ordering. det(Phi) = det(Phi.T) = 1, so the determinant
    # conservation law above cannot see a transposition -- this test is what does. Measured
    # separation at a quarter period: the largest entry differs by a factor of 1.5e+05
    # relative to the correct value, so the 1.0 relative bound below is not a knife edge.
    covariance = _correlated_covariance()
    dt_s = 0.25 * _PERIOD_S
    phi = cw_stm(_N_RAD_S, dt_s)
    correct = propagate_covariance(covariance, _N_RAD_S, dt_s)
    transposed = phi.T @ covariance @ phi
    relative = np.max(np.abs(transposed - correct)) / np.max(np.abs(correct))
    assert relative > 1.0


@pytest.mark.unit
def test_chained_propagation_stays_symmetric():
    # 20 chained half-period steps, which is inside the conditioning limit measured in the
    # next test. Re-symmetrising each step holds the asymmetry at exactly zero (measured);
    # the bare product Phi P Phi.T drifts to 1.045e-15 relative over 200 steps. The
    # assertion is exact equality rather than a tolerance because that is what was measured
    # -- a tolerance here would be a weaker claim than the code actually supports.
    covariance = _diagonal(5.0, 1.0e-2)
    bare = covariance.copy()
    dt_s = 0.5 * _PERIOD_S
    phi = cw_stm(_N_RAD_S, dt_s)
    worst = 0.0
    worst_bare = 0.0
    for _ in range(20):
        covariance = propagate_covariance(covariance, _N_RAD_S, dt_s)
        assert np.array_equal(covariance, covariance.T)
        scale = max(1.0, float(np.max(np.abs(covariance))))
        worst = max(worst, float(np.max(np.abs(covariance - covariance.T))) / scale)
        bare = phi @ bare @ phi.T
        worst_bare = max(worst_bare, float(np.max(np.abs(bare - bare.T)) / np.max(np.abs(bare))))
    assert worst == 0.0, f"re-symmetrised chain should be exactly symmetric, got {worst:.3e}"
    # Complement: the re-symmetrisation is doing something. Without it the product drifts
    # off symmetry, and although it stays far inside DEFAULT_SYMMETRY_RTOL over this many
    # steps, it is not zero -- so the exact-equality assertion above is a real property of
    # propagate_covariance rather than an accident of the arithmetic.
    assert worst_bare > 0.0


@pytest.mark.unit
def test_a_long_chain_eventually_loses_positive_definiteness():
    # Not a defect, a documented limit, and one worth a test rather than a footnote. The CW
    # along-track secular drift makes the condition number grow quadratically in elapsed
    # time, and a chained Phi P Phi.T eventually has a round-off-negative eigenvalue.
    # Measured: fine at 40 half-period steps (cond 1.1e+17), refused at 43 (smallest
    # eigenvalue -1.8e-10 against a largest of 2.8e+07). The refusal is right -- the matrix
    # has no Cholesky factor by then -- and the range checked below (survives 20, fails by
    # 100) brackets the measured transition with room on both sides so that a change in the
    # BLAS or the platform moves the number without breaking the test.
    covariance = _diagonal(5.0, 1.0e-2)
    dt_s = 0.5 * _PERIOD_S
    for _ in range(20):
        covariance = propagate_covariance(covariance, _N_RAD_S, dt_s)
    assert float(np.linalg.cond(covariance)) > 1.0e12

    with pytest.raises(CovarianceDefinitionError, match=r"not positive definite"):
        for _ in range(100):
            covariance = propagate_covariance(covariance, _N_RAD_S, dt_s)


@pytest.mark.unit
def test_propagate_covariance_rejects_a_bad_covariance():
    with pytest.raises(CovarianceDefinitionError, match=r"covariance must have shape \(6, 6\)"):
        propagate_covariance(np.eye(3), _N_RAD_S, 100.0)


@pytest.mark.unit
def test_propagate_covariance_rejects_bad_process_noise():
    with pytest.raises(CovarianceDefinitionError, match=r"process_noise.*not symmetric"):
        asymmetric = np.eye(6)
        asymmetric[1, 0] = 1.0
        propagate_covariance(np.eye(6), _N_RAD_S, 100.0, process_noise=asymmetric)


@pytest.mark.unit
def test_propagate_covariance_rejects_a_non_positive_mean_motion():
    with pytest.raises(ValueError, match=r"n_rad_s must be a finite positive mean motion"):
        propagate_covariance(np.eye(6), 0.0, 100.0)


@pytest.mark.unit
def test_propagate_covariance_rejects_a_non_finite_step():
    with pytest.raises(ValueError, match=r"dt_s must be finite"):
        propagate_covariance(np.eye(6), _N_RAD_S, math.inf)


# --------------------------------------------------------------------------------------
# Model M9's independent check: Monte Carlo covariance vs the linear prediction
# --------------------------------------------------------------------------------------


def _monte_carlo_final_states(
    covariance: np.ndarray, n_samples: int, dt_s: float, *, nonlinear: bool, seed: int
) -> np.ndarray:
    """Draw dispersed initial states, propagate each, and return the final states."""
    nominal = np.array([0.0, -200.0, 0.0, 0.0, 0.0, 0.0])
    model = NavigationErrorModel(noise_covariance=covariance)
    rng = np.random.default_rng(seed)
    times_s = np.array([0.0, dt_s])
    if nonlinear:
        r_target, v_target = _target_state_eci()
        return np.array(
            [
                propagate_relative_nonlinear(
                    r_target, v_target, nominal + model.draw_noise(rng), times_s
                )[-1]
                for _ in range(n_samples)
            ]
        )
    return np.array(
        [propagate_cw(_N_RAD_S, nominal + model.draw_noise(rng), dt_s) for _ in range(n_samples)]
    )


@pytest.mark.unit
def test_monte_carlo_covariance_matches_the_linear_prediction_under_cw():
    # The fast tier of model M9's independent check. Under CW the truth flow is the same
    # linear map the prediction uses, so what this measures is the sampling agreement and
    # the Phi P Phi.T ordering -- it is the test that fails when the STM is transposed or
    # the product is written the wrong way round. The nonlinear version below is the
    # genuinely independent oracle.
    n_samples = 4000
    dt_s = 0.25 * _PERIOD_S
    covariance = _diagonal(0.1, 1.0e-4)
    final = _monte_carlo_final_states(covariance, n_samples, dt_s, nonlinear=False, seed=20260824)
    predicted = propagate_covariance(covariance, _N_RAD_S, dt_s)
    z = _covariance_z_scores(np.cov(final, rowvar=False, ddof=1), predicted, n_samples)
    assert float(np.max(z)) < _Z_BOUND, f"worst covariance entry off by {np.max(z):.2f} sigma"


@pytest.mark.slow
@pytest.mark.integration
def test_monte_carlo_covariance_matches_the_linear_prediction_under_nonlinear_dynamics():
    # Model M9's stated independent check, done properly: the samples are flown through the
    # *nonlinear* relative dynamics (rpo_core.relative.nonlinear, the oracle CW is a
    # linearisation of), and the sample covariance is compared with the analytic
    # Phi P Phi.T. Agreement here is a statement about physics, not about arithmetic.
    #
    # Why the nominal separation is 200 m and the dispersion 0.1 m: the systematic gap
    # between the nonlinear flow's Jacobian and the CW STM scales as rho/r, which is
    # 200 / 6.8e6 = 2.9e-05 here, while the sampling standard error of a covariance entry is
    # sqrt(2/n) = 2.2e-02 at n = 4000. Sampling noise therefore dominates the systematic
    # difference by ~750x, which is what "small-dispersion limit" has to mean for the
    # tolerance below to be derived from the standard error rather than from the physics.
    #
    # Measured worst entry: 1.72 sigma at n = 300 and 2.26 sigma at n = 1200. The bound is
    # 5 sigma, so this is not close to failing, and the 21 distinct entries of a 6x6
    # symmetric matrix have an expected worst |z| near 2.5 -- the measurement is where an
    # honest sample should be, not comfortably inside a loose bound.
    n_samples = 1200
    dt_s = 0.25 * _PERIOD_S
    covariance = _diagonal(0.1, 1.0e-4)
    final = _monte_carlo_final_states(covariance, n_samples, dt_s, nonlinear=True, seed=20260824)
    predicted = propagate_covariance(covariance, _N_RAD_S, dt_s)
    z = _covariance_z_scores(np.cov(final, rowvar=False, ddof=1), predicted, n_samples)
    assert float(np.max(z)) < _Z_BOUND, f"worst covariance entry off by {np.max(z):.2f} sigma"


@pytest.mark.slow
@pytest.mark.integration
def test_large_dispersion_departs_from_the_linear_prediction():
    # Complement: the agreement above is a *small-dispersion* claim, and a test that only
    # shows agreement cannot tell whether the linear prediction is right or merely
    # untestable. If this test ever passes at _Z_BOUND, the one above has stopped measuring
    # anything.
    #
    # The dispersion needed here was **measured, not assumed, and the first guess was
    # wrong**. A 50 km one-sigma (rho/r = 7.4e-3) still agrees at 1.28 sigma over 1000
    # samples: the linear covariance propagation is far more robust than the CW *trajectory*
    # error suggests, because a covariance feels only the flow's Jacobian and not the
    # accumulated position error. Measured worst |z| at 400 samples over a quarter period:
    #
    #     sigma_pos = 150 km (rho/r = 0.0221):   2.16 sigma  -- still agrees
    #     sigma_pos = 300 km (rho/r = 0.0441):   9.12 sigma
    #     sigma_pos = 600 km (rho/r = 0.0883):  28.37 sigma
    #
    # 600 km is used: 28.4 sigma against a 5 sigma bound is 5.7x headroom, and it is well
    # past the 300 km point where the effect first clears the bound. At that separation the
    # "chaser" is on a visibly different orbit, which is the honest statement of where a
    # linear covariance stops meaning anything.
    n_samples = 400
    dt_s = 0.25 * _PERIOD_S
    covariance = _diagonal(600.0e3, 120.0)
    final = _monte_carlo_final_states(covariance, n_samples, dt_s, nonlinear=True, seed=7)
    predicted = propagate_covariance(covariance, _N_RAD_S, dt_s)
    z = _covariance_z_scores(np.cov(final, rowvar=False, ddof=1), predicted, n_samples)
    assert float(np.max(z)) > _Z_BOUND, (
        f"large dispersion should break the linear prediction, worst entry only "
        f"{np.max(z):.2f} sigma out"
    )


# --------------------------------------------------------------------------------------
# NavigationErrorModel: bias is a bias
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_model_with_no_error_returns_the_truth_exactly():
    # Limiting case, and the control every dispersion study needs: zero navigation error
    # must be bit-for-bit identity, not "close".
    truth = np.array([1.0, -2.0, 3.0, -0.01, 0.02, -0.03])
    solution = NavigationErrorModel().begin_run(np.random.default_rng(0))
    assert np.array_equal(solution.estimate(truth, np.random.default_rng(0)), truth)
    assert np.array_equal(solution.bias_hill, np.zeros(STATE_DIMENSION))


@pytest.mark.unit
def test_the_bias_is_constant_across_every_estimate_in_a_run():
    # The defining property. A bias-only model must return exactly the same estimate every
    # time it is asked, however many times the generator is consulted.
    model = NavigationErrorModel(bias_covariance=_diagonal(1.0, 1.0e-3))
    truth = np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(11)
    solution = model.begin_run(rng)
    estimates = [solution.estimate(truth, rng) for _ in range(20)]
    for estimate in estimates[1:]:
        assert np.array_equal(estimate, estimates[0])
    assert not np.array_equal(estimates[0], truth)


@pytest.mark.unit
def test_the_bias_differs_between_runs():
    # Complement of the previous test: constant *within* a run, not constant *everywhere*,
    # which would be a fixed offset rather than a random bias.
    model = NavigationErrorModel(bias_covariance=_diagonal(1.0, 1.0e-3))
    rng = np.random.default_rng(11)
    first = model.begin_run(rng).bias_hill
    second = model.begin_run(rng).bias_hill
    assert not np.array_equal(first, second)


@pytest.mark.unit
def test_the_white_noise_is_redrawn_at_every_estimate():
    model = NavigationErrorModel(noise_covariance=_diagonal(1.0, 1.0e-3))
    truth = np.zeros(STATE_DIMENSION)
    rng = np.random.default_rng(3)
    solution = model.begin_run(rng)
    first = solution.estimate(truth, rng)
    second = solution.estimate(truth, rng)
    assert not np.array_equal(first, second)


@pytest.mark.unit
def test_a_single_estimate_cannot_distinguish_bias_from_white_noise():
    # The knife edge that makes the next test meaningful. Two models with the same *total*
    # covariance produce the same distribution of single-look errors, so any test that
    # separated them at m = 1 would be measuring an implementation artefact rather than the
    # bias. Checked as an identity on the closed form and statistically on the samples.
    total = _diagonal(2.0, 5.0e-3)
    half = total / 2.0
    split = NavigationErrorModel(noise_covariance=half, bias_covariance=half)
    white = NavigationErrorModel(noise_covariance=total)
    assert np.allclose(split.total_covariance, white.total_covariance, rtol=0.0, atol=0.0)
    assert np.allclose(split.averaged_covariance(1), white.averaged_covariance(1))

    n_runs = 20000
    truth = np.zeros(STATE_DIMENSION)
    errors = {}
    for name, model in (("split", split), ("white", white)):
        rng = np.random.default_rng(2024)
        errors[name] = np.array([model.begin_run(rng).estimate(truth, rng) for _ in range(n_runs)])
    for name in ("split", "white"):
        z = _covariance_z_scores(np.cov(errors[name], rowvar=False, ddof=1), total, n_runs)
        assert float(np.max(z)) < _Z_BOUND, f"{name} single-look covariance off by {np.max(z):.2f}"


@pytest.mark.unit
def test_averaging_estimates_separates_a_bias_from_white_noise():
    # The closed form P_b + P_w / m, measured. With eight looks the white-only model's error
    # covariance is 8x smaller while the half-and-half model's is only 1.78x smaller, because
    # its bias half does not average at all.
    total = _diagonal(2.0, 5.0e-3)
    half = total / 2.0
    split = NavigationErrorModel(noise_covariance=half, bias_covariance=half)
    white = NavigationErrorModel(noise_covariance=total)
    m_looks = 8

    assert np.allclose(split.averaged_covariance(m_looks), half + half / m_looks)
    assert np.allclose(white.averaged_covariance(m_looks), total / m_looks)

    n_runs = 20000
    truth = np.zeros(STATE_DIMENSION)
    for model in (split, white):
        rng = np.random.default_rng(99)
        averaged = np.array(
            [np.mean(_averaged_looks(model, truth, rng, m_looks), axis=0) for _ in range(n_runs)]
        )
        z = _covariance_z_scores(
            np.cov(averaged, rowvar=False, ddof=1), model.averaged_covariance(m_looks), n_runs
        )
        assert float(np.max(z)) < _Z_BOUND, f"averaged covariance off by {np.max(z):.2f} sigma"


@pytest.mark.unit
def test_averaged_covariance_rejects_a_non_positive_look_count():
    model = NavigationErrorModel(noise_covariance=_diagonal(1.0, 1.0e-3))
    with pytest.raises(CovarianceDefinitionError, match=r"n_estimates must be a positive int"):
        model.averaged_covariance(0)


@pytest.mark.unit
def test_the_model_rejects_a_malformed_covariance():
    with pytest.raises(CovarianceDefinitionError, match=r"noise_covariance.*shape \(6, 6\)"):
        NavigationErrorModel(noise_covariance=np.eye(4))


@pytest.mark.unit
def test_with_bias_rejects_a_malformed_bias():
    model = NavigationErrorModel(bias_covariance=_diagonal(1.0, 1.0e-3))
    with pytest.raises(GuidanceDefinitionError, match=r"bias_hill must have shape \(6,\)"):
        model.with_bias(np.zeros(3))


@pytest.mark.unit
def test_estimate_rejects_a_malformed_truth_state():
    model = NavigationErrorModel(noise_covariance=_diagonal(1.0, 1.0e-3))
    solution = model.begin_run(np.random.default_rng(0))
    with pytest.raises(GuidanceDefinitionError, match=r"truth_state_hill must have shape"):
        solution.estimate(np.zeros(5), np.random.default_rng(0))


@pytest.mark.unit
def test_the_bias_dispersion_round_trips_through_with_bias():
    # The campaign draws the bias as a montecarlo Dispersion (once per run, by construction)
    # and hands it back through with_bias. The two paths must agree.
    model = NavigationErrorModel(bias_covariance=_correlated_covariance())
    dispersion = model.bias_dispersion()
    assert dispersion is not None
    drawn = dispersion.sample(np.random.default_rng(5))
    assert np.array_equal(model.with_bias(drawn).bias_hill, drawn)


@pytest.mark.unit
def test_a_model_without_a_bias_declares_no_dispersion():
    assert NavigationErrorModel(noise_covariance=_diagonal(1.0, 1.0e-3)).bias_dispersion() is None
    assert NavigationErrorModel().draw_noise(np.random.default_rng(0)) is None


# --------------------------------------------------------------------------------------
# plan_from_estimate: the -Phi @ e closed form
# --------------------------------------------------------------------------------------


def _guided(estimate: np.ndarray, truth: np.ndarray, execute_fn=None, n_samples: int = 3):
    """Plan from ``estimate``, fly ``truth`` under CW, return the transfer."""
    return plan_from_estimate(
        _N_RAD_S,
        np.linspace(0.0, _TOF_S, n_samples),
        estimated_state_hill=estimate,
        truth_state_hill=truth,
        commanded_terminal_state_hill=_TARGET_HILL,
        propagate_fn=cw_truth_propagator(_N_RAD_S),
        execute_fn=execute_fn,
    )


@pytest.mark.unit
def test_zero_estimation_error_reaches_the_commanded_state_exactly():
    # The limiting case the whole campaign is calibrated against: with no navigation error
    # and no execution error the guided flight is the nominal plan. Measured terminal miss
    # 3.9e-13 m on a 750 m hop (5e-16 relative); 1e-9 m is ~2500x headroom and is still far
    # below anything a constraint could notice.
    result = _guided(_START_HILL, _START_HILL)
    assert result.terminal_position_error_m < 1.0e-9
    assert result.terminal_velocity_error_m_s < 1.0e-12
    assert np.array_equal(result.dv1_executed_hill_m_s, result.dv1_commanded_hill_m_s)


@pytest.mark.unit
def test_the_terminal_error_is_minus_phi_times_the_estimation_error():
    # The closed form. This is the test that fails if the plan is flown against the estimate
    # instead of against truth (the terminal error would collapse to zero) or if the plan is
    # built from truth instead of the estimate (likewise). Measured worst residual over the
    # five cases below: 1.4e-11 m against errors of order 1 m, i.e. 1.4e-11 relative;
    # 1e-7 m is ~7000x headroom.
    rng = np.random.default_rng(31337)
    model = NavigationErrorModel(noise_covariance=_diagonal(1.0, 2.0e-3))
    phi = cw_stm(_N_RAD_S, _TOF_S)
    for _ in range(5):
        error = model.draw_noise(rng)
        result = _guided(_START_HILL + error, _START_HILL)
        predicted = -phi @ error
        achieved = result.terminal_state_hill - _TARGET_HILL
        assert np.allclose(achieved, predicted, rtol=1.0e-7, atol=1.0e-7)
        assert np.allclose(result.estimation_error_hill, error)


@pytest.mark.unit
def test_planning_on_truth_reports_no_error_at_all():
    # Complement for the closed form above: pass the same state as both estimate and truth
    # -- which is what "accidentally planned on truth" looks like -- and the terminal error
    # vanishes regardless of how large the estimation error actually was. A campaign built
    # that way would report a reassuring answer that means nothing, which is why the two
    # states are separate keyword-only arguments.
    error = np.array([3.0, -2.0, 1.5, 0.01, -0.02, 0.005])
    truth = _START_HILL
    honest = _guided(truth + error, truth)
    self_congratulatory = _guided(truth + error, truth + error)
    assert honest.terminal_position_error_m > 1.0
    assert self_congratulatory.terminal_position_error_m < 1.0e-9


@pytest.mark.unit
def test_burn_execution_error_changes_the_flown_impulse_and_the_miss():
    # execute_fn is wired to both impulses, not just the departure one.
    sample = MagnitudePointingDispersion(
        sigma_magnitude=0.05, sigma_pointing_rad=math.radians(3.0)
    ).sample(np.random.default_rng(17))
    perfect = _guided(_START_HILL, _START_HILL)
    dispersed = _guided(_START_HILL, _START_HILL, execute_fn=sample.apply)
    assert not np.allclose(dispersed.dv1_executed_hill_m_s, dispersed.dv1_commanded_hill_m_s)
    assert not np.allclose(dispersed.dv2_executed_hill_m_s, dispersed.dv2_commanded_hill_m_s)
    assert dispersed.terminal_position_error_m > perfect.terminal_position_error_m
    # The rotation is exact, so the executed magnitude is the scale factor times the
    # commanded one to machine precision -- the property MagnitudePointingDispersion
    # guarantees, re-checked here because plan_from_estimate is what applies it.
    assert float(np.linalg.norm(dispersed.dv1_executed_hill_m_s)) == pytest.approx(
        abs(sample.scale) * float(np.linalg.norm(dispersed.dv1_commanded_hill_m_s)), rel=1.0e-12
    )


@pytest.mark.unit
def test_the_final_sample_carries_the_arrival_impulse():
    # Without this the terminal velocity error would report the pre-burn coast velocity and
    # every campaign would look like a manoeuvre that failed to stop.
    result = _guided(_START_HILL, _START_HILL, n_samples=11)
    assert np.allclose(result.states_hill[-1], result.terminal_state_hill)
    coast = cw_truth_propagator(_N_RAD_S)(
        np.concatenate((_START_HILL[:3], _START_HILL[3:] + result.dv1_executed_hill_m_s)),
        result.times_s,
    )
    assert np.allclose(result.states_hill[-1, 3:] - coast[-1, 3:], result.dv2_executed_hill_m_s)
    assert np.allclose(result.states_hill[:-1], coast[:-1])


@pytest.mark.unit
def test_cw_truth_propagator_agrees_with_the_state_transition_matrix():
    state = np.array([10.0, -1000.0, 5.0, 0.01, -0.02, 0.003])
    times_s = np.linspace(0.0, _TOF_S, 7)
    flown = cw_truth_propagator(_N_RAD_S)(state, times_s)
    for index, t in enumerate(times_s):
        assert np.allclose(flown[index], cw_stm(_N_RAD_S, float(t)) @ state)


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_time_grid_that_does_not_start_at_zero():
    with pytest.raises(GuidanceDefinitionError, match=r"times_s must start at 0\.0 s"):
        plan_from_estimate(
            _N_RAD_S,
            np.linspace(10.0, _TOF_S, 5),
            estimated_state_hill=_START_HILL,
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=cw_truth_propagator(_N_RAD_S),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_single_output_time():
    with pytest.raises(GuidanceDefinitionError, match=r"at least two output times"):
        plan_from_estimate(
            _N_RAD_S,
            np.array([0.0]),
            estimated_state_hill=_START_HILL,
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=cw_truth_propagator(_N_RAD_S),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_non_increasing_time_grid():
    with pytest.raises(GuidanceDefinitionError, match=r"strictly increasing"):
        plan_from_estimate(
            _N_RAD_S,
            np.array([0.0, 100.0, 100.0]),
            estimated_state_hill=_START_HILL,
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=cw_truth_propagator(_N_RAD_S),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_non_finite_time_grid():
    with pytest.raises(GuidanceDefinitionError, match=r"times_s must be finite"):
        plan_from_estimate(
            _N_RAD_S,
            np.array([0.0, math.inf]),
            estimated_state_hill=_START_HILL,
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=cw_truth_propagator(_N_RAD_S),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_malformed_state():
    with pytest.raises(GuidanceDefinitionError, match=r"estimated_state_hill must have shape"):
        plan_from_estimate(
            _N_RAD_S,
            np.linspace(0.0, _TOF_S, 3),
            estimated_state_hill=np.zeros(3),
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=cw_truth_propagator(_N_RAD_S),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_non_finite_state():
    with pytest.raises(GuidanceDefinitionError, match=r"truth_state_hill must be finite"):
        plan_from_estimate(
            _N_RAD_S,
            np.linspace(0.0, _TOF_S, 3),
            estimated_state_hill=_START_HILL,
            truth_state_hill=np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0]),
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=cw_truth_propagator(_N_RAD_S),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_propagator_returning_the_wrong_shape():
    with pytest.raises(GuidanceDefinitionError, match=r"propagate_fn must return an array"):
        plan_from_estimate(
            _N_RAD_S,
            np.linspace(0.0, _TOF_S, 4),
            estimated_state_hill=_START_HILL,
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=lambda state, times: np.zeros((len(times) - 1, 6)),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_propagator_returning_non_finite_states():
    with pytest.raises(GuidanceDefinitionError, match=r"non-finite state"):
        plan_from_estimate(
            _N_RAD_S,
            np.linspace(0.0, _TOF_S, 4),
            estimated_state_hill=_START_HILL,
            truth_state_hill=_START_HILL,
            commanded_terminal_state_hill=_TARGET_HILL,
            propagate_fn=lambda state, times: np.full((len(times), 6), np.nan),
        )


@pytest.mark.unit
def test_plan_from_estimate_rejects_a_burn_model_returning_the_wrong_shape():
    with pytest.raises(GuidanceDefinitionError, match=r"dv1_executed must have shape \(3,\)"):
        _guided(_START_HILL, _START_HILL, execute_fn=lambda dv: np.zeros(2))


# --------------------------------------------------------------------------------------
# The two halves tied together: guided miss covariance == Phi P Phi.T
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_guided_terminal_miss_covariance_matches_the_linear_prediction():
    # The covariance result and the guidance result are two statements about the same Phi,
    # and this is the test that refuses to let them drift apart: fly many guided runs with
    # navigation error of covariance P and the sample covariance of the terminal misses must
    # be Phi P Phi.T. It is the campaign's terminal-error percentiles in miniature.
    n_runs = 3000
    covariance = _diagonal(1.0, 2.0e-3)
    model = NavigationErrorModel(noise_covariance=covariance)
    rng = np.random.default_rng(56789)
    misses = np.array(
        [
            _guided(_START_HILL + model.draw_noise(rng), _START_HILL).terminal_state_hill
            - _TARGET_HILL
            for _ in range(n_runs)
        ]
    )
    predicted = terminal_error_covariance(covariance, _N_RAD_S, _TOF_S)
    z = _covariance_z_scores(np.cov(misses, rowvar=False, ddof=1), predicted, n_runs)
    assert float(np.max(z)) < _Z_BOUND, f"terminal miss covariance off by {np.max(z):.2f} sigma"


def _redrawn_bias_looks(
    model: NavigationErrorModel, truth: np.ndarray, rng: np.random.Generator, m_looks: int
) -> list[np.ndarray]:
    """Return ``m_looks`` estimates with the bias **redrawn every look** -- the defect.

    Differs from :func:`_averaged_looks` by the position of ``begin_run`` and by nothing
    else. It consumes the generator in exactly the same order, so at ``m_looks == 1`` the
    two functions return bitwise identical values; the models separate only when more than
    one look is combined.
    """
    return [model.begin_run(rng).estimate(truth, rng) for _ in range(m_looks)]


def _guided_mean_square_miss(
    model: NavigationErrorModel,
    looks: object,
    m_looks: int,
    n_runs: int,
    seed: int,
) -> float:
    """Return the mean squared terminal position miss over ``n_runs`` guided runs."""
    rng = np.random.default_rng(seed)
    squared = np.empty(n_runs)
    for index in range(n_runs):
        estimate = np.mean(looks(model, _START_HILL, rng, m_looks), axis=0)  # type: ignore[operator]
        squared[index] = _guided(estimate, _START_HILL).terminal_position_error_m ** 2
    return float(np.mean(squared))


@pytest.mark.unit
def test_a_constant_bias_gives_a_different_terminal_error_than_white_noise():
    # The headline complement, and the reason NavigationSolution holds the bias as a frozen
    # field. ONE model, half its estimation-error covariance in a per-run bias and half in
    # white noise, run two ways: with the bias drawn once per run (correct) and with it
    # redrawn at every look (the defect -- that is white noise wearing a bias's name).
    # Guidance uses an eight-look average of the navigation solution, which is what any real
    # filter does with more than one measurement.
    #
    # Closed form: the mean-square terminal position miss is trace(Sigma) with
    # Sigma = (Phi P_e Phi.T)[:3,:3]. Correct model: P_e = P_b + P_w/m = 0.5625 * P_total.
    # Defect: P_e = (P_b + P_w)/m = 0.125 * P_total. Predicted RMS ratio sqrt(4.5) = 2.1213.
    # The tolerance is the standard error of a mean of squared Gaussian norms,
    # sqrt(2 trace(Sigma**2) / n), which is derived rather than chosen. Measured at
    # n = 2000, m = 8: ratio 2.16 against the predicted 2.12, i.e. the defect *understates*
    # the terminal miss by a factor of two.
    total = _diagonal(2.0, 5.0e-3)
    half = total / 2.0
    model = NavigationErrorModel(noise_covariance=half, bias_covariance=half)
    m_looks = 8
    n_runs = 2000
    phi = cw_stm(_N_RAD_S, _TOF_S)

    measured_bias = _guided_mean_square_miss(model, _averaged_looks, m_looks, n_runs, 20260824)
    measured_white = _guided_mean_square_miss(model, _redrawn_bias_looks, m_looks, n_runs, 20260824)

    sigma_bias = (phi @ model.averaged_covariance(m_looks) @ phi.T)[:3, :3]
    sigma_white = (phi @ (model.total_covariance / m_looks) @ phi.T)[:3, :3]
    for measured, sigma, label in (
        (measured_bias, sigma_bias, "constant bias"),
        (measured_white, sigma_white, "redrawn bias"),
    ):
        expected = float(np.trace(sigma))
        standard_error = math.sqrt(2.0 * float(np.trace(sigma @ sigma)) / n_runs)
        assert abs(measured - expected) < _Z_BOUND * standard_error, (
            f"{label}: mean-square miss {measured:.4f} m^2 against predicted {expected:.4f}, "
            f"off by {abs(measured - expected) / standard_error:.2f} sigma"
        )

    ratio = math.sqrt(measured_bias / measured_white)
    predicted_ratio = math.sqrt(float(np.trace(sigma_bias)) / float(np.trace(sigma_white)))
    assert predicted_ratio == pytest.approx(math.sqrt(4.5), rel=1.0e-12)
    assert ratio == pytest.approx(predicted_ratio, rel=0.05)


@pytest.mark.unit
def test_the_two_bias_models_are_bitwise_identical_at_one_look():
    # The knife edge of the previous test, and it is sharper than a statistical statement:
    # at m = 1 the correct model and the defect consume the generator identically and return
    # *bitwise equal* estimates. A version of the test above written without averaging would
    # therefore pass with the bias modelled as white noise, no matter how many runs it made.
    model = NavigationErrorModel(
        noise_covariance=_diagonal(2.0, 5.0e-3), bias_covariance=_diagonal(2.0, 5.0e-3)
    )
    truth = np.zeros(STATE_DIMENSION)
    correct = _averaged_looks(model, truth, np.random.default_rng(1234), 1)
    defect = _redrawn_bias_looks(model, truth, np.random.default_rng(1234), 1)
    assert np.array_equal(correct[0], defect[0])
    assert np.allclose(model.averaged_covariance(1), model.total_covariance)

    # And at eight looks the closed forms have already parted company by a factor of 4.5 in
    # covariance, which is what the previous test measures.
    assert np.allclose(model.averaged_covariance(8), model.total_covariance * 0.5625)
