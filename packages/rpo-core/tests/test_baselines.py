"""Three rendezvous baselines, checked against closed forms and an external oracle.

The load-bearing tests in this file are the two that decide whether the comparison table can
be trusted at all:

* :func:`test_phasing_is_cheaper_and_slower_than_lambert` -- the known direction of the
  Δv-versus-time trade. If a phasing profile ever came out costing more Δv *and* less time
  than a direct Lambert transfer on the same problem, the implementation would be wrong; the
  correct response is to find the error, not to relax the assertion.
* :func:`test_cw_terminal_error_follows_the_measured_rho_squared_law` -- the CW baseline's
  terminal miss, measured under nonlinear dynamics, must obey the ``6*pi*rho**2/r`` law from
  ``docs/cw_validity.md``. That is what ties this module's error column back to the measured
  validity study rather than to a number this code produced about itself.

Two properties keep the checks genuine rather than self-referential:

* The Hohmann closed form is checked against vis-viva speeds on the transfer ellipse, and
  the Hohmann *time* is checked by propagating the burn under
  :func:`~rpo_core.propagate.propagate_two_body` and confirming the arrival radius. The
  second shares no code with the formula it validates.
* The CW baseline's terminal error is re-derived through
  :func:`~rpo_core.relative.nonlinear.propagate_relative_nonlinear`, which is a different
  code path from :func:`rpo_core.baselines._fly_impulse_schedule`. A scoring function that
  quietly evaluated CW under CW would pass every other test in this file and fail that one.

Numbers quoted in tolerance comments were measured on this machine before the bound was
chosen; the headroom is stated in each case.
"""

import math

import numpy as np
import pytest
from rpo_core.baselines import (
    DEFAULT_CW_TOLERANCE_M,
    BaselineResult,
    Method,
    PhasingGeometryError,
    RendezvousProblem,
    Validity,
    cw_two_impulse_baseline,
    hohmann_delta_v_m_s,
    hohmann_transfer_time_s,
    lambert_baseline,
    phasing_baseline,
)
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from rpo_core.exceptions import (
    InfeasibleTransferError,
    SingularTransferTimeError,
)
from rpo_core.propagate import propagate_two_body
from rpo_core.relative.cw import two_impulse_transfer
from rpo_core.relative.nonlinear import (
    CW_ERROR_COEFFICIENT,
    conservative_cw_error_bound_m,
    propagate_relative_nonlinear,
)

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
V_CIRCULAR = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
N_RAD_S = mean_motion_rad_s(A_ISS_M)
PERIOD_S = orbital_period_s(A_ISS_M)

# Exactly circular target inclined 51.6 deg -- the same reference the CW validity study uses,
# so linearisation error is isolated from eccentricity error.
_INC = math.radians(51.6)
R_TARGET = np.array([A_ISS_M, 0.0, 0.0])
V_TARGET = V_CIRCULAR * np.array([0.0, math.cos(_INC), math.sin(_INC)])
AT_REST = np.zeros(3)

#: A transfer time away from both singular families: not a whole period (in-plane rank loss)
#: and not a half period (cross-track rank loss).
GENERIC_TOF_S = 0.4 * PERIOD_S


def hop_problem(separation_m: float, tof_s: float = GENERIC_TOF_S) -> RendezvousProblem:
    """Return a V-bar hop from ``-rho`` to ``-rho/4``, both at rest in the Hill frame."""
    return RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -separation_m, 0.0]),
        AT_REST,
        np.array([0.0, -0.25 * separation_m, 0.0]),
        AT_REST,
        tof_s,
    )


def three_dimensional_problem(tof_s: float = GENERIC_TOF_S) -> RendezvousProblem:
    """Return a transfer with radial and cross-track content in both endpoints."""
    return RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([100.0, -1000.0, 50.0]),
        AT_REST,
        np.array([-30.0, -250.0, -20.0]),
        AT_REST,
        tof_s,
    )


# --------------------------------------------------------------------------------------
# Hohmann closed form
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("r1_m", "r2_m"),
    [
        (A_ISS_M, A_ISS_M + 100.0e3),
        (A_ISS_M, A_ISS_M - 1000.0),
        (7.0e6, 4.2164e7),
    ],
)
def test_hohmann_delta_v_matches_vis_viva_on_the_transfer_ellipse(r1_m, r2_m):
    """The closed form must equal the vis-viva speed difference at each apsis.

    Independent because vis-viva is computed here from the transfer semi-major axis and the
    two radii, sharing no expression with ``sqrt(mu/r1)*(sqrt(2*r2/(r1+r2)) - 1)``.
    """
    dv1, dv2 = hohmann_delta_v_m_s(r1_m, r2_m)
    a_transfer = 0.5 * (r1_m + r2_m)
    v_transfer_1 = math.sqrt(MU_EARTH_M3_S2 * (2.0 / r1_m - 1.0 / a_transfer))
    v_transfer_2 = math.sqrt(MU_EARTH_M3_S2 * (2.0 / r2_m - 1.0 / a_transfer))
    expected_1 = v_transfer_1 - math.sqrt(MU_EARTH_M3_S2 / r1_m)
    expected_2 = math.sqrt(MU_EARTH_M3_S2 / r2_m) - v_transfer_2

    # Measured worst relative error over these three cases: 4.27e-12, on the 1 km transfer
    # where both forms suffer catastrophic cancellation. 1e-10 clears it with 23x headroom.
    assert abs(dv1 - expected_1) <= 1.0e-10 * abs(expected_1)
    assert abs(dv2 - expected_2) <= 1.0e-10 * abs(expected_2)


@pytest.mark.unit
def test_hohmann_degenerates_to_zero_and_half_a_period_when_the_radii_agree():
    """Limiting case: no transfer at all costs nothing and takes half a circular period."""
    dv1, dv2 = hohmann_delta_v_m_s(A_ISS_M, A_ISS_M)
    assert dv1 == 0.0
    assert dv2 == 0.0
    assert hohmann_transfer_time_s(A_ISS_M, A_ISS_M) == pytest.approx(0.5 * PERIOD_S, rel=0.0)


@pytest.mark.unit
def test_hohmann_delta_v_is_signed_by_transfer_direction():
    """Complement: raising and lowering must differ in sign, not only in magnitude.

    Without this the module could return absolute values and every Δv-budget test would
    still pass, while a caller flying the profile would burn the wrong way twice.
    """
    up = hohmann_delta_v_m_s(A_ISS_M, A_ISS_M + 50.0e3)
    down = hohmann_delta_v_m_s(A_ISS_M, A_ISS_M - 50.0e3)
    assert up[0] > 0.0 and up[1] > 0.0
    assert down[0] < 0.0 and down[1] < 0.0


@pytest.mark.integration
@pytest.mark.parametrize("r2_m", [A_ISS_M + 100.0e3, A_ISS_M - 50.0e3])
def test_hohmann_burn_actually_arrives_after_the_closed_form_time(r2_m):
    """External oracle: fly the departure impulse and check where it is at ``t_h``.

    :func:`~rpo_core.propagate.propagate_two_body` is already validated against energy and
    angular-momentum conservation, so this is a genuine check on the closed form rather than
    a restatement of it.
    """
    dv1, dv2 = hohmann_delta_v_m_s(A_ISS_M, r2_m)
    transfer_time_s = hohmann_transfer_time_s(A_ISS_M, r2_m)
    state0 = np.array([A_ISS_M, 0.0, 0.0, 0.0, V_CIRCULAR + dv1, 0.0])
    arrival = propagate_two_body(state0, np.array([0.0, transfer_time_s]))[-1]

    # Measured arrival-radius error: 3.23e-06 m (raise) and 2.48e-06 m (lower), integrator
    # limited at rtol = atol = 1e-12. 1e-4 m clears the worst with 31x headroom.
    assert abs(float(np.linalg.norm(arrival[:3])) - r2_m) <= 1.0e-4
    # And the arrival burn must circularise it. Measured speed residual: 3.37e-09 m/s.
    arrival_speed = float(np.linalg.norm(arrival[3:])) + dv2
    assert abs(arrival_speed - math.sqrt(MU_EARTH_M3_S2 / r2_m)) <= 1.0e-6


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"r1_m": 0.0, "r2_m": 7.0e6}, r"r1_m must be finite and strictly positive"),
        ({"r1_m": 7.0e6, "r2_m": -1.0}, r"r2_m must be finite and strictly positive"),
        ({"r1_m": 7.0e6, "r2_m": math.nan}, r"r2_m must be finite and strictly positive"),
        (
            {"r1_m": 7.0e6, "r2_m": 7.0e6, "mu_m3_s2": 0.0},
            r"mu_m3_s2 must be finite and strictly positive",
        ),
    ],
)
def test_hohmann_rejects_non_positive_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hohmann_delta_v_m_s(**kwargs)
    with pytest.raises(ValueError, match=match):
        hohmann_transfer_time_s(**kwargs)


# --------------------------------------------------------------------------------------
# Problem statement
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_problem_derives_mean_motion_and_separation_from_the_state():
    problem = hop_problem(10_000.0)
    assert problem.orbit_radius_m == pytest.approx(A_ISS_M, rel=1.0e-15)
    assert problem.n_rad_s == pytest.approx(N_RAD_S, rel=1.0e-12)
    assert problem.period_s == pytest.approx(PERIOD_S, rel=1.0e-12)
    # Separation is the worst point of the transfer, not the initial one.
    assert problem.separation_m == pytest.approx(10_000.0, rel=1.0e-15)


@pytest.mark.unit
def test_problem_separation_takes_the_larger_endpoint():
    """Complement: an outbound transfer must report the *terminal* separation.

    Reporting the initial one would understate the CW envelope by the square of the ratio,
    which is exactly the direction that turns an INVALID scenario into a VALID-looking one.
    """
    outbound = RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -250.0, 0.0]),
        AT_REST,
        np.array([0.0, -10_000.0, 0.0]),
        AT_REST,
        GENERIC_TOF_S,
    )
    assert outbound.separation_m == pytest.approx(10_000.0, rel=1.0e-15)


@pytest.mark.unit
def test_problem_chaser_state_round_trips_through_the_hill_frame():
    """Conservation-style check: converting out and back must recover the relative state."""
    problem = hop_problem(10_000.0)
    chaser = problem.chaser_state0_eci
    from rpo_core.frames import relative_state_eci_to_hill

    recovered = relative_state_eci_to_hill(
        problem.r_target0_eci_m, problem.v_target0_eci_m_s, chaser[:3], chaser[3:]
    )
    assert recovered[:3] == pytest.approx(problem.r0_hill_m, abs=1.0e-9)
    assert recovered[3:] == pytest.approx(problem.v0_hill_m_s, abs=1.0e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("r0_hill_m", np.zeros(2), r"r0_hill_m must have shape \(3,\)"),
        ("vf_hill_m_s", np.array([0.0, math.inf, 0.0]), r"vf_hill_m_s must be finite"),
        ("tof_s", -1.0, r"tof_s must be finite and strictly positive"),
        ("mu_m3_s2", 0.0, r"mu_m3_s2 must be finite and strictly positive"),
        ("r_target0_eci_m", np.zeros(3), r"non-zero position vector"),
        ("v_target0_eci_m_s", np.zeros(3), r"non-zero velocity vector"),
    ],
)
def test_problem_rejects_malformed_input(field, value, match):
    kwargs = {
        "r_target0_eci_m": R_TARGET,
        "v_target0_eci_m_s": V_TARGET,
        "r0_hill_m": np.array([0.0, -1000.0, 0.0]),
        "v0_hill_m_s": AT_REST,
        "rf_hill_m": np.array([0.0, -250.0, 0.0]),
        "vf_hill_m_s": AT_REST,
        "tof_s": GENERIC_TOF_S,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        RendezvousProblem(**kwargs)


@pytest.mark.unit
def test_problem_on_a_hyperbolic_target_refuses_to_invent_a_mean_motion():
    escape = RendezvousProblem(
        R_TARGET,
        2.0 * V_TARGET,
        np.array([0.0, -1000.0, 0.0]),
        AT_REST,
        np.array([0.0, -250.0, 0.0]),
        AT_REST,
        GENERIC_TOF_S,
    )
    with pytest.raises(ValueError, match=r"target orbit is not closed"):
        _ = escape.n_rad_s


# --------------------------------------------------------------------------------------
# The common result contract
# --------------------------------------------------------------------------------------


def _all_baselines(problem: RendezvousProblem) -> tuple[BaselineResult, ...]:
    return (
        phasing_baseline(problem),
        lambert_baseline(problem),
        cw_two_impulse_baseline(problem),
        cw_two_impulse_baseline(problem, correct=True),
    )


@pytest.mark.integration
def test_every_baseline_returns_the_same_contract():
    """One result type, and the total is the sum of the burns for all of them.

    A total that drifted from its own burn list is how a Δv column ends up quietly missing a
    burn, which is exactly the defect this assertion exists to catch.
    """
    results = _all_baselines(hop_problem(10_000.0))
    assert {result.method for result in results} == {
        Method.PHASING,
        Method.LAMBERT,
        Method.CW_TWO_IMPULSE,
        Method.CW_CORRECTED,
    }
    for result in results:
        assert result.total_delta_v_m_s == pytest.approx(
            sum(result.burn_delta_v_m_s), rel=1.0e-15, abs=1.0e-15
        )
        assert result.burn_count == len(result.burn_times_s)
        assert all(math.isfinite(value) for value in result.burn_delta_v_m_s)
        assert result.tof_s > 0.0
        assert result.min_separation_m > 0.0
        assert result.validity_detail.strip() != ""


@pytest.mark.integration
def test_burn_counts_are_four_for_phasing_and_two_for_the_fixed_time_methods():
    problem = hop_problem(10_000.0)
    assert phasing_baseline(problem).burn_count == 4
    assert lambert_baseline(problem).burn_count == 2
    assert cw_two_impulse_baseline(problem).burn_count == 2


# --------------------------------------------------------------------------------------
# The known trade direction
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("separation_m", [10_000.0, 1000.0, 250.0])
def test_phasing_is_cheaper_and_slower_than_lambert(separation_m):
    """The known direction of the trade, on identical problems.

    Phasing buys Δv with time: it uses a small radius offset held for several revolutions
    instead of a large impulse held for a fraction of one. If this ever inverted, the fault
    would be in the implementation -- most likely a drift radius solved with the wrong sign,
    or a phase budget that forgot the transfer arcs -- and the assertion is the detector, not
    the thing to loosen.

    Measured ratios: Δv 0.0445 / 0.0418 / 0.0416 and TOF 10.0 at 10 km / 1 km / 250 m.
    """
    problem = hop_problem(separation_m)
    phasing = phasing_baseline(problem)
    lambert = lambert_baseline(problem)

    assert phasing.total_delta_v_m_s < lambert.total_delta_v_m_s
    assert phasing.tof_s > lambert.tof_s
    # Bounds well inside the measured 0.0416-0.0445 and 9.9996-10.0 so the assertion is
    # about the direction of the trade, not about the exact revolution count.
    assert phasing.total_delta_v_m_s / lambert.total_delta_v_m_s < 0.25
    assert phasing.tof_s / lambert.tof_s > 4.0


@pytest.mark.integration
def test_more_drift_revolutions_buy_less_delta_v_and_more_time():
    """Monotone behaviour of the trade knob, which is what makes a Pareto front meaningful."""
    problem = hop_problem(10_000.0)
    results = [phasing_baseline(problem, drift_revolutions=k) for k in (1.0, 2.0, 3.0, 6.0)]
    delta_v = [result.total_delta_v_m_s for result in results]
    times = [result.tof_s for result in results]
    assert delta_v == sorted(delta_v, reverse=True)
    assert times == sorted(times)


@pytest.mark.integration
def test_phasing_drift_radius_offset_obeys_the_measured_one_over_k_plus_half_law():
    r"""``drift_radius_offset_m * (k + 1/2)`` is constant, and the half is not decoration.

    The phase is delivered over ``k`` drift revolutions at the full offset plus one further
    revolution split between two half-transfers, which sit at *half* the offset -- hence
    ``k + 1/2`` rather than ``k + 1``. Measured spread of the product across k = 1..12:
    1.39e-05 relative at 10 km separation. The bound below clears it with 7x headroom.

    The complement matters as much as the law: a ``k + 1`` normalisation is measurably
    non-constant, so this assertion is a knife edge rather than a plateau.
    """
    problem = hop_problem(10_000.0)
    revolutions = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    offsets = np.array(
        [
            phasing_baseline(problem, drift_revolutions=k).detail["drift_radius_offset_m"]
            for k in revolutions
        ]
    )
    product = offsets * (revolutions + 0.5)
    spread = float(product.max() - product.min()) / abs(float(product.mean()))
    assert spread < 1.0e-4

    wrong = offsets * (revolutions + 1.0)
    wrong_spread = float(wrong.max() - wrong.min()) / abs(float(wrong.mean()))
    assert wrong_spread > 100.0 * spread


# --------------------------------------------------------------------------------------
# Terminal error under nonlinear dynamics
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_terminal_errors_are_measured_under_nonlinear_dynamics_for_every_method():
    """CW must be *seen* to miss. A scoring path that flattered it would report ~1e-9 m.

    :func:`~rpo_core.relative.cw.two_impulse_transfer` reproduces its own commanded terminal
    state to about 1e-9 m by construction. The measured miss below is 73 m, seventy billion
    times larger, which is only possible if the scoring flight uses different dynamics from
    the design.
    """
    problem = hop_problem(10_000.0)
    cw = cw_two_impulse_baseline(problem)
    assert cw.terminal_position_error_m > 1.0

    # The linear model's own account of the same burn, for contrast.
    dv1, _ = two_impulse_transfer(
        problem.n_rad_s,
        problem.r0_hill_m,
        problem.v0_hill_m_s,
        problem.rf_hill_m,
        problem.vf_hill_m_s,
        problem.tof_s,
    )
    from rpo_core.relative.cw import propagate_cw

    linear_arrival = propagate_cw(
        problem.n_rad_s,
        np.concatenate((problem.r0_hill_m, problem.v0_hill_m_s + dv1)),
        problem.tof_s,
    )
    linear_miss = float(np.linalg.norm(linear_arrival[:3] - problem.rf_hill_m))
    assert linear_miss < 1.0e-6
    assert cw.terminal_position_error_m > 1.0e6 * linear_miss


@pytest.mark.integration
@pytest.mark.parametrize("separation_m", [10_000.0, 1000.0])
def test_cw_scoring_agrees_with_the_documented_nonlinear_oracle(separation_m):
    """The shared scoring flight must reproduce :func:`propagate_relative_nonlinear`.

    Different code path -- the baseline flies ECI impulses through
    :func:`~rpo_core.propagate.propagate_two_body` leg by leg, the oracle reconstructs the
    chaser from a Hill-frame relative state -- so agreement here is evidence the scoring
    function is measuring what it claims.
    """
    problem = hop_problem(separation_m)
    dv1, dv2 = two_impulse_transfer(
        problem.n_rad_s,
        problem.r0_hill_m,
        problem.v0_hill_m_s,
        problem.rf_hill_m,
        problem.vf_hill_m_s,
        problem.tof_s,
    )
    oracle = propagate_relative_nonlinear(
        problem.r_target0_eci_m,
        problem.v_target0_eci_m_s,
        np.concatenate((problem.r0_hill_m, problem.v0_hill_m_s + dv1)),
        np.array([0.0, problem.tof_s]),
    )[-1]
    oracle_position_error = float(np.linalg.norm(oracle[:3] - problem.rf_hill_m))
    oracle_velocity_error = float(np.linalg.norm(oracle[3:] + dv2 - problem.vf_hill_m_s))

    result = cw_two_impulse_baseline(problem)
    # Measured agreement: 0.0 m in position and 1.07e-13 m/s in velocity at both
    # separations. The bounds below are integrator-noise scale with wide headroom.
    assert result.terminal_position_error_m == pytest.approx(oracle_position_error, abs=1.0e-6)
    assert result.terminal_velocity_error_m_s == pytest.approx(oracle_velocity_error, abs=1.0e-9)


@pytest.mark.integration
def test_cw_terminal_error_follows_the_measured_rho_squared_law():
    r"""The CW miss must scale as ``rho**2 / r``, tying this module to ``docs/cw_validity.md``.

    The hop scales both endpoints together, so a quadratic law predicts exactly 100x between
    10 km and 1 km and exactly 16x between 1 km and 250 m. Measured: 100.019 and 16.0002.

    The ratio of the measured miss to ``6*pi*rho**2/r * n_orbits`` is also constant across
    separations -- measured 0.66172, 0.66159, 0.66158 -- which is the same law with a
    transfer-specific coefficient. It is below 1 because the law is quoted for a stationary
    along-track offset over whole orbits and this is a 0.4-orbit hop; the *scaling* is what
    the law asserts and the *coefficient* is what the transfer sets.

    Complement: a linear-in-``rho`` error law would give ratios of 10 and 4, which the
    assertions below reject by three orders of magnitude.
    """
    errors = {
        separation: cw_two_impulse_baseline(hop_problem(separation)).terminal_position_error_m
        for separation in (10_000.0, 1000.0, 250.0)
    }
    # Bound 5e-4 relative: measured departures from the exact ratios are 1.93e-4 and 1.30e-5,
    # so 2.6x headroom on the worse of the two.
    assert errors[10_000.0] / errors[1000.0] == pytest.approx(100.0, rel=5.0e-4)
    assert errors[1000.0] / errors[250.0] == pytest.approx(16.0, rel=5.0e-4)

    n_orbits = GENERIC_TOF_S / PERIOD_S
    coefficients = [
        errors[separation] / (CW_ERROR_COEFFICIENT * separation**2 / A_ISS_M * n_orbits)
        for separation in (10_000.0, 1000.0, 250.0)
    ]
    assert min(coefficients) == pytest.approx(max(coefficients), rel=1.0e-3)

    # And the whole point of the conservative bound: it must not be exceeded.
    for separation, error in errors.items():
        assert error < conservative_cw_error_bound_m(separation, A_ISS_M, n_orbits)


@pytest.mark.integration
def test_lambert_terminal_error_is_integrator_noise_and_separation_independent():
    """Complement to the CW law: an exact method's error must *not* scale with separation.

    If Lambert's miss also grew as ``rho**2`` it would mean the scoring flight, not the
    method, was producing the CW result's error signature.
    """
    errors = [
        lambert_baseline(hop_problem(separation)).terminal_position_error_m
        for separation in (10_000.0, 1000.0, 250.0)
    ]
    # Measured 4.45e-06, 4.33e-06, 4.56e-06 m -- flat across a 40x separation range, and
    # consistent with the 7.09e-04 m worst case rpo_core.lambert reports for itself.
    assert max(errors) < 1.0e-3
    assert max(errors) / min(errors) < 2.0


@pytest.mark.integration
def test_nonlinear_correction_buys_accuracy_for_a_small_delta_v_premium():
    """The corrected variant must beat the raw CW seed under the same scoring flight."""
    problem = hop_problem(10_000.0)
    raw = cw_two_impulse_baseline(problem)
    corrected = cw_two_impulse_baseline(problem, correct=True)

    # Measured: 73.39 m -> 2.52e-04 m, a factor of 2.9e5, for 0.135 % more delta-v.
    assert corrected.terminal_position_error_m < 1.0e-3
    assert raw.terminal_position_error_m / corrected.terminal_position_error_m > 1.0e4
    premium = corrected.total_delta_v_m_s / raw.total_delta_v_m_s - 1.0
    assert 0.0 < premium < 0.01
    assert corrected.detail["final_residual_m"] < corrected.detail["initial_residual_m"]


# --------------------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_cw_is_invalid_at_ten_kilometres_and_valid_at_two_hundred_and_fifty_metres():
    """The anti-misleading-benchmark check, at the level of a single result.

    ``docs/cw_validity.md`` measures the one-orbit CW error at 10 km as 277 m; against the
    5 m budget the conservative bound is 33x over. At 250 m it is 0.104 m, well inside. A
    build in which CW reported VALID at 10 km would produce a table showing it winning on
    Δv in a regime where its own error dwarfs the keep-out sphere.
    """
    far = cw_two_impulse_baseline(hop_problem(10_000.0))
    near = cw_two_impulse_baseline(hop_problem(250.0))

    assert far.validity is Validity.INVALID
    assert not far.is_valid
    assert near.validity is Validity.VALID
    assert near.is_valid

    assert far.cw_error_bound_m > DEFAULT_CW_TOLERANCE_M
    assert near.cw_error_bound_m < DEFAULT_CW_TOLERANCE_M
    assert "EXCEEDS" in far.validity_detail
    assert "10,000 m separation" in far.validity_detail


@pytest.mark.unit
def test_cw_validity_verdict_moves_with_the_budget_it_is_judged_against():
    """Complement: the flag is a comparison against a budget, not a hard-coded separation."""
    problem = hop_problem(10_000.0)
    generous = cw_two_impulse_baseline(problem, cw_tolerance_m=1000.0)
    strict = cw_two_impulse_baseline(hop_problem(250.0), cw_tolerance_m=0.01)
    assert generous.validity is Validity.VALID
    assert strict.validity is Validity.INVALID


@pytest.mark.integration
def test_phasing_is_invalid_when_the_transfer_leaves_its_premise():
    """Knife edge: the same target and the same hop, plus content phasing cannot deliver.

    CW stays VALID on the cross-track problem, so the two flags are answering different
    questions rather than both tracking separation.
    """
    coplanar = phasing_baseline(hop_problem(1000.0))
    assert coplanar.validity is Validity.VALID

    cross_track = RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -1000.0, 0.0]),
        AT_REST,
        np.array([0.0, -250.0, 80.0]),
        AT_REST,
        GENERIC_TOF_S,
    )
    result = phasing_baseline(cross_track)
    assert result.validity is Validity.INVALID
    assert "cross-track" in result.validity_detail
    assert cw_two_impulse_baseline(cross_track).validity is Validity.VALID

    moving = RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -1000.0, 0.0]),
        np.array([0.0, 0.0, 0.05]),
        np.array([0.0, -250.0, 0.0]),
        AT_REST,
        GENERIC_TOF_S,
    )
    moving_result = phasing_baseline(moving)
    assert moving_result.validity is Validity.INVALID
    assert "at-rest tolerance" in moving_result.validity_detail


@pytest.mark.integration
def test_phasing_is_invalid_on_an_eccentric_chaser():
    """The Hohmann closed form assumes circular endpoints; a 0.01 chaser is not one."""
    # A relative velocity large enough to make the chaser orbit measurably eccentric while
    # keeping the transfer coplanar, so the eccentricity branch is what fires.
    eccentric = RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -1000.0, 0.0]),
        np.array([20.0, 0.0, 0.0]),
        np.array([0.0, -250.0, 0.0]),
        AT_REST,
        GENERIC_TOF_S,
    )
    result = phasing_baseline(eccentric, rest_tol_m_s=100.0)
    assert result.validity is Validity.INVALID
    assert "eccentricity" in result.validity_detail
    assert result.detail["chaser_eccentricity"] > 1.0e-3


@pytest.mark.integration
def test_lambert_is_valid_regardless_of_separation():
    """Lambert makes no linearisation, so its premise cannot be broken by distance."""
    for separation in (250.0, 10_000.0, 200_000.0):
        result = lambert_baseline(hop_problem(separation))
        assert result.validity is Validity.VALID
    # It still reports the CW bound, so a reader can see how far the linear model is out.
    far = lambert_baseline(hop_problem(10_000.0))
    assert far.cw_error_bound_m > DEFAULT_CW_TOLERANCE_M


# --------------------------------------------------------------------------------------
# Minimum separation
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_minimum_separation_is_reported_and_never_exceeds_the_endpoints():
    """A Δv table that ignores where the trajectory went is its own misleading benchmark."""
    problem = hop_problem(10_000.0)
    for result in _all_baselines(problem):
        assert result.min_separation_m <= problem.separation_m
        assert result.min_separation_m > 0.0
    # Measured: the CW arc bulges outward and never comes inside the 2 500 m terminal point,
    # while the phasing profile dips 3.2 m below it on the way in.
    assert cw_two_impulse_baseline(problem).min_separation_m >= 2500.0
    assert phasing_baseline(problem).min_separation_m < 2500.0


# --------------------------------------------------------------------------------------
# Raise paths
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_phasing_raises_when_the_bracket_cannot_deliver_the_phase():
    with pytest.raises(PhasingGeometryError, match=r"no drift radius within") as info:
        phasing_baseline(
            hop_problem(10_000.0),
            drift_revolutions=0.0,
            max_drift_radius_fraction=1.0e-9,
        )
    error = info.value
    assert error.required_phase_rad == pytest.approx(7500.0 / A_ISS_M, rel=1.0e-9)
    assert len(error.achievable_phase_rad) == 2
    assert not (
        error.achievable_phase_rad[0] <= error.required_phase_rad <= error.achievable_phase_rad[1]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"drift_revolutions": -1.0}, r"drift_revolutions must be finite and >= 0"),
        ({"drift_revolutions": math.nan}, r"drift_revolutions must be finite and >= 0"),
        ({"max_drift_radius_fraction": 0.0}, r"max_drift_radius_fraction must lie in"),
        ({"max_drift_radius_fraction": 1.5}, r"max_drift_radius_fraction must lie in"),
    ],
)
def test_phasing_rejects_malformed_knobs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        phasing_baseline(hop_problem(1000.0), **kwargs)


@pytest.mark.unit
def test_cw_baseline_raises_at_a_whole_period_transfer_time():
    """F-2.2: the singular time must raise, not return a plausible-looking wrong answer."""
    with pytest.raises(SingularTransferTimeError, match=r"in-plane Phi_rv block is singular"):
        cw_two_impulse_baseline(hop_problem(1000.0, tof_s=PERIOD_S))


@pytest.mark.unit
def test_cw_baseline_raises_at_a_half_period_when_cross_track_is_requested():
    problem = RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -1000.0, 0.0]),
        AT_REST,
        np.array([0.0, -250.0, 80.0]),
        AT_REST,
        0.5 * PERIOD_S,
    )
    with pytest.raises(InfeasibleTransferError, match=r"cross-track targeting is rank-deficient"):
        cw_two_impulse_baseline(problem)


@pytest.mark.unit
def test_cw_baseline_refuses_to_silently_ignore_correction_arguments():
    """A tolerance passed with ``correct=False`` would otherwise do nothing at all."""
    with pytest.raises(ValueError, match=r"were supplied with correct=False"):
        cw_two_impulse_baseline(hop_problem(1000.0), tolerance_m=1.0e-6)


@pytest.mark.slow
@pytest.mark.integration
def test_correction_raises_rather_than_returning_a_non_converged_iterate():
    """The corrected variant must propagate a targeting failure, not absorb it."""
    from rpo_core.targeting import TargetingConvergenceError

    with pytest.raises(TargetingConvergenceError):
        cw_two_impulse_baseline(
            hop_problem(10_000.0), correct=True, tolerance_m=1.0e-15, max_iterations=2
        )


# --------------------------------------------------------------------------------------
# Half-period V-bar hop: the closed form from math-model.md M4
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_half_period_vbar_hop_matches_the_closed_form_total():
    r"""``total dv = n * dy / 2`` for a coplanar V-bar hop over exactly half a period.

    From ``docs/project1/math-model.md`` M4. Measured relative error through the baseline:
    2.63e-16, i.e. floating-point exact.
    """
    problem = hop_problem(1000.0, tof_s=0.5 * PERIOD_S)
    result = cw_two_impulse_baseline(problem)
    expected = N_RAD_S * 750.0 / 2.0
    assert result.total_delta_v_m_s == pytest.approx(expected, rel=1.0e-12)
    # Two equal purely-radial impulses, so the burn list must also be symmetric.
    assert result.burn_delta_v_m_s[0] == pytest.approx(result.burn_delta_v_m_s[1], rel=1.0e-12)


@pytest.mark.integration
def test_three_dimensional_problem_is_handled_by_every_fixed_time_method():
    """Radial and cross-track content must not be quietly dropped."""
    problem = three_dimensional_problem()
    lambert = lambert_baseline(problem)
    cw = cw_two_impulse_baseline(problem)
    assert lambert.terminal_position_error_m < 1.0e-3
    assert cw.total_delta_v_m_s > 0.0
    # The cross-track component is genuinely targeted, not ignored: dropping it would leave
    # a terminal miss of order the 70 m cross-track change itself.
    assert cw.terminal_position_error_m < 1.0
