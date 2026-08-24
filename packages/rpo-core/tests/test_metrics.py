"""Mission metrics: closed forms, knife-edge complements, and an exact JSON round trip.

Nothing here is a golden number read back out of this implementation. The scalar metrics
are asserted against arithmetic that can be done on paper:

* The **straight-line pass**. The chaser holds ``x = +30 m`` and ``z = +40 m`` while
  sliding along-track at a constant 2 m/s from ``y = -400 m``, so

      |rho(t)|**2 = 30**2 + 40**2 + (2t - 400)**2 = 2500 + (2t - 400)**2

  which is *exactly* quadratic in ``t``. Closest approach is therefore at exactly
  ``t = 200 s`` and exactly 50 m -- the 30-40-50 right triangle -- and the parabolic
  sub-sample refinement in ``rpo_core.constraints`` recovers both to machine precision. The
  sample grid is chosen so that ``t = 200 s`` is *not* a sample, which is what makes the
  sampled and refined minima measurably different numbers rather than the same one twice.

* The **impulses**, ``(3, 4, 0)`` and ``(0, -6, 8)`` m/s: a 3-4-5 and a 6-8-10 triangle, so
  the magnitudes are exactly 5 and 10 and the budget is exactly 15 m/s in binary floating
  point.

* The **terminal errors**, a 3-4-12-13 position difference and a 6-8-10 velocity
  difference, again exact.

* The **half-period V-bar hop**, whose total delta-v has the closed form ``n*dy/2`` derived
  in the ``two_impulse_transfer`` docstring's STM partition. That one is a genuine external
  check: it comes from the CW state transition matrix at ``tau = pi``, not from this module.

Where a tolerance appears, the comment above it states the value measured on this machine
before the bound was chosen, and the headroom the bound leaves.
"""

import json
import math
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rpo_core.config import ScenarioConfig, load_scenario
from rpo_core.constraints import (
    ApproachCorridor,
    ApproachEllipsoid,
    ClosingVelocityLimit,
    KeepOutSphere,
    SafetyReport,
    evaluate_constraints,
)
from rpo_core.exceptions import RpoCoreError
from rpo_core.metrics import (
    METRICS_FILENAME,
    METRICS_SCHEMA_VERSION,
    Burn,
    MetricsError,
    TrajectoryMetrics,
    compute_metrics,
    read_metrics,
    write_metrics,
)
from rpo_core.relative.cw import propagate_cw, two_impulse_transfer
from rpo_core.relative.nonlinear import CW_ERROR_SAFETY_FACTOR, estimated_cw_error_m

# --------------------------------------------------------------------------------------
# The straight-line pass -- every constant below is read off the 30-40-50 identity
# --------------------------------------------------------------------------------------

PASS_X_M = 30.0
PASS_Z_M = 40.0
PASS_SPEED_M_S = 2.0
PASS_Y0_M = -400.0
PASS_DURATION_S = 400.0
PASS_CLOSEST_TIME_S = 200.0
PASS_CLOSEST_RANGE_M = 50.0

#: 24 samples over 400 s puts the closest approach at fractional index 11.5 -- squarely
#: between two samples, and symmetric about them. If it landed *on* a sample the sampled and
#: refined minima would coincide and the test that they differ would pass for the wrong
#: reason.
PASS_N_SAMPLES = 24
_STEP_S = PASS_DURATION_S / (PASS_N_SAMPLES - 1)

#: |rho| at the nearer of the two samples bracketing the true minimum:
#: sqrt(2500 + (400/23)**2). Analytic, from the identity above.
PASS_SAMPLED_MIN_RANGE_M = math.sqrt(2500.0 + (400.0 / 23.0) ** 2)

#: Initial separation, sqrt(30**2 + 40**2 + 400**2) = sqrt(162500).
PASS_INITIAL_RANGE_M = math.sqrt(162500.0)

#: Closing velocity is -(rho . rhodot)/|rho| = -2u/sqrt(2500 + u**2) with u = 2t - 400.
#: Its derivative in u is -5000 * (2500 + u**2)**(-3/2) < 0 everywhere, so the maximum is at
#: the smallest u -- the first sample, u = -400.
PASS_MAX_CLOSING_M_S = 2.0 * 400.0 / PASS_INITIAL_RANGE_M

DEPARTURE_DV_M_S = (3.0, 4.0, 0.0)  # 3-4-5: magnitude exactly 5
ARRIVAL_DV_M_S = (0.0, -6.0, 8.0)  # 6-8-10: magnitude exactly 10
TOTAL_DV_M_S = 15.0

#: Commanded terminal state chosen so the achieved-minus-commanded difference is
#: (3, 4, 12) in position -- a 3-4-12-13 quadruple -- and (0, 6, 8) in velocity.
COMMANDED_TERMINAL = (27.0, 396.0, 28.0, 0.0, -4.0, -8.0)
TERMINAL_POSITION_ERROR_M = 13.0
TERMINAL_VELOCITY_ERROR_M_S = 10.0

_SCENARIO: dict[str, Any] = {
    "name": "analytic_pass",
    "description": "straight-line 30-40-50 pass, used only to exercise the metrics builder",
    "orbit": {"altitude_m": 420_000.0, "inclination_deg": 51.6},
    "start_hold_point": {"name": "start", "position_hill_m": [0.0, -400.0, 0.0]},
    "target_hold_point": {"name": "finish", "position_hill_m": [0.0, 400.0, 0.0]},
    "constraints": {
        # 40 m sits comfortably inside the 50 m closest approach, so the nominal pass is
        # clean and the complement test can tighten it to breach.
        "keep_out_sphere_radius_m": 40.0,
        "approach_ellipsoid_semi_axes_m": [2000.0, 4000.0, 2000.0],
        "approach_cone_half_angle_deg": 10.0,
        "approach_cone_activation_range_m": 1000.0,
        "max_closing_velocity_m_s": 2.5,
        "max_closing_velocity_activation_range_m": 5000.0,
    },
    "maneuver": {"tof_periods": 0.5},
    "seed": 7,
}


def scenario(**constraint_overrides: float) -> ScenarioConfig:
    """Return the analytic scenario, optionally with constraint fields overridden."""
    raw = json.loads(json.dumps(_SCENARIO))
    raw["constraints"].update(constraint_overrides)
    return ScenarioConfig.model_validate(raw)


def straight_line_pass(
    n_samples: int = PASS_N_SAMPLES, duration_s: float = PASS_DURATION_S
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, states)`` for the 30-40-50 pass described in the module docstring."""
    times = np.linspace(0.0, duration_s, n_samples)
    states = np.zeros((n_samples, 6), dtype=np.float64)
    states[:, 0] = PASS_X_M
    states[:, 1] = PASS_Y0_M + PASS_SPEED_M_S * times
    states[:, 2] = PASS_Z_M
    states[:, 4] = PASS_SPEED_M_S
    return times, states


def receding_pass(n_samples: int = PASS_N_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, states)`` for a pass that only ever opens the range.

    Same 30 m radial / 40 m cross-track offsets, but starting at ``y = +50 m`` and moving
    further along-track, so ``|rho|`` increases monotonically and the closing velocity is
    negative at every sample.
    """
    times = np.linspace(0.0, 200.0, n_samples)
    states = np.zeros((n_samples, 6), dtype=np.float64)
    states[:, 0] = PASS_X_M
    states[:, 1] = 50.0 + PASS_SPEED_M_S * times
    states[:, 2] = PASS_Z_M
    states[:, 4] = PASS_SPEED_M_S
    return times, states


def safety_report(
    config: ScenarioConfig,
    times: np.ndarray,
    states: np.ndarray,
    *,
    with_corridor: bool = False,
) -> SafetyReport:
    """Evaluate the scenario's own constraints on a trajectory.

    The corridor is off by default: the pass runs from ``y = -400`` to ``y = +400`` and so
    leaves a V-bar cone by construction, which would put unrelated violations into every
    assertion about the keep-out sphere.
    """
    limits = config.constraints
    return evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(limits.keep_out_sphere_radius_m),
        ellipsoid=ApproachEllipsoid(limits.approach_ellipsoid_semi_axes_m),
        corridor=(
            ApproachCorridor(
                math.radians(limits.approach_cone_half_angle_deg),
                limits.approach_cone_activation_range_m,
            )
            if with_corridor
            else None
        ),
        closing_velocity=ClosingVelocityLimit(
            limits.max_closing_velocity_m_s,
            limits.max_closing_velocity_activation_range_m,
        ),
    )


def analytic_burns() -> list[Burn]:
    """Return the two hand-computed impulses."""
    return [
        Burn("depart", 0.0, DEPARTURE_DV_M_S),
        Burn("arrive", PASS_DURATION_S, ARRIVAL_DV_M_S),
    ]


def analytic_metrics(config: ScenarioConfig | None = None, **kwargs: Any) -> TrajectoryMetrics:
    """Return the metrics for the analytic pass under ``config`` (default scenario)."""
    config = scenario() if config is None else config
    times, states = straight_line_pass()
    return compute_metrics(
        config,
        times,
        states,
        analytic_burns(),
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
        **kwargs,
    )


def _walk_floats(value: object, path: str = "") -> list[tuple[str, float]]:
    """Return every float in a decoded JSON structure, with a dotted path to each."""
    found: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_walk_floats(item, f"{path}.{key}"))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.extend(_walk_floats(item, f"{path}[{index}]"))
    elif isinstance(value, float):
        found.append((path, value))
    return found


# --------------------------------------------------------------------------------------
# Delta-v budget
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_total_delta_v_is_the_exact_sum_of_impulse_magnitudes():
    metrics = analytic_metrics()
    # Exact, not approximate: 3-4-5 and 6-8-10 make sqrt(25) and sqrt(100), both exactly
    # representable, and 5.0 + 10.0 is exact. Measured error on this machine: 0.0.
    assert metrics.total_delta_v_m_s == TOTAL_DV_M_S
    assert [burn.magnitude_m_s for burn in metrics.burns] == [5.0, 10.0]


@pytest.mark.unit
def test_delta_v_sums_magnitudes_not_vectors():
    """Complement: two opposed impulses cost twice the magnitude, never zero."""
    config = scenario()
    times, states = straight_line_pass()
    burns = [Burn("out", 0.0, (0.0, 0.5, 0.0)), Burn("back", PASS_DURATION_S, (0.0, -0.5, 0.0))]
    metrics = compute_metrics(
        config,
        times,
        states,
        burns,
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    # The vector sum is exactly zero here, so a budget that summed vectors would report 0.0
    # and this assertion is the knife edge between the two implementations.
    assert metrics.total_delta_v_m_s == 1.0


@pytest.mark.unit
def test_dropping_a_burn_changes_the_total():
    """Complement: the total is not dominated by one burn and blind to the rest."""
    config = scenario()
    times, states = straight_line_pass()
    report = safety_report(config, times, states)
    both = compute_metrics(
        config,
        times,
        states,
        analytic_burns(),
        report,
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    for dropped in range(2):
        kept = [burn for index, burn in enumerate(analytic_burns()) if index != dropped]
        one = compute_metrics(
            config,
            times,
            states,
            kept,
            report,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )
        assert one.total_delta_v_m_s < both.total_delta_v_m_s
        assert len(one.burns) == 1
    assert both.total_delta_v_m_s == TOTAL_DV_M_S


@pytest.mark.unit
def test_a_plan_with_no_burns_has_a_zero_budget():
    config = scenario()
    times, states = straight_line_pass()
    metrics = compute_metrics(
        config,
        times,
        states,
        [],
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    assert metrics.total_delta_v_m_s == 0.0
    assert metrics.burns == ()


@pytest.mark.integration
def test_half_period_vbar_hop_matches_the_closed_form_total_delta_v():
    """``n*dy/2`` for a planar half-period hop, from the STM at ``tau = pi``.

    At ``tau = pi`` the in-plane ``Phi_rv`` block is ``[[0, 4/n], [-4/n, -3*pi/n]]``, so a
    pure along-track shortfall ``dy`` is met by ``dvx = -n*dy/4`` and, by the ``Phi_vv``
    block ``[[-1, 0], [0, -7]]``, an arrival impulse of the same magnitude. Total
    ``n*dy/2``. This is CW algebra, not a number this module produced.
    """
    config = load_scenario(Path(__file__).resolve().parents[3] / "configs" / "vbar_baseline.yaml")
    metrics = vbar_baseline_metrics(config)

    delta_y_m = abs(
        config.target_hold_point.position_hill_m[1] - config.start_hold_point.position_hill_m[1]
    )
    closed_form = config.orbit.mean_motion_rad_s * delta_y_m / 2.0
    # Measured relative error on this machine: 0.0 (the solve lands on the closed form to
    # the last bit). 1e-14 leaves ample headroom for a different BLAS.
    assert metrics.total_delta_v_m_s == pytest.approx(closed_form, rel=1e-14)
    # And the two impulses are equal in magnitude, which is the other half of the algebra.
    assert metrics.burns[0].magnitude_m_s == pytest.approx(closed_form / 2.0, rel=1e-14)
    assert metrics.burns[1].magnitude_m_s == pytest.approx(closed_form / 2.0, rel=1e-14)


def vbar_baseline_metrics(config: ScenarioConfig, n_samples: int = 241) -> TrajectoryMetrics:
    """Plan and score the shipped V-bar baseline scenario.

    The arrival impulse is folded into the final sample rather than appended as a new one:
    ``compute_metrics`` reads the terminal state off the last sample, so a trajectory that
    stops before the arrival burn would report the coast velocity as a terminal error.
    """
    n = config.orbit.mean_motion_rad_s
    tof_s = config.tof_s
    r0 = np.array(config.start_hold_point.position_hill_m)
    rf = np.array(config.target_hold_point.position_hill_m)
    dv1, dv2 = two_impulse_transfer(n, r0, np.zeros(3), rf, np.zeros(3), tof_s)

    times = np.linspace(0.0, tof_s, n_samples)
    states = np.array([propagate_cw(n, np.concatenate((r0, dv1)), float(t)) for t in times])
    states[-1, 3:] += dv2

    limits = config.constraints
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(limits.keep_out_sphere_radius_m),
        ellipsoid=ApproachEllipsoid(limits.approach_ellipsoid_semi_axes_m),
        corridor=ApproachCorridor(
            math.radians(limits.approach_cone_half_angle_deg),
            limits.approach_cone_activation_range_m,
        ),
        closing_velocity=ClosingVelocityLimit(
            limits.max_closing_velocity_m_s,
            limits.max_closing_velocity_activation_range_m,
        ),
    )
    return compute_metrics(
        config,
        times,
        states,
        [Burn("departure", 0.0, tuple(dv1)), Burn("arrival", float(times[-1]), tuple(dv2))],
        report,
        commanded_terminal_state_hill=np.concatenate((rf, np.zeros(3))),
    )


# --------------------------------------------------------------------------------------
# Time of flight and terminal accuracy
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_time_of_flight_is_the_span_of_the_time_base():
    metrics = analytic_metrics()
    assert metrics.time_of_flight_s == PASS_DURATION_S
    assert metrics.time_of_flight_periods == PASS_DURATION_S / scenario().orbit.orbital_period_s


@pytest.mark.unit
def test_time_of_flight_is_a_span_not_a_final_time():
    """Complement: shifting the epoch leaves the time of flight unchanged."""
    config = scenario()
    times, states = straight_line_pass()
    shifted = times + 1_000.0
    baseline = compute_metrics(
        config,
        times,
        states,
        [],
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    moved = compute_metrics(
        config,
        shifted,
        states,
        [],
        safety_report(config, shifted, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    assert moved.time_of_flight_s == baseline.time_of_flight_s == PASS_DURATION_S


@pytest.mark.unit
def test_terminal_errors_are_the_norms_of_the_state_difference():
    metrics = analytic_metrics()
    # 3-4-12-13 and 6-8-10, both exact in binary. Measured error: 0.0.
    assert metrics.terminal_position_error_m == TERMINAL_POSITION_ERROR_M
    assert metrics.terminal_velocity_error_m_s == TERMINAL_VELOCITY_ERROR_M_S
    assert metrics.commanded_terminal_state_hill == COMMANDED_TERMINAL
    assert metrics.achieved_terminal_state_hill == (30.0, 400.0, 40.0, 0.0, 2.0, 0.0)


@pytest.mark.unit
def test_terminal_errors_are_taken_at_the_last_sample_not_the_first():
    """Complement: comparing against the first sample would give a different number."""
    metrics = analytic_metrics()
    first_sample = np.array([PASS_X_M, PASS_Y0_M, PASS_Z_M, 0.0, PASS_SPEED_M_S, 0.0])
    error_at_first = float(np.linalg.norm(first_sample[:3] - np.array(COMMANDED_TERMINAL[:3])))
    assert not math.isclose(metrics.terminal_position_error_m, error_at_first, rel_tol=1e-3)


@pytest.mark.integration
def test_the_baseline_hop_arrives_on_target():
    """The planned two-impulse hop closes to the commanded hold point.

    Terminal errors measured on this machine: 2.84e-14 m and 0.0 m/s. The bounds below are
    1e-9 m and 1e-12 m/s -- roughly 3e4x and unbounded headroom -- chosen to be loose enough
    to survive a different linear-algebra backend and tight enough that a genuinely missed
    target (metres, not femtometres) fails.
    """
    config = load_scenario(Path(__file__).resolve().parents[3] / "configs" / "vbar_baseline.yaml")
    metrics = vbar_baseline_metrics(config)
    assert metrics.terminal_position_error_m < 1.0e-9
    assert metrics.terminal_velocity_error_m_s < 1.0e-12


# --------------------------------------------------------------------------------------
# Keep-out zone: sampled and refined must stay two numbers
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_sampled_and_refined_keep_out_minima_are_both_reported_and_differ():
    metrics = analytic_metrics()
    # Refined minimum: measured |50.0000000000000071 - 50| = 7.1e-15 m. The bound below is
    # 1e-12 m, ~140x headroom, and 3e12x tighter than the 2.94 m gap it has to resolve.
    assert metrics.min_koz_range_refined_m == pytest.approx(PASS_CLOSEST_RANGE_M, abs=1e-12)
    assert metrics.min_koz_time_refined_s == pytest.approx(PASS_CLOSEST_TIME_S, abs=1e-9)
    # Sampled minimum: the analytic value at the bracketing sample, sqrt(2500 + (400/23)^2)
    # = 52.938242 m. Measured error 0.0.
    assert metrics.min_koz_range_sampled_m == pytest.approx(PASS_SAMPLED_MIN_RANGE_M, abs=1e-12)
    # The two are 2.94 m apart. Collapsing them into one field cannot pass this.
    assert metrics.min_koz_range_sampled_m - metrics.min_koz_range_refined_m > 2.9
    assert metrics.koz_refinement_applied is True


@pytest.mark.unit
def test_keep_out_clearances_are_the_minima_less_the_radius():
    config = scenario()
    metrics = analytic_metrics(config)
    radius = config.constraints.keep_out_sphere_radius_m
    assert metrics.keep_out_radius_m == radius
    assert metrics.min_koz_clearance_refined_m == metrics.min_koz_range_refined_m - radius
    assert metrics.min_koz_clearance_sampled_m == metrics.min_koz_range_sampled_m - radius


@pytest.mark.unit
def test_the_sampled_minimum_lands_on_a_sample_and_the_refined_one_does_not():
    metrics = analytic_metrics()
    assert metrics.min_koz_time_sampled_s in metrics.series.times_s
    assert metrics.min_koz_time_refined_s not in metrics.series.times_s
    # The grid is symmetric about the true minimum, so whichever of the two bracketing
    # samples wins the tie sits half a step away: 200/23 = 8.6957 s.
    assert abs(metrics.min_koz_time_sampled_s - PASS_CLOSEST_TIME_S) == pytest.approx(
        _STEP_S / 2.0, rel=1e-12
    )


@pytest.mark.unit
def test_refinement_is_not_claimed_when_the_minimum_sits_on_an_endpoint():
    """Limiting case: a monotone approach has no bracketing triple, so nothing is refined."""
    config = scenario()
    # Truncate the pass before closest approach: |rho| is still falling at the last sample.
    times, states = straight_line_pass(n_samples=12, duration_s=150.0)
    metrics = compute_metrics(
        config,
        times,
        states,
        [],
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    assert metrics.koz_refinement_applied is False
    assert metrics.min_koz_range_refined_m == metrics.min_koz_range_sampled_m
    assert metrics.min_koz_time_refined_s == metrics.min_koz_time_sampled_s


@pytest.mark.unit
def test_refinement_converges_as_the_grid_is_tightened():
    """Convergence behaviour, not a single threshold: both minima approach 50 m."""
    config = scenario()
    sampled_errors: list[float] = []
    refined_errors: list[float] = []
    for n_samples in (24, 48, 96, 192):
        times, states = straight_line_pass(n_samples=n_samples)
        metrics = compute_metrics(
            config,
            times,
            states,
            [],
            safety_report(config, times, states),
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )
        sampled_errors.append(abs(metrics.min_koz_range_sampled_m - PASS_CLOSEST_RANGE_M))
        refined_errors.append(abs(metrics.min_koz_range_refined_m - PASS_CLOSEST_RANGE_M))
    assert all(later < earlier for earlier, later in pairwise(sampled_errors))
    # The refined minimum is already exact at the coarsest grid because |rho|**2 is exactly
    # quadratic in t here, so it is asserted flat rather than monotone. Measured worst
    # value across the four grids: 1.4e-14 m; the bound is 1e-11, ~700x headroom.
    assert max(refined_errors) < 1.0e-11
    # And the refinement is doing real work at every grid, not merely echoing the sample.
    assert all(
        refined < sampled for refined, sampled in zip(refined_errors, sampled_errors, strict=True)
    )


# --------------------------------------------------------------------------------------
# Closing velocity
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_maximum_closing_velocity_matches_the_projection_closed_form():
    metrics = analytic_metrics()
    # -2u/sqrt(2500 + u^2) is strictly decreasing in u, so the maximum is at the first
    # sample: 800/sqrt(162500) = 1.98455575 m/s. Measured error 0.0.
    assert metrics.max_closing_velocity_m_s == pytest.approx(PASS_MAX_CLOSING_M_S, abs=1e-12)
    assert metrics.max_closing_velocity_time_s == 0.0


@pytest.mark.unit
def test_closing_velocity_is_positive_when_approaching_and_negative_when_receding():
    """Complement: the sign convention is a knife edge, not a plateau.

    The receding pass departs from ``y = +50`` at 2 m/s with the same 30/40 offsets, so at
    the first sample ``|rho|**2 = 900 + 1600 + 2500 = 5000`` and the closing velocity is
    ``-(50 * 2) / sqrt(5000) = -sqrt(2)`` exactly. Even the *largest* closing velocity over
    the arc is negative, so a sign flip in the projection cannot pass both halves of this.
    """
    config = scenario()
    approaching = analytic_metrics(config)
    times, states = receding_pass()
    away = compute_metrics(
        config,
        times,
        states,
        [],
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    assert approaching.max_closing_velocity_m_s > 0.0
    assert away.max_closing_velocity_m_s < 0.0
    # Measured error against -sqrt(2): 0.0. Bound 1e-12, unbounded headroom.
    assert away.max_closing_velocity_m_s == pytest.approx(-math.sqrt(2.0), abs=1e-12)
    assert away.max_closing_velocity_time_s == 0.0


@pytest.mark.unit
def test_closing_velocity_is_none_when_the_limit_was_never_active():
    """Never-enforced and enforced-and-fine must not collapse into the same number."""
    # Activation at 10 m, and the pass never comes closer than 50 m.
    config = scenario(max_closing_velocity_activation_range_m=10.0)
    metrics = analytic_metrics(config)
    assert metrics.max_closing_velocity_m_s is None
    assert metrics.max_closing_velocity_time_s is None
    assert metrics.closing_velocity_activation_range_m == 10.0


@pytest.mark.unit
def test_the_violating_mask_marks_exactly_the_samples_above_the_limit():
    # 1.5 m/s sits below the 1.98 m/s peak, so an initial run of samples violates.
    config = scenario(max_closing_velocity_m_s=1.5)
    metrics = analytic_metrics(config)
    flagged = [
        index for index, flag in enumerate(metrics.series.closing_velocity_violating) if flag
    ]
    assert flagged, "expected some samples above the 1.5 m/s limit"
    # The mask is the same decision as the count in the record; they cannot drift.
    assert len(flagged) == metrics.constraint_violation_count
    for index, rate in enumerate(metrics.series.range_rate_m_s):
        assert metrics.series.closing_velocity_violating[index] == (-rate > 1.5)


# --------------------------------------------------------------------------------------
# Clohessy-Wiltshire error: the bound, never the estimate
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_cw_error_uses_the_conservative_bound_not_the_optimistic_estimate():
    """The single most consequential line in this module.

    ``estimated_cw_error_m`` under-predicts by up to 1.23x between 0.4 and 1.0 orbits --
    the regime of the flagship half-orbit hop. This asserts the recorded number is the
    bound, which is exactly ``CW_ERROR_SAFETY_FACTOR`` times the estimate.
    """
    config = scenario()
    metrics = analytic_metrics(config)
    optimistic = estimated_cw_error_m(
        metrics.cw_error_separation_m,
        config.orbit.semi_major_axis_m,
        metrics.cw_error_n_orbits,
    )
    assert metrics.cw_error_bound_m > optimistic
    # Measured absolute difference from CW_ERROR_SAFETY_FACTOR * estimate: 6.9e-18 m.
    # The relative bound below is 1e-15, ~7000x headroom on a 0.048 m quantity.
    assert metrics.cw_error_bound_m == pytest.approx(CW_ERROR_SAFETY_FACTOR * optimistic, rel=1e-15)
    assert metrics.cw_error_bound_m / optimistic == pytest.approx(CW_ERROR_SAFETY_FACTOR, rel=1e-12)


@pytest.mark.unit
def test_cw_error_scales_quadratically_in_separation():
    """Limiting behaviour of the measured 6*pi*rho^2/r law: doubling rho quadruples it."""
    config = scenario()
    bounds: list[float] = []
    for scale in (1.0, 2.0):
        times, states = straight_line_pass()
        scaled = states * scale
        metrics = compute_metrics(
            config,
            times,
            scaled,
            [],
            safety_report(config, times, scaled),
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )
        bounds.append(metrics.cw_error_bound_m)
    assert bounds[1] / bounds[0] == pytest.approx(4.0, rel=1e-12)


@pytest.mark.unit
def test_cw_error_separation_is_the_largest_sampled_range_not_the_hold_point_radius():
    """The transfer arc bulges away from the chord, so the hold-point radius understates it."""
    config = scenario()
    metrics = analytic_metrics(config)
    assert metrics.cw_error_separation_m == max(metrics.series.range_m)
    assert metrics.cw_error_separation_m == pytest.approx(PASS_INITIAL_RANGE_M, abs=1e-12)
    assert metrics.cw_error_separation_m > config.max_separation_m


@pytest.mark.unit
def test_cw_budget_is_the_documented_fraction_of_the_keep_out_radius():
    config = scenario()
    metrics = analytic_metrics(config)
    assert metrics.cw_error_budget_m == 0.025 * config.constraints.keep_out_sphere_radius_m
    assert metrics.cw_within_budget == (metrics.cw_error_bound_m <= metrics.cw_error_budget_m)
    assert metrics.cw_within_budget is True


# --------------------------------------------------------------------------------------
# Constraint outcome, and the complement that proves it is wired up
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_nominal_pass_is_clean():
    metrics = analytic_metrics()
    assert metrics.constraint_violation_count == 0
    assert metrics.first_violation_time_s is None
    assert metrics.all_constraints_satisfied is True
    assert metrics.min_koz_clearance_refined_m > 0.0


@pytest.mark.unit
def test_a_violated_keep_out_sphere_changes_the_metrics_and_nothing_else():
    """Complement: the same trajectory, a sphere it breaches, and the metrics must move.

    Only the keep-out radius changes between the two runs, so anything that moves is
    attributable to the violation and anything that does not move (delta-v, time of flight,
    terminal errors) is proven independent of it.
    """
    clean = analytic_metrics(scenario())
    breached = analytic_metrics(scenario(keep_out_sphere_radius_m=60.0))

    assert clean.all_constraints_satisfied is True
    assert breached.all_constraints_satisfied is False
    assert breached.constraint_violation_count > 0
    assert breached.first_violation_time_s is not None
    assert breached.min_koz_clearance_refined_m < 0.0
    # 60 m sphere, 50 m closest approach: exactly 10 m inside. Measured error 7.1e-15 m.
    assert breached.min_koz_clearance_refined_m == pytest.approx(-10.0, abs=1e-12)

    assert breached.total_delta_v_m_s == clean.total_delta_v_m_s
    assert breached.time_of_flight_s == clean.time_of_flight_s
    assert breached.terminal_position_error_m == clean.terminal_position_error_m
    assert breached.min_koz_range_refined_m == clean.min_koz_range_refined_m


@pytest.mark.unit
def test_a_violated_closing_velocity_limit_changes_the_metrics():
    """Complement, on the other constraint the record is required to carry."""
    clean = analytic_metrics(scenario())
    breached = analytic_metrics(scenario(max_closing_velocity_m_s=1.0))
    assert clean.constraint_violation_count == 0
    assert breached.constraint_violation_count > 0
    assert breached.first_violation_time_s == 0.0
    assert any(breached.series.closing_velocity_violating)
    # The peak closing velocity is a property of the trajectory, not of the limit.
    assert breached.max_closing_velocity_m_s == clean.max_closing_velocity_m_s


@pytest.mark.integration
def test_the_shipped_baseline_scenario_violates_its_own_approach_corridor():
    """A real finding, asserted so it cannot regress silently.

    A half-period V-bar hop of ``dy`` bulges radially by exactly ``dy/4`` -- the departure
    impulse is ``n*dy/4`` and the radial response is ``(sin(n t)/n) * dv_x``, peaking at
    ``dv_x/n``. For the shipped 750 m hop that is 187.5 m, at ``y = -625 m``, which is
    16.7 deg off V-bar against a 10 deg corridor. The comment in
    ``configs/vbar_baseline.yaml`` estimates the bulge at "~40 m" and concludes "tight, but
    not violated"; that comment is wrong.
    """
    config = load_scenario(Path(__file__).resolve().parents[3] / "configs" / "vbar_baseline.yaml")
    metrics = vbar_baseline_metrics(config)
    assert metrics.max_corridor_angle_rad is not None
    assert metrics.max_corridor_angle_rad > metrics.corridor_half_angle_rad
    assert metrics.all_constraints_satisfied is False
    assert metrics.constraint_violation_count > 0
    # Bulge dy/4 = 187.5 m exactly; the CW solve reproduces it to the last bit here.
    radial_excursion_m = max(abs(position[0]) for position in metrics.series.position_hill_m)
    delta_y_m = abs(
        config.target_hold_point.position_hill_m[1] - config.start_hold_point.position_hill_m[1]
    )
    assert radial_excursion_m == pytest.approx(delta_y_m / 4.0, rel=1e-12)


@pytest.mark.unit
def test_optional_constraint_results_are_none_when_not_evaluated():
    config = scenario()
    times, states = straight_line_pass()
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(config.constraints.keep_out_sphere_radius_m),
        closing_velocity=ClosingVelocityLimit(
            config.constraints.max_closing_velocity_m_s,
            config.constraints.max_closing_velocity_activation_range_m,
        ),
    )
    metrics = compute_metrics(
        config, times, states, [], report, commanded_terminal_state_hill=COMMANDED_TERMINAL
    )
    assert metrics.max_corridor_angle_rad is None
    assert metrics.max_ellipsoid_quadratic_form is None
    # The corridor *geometry* is still recorded, because the figure draws it either way.
    assert metrics.corridor_half_angle_rad == pytest.approx(math.radians(10.0), rel=1e-15)


# --------------------------------------------------------------------------------------
# JSON: exact round trip
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_json_round_trip_reproduces_every_float_bitwise(tmp_path: Path):
    """Not "close": identical to the last bit, compared as raw IEEE-754 doubles."""
    metrics = analytic_metrics()
    write_metrics(tmp_path, metrics)
    restored = read_metrics(tmp_path)

    assert restored == metrics

    original_floats = _walk_floats(metrics.to_json_dict())
    restored_floats = _walk_floats(restored.to_json_dict())
    assert [path for path, _ in original_floats] == [path for path, _ in restored_floats]
    assert original_floats, "expected the record to contain floats at all"
    for (path, before), (_, after) in zip(original_floats, restored_floats, strict=True):
        assert struct.pack("<d", before) == struct.pack("<d", after), f"{path} lost precision"


@pytest.mark.unit
def test_json_round_trip_survives_awkward_magnitudes(tmp_path: Path):
    """Complement: values that a %.6g or a round(x, 6) would visibly damage."""
    config = scenario()
    times, states = straight_line_pass()
    # 17 significant digits and a subnormal-adjacent exponent: nothing short of repr
    # precision reproduces these.
    burns = [
        Burn("tiny", 0.0, (1.2345678901234567e-11, 0.0, 0.0)),
        Burn("odd", PASS_DURATION_S, (0.0, math.pi * 1e-3, math.e * 1e-5)),
    ]
    metrics = compute_metrics(
        config,
        times,
        states,
        burns,
        safety_report(config, times, states),
        commanded_terminal_state_hill=COMMANDED_TERMINAL,
    )
    write_metrics(tmp_path, metrics)
    restored = read_metrics(tmp_path)
    for before, after in zip(metrics.burns, restored.burns, strict=True):
        assert struct.pack("<d", before.magnitude_m_s) == struct.pack("<d", after.magnitude_m_s)
        assert before.delta_v_hill_m_s == after.delta_v_hill_m_s
    assert struct.pack("<d", metrics.total_delta_v_m_s) == struct.pack(
        "<d", restored.total_delta_v_m_s
    )


@pytest.mark.unit
def test_write_metrics_creates_metrics_json_and_returns_its_path(tmp_path: Path):
    metrics = analytic_metrics()
    destination = write_metrics(tmp_path / "run", metrics)
    assert destination == tmp_path / "run" / METRICS_FILENAME
    assert destination.stat().st_size > 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == METRICS_SCHEMA_VERSION
    assert payload["scenario_name"] == "analytic_pass"
    assert payload["seed"] == 7
    assert len(payload["series"]["times_s"]) == PASS_N_SAMPLES


@pytest.mark.unit
def test_metrics_json_is_strict_json_with_no_nan_tokens(tmp_path: Path):
    """A record containing NaN is unreadable by anything but Python's own decoder."""
    # The corridor was not evaluated here, so max_corridor_angle_rad is a genuine "no value".
    metrics = analytic_metrics()
    text = write_metrics(tmp_path, metrics).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "null" in text  # the undefined metrics are recorded as null, not as NaN
    json.loads(text, parse_constant=_reject_constant)


def _reject_constant(token: str) -> float:
    raise AssertionError(f"metrics.json contained the non-standard JSON constant {token!r}")


@pytest.mark.unit
def test_the_same_inputs_produce_byte_identical_records(tmp_path: Path):
    """A1's acceptance criterion, restated where it can actually be checked."""
    first = write_metrics(tmp_path / "a", analytic_metrics()).read_bytes()
    second = write_metrics(tmp_path / "b", analytic_metrics()).read_bytes()
    assert first == second


@pytest.mark.unit
def test_the_config_hash_and_seed_identify_the_run():
    config = scenario()
    metrics = analytic_metrics(config)
    from rpo_core.config import config_hash

    assert metrics.config_hash == config_hash(config)
    assert metrics.seed == config.seed
    assert analytic_metrics(config, seed=99).seed == 99
    # A different scenario must hash differently, otherwise the identity is decorative.
    assert analytic_metrics(scenario(keep_out_sphere_radius_m=41.0)).config_hash != (
        metrics.config_hash
    )


# --------------------------------------------------------------------------------------
# Every raise path
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"label": "  "}, "non-empty name"),
        ({"time_s": math.nan}, "non-finite time_s"),
        ({"delta_v_hill_m_s": (1.0, 2.0)}, "three entries"),
        ({"delta_v_hill_m_s": (1.0, 2.0, math.inf)}, "must be finite"),
        ({"delta_v_hill_m_s": ("a", "b", "c")}, "three finite numbers"),
    ],
)
def test_burn_rejects_malformed_input(kwargs: dict[str, Any], message: str):
    fields: dict[str, Any] = {"label": "burn", "time_s": 0.0, "delta_v_hill_m_s": (1.0, 0.0, 0.0)}
    fields.update(kwargs)
    with pytest.raises(MetricsError, match=message):
        Burn(**fields)


@pytest.mark.unit
def test_compute_metrics_rejects_a_trajectory_with_the_wrong_state_width():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match=r"shape \(N, 6\)"):
        compute_metrics(
            config,
            times,
            states[:, :3],
            [],
            good,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )


@pytest.mark.unit
def test_compute_metrics_rejects_a_single_sample_trajectory():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match="at least two samples"):
        compute_metrics(
            config,
            times[:1],
            states[:1],
            [],
            good,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )


@pytest.mark.unit
def test_compute_metrics_rejects_mismatched_lengths():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match="same length"):
        compute_metrics(
            config, times[:-1], states, [], good, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
def test_compute_metrics_rejects_a_two_dimensional_time_base():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match=r"times_s must have shape \(N,\)"):
        compute_metrics(
            config,
            times.reshape(-1, 1),
            states,
            [],
            good,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )


@pytest.mark.unit
def test_compute_metrics_rejects_non_finite_times():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    broken = times.copy()
    broken[3] = math.nan
    with pytest.raises(MetricsError, match="times_s must be finite"):
        compute_metrics(
            config, broken, states, [], good, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
def test_compute_metrics_rejects_non_finite_states():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    broken = states.copy()
    broken[5, 2] = math.inf
    with pytest.raises(MetricsError, match="states_hill must be finite"):
        compute_metrics(
            config, times, broken, [], good, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
def test_compute_metrics_rejects_a_time_base_that_runs_backwards():
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    broken = times.copy()
    broken[4], broken[5] = broken[5], broken[4]
    with pytest.raises(MetricsError, match="strictly increasing"):
        compute_metrics(
            config, broken, states, [], good, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("commanded", "message"),
    [
        ((1.0, 2.0, 3.0), r"shape \(6,\)"),
        ((1.0, 2.0, 3.0, 4.0, 5.0, math.nan), "must be finite"),
    ],
)
def test_compute_metrics_rejects_a_malformed_commanded_state(
    commanded: tuple[float, ...], message: str
):
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match=message):
        compute_metrics(config, times, states, [], good, commanded_terminal_state_hill=commanded)


@pytest.mark.unit
def test_compute_metrics_rejects_a_report_without_a_keep_out_result():
    config = scenario()
    times, states = straight_line_pass()
    report = evaluate_constraints(
        times,
        states,
        closing_velocity=ClosingVelocityLimit(
            config.constraints.max_closing_velocity_m_s,
            config.constraints.max_closing_velocity_activation_range_m,
        ),
    )
    with pytest.raises(MetricsError, match="no keep_out result"):
        compute_metrics(
            config, times, states, [], report, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
def test_compute_metrics_rejects_a_report_without_a_closing_velocity_result():
    config = scenario()
    times, states = straight_line_pass()
    report = evaluate_constraints(
        times, states, keep_out=KeepOutSphere(config.constraints.keep_out_sphere_radius_m)
    )
    with pytest.raises(MetricsError, match="no closing_velocity result"):
        compute_metrics(
            config, times, states, [], report, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
def test_compute_metrics_rejects_a_report_from_a_different_trajectory():
    """Pairing a stale report with a new trajectory is a silent-wrong-answer machine."""
    config = scenario()
    long_times, long_states = straight_line_pass(n_samples=PASS_N_SAMPLES * 4)
    stale = safety_report(config, long_times, long_states)
    short_times, short_states = straight_line_pass()
    with pytest.raises(MetricsError, match="does not describe this trajectory"):
        compute_metrics(
            config,
            short_times,
            short_states,
            [],
            stale,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )


@pytest.mark.unit
def test_compute_metrics_rejects_a_closing_velocity_result_from_a_different_limit():
    """The shaded spans in a figure and the count in the record must be one decision."""
    config = scenario()
    times, states = straight_line_pass()
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(config.constraints.keep_out_sphere_radius_m),
        # 1.0 m/s, not the scenario's 2.5 m/s.
        closing_velocity=ClosingVelocityLimit(1.0, 5000.0),
    )
    with pytest.raises(MetricsError, match="different limit than the scenario"):
        compute_metrics(
            config, times, states, [], report, commanded_terminal_state_hill=COMMANDED_TERMINAL
        )


@pytest.mark.unit
@pytest.mark.parametrize("burn_time_s", [-1.0, PASS_DURATION_S + 1.0])
def test_compute_metrics_rejects_a_burn_outside_the_trajectory_span(burn_time_s: float):
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match="outside the trajectory span"):
        compute_metrics(
            config,
            times,
            states,
            [Burn("stray", burn_time_s, (1.0, 0.0, 0.0))],
            good,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
        )


@pytest.mark.unit
@pytest.mark.parametrize("axis", [(0.0, 0.0), (0.0, -1.0, math.nan)])
def test_compute_metrics_rejects_a_malformed_corridor_axis(axis: tuple[float, ...]):
    config = scenario()
    times, states = straight_line_pass()
    good = safety_report(config, times, states)
    with pytest.raises(MetricsError, match="corridor_axis_hill"):
        compute_metrics(
            config,
            times,
            states,
            [],
            good,
            commanded_terminal_state_hill=COMMANDED_TERMINAL,
            corridor_axis_hill=axis,
        )


@pytest.mark.unit
def test_to_json_dict_refuses_a_non_finite_metric():
    metrics = analytic_metrics()
    import dataclasses

    poisoned = dataclasses.replace(metrics, total_delta_v_m_s=math.nan)
    with pytest.raises(MetricsError, match="total_delta_v_m_s"):
        poisoned.to_json_dict()


@pytest.mark.unit
def test_write_metrics_reports_a_path_it_cannot_write(tmp_path: Path):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied", encoding="utf-8")
    with pytest.raises(MetricsError, match="cannot write"):
        write_metrics(blocked, analytic_metrics())


@pytest.mark.unit
def test_read_metrics_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(MetricsError, match="cannot read"):
        read_metrics(tmp_path / "nowhere")


@pytest.mark.unit
def test_read_metrics_rejects_non_json(tmp_path: Path):
    (tmp_path / METRICS_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(MetricsError, match="not valid JSON"):
        read_metrics(tmp_path)


@pytest.mark.unit
def test_read_metrics_rejects_a_json_array(tmp_path: Path):
    (tmp_path / METRICS_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(MetricsError, match="JSON object at the top level"):
        read_metrics(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("version", [None, 0, METRICS_SCHEMA_VERSION + 1])
def test_read_metrics_rejects_a_foreign_schema_version(tmp_path: Path, version: int | None):
    payload = analytic_metrics().to_json_dict()
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    (tmp_path / METRICS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MetricsError, match="schema_version"):
        read_metrics(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("total_delta_v_m_s"),
        lambda payload: payload.__setitem__("commanded_terminal_state_hill", [1.0, 2.0]),
        lambda payload: payload["series"].__setitem__("times_s", ["not-a-number"]),
    ],
    ids=["missing-field", "wrong-length-vector", "non-numeric-series"],
)
def test_read_metrics_rejects_a_malformed_body(tmp_path: Path, mutate: Any):
    payload = analytic_metrics().to_json_dict()
    mutate(payload)
    (tmp_path / METRICS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MetricsError, match="malformed"):
        read_metrics(tmp_path)


@pytest.mark.unit
def test_metrics_error_is_an_rpo_core_error():
    """Callers catch RpoCoreError; a metrics failure must not escape that net."""
    assert issubclass(MetricsError, RpoCoreError)
    assert issubclass(MetricsError, ValueError)
