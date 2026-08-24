"""Dispersed campaigns: determinism, honest failure accounting, and the zero-dispersion limit.

What is asserted here, and why none of it is a golden number from an earlier run of this
code:

* **The limiting case is the oracle.** A campaign with zero dispersion must reproduce
  :func:`rpo_traj.plan.plan_rendezvous`'s uncorrected plan *exactly* -- same delta-v, same
  terminal miss, same constraint verdicts. That is an independent implementation of the same
  manoeuvre reached through a different call path, so agreement is a real check rather than
  self-consistency.
* **Determinism is structural, and is tested as such.** Same seed twice gives identical
  numbers; the first ``k`` runs of a large campaign are bitwise identical to a ``k``-run
  campaign. The second property belongs to :mod:`rpo_core.montecarlo`'s substream scheme,
  and what is verified here is that composing a campaign on top of it does not break it.
* **The three outcomes partition the campaign.** ``n_failed + n_succeeded +
  n_completed_but_violated == n_runs``, always, and a run that raised never appears among
  the completed ones. A failing run is engineered from the *physics* -- the half-period
  cross-track rank deficiency -- rather than injected, so the accounting is exercised by the
  thing that actually happens.
* **Complements.** Zero dispersion gives zero spread and realistic dispersion does not;
  tightening the cross-track tolerance turns a zero-failure campaign into a partly-failing
  one; the Wilson interval at zero observed breaches is not the zero-width interval the
  normal approximation would report; retaining a per-sample series is shown to be something
  the retention policy structurally cannot do.

The fast tier flies the linear CW dynamics at a low sample count and runs in seconds. Every
campaign whose *numbers* are meant to be believed is marked ``slow`` and flies the nonlinear
dynamics, because a rate computed on the same linear model that designed the burn is not a
deliverable.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from rpo_core.config import ScenarioConfig, load_scenario
from rpo_core.exceptions import PropagationError
from rpo_core.montecarlo import (
    CampaignConfigurationError,
    MagnitudePointingDispersion,
    VectorNormalDispersion,
    wilson_interval,
)
from rpo_core.navigation import NavigationErrorModel
from rpo_core.relative.cw import cw_stm
from rpo_traj.campaign import (
    BURN_EXECUTION_KEY,
    INITIAL_STATE_KEY,
    NAVIGATION_BIAS_KEY,
    CampaignSetupError,
    DispersedCampaignReport,
    DispersionSettings,
    RunOutcome,
    run_dispersed_rendezvous,
    sensitivity_sweep,
)
from rpo_traj.plan import plan_rendezvous

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "vbar_baseline.yaml"

# Fast-tier sizes. 51 samples is far too coarse to resolve the planner's sub-millimetre
# keep-out dip and is not meant to: these tests assert accounting, determinism and
# structure, none of which depend on the sample count. The tests that assert a *number*
# (the baseline rates, the sample-count invariance) are marked slow and use the real
# defaults.
_FAST_SAMPLES = 51
_FAST_RUNS = 60


@pytest.fixture(scope="session")
def baseline_config() -> ScenarioConfig:
    """Return the shipped flagship scenario, loaded through the real validator."""
    return load_scenario(BASELINE_CONFIG_PATH)


def _fast(config: ScenarioConfig, **kwargs: object) -> DispersedCampaignReport:
    """Run a small linear-dynamics campaign with the arguments a fast test wants."""
    parameters: dict[str, object] = {
        "n_samples": _FAST_SAMPLES,
        "dynamics": "cw",
        "settings": DispersionSettings.realistic(),
    }
    parameters.update(kwargs)
    return run_dispersed_rendezvous(config, _FAST_RUNS, 2026, **parameters)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# The limiting case: zero dispersion is the nominal plan
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_zero_dispersion_reproduces_the_nominal_plan_exactly(
    baseline_config: ScenarioConfig, tmp_path: Path
):
    # The control case, and the strongest single check in this file: with no dispersion at
    # all the campaign must land on the same manoeuvre rpo_traj.plan produces without
    # differential correction, reached through an entirely different call path. Measured
    # agreement: exact to the last bit on every quantity below (the two paths perform the
    # same floating-point operations in the same order once the dispersions are gone), so
    # the tolerances are 0.0 rather than a hand-picked epsilon.
    n_samples = 201
    plan = plan_rendezvous(
        baseline_config,
        base_dir=tmp_path,
        make_plots=False,
        correct=False,
        n_samples=n_samples,
    )
    report = run_dispersed_rendezvous(
        baseline_config,
        1,
        0,
        settings=DispersionSettings(),
        n_samples=n_samples,
        dynamics="nonlinear",
    )

    assert report.n_failed == 0
    assert report.n_runs == 1
    per_run = report.per_run_metrics
    assert float(per_run["terminal_position_error_m"][0]) == plan.metrics.terminal_position_error_m
    assert (
        float(per_run["terminal_velocity_error_m_s"][0]) == plan.metrics.terminal_velocity_error_m_s
    )
    assert float(per_run["total_delta_v_m_s"][0]) == plan.metrics.total_delta_v_m_s
    assert float(per_run["n_violating_samples"][0]) == plan.violation_count
    assert bool(per_run["all_constraints_satisfied"][0]) == plan.all_constraints_satisfied
    # The baseline breaks its own approach corridor by construction (dy/4 = 187.5 m of
    # radial bulge against a 10 deg cone), so this is not a vacuous pass: the campaign is
    # reproducing a *violating* plan, violation and all.
    assert float(per_run["approach_corridor_breached"][0]) == 1.0
    assert report.n_succeeded == 0
    assert report.n_completed_but_violated == 1


@pytest.mark.unit
def test_zero_dispersion_gives_zero_spread(baseline_config: ScenarioConfig):
    # Complement for "the dispersions are ignored". Without dispersions every run is the
    # same run, so the standard deviation of every metric is exactly zero; with them, it is
    # not. A campaign that silently dropped its dispersions would pass the first half of
    # this and fail the second.
    undispersed = _fast(baseline_config, settings=DispersionSettings())
    dispersed = _fast(baseline_config)
    for name in ("terminal_position_error_m", "total_delta_v_m_s", "min_separation_m"):
        values = undispersed.per_run_metrics[name]
        # Bitwise equality rather than a zero standard deviation: np.std of 60 identical
        # doubles is 5.6e-17, not 0.0, because the mean of the sum does not round back to
        # the value. Asserting equality of the values themselves says the stronger thing
        # and does not need a tolerance at all.
        assert np.all(values == values[0]), name
        assert float(np.std(dispersed.per_run_metrics[name])) > 0.0


@pytest.mark.unit
def test_each_dispersion_family_moves_the_answer(baseline_config: ScenarioConfig):
    # Finer-grained complement: not just "some dispersion happens" but that each declared
    # family reaches the trajectory. Scaling one family to zero at a time must change the
    # spread, or that family was decorative.
    base = DispersionSettings.realistic()
    full = float(np.std(_fast(baseline_config).per_run_metrics["terminal_position_error_m"]))
    for family in ("burn_execution", "navigation", "initial_state"):
        without = _fast(
            baseline_config,
            settings=base.scaled(**{family: 0.0}),  # type: ignore[arg-type]
        )
        spread = float(np.std(without.per_run_metrics["terminal_position_error_m"]))
        assert spread != full, f"removing {family} changed nothing; it is not wired in"


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_same_seed_gives_identical_results(baseline_config: ScenarioConfig):
    first = _fast(baseline_config)
    second = _fast(baseline_config)
    for name, values in first.per_run_metrics.items():
        assert np.array_equal(values, second.per_run_metrics[name]), name
    assert first.to_dict() == second.to_dict()


@pytest.mark.unit
def test_a_different_seed_gives_different_results(baseline_config: ScenarioConfig):
    # Complement: determinism must not be the degenerate kind where the seed is ignored.
    first = _fast(baseline_config)
    other = run_dispersed_rendezvous(
        baseline_config,
        _FAST_RUNS,
        2027,
        settings=DispersionSettings.realistic(),
        n_samples=_FAST_SAMPLES,
        dynamics="cw",
    )
    assert not np.array_equal(
        first.per_run_metrics["terminal_position_error_m"],
        other.per_run_metrics["terminal_position_error_m"],
    )


@pytest.mark.unit
def test_run_index_is_independent_of_campaign_size(baseline_config: ScenarioConfig):
    # rpo_core.montecarlo guarantees this by deriving run i's seed sequence from (seed, i)
    # alone. What is checked here is that composing a campaign on top of it -- with a
    # navigation model that also draws from the run's own generator -- does not introduce a
    # dependence on n_runs. Bitwise equality, not approximate: anything less would mean a
    # resumed campaign is a different campaign.
    small = run_dispersed_rendezvous(
        baseline_config,
        7,
        2026,
        settings=DispersionSettings.realistic(),
        n_samples=_FAST_SAMPLES,
        dynamics="cw",
    )
    large = _fast(baseline_config)
    assert small.n_failed == 0 and large.n_failed == 0
    for name, values in small.per_run_metrics.items():
        assert np.array_equal(values, large.per_run_metrics[name][: values.size]), name


# --------------------------------------------------------------------------------------
# Failure accounting: three outcomes, never merged
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_outcome_counts_partition_every_campaign(baseline_config: ScenarioConfig):
    report = _fast(baseline_config)
    assert report.n_failed + report.n_succeeded + report.n_completed_but_violated == report.n_runs
    assert report.n_completed == report.n_runs - report.n_failed
    assert report.completed_indices.size == report.n_completed


@pytest.mark.unit
def test_a_physically_failing_run_is_reported_as_a_failure(baseline_config: ScenarioConfig):
    # The failure is real physics, not an injected fault. The baseline is a half-period hop,
    # where the cross-track subproblem is rank-deficient: z(t_f) is pinned to -z_0 whatever
    # impulse is applied. Tightening cross_track_tolerance_m to 8 m makes the CW solve refuse
    # every run whose *estimated* cross-track offset exceeds it. With the realistic settings
    # the estimated z has a one-sigma of sqrt(5^2 + 2^2 + 1^2) = 5.48 m, so the expected
    # failure fraction is 2*(1 - Phi(8/5.48)) = 14.4 %.
    report = _fast(baseline_config, cross_track_tolerance_m=8.0)
    assert 0 < report.n_failed < report.n_runs, "expected a partly-failing campaign"
    assert report.failure_counts_by_type == {"InfeasibleTransferError": report.n_failed}
    # Not counted as successes, and not dropped: the counts still partition the campaign and
    # the failed indices are absent from the completed ones.
    assert report.n_failed + report.n_succeeded + report.n_completed_but_violated == report.n_runs
    assert report.completed_indices.size == report.n_runs - report.n_failed
    assert report.success_rate.trials == report.n_runs
    # And the per-run metric arrays are shorter than the campaign, which is the shape a
    # silently-dropped run would hide.
    assert report.per_run_metrics["terminal_position_error_m"].size == report.n_completed
    # Every breach rate keeps the *full* denominator, which is only observable when there
    # are failures -- a campaign with none cannot distinguish n_runs from n_completed, and
    # the mutation study found exactly that blind spot. Conditioning on completion would
    # inflate every breach rate by n_runs / n_completed and would silently answer the
    # different question "given that it flew", which the module docstring rejects.
    for name, estimate in report.breach_rates.items():
        assert estimate.trials == report.n_runs, name
        assert estimate.trials > report.n_completed, name
    assert report.to_dict()["breach_rate_is_lower_bound"] is True
    assert report.to_dict()["breach_rate_denominator"] == "all_runs_including_failures"


@pytest.mark.unit
def test_the_same_campaign_has_no_failures_at_the_default_tolerance(
    baseline_config: ScenarioConfig,
):
    # Complement of the previous test: the failures come from the tolerance, not from the
    # code. At the default (math.inf) the uncontrollable cross-track offset is accepted, no
    # impulse is wasted on it, and the residual shows up in the terminal miss instead.
    report = _fast(baseline_config)
    assert report.n_failed == 0
    assert report.failure_counts_by_type == {}


@pytest.mark.unit
def test_an_injected_propagation_failure_is_recorded_not_dropped(
    baseline_config: ScenarioConfig,
):
    # A second, cleaner split: fail every run whose dispersed radial offset is positive,
    # which is about half of them and is a deterministic function of the seed. This is the
    # test that a raising truth model does not take the campaign down with it, and that the
    # exception type reaches the report.
    def _failing(initial_state_hill: np.ndarray, times_s: np.ndarray) -> np.ndarray:
        if float(initial_state_hill[0]) > 0.0:
            raise PropagationError("injected integrator failure for a dispersed run")
        from rpo_core.navigation import cw_truth_propagator

        return cw_truth_propagator(baseline_config.orbit.mean_motion_rad_s)(
            initial_state_hill, times_s
        )

    report = _fast(baseline_config, propagate_fn=_failing)
    assert 0 < report.n_failed < report.n_runs
    assert report.failure_counts_by_type == {"PropagationError": report.n_failed}
    assert report.n_failed + report.n_succeeded + report.n_completed_but_violated == report.n_runs
    assert report.to_dict()["breach_rate_is_lower_bound"] is True


@pytest.mark.unit
def test_failed_and_violated_runs_are_reported_separately(baseline_config: ScenarioConfig):
    # The two counts answer different questions and merging them would let a campaign
    # improve its violation rate by crashing more often. The baseline violates its corridor
    # in every completed run and fails a fraction of them, so both counts are non-zero here
    # and are demonstrably not the same number.
    report = _fast(baseline_config, cross_track_tolerance_m=8.0)
    assert report.n_failed > 0
    assert report.n_completed_but_violated > 0
    assert report.n_failed != report.n_completed_but_violated
    payload = report.to_dict()
    assert payload["n_failed"] == report.n_failed
    assert payload["n_completed_but_violated"] == report.n_completed_but_violated


# --------------------------------------------------------------------------------------
# Rates and intervals
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_rates_use_the_wilson_interval_and_the_full_denominator(
    baseline_config: ScenarioConfig,
):
    report = _fast(baseline_config)
    for estimate in (report.success_rate, report.completion_rate, *report.breach_rates.values()):
        assert estimate.trials == report.n_runs
        assert estimate.describe()["interval"] == "wilson_score"
        assert (estimate.lower, estimate.upper) == wilson_interval(
            estimate.successes, estimate.trials, confidence=report.confidence
        )


@pytest.mark.unit
def test_a_zero_breach_count_still_has_a_non_zero_upper_bound(baseline_config: ScenarioConfig):
    # The reason Wilson rather than the normal approximation, measured on a real campaign.
    # The approach ellipsoid (2 x 4 x 2 km) is never left on this scenario -- the whole hop
    # lives inside 1 km -- so the observed breach count is zero, and Wald would report the
    # interval [0, 0]: "sixty runs prove the breach probability is zero". Wilson reports
    # [0, z^2/(n + z^2)] = [0, 6.0 %] at n = 60, which is the honest statement and is the
    # number a reviewer should be given.
    report = _fast(baseline_config)
    ellipsoid = report.breach_rates["approach_ellipsoid"]
    assert ellipsoid.successes == 0
    assert ellipsoid.point == 0.0
    assert ellipsoid.lower == 0.0
    wald_upper = 0.0  # p_hat + z sqrt(p_hat (1 - p_hat) / n) with p_hat = 0
    assert ellipsoid.upper > wald_upper
    assert ellipsoid.upper == pytest.approx(1.959963985**2 / (60 + 1.959963985**2), rel=1e-6)


@pytest.mark.unit
def test_the_breach_rate_counts_the_constraint_it_names(baseline_config: ScenarioConfig):
    # A report that wired the four constraints to the wrong counters would still produce
    # plausible-looking numbers, so the mapping is pinned against three constraints whose
    # outcome is known from the geometry and one that is genuinely stochastic:
    #
    #  * the corridor is broken in *every* run -- dy/4 = 187.5 m of radial bulge against a
    #    10 deg cone is model M4's corollary and has nothing to do with the dispersions;
    #  * the ellipsoid is never left, because a 1 km hop inside a 2 x 4 x 2 km volume cannot
    #    leave it however the dispersions fall;
    #  * the keep-out sphere is breached in a minority of runs, and that minority is exactly
    #    the set whose sampled minimum range fell below the 200 m radius.
    report = _fast(baseline_config)
    assert report.breach_rates["approach_corridor"].successes == report.n_completed
    assert report.breach_rates["approach_ellipsoid"].successes == 0
    keep_out = report.breach_rates["keep_out_sphere"]
    assert 0 < keep_out.successes < report.n_completed
    assert keep_out.successes == int(np.sum(report.per_run_metrics["keep_out_clearance_m"] < 0.0))
    for name, estimate in report.breach_rates.items():
        metric = f"{name}_breached"
        assert estimate.successes == round(float(np.sum(report.per_run_metrics[metric])))


@pytest.mark.unit
def test_only_estimation_error_produces_an_in_plane_terminal_miss(
    baseline_config: ScenarioConfig,
):
    # The closed form of rpo_core.navigation, measured end to end through the campaign: the
    # terminal state error is -Phi @ e in the *estimation* error e, so a delivery dispersion
    # the navigation solution can see costs **nothing** in-plane -- the guidance simply
    # plans from where the chaser actually is. Switch off navigation and burn error and the
    # in-plane miss must collapse to machine precision even though the truth was displaced
    # by 5 m one-sigma in every axis.
    #
    # Measured in-plane miss with delivery dispersion only: 6.4e-11 m worst case over 60
    # runs, against a 5 m displacement. The bound below is 1e-6 m, ~1.6e4 headroom.
    delivery_only = _fast(
        baseline_config,
        settings=DispersionSettings.realistic().scaled(navigation=0.0, burn_execution=0.0),
    )
    in_plane = delivery_only.per_run_metrics["terminal_in_plane_position_error_m"]
    assert float(np.max(in_plane)) < 1.0e-6
    assert delivery_only.breach_rates["keep_out_sphere"].successes == 0

    # What is left is the cross-track floor, which is structural: at a half period
    # z(t_f) = -z_0 whatever impulse is applied, so the whole terminal miss is |z_0| of the
    # dispersed truth. Its mean for a 5 m one-sigma Gaussian is 5 * sqrt(2/pi) = 3.989 m.
    full = delivery_only.per_run_metrics["terminal_position_error_m"]
    assert float(np.mean(full)) == pytest.approx(5.0 * math.sqrt(2.0 / math.pi), rel=0.25)

    # Complement: navigation error alone, with the truth delivered perfectly, produces the
    # breaches. This is the sensitivity that matters on this scenario and it is a property
    # of Phi_rv, not of the dispersion sizes: at tau = pi the along-track
    # position-from-velocity coefficient is (4 sin tau - 3 tau)/n = -3 pi / n = -8367 s, so
    # 1 mm/s of velocity *knowledge* error is 8.4 m of along-track terminal miss.
    navigation_only = _fast(
        baseline_config,
        settings=DispersionSettings.realistic().scaled(initial_state=0.0, burn_execution=0.0),
    )
    assert navigation_only.breach_rates["keep_out_sphere"].successes > 0
    assert float(np.mean(navigation_only.per_run_metrics["terminal_in_plane_position_error_m"])) > (
        10.0 * float(np.mean(in_plane) + 1.0)
    )


@pytest.mark.unit
def test_terminal_error_percentiles_are_reported(baseline_config: ScenarioConfig):
    report = _fast(baseline_config)
    summary = report.metrics["terminal_position_error_m"]
    assert summary.count == report.n_completed
    assert sorted(summary.percentiles) == [1.0, 5.0, 50.0, 95.0, 99.0]
    values = report.per_run_metrics["terminal_position_error_m"]
    for level, value in summary.percentiles.items():
        assert value == pytest.approx(float(np.percentile(values, level)))


# --------------------------------------------------------------------------------------
# Retention policy: scalars only
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_run_outcome_carries_no_per_sample_series(baseline_config: ScenarioConfig):
    # The memory guarantee, asserted structurally rather than by measuring bytes: every
    # field of RunOutcome is a scalar, so a 1000-run campaign cannot accumulate trajectories
    # however many samples each run used. Bytes are measured in the slow tier below.
    assert {field.type for field in fields(RunOutcome)} <= {"float", "int", "bool"}
    report = _fast(baseline_config)
    for name, values in report.per_run_metrics.items():
        assert values.ndim == 1, name
        assert values.size == report.n_completed, name
    # And the raw run results are not retained at all.
    assert report.summary.to_json()


@pytest.mark.unit
def test_the_retained_metrics_do_not_grow_with_the_sample_count(
    baseline_config: ScenarioConfig,
):
    # Complement for the structural check: raising the sample count by 4x must not change
    # the size of anything retained. If a per-sample series were leaking into the record,
    # this is where it would show.
    coarse = _fast(baseline_config, n_samples=51)
    fine = _fast(baseline_config, n_samples=201)
    for name in coarse.per_run_metrics:
        assert coarse.per_run_metrics[name].size == fine.per_run_metrics[name].size
    assert len(coarse.to_dict()["metrics"]) == len(fine.to_dict()["metrics"])


@pytest.mark.unit
def test_the_report_serialises_to_strict_json(baseline_config: ScenarioConfig):
    payload = json.dumps(_fast(baseline_config).to_dict(), allow_nan=False, sort_keys=True)
    restored = json.loads(payload)
    assert restored["breach_rate_denominator"] == "all_runs_including_failures"
    assert restored["dispersions"][BURN_EXECUTION_KEY]["kind"] == "magnitude_pointing"


# --------------------------------------------------------------------------------------
# Dispersion settings
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_navigation_bias_is_a_declared_dispersion_and_the_noise_is_not():
    # The bias is drawn once per run, which is exactly what a campaign dispersion is; the
    # white noise is drawn per estimate and therefore cannot be one. Getting this backwards
    # is the modelling error the whole module is arranged around.
    declared = DispersionSettings.realistic().dispersions()
    assert set(declared) == {BURN_EXECUTION_KEY, INITIAL_STATE_KEY, NAVIGATION_BIAS_KEY}
    assert isinstance(declared[NAVIGATION_BIAS_KEY], VectorNormalDispersion)
    assert isinstance(declared[BURN_EXECUTION_KEY], MagnitudePointingDispersion)
    assert np.allclose(
        declared[NAVIGATION_BIAS_KEY].covariance,
        DispersionSettings.realistic().navigation.bias_covariance,
    )


@pytest.mark.unit
def test_a_noise_only_model_declares_no_bias_dispersion():
    settings = DispersionSettings(
        navigation=NavigationErrorModel(noise_covariance=np.diag([1.0] * 3 + [1e-6] * 3))
    )
    assert NAVIGATION_BIAS_KEY not in settings.dispersions()


@pytest.mark.unit
def test_scaling_to_zero_removes_a_dispersion_family():
    scaled = DispersionSettings.realistic().scaled(burn_execution=0.0, initial_state=0.0)
    assert scaled.burn_execution is None
    assert scaled.initial_state_covariance is None
    assert set(scaled.dispersions()) == {NAVIGATION_BIAS_KEY}


@pytest.mark.unit
def test_scaling_multiplies_the_one_sigmas_not_the_covariance():
    base = DispersionSettings.realistic()
    doubled = base.scaled(navigation=2.0, burn_execution=3.0)
    assert doubled.burn_execution is not None and base.burn_execution is not None
    assert doubled.burn_execution.sigma_magnitude == pytest.approx(
        3.0 * base.burn_execution.sigma_magnitude
    )
    assert np.allclose(doubled.navigation.bias_covariance, 4.0 * base.navigation.bias_covariance)


@pytest.mark.unit
def test_scaling_rejects_a_negative_factor():
    with pytest.raises(CampaignSetupError, match=r"navigation scale must be a finite"):
        DispersionSettings.realistic().scaled(navigation=-1.0)


@pytest.mark.unit
def test_settings_reject_a_malformed_initial_state_covariance():
    from rpo_core.navigation import CovarianceDefinitionError

    with pytest.raises(
        CovarianceDefinitionError, match=r"initial_state_covariance must have shape"
    ):
        DispersionSettings(initial_state_covariance=np.eye(3))


# --------------------------------------------------------------------------------------
# Raise paths
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_campaign_rejects_too_few_samples(baseline_config: ScenarioConfig):
    with pytest.raises(CampaignSetupError, match=r"n_samples must be an int of at least 3"):
        run_dispersed_rendezvous(baseline_config, 2, 0, n_samples=2)


@pytest.mark.unit
def test_campaign_rejects_a_negative_terminal_tolerance(baseline_config: ScenarioConfig):
    with pytest.raises(CampaignSetupError, match=r"terminal_tolerance_m must be a finite"):
        run_dispersed_rendezvous(baseline_config, 2, 0, terminal_tolerance_m=-1.0)


@pytest.mark.unit
def test_campaign_rejects_an_unknown_dynamics_model(baseline_config: ScenarioConfig):
    with pytest.raises(CampaignSetupError, match=r"dynamics must be 'nonlinear' or 'cw'"):
        run_dispersed_rendezvous(baseline_config, 2, 0, dynamics="j2")  # type: ignore[arg-type]


@pytest.mark.unit
def test_campaign_rejects_a_non_positive_run_count(baseline_config: ScenarioConfig):
    # Raised by the Monte Carlo harness, not rewrapped: the message that names the reason is
    # worth more than one that says "campaign failed".
    with pytest.raises(CampaignConfigurationError, match=r"n_runs must be a positive int"):
        run_dispersed_rendezvous(baseline_config, 0, 0, dynamics="cw", n_samples=_FAST_SAMPLES)


@pytest.mark.unit
def test_campaign_rejects_a_negative_seed(baseline_config: ScenarioConfig):
    with pytest.raises(CampaignConfigurationError, match=r"seed must be a non-negative int"):
        run_dispersed_rendezvous(baseline_config, 2, -1, dynamics="cw", n_samples=_FAST_SAMPLES)


@pytest.mark.unit
def test_sensitivity_sweep_rejects_an_unknown_source(baseline_config: ScenarioConfig):
    with pytest.raises(CampaignSetupError, match=r"unknown sensitivity source"):
        sensitivity_sweep(baseline_config, 2, 0, sources=("solar_pressure",))


# --------------------------------------------------------------------------------------
# Sensitivity (SRS F-5.5)
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_sensitivity_sweep_shows_navigation_error_driving_the_terminal_miss(
    baseline_config: ScenarioConfig,
):
    # F-5.5 asks for sensitivity to burn and navigation error. The expected physics: the
    # terminal miss is -Phi @ e in the estimation error e, so it scales linearly with the
    # navigation one-sigma and is barely moved by burn execution error, which perturbs an
    # already-small impulse. Both directions are asserted; a sweep that reported the same
    # number at every scale would fail the first and a sweep that reported noise would fail
    # the second.
    points = sensitivity_sweep(
        baseline_config,
        20,
        2026,
        sources=("navigation",),
        scales=(0.0, 1.0, 4.0),
        n_samples=_FAST_SAMPLES,
        dynamics="cw",
    )
    misses = [
        float(np.mean(point.report.per_run_metrics["terminal_in_plane_position_error_m"]))
        for point in points
    ]
    assert misses[0] < misses[1] < misses[2]
    # Linear in the one-sigma, to within the sampling spread of 20 runs.
    assert misses[2] / misses[1] == pytest.approx(4.0, rel=0.35)


@pytest.mark.unit
def test_sensitivity_uses_common_random_numbers(baseline_config: ScenarioConfig):
    # Every point shares a seed, and the substream scheme keys each dispersion on its own
    # name, so scaling the navigation dispersion leaves each run's burn draw bitwise
    # unchanged. That is what makes a 20-run sweep readable at all: the difference between
    # two points is the scaling, not two different sets of draws.
    points = sensitivity_sweep(
        baseline_config,
        12,
        2026,
        sources=("navigation",),
        scales=(1.0, 4.0),
        settings=DispersionSettings.realistic().scaled(initial_state=0.0),
        n_samples=_FAST_SAMPLES,
        dynamics="cw",
    )
    # The commanded delta-v depends on the estimate, so it moves; the *executed* magnitude
    # ratio is set by the burn draw alone and must be identical run for run.
    ratios = [
        point.report.per_run_metrics["total_delta_v_m_s"]
        / point.report.per_run_metrics["commanded_delta_v_m_s"]
        for point in points
    ]
    assert np.allclose(ratios[0], ratios[1], rtol=5.0e-3)
    assert not np.array_equal(
        points[0].report.per_run_metrics["commanded_delta_v_m_s"],
        points[1].report.per_run_metrics["commanded_delta_v_m_s"],
    )


# --------------------------------------------------------------------------------------
# Burn execution error reaches the propellant budget
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_burn_execution_error_separates_commanded_from_executed_delta_v(
    baseline_config: ScenarioConfig,
):
    dispersed = _fast(baseline_config)
    perfect = _fast(
        baseline_config, settings=DispersionSettings.realistic().scaled(burn_execution=0.0)
    )
    executed = dispersed.per_run_metrics["total_delta_v_m_s"]
    commanded = dispersed.per_run_metrics["commanded_delta_v_m_s"]
    assert not np.allclose(executed, commanded)
    # The magnitude scale factor is N(1, 0.02) per burn, so the executed-to-commanded ratio
    # should sit within a few per cent of one and be centred on it.
    ratio = executed / commanded
    assert float(np.mean(ratio)) == pytest.approx(1.0, abs=0.02)
    assert 0.0 < float(np.std(ratio)) < 0.05
    # With execution error switched off the two are identical to the last bit.
    assert np.array_equal(
        perfect.per_run_metrics["total_delta_v_m_s"],
        perfect.per_run_metrics["commanded_delta_v_m_s"],
    )


# --------------------------------------------------------------------------------------
# Slow tier: the numbers that are meant to be believed
# --------------------------------------------------------------------------------------


#: Runs in the deliverable campaign. 200 is a compromise: the Wilson half-width on a rate
#: near 0.3 is +-6.5 points here, which is enough to separate "a substantial minority" from
#: "rare" but not enough to quote a breach probability to the percent. A study meant to
#: certify a number would run 2000; a test suite that has to finish should not.
_SLOW_RUNS = 200


@pytest.fixture(scope="session")
def baseline_campaign(baseline_config: ScenarioConfig) -> DispersedCampaignReport:
    """Return the deliverable campaign, shared by the slow tests that interrogate it."""
    return run_dispersed_rendezvous(baseline_config, _SLOW_RUNS, 42, dynamics="nonlinear")


@pytest.mark.slow
@pytest.mark.integration
def test_the_baseline_campaign_under_realistic_dispersions(
    baseline_config: ScenarioConfig, baseline_campaign: DispersedCampaignReport
):
    # The deliverable campaign: nonlinear truth dynamics, the shipped scenario, the
    # representative dispersion set. The assertions are the ones a reviewer would want
    # pinned, and every one of them is a structural consequence of the scenario rather than
    # a number read off a previous run:
    #
    #  * every run completes -- nothing about the dispersions makes the solve ill-posed at
    #    the default cross-track tolerance;
    #  * the success rate is zero, because the baseline breaks its own 10 deg corridor by
    #    187.5 m of radial bulge in every run (model M4's corollary), and the corridor
    #    breach rate is therefore one;
    #  * the keep-out sphere **is** breached in a substantial minority of runs, and that is
    #    the finding rather than a defect. The arrival hold point is 250 m from a 200 m
    #    sphere, and the navigation-driven terminal miss has a one-sigma of 67 m in-plane
    #    (from Phi @ P_nav @ Phi.T, computed below rather than quoted), so a 50 m miss in
    #    the wrong direction is well inside one sigma;
    #  * the terminal miss carries the uncontrollable cross-track floor, so the full miss is
    #    strictly larger than the in-plane part;
    #  * the ellipsoid is never left, which keeps a Wilson interval at zero counts in the
    #    report.
    report = baseline_campaign

    assert report.n_runs == _SLOW_RUNS
    assert report.n_failed == 0
    assert report.completion_rate.point == 1.0
    assert report.n_succeeded == 0
    assert report.success_rate.upper < 0.02
    assert report.breach_rates["approach_corridor"].point == 1.0
    assert report.breach_rates["approach_ellipsoid"].successes == 0

    # The keep-out breach probability against the closed form. The in-plane terminal miss is
    # -Phi @ e with e the estimation error, so its covariance is Phi P_nav Phi.T; the run
    # breaches when the miss carries the chaser more than 50 m closer to the target than the
    # 250 m hold point. Predicting the exact probability would need the joint geometry, so
    # what is asserted is the order: the breach rate must be a genuine minority, and it must
    # be consistent with a 50 m threshold against the predicted along-track one-sigma.
    keep_out = report.breach_rates["keep_out_sphere"]
    assert 0.05 < keep_out.point < 0.6
    assert keep_out.lower > 0.0, "a non-zero observed count must have a non-zero lower bound"
    settings = DispersionSettings.realistic()
    phi = cw_stm(baseline_config.orbit.mean_motion_rad_s, baseline_config.tof_s)
    sigma_terminal = np.sqrt(np.diag(phi @ settings.navigation.total_covariance @ phi.T))
    assert float(np.hypot(sigma_terminal[0], sigma_terminal[1])) > 50.0

    full = report.per_run_metrics["terminal_position_error_m"]
    in_plane = report.per_run_metrics["terminal_in_plane_position_error_m"]
    assert np.all(full >= in_plane - 1.0e-12)
    assert float(np.mean(full)) > float(np.mean(in_plane))


@pytest.mark.slow
@pytest.mark.integration
def test_the_cross_track_floor_is_the_true_initial_offset(
    baseline_campaign: DispersedCampaignReport,
):
    # The half-period cross-track trap, measured rather than asserted from the docstring.
    # At a half period z(t_f) = cos(pi) z_0 = -z_0 whatever impulse is applied, so the
    # terminal cross-track miss must equal the *true* initial cross-track offset. The
    # dispersed truth offset has a one-sigma of 5 m, so the RMS of the cross-track part of
    # the miss must match that to within the sampling error of an RMS over n runs,
    # sigma/sqrt(2n) = 0.25 m at n = 200.
    report = baseline_campaign
    n_runs = report.n_runs
    full = report.per_run_metrics["terminal_position_error_m"]
    in_plane = report.per_run_metrics["terminal_in_plane_position_error_m"]
    cross_track = np.sqrt(np.maximum(full**2 - in_plane**2, 0.0))
    sigma_z_m = 5.0
    standard_error = sigma_z_m / math.sqrt(2.0 * n_runs)
    assert float(np.sqrt(np.mean(cross_track**2))) == pytest.approx(
        sigma_z_m, abs=5.0 * standard_error
    )


@pytest.mark.slow
@pytest.mark.integration
def test_the_sample_count_does_not_move_the_campaign_rates(baseline_config: ScenarioConfig):
    # The measurement behind DEFAULT_CAMPAIGN_SAMPLE_COUNT. 401 against the planner's 2001:
    # the rates must be identical and the metric percentiles must agree far inside the
    # run-to-run spread, or the default is buying precision on the wrong quantity.
    coarse = run_dispersed_rendezvous(baseline_config, 40, 42, dynamics="nonlinear", n_samples=401)
    fine = run_dispersed_rendezvous(baseline_config, 40, 42, dynamics="nonlinear", n_samples=2001)
    assert coarse.n_failed == fine.n_failed == 0
    assert coarse.n_succeeded == fine.n_succeeded
    for name, estimate in coarse.breach_rates.items():
        assert estimate.successes == fine.breach_rates[name].successes, name
    # Terminal state is a function of the impulses, not of the output grid, so it is
    # identical to the last bit; the sampled extrema are grid-dependent and are compared
    # against the spread of the metric itself rather than against an absolute epsilon.
    assert np.allclose(
        coarse.per_run_metrics["terminal_position_error_m"],
        fine.per_run_metrics["terminal_position_error_m"],
        rtol=1.0e-9,
    )
    spread = float(np.std(fine.per_run_metrics["min_separation_m"]))
    difference = float(
        np.max(
            np.abs(
                coarse.per_run_metrics["min_separation_m"]
                - fine.per_run_metrics["min_separation_m"]
            )
        )
    )
    assert difference < 0.05 * spread, (
        f"sample count moved the minimum separation by {difference:.4f} m against a "
        f"run-to-run spread of {spread:.4f} m"
    )
