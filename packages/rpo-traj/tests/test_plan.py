"""Tests for :mod:`rpo_traj.plan`.

Every tolerance below was set by printing the value first; the measurement and the headroom
it buys are named in a comment beside each one. Nothing here asserts a golden number taken
from an earlier run of this same code: the delta-v check is a closed form derived
independently in ``docs/project1/math-model.md``, the correction check is a comparison
against a baseline the same run produces, the convergence check compares a grid quantity
against a limit found by an independent root-find, and the invariance check is a symmetry of
the two-body problem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rpo_core.config import ScenarioConfig, load_scenario
from rpo_core.constants import MU_EARTH_M3_S2
from rpo_core.constraints import range_rate_m_s, separation_m
from rpo_core.metrics import read_metrics
from rpo_core.relative.nonlinear import propagate_relative_nonlinear
from rpo_core.targeting import TargetingConvergenceError
from rpo_traj.plan import (
    DEFAULT_SAMPLE_COUNT,
    MIN_SAMPLE_COUNT,
    PlanningError,
    RendezvousPlan,
    plan_rendezvous,
    target_state_eci,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "vbar_baseline.yaml"


@pytest.fixture(scope="session")
def baseline_config() -> ScenarioConfig:
    """Return the shipped flagship scenario, loaded through the real validator."""
    return load_scenario(BASELINE_CONFIG_PATH)


@pytest.fixture(scope="session")
def corrected_plan(
    baseline_config: ScenarioConfig, tmp_path_factory: pytest.TempPathFactory
) -> RendezvousPlan:
    """Return the baseline planned at the shipped defaults. Session-scoped: ~0.3 s."""
    return plan_rendezvous(
        baseline_config, base_dir=tmp_path_factory.mktemp("corrected"), make_plots=False
    )


@pytest.fixture(scope="session")
def uncorrected_plan(
    baseline_config: ScenarioConfig, tmp_path_factory: pytest.TempPathFactory
) -> RendezvousPlan:
    """Return the baseline flying the raw Clohessy-Wiltshire impulses."""
    return plan_rendezvous(
        baseline_config,
        base_dir=tmp_path_factory.mktemp("uncorrected"),
        make_plots=False,
        correct=False,
    )


#: A hop short enough that its ``dy/4`` radial bulge stays inside the same 10 deg corridor
#: the baseline breaks. Defined in code rather than in ``configs/`` because it exists to be
#: the baseline's complement in this suite, not to be a scenario anybody would fly.
#:
#: Both activation ranges are widened to 1200 m so the corridor and the closing-velocity
#: limit are genuinely enforced over the whole 1000 m -> 900 m hop. Left at the baseline's
#: 1000 m / 300 m they would go unevaluated for most or all of the arc, and the run would
#: report a pass that had never been tested -- the vacuous satisfaction the baseline config's
#: own comments warn about.
SHORT_HOP_SCENARIO: dict[str, object] = {
    "name": "vbar_short_hop",
    "description": "Short V-bar hop whose radial bulge stays inside the corridor.",
    "orbit": {"altitude_m": 420000.0, "inclination_deg": 51.6},
    "start_hold_point": {"name": "vbar_minus_1000", "position_hill_m": [0.0, -1000.0, 0.0]},
    "target_hold_point": {"name": "vbar_minus_900", "position_hill_m": [0.0, -900.0, 0.0]},
    "constraints": {
        "keep_out_sphere_radius_m": 200.0,
        "approach_ellipsoid_semi_axes_m": [2000.0, 4000.0, 2000.0],
        "approach_cone_half_angle_deg": 10.0,
        "approach_cone_activation_range_m": 1200.0,
        "max_closing_velocity_m_s": 0.1,
        "max_closing_velocity_activation_range_m": 1200.0,
    },
    "maneuver": {"tof_periods": 0.5},
    "integrator": {"method": "DOP853", "rtol": 1.0e-12, "atol": 1.0e-12},
    "seed": 42,
}


@pytest.fixture(scope="session")
def short_hop_config() -> ScenarioConfig:
    """Return a scenario that satisfies every constraint, on the baseline's machinery."""
    return ScenarioConfig.model_validate(SHORT_HOP_SCENARIO)


def _hop_length_m(config: ScenarioConfig) -> float:
    """Return the along-track hop length ``|dy|``, metres."""
    return abs(
        config.target_hold_point.position_hill_m[1] - config.start_hold_point.position_hill_m[1]
    )


# ======================================================================================
# The closed form: total delta-v of a half-period V-bar hop
# ======================================================================================


@pytest.mark.integration
def test_half_period_hop_total_delta_v_matches_closed_form(
    baseline_config: ScenarioConfig, uncorrected_plan: RendezvousPlan
) -> None:
    """A half-period coplanar V-bar hop costs exactly ``n*dy/2``.

    Derived independently of this code in ``docs/project1/math-model.md`` (model M4): at
    ``tau = pi`` the two-impulse solve collapses to two equal purely radial impulses of
    ``n*dy/4``. This is a genuine analytic solution, not a stored output.
    """
    mean_motion_rad_s = baseline_config.orbit.mean_motion_rad_s
    closed_form_m_s = mean_motion_rad_s * _hop_length_m(baseline_config) / 2.0

    achieved_m_s = uncorrected_plan.metrics.total_delta_v_m_s
    # Measured absolute difference: exactly 0.0 -- the two expressions agree bit for bit,
    # because both reduce to the same product of the same doubles. 1e-15 relative is
    # nominal headroom against a future change in the order of operations, not slack the
    # current implementation needs.
    assert achieved_m_s == pytest.approx(closed_form_m_s, rel=1e-15)

    # Both impulses are equal in magnitude and purely radial (Hill x), which is the
    # structure the closed form comes from. Anything else reaching the same total would be
    # a different manoeuvre with a coincidental budget.
    depart, arrive = uncorrected_plan.burns
    quarter_hop_m_s = mean_motion_rad_s * _hop_length_m(baseline_config) / 4.0
    assert depart.magnitude_m_s == pytest.approx(quarter_hop_m_s, rel=1e-15)
    assert arrive.magnitude_m_s == pytest.approx(quarter_hop_m_s, rel=1e-15)
    for burn in (depart, arrive):
        # Measured off-radial components: 6.5e-18 and 0.0 m/s, i.e. round-off on a
        # 0.211 m/s impulse. 1e-12 m/s is six decades of headroom and still far below
        # anything a thruster could deliver.
        assert abs(burn.delta_v_hill_m_s[1]) < 1e-12
        assert abs(burn.delta_v_hill_m_s[2]) < 1e-12


@pytest.mark.integration
def test_closed_form_is_a_knife_edge_at_the_half_period(
    baseline_config: ScenarioConfig, tmp_path: Path
) -> None:
    """Off the half period the closed form does not hold, so the check tests something.

    The complement required by ``docs/CONTRIBUTING.md``: without it, a test asserting
    ``n*dy/2`` would still pass if the transfer time were being ignored entirely.
    """
    off_period_config = ScenarioConfig.model_validate(
        {**baseline_config.model_dump(mode="python"), "maneuver": {"tof_periods": 0.4}}
    )
    plan = plan_rendezvous(
        off_period_config, base_dir=tmp_path, make_plots=False, correct=False, n_samples=301
    )
    closed_form_m_s = baseline_config.orbit.mean_motion_rad_s * _hop_length_m(baseline_config) / 2.0
    # Measured: 0.616817 m/s against the half-period 0.422392 m/s, a ratio of 1.4603. The
    # 1.2x floor is far below that and far above any conceivable round-off.
    assert plan.metrics.total_delta_v_m_s > 1.2 * closed_form_m_s


# ======================================================================================
# Differential correction versus raw CW
# ======================================================================================


@pytest.mark.integration
def test_correction_measurably_beats_raw_cw_on_terminal_miss(
    corrected_plan: RendezvousPlan, uncorrected_plan: RendezvousPlan
) -> None:
    """Correcting onto nonlinear dynamics removes the CW linearisation miss.

    Measured on the baseline scenario at the shipped defaults:

    ==========================  ==============
    raw CW terminal miss        1.2252713 m
    corrected terminal miss     5.521457e-07 m
    improvement                 2.219e+06 x
    Newton iterations           1
    ==========================  ==============

    The raw figure is not an error in CW's own terms -- CW hits its target exactly. It is
    the price of flying a linear solution through nonlinear dynamics, and it agrees with
    the ``6*pi*rho^2/r`` law of ``docs/cw_validity.md`` to within its stated envelope.
    """
    raw_miss_m = uncorrected_plan.metrics.terminal_position_error_m
    corrected_miss_m = corrected_plan.metrics.terminal_position_error_m

    # Both plans report the same uncorrected baseline, one from the shooting solver's first
    # residual and one from the trajectory it actually flew. That they agree exactly is the
    # cross-check that makes the comparison below meaningful.
    assert uncorrected_plan.raw_cw_terminal_miss_m == raw_miss_m
    assert corrected_plan.raw_cw_terminal_miss_m == pytest.approx(raw_miss_m, rel=1e-12)

    # Measured 1.2252713 m. Bracketed rather than pinned: the value is a physical
    # consequence of the scenario, and +/-20 % is wide enough to survive an integrator
    # tolerance change while being nowhere near 0 or the 200 m keep-out sphere.
    assert 1.0 < raw_miss_m < 1.5

    # Measured 5.52e-07 m against the solver's own 1e-03 m default tolerance.
    assert corrected_miss_m < 1e-6
    # Measured improvement 2.219e+06. 1e+05 is a factor of 22 of headroom and still an
    # improvement no accidental no-op could produce.
    assert raw_miss_m / corrected_miss_m > 1e5

    # The correction is not free, and a correction that cost nothing would mean the burn
    # never changed. Measured delta-v difference 9.002e-05 m/s on a 0.422392 m/s budget
    # (2.13e-04 relative), and measured |dv1 - dv1_cw| 1.267e-04 m/s.
    delta_budget_m_s = abs(
        corrected_plan.metrics.total_delta_v_m_s - uncorrected_plan.metrics.total_delta_v_m_s
    )
    assert 1e-5 < delta_budget_m_s < 1e-3

    # Terminal velocity is matched by construction once corrected: dv2 is computed against
    # the velocity the nonlinear coast actually delivers. Uncorrected, it is not.
    assert corrected_plan.metrics.terminal_velocity_error_m_s == pytest.approx(0.0, abs=1e-15)
    # Measured 8.310e-04 m/s of residual drift after the raw CW arrival burn.
    assert uncorrected_plan.metrics.terminal_velocity_error_m_s > 1e-4

    assert corrected_plan.targeting_iterations == 1
    assert uncorrected_plan.targeting_iterations is None
    assert uncorrected_plan.corrected_terminal_miss_m is None


# ======================================================================================
# Sample-density convergence -- the justification for DEFAULT_SAMPLE_COUNT
# ======================================================================================


def _activation_crossing_closing_velocity_m_s(plan: RendezvousPlan) -> float:
    """Return the closing velocity exactly at the closing-velocity activation range.

    Found by bisecting the range along the coast, which is an independent method from the
    grid maximum it is used to check: a root-find on ``|rho|(t) - R_activation`` rather than
    a maximum over samples. Closing velocity decreases monotonically through the final
    approach, so the constraint's supremum sits at that crossing and the grid maximum can
    only approach it from below.
    """
    config = plan.config
    r_target_eci_m, v_target_eci_m_s = target_state_eci(config.orbit)
    initial_state_hill = np.concatenate(
        (
            np.asarray(config.start_hold_point.position_hill_m, dtype=np.float64),
            np.asarray(plan.burns[0].delta_v_hill_m_s, dtype=np.float64),
        )
    )
    activation_range_m = config.constraints.max_closing_velocity_activation_range_m

    def state_at(time_s: float) -> np.ndarray:
        trajectory = propagate_relative_nonlinear(
            r_target_eci_m,
            v_target_eci_m_s,
            initial_state_hill,
            np.array([0.0, time_s], dtype=np.float64),
            rtol=config.integrator.rtol,
            atol=config.integrator.atol,
        )
        return np.asarray(trajectory[-1:], dtype=np.float64)

    low_s, high_s = 0.5 * config.tof_s, config.tof_s
    for _ in range(60):  # 60 halvings of a ~1400 s bracket lands well below float spacing
        middle_s = 0.5 * (low_s + high_s)
        if float(separation_m(state_at(middle_s))[0]) > activation_range_m:
            low_s = middle_s
        else:
            high_s = middle_s
    return -float(range_rate_m_s(state_at(0.5 * (low_s + high_s)))[0])


# Not marked slow: measured at 1.5 s, and it is the only thing that justifies
# DEFAULT_SAMPLE_COUNT. Excluding it from the fast CI job would leave the shipped default
# unchecked on every pull request, which is exactly where a silent change to it would land.
@pytest.mark.integration
def test_reported_extrema_converge_with_sample_density(
    baseline_config: ScenarioConfig, tmp_path: Path, uncorrected_plan: RendezvousPlan
) -> None:
    """The two reported extrema stop moving as the grid refines, which fixes the default.

    Run uncorrected, because that is the mode where the keep-out minimum is genuinely
    interior and therefore genuinely resolution-limited. Corrected, the arc arrives on its
    hold point and the minimum sits on the endpoint, where no sample count can be wrong.

    The sample counts are deliberately **non-nested** (101, 1009, 2003, 3001). A nested
    sweep -- 1001, 2001, 4001 -- shares grid points, so a finer grid can reproduce a coarser
    one's extremum exactly and read as converged whether or not either found the feature.
    Measured: at N = 2001 and N = 4001 the closing-velocity maximum agrees to 0.0e+00, which
    is an artefact of the shared points, not convergence.
    """
    counts = (101, 1009, 2003, 3001)
    plans = {
        count: plan_rendezvous(
            baseline_config,
            base_dir=tmp_path / f"n{count}",
            make_plots=False,
            correct=False,
            n_samples=count,
        )
        for count in counts
    }

    # --- Minimum keep-out clearance: a resolution threshold -----------------------------
    # The endpoint clearance is an independent reference: it is the terminal position the
    # metrics record already carries, not a number this sweep produced.
    terminal_position_hill_m = np.asarray(
        uncorrected_plan.metrics.achieved_terminal_state_hill[:3], dtype=np.float64
    )
    endpoint_clearance_m = (
        float(np.linalg.norm(terminal_position_hill_m))
        - baseline_config.constraints.keep_out_sphere_radius_m
    )

    coarse = plans[101].metrics
    # 101 samples (spacing 27.9 s) steps clean over a 3.6 s minimum and reports the
    # endpoint. Measured agreement with the endpoint clearance: 1.4e-10 m.
    assert coarse.min_koz_clearance_refined_m == pytest.approx(endpoint_clearance_m, abs=1e-8)
    assert not coarse.koz_refinement_applied

    fine_clearances_m = [plans[count].metrics.min_koz_clearance_refined_m for count in counts[1:]]
    for count, clearance_m in zip(counts[1:], fine_clearances_m, strict=True):
        # Every fine grid finds a genuinely interior minimum and refines it. Measured depth
        # below the endpoint: 1.074e-03 m at all three counts.
        assert plans[count].metrics.koz_refinement_applied
        assert clearance_m < endpoint_clearance_m - 1e-4, count
    # And they agree with each other: measured spread 2.1e-11 m across 1009/2003/3001.
    # 1e-6 m is five decades of headroom, and still 1e-8 of the 200 m sphere.
    assert max(fine_clearances_m) - min(fine_clearances_m) < 1e-6

    # --- Maximum closing velocity: a supremum at an activation boundary -----------------
    supremum_m_s = _activation_crossing_closing_velocity_m_s(uncorrected_plan)
    # Measured 0.2288959 m/s by bisection, against 0.2286076 m/s from the default grid.
    assert 0.2 < supremum_m_s < 0.25

    errors_m_s = {}
    for count in counts:
        maximum_m_s = plans[count].metrics.max_closing_velocity_m_s
        assert maximum_m_s is not None
        # Approached from below, never exceeded: a grid maximum of a decreasing function
        # cannot beat the value at the boundary it starts from.
        assert maximum_m_s <= supremum_m_s
        errors_m_s[count] = supremum_m_s - maximum_m_s
        # The predicted bound is spacing x slope, with the slope measured at
        # 4.4e-04 m/s per second: eps <= 1.23 / (N - 1) m/s. Measured errors 5.271e-03,
        # 4.746e-04, 4.660e-04, 2.883e-04 against bounds 1.23e-02, 1.22e-03, 6.14e-04,
        # 4.10e-04 -- inside by 2.1x to 2.6x at every count.
        assert errors_m_s[count] < 1.23 / (count - 1), count

    # Convergence, stated as behaviour rather than as a single threshold: refining by a
    # factor of ~30 in sample count cuts the error by at least 5x. Measured 5.271e-03 ->
    # 2.883e-04, a factor of 18.3.
    assert errors_m_s[counts[0]] / errors_m_s[counts[-1]] > 5.0

    # Finally, the shipped default sits on the converged side of both criteria.
    default_plan = plan_rendezvous(
        baseline_config,
        base_dir=tmp_path / "default",
        make_plots=False,
        correct=False,
        n_samples=DEFAULT_SAMPLE_COUNT,
    )
    default_maximum_m_s = default_plan.metrics.max_closing_velocity_m_s
    assert default_maximum_m_s is not None
    assert default_plan.metrics.koz_refinement_applied
    assert default_plan.metrics.min_koz_clearance_refined_m == pytest.approx(
        fine_clearances_m[0], abs=1e-6
    )
    # 0.5 % of the 0.1 m/s limit the number is judged against. Measured 2.883e-04 m/s.
    assert supremum_m_s - default_maximum_m_s < 0.005 * (
        baseline_config.constraints.max_closing_velocity_m_s
    )


# ======================================================================================
# Determinism (N-1)
# ======================================================================================


@pytest.mark.integration
def test_same_config_and_seed_produce_identical_metrics(
    baseline_config: ScenarioConfig, tmp_path: Path
) -> None:
    """Two runs of the same scenario at the same seed write the same ``metrics.json``.

    Compared as parsed content, then as bytes. ``metrics.json`` carries no timestamp at all;
    the timestamp lives in ``provenance.json``, deliberately outside the config hash, so a
    re-run is idempotent in its path rather than accumulating one directory per invocation.
    """
    first = plan_rendezvous(
        baseline_config, base_dir=tmp_path / "a", make_plots=False, n_samples=301
    )
    second = plan_rendezvous(
        baseline_config, base_dir=tmp_path / "b", make_plots=False, n_samples=301
    )

    assert json.loads(first.metrics_path.read_text()) == json.loads(second.metrics_path.read_text())
    # Stronger than equality of parsed content: floats are written at shortest-round-trip
    # repr, so identical bytes means identical doubles, not merely close ones.
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    assert read_metrics(first.run_dir) == read_metrics(second.run_dir)

    # The run directory name is content-addressed, so both runs chose the same leaf.
    assert first.run_dir.name == second.run_dir.name

    first_provenance = json.loads((first.run_dir / "provenance.json").read_text())
    second_provenance = json.loads((second.run_dir / "provenance.json").read_text())
    differing = {key for key in first_provenance if first_provenance[key] != second_provenance[key]}
    # Measured: the timestamp is the only field that moves, and only because the two runs
    # happened at different instants. Everything identifying the run is stable.
    assert differing <= {"created_utc"}
    assert first_provenance["config_hash"] == first.metrics.config_hash


@pytest.mark.integration
def test_seed_override_renames_the_run_without_changing_the_physics(
    baseline_config: ScenarioConfig, tmp_path: Path
) -> None:
    """``seed`` names the run and rides in provenance; it changes no computed number.

    Nothing in a two-impulse plan is stochastic, so this is the honest behaviour rather than
    a bug: pinning it here means a future stochastic entry point cannot quietly start
    ignoring the seed, and cannot quietly start perturbing a deterministic plan either.
    """
    default_seed = plan_rendezvous(
        baseline_config, base_dir=tmp_path / "a", make_plots=False, n_samples=301
    )
    overridden = plan_rendezvous(
        baseline_config, base_dir=tmp_path / "b", make_plots=False, n_samples=301, seed=7
    )

    assert default_seed.seed == baseline_config.seed == 42
    assert overridden.seed == 7
    assert overridden.metrics.seed == 7
    assert overridden.run_dir.name.endswith("-7")
    assert default_seed.run_dir.name.endswith("-42")
    # The config hash covers the scenario, not the invocation, so it must not move.
    assert overridden.metrics.config_hash == default_seed.metrics.config_hash
    assert json.loads((overridden.run_dir / "provenance.json").read_text())["seed"] == 7

    assert overridden.metrics.total_delta_v_m_s == default_seed.metrics.total_delta_v_m_s
    assert (
        overridden.metrics.min_koz_clearance_refined_m
        == default_seed.metrics.min_koz_clearance_refined_m
    )
    assert (
        overridden.metrics.constraint_violation_count
        == default_seed.metrics.constraint_violation_count
    )


# ======================================================================================
# The baseline scenario violates its own corridor, on purpose
# ======================================================================================


@pytest.mark.integration
def test_shipped_baseline_violates_its_corridor_and_still_writes_everything(
    corrected_plan: RendezvousPlan,
) -> None:
    """The flagship scenario fails a constraint by geometry, and reports it completely.

    A half-period two-impulse V-bar hop bulges radially by exactly ``dy/4`` -- 187.5 m for
    the baseline's 750 m hop -- independent of altitude and of transfer time
    (``docs/project1/math-model.md``, model M4 corollary). Against a 10 deg corridor that is
    a violation the manoeuvre cannot avoid, so the requirement is that it be *reported*, not
    that it be tuned away. Widening the cone to make this test pass would be the failure.
    """
    metrics = corrected_plan.metrics
    assert not corrected_plan.all_constraints_satisfied
    assert metrics.constraint_violation_count > 0
    assert metrics.first_violation_time_s is not None

    outcomes = {result.name: result for result in corrected_plan.report.results}
    assert set(outcomes) == {
        "keep_out_sphere",
        "approach_ellipsoid",
        "approach_corridor",
        "closing_velocity",
    }
    assert outcomes["keep_out_sphere"].satisfied
    assert outcomes["approach_ellipsoid"].satisfied
    assert not outcomes["approach_corridor"].satisfied
    assert not outcomes["closing_velocity"].satisfied

    # The radial bulge is dy/4 = 187.5 m exactly, and the worst cone angle follows from the
    # geometry. Measured 20.5585 deg; the M4 corollary quotes 20.56 deg for the CW arc.
    assert metrics.max_corridor_angle_rad is not None
    assert math.degrees(metrics.max_corridor_angle_rad) == pytest.approx(20.56, abs=0.02)
    radial_excursion_m = float(np.abs(corrected_plan.states_hill[:, 0]).max())
    hop_length_m = _hop_length_m(corrected_plan.config)
    # Measured 187.5 m against dy/4 = 187.5 m. 0.2 % covers the nonlinear correction's
    # effect on an otherwise exactly linear result.
    assert radial_excursion_m == pytest.approx(hop_length_m / 4.0, rel=2e-3)

    # A majority of samples are outside the corridor: measured 1252 of 2001 (62.6 %). This
    # is a wholesale breach of the constraint, not a boundary graze.
    assert outcomes["approach_corridor"].n_violating_samples > 0.5 * corrected_plan.n_samples

    # Everything is still on disk. A violated constraint is a result, not a crash.
    assert corrected_plan.metrics_path.is_file()
    assert (corrected_plan.run_dir / "provenance.json").is_file()
    written = read_metrics(corrected_plan.run_dir)
    assert not written.all_constraints_satisfied
    assert written.constraint_violation_count == metrics.constraint_violation_count


@pytest.mark.integration
def test_a_scenario_that_respects_its_corridor_reports_no_violation(
    short_hop_config: ScenarioConfig, tmp_path: Path
) -> None:
    """Complement: shorten the hop and every constraint passes on the same machinery.

    Without this, the baseline test above would still pass if the corridor evaluation were
    hard-wired to fail. The hop is 100 m rather than 750 m, so the ``dy/4`` bulge is 25 m and
    the worst cone angle is 1.51 deg against the same 10 deg limit.
    """
    plan = plan_rendezvous(short_hop_config, base_dir=tmp_path, make_plots=False)
    assert plan.all_constraints_satisfied
    assert plan.violation_count == 0
    assert plan.metrics.first_violation_time_s is None
    assert plan.metrics.max_corridor_angle_rad is not None
    # Measured 1.5142 deg.
    assert math.degrees(plan.metrics.max_corridor_angle_rad) < 2.0
    # The closing-velocity limit is genuinely active here rather than vacuously satisfied:
    # a "pass" from zero evaluated samples would report None. Measured 0.056368 m/s against
    # the 0.1 m/s limit.
    assert plan.metrics.max_closing_velocity_m_s is not None
    assert 0.05 < plan.metrics.max_closing_velocity_m_s < 0.1


# ======================================================================================
# Two-body invariance: the orbit's orientation is a coordinate choice
# ======================================================================================


@pytest.mark.integration
def test_inclination_does_not_change_the_plan(
    baseline_config: ScenarioConfig, tmp_path: Path
) -> None:
    """Relative motion in a spherical field depends on orbit size, never on orientation.

    :func:`~rpo_traj.plan.target_state_eci` picks a right ascension and argument of latitude
    of zero and tilts the velocity by the configured inclination. If the Hill frame were
    built wrongly -- a sign on the orbit normal, a dropped ``omega x r`` transport term --
    the inclination would leak into the answer. It does not, and this test is what says so
    rather than the docstring.
    """
    equatorial_config = ScenarioConfig.model_validate(
        {
            **baseline_config.model_dump(mode="python"),
            "orbit": {"altitude_m": 420000.0, "inclination_deg": 0.0},
        }
    )
    inclined = plan_rendezvous(
        baseline_config, base_dir=tmp_path / "i51", make_plots=False, n_samples=301
    )
    equatorial = plan_rendezvous(
        equatorial_config, base_dir=tmp_path / "i0", make_plots=False, n_samples=301
    )

    # Measured absolute differences: delta-v 4.1e-10 m/s on 0.42 m/s (1.0e-09 relative),
    # keep-out clearance 5.5e-07 m on 50 m, max closing velocity 2.4e-09 m/s. All are
    # integrator round-off, and the violation count is identical.
    assert equatorial.metrics.total_delta_v_m_s == pytest.approx(
        inclined.metrics.total_delta_v_m_s, rel=1e-8
    )
    assert equatorial.metrics.min_koz_clearance_refined_m == pytest.approx(
        inclined.metrics.min_koz_clearance_refined_m, abs=1e-5
    )
    assert equatorial.metrics.max_closing_velocity_m_s == pytest.approx(
        inclined.metrics.max_closing_velocity_m_s, abs=1e-7
    )
    assert (
        equatorial.metrics.constraint_violation_count == inclined.metrics.constraint_violation_count
    )
    # Terminal position error is deliberately NOT compared: it is the shooting solver's
    # residual, which lands wherever the Newton step happens to (measured 1.3e-07 versus
    # 5.5e-07 m). Both are five decades below the 1e-03 m tolerance asked for; neither is a
    # physical quantity of the scenario.


@pytest.mark.unit
def test_target_state_eci_is_circular_and_correctly_inclined(
    baseline_config: ScenarioConfig,
) -> None:
    """The epoch state is a circular orbit at the configured radius and inclination."""
    r_eci_m, v_eci_m_s = target_state_eci(baseline_config.orbit)
    radius_m = baseline_config.orbit.semi_major_axis_m

    assert float(np.linalg.norm(r_eci_m)) == pytest.approx(radius_m, rel=1e-15)
    # Circular: r . v = 0 exactly by construction, and |v| = sqrt(mu/a).
    assert float(np.dot(r_eci_m, v_eci_m_s)) == 0.0
    assert float(np.linalg.norm(v_eci_m_s)) == pytest.approx(
        math.sqrt(MU_EARTH_M3_S2 / radius_m), rel=1e-9
    )
    # Inclination is the angle between the orbit normal and the pole.
    angular_momentum = np.cross(r_eci_m, v_eci_m_s)
    inclination_rad = math.acos(
        float(angular_momentum[2]) / float(np.linalg.norm(angular_momentum))
    )
    assert math.degrees(inclination_rad) == pytest.approx(
        baseline_config.orbit.inclination_deg, abs=1e-9
    )
    # And the mean motion the plan uses is the one this orbit actually has.
    period_s = 2.0 * math.pi * radius_m / float(np.linalg.norm(v_eci_m_s))
    assert period_s == pytest.approx(baseline_config.orbit.orbital_period_s, rel=1e-9)


# ======================================================================================
# Figures
# ======================================================================================


@pytest.mark.integration
def test_plots_are_written_when_requested_and_skipped_when_not(
    short_hop_config: ScenarioConfig, tmp_path: Path
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None

    with_plots = plan_rendezvous(
        short_hop_config, base_dir=tmp_path / "yes", make_plots=True, n_samples=201
    )
    assert set(with_plots.figure_paths) == {
        "hill_trajectory",
        "range_and_rate",
        "delta_v_budget",
    }
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in with_plots.figure_paths.values()
    )
    assert with_plots.plots_skipped_reason is None

    without_plots = plan_rendezvous(
        short_hop_config, base_dir=tmp_path / "no", make_plots=False, n_samples=201
    )
    assert without_plots.figure_paths == {}
    assert without_plots.plots_skipped_reason == "figures not requested"
    assert not list(without_plots.run_dir.glob("*.png"))


# ======================================================================================
# Raise paths
# ======================================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n_samples", "match"),
    [
        (MIN_SAMPLE_COUNT - 1, "below MIN_SAMPLE_COUNT"),
        (0, "below MIN_SAMPLE_COUNT"),
        (-5, "below MIN_SAMPLE_COUNT"),
        (2001.5, "whole number of samples"),
        ("many", "integer number of trajectory samples"),
        (None, "integer number of trajectory samples"),
    ],
)
def test_bad_sample_count_raises(
    baseline_config: ScenarioConfig, tmp_path: Path, n_samples: object, match: str
) -> None:
    with pytest.raises(PlanningError, match=match):
        plan_rendezvous(
            baseline_config,
            base_dir=tmp_path,
            make_plots=False,
            n_samples=n_samples,  # type: ignore[arg-type]
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("seed", "match"),
    [(-1, "must be non-negative"), (-99, "must be non-negative"), ("x", "non-negative integer")],
)
def test_bad_seed_raises(
    baseline_config: ScenarioConfig, tmp_path: Path, seed: object, match: str
) -> None:
    with pytest.raises(PlanningError, match=match):
        plan_rendezvous(
            baseline_config,
            base_dir=tmp_path,
            make_plots=False,
            n_samples=11,
            seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.unit
@pytest.mark.parametrize("mu_m3_s2", [0.0, -1.0, float("nan"), float("inf")])
def test_target_state_eci_rejects_a_bad_gravitational_parameter(
    baseline_config: ScenarioConfig, mu_m3_s2: float
) -> None:
    with pytest.raises(PlanningError, match="finite positive gravitational parameter"):
        target_state_eci(baseline_config.orbit, mu_m3_s2)


@pytest.mark.integration
def test_a_targeting_failure_is_raised_not_swallowed(
    baseline_config: ScenarioConfig, tmp_path: Path
) -> None:
    """An unreachable tolerance must raise, not return a best-effort plan.

    The nonlinear oracle's own noise floor is ~5e-09 m, so 1e-12 m cannot be delivered. The
    solver correctly refuses; :func:`plan_rendezvous` must let that refusal through with its
    residual history intact rather than rewrapping it or, worse, planning around it.
    """
    with pytest.raises(TargetingConvergenceError) as excinfo:
        plan_rendezvous(
            baseline_config,
            base_dir=tmp_path,
            make_plots=False,
            n_samples=11,
            targeting_tolerance_m=1e-12,
        )
    error = excinfo.value
    # Measured: stalls after 6 iterations at a residual of 5.771e-09 m, which is the floor
    # the module documents. The history is the diagnosis, and it must survive the trip.
    assert error.iterations >= 1
    assert error.residual_m < 1e-6
    assert len(error.residual_history_m) >= 2
    assert error.residual_history_m[0] > error.residual_history_m[-1]

    # Nothing was written: the run directory is created only after the solve succeeds, so a
    # failed plan leaves no half-finished record for a later reader to mistake for one.
    assert not list(tmp_path.rglob("metrics.json"))


@pytest.mark.integration
def test_a_scenario_config_error_propagates_from_the_output_stage(
    baseline_config: ScenarioConfig, tmp_path: Path
) -> None:
    """A run directory that cannot be created is an error, not a silently skipped write."""
    from rpo_core.config import ScenarioConfigError

    blocker = tmp_path / "runs"
    blocker.write_text("not a directory\n")
    with pytest.raises(ScenarioConfigError, match="cannot create run directory"):
        plan_rendezvous(baseline_config, base_dir=blocker, make_plots=False, n_samples=11)


@pytest.mark.integration
def test_corrected_and_raw_runs_do_not_overwrite_each_other(tmp_path):
    """Two different results must not resolve to one run directory.

    Regression guard. The run directory is content-addressed on the scenario, and
    `correct` is a run option rather than part of the scenario, so both modes previously
    landed on the same path and the second silently clobbered the first. Silent, and
    exactly the kind of thing that makes a results table untraceable.
    """
    config = load_scenario(BASELINE_CONFIG_PATH)
    corrected = plan_rendezvous(
        config, seed=42, correct=True, base_dir=tmp_path, make_plots=False, n_samples=201
    )
    raw = plan_rendezvous(
        config, seed=42, correct=False, base_dir=tmp_path, make_plots=False, n_samples=201
    )

    assert corrected.run_dir != raw.run_dir
    assert corrected.metrics_path.exists() and raw.metrics_path.exists()
    # And the two really do differ, so the separation is load-bearing rather than cosmetic.
    assert raw.corrected_terminal_miss_m is None or (
        corrected.metrics.terminal_position_error_m < raw.metrics.terminal_position_error_m
    )
