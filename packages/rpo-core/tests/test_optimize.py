"""The Δv-versus-TOF trade, the optimiser, dominance, and the comparison table.

The most important test in this file is
:func:`test_table_reports_cw_invalid_at_ten_kilometres_and_valid_at_two_fifty`. Everything
else supports it. A comparison table that shows Clohessy-Wiltshire winning on Δv at a
separation where CW's own linearisation error is 277 m -- larger than the keep-out sphere it
is supposed to respect -- is worse than no table at all, because it launders a modelling
failure into a recommendation. That test is the guard against shipping one.

Three properties keep the rest of the file from grading its own homework:

* The optimiser is cross-checked against a dense brute-force sweep of the same objective,
  and must be **at least as good**, not merely close. A minimiser that returned its starting
  bound would pass an "approximately equal" assertion on a flat objective and fail this one.
* The singular-time exclusion is checked in both directions: guarded, the sweep's worst Δv
  is 215 m/s; unguarded, on the same times, it is 5.3e+05 m/s. Asserting only the first
  would still pass if the guard were removed and the grid happened to miss the pole.
* Dominance is tested against hand-built point sets whose answer is known by inspection,
  including the two cases that separate weak from strict dominance.

Numbers quoted in tolerance comments were measured on this machine before the bound was
chosen; the headroom is stated in each case.
"""

import itertools
import math

import numpy as np
import pytest
from rpo_core.baselines import (
    DEFAULT_CW_TOLERANCE_M,
    Method,
    RendezvousProblem,
    Validity,
)
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from rpo_core.optimize import (
    SWEEP_CROSS_TRACK_SIN_TOL,
    SWEEP_MAX_CONDITION,
    DeltaVConvergenceError,
    DeltaVSweep,
    NoRegularIntervalError,
    compare_baselines,
    delta_v_vs_tof,
    dominates,
    minimise_delta_v,
    pareto_front,
    phasing_delta_v_vs_tof,
    regular_intervals_s,
    singular_transfer_times_s,
)
from rpo_core.relative.cw import (
    DEFAULT_CROSS_TRACK_SIN_TOL,
    SINGULARITY_CONDITION_LIMIT,
)

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
V_CIRCULAR = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
N_RAD_S = mean_motion_rad_s(A_ISS_M)
PERIOD_S = orbital_period_s(A_ISS_M)

_INC = math.radians(51.6)
R_TARGET = np.array([A_ISS_M, 0.0, 0.0])
V_TARGET = V_CIRCULAR * np.array([0.0, math.cos(_INC), math.sin(_INC)])
AT_REST = np.zeros(3)

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
    """Return a transfer that excites both singular families.

    Radial content makes the whole-period in-plane singularity bite; cross-track content
    makes the half-period one bite. A pure V-bar hop excites neither, which is exactly why
    the sweep tests need this problem and the table tests do not.
    """
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
# Singular transfer times and the intervals between them
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_singular_times_are_the_half_period_multiples_in_range():
    """Both families at once: ``k*T`` (in-plane) is the even subset of ``k*T/2``.

    The range deliberately stops at 1.9 periods rather than 2.0. The interval is closed and
    ``problem.period_s`` comes from the target's specific energy rather than from the literal
    semi-major axis -- they agree to 3.3e-16 relative, which is exactly enough to make a
    bound sitting *on* a multiple round either way. Pinning an endpoint whose answer depends
    on the last bit would be testing floating point, not the physics; the inclusive case gets
    its own test below, driven from the problem's own period.
    """
    problem = three_dimensional_problem()
    period_s = problem.period_s
    times = singular_transfer_times_s(problem, 0.02 * period_s, 1.9 * period_s)
    expected = (0.5 * period_s, 1.0 * period_s, 1.5 * period_s)
    assert times == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.unit
def test_singular_times_include_a_bound_that_lands_exactly_on_a_multiple():
    """The range is a closed interval, and the documented behaviour is inclusion."""
    problem = three_dimensional_problem()
    half_period_s = 0.5 * problem.period_s
    times = singular_transfer_times_s(problem, half_period_s, 2.0 * half_period_s)
    assert times == pytest.approx((half_period_s, 2.0 * half_period_s), rel=1.0e-12)


@pytest.mark.unit
def test_singular_times_are_empty_inside_a_regular_stretch():
    """Complement: the function must not invent singularities where there are none."""
    problem = three_dimensional_problem()
    assert singular_transfer_times_s(problem, 0.05 * PERIOD_S, 0.45 * PERIOD_S) == ()


@pytest.mark.unit
def test_regular_intervals_exclude_a_guard_band_around_every_singular_time():
    problem = three_dimensional_problem()
    period_s = problem.period_s
    intervals = regular_intervals_s(problem, 0.02 * period_s, 1.9 * period_s)
    fractions = [value / period_s for pair in intervals for value in pair]
    assert fractions == pytest.approx(
        [0.02, 0.499, 0.501, 0.999, 1.001, 1.499, 1.501, 1.9], rel=1.0e-9
    )
    # Disjoint and ascending.
    for (_, hi), (lo, _) in itertools.pairwise(intervals):
        assert lo > hi


@pytest.mark.unit
def test_regular_intervals_collapse_to_nothing_inside_a_guard_band():
    """Complement: a range entirely inside a guard band must yield no interval at all."""
    problem = three_dimensional_problem()
    assert regular_intervals_s(problem, 0.4999 * PERIOD_S, 0.5001 * PERIOD_S) == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lo", "hi", "match"),
    [
        (-1.0, 100.0, r"tof_lo_s must be finite and strictly positive"),
        (100.0, 100.0, r"tof_hi_s must exceed tof_lo_s"),
        (200.0, 100.0, r"tof_hi_s must exceed tof_lo_s"),
    ],
)
def test_singular_times_reject_a_malformed_range(lo, hi, match):
    with pytest.raises(ValueError, match=match):
        singular_transfer_times_s(three_dimensional_problem(), lo, hi)


@pytest.mark.unit
def test_regular_intervals_reject_a_non_positive_guard():
    with pytest.raises(ValueError, match=r"guard_fraction must be finite and strictly positive"):
        regular_intervals_s(three_dimensional_problem(), 100.0, 5000.0, guard_fraction=0.0)


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_sweep_returns_finite_delta_v_and_records_every_exclusion():
    problem = three_dimensional_problem()
    times = np.linspace(0.02 * PERIOD_S, 2.0 * PERIOD_S, 401)
    sweep = delta_v_vs_tof(problem, times)

    assert isinstance(sweep, DeltaVSweep)
    assert sweep.method is Method.CW_TWO_IMPULSE
    assert np.all(np.isfinite(sweep.delta_v_m_s))
    assert np.all(sweep.delta_v_m_s > 0.0)
    assert sweep.tof_s.size + len(sweep.excluded) == times.size
    assert np.all(np.diff(sweep.tof_s) > 0.0)
    for excluded in sweep.excluded:
        assert excluded.reason.strip() != ""


@pytest.mark.unit
def test_sweep_excludes_every_exact_singular_transfer_time():
    """At ``k*T/2`` exactly, a transfer with radial and cross-track content has no solution."""
    problem = three_dimensional_problem()
    times = np.array([0.5, 1.0, 1.5, 2.0]) * PERIOD_S
    sweep = delta_v_vs_tof(problem, times)
    assert sweep.tof_s.size == 0
    assert len(sweep.excluded) == 4
    assert sorted(excluded.tof_s for excluded in sweep.excluded) == pytest.approx(
        sorted(times), rel=1.0e-15
    )


@pytest.mark.unit
def test_sweep_near_a_whole_period_returns_no_nan_and_no_absurd_delta_v():
    """The headline sweep property, checked in both directions.

    Guarded, the worst retained Δv over a grid that includes points 1e-07 of a period from
    ``k*T`` is 215 m/s. Unguarded, on exactly the same times, it is 5.29e+05 m/s -- three
    thousand times larger. Asserting only the guarded bound would still pass with the guard
    removed if the grid happened to miss the pole, so the complement is what makes this a
    knife edge.
    """
    problem = three_dimensional_problem()
    times = np.concatenate(
        (
            np.linspace(0.02 * PERIOD_S, 2.0 * PERIOD_S, 401),
            np.array([0.5, 1.0, 1.5, 2.0]) * PERIOD_S,
            np.array([1.0 - 1.0e-7, 1.0 + 1.0e-7, 0.5 - 1.0e-7]) * PERIOD_S,
        )
    )

    guarded = delta_v_vs_tof(problem, times)
    assert not np.any(np.isnan(guarded.delta_v_m_s))
    # Measured worst retained value 215.12 m/s; 1e3 leaves 4.6x headroom and is still three
    # orders below the unguarded worst case.
    assert float(guarded.delta_v_m_s.max()) < 1.0e3

    unguarded = delta_v_vs_tof(
        problem,
        times,
        max_condition=SINGULARITY_CONDITION_LIMIT,
        cross_track_sin_tol=DEFAULT_CROSS_TRACK_SIN_TOL,
    )
    # Measured 5.29e+05 m/s. If this ever dropped below the guarded bound the guard would
    # be doing nothing and the test above would be vacuous.
    assert float(unguarded.delta_v_m_s.max()) > 1.0e5
    assert float(unguarded.delta_v_m_s.max()) > 100.0 * float(guarded.delta_v_m_s.max())


@pytest.mark.unit
def test_sweep_guards_cut_where_they_are_documented_to_cut():
    """The exclusion boundary must follow the measured ``cond ~ 3/(1 - t/T)`` law.

    Measured: 4.0e-04 of a period from ``T`` is retained at 132 m/s, 3.0e-04 is excluded
    (condition number crosses 1e4 there). The cross-track cut sits at ``|sin| = 3e-04``,
    i.e. 9.5e-05 of a half period.
    """
    problem = three_dimensional_problem()
    retained = delta_v_vs_tof(problem, np.array([(1.0 - 4.0e-4) * PERIOD_S]))
    excluded = delta_v_vs_tof(problem, np.array([(1.0 - 3.0e-4) * PERIOD_S]))
    assert retained.tof_s.size == 1
    assert excluded.tof_s.size == 0
    assert "SingularTransferTimeError" in excluded.excluded[0].reason

    cross_track_excluded = delta_v_vs_tof(problem, np.array([(1.0 - 5.0e-5) * 0.5 * PERIOD_S]))
    assert cross_track_excluded.tof_s.size == 0
    assert "InfeasibleTransferError" in cross_track_excluded.excluded[0].reason


@pytest.mark.unit
def test_sweep_keeps_the_half_period_hop_that_a_time_based_rule_would_throw_away():
    """A pure V-bar hop is well posed at ``T/2`` and at ``T``; the guards must let it through.

    This is why the exclusion is measured per sample rather than applied by transfer time.
    Excluding a fixed neighbourhood of ``k*T/2`` would reject the suite's flagship
    manoeuvre.
    """
    problem = hop_problem(1000.0)
    sweep = delta_v_vs_tof(problem, np.array([0.5, 1.5]) * PERIOD_S)
    assert sweep.tof_s.size == 2
    assert sweep.excluded == ()
    # The half-period value is the closed form n*dy/2 from math-model.md M4.
    assert float(sweep.delta_v_m_s[0]) == pytest.approx(N_RAD_S * 750.0 / 2.0, rel=1.0e-12)


@pytest.mark.unit
def test_sweep_minimum_raises_rather_than_reporting_nan_when_everything_was_excluded():
    problem = three_dimensional_problem()
    sweep = delta_v_vs_tof(problem, np.array([0.5, 1.0]) * PERIOD_S)
    with pytest.raises(ValueError, match=r"retained no samples"):
        _ = sweep.minimum


@pytest.mark.unit
@pytest.mark.parametrize(
    ("times", "match"),
    [
        (np.zeros((2, 2)), r"tof_values_s must be a non-empty 1-D array"),
        (np.array([]), r"tof_values_s must be a non-empty 1-D array"),
        (np.array([100.0, math.nan]), r"tof_values_s must be finite"),
        (np.array([100.0, -1.0]), r"tof_values_s must be strictly positive"),
    ],
)
def test_sweep_rejects_malformed_times(times, match):
    with pytest.raises(ValueError, match=match):
        delta_v_vs_tof(three_dimensional_problem(), times)


@pytest.mark.unit
def test_sweep_refuses_a_method_whose_time_of_flight_is_an_output():
    with pytest.raises(ValueError, match=r"has no time of flight to sweep"):
        delta_v_vs_tof(hop_problem(1000.0), np.array([1000.0]), method=Method.PHASING)


@pytest.mark.slow
@pytest.mark.integration
def test_lambert_sweep_tracks_the_cw_sweep_where_cw_is_valid():
    """Two independent methods, one curve. They must agree where the linear model holds.

    At 1 km separation the CW conservative bound is 1.66 m against a 5 m budget, so the two
    Δv curves should differ only by the linearisation. Measured worst relative difference
    over the sampled range: 1.4e-03.
    """
    problem = hop_problem(1000.0)
    times = np.linspace(0.1 * PERIOD_S, 0.9 * PERIOD_S, 40)
    cw = delta_v_vs_tof(problem, times)
    lambert = delta_v_vs_tof(problem, times, method=Method.LAMBERT)
    assert cw.tof_s.size == lambert.tof_s.size == times.size
    relative = np.abs(lambert.delta_v_m_s - cw.delta_v_m_s) / cw.delta_v_m_s
    assert float(relative.max()) < 1.0e-2


@pytest.mark.integration
def test_phasing_sweep_walks_the_cheap_and_slow_end_of_the_trade():
    problem = hop_problem(10_000.0)
    sweep = phasing_delta_v_vs_tof(problem, np.array([1.0, 2.0, 3.0, 5.0, 9.0]))
    assert sweep.method is Method.PHASING
    assert sweep.excluded == ()
    assert np.all(np.diff(sweep.tof_s) > 0.0)
    # Monotone: more time bought less delta-v at every step.
    assert np.all(np.diff(sweep.delta_v_m_s) < 0.0)
    # Measured 0.6197 m/s over 2 orbits down to 0.1118 m/s over 10.
    assert float(sweep.delta_v_m_s[0]) == pytest.approx(0.6197, rel=1.0e-3)
    assert float(sweep.tof_s[0]) == pytest.approx(2.0 * PERIOD_S, rel=1.0e-3)


@pytest.mark.unit
def test_phasing_sweep_records_an_unreachable_revolution_count_without_inventing_a_time():
    """The failed profile has no time of flight, so ``tof_s`` is ``nan`` and says so.

    Reporting a plausible-looking time for a profile that does not exist is the exact
    failure mode this package refuses elsewhere.
    """
    sweep = phasing_delta_v_vs_tof(
        hop_problem(10_000.0),
        np.array([0.0, 3.0]),
        max_drift_radius_fraction=1.0e-9,
    )
    assert sweep.tof_s.size == 0
    assert len(sweep.excluded) == 2
    assert all(math.isnan(excluded.tof_s) for excluded in sweep.excluded)
    assert "drift_revolutions=" in sweep.excluded[0].reason


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.zeros((2, 2)), r"must be a non-empty 1-D array"),
        (np.array([1.0, -1.0]), r"must be finite and non-negative"),
        (np.array([1.0, math.nan]), r"must be finite and non-negative"),
    ],
)
def test_phasing_sweep_rejects_malformed_revolution_counts(values, match):
    with pytest.raises(ValueError, match=match):
        phasing_delta_v_vs_tof(hop_problem(1000.0), values)


# --------------------------------------------------------------------------------------
# The optimiser
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    ("problem_factory", "lo_fraction", "hi_fraction"),
    [
        (three_dimensional_problem, 0.02, 0.999),
        (lambda: hop_problem(1000.0), 0.05, 0.999),
    ],
)
def test_optimiser_agrees_with_a_dense_brute_force_sweep(problem_factory, lo_fraction, hi_fraction):
    """Cross-check against brute force, and require the optimiser to be *at least as good*.

    Equality alone would be satisfied by a minimiser that happened to land on a grid point;
    the ``<=`` is what proves it is actually searching. Measured: the optimiser beats the
    800-point grid by 7.0e-10 m/s (3-D) and 1.9e-07 m/s (V-bar), and its minimising time of
    flight lands 1.4e-05 and 3.3e-04 periods from the grid's, both inside one grid cell of
    1.2e-03 periods.
    """
    problem = problem_factory()
    lo_s, hi_s = lo_fraction * PERIOD_S, hi_fraction * PERIOD_S
    minimum = minimise_delta_v(problem, lo_s, hi_s)

    grid = np.linspace(lo_s, hi_s, 800)
    brute_tof_s, brute_delta_v = delta_v_vs_tof(problem, grid).minimum

    assert minimum.delta_v_m_s <= brute_delta_v
    assert minimum.delta_v_m_s == pytest.approx(brute_delta_v, rel=1.0e-4)
    spacing_s = (hi_s - lo_s) / 799
    assert abs(minimum.tof_s - brute_tof_s) <= 2.0 * spacing_s
    assert lo_s <= minimum.tof_s <= hi_s
    assert minimum.function_evaluations > 0
    assert minimum.interval_s[0] <= minimum.tof_s <= minimum.interval_s[1]


@pytest.mark.integration
def test_optimiser_searches_every_sub_interval_not_just_the_first():
    """The 3-D transfer's global minimum lives in the *second* sub-interval.

    Measured: the best time of flight inside ``(0, T/2)`` costs 0.7099 m/s while the global
    minimum at 0.744 T costs 0.4459 m/s. An optimiser that stopped at the first
    singularity-free stretch would return the former and look entirely plausible.
    """
    problem = three_dimensional_problem()
    minimum = minimise_delta_v(problem, 0.02 * PERIOD_S, 0.999 * PERIOD_S)
    assert minimum.sub_intervals == 2
    assert minimum.tof_s / PERIOD_S > 0.5
    assert minimum.delta_v_m_s == pytest.approx(0.44589, rel=1.0e-4)

    first_only = minimise_delta_v(problem, 0.02 * PERIOD_S, 0.499 * PERIOD_S)
    assert first_only.sub_intervals == 1
    assert first_only.delta_v_m_s > minimum.delta_v_m_s


@pytest.mark.unit
def test_optimiser_raises_rather_than_returning_the_last_iterate():
    """Non-convergence must be a failure, never a number.

    A one-iteration cap makes SciPy report ``success=False``; anything that came back would
    be a point of the bracket wearing the costume of a minimum.
    """
    with pytest.raises(DeltaVConvergenceError, match=r"did not converge") as info:
        minimise_delta_v(hop_problem(1000.0), 0.05 * PERIOD_S, 0.999 * PERIOD_S, max_iterations=1)
    error = info.value
    assert error.iterations >= 1
    assert len(error.interval_s) == 2
    assert error.detail.strip() != ""


@pytest.mark.unit
def test_optimiser_raises_when_the_guards_consume_the_whole_range():
    with pytest.raises(NoRegularIntervalError, match=r"no singularity-free sub-interval") as info:
        minimise_delta_v(three_dimensional_problem(), 0.4999 * PERIOD_S, 0.5001 * PERIOD_S)
    assert info.value.singular_times_s == pytest.approx((0.5 * PERIOD_S,), rel=1.0e-12)
    assert info.value.guard_s > 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tof_lo_s": 0.0}, r"tof_lo_s must be finite and strictly positive"),
        ({"tof_hi_s": 100.0}, r"tof_hi_s must exceed tof_lo_s"),
        ({"xatol_s": 0.0}, r"xatol_s must be finite and strictly positive"),
        ({"max_iterations": 0}, r"max_iterations must be >= 1"),
        ({"method": Method.PHASING}, r"has no time of flight to optimise"),
    ],
)
def test_optimiser_rejects_malformed_arguments(kwargs, match):
    call = {
        "problem": hop_problem(1000.0),
        "tof_lo_s": 0.05 * PERIOD_S,
        "tof_hi_s": 0.45 * PERIOD_S,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        minimise_delta_v(**call)


@pytest.mark.slow
@pytest.mark.integration
def test_optimiser_error_shrinks_as_the_tolerance_tightens():
    """Convergence *behaviour*, not a single hand-picked threshold.

    Two properties, both measured. First, a looser tolerance never reports a *lower* Δv than
    the converged answer -- a reported optimum below the true minimum would mean the search
    had left its interval. Second, the error genuinely shrinks: measured excess over the
    converged value is 4.7e-07 and 1.3e-06 m/s at ``xatol_s`` of 100 s and 30 s, against
    7.8e-16 m/s at 1e-02 s and 1e-03 s -- nine orders of magnitude.

    Step-by-step monotonicity is deliberately *not* asserted, and would be wrong to assert:
    bounded Brent's bracket depends discontinuously on the tolerance, so an intermediate
    tolerance can land closer than the next one down (measured 7.7e-11 at 10 s against
    2.5e-09 at 3 s). Requiring monotonicity there would be a test of luck.
    """
    problem = three_dimensional_problem()
    lo_s, hi_s = 0.501 * PERIOD_S, 0.999 * PERIOD_S
    converged = minimise_delta_v(problem, lo_s, hi_s, xatol_s=1.0e-6).delta_v_m_s

    def excess(tolerance_s: float) -> float:
        found = minimise_delta_v(problem, lo_s, hi_s, xatol_s=tolerance_s)
        return found.delta_v_m_s - converged

    coarse = [excess(tolerance) for tolerance in (100.0, 30.0)]
    fine = [excess(tolerance) for tolerance in (1.0e-2, 1.0e-3)]
    assert min(coarse + fine) >= -1.0e-12
    # Measured ratio 1.6e+09; 100x leaves seven orders of headroom and still rejects a
    # minimiser that ignores its tolerance entirely.
    assert max(coarse) > 100.0 * max(max(fine), 1.0e-15)


# --------------------------------------------------------------------------------------
# Dominance and the Pareto front
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Strictly better on both.
        ((1.0, 100.0), (2.0, 200.0), True),
        # Tie on delta-v, strictly better on time: still dominance.
        ((1.0, 100.0), (1.0, 200.0), True),
        # Tie on time, strictly better on delta-v: still dominance.
        ((1.0, 100.0), (2.0, 100.0), True),
        # Identical points do not dominate each other.
        ((1.0, 100.0), (1.0, 100.0), False),
        # Better on one, worse on the other: the trade, not dominance.
        ((1.0, 200.0), (2.0, 100.0), False),
        ((2.0, 100.0), (1.0, 200.0), False),
        # Strictly worse on both.
        ((2.0, 200.0), (1.0, 100.0), False),
    ],
)
def test_dominance_relation_on_hand_built_pairs(a, b, expected):
    assert dominates(a, b) is expected


@pytest.mark.unit
def test_dominance_is_irreflexive_and_antisymmetric():
    """Two structural properties a sign-flipped implementation would violate immediately."""
    points = [(1.0, 100.0), (2.0, 50.0), (1.5, 75.0), (3.0, 300.0)]
    for point in points:
        assert not dominates(point, point)
    for first in points:
        for second in points:
            if first != second:
                assert not (dominates(first, second) and dominates(second, first))


@pytest.mark.unit
def test_pareto_front_drops_strictly_dominated_points():
    points = [
        (1.0, 300.0),  # cheap and slow -- on the front
        (3.0, 100.0),  # expensive and fast -- on the front
        (2.0, 200.0),  # in between -- on the front
        (4.0, 400.0),  # dominated by all three
        (3.5, 250.0),  # dominated by (2.0, 200.0)
    ]
    assert pareto_front(points) == ((3.0, 100.0), (2.0, 200.0), (1.0, 300.0))


@pytest.mark.unit
def test_pareto_front_handles_ties_on_one_objective():
    """Equal Δv at two times: only the faster survives."""
    assert pareto_front([(1.0, 100.0), (1.0, 200.0)]) == ((1.0, 100.0),)
    assert pareto_front([(1.0, 100.0), (2.0, 100.0)]) == ((1.0, 100.0),)


@pytest.mark.unit
def test_pareto_front_collapses_exact_duplicates_to_one_entry():
    """Under weak dominance identical points survive each other, so dedupe is explicit.

    Without it the front would contain a point twice, and a front is a set.
    """
    assert pareto_front([(1.0, 100.0)] * 4) == ((1.0, 100.0),)
    assert pareto_front([(1.0, 100.0), (2.0, 50.0), (1.0, 100.0), (2.0, 50.0)]) == (
        (2.0, 50.0),
        (1.0, 100.0),
    )


@pytest.mark.unit
def test_pareto_front_handles_empty_and_single_point_inputs():
    assert pareto_front([]) == ()
    assert pareto_front([(1.0, 100.0)]) == ((1.0, 100.0),)


@pytest.mark.unit
def test_pareto_front_of_a_totally_ordered_set_is_one_point():
    """Complement to the trade case: when nothing trades off, the front is a singleton."""
    points = [(float(i), float(i)) for i in range(1, 6)]
    assert pareto_front(points) == ((1.0, 1.0),)


@pytest.mark.unit
def test_pareto_front_of_a_pure_trade_keeps_everything():
    points = [(1.0, 500.0), (2.0, 400.0), (3.0, 300.0), (4.0, 200.0), (5.0, 100.0)]
    assert len(pareto_front(points)) == len(points)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ([(1.0, 2.0, 3.0)], r"points\[0\] must be a \(delta_v_m_s, tof_s\) pair"),
        ([(1.0,)], r"points\[0\] must be a \(delta_v_m_s, tof_s\) pair"),
        ([(1.0, math.nan)], r"points\[0\] must be finite"),
        ([(1.0, 2.0), (math.inf, 3.0)], r"points\[1\] must be finite"),
    ],
)
def test_pareto_front_rejects_malformed_points(bad, match):
    with pytest.raises(ValueError, match=match):
        pareto_front(bad)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        ((1.0, 2.0, 3.0), (1.0, 2.0), r"a must be a \(delta_v_m_s, tof_s\) pair"),
        ((1.0, 2.0), (math.nan, 2.0), r"b must be finite"),
    ],
)
def test_dominates_rejects_malformed_points(a, b, match):
    with pytest.raises(ValueError, match=match):
        dominates(a, b)


@pytest.mark.integration
def test_pareto_front_of_real_baselines_spans_the_whole_trade():
    """Phasing owns the cheap-and-slow end; the fixed-time methods own the fast end."""
    problem = hop_problem(10_000.0)
    phasing = phasing_delta_v_vs_tof(problem, np.array([1.0, 3.0, 9.0]))
    fixed = delta_v_vs_tof(problem, np.linspace(0.2, 0.9, 8) * PERIOD_S)
    front = pareto_front([*phasing.points, *fixed.points])

    assert len(front) >= 4
    # Ascending in time of flight and strictly descending in delta-v: that is what a front is.
    times = [point[1] for point in front]
    costs = [point[0] for point in front]
    assert times == sorted(times)
    assert all(later < earlier for earlier, later in itertools.pairwise(costs))
    # The cheapest member is a phasing profile and the fastest is a fixed-time one.
    assert front[-1][1] > 5.0 * PERIOD_S
    assert front[0][1] < PERIOD_S


# --------------------------------------------------------------------------------------
# The comparison table
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_table_reports_cw_invalid_at_ten_kilometres_and_valid_at_two_fifty():
    """The anti-misleading-benchmark test. The most important one in this file.

    At 10 km the CW conservative bound is 166 m over 0.4 orbits against a 5 m budget, and
    the measured terminal miss is 73 m -- larger than a third of the 200 m keep-out sphere.
    The rendered table must say INVALID next to that row, in the row's own words, so nobody
    reads CW's lower Δv as a recommendation. At 250 m the bound is 0.104 m and CW is a
    perfectly good design tool, so the same table must say VALID.

    Both halves matter. A build that always printed INVALID would pass the first assertion
    and be useless; a build that always printed VALID would pass the second and be
    dangerous.
    """
    far = compare_baselines(hop_problem(10_000.0))
    near = compare_baselines(hop_problem(250.0))

    far_cw = far.by_method(Method.CW_TWO_IMPULSE)
    near_cw = near.by_method(Method.CW_TWO_IMPULSE)
    assert far_cw.validity is Validity.INVALID
    assert near_cw.validity is Validity.VALID

    far_table = far.render_table()
    near_table = near.render_table()

    # Checking the rendered string for "INVALID" alone would be satisfied by the footer,
    # which explains the word in prose on every table including the valid one. The verdict
    # has to be attached to the CW row.
    assert "[INVALID] CW two-impulse:" in far_table
    assert "[INVALID] CW two-impulse:" not in near_table
    assert "[VALID] CW two-impulse:" in near_table

    far_row = next(line for line in far_table.splitlines() if line.startswith("CW two-impulse "))
    near_row = next(line for line in near_table.splitlines() if line.startswith("CW two-impulse "))
    assert far_row.rstrip().endswith("INVALID")
    assert near_row.rstrip().endswith("VALID")

    # And the reason has to carry the numbers that decided it.
    assert "166.4" in far_table
    assert "5 m position-error budget" in far_table


@pytest.mark.integration
def test_table_shows_cw_winning_on_delta_v_exactly_where_it_is_invalid():
    """The situation the validity column exists for, made explicit.

    At 10 km CW reports the lowest Δv of the two fixed-time methods while being outside its
    own envelope. If the table ever gained a "best" column, this is the row it would
    recommend.
    """
    comparison = compare_baselines(hop_problem(10_000.0))
    cw = comparison.by_method(Method.CW_TWO_IMPULSE)
    lambert = comparison.by_method(Method.LAMBERT)

    assert cw.total_delta_v_m_s < lambert.total_delta_v_m_s
    assert cw.validity is Validity.INVALID
    assert lambert.validity is Validity.VALID
    assert cw.terminal_position_error_m > 1.0e6 * lambert.terminal_position_error_m

    # cheapest_valid must skip it, and it must not be reachable from the table itself.
    cheapest = comparison.cheapest_valid
    assert cheapest is not None
    assert cheapest.method is not Method.CW_TWO_IMPULSE
    assert "best" not in comparison.render_table().lower().replace('no "best" column', "")


@pytest.mark.integration
def test_table_has_no_best_column_and_says_why():
    comparison = compare_baselines(hop_problem(10_000.0))
    table = comparison.render_table()
    header = table.splitlines()[6]
    assert "Best" not in header
    assert "Model" in header
    assert 'no "best" column' in table
    assert "Model premise is not accuracy" in table


@pytest.mark.integration
def test_table_contains_one_row_per_method_with_its_measured_numbers():
    problem = hop_problem(10_000.0)
    comparison = compare_baselines(problem)
    table = comparison.render_table()

    assert len(comparison.results) == 4
    for result in comparison.results:
        assert result.method.label in table
        assert result.validity.label in table
    # Header lines quote the scenario so a pasted table is self-describing.
    assert "6,798,137 m" in table
    assert "nonlinear two-body dynamics" in table
    assert "10,000.0 m" in table


@pytest.mark.integration
def test_table_body_rows_line_up_under_the_rule():
    """A table that goes in a README has to survive being pasted into one."""
    table = compare_baselines(hop_problem(10_000.0)).render_table()
    lines = table.splitlines()
    rule = next(line for line in lines if set(line) == {"-"})
    body_start = lines.index(rule) + 1
    body = lines[body_start : body_start + 4]
    for line in body:
        assert len(line) <= len(rule)
    assert all(len(line.split()) >= 9 for line in body)


@pytest.mark.integration
def test_comparison_can_omit_the_corrected_variant():
    comparison = compare_baselines(hop_problem(1000.0), include_corrected=False)
    assert len(comparison.results) == 3
    with pytest.raises(KeyError, match=r"'cw-corrected' was not run"):
        comparison.by_method(Method.CW_CORRECTED)


@pytest.mark.integration
def test_comparison_scores_every_method_by_the_same_nonlinear_flight():
    """The apples-to-apples property, asserted rather than assumed.

    Each result's terminal error must be reproducible from its own reported time of flight
    against the same commanded terminal state -- so a method cannot be scored generously by
    being handed a different arrival target.
    """
    problem = hop_problem(1000.0)
    comparison = compare_baselines(problem)
    for result in comparison.results:
        assert result.terminal_position_error_m >= 0.0
        assert result.terminal_velocity_error_m_s >= 0.0
        assert result.cw_error_bound_m > 0.0
    # Same problem, so every row's CW bound agrees except for its own time of flight.
    fixed_time = [
        result
        for result in comparison.results
        if result.method in (Method.LAMBERT, Method.CW_TWO_IMPULSE, Method.CW_CORRECTED)
    ]
    bounds = {round(result.cw_error_bound_m, 9) for result in fixed_time}
    assert len(bounds) == 1


@pytest.mark.integration
def test_cheapest_valid_is_none_when_nothing_is_valid():
    """Complement: the filter must be able to return nothing rather than falling back."""
    comparison = compare_baselines(hop_problem(10_000.0), cw_tolerance_m=1.0e-12)
    # Force phasing and Lambert out too by asking for a cross-track change phasing cannot do
    # and judging CW against an impossible budget.
    cross_track = RendezvousProblem(
        R_TARGET,
        V_TARGET,
        np.array([0.0, -10_000.0, 0.0]),
        AT_REST,
        np.array([0.0, -2500.0, 80.0]),
        AT_REST,
        GENERIC_TOF_S,
    )
    strict = compare_baselines(cross_track, cw_tolerance_m=1.0e-12)
    assert comparison.cheapest_valid is not None  # Lambert is still valid here.
    assert all(
        result.validity is Validity.INVALID
        for result in strict.results
        if result.method is not Method.LAMBERT
    )


@pytest.mark.unit
def test_sweep_constants_are_tighter_than_the_cw_module_backstops():
    """The sweep guards must be *tighter*, or they would do nothing at all."""
    assert SWEEP_MAX_CONDITION < SINGULARITY_CONDITION_LIMIT
    assert SWEEP_CROSS_TRACK_SIN_TOL > DEFAULT_CROSS_TRACK_SIN_TOL
    assert DEFAULT_CW_TOLERANCE_M == 5.0
