"""RPO safety constraint checks.

Everything asserted here is analytic geometry -- a 30-40-50 right triangle, a cone
half-angle put in by construction, an orthogonality that makes a dot product vanish -- or
a convergence behaviour. Nothing is a golden number read back out of this implementation.

Where a tolerance appears, the comment above it states the value that was measured on this
machine and the headroom the bound leaves. Measurements were taken with the module's own
public API on the exact trajectories used below, printed before the bound was chosen.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
from rpo_core.constraints import (
    ApproachCorridor,
    ApproachEllipsoid,
    ClosingVelocityLimit,
    ConstraintDefinitionError,
    KeepOutSphere,
    TrajectorySamplingError,
    evaluate_approach_corridor,
    evaluate_approach_ellipsoid,
    evaluate_closing_velocity,
    evaluate_constraints,
    evaluate_keep_out_sphere,
    range_rate_m_s,
    separation_m,
)
from rpo_core.exceptions import RpoCoreError

# --------------------------------------------------------------------------------------
# Analytic trajectories
# --------------------------------------------------------------------------------------

#: Straight-line pass geometry. The chaser holds x = +30 m and z = +40 m while sliding
#: along-track at a constant 2 m/s from y = -400 m, so
#:
#:     |rho(t)|**2 = 30**2 + 40**2 + (2t - 400)**2 = 2500 + (2t - 400)**2
#:
#: The closest approach is therefore at exactly t = 200 s and exactly 50 m -- the 30-40-50
#: right triangle. Every keep-out number below is read off that identity, not off a run.
PASS_X_M = 30.0
PASS_Z_M = 40.0
PASS_SPEED_M_S = 2.0
PASS_Y0_M = -400.0
PASS_CLOSEST_TIME_S = 200.0
PASS_CLOSEST_RANGE_M = 50.0

#: Offset-circle geometry: the chaser walks a circle of radius 150 m whose centre sits
#: 200 m out along R-bar, so |rho| ranges over [50, 350] m and the minimum, |200 - 150|,
#: falls at half a revolution. |rho|**2 is *not* quadratic in t here, which is what makes
#: it the honest test of the parabolic refinement.
CIRCLE_CENTRE_M = 200.0
CIRCLE_RADIUS_M = 150.0
CIRCLE_PERIOD_S = 600.0
CIRCLE_RATE_RAD_S = 2.0 * math.pi / CIRCLE_PERIOD_S
CIRCLE_CLOSEST_TIME_S = 0.5 * CIRCLE_PERIOD_S
CIRCLE_CLOSEST_RANGE_M = CIRCLE_CENTRE_M - CIRCLE_RADIUS_M


def straight_line_pass(times_s):
    """Return ``(times, states)`` for the 30-40-50 straight-line pass described above."""
    times = np.asarray(times_s, dtype=np.float64)
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = PASS_X_M
    states[:, 1] = PASS_Y0_M + PASS_SPEED_M_S * times
    states[:, 2] = PASS_Z_M
    states[:, 4] = PASS_SPEED_M_S
    return times, states


def offset_circle(times_s):
    """Return ``(times, states)`` for the offset-circle pass described above."""
    times = np.asarray(times_s, dtype=np.float64)
    phase = CIRCLE_RATE_RAD_S * times
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = CIRCLE_CENTRE_M + CIRCLE_RADIUS_M * np.cos(phase)
    states[:, 1] = CIRCLE_RADIUS_M * np.sin(phase)
    states[:, 3] = -CIRCLE_RADIUS_M * CIRCLE_RATE_RAD_S * np.sin(phase)
    states[:, 4] = CIRCLE_RADIUS_M * CIRCLE_RATE_RAD_S * np.cos(phase)
    return times, states


def single_sample(position_m, velocity_m_s=(0.0, 0.0, 0.0)):
    """Return a one-sample ``(times, states)`` pair at t = 0."""
    states = np.zeros((1, 6), dtype=np.float64)
    states[0, :3] = position_m
    states[0, 3:] = velocity_m_s
    return np.array([0.0]), states


# --------------------------------------------------------------------------------------
# Keep-out sphere: closest approach
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_straight_line_pass_reproduces_the_hand_computed_closest_approach():
    """Closest approach is sqrt(30**2 + 40**2) = 50 m at t = 200 s, by Pythagoras alone."""
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(25.0))

    # The grid lands on t = 200 exactly, so the sampled minimum is the true one and the
    # arithmetic is exact in binary: measured error 0.0, asserted as equality.
    assert result.sampled_min_range_m == PASS_CLOSEST_RANGE_M
    assert result.worst_time_s == PASS_CLOSEST_TIME_S
    assert result.worst_index == 20  # (200 - 0) / 10
    assert result.worst_value == PASS_CLOSEST_RANGE_M - 25.0
    assert result.satisfied
    assert result.n_violating_samples == 0
    assert result.first_violation_time_s is None


@pytest.mark.unit
def test_trajectory_tangent_to_the_sphere_has_exactly_zero_clearance():
    """A sphere of exactly 50 m is tangent to the pass; tangency is not a violation."""
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(PASS_CLOSEST_RANGE_M))

    # Measured: worst_value and refined_clearance_m both come out at exactly 0.0, because
    # 30, 40 and 50 are all exact in binary floating point and so is their Pythagorean
    # identity. Asserted as equality rather than approx, deliberately.
    assert result.worst_value == 0.0
    assert result.refined_clearance_m == 0.0
    assert result.satisfied
    assert result.n_violating_samples == 0


@pytest.mark.unit
def test_tangency_is_a_knife_edge_not_a_plateau():
    """Complement of the tangency test: one micrometre more radius and it is a breach.

    Without this, the tangency test above would still pass if the clearance comparison
    were deleted and ``satisfied`` hard-coded to ``True``.
    """
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    grazed = evaluate_keep_out_sphere(times, states, KeepOutSphere(PASS_CLOSEST_RANGE_M + 1.0e-6))

    assert not grazed.satisfied
    assert grazed.n_violating_samples == 1
    assert grazed.first_violation_time_s == PASS_CLOSEST_TIME_S
    # Measured worst_value -9.999999974752427e-07 against an exact -1e-6; the residual is
    # the float64 representation of 50 + 1e-6, ~2.5e-15 relative. rel=1e-9 leaves ~4e5x.
    assert grazed.worst_value == pytest.approx(-1.0e-6, rel=1.0e-9)


@pytest.mark.unit
def test_sphere_entry_and_exit_give_the_right_count_and_first_violation_time():
    """R = 100 m: the chord is |2t - 400| <= sqrt(100**2 - 50**2), i.e. t in [156.7, 243.3].

    On a 10 s grid that is the nine samples t = 160 .. 240, and the first violation is at
    t = 160 s. Both numbers come from the chord half-width sqrt(7500) / 2 = 43.30127 s,
    not from running the checker.
    """
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(100.0))

    half_chord_s = math.sqrt(100.0**2 - PASS_CLOSEST_RANGE_M**2) / PASS_SPEED_M_S
    entry_s = PASS_CLOSEST_TIME_S - half_chord_s
    exit_s = PASS_CLOSEST_TIME_S + half_chord_s
    expected = int(np.count_nonzero((times > entry_s) & (times < exit_s)))

    assert expected == 9  # guards the arithmetic above against a silent change of grid
    assert result.n_violating_samples == expected
    assert result.first_violation_time_s == 160.0
    assert not result.satisfied
    assert result.worst_value == PASS_CLOSEST_RANGE_M - 100.0


@pytest.mark.unit
def test_a_sample_exactly_on_the_sphere_is_not_counted_as_inside():
    """Complement at the other end: violation is a strict inequality on the clearance."""
    times, states = single_sample((PASS_X_M, 0.0, PASS_Z_M))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(PASS_CLOSEST_RANGE_M))
    assert result.n_violating_samples == 0
    assert result.satisfied


# --------------------------------------------------------------------------------------
# Keep-out sphere: parabolic refinement of the between-sample minimum
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_refinement_recovers_a_between_sample_minimum_of_a_rectilinear_pass():
    """A 30 s grid steps over t = 200 s; the refinement puts the minimum back exactly.

    For constant relative velocity |rho|**2 is exactly quadratic in t, and the module
    refines on |rho|**2 precisely so that this case is recovered rather than approximated.
    """
    times, states = straight_line_pass(np.arange(0.0, 421.0, 30.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(25.0))

    sampled_error = abs(result.sampled_min_range_m - PASS_CLOSEST_RANGE_M)
    refined_error = abs(result.refined_min_range_m - PASS_CLOSEST_RANGE_M)

    assert result.refinement_applied
    # Measured: sampled_error = 3.851648071345039 m (the nearest sample is t = 210 s),
    # refined_error = 0.0 exactly. The bound of 1e-9 m leaves the exact answer nine orders
    # of headroom and would still fail on any real loss of the quadratic identity.
    # sqrt(2500 + (2*210 - 400)**2) = sqrt(2900) is the nearest sample, at t = 210 s.
    assert sampled_error == pytest.approx(math.sqrt(2900.0) - 50.0, rel=1e-12)
    assert refined_error < 1.0e-9
    assert refined_error < sampled_error
    # Measured refined_time_s error vs the analytic 200.0 s: 0.0.
    assert result.refined_time_s == pytest.approx(PASS_CLOSEST_TIME_S, abs=1.0e-9)


@pytest.mark.unit
def test_refinement_beats_the_sampled_minimum_on_a_curved_path():
    """Curved path, where the parabola is only the local model and not the exact one.

    Truth is |200 - 150| = 50 m at t = 300 s. On a 70 s grid the nearest sample is 42 m
    away in range; the refinement gets that down to 1.5 m.
    """
    times, states = offset_circle(np.arange(0.0, 601.0, 70.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(25.0))

    sampled_error = abs(result.sampled_min_range_m - CIRCLE_CLOSEST_RANGE_M)
    refined_error = abs(result.refined_min_range_m - CIRCLE_CLOSEST_RANGE_M)

    assert result.refinement_applied
    # Measured on this grid: sampled_error = 11.7345 m, refined_error = 1.5327 m, a factor
    # of 7.656. Asserting > 3x leaves better than 2x headroom while still failing outright
    # if the refinement were replaced by the sampled value (which would give a factor 1.0).
    assert refined_error < sampled_error / 3.0
    # Measured refined_time_s = 299.38 s against an analytic 300.0 s: 0.62 s out on a 70 s
    # grid. A bound of 5 s leaves 8x and still fails if the vertex were misplaced by a
    # whole sample.
    assert result.refined_time_s == pytest.approx(CIRCLE_CLOSEST_TIME_S, abs=5.0)


@pytest.mark.unit
def test_refined_minimum_converges_as_the_grid_is_tightened():
    """Convergence behaviour, which is a stronger statement than any single tolerance.

    The grid is offset by 1 s so that t = 300 s is never sampled exactly at any step.
    """
    errors = []
    for step_s in (70.0, 35.0, 17.5, 8.75, 4.375):
        times, states = offset_circle(np.arange(1.0, 600.0, step_s))
        result = evaluate_keep_out_sphere(times, states, KeepOutSphere(25.0))
        errors.append(abs(result.refined_min_range_m - CIRCLE_CLOSEST_RANGE_M))

    # Measured: 1.397, 0.2209, 6.152e-4, 1.507e-4, 3.427e-5 m -- monotone, and roughly a
    # factor 4 per halving once the parabola is a good local model. Monotone improvement is
    # asserted rather than a threshold, per docs/CONTRIBUTING.md.
    for coarse, fine in pairwise(errors):
        assert fine < coarse, f"refinement error did not improve: {errors}"
    assert errors[-1] < 1.0e-4  # measured 3.427e-5, ~3x headroom


@pytest.mark.unit
def test_refinement_exposes_a_breach_that_the_sampled_grid_missed():
    """The point of the whole exercise: sampled clearance positive, refined clearance not.

    On the 30 s grid the nearest sample sits 53.85 m out, so a 52 m sphere looks clear from
    the samples alone. The continuous path passes through 50 m, so it is not.
    """
    times, states = straight_line_pass(np.arange(0.0, 421.0, 30.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(52.0))

    assert result.n_violating_samples == 0
    assert result.worst_value > 0.0  # measured +1.8516 m: every sample is outside
    # Measured refined_clearance_m = -2.0 m exactly, i.e. 50 - 52, because the rectilinear
    # refinement is exact. rel=1e-9 leaves ample room for the sqrt round-off.
    assert result.refined_clearance_m == pytest.approx(-2.0, rel=1.0e-9)
    assert not result.satisfied, "a refined breach must not be reported as safe"


@pytest.mark.unit
@pytest.mark.parametrize("step_s", [7.0, 13.0, 29.0, 61.0])
def test_refined_minimum_never_exceeds_the_sampled_minimum(step_s):
    """Invariant: the vertex of an upward parabola lies at or below its middle point.

    Measured over steps 3 s to 89 s on the offset circle, max(refined - sampled) = 0.0.
    A refinement that reported a *larger* minimum would be reporting extra clearance the
    trajectory does not have, which is the one direction of error that is unsafe.
    """
    times, states = offset_circle(np.arange(0.0, 601.0, step_s))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(25.0))
    assert result.refined_min_range_m <= result.sampled_min_range_m


@pytest.mark.unit
def test_endpoint_minimum_is_reported_unrefined():
    """A monotone approach truncated mid-flight has no bracketing triple to fit."""
    times = np.arange(0.0, 101.0, 10.0)
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = 500.0 - 4.0 * times
    states[:, 3] = -4.0
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(50.0))

    assert result.worst_index == times.size - 1
    assert not result.refinement_applied
    assert result.refined_min_range_m == result.sampled_min_range_m
    assert result.refined_time_s == result.worst_time_s
    assert result.sampled_min_range_m == pytest.approx(100.0, abs=1e-12)  # 500 - 4*100


@pytest.mark.unit
def test_single_sample_trajectory_is_scored_without_refinement():
    times, states = single_sample((0.0, -120.0, 0.0))
    result = evaluate_keep_out_sphere(times, states, KeepOutSphere(100.0))
    assert result.worst_index == 0
    assert not result.refinement_applied
    assert result.worst_value == pytest.approx(20.0, abs=1e-12)


# --------------------------------------------------------------------------------------
# Approach ellipsoid
# --------------------------------------------------------------------------------------

ELLIPSOID_AXES_M = (100.0, 400.0, 60.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("position_m", "expected_q"),
    [
        ((100.0, 0.0, 0.0), 1.0),  # on the +x semi-axis
        ((0.0, -400.0, 0.0), 1.0),  # on the -y semi-axis
        ((0.0, 0.0, -60.0), 1.0),  # on the -z semi-axis
        ((50.0, 200.0, 30.0), 0.75),  # half of each axis: 3 * 0.25
        ((0.0, 0.0, 0.0), 0.0),  # at the target
    ],
)
def test_ellipsoid_quadratic_form_matches_the_hand_computed_value(position_m, expected_q):
    """Measured: each of these comes out bit-exact, because the ratios are exact binary."""
    times, states = single_sample(position_m)
    result = evaluate_approach_ellipsoid(times, states, ApproachEllipsoid(ELLIPSOID_AXES_M))
    assert result.worst_value == pytest.approx(expected_q, abs=1e-15)
    assert result.satisfied


@pytest.mark.unit
def test_a_point_on_the_ellipsoid_surface_is_not_a_violation():
    """Boundary case: q == 1 exactly is contained. Complemented by the test below."""
    times, states = single_sample((ELLIPSOID_AXES_M[0], 0.0, 0.0))
    result = evaluate_approach_ellipsoid(times, states, ApproachEllipsoid(ELLIPSOID_AXES_M))
    assert result.worst_value == 1.0
    assert result.n_violating_samples == 0
    assert result.satisfied


@pytest.mark.unit
def test_ellipsoid_containment_is_a_knife_edge():
    """Complement: 0.1% outside the +x semi-axis is a violation, q = 1.002001."""
    times, states = single_sample((ELLIPSOID_AXES_M[0] * 1.001, 0.0, 0.0))
    result = evaluate_approach_ellipsoid(times, states, ApproachEllipsoid(ELLIPSOID_AXES_M))
    assert result.worst_value == pytest.approx(1.001**2, rel=1e-12)
    assert result.n_violating_samples == 1
    assert not result.satisfied
    assert result.first_violation_time_s == 0.0


@pytest.mark.unit
def test_ellipsoid_is_not_a_bounding_box():
    """80% of two semi-axes at once is outside: q = 0.64 + 0.64 = 1.28 > 1.

    An implementation that checked each axis independently -- |x| <= a and |y| <= b and
    |z| <= c -- would call this contained. The quadratic form does not.
    """
    times, states = single_sample((0.8 * ELLIPSOID_AXES_M[0], 0.8 * ELLIPSOID_AXES_M[1], 0.0))
    result = evaluate_approach_ellipsoid(times, states, ApproachEllipsoid(ELLIPSOID_AXES_M))
    assert result.worst_value == pytest.approx(1.28, rel=1e-12)
    assert not result.satisfied


@pytest.mark.unit
def test_equal_semi_axes_degenerate_to_a_sphere_of_that_radius():
    """Limiting case: with a = b = c = R the quadratic form is (|rho| / R)**2."""
    radius_m = 500.0
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    result = evaluate_approach_ellipsoid(
        times, states, ApproachEllipsoid((radius_m, radius_m, radius_m))
    )
    ranges = separation_m(states)
    # Measured worst_value 0.6500000000000001 against the analytic (400 / 500)**2 + ...
    # = 0.65 at t = 0 and t = 400. rel=1e-12 leaves ~1e4x on the observed 1.1e-16 residual.
    assert result.worst_value == pytest.approx(float(np.max(ranges**2)) / radius_m**2, rel=1e-12)
    assert result.worst_value == pytest.approx(0.65, rel=1e-12)


@pytest.mark.unit
def test_ellipsoid_violations_are_counted_across_the_trajectory():
    """Containment inside |y| <= 200 m holds only over the middle of the pass."""
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    # x = 30 and z = 40 are well inside their semi-axes; only y decides.
    result = evaluate_approach_ellipsoid(times, states, ApproachEllipsoid((1000.0, 200.0, 1000.0)))
    y_m = states[:, 1]
    outside = np.abs(y_m) / 200.0
    expected = int(np.count_nonzero(outside**2 + (30.0 / 1000.0) ** 2 + (40.0 / 1000.0) ** 2 > 1.0))
    assert result.n_violating_samples == expected
    assert result.first_violation_time_s == 0.0
    assert not result.satisfied


# --------------------------------------------------------------------------------------
# Approach corridor cone
# --------------------------------------------------------------------------------------

CONE_HALF_ANGLE_RAD = math.radians(10.0)
VBAR_AXIS = (0.0, -1.0, 0.0)


def cone_point(angle_rad, range_m):
    """Return a position exactly ``angle_rad`` off the -V-bar axis at ``range_m``."""
    return (range_m * math.sin(angle_rad), -range_m * math.cos(angle_rad), 0.0)


@pytest.mark.unit
def test_cone_boundary_point_is_not_a_violation():
    """A point placed exactly on the cone surface must be inside it.

    This is the case that motivated deciding on cosines: the arccos round trip returns the
    constructed 10-degree half-angle 2.220446e-16 rad *larger* than it went in, so an
    angle-space strict comparison flags an exactly-tangent trajectory as unsafe.
    """
    times, states = single_sample(cone_point(CONE_HALF_ANGLE_RAD, 100.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)
    )
    assert result.satisfied
    assert result.n_violating_samples == 0
    # The reported angle is still the arccos value. Measured departure from the
    # constructed half-angle: 2.220446e-16 rad. A bound of 1e-14 leaves ~45x.
    assert result.worst_value == pytest.approx(CONE_HALF_ANGLE_RAD, abs=1.0e-14)


@pytest.mark.unit
def test_point_just_inside_the_cone_is_satisfied():
    """1e-6 relative inside the half-angle, i.e. 1.745e-7 rad -- nine orders above noise."""
    times, states = single_sample(cone_point(CONE_HALF_ANGLE_RAD * (1.0 - 1.0e-6), 100.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)
    )
    assert result.satisfied
    assert result.worst_value < CONE_HALF_ANGLE_RAD


@pytest.mark.unit
def test_point_just_outside_the_cone_is_violated():
    """Complement of the two tests above: 1e-6 relative the other way is a violation."""
    times, states = single_sample(cone_point(CONE_HALF_ANGLE_RAD * (1.0 + 1.0e-6), 100.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)
    )
    assert not result.satisfied
    assert result.n_violating_samples == 1
    assert result.first_violation_time_s == 0.0
    assert result.worst_value > CONE_HALF_ANGLE_RAD


@pytest.mark.unit
def test_a_point_far_outside_the_cone_but_beyond_activation_range_is_not_a_violation():
    """R-bar at 1 km is 90 degrees off a V-bar corridor, but a corridor at 1 km is fiction.

    With a 500 m activation range no sample is active, so the constraint is vacuously
    satisfied and the worst-case fields say "never evaluated" rather than fabricating a
    zero.
    """
    times, states = single_sample((1000.0, 0.0, 0.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, 500.0, VBAR_AXIS)
    )
    assert result.satisfied
    assert result.n_violating_samples == 0
    assert result.worst_index == -1
    assert math.isnan(result.worst_value)
    assert math.isnan(result.worst_time_s)
    assert result.first_violation_time_s is None


@pytest.mark.unit
def test_the_activation_range_is_the_only_thing_making_that_point_safe():
    """Complement: the identical sample with an always-on corridor is a 90-degree breach.

    Without this, the previous test would still pass if the cone check were removed.
    """
    times, states = single_sample((1000.0, 0.0, 0.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)
    )
    assert not result.satisfied
    assert result.n_violating_samples == 1
    # R-bar is exactly perpendicular to V-bar; measured pi/2 to the last bit.
    assert result.worst_value == pytest.approx(math.pi / 2.0, abs=1e-15)


@pytest.mark.unit
def test_activation_range_boundary_sample_is_evaluated():
    """``|rho| <= activation_range_m`` is inclusive; a sample exactly at the gate counts."""
    times, states = single_sample((500.0, 0.0, 0.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, 500.0, VBAR_AXIS)
    )
    assert result.worst_index == 0
    assert not result.satisfied


@pytest.mark.unit
def test_the_cone_axis_direction_is_what_distinguishes_leading_from_trailing():
    """A chaser at +V-bar is 180 degrees from a trailing-approach corridor axis.

    Getting the axis sign wrong is the classic Hill-frame error, and it flips a perfectly
    safe approach into a breach and back. Both directions are asserted here.
    """
    trailing = single_sample((0.0, -100.0, 0.0))
    leading = single_sample((0.0, +100.0, 0.0))
    corridor = ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)

    behind = evaluate_approach_corridor(*trailing, corridor)
    ahead = evaluate_approach_corridor(*leading, corridor)

    assert behind.satisfied
    assert behind.worst_value == pytest.approx(0.0, abs=1e-15)
    assert not ahead.satisfied
    assert ahead.worst_value == pytest.approx(math.pi, abs=1e-15)


@pytest.mark.unit
def test_a_non_unit_axis_is_normalised_and_scores_identically():
    """Documented behaviour: the axis is normalised on construction, not rejected."""
    scaled = ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, (0.0, -37.5, 0.0))
    assert scaled.axis_hill == (0.0, -1.0, 0.0)

    times, states = single_sample(cone_point(CONE_HALF_ANGLE_RAD * 0.5, 250.0))
    unit = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)
    )
    assert evaluate_approach_corridor(times, states, scaled).worst_value == unit.worst_value


@pytest.mark.unit
def test_an_oblique_axis_is_normalised_to_unit_length():
    corridor = ApproachCorridor(0.2, math.inf, (0.0, -3.0, 4.0))
    assert corridor.axis_hill == pytest.approx((0.0, -0.6, 0.8), abs=1e-15)
    assert math.hypot(*corridor.axis_hill) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.unit
def test_cone_angle_at_the_target_origin_is_defined_as_zero():
    """|rho| -> 0 leaves the line of sight undefined; the module defines the angle as 0.

    The keep-out sphere is the constraint that has something to say about sitting on the
    target, so the corridor deliberately declines to invent an answer here.
    """
    times, states = single_sample((0.0, 0.0, 0.0))
    result = evaluate_approach_corridor(
        times, states, ApproachCorridor(CONE_HALF_ANGLE_RAD, math.inf, VBAR_AXIS)
    )
    assert result.worst_value == 0.0
    assert result.satisfied


# --------------------------------------------------------------------------------------
# Closing velocity
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_pure_radial_approach_closes_at_exactly_the_approach_speed():
    """Radial rho = [r, 0, 0] with rhodot = [-s, 0, 0] closes at -(rho . rhodot)/|rho| = s."""
    speed_m_s = 0.5
    times = np.arange(0.0, 101.0, 10.0)
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = 1000.0 - speed_m_s * times
    states[:, 3] = -speed_m_s
    result = evaluate_closing_velocity(times, states, ClosingVelocityLimit(1.0, math.inf))

    # Measured error against 0.5: exactly 0.0 -- the projection divides r by itself.
    assert result.worst_value == speed_m_s
    assert result.satisfied
    np.testing.assert_array_equal(-range_rate_m_s(states), np.full(times.size, speed_m_s))


@pytest.mark.unit
def test_circular_relative_orbit_at_constant_range_has_zero_closing_velocity():
    """Circular motion has rho . rhodot = 0, so the range rate vanishes identically."""
    amplitude_m, rate_rad_s = 250.0, 2.0 * math.pi / 900.0
    times = np.arange(0.0, 901.0, 25.0)
    phase = rate_rad_s * times
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = amplitude_m * np.cos(phase)
    states[:, 1] = amplitude_m * np.sin(phase)
    states[:, 3] = -amplitude_m * rate_rad_s * np.sin(phase)
    states[:, 4] = amplitude_m * rate_rad_s * np.cos(phase)

    # Sanity: the construction really is constant-range. Measured max |range - A| =
    # 2.84e-14 m on a 250 m circle, i.e. 1.1e-16 relative -- pure round-off.
    ranges = separation_m(states)
    np.testing.assert_allclose(ranges, amplitude_m, atol=1.0e-12)

    result = evaluate_closing_velocity(times, states, ClosingVelocityLimit(1.0e-6, math.inf))
    # Measured worst closing velocity 1.1368684e-16 m/s. A bound of 1e-15 leaves ~9x and
    # is still 1e13 times tighter than any physically meaningful closing rate.
    assert abs(result.worst_value) < 1.0e-15
    assert result.satisfied
    assert result.n_violating_samples == 0


@pytest.mark.unit
def test_receding_gives_a_negative_closing_velocity():
    """Complement: reverse the velocity and the sign flips, so the limit cannot fire.

    Without this, the radial-approach test would still pass if the negation that turns
    range rate into closing velocity were dropped and the sign convention inverted.
    """
    speed_m_s = 0.5
    times = np.arange(0.0, 101.0, 10.0)
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = 1000.0 + speed_m_s * times
    states[:, 3] = +speed_m_s
    result = evaluate_closing_velocity(times, states, ClosingVelocityLimit(0.0, math.inf))
    assert result.worst_value == -speed_m_s
    assert result.satisfied
    assert result.n_violating_samples == 0


@pytest.mark.unit
def test_closing_velocity_limit_is_a_knife_edge():
    """0.5 m/s of approach passes a 0.6 m/s limit and fails a 0.4 m/s one."""
    speed_m_s = 0.5
    times = np.arange(0.0, 101.0, 10.0)
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = 1000.0 - speed_m_s * times
    states[:, 3] = -speed_m_s

    lenient = evaluate_closing_velocity(times, states, ClosingVelocityLimit(0.6, math.inf))
    strict = evaluate_closing_velocity(times, states, ClosingVelocityLimit(0.4, math.inf))

    assert lenient.satisfied and lenient.n_violating_samples == 0
    assert not strict.satisfied
    assert strict.n_violating_samples == times.size
    assert strict.first_violation_time_s == 0.0


@pytest.mark.unit
def test_closing_velocity_is_not_enforced_beyond_the_activation_range():
    """5 m/s at 5 km is a transfer, not a violation -- until the activation range says so.

    Both halves are asserted, so the test cannot pass with the activation gate removed.
    """
    times = np.arange(0.0, 101.0, 10.0)
    states = np.zeros((times.size, 6), dtype=np.float64)
    states[:, 0] = 5000.0 - 5.0 * times
    states[:, 3] = -5.0

    gated = evaluate_closing_velocity(times, states, ClosingVelocityLimit(1.0, 500.0))
    always = evaluate_closing_velocity(times, states, ClosingVelocityLimit(1.0, math.inf))

    assert gated.satisfied and gated.worst_index == -1 and math.isnan(gated.worst_value)
    assert not always.satisfied
    assert always.n_violating_samples == times.size
    assert always.worst_value == pytest.approx(5.0, abs=1e-15)


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_separation_is_the_euclidean_norm_of_the_position_block():
    """3-4-12-13 Pythagorean quadruple, and the velocity block must not contribute."""
    states = np.array([[3.0, 4.0, 12.0, 900.0, -900.0, 900.0]])
    assert separation_m(states)[0] == 13.0


@pytest.mark.unit
def test_range_rate_agrees_with_a_central_difference_of_the_range():
    """Independent derivation: (rho . rhodot)/|rho| against d|rho|/dt by finite difference.

    Two separate routes to the same quantity, so a transposed index or a missing component
    in the projection shows up immediately.
    """
    times = np.arange(0.0, 601.0, 25.0)
    _, states = offset_circle(times)
    step_s = 1.0e-3
    _, ahead = offset_circle(times + step_s)
    _, behind = offset_circle(times - step_s)
    difference = (separation_m(ahead) - separation_m(behind)) / (2.0 * step_s)

    # Measured max |analytic - central difference| = 2.936e-10 m/s against range rates of
    # order 1.57 m/s, i.e. ~1.9e-10 relative, dominated by the O(h**2) truncation error of
    # the difference itself. A bound of 1e-8 leaves ~34x.
    np.testing.assert_allclose(range_rate_m_s(states), difference, atol=1.0e-8)


@pytest.mark.unit
def test_range_rate_at_coincident_positions_is_defined_as_zero_not_infinite():
    states = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
    assert range_rate_m_s(states)[0] == 0.0


# --------------------------------------------------------------------------------------
# Aggregate report
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_report_totals_and_first_violation_time_across_constraints():
    """Every number below is derived from the pass geometry, then cross-checked.

    Keep-out R = 100 m breaches over t in [156.70, 243.30] s -> 9 samples, first at 160 s.
    Closing velocity is gated at 200 m, which is |2t - 400| <= sqrt(200**2 - 50**2), i.e.
    t in [103.18, 296.83] s. Inside that window the closing velocity 2(400 - 2t)/|rho|
    exceeds 1 m/s while 3(400 - 2t)**2 > 2500, i.e. t < 185.57 s -> 8 samples, first at
    110 s. The ellipsoid at 500 m a side is never breached (max q = 0.65).
    """
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(100.0),
        ellipsoid=ApproachEllipsoid((500.0, 500.0, 500.0)),
        closing_velocity=ClosingVelocityLimit(1.0, 200.0),
    )

    assert report.keep_out is not None and report.keep_out.n_violating_samples == 9
    assert report.ellipsoid is not None and report.ellipsoid.n_violating_samples == 0
    assert report.closing_velocity is not None
    assert report.closing_velocity.n_violating_samples == 8
    assert report.closing_velocity.first_violation_time_s == 110.0

    assert report.total_violating_samples == 9 + 0 + 8
    assert report.first_violation_time_s == min(110.0, 160.0)
    assert not report.all_satisfied
    assert len(report.results) == 3
    assert report.corridor is None


@pytest.mark.unit
def test_report_of_a_clean_trajectory_has_no_violations_and_no_first_time():
    """Complement: the same machinery must say "safe" when the trajectory is safe."""
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(10.0),
        ellipsoid=ApproachEllipsoid((500.0, 500.0, 500.0)),
        closing_velocity=ClosingVelocityLimit(10.0, math.inf),
    )
    assert report.all_satisfied
    assert report.total_violating_samples == 0
    assert report.first_violation_time_s is None


@pytest.mark.unit
def test_report_only_evaluates_the_constraints_it_was_given():
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    report = evaluate_constraints(times, states, keep_out=KeepOutSphere(10.0))
    assert report.results == (report.keep_out,)
    assert report.ellipsoid is None
    assert report.corridor is None
    assert report.closing_velocity is None


@pytest.mark.unit
def test_report_matches_the_individual_evaluations():
    """The aggregate must not be a second, divergent implementation of the same checks."""
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    keep_out = KeepOutSphere(100.0)
    corridor = ApproachCorridor(math.radians(30.0), 300.0, VBAR_AXIS)
    report = evaluate_constraints(times, states, keep_out=keep_out, corridor=corridor)

    assert report.keep_out == evaluate_keep_out_sphere(times, states, keep_out)
    assert report.corridor == evaluate_approach_corridor(times, states, corridor)


@pytest.mark.unit
def test_results_are_frozen():
    """A report handed to a downstream metrics table must not be editable in place."""
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    report = evaluate_constraints(times, states, keep_out=KeepOutSphere(100.0))
    with pytest.raises(AttributeError):
        report.total_violating_samples = 0  # type: ignore[misc]
    assert report.keep_out is not None
    with pytest.raises(AttributeError):
        report.keep_out.satisfied = True  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Constraint-definition validation -- every raise path
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("bad_radius", [0.0, -1.0, -1e-9, float("nan"), float("inf")])
def test_non_positive_or_non_finite_keep_out_radius_raises(bad_radius):
    with pytest.raises(ConstraintDefinitionError, match="radius_m"):
        KeepOutSphere(bad_radius)


@pytest.mark.unit
def test_keep_out_radius_error_reports_the_offending_value():
    with pytest.raises(ConstraintDefinitionError, match=r"-42\.0"):
        KeepOutSphere(-42.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_axes",
    [(0.0, 1.0, 1.0), (1.0, -2.0, 1.0), (1.0, 1.0, float("nan")), (1.0, float("inf"), 1.0)],
)
def test_non_positive_or_non_finite_semi_axis_raises(bad_axes):
    with pytest.raises(ConstraintDefinitionError, match="semi-axis"):
        ApproachEllipsoid(bad_axes)


@pytest.mark.unit
@pytest.mark.parametrize("bad_axes", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), ()])
def test_wrong_number_of_semi_axes_raises(bad_axes):
    with pytest.raises(ConstraintDefinitionError, match="three entries"):
        ApproachEllipsoid(bad_axes)


@pytest.mark.unit
def test_non_sequence_semi_axes_raises():
    with pytest.raises(ConstraintDefinitionError, match="three finite positive lengths"):
        ApproachEllipsoid(100.0)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_angle", [0.0, -0.1, math.pi, math.pi + 0.1, float("nan"), float("inf")]
)
def test_half_angle_outside_the_open_interval_raises(bad_angle):
    with pytest.raises(ConstraintDefinitionError, match="half_angle_rad"):
        ApproachCorridor(bad_angle, math.inf)


@pytest.mark.unit
@pytest.mark.parametrize("bad_range", [0.0, -1.0, float("nan")])
def test_non_positive_activation_range_raises(bad_range):
    with pytest.raises(ConstraintDefinitionError, match="activation_range_m"):
        ApproachCorridor(0.1, bad_range)


@pytest.mark.unit
@pytest.mark.parametrize("bad_range", [0.0, -1.0, float("nan")])
def test_non_positive_activation_range_on_the_closing_limit_raises(bad_range):
    with pytest.raises(ConstraintDefinitionError, match="activation_range_m"):
        ClosingVelocityLimit(1.0, bad_range)


@pytest.mark.unit
def test_infinite_activation_range_is_accepted_as_always_active():
    """The documented escape hatch; asserted so a tightened check cannot break it silently."""
    assert ApproachCorridor(0.1, math.inf).activation_range_m == math.inf
    assert ClosingVelocityLimit(1.0, math.inf).activation_range_m == math.inf


@pytest.mark.unit
@pytest.mark.parametrize("bad_axis", [(0.0, 0.0, 0.0), (0.0, 1e-13, 0.0)])
def test_zero_length_cone_axis_raises(bad_axis):
    with pytest.raises(ConstraintDefinitionError, match="non-zero length"):
        ApproachCorridor(0.1, math.inf, bad_axis)


@pytest.mark.unit
@pytest.mark.parametrize("bad_axis", [(0.0, -1.0), (0.0, -1.0, 0.0, 0.0)])
def test_wrong_number_of_cone_axis_components_raises(bad_axis):
    with pytest.raises(ConstraintDefinitionError, match="three entries"):
        ApproachCorridor(0.1, math.inf, bad_axis)


@pytest.mark.unit
@pytest.mark.parametrize("bad_axis", [(float("nan"), -1.0, 0.0), (0.0, float("inf"), 0.0)])
def test_non_finite_cone_axis_raises(bad_axis):
    with pytest.raises(ConstraintDefinitionError, match="axis_hill must be finite"):
        ApproachCorridor(0.1, math.inf, bad_axis)


@pytest.mark.unit
def test_non_sequence_cone_axis_raises():
    with pytest.raises(ConstraintDefinitionError, match="three finite components"):
        ApproachCorridor(0.1, math.inf, 1.0)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("bad_speed", [-1.0, -1e-12, float("nan"), float("inf")])
def test_negative_or_non_finite_closing_speed_limit_raises(bad_speed):
    with pytest.raises(ConstraintDefinitionError, match="max_closing_speed_m_s"):
        ClosingVelocityLimit(bad_speed, math.inf)


@pytest.mark.unit
def test_zero_closing_speed_limit_is_accepted():
    """Zero is a meaningful limit (no net approach permitted), not a definition error."""
    assert ClosingVelocityLimit(0.0, math.inf).max_closing_speed_m_s == 0.0


@pytest.mark.unit
def test_a_report_with_no_constraints_raises_rather_than_reporting_safe():
    times, states = straight_line_pass(np.arange(0.0, 401.0, 10.0))
    with pytest.raises(ConstraintDefinitionError, match="at least one constraint"):
        evaluate_constraints(times, states)


# --------------------------------------------------------------------------------------
# Trajectory validation -- every raise path
# --------------------------------------------------------------------------------------

KEEP_OUT = KeepOutSphere(50.0)


@pytest.mark.unit
def test_zero_length_trajectory_raises():
    with pytest.raises(TrajectorySamplingError, match="at least one sample"):
        evaluate_keep_out_sphere(np.array([]), np.zeros((0, 6)), KEEP_OUT)


@pytest.mark.unit
def test_empty_times_with_populated_states_raises():
    with pytest.raises(TrajectorySamplingError, match="at least one sample"):
        evaluate_keep_out_sphere(np.array([]), np.zeros((3, 6)), KEEP_OUT)


@pytest.mark.unit
def test_mismatched_times_and_states_lengths_raise_with_both_lengths():
    with pytest.raises(TrajectorySamplingError, match=r"got 3 times and 4 states"):
        evaluate_keep_out_sphere(np.arange(3.0), np.zeros((4, 6)), KEEP_OUT)


@pytest.mark.unit
@pytest.mark.parametrize("bad_shape", [(3, 5), (3, 7), (3, 3)])
def test_states_that_are_not_six_wide_raise(bad_shape):
    with pytest.raises(TrajectorySamplingError, match=r"shape \(N, 6\)"):
        evaluate_keep_out_sphere(np.arange(float(bad_shape[0])), np.zeros(bad_shape), KEEP_OUT)


@pytest.mark.unit
def test_one_dimensional_states_raise():
    with pytest.raises(TrajectorySamplingError, match=r"shape \(N, 6\)"):
        evaluate_keep_out_sphere(np.array([0.0]), np.zeros(6), KEEP_OUT)


@pytest.mark.unit
def test_two_dimensional_times_raise():
    with pytest.raises(TrajectorySamplingError, match="one-dimensional"):
        evaluate_keep_out_sphere(np.zeros((2, 2)), np.zeros((4, 6)), KEEP_OUT)


@pytest.mark.unit
@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_times_raise_with_the_offending_index(bad_time):
    times = np.array([0.0, 10.0, 20.0])
    times[1] = bad_time
    with pytest.raises(TrajectorySamplingError, match=r"times_s must be finite.*index 1"):
        evaluate_keep_out_sphere(times, np.ones((3, 6)), KEEP_OUT)


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_states_raise_with_the_offending_row(bad_value):
    states = np.ones((3, 6))
    states[2, 4] = bad_value
    with pytest.raises(TrajectorySamplingError, match=r"states_hill must be finite.*row 2"):
        evaluate_keep_out_sphere(np.arange(3.0), states, KEEP_OUT)


@pytest.mark.unit
def test_decreasing_times_raise_with_both_offending_values():
    with pytest.raises(TrajectorySamplingError, match="strictly increasing") as excinfo:
        evaluate_keep_out_sphere(np.array([0.0, 10.0, 5.0]), np.ones((3, 6)), KEEP_OUT)
    message = str(excinfo.value)
    assert "times_s[1]=10.0" in message
    assert "times_s[2]=5.0" in message


@pytest.mark.unit
def test_repeated_times_raise():
    """Equal timestamps are rejected too: a zero-length step makes the refinement singular."""
    with pytest.raises(TrajectorySamplingError, match="strictly increasing"):
        evaluate_keep_out_sphere(np.array([0.0, 10.0, 10.0]), np.ones((3, 6)), KEEP_OUT)


@pytest.mark.unit
@pytest.mark.parametrize(
    "evaluate",
    [
        lambda t, s: evaluate_keep_out_sphere(t, s, KEEP_OUT),
        lambda t, s: evaluate_approach_ellipsoid(t, s, ApproachEllipsoid((1.0, 1.0, 1.0))),
        lambda t, s: evaluate_approach_corridor(t, s, ApproachCorridor(0.1, math.inf)),
        lambda t, s: evaluate_closing_velocity(t, s, ClosingVelocityLimit(1.0, math.inf)),
        lambda t, s: evaluate_constraints(t, s, keep_out=KEEP_OUT),
    ],
)
def test_every_entry_point_validates_the_trajectory(evaluate):
    """A validation branch that only one entry point exercises is not a validation branch."""
    with pytest.raises(TrajectorySamplingError, match="strictly increasing"):
        evaluate(np.array([0.0, 10.0, 5.0]), np.ones((3, 6)))


@pytest.mark.unit
def test_helpers_validate_their_states():
    with pytest.raises(TrajectorySamplingError, match=r"shape \(N, 6\)"):
        separation_m(np.zeros((3, 4)))
    with pytest.raises(TrajectorySamplingError, match="must be finite"):
        range_rate_m_s(np.full((2, 6), np.nan))


@pytest.mark.unit
def test_both_error_types_are_rpo_core_errors_and_value_errors():
    """Callers catching RpoCoreError or ValueError must not miss these."""
    assert issubclass(ConstraintDefinitionError, RpoCoreError)
    assert issubclass(ConstraintDefinitionError, ValueError)
    assert issubclass(TrajectorySamplingError, RpoCoreError)
    assert issubclass(TrajectorySamplingError, ValueError)
