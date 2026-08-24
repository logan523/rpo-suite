"""CW measured against nonlinear two-body relative motion.

This is the first external oracle in the suite. Every assertion in ``test_cw.py`` checks
CW against closed-form properties of CW; these check it against the dynamics it
approximates. The target orbit is exactly circular throughout, which isolates
*linearisation* error from *eccentricity* error -- conflating the two produces an error
budget that cannot be attributed to anything.
"""

import itertools
import math
import warnings

import numpy as np
import pytest
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from rpo_core.relative.nonlinear import (
    CW_ERROR_SAFETY_FACTOR,
    check_cw_validity,
    conservative_cw_error_bound_m,
    cw_position_error_m,
    estimated_cw_error_m,
    propagate_relative_nonlinear,
)

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
V_CIRCULAR = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
N = mean_motion_rad_s(A_ISS_M)
PERIOD_S = orbital_period_s(A_ISS_M)

# Exactly circular target, inclined 51.6 deg.
_INC = math.radians(51.6)
R_TARGET = np.array([A_ISS_M, 0.0, 0.0])
V_TARGET = V_CIRCULAR * np.array([0.0, math.cos(_INC), math.sin(_INC)])


def _drift_free(x0: float, y0: float) -> np.ndarray:
    """Return a bounded (non-drifting) relative state at the given offsets."""
    return np.array([x0, y0, 0.0, 0.0, -2.0 * N * x0, 0.0])


@pytest.mark.integration
def test_cw_matches_nonlinear_motion_in_the_small_separation_limit():
    """At 1 m separation CW is essentially exact -- the linearisation has nothing to drop."""
    times = np.linspace(0.0, 0.5 * PERIOD_S, 51)
    error = cw_position_error_m(R_TARGET, V_TARGET, _drift_free(1.0, -1.0), times, N)
    assert error.max() < 1e-4, f"CW deviated by {error.max():.3e} m at 1 m separation"


@pytest.mark.integration
def test_cw_error_grows_with_separation():
    """The defining property of a linearisation. Error must scale up with rho, not stay flat.

    A test that only checked small separations would also pass if the nonlinear reference
    were accidentally computing CW.
    """
    times = np.linspace(0.0, 0.5 * PERIOD_S, 51)
    errors = [
        cw_position_error_m(R_TARGET, V_TARGET, _drift_free(0.0, -sep), times, N).max()
        for sep in (100.0, 1_000.0, 10_000.0, 100_000.0)
    ]
    assert all(a < b for a, b in itertools.pairwise(errors)), (
        f"error did not grow monotonically with separation: {errors}"
    )
    # Leading neglected term is O(rho/r), so a 10x separation increase should cost
    # roughly 100x in absolute error. Bracket it loosely; the point is the scaling law
    # holds, not that it hits an exact power.
    ratio = errors[-1] / errors[-2]
    assert 20.0 < ratio < 500.0, f"error scaling {ratio:.1f}x per decade is not O(rho^2)"


@pytest.mark.integration
def test_cw_error_grows_with_elapsed_time():
    """Linearisation error accumulates; it does not stay bounded."""
    times = np.linspace(0.0, 2.0 * PERIOD_S, 201)
    error = cw_position_error_m(R_TARGET, V_TARGET, _drift_free(0.0, -1_000.0), times, N)
    first_orbit = error[: len(error) // 2].max()
    second_orbit = error[len(error) // 2 :].max()
    assert second_orbit > first_orbit


@pytest.mark.integration
def test_nonlinear_reference_conserves_the_target_orbit_it_is_built_on():
    """Guard the oracle itself: a drifting reference would silently corrupt every bound."""
    times = np.linspace(0.0, PERIOD_S, 101)
    relative = propagate_relative_nonlinear(R_TARGET, V_TARGET, np.zeros(6), times, MU_EARTH_M3_S2)
    # A chaser initialised exactly on the target must stay exactly on it.
    assert np.abs(relative[:, :3]).max() < 1e-6
    assert np.abs(relative[:, 3:]).max() < 1e-9


@pytest.mark.integration
def test_drift_free_condition_still_closes_under_nonlinear_dynamics_at_short_range():
    """The CW closed-orbit condition should very nearly close in the real dynamics too.

    It does not close exactly -- that residual *is* the linearisation error, and this test
    pins how large it is at operational range.
    """
    times = np.array([0.0, PERIOD_S])
    relative = propagate_relative_nonlinear(
        R_TARGET, V_TARGET, _drift_free(100.0, -500.0), times, MU_EARTH_M3_S2
    )
    closure_error_m = float(np.linalg.norm(relative[-1, :3] - relative[0, :3]))
    assert closure_error_m < 50.0, f"closure error {closure_error_m:.2f} m is implausibly large"
    assert closure_error_m > 1e-3, "closure was exact; the reference may not be nonlinear"


# --------------------------------------------------------------------------------------
# The measured error law
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("altitude_m", [400.0e3, 800.0e3, 1500.0e3])
@pytest.mark.parametrize("separation_m", [1.0e3, 1.0e4])
def test_one_orbit_error_follows_the_six_pi_law(altitude_m, separation_m):
    """err(1 orbit) == 6*pi * rho**2 / r, to better than 0.1 %.

    This is the measured result that turns "CW degrades with separation" from a hand-wave
    into a budget. Parametrised across altitude and separation because a law that only
    held at one operating point would be a fit, not a law.
    """
    radius_m = R_EARTH_EQUATORIAL_M + altitude_m
    v_circular = math.sqrt(MU_EARTH_M3_S2 / radius_m)
    n = mean_motion_rad_s(radius_m)
    period = orbital_period_s(radius_m)

    r_target = np.array([radius_m, 0.0, 0.0])
    v_target = v_circular * np.array([0.0, math.cos(_INC), math.sin(_INC)])
    state0 = np.array([0.0, -separation_m, 0.0, 0.0, 0.0, 0.0])

    measured = cw_position_error_m(
        r_target, v_target, state0, np.linspace(0.0, period, 61), n
    ).max()
    predicted = 6.0 * math.pi * separation_m**2 / radius_m
    assert measured == pytest.approx(predicted, rel=1e-3)


@pytest.mark.unit
def test_validity_estimator_agrees_with_the_measured_law():
    assert estimated_cw_error_m(1_000.0, A_ISS_M) == pytest.approx(2.773, rel=1e-3)
    assert estimated_cw_error_m(10_000.0, A_ISS_M) == pytest.approx(277.3, rel=1e-3)


@pytest.mark.unit
def test_error_estimate_is_quadratic_in_separation():
    """Ten times the separation costs a hundred times the error."""
    small = estimated_cw_error_m(1_000.0, A_ISS_M)
    large = estimated_cw_error_m(10_000.0, A_ISS_M)
    assert large / small == pytest.approx(100.0, rel=1e-12)


@pytest.mark.unit
def test_validity_check_stays_silent_inside_the_envelope():
    """The MVP scenario -- 1 km separation, half an orbit -- is valid.

    Budget is 5 m, i.e. 2.5 % of the 200 m keep-out sphere. An earlier 1 % (2 m) budget
    was arbitrary and sat awkwardly: the conservative bound for this scenario is 2.08 m,
    so the flagship baseline warned about itself while the measured error was only
    1.455 m. The guard exists to catch CW being used at 10 km where the error is 277 m,
    not to police the difference between 2.00 and 2.08 m against a 200 m sphere.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        check_cw_validity(1_000.0, N, n_orbits=0.5, tolerance_m=5.0)


@pytest.mark.unit
def test_baseline_scenario_bound_sits_well_inside_the_five_metre_budget():
    """Pin the actual margin so a future change to the law or factor is visible."""
    bound = conservative_cw_error_bound_m(1_000.0, A_ISS_M, 0.5)
    assert bound == pytest.approx(2.08, abs=0.02)
    assert bound < 5.0


@pytest.mark.unit
def test_validity_check_warns_outside_the_envelope():
    """Far-range work is exactly where CW must stop being trusted."""
    with pytest.warns(UserWarning, match="outside its useful envelope"):
        check_cw_validity(50_000.0, N, n_orbits=1.0, tolerance_m=2.0)


@pytest.mark.unit
def test_validity_check_rejects_a_non_positive_tolerance():
    with pytest.raises(ValueError, match="tolerance_m"):
        check_cw_validity(1_000.0, N, tolerance_m=0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"separation_m": -1.0, "orbit_radius_m": A_ISS_M}, "separation_m"),
        ({"separation_m": 1.0, "orbit_radius_m": 0.0}, "orbit_radius_m"),
        ({"separation_m": 1.0, "orbit_radius_m": A_ISS_M, "n_orbits": -1.0}, "n_orbits"),
    ],
)
def test_error_estimator_rejects_invalid_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        estimated_cw_error_m(**kwargs)


# --------------------------------------------------------------------------------------
# The conservative bound must actually bound
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("altitude_m", [400.0e3, 800.0e3, 1500.0e3])
@pytest.mark.parametrize("separation_m", [100.0, 1_000.0, 5_000.0])
@pytest.mark.parametrize("n_orbits", [0.1, 0.4, 0.5, 0.7, 1.0, 2.0])
def test_conservative_bound_is_never_exceeded_by_measurement(altitude_m, separation_m, n_orbits):
    """The whole point of a bound. If measurement beats it anywhere, it is not a bound.

    The plain linear estimate FAILS this test between ~0.4 and 1.0 orbits — measured error
    exceeds it by up to 1.23x. That is why `check_cw_validity` guards on this function and
    not on `estimated_cw_error_m`.
    """
    radius_m = R_EARTH_EQUATORIAL_M + altitude_m
    v_circular = math.sqrt(MU_EARTH_M3_S2 / radius_m)
    n = mean_motion_rad_s(radius_m)
    period = orbital_period_s(radius_m)

    r_target = np.array([radius_m, 0.0, 0.0])
    v_target = v_circular * np.array([0.0, math.cos(_INC), math.sin(_INC)])
    state0 = np.array([0.0, -separation_m, 0.0, 0.0, 0.0, 0.0])

    measured = cw_position_error_m(
        r_target, v_target, state0, np.linspace(0.0, n_orbits * period, 121), n
    ).max()
    bound = conservative_cw_error_bound_m(separation_m, radius_m, n_orbits)
    assert measured <= bound, (
        f"bound {bound:.4g} m was exceeded by measurement {measured:.4g} m "
        f"at {separation_m} m / {n_orbits} orbits / {altitude_m / 1e3:.0f} km"
    )


@pytest.mark.integration
def test_the_plain_linear_estimate_is_genuinely_optimistic_near_three_quarter_orbit():
    """Complement test: prove the safety factor is load-bearing, not decoration.

    If this ever stops failing to bound, the factor could be dropped — but as measured,
    the linear estimate under-predicts here, so removing it would silently weaken the guard.
    """
    radius_m = R_EARTH_EQUATORIAL_M + 800.0e3
    v_circular = math.sqrt(MU_EARTH_M3_S2 / radius_m)
    n = mean_motion_rad_s(radius_m)
    period = orbital_period_s(radius_m)
    r_target = np.array([radius_m, 0.0, 0.0])
    v_target = v_circular * np.array([0.0, math.cos(_INC), math.sin(_INC)])

    measured = cw_position_error_m(
        r_target,
        v_target,
        np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]),
        np.linspace(0.0, 0.7 * period, 121),
        n,
    ).max()
    assert measured > estimated_cw_error_m(100.0, radius_m, 0.7)
    assert measured <= conservative_cw_error_bound_m(100.0, radius_m, 0.7)


@pytest.mark.unit
def test_safety_factor_clears_the_measured_worst_case():
    """Documented worst-case under-prediction is 1.2253x; the factor must exceed it."""
    assert CW_ERROR_SAFETY_FACTOR > 1.2253
