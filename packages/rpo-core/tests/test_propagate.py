"""Two-body propagation: conservation laws, known solutions, and failure behaviour."""

import math

import numpy as np
import pytest
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    orbital_period_s,
)
from rpo_core.exceptions import PropagationError
from rpo_core.propagate import (
    propagate_two_body,
    specific_angular_momentum,
    specific_angular_momentum_batch,
    specific_energy_batch_j_kg,
    specific_energy_j_kg,
    two_body_derivative,
)

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
V_CIRCULAR = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
PERIOD_S = orbital_period_s(A_ISS_M)

CIRCULAR_STATE = np.array([A_ISS_M, 0.0, 0.0, 0.0, V_CIRCULAR, 0.0])
# e = 0.1 ellipse at the same periapsis radius, for the non-circular checks.
ELLIPTIC_STATE = np.array([A_ISS_M, 0.0, 0.0, 0.0, V_CIRCULAR * math.sqrt(1.1), 0.0])


@pytest.mark.unit
def test_specific_energy_is_conserved_over_ten_orbits():
    times = np.linspace(0.0, 10.0 * PERIOD_S, 201)
    states = propagate_two_body(ELLIPTIC_STATE, times)
    energies = np.array([specific_energy_j_kg(s) for s in states])
    drift = np.abs(energies - energies[0]) / abs(energies[0])
    assert drift.max() < 1e-10, f"energy drifted by {drift.max():.3e} relative"


@pytest.mark.unit
def test_specific_angular_momentum_is_conserved_over_ten_orbits():
    times = np.linspace(0.0, 10.0 * PERIOD_S, 201)
    states = propagate_two_body(ELLIPTIC_STATE, times)
    h = np.array([specific_angular_momentum(s) for s in states])
    drift = np.linalg.norm(h - h[0], axis=1) / np.linalg.norm(h[0])
    assert drift.max() < 1e-10, f"angular momentum drifted by {drift.max():.3e} relative"


@pytest.mark.unit
def test_circular_orbit_returns_to_its_initial_state_after_one_period():
    """Known solution: a circular orbit is periodic with period 2*pi*sqrt(a**3/mu)."""
    final = propagate_two_body(CIRCULAR_STATE, np.array([0.0, PERIOD_S]))[-1]
    np.testing.assert_allclose(final[:3], CIRCULAR_STATE[:3], atol=1e-3)  # mm..mm scale
    np.testing.assert_allclose(final[3:], CIRCULAR_STATE[3:], atol=1e-6)


@pytest.mark.unit
def test_circular_orbit_keeps_constant_radius_and_speed():
    times = np.linspace(0.0, 2.0 * PERIOD_S, 201)
    states = propagate_two_body(CIRCULAR_STATE, times)
    radii = np.linalg.norm(states[:, :3], axis=1)
    speeds = np.linalg.norm(states[:, 3:], axis=1)
    # Bound set from the integrator's own noise floor: at rtol = 1e-12 accumulated over
    # two orbits the measured spread is ~2e-12 relative, so 1e-10 is two decades of
    # headroom while still catching any real secular drift in the radius.
    assert np.ptp(radii) / A_ISS_M < 1e-10
    assert np.ptp(speeds) / V_CIRCULAR < 1e-10


@pytest.mark.unit
def test_orbit_stays_in_its_initial_plane():
    """Two-body motion is planar: the out-of-plane component must stay at zero."""
    times = np.linspace(0.0, 3.0 * PERIOD_S, 151)
    states = propagate_two_body(ELLIPTIC_STATE, times)
    assert np.abs(states[:, 2]).max() < 1e-6


@pytest.mark.unit
def test_elliptic_apoapsis_matches_the_vis_viva_prediction():
    """Independent closed-form check: r_apo = a(1+e), v_apo from vis-viva."""
    energy = specific_energy_j_kg(ELLIPTIC_STATE)
    a = -MU_EARTH_M3_S2 / (2.0 * energy)
    h = float(np.linalg.norm(specific_angular_momentum(ELLIPTIC_STATE)))
    e = math.sqrt(1.0 + 2.0 * energy * h**2 / MU_EARTH_M3_S2**2)

    period = 2.0 * math.pi * math.sqrt(a**3 / MU_EARTH_M3_S2)
    apoapsis = propagate_two_body(ELLIPTIC_STATE, np.array([0.0, 0.5 * period]))[-1]

    assert float(np.linalg.norm(apoapsis[:3])) == pytest.approx(a * (1.0 + e), rel=1e-9)
    expected_v = math.sqrt(MU_EARTH_M3_S2 * (2.0 / (a * (1.0 + e)) - 1.0 / a))
    assert float(np.linalg.norm(apoapsis[3:])) == pytest.approx(expected_v, rel=1e-9)


@pytest.mark.unit
def test_backward_then_forward_propagation_recovers_the_initial_state():
    forward = propagate_two_body(ELLIPTIC_STATE, np.array([0.0, 0.4 * PERIOD_S]))[-1]
    reverse = forward.copy()
    reverse[3:] *= -1.0  # time reversal: flip velocity, propagate forward, flip back
    back = propagate_two_body(reverse, np.array([0.0, 0.4 * PERIOD_S]))[-1]
    back[3:] *= -1.0
    np.testing.assert_allclose(back[:3], ELLIPTIC_STATE[:3], atol=1e-3)
    np.testing.assert_allclose(back[3:], ELLIPTIC_STATE[3:], atol=1e-6)


@pytest.mark.unit
def test_solution_converges_monotonically_as_tolerance_tightens():
    """A number that moves when rtol moves is an integrator setting, not a physical result.

    Asserting convergence *behaviour* rather than one hand-picked threshold: the deviation
    from a reference run at rtol = 1e-13 must shrink at every step as the tolerance
    tightens, and the package default must already be converged to well under a
    millimetre over two orbits. A single arbitrary bound would pass for the wrong reason
    if the integrator silently stopped responding to its tolerance argument.
    """
    times = np.array([0.0, 2.0 * PERIOD_S])
    reference = propagate_two_body(ELLIPTIC_STATE, times, rtol=1e-13, atol=1e-13)[-1]

    deviations = [
        float(
            np.linalg.norm(
                propagate_two_body(ELLIPTIC_STATE, times, rtol=tol, atol=tol)[-1][:3]
                - reference[:3]
            )
        )
        for tol in (1e-10, 1e-11, 1e-12)
    ]
    assert deviations[0] > deviations[1] > deviations[2], (
        f"deviation did not shrink monotonically with tolerance: {deviations}"
    )
    # At the package default (1e-12) the trajectory is converged to ~0.04 mm over two
    # orbits of a 6.8e6 m orbit, i.e. ~6e-12 relative.
    assert deviations[-1] < 1e-4


@pytest.mark.unit
def test_singularity_at_the_origin_raises_rather_than_returning_inf():
    with pytest.raises(PropagationError, match="singularity"):
        two_body_derivative(0.0, np.zeros(6), MU_EARTH_M3_S2)


@pytest.mark.unit
def test_single_output_time_returns_the_initial_state_unchanged():
    result = propagate_two_body(CIRCULAR_STATE, np.array([0.0]))
    assert result.shape == (1, 6)
    np.testing.assert_array_equal(result[0], CIRCULAR_STATE)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("times", "match"),
    [
        (np.array([10.0, 20.0]), "must start at 0.0"),
        (np.array([0.0, 20.0, 10.0]), "non-decreasing"),
        (np.array([]), "non-empty"),
        (np.array([0.0, np.nan]), "finite"),
    ],
)
def test_malformed_time_schedule_raises(times, match):
    with pytest.raises(ValueError, match=match):
        propagate_two_body(CIRCULAR_STATE, times)


@pytest.mark.unit
def test_malformed_initial_state_raises():
    with pytest.raises(ValueError, match="shape"):
        propagate_two_body(np.zeros(3), np.array([0.0, 100.0]))


# --------------------------------------------------------------------------------------
# Batch handling — regression guard for a real bug
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_stacked_states_are_rejected_rather_than_silently_collapsed():
    """An (N, 6) array must raise, not return one meaningless number.

    Regression guard. An earlier version indexed with an ellipsis, advertising batch
    support that `np.linalg.norm` did not honour: three identical LEO states returned
    +54 MJ/kg instead of -29.3 MJ/kg, flipping the sign so a bound orbit read as
    hyperbolic. Silent and catastrophic exactly where an energy check is supposed to be
    the thing that catches problems.
    """
    batch = np.vstack([CIRCULAR_STATE, CIRCULAR_STATE, CIRCULAR_STATE])
    with pytest.raises(ValueError, match="batch variant"):
        specific_energy_j_kg(batch)
    with pytest.raises(ValueError, match="batch variant"):
        specific_angular_momentum(batch)


@pytest.mark.unit
def test_batch_variants_match_the_scalar_versions_elementwise():
    states = np.vstack([CIRCULAR_STATE, ELLIPTIC_STATE])
    np.testing.assert_allclose(
        specific_energy_batch_j_kg(states),
        [specific_energy_j_kg(s) for s in states],
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        specific_angular_momentum_batch(states),
        [specific_angular_momentum(s) for s in states],
        rtol=1e-15,
    )


@pytest.mark.unit
def test_batch_energy_is_negative_for_bound_orbits():
    """The sign is the whole point: bound orbits have negative specific energy."""
    states = np.vstack([CIRCULAR_STATE, ELLIPTIC_STATE])
    assert np.all(specific_energy_batch_j_kg(states) < 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros((2, 5)), np.zeros(3)])
def test_batch_variants_reject_malformed_shapes(bad):
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        specific_energy_batch_j_kg(bad)
