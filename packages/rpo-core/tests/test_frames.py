"""ECI <-> Hill (LVLH) frame transformations."""

import math

import numpy as np
import pytest
from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M, mean_motion_rad_s
from rpo_core.exceptions import DegenerateGeometryError
from rpo_core.frames import (
    hill_basis,
    relative_state_eci_to_hill,
    relative_state_hill_to_eci,
)

A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3
V_CIRCULAR_M_S = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
N_RAD_S = mean_motion_rad_s(A_ISS_M)

# Equatorial circular reference state, chosen so the expected triad is written by hand.
R_EQUATORIAL = np.array([A_ISS_M, 0.0, 0.0])
V_EQUATORIAL = np.array([0.0, V_CIRCULAR_M_S, 0.0])

# A 51.6 deg inclined circular state, for the convention checks that must hold generally.
_INC = math.radians(51.6)
R_INCLINED = np.array([A_ISS_M, 0.0, 0.0])
V_INCLINED = V_CIRCULAR_M_S * np.array([0.0, math.cos(_INC), math.sin(_INC)])


@pytest.mark.unit
def test_rotation_is_orthonormal_and_right_handed():
    rotation, _ = hill_basis(R_INCLINED, V_INCLINED)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    # det = +1 means a proper rotation (no reflection). A det of -1 would mean the triad
    # is left-handed, which flips the sign of every cross-track result downstream.
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.unit
def test_equatorial_circular_triad_matches_hand_computed_axes():
    rotation, omega = hill_basis(R_EQUATORIAL, V_EQUATORIAL)
    np.testing.assert_allclose(rotation[0], [1.0, 0.0, 0.0], atol=1e-14)  # x = radial out
    np.testing.assert_allclose(rotation[1], [0.0, 1.0, 0.0], atol=1e-14)  # y = along-track
    np.testing.assert_allclose(rotation[2], [0.0, 0.0, 1.0], atol=1e-14)  # z = +orbit normal
    np.testing.assert_allclose(omega, [0.0, 0.0, N_RAD_S], rtol=1e-14)


@pytest.mark.unit
def test_cross_track_axis_is_the_positive_orbit_normal():
    """z_hat == h_hat, not -h_hat.

    This is the convention check. With x radial-outward and y along-velocity, the
    right-handed completion z = x cross y is r_hat cross v_hat, which *is* the specific
    angular momentum direction. Getting this backwards silently flips every cross-track
    sign in the suite.
    """
    rotation, _ = hill_basis(R_INCLINED, V_INCLINED)
    h = np.cross(R_INCLINED, V_INCLINED)
    np.testing.assert_allclose(rotation[2], h / np.linalg.norm(h), atol=1e-14)


@pytest.mark.unit
def test_radial_axis_points_away_from_earth():
    rotation, _ = hill_basis(R_INCLINED, V_INCLINED)
    assert float(rotation[0] @ R_INCLINED) > 0.0


@pytest.mark.unit
def test_along_track_axis_aligns_with_velocity_for_circular_orbit():
    rotation, _ = hill_basis(R_INCLINED, V_INCLINED)
    np.testing.assert_allclose(rotation[1], V_INCLINED / np.linalg.norm(V_INCLINED), atol=1e-14)


@pytest.mark.unit
def test_frame_angular_velocity_magnitude_equals_mean_motion():
    _, omega = hill_basis(R_INCLINED, V_INCLINED)
    assert float(np.linalg.norm(omega)) == pytest.approx(N_RAD_S, rel=1e-14)


@pytest.mark.unit
def test_chaser_colocated_with_target_has_zero_relative_state():
    state = relative_state_eci_to_hill(R_INCLINED, V_INCLINED, R_INCLINED, V_INCLINED)
    np.testing.assert_allclose(state, np.zeros(6), atol=1e-12)


@pytest.mark.unit
def test_chaser_trailing_on_v_bar_has_negative_along_track_position():
    """A chaser 1 km behind the target must land at y = -1000 m, not +1000 m."""
    separation_m = 1000.0
    v_hat = V_EQUATORIAL / np.linalg.norm(V_EQUATORIAL)
    r_chaser = R_EQUATORIAL - separation_m * v_hat
    state = relative_state_eci_to_hill(R_EQUATORIAL, V_EQUATORIAL, r_chaser, V_EQUATORIAL)
    assert state[1] == pytest.approx(-separation_m, abs=1e-6)


@pytest.mark.unit
def test_transport_theorem_term_is_applied():
    """A chaser with identical inertial velocity is *moving* in the rotating Hill frame.

    Omitting the ``omega x dr`` term would return zero relative velocity here. The
    expected magnitude is n * separation -- about 1.1 m/s per km in this orbit, the same
    order as the manoeuvres being designed, so the error would not be subtle.
    """
    separation_m = 1000.0
    r_chaser = R_EQUATORIAL + np.array([0.0, -separation_m, 0.0])
    state = relative_state_eci_to_hill(R_EQUATORIAL, V_EQUATORIAL, r_chaser, V_EQUATORIAL)
    assert float(np.linalg.norm(state[3:])) == pytest.approx(N_RAD_S * separation_m, rel=1e-9)


@pytest.mark.unit
def test_hill_to_eci_round_trip_recovers_relative_state():
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = np.concatenate((rng.uniform(-5e3, 5e3, 3), rng.uniform(-10.0, 10.0, 3)))
        r_c, v_c = relative_state_hill_to_eci(R_INCLINED, V_INCLINED, state)
        recovered = relative_state_eci_to_hill(R_INCLINED, V_INCLINED, r_c, v_c)
        np.testing.assert_allclose(recovered[:3], state[:3], atol=1e-8)
        np.testing.assert_allclose(recovered[3:], state[3:], atol=1e-11)


@pytest.mark.unit
def test_zero_position_raises():
    with pytest.raises(DegenerateGeometryError, match="zero magnitude"):
        hill_basis(np.zeros(3), V_EQUATORIAL)


@pytest.mark.unit
def test_zero_velocity_raises():
    with pytest.raises(DegenerateGeometryError, match="zero magnitude"):
        hill_basis(R_EQUATORIAL, np.zeros(3))


@pytest.mark.unit
def test_purely_radial_trajectory_raises():
    """Position parallel to velocity means zero angular momentum, so no LVLH frame."""
    with pytest.raises(DegenerateGeometryError, match="parallel"):
        hill_basis(R_EQUATORIAL, R_EQUATORIAL * 1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros((3, 1)), np.zeros(6)])
def test_wrong_shape_raises(bad):
    with pytest.raises(ValueError, match="shape"):
        hill_basis(bad, V_EQUATORIAL)


@pytest.mark.unit
def test_non_finite_input_raises():
    with pytest.raises(ValueError, match="finite"):
        hill_basis(np.array([np.nan, 0.0, 0.0]), V_EQUATORIAL)
