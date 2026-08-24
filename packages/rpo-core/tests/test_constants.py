"""Constants and derived Keplerian quantities."""

import math

import pytest
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)

# ISS-like reference orbit used across the whole suite: 420 km circular.
A_ISS_M = R_EARTH_EQUATORIAL_M + 420.0e3


@pytest.mark.unit
def test_iss_like_mean_motion_and_period_match_published_values():
    n = mean_motion_rad_s(A_ISS_M)
    period = orbital_period_s(A_ISS_M)
    # A 420 km circular orbit has a period of ~93 minutes; anything outside 90-96 min
    # means mu, R_earth, or the formula is wrong.
    assert 90.0 * 60.0 < period < 96.0 * 60.0
    assert n == pytest.approx(2.0 * math.pi / period, rel=1e-15)


@pytest.mark.unit
def test_mean_motion_matches_circular_speed_over_radius():
    """For a circular orbit, n == v_circular / a. Independent route to the same number."""
    v_circular = math.sqrt(MU_EARTH_M3_S2 / A_ISS_M)
    assert mean_motion_rad_s(A_ISS_M) == pytest.approx(v_circular / A_ISS_M, rel=1e-14)


@pytest.mark.unit
def test_period_scales_as_three_halves_power():
    """Kepler's third law: doubling a multiplies the period by 2**1.5."""
    assert orbital_period_s(2.0 * A_ISS_M) == pytest.approx(
        orbital_period_s(A_ISS_M) * 2.0**1.5, rel=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize("bad_a", [0.0, -1.0, -A_ISS_M])
def test_non_positive_semi_major_axis_raises(bad_a):
    with pytest.raises(ValueError, match="semi_major_axis_m"):
        mean_motion_rad_s(bad_a)


@pytest.mark.unit
def test_non_positive_mu_raises():
    with pytest.raises(ValueError, match="mu_m3_s2"):
        mean_motion_rad_s(A_ISS_M, mu_m3_s2=0.0)
