r"""External-validation layer: ingest, alignment, interpolation, and comparison.

The oracle
----------
GMAT and STK are not installed on this machine (macOS arm64; no ``agi``, ``pystk``,
``comtypes`` or ``win32com``), so nothing here is an external-tool validation and nothing
here pretends to be. What *is* available is a genuinely independent analytic solution of the
same equations of motion: Kepler's equation plus the Lagrange :math:`f` and :math:`g`
functions, evaluated in closed form with **no numerical integration anywhere**.

:func:`kepler_propagate` below shares no code path with
:func:`rpo_core.propagate.propagate_two_body`. The repository propagates by integrating

.. math:: \ddot{\mathbf{r}} = -\mu \mathbf{r} / r^{3}

with DOP853; this file instead solves

.. math::

    \sqrt{\mu/a^{3}}\,\Delta t = \Delta E
        + \frac{\sigma_{0}}{\sqrt{a}}\bigl(1 - \cos\Delta E\bigr)
        - \Bigl(1 - \frac{r_{0}}{a}\Bigr)\sin\Delta E,
    \qquad \sigma_{0} = \frac{\mathbf{r}_{0}\cdot\mathbf{v}_{0}}{\sqrt{\mu}}

by Newton iteration and maps the result forward with

.. math::

    f = 1 - \frac{a}{r_{0}}\bigl(1 - \cos\Delta E\bigr), \qquad
    g = \Delta t - \sqrt{a^{3}/\mu}\,\bigl(\Delta E - \sin\Delta E\bigr),

.. math::

    \dot{f} = -\frac{\sqrt{\mu a}}{r\,r_{0}}\sin\Delta E, \qquad
    \dot{g} = 1 - \frac{a}{r}\bigl(1 - \cos\Delta E\bigr),

with :math:`\mathbf{r} = f\mathbf{r}_{0} + g\mathbf{v}_{0}` and
:math:`\mathbf{v} = \dot{f}\mathbf{r}_{0} + \dot{g}\mathbf{v}_{0}`. Agreement between the two
is therefore a real result about the integrator, not a regression test against stored
self-output. The oracle is itself checked first (against the exact circular solution, and
against energy and angular-momentum conservation) so that a failure cannot be blamed on the
wrong side.
"""

import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M, orbital_period_s
from rpo_core.frames import hill_basis
from rpo_core.propagate import propagate_two_body
from rpo_core.validation import (
    ARCSEC_RAD,
    DEFAULT_LAGRANGE_POINTS,
    EPHEMERIS_FORMAT,
    ComparisonReport,
    Ephemeris,
    EphemerisFormatError,
    EphemerisGapError,
    EphemerisTimeError,
    EphemerisUnitError,
    Epoch,
    EpochMismatchError,
    FrameMismatchError,
    InterpolationMethod,
    InterpolationRangeError,
    Provenance,
    ReferenceFrame,
    TimeScale,
    align_ephemeris,
    along_track_error_from_time_offset_m,
    check_alignment,
    compare_ephemerides,
    ephemeris_from_states,
    estimate_frame_tie_error_m,
    estimate_interpolation_error_m,
    interpolate_states,
    read_ephemeris,
    write_comparison_report,
    write_ephemeris,
)

# SRS §1.3 reference scenario: ISS-like 420 km circular, 51.6 deg inclination.
A_M = R_EARTH_EQUATORIAL_M + 420.0e3
PERIOD_S = orbital_period_s(A_M)
INCLINATION_RAD = math.radians(51.6)
V_CIRC_M_S = math.sqrt(MU_EARTH_M3_S2 / A_M)

STATE_CIRCULAR = np.array(
    [
        A_M,
        0.0,
        0.0,
        0.0,
        V_CIRC_M_S * math.cos(INCLINATION_RAD),
        V_CIRC_M_S * math.sin(INCLINATION_RAD),
    ]
)

# A mildly eccentric companion case, so nothing is proved only on the degenerate circle.
_E = 0.01
_RP = A_M * (1.0 - _E)
_VP = math.sqrt(MU_EARTH_M3_S2 * (1.0 + _E) / _RP)
STATE_ECCENTRIC = np.array(
    [_RP, 0.0, 0.0, 0.0, _VP * math.cos(INCLINATION_RAD), _VP * math.sin(INCLINATION_RAD)]
)

EPOCH = Epoch.from_iso("2026-03-01T00:00:00.000000", TimeScale.TAI)


# ======================================================================================
# The independent oracle: Kepler's equation + Lagrange f and g. No integration.
# ======================================================================================


def _solve_delta_eccentric_anomaly(mean_arg: float, sigma0: float, r0: float, a: float) -> float:
    """Solve the universal-in-DeltaE Kepler equation by Newton iteration.

    ``mean_arg = sqrt(mu / a**3) * dt``. The unknown ``dE`` satisfies

        mean_arg = dE + sigma0 / sqrt(a) * (1 - cos dE) - (1 - r0/a) * sin dE

    which is monotone in ``dE`` for a bound orbit (its derivative is ``r/a > 0``), so Newton
    from ``dE = mean_arg`` converges for every eccentricity used here.
    """
    root_a = math.sqrt(a)
    one_minus = 1.0 - r0 / a
    delta_e = mean_arg
    for _ in range(200):
        residual = (
            delta_e
            + sigma0 / root_a * (1.0 - math.cos(delta_e))
            - one_minus * math.sin(delta_e)
            - mean_arg
        )
        derivative = 1.0 + sigma0 / root_a * math.sin(delta_e) - one_minus * math.cos(delta_e)
        step = residual / derivative
        delta_e -= step
        if abs(step) < 1.0e-15 * max(1.0, abs(delta_e)):
            break
    else:  # pragma: no cover - defensive; never reached for the cases exercised here
        raise AssertionError(f"Kepler oracle failed to converge, mean_arg={mean_arg}")
    return delta_e


def kepler_propagate(
    state0: npt.NDArray[np.float64], times_s: npt.NDArray[np.float64], mu: float = MU_EARTH_M3_S2
) -> npt.NDArray[np.float64]:
    """Propagate a bound two-body orbit analytically. Independent of ``propagate_two_body``."""
    r0_vec = state0[:3]
    v0_vec = state0[3:]
    r0 = float(np.linalg.norm(r0_vec))
    v0 = float(np.linalg.norm(v0_vec))
    a = 1.0 / (2.0 / r0 - v0**2 / mu)
    sigma0 = float(r0_vec @ v0_vec) / math.sqrt(mu)
    root_mu_over_a3 = math.sqrt(mu / a**3)

    out = np.empty((times_s.size, 6), dtype=np.float64)
    for index, dt in enumerate(times_s):
        delta_e = _solve_delta_eccentric_anomaly(root_mu_over_a3 * float(dt), sigma0, r0, a)
        cos_de = math.cos(delta_e)
        sin_de = math.sin(delta_e)
        f = 1.0 - a / r0 * (1.0 - cos_de)
        g = float(dt) - math.sqrt(a**3 / mu) * (delta_e - sin_de)
        r_vec = f * r0_vec + g * v0_vec
        r = float(np.linalg.norm(r_vec))
        f_dot = -math.sqrt(mu * a) / (r * r0) * sin_de
        g_dot = 1.0 - a / r * (1.0 - cos_de)
        out[index, :3] = r_vec
        out[index, 3:] = f_dot * r0_vec + g_dot * v0_vec
    return out


@pytest.mark.unit
def test_oracle_reproduces_the_exact_circular_solution():
    """The oracle is checked before it is trusted.

    For an exactly circular orbit the closed-form answer is a rigid rotation of the initial
    state through ``n*t`` about the angular-momentum axis. Nothing in that statement involves
    Kepler's equation, so agreement pins the oracle independently.
    """
    n = math.sqrt(MU_EARTH_M3_S2 / A_M**3)
    times = np.linspace(0.0, PERIOD_S, 37)
    got = kepler_propagate(STATE_CIRCULAR, times)

    h = np.cross(STATE_CIRCULAR[:3], STATE_CIRCULAR[3:])
    axis = h / np.linalg.norm(h)
    for index, t in enumerate(times):
        angle = n * float(t)
        cross = np.cross(axis, STATE_CIRCULAR[:3])
        expected_r = (
            STATE_CIRCULAR[:3] * math.cos(angle)
            + cross * math.sin(angle)
            + axis * float(axis @ STATE_CIRCULAR[:3]) * (1.0 - math.cos(angle))
        )
        # 1e-7 m over a full orbit: measured worst is ~2e-8 m, set by the double-precision
        # round-off of the trigonometric reconstruction, not by either algorithm.
        np.testing.assert_allclose(got[index, :3], expected_r, atol=1.0e-7)


@pytest.mark.unit
def test_oracle_conserves_energy_and_angular_momentum_on_an_eccentric_orbit():
    times = np.linspace(0.0, 10.0 * PERIOD_S, 501)
    states = kepler_propagate(STATE_ECCENTRIC, times)
    radius = np.linalg.norm(states[:, :3], axis=1)
    speed = np.linalg.norm(states[:, 3:], axis=1)
    energy = 0.5 * speed**2 - MU_EARTH_M3_S2 / radius
    momentum = np.cross(states[:, :3], states[:, 3:])
    # Relative drift over 10 orbits, measured: ~1e-15 in energy, ~1e-15 in |h|.
    assert float(np.ptp(energy) / abs(energy[0])) < 1.0e-12
    assert float(np.ptp(np.linalg.norm(momentum, axis=1)) / np.linalg.norm(momentum[0])) < 1.0e-12


@pytest.mark.unit
def test_oracle_is_not_trivially_the_initial_state():
    """Complement: the oracle actually moves the spacecraft.

    Without this, every "agreement" test below would still pass if ``kepler_propagate``
    returned ``state0`` broadcast over the time grid and the propagator did too.
    """
    half = kepler_propagate(STATE_CIRCULAR, np.array([0.5 * PERIOD_S]))[0]
    assert float(np.linalg.norm(half[:3] - STATE_CIRCULAR[:3])) > 1.0e7


# ======================================================================================
# The headline validation result
# ======================================================================================


@pytest.mark.integration
def test_propagate_two_body_agrees_with_the_analytic_kepler_reference(capsys):
    """Measured agreement between the repository propagator and an independent solution.

    This is a real validation result and the project may quote it. Both sides solve the same
    equations of motion by methods with nothing in common: DOP853 numerical integration
    versus Kepler's equation with f and g. The bound below was set by printing the measured
    value first; see the comment.
    """
    times = np.arange(0.0, 86400.0 + 1.0, 60.0)
    integrated = propagate_two_body(STATE_CIRCULAR, times)
    analytic = kepler_propagate(STATE_CIRCULAR, times)

    position_error = np.linalg.norm(integrated[:, :3] - analytic[:, :3], axis=1)
    velocity_error = np.linalg.norm(integrated[:, 3:] - analytic[:, 3:], axis=1)
    with capsys.disabled():
        print(
            f"\n[measured] propagate_two_body vs analytic Kepler over 86400 s "
            f"({86400.0 / PERIOD_S:.2f} orbits): "
            f"max |dr| = {position_error.max():.4e} m, "
            f"RMS |dr| = {np.sqrt(np.mean(position_error**2)):.4e} m, "
            f"max |dv| = {velocity_error.max():.4e} m/s"
        )
    # Bound from measurement with an order of magnitude of headroom: the measured worst is
    # printed above and is O(1e-5) m at rtol=atol=1e-12 over 15.5 orbits.
    assert float(position_error.max()) < 1.0e-3
    assert float(velocity_error.max()) < 1.0e-6


@pytest.mark.integration
def test_agreement_degrades_when_the_integrator_tolerance_is_loosened():
    """Complement: the agreement above is a knife edge on tolerance, not a plateau.

    If the comparison were insensitive to ``rtol``/``atol`` it would be measuring something
    other than integration error, and the tight bound above would be meaningless.
    """
    times = np.arange(0.0, 4.0 * PERIOD_S, 120.0)
    analytic = kepler_propagate(STATE_CIRCULAR, times)
    errors = []
    for tol in (1.0e-12, 1.0e-9, 1.0e-6):
        integrated = propagate_two_body(STATE_CIRCULAR, times, rtol=tol, atol=tol)
        errors.append(float(np.max(np.linalg.norm(integrated[:, :3] - analytic[:, :3], axis=1))))
    assert errors[0] < errors[1] < errors[2]
    assert errors[2] / max(errors[0], 1.0e-30) > 1.0e3


# ======================================================================================
# Epoch arithmetic
# ======================================================================================


@pytest.mark.unit
def test_epoch_iso_round_trip_and_known_mjd():
    # 2000-01-01T00:00:00 is MJD 51544 by definition; an independent anchor for the
    # calendar arithmetic, which is otherwise self-referential.
    assert Epoch.from_iso("2000-01-01T00:00:00", TimeScale.TAI).mjd_day == 51544
    epoch = Epoch.from_iso("2026-03-01T12:34:56.789012", TimeScale.TAI)
    assert epoch.to_iso() == "2026-03-01T12:34:56.789012"


@pytest.mark.unit
def test_epoch_shift_across_midnight_normalises_the_day():
    epoch = Epoch.from_iso("2026-03-01T23:59:59.000000", TimeScale.TAI)
    later = epoch.shifted(2.0)
    assert later.mjd_day == epoch.mjd_day + 1
    assert later.seconds_of_day == pytest.approx(1.0, abs=1e-12)
    assert later.offset_s(epoch) == pytest.approx(2.0, abs=1e-9)


@pytest.mark.unit
def test_tai_to_tt_offset_is_exactly_32_184_seconds():
    tai = Epoch.from_iso("2026-03-01T00:00:00", TimeScale.TAI)
    tt = tai.in_scale(TimeScale.TT)
    assert tt.scale is TimeScale.TT
    assert tt.seconds_of_day == pytest.approx(32.184, abs=1e-12)


@pytest.mark.unit
def test_epoch_rejects_a_leap_second_rather_than_wrapping_it():
    with pytest.raises(ValueError, match=r"out-of-range time field.*7\.7 km"):
        Epoch.from_iso("2026-06-30T23:59:60.000", TimeScale.UTC)


@pytest.mark.unit
def test_epoch_rejects_an_unparseable_string():
    with pytest.raises(ValueError, match="is not of the form YYYY-MM-DD"):
        Epoch.from_iso("1 March 2026 noon", TimeScale.TAI)


@pytest.mark.unit
def test_epoch_rejects_an_out_of_range_month():
    with pytest.raises(ValueError, match="has month 13, expected 1-12"):
        Epoch.from_iso("2026-13-01T00:00:00", TimeScale.TAI)


@pytest.mark.unit
def test_epoch_rejects_a_non_normalised_seconds_of_day():
    with pytest.raises(ValueError, match=r"seconds_of_day must lie in \[0, 86400\)"):
        Epoch(60000, 86400.0, TimeScale.TAI)


@pytest.mark.unit
def test_epoch_rejects_a_non_finite_shift():
    with pytest.raises(ValueError, match="seconds must be finite"):
        EPOCH.shifted(math.nan)


@pytest.mark.unit
def test_epoch_rejects_a_non_finite_mjd():
    with pytest.raises(ValueError, match="mjd must be finite"):
        Epoch.from_mjd(math.inf, TimeScale.TAI)


@pytest.mark.unit
def test_differencing_epochs_across_time_scales_raises():
    tai = Epoch.from_iso("2026-03-01T00:00:00", TimeScale.TAI)
    utc = Epoch.from_iso("2026-03-01T00:00:00", TimeScale.UTC)
    with pytest.raises(EpochMismatchError, match="different time scales"):
        tai.offset_s(utc)


@pytest.mark.unit
def test_converting_to_utc_without_a_supplied_offset_raises_and_quotes_the_cost():
    tai = Epoch.from_iso("2026-03-01T00:00:00", TimeScale.TAI)
    with pytest.raises(EpochMismatchError, match=r"leap-second.*table.*along-track"):
        tai.in_scale(TimeScale.UTC)


@pytest.mark.unit
def test_supplied_time_scale_offset_is_applied():
    tai = Epoch.from_iso("2026-03-01T00:00:00", TimeScale.TAI)
    utc = tai.in_scale(TimeScale.UTC, offset_s=-37.0)
    assert utc.scale is TimeScale.UTC
    assert utc.mjd_day == tai.mjd_day - 1
    assert utc.seconds_of_day == pytest.approx(86400.0 - 37.0, abs=1e-9)


@pytest.mark.unit
def test_one_second_of_epoch_error_is_about_seven_point_seven_kilometres():
    # Independent hand check: v = sqrt(mu/a) at 420 km altitude = 7658.6 m/s.
    assert along_track_error_from_time_offset_m(1.0) == pytest.approx(V_CIRC_M_S, rel=1e-12)
    assert along_track_error_from_time_offset_m(37.0) == pytest.approx(37.0 * V_CIRC_M_S, rel=1e-12)
    assert along_track_error_from_time_offset_m(-2.0) == pytest.approx(2.0 * V_CIRC_M_S, rel=1e-12)


@pytest.mark.unit
def test_along_track_helper_rejects_bad_inputs():
    with pytest.raises(ValueError, match="must be finite"):
        along_track_error_from_time_offset_m(math.nan)
    with pytest.raises(ValueError, match="speed_m_s must be >= 0"):
        along_track_error_from_time_offset_m(1.0, -1.0)


# ======================================================================================
# The GCRF-approximation budget
# ======================================================================================


@pytest.mark.unit
def test_frame_tie_error_over_one_day_in_leo_matches_a_hand_computation(capsys):
    """Precession + nutation over a day, computed by hand from published rates."""
    precession_rad = 50.2879 * ARCSEC_RAD / (365.25 * 86400.0) * 86400.0
    nutation_rad = 0.15 * ARCSEC_RAD
    expected_m = (precession_rad + nutation_rad) * A_M
    got = estimate_frame_tie_error_m(86400.0)
    with capsys.disabled():
        print(
            f"\n[measured] GCRF-approximation budget in LEO: {got:.2f} m over 1 day "
            f"(precession {precession_rad * A_M:.2f} m + nutation {nutation_rad * A_M:.2f} m), "
            f"{estimate_frame_tie_error_m(7.0 * 86400.0):.1f} m over 7 days; "
            f"EME2000 frame bias adds "
            f"{estimate_frame_tie_error_m(0.0, include_frame_bias=True):.2f} m, constant"
        )
    assert got == pytest.approx(expected_m, rel=1e-12)
    # Order 10 m over a day is the claim made in the module docstring and the runbook.
    assert 5.0 < got < 20.0


@pytest.mark.unit
def test_frame_tie_error_grows_and_the_nutation_term_saturates():
    day = estimate_frame_tie_error_m(86400.0)
    week = estimate_frame_tie_error_m(7.0 * 86400.0)
    assert week > day
    # Both terms are still growing over the first week, so the budget is very nearly linear.
    assert week == pytest.approx(7.0 * day, rel=1e-9)

    # The nutation term is bounded by the 17.2 arcsec amplitude of the 18.6-year principal
    # term, which the 0.15 arcsec/day drift bound reaches after 17.2/0.15 = 115 days. Beyond
    # that only precession keeps growing, so a daily increment taken at 200 days must equal
    # the precession-only rate: 50.2879"/yr / 365.25 * 6.798e6 m = 4.54 m/day. Without the
    # saturation the increment would still be the ~9.5 m/day of the first week.
    late = estimate_frame_tie_error_m(201.0 * 86400.0) - estimate_frame_tie_error_m(200.0 * 86400.0)
    precession_only_m = 50.2879 * ARCSEC_RAD / 365.25 * A_M
    assert late == pytest.approx(precession_only_m, rel=1e-9)
    assert late < 0.55 * day


@pytest.mark.unit
def test_frame_bias_term_is_about_a_metre_at_leo():
    bias_only = estimate_frame_tie_error_m(0.0, include_frame_bias=True)
    assert bias_only == pytest.approx(23.0e-3 * ARCSEC_RAD * A_M, rel=1e-12)
    assert 0.5 < bias_only < 1.5


@pytest.mark.unit
def test_frame_tie_estimate_rejects_bad_inputs():
    with pytest.raises(ValueError, match="elapsed_s must be finite and >= 0"):
        estimate_frame_tie_error_m(-1.0)
    with pytest.raises(ValueError, match="radius_m must be finite and > 0"):
        estimate_frame_tie_error_m(1.0, 0.0)


# ======================================================================================
# Ephemeris construction, writing and reading
# ======================================================================================


def _reference_ephemeris(
    duration_s: float = 86400.0,
    step_s: float = 60.0,
    *,
    state0: npt.NDArray[np.float64] | None = None,
    epoch: Epoch = EPOCH,
    frame: ReferenceFrame = ReferenceFrame.GCRF_APPROX,
    provenance: Provenance = Provenance.SYNTHETIC_REFERENCE,
) -> Ephemeris:
    """Build an analytic-Kepler ephemeris: the independent reference the harness ingests."""
    times = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    states = kepler_propagate(STATE_CIRCULAR if state0 is None else state0, times)
    return ephemeris_from_states(
        times,
        states,
        epoch=epoch,
        frame=frame,
        provenance=provenance,
        source="analytic Kepler f-and-g (tests)",
    )


def _internal_ephemeris(
    duration_s: float = 86400.0, step_s: float = 60.0, *, epoch: Epoch = EPOCH
) -> Ephemeris:
    """Build the repository's own ephemeris via ``propagate_two_body``."""
    times = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    states = propagate_two_body(STATE_CIRCULAR, times)
    return ephemeris_from_states(
        times,
        states,
        epoch=epoch,
        frame=ReferenceFrame.GCRF_APPROX,
        provenance=Provenance.SYNTHETIC_REFERENCE,
        source="rpo_core.propagate.propagate_two_body",
    )


@pytest.mark.unit
def test_ephemeris_round_trips_through_the_file_format(tmp_path: Path):
    original = _reference_ephemeris(duration_s=3600.0)
    path = write_ephemeris(tmp_path / "ref.eph", original)
    restored = read_ephemeris(path)

    # repr-formatted floats round-trip bitwise, so this is exact equality, not a tolerance.
    np.testing.assert_array_equal(restored.times_s, original.times_s)
    np.testing.assert_array_equal(restored.states, original.states)
    assert restored.frame is original.frame
    assert restored.provenance is original.provenance
    assert restored.source == original.source
    assert restored.epoch == original.epoch


@pytest.mark.unit
def test_written_header_declares_the_contract(tmp_path: Path):
    path = write_ephemeris(tmp_path / "ref.eph", _reference_ephemeris(duration_s=600.0))
    text = path.read_text(encoding="utf-8")
    assert f"# format: {EPHEMERIS_FORMAT}" in text
    assert "# frame: GCRF_APPROX" in text
    assert "# time_scale: TAI" in text
    assert "# provenance: synthetic_reference" in text
    assert "t,x,y,z,vx,vy,vz" in text


@pytest.mark.unit
def test_kilometre_units_are_converted_at_ingest(tmp_path: Path):
    reference = _reference_ephemeris(duration_s=600.0)
    lines = [
        f"# format: {EPHEMERIS_FORMAT}",
        "# frame: GCRF_APPROX",
        "# time_scale: TAI",
        f"# epoch: {EPOCH.to_iso()}",
        "# position_unit: km",
        "# velocity_unit: km/s",
        "t,x,y,z,vx,vy,vz",
    ]
    for index in range(reference.size):
        row = [repr(float(reference.times_s[index]))]
        row += [repr(float(v) / 1.0e3) for v in reference.states[index]]
        lines.append(",".join(row))
    path = tmp_path / "km.eph"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    restored = read_ephemeris(path)
    np.testing.assert_allclose(restored.states, reference.states, rtol=1e-12)


@pytest.mark.unit
def test_columns_are_matched_by_name_not_position(tmp_path: Path):
    """A reordered export still reads correctly; positional parsing would silently swap axes."""
    reference = _reference_ephemeris(duration_s=600.0)
    header = [
        f"# format: {EPHEMERIS_FORMAT}",
        "# frame: GCRF_APPROX",
        "# time_scale: TAI",
        f"# epoch: {EPOCH.to_iso()}",
        "# position_unit: m",
        "# velocity_unit: m/s",
        "vx,vy,vz,t,x,y,z",
    ]
    rows = []
    for index in range(reference.size):
        s = reference.states[index]
        rows.append(
            ",".join(
                repr(float(v))
                for v in (s[3], s[4], s[5], float(reference.times_s[index]), s[0], s[1], s[2])
            )
        )
    path = tmp_path / "reordered.eph"
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")

    restored = read_ephemeris(path)
    np.testing.assert_array_equal(restored.states, reference.states)


@pytest.mark.unit
def test_whitespace_separated_rows_are_accepted(tmp_path: Path):
    reference = _reference_ephemeris(duration_s=600.0)
    lines = [
        f"# format: {EPHEMERIS_FORMAT}",
        "# frame: GCRF_APPROX",
        "# time_scale: TAI",
        f"# epoch: {EPOCH.to_iso()}",
        "# position_unit: m",
        "# velocity_unit: m/s",
        "t x y z vx vy vz",
    ]
    for index in range(reference.size):
        values = (float(reference.times_s[index]), *(float(v) for v in reference.states[index]))
        lines.append(" ".join(repr(v) for v in values))
    path = tmp_path / "ws.eph"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert read_ephemeris(path).size == reference.size


# --------------------------------------------------------------------------------------
# Malformed ephemerides: every one raises, and each with its own message
# --------------------------------------------------------------------------------------


def _write_variant(tmp_path: Path, name: str, *, header: list[str], rows: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    return path


_GOOD_HEADER = [
    f"# format: {EPHEMERIS_FORMAT}",
    "# frame: GCRF_APPROX",
    "# time_scale: TAI",
    f"# epoch: {EPOCH.to_iso()}",
    "# position_unit: m",
    "# velocity_unit: m/s",
    "t,x,y,z,vx,vy,vz",
]


def _good_rows(count: int = 20, step_s: float = 60.0) -> list[str]:
    times = np.arange(count, dtype=np.float64) * step_s
    states = kepler_propagate(STATE_CIRCULAR, times)
    return [
        ",".join(repr(v) for v in (float(times[i]), *(float(x) for x in states[i])))
        for i in range(count)
    ]


@pytest.mark.unit
def test_non_monotonic_times_raise(tmp_path: Path):
    rows = _good_rows()
    rows[5], rows[6] = rows[6], rows[5]
    path = _write_variant(tmp_path, "swapped.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisTimeError, match="times must be strictly increasing"):
        read_ephemeris(path)


@pytest.mark.unit
def test_duplicated_times_raise(tmp_path: Path):
    rows = _good_rows()
    rows[7] = rows[6]
    path = _write_variant(tmp_path, "duplicate.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisTimeError, match="times must be strictly increasing"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_gap_raises(tmp_path: Path):
    times = np.arange(20, dtype=np.float64) * 60.0
    times[10:] += 600.0
    states = kepler_propagate(STATE_CIRCULAR, times)
    rows = [
        ",".join(repr(v) for v in (float(times[i]), *(float(x) for x in states[i])))
        for i in range(times.size)
    ]
    path = _write_variant(tmp_path, "gap.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisGapError, match="sample spacing jumps at index 9"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_gap_is_accepted_when_the_check_is_explicitly_disabled(tmp_path: Path):
    """Complement: the gap check is a real gate, not an unreachable branch."""
    times = np.arange(20, dtype=np.float64) * 60.0
    times[10:] += 600.0
    states = kepler_propagate(STATE_CIRCULAR, times)
    rows = [
        ",".join(repr(v) for v in (float(times[i]), *(float(x) for x in states[i])))
        for i in range(times.size)
    ]
    path = _write_variant(tmp_path, "gap_ok.eph", header=_GOOD_HEADER, rows=rows)
    # The velocity cross-check still fires: a central difference straddling the gap spans
    # 660 s instead of 120 s, and its truncation error is O(h**2), so it disagrees with the
    # tabulated velocity by 33 %. That is the gap showing up through a second, independent
    # check -- which is why both have to be switched off to read a gapped file at all.
    with pytest.raises(EphemerisUnitError, match="velocity columns disagree"):
        read_ephemeris(path, max_step_ratio=None)
    assert read_ephemeris(path, max_step_ratio=None, velocity_rel_tol=None).size == 20


@pytest.mark.unit
def test_a_missing_column_raises_and_names_it(tmp_path: Path):
    header = list(_GOOD_HEADER)
    header[-1] = "t,x,y,z,vx,vy"
    rows = [",".join(row.split(",")[:-1]) for row in _good_rows()]
    path = _write_variant(tmp_path, "missing_col.eph", header=header, rows=rows)
    with pytest.raises(EphemerisFormatError, match=r"missing required column\(s\) \['vz'\]"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_short_row_raises_rather_than_being_skipped(tmp_path: Path):
    rows = _good_rows()
    rows[4] = ",".join(rows[4].split(",")[:-1])
    path = _write_variant(tmp_path, "short_row.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisFormatError, match=r"row has 6 field\(s\), expected 7"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_non_numeric_field_raises(tmp_path: Path):
    rows = _good_rows()
    fields = rows[3].split(",")
    fields[2] = "n/a"
    rows[3] = ",".join(fields)
    path = _write_variant(tmp_path, "nan_field.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisFormatError, match="field is not a number"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_non_finite_value_raises(tmp_path: Path):
    rows = _good_rows()
    fields = rows[3].split(",")
    fields[2] = "nan"
    rows[3] = ",".join(fields)
    path = _write_variant(tmp_path, "inf.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisFormatError, match="is not finite"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_missing_header_key_raises_and_names_it(tmp_path: Path):
    header = [line for line in _GOOD_HEADER if not line.startswith("# epoch")]
    path = _write_variant(tmp_path, "no_epoch.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisFormatError, match=r"missing required header key\(s\) \['epoch'\]"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_foreign_format_token_raises(tmp_path: Path):
    header = list(_GOOD_HEADER)
    header[0] = "# format: stk-ephemeris/1.0"
    path = _write_variant(tmp_path, "foreign.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisFormatError, match="does not sniff foreign ephemeris formats"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_malformed_header_comment_raises(tmp_path: Path):
    header = [*_GOOD_HEADER[:-1], "# this is not a key value pair", _GOOD_HEADER[-1]]
    path = _write_variant(tmp_path, "bad_comment.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisFormatError, match="is not 'key: value'"):
        read_ephemeris(path)


@pytest.mark.unit
def test_an_unknown_frame_raises(tmp_path: Path):
    header = list(_GOOD_HEADER)
    header[1] = "# frame: J2000ISH"
    path = _write_variant(tmp_path, "bad_frame.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisFormatError, match="unknown frame 'J2000ISH'"):
        read_ephemeris(path)


@pytest.mark.unit
def test_an_unknown_time_scale_raises(tmp_path: Path):
    header = list(_GOOD_HEADER)
    header[2] = "# time_scale: GPS"
    path = _write_variant(tmp_path, "bad_scale.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisFormatError, match="unknown time_scale 'GPS'"):
        read_ephemeris(path)


@pytest.mark.unit
def test_an_unknown_provenance_raises(tmp_path: Path):
    header = [*_GOOD_HEADER[:-1], "# provenance: probably_fine", _GOOD_HEADER[-1]]
    path = _write_variant(tmp_path, "bad_prov.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisFormatError, match="unknown provenance 'probably_fine'"):
        read_ephemeris(path)


@pytest.mark.unit
def test_an_unknown_unit_token_raises(tmp_path: Path):
    header = list(_GOOD_HEADER)
    header[4] = "# position_unit: furlong"
    path = _write_variant(tmp_path, "furlong.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisUnitError, match="unknown position_unit 'furlong'"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_metre_kilometre_unit_mismatch_raises(tmp_path: Path):
    """Data in metres, header claiming kilometres: the factor-of-1000 classic."""
    header = list(_GOOD_HEADER)
    header[4] = "# position_unit: km"
    path = _write_variant(tmp_path, "unit_mismatch.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisUnitError, match="outside the plausible range"):
        read_ephemeris(path)


@pytest.mark.unit
def test_the_plausibility_check_can_be_disabled(tmp_path: Path):
    """Complement: the magnitude gate is what fires above, not some other check."""
    header = list(_GOOD_HEADER)
    header[4] = "# position_unit: km"
    path = _write_variant(tmp_path, "unit_ok.eph", header=header, rows=_good_rows())
    with pytest.raises(EphemerisUnitError, match="velocity columns disagree"):
        read_ephemeris(path, check_units=False)
    assert read_ephemeris(path, check_units=False, velocity_rel_tol=None).size == 20


@pytest.mark.unit
def test_velocity_columns_inconsistent_with_position_raise(tmp_path: Path):
    rows = []
    times = np.arange(20, dtype=np.float64) * 60.0
    states = kepler_propagate(STATE_CIRCULAR, times)
    states[:, 3:] *= -1.0  # sign-flipped velocity block: a 200 % disagreement
    for i in range(times.size):
        rows.append(",".join(repr(v) for v in (float(times[i]), *(float(x) for x in states[i]))))
    path = _write_variant(tmp_path, "vflip.eph", header=_GOOD_HEADER, rows=rows)
    with pytest.raises(EphemerisUnitError, match="velocity columns disagree"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_clean_ephemeris_passes_the_velocity_check_with_headroom(capsys, tmp_path: Path):
    """Complement, and the measurement the 1e-2 default is set from."""
    reference = _reference_ephemeris(duration_s=3600.0)
    dt = reference.times_s[2:] - reference.times_s[:-2]
    difference = (reference.states[2:, :3] - reference.states[:-2, :3]) / dt[:, None]
    residual = np.linalg.norm(difference - reference.states[1:-1, 3:], axis=1) / np.linalg.norm(
        reference.states[1:-1, 3:], axis=1
    )
    with capsys.disabled():
        print(
            f"\n[measured] velocity/central-difference residual at 60 s spacing: "
            f"worst {residual.max():.3e} relative (limit 1e-2)"
        )
    assert float(residual.max()) < 1.0e-2
    path = write_ephemeris(tmp_path / "clean.eph", reference)
    assert read_ephemeris(path).size == reference.size


@pytest.mark.unit
def test_a_file_with_no_data_rows_raises(tmp_path: Path):
    path = _write_variant(tmp_path, "empty.eph", header=_GOOD_HEADER, rows=[])
    with pytest.raises(EphemerisFormatError, match="no data rows found"):
        read_ephemeris(path)


@pytest.mark.unit
def test_a_file_with_no_column_header_raises(tmp_path: Path):
    path = _write_variant(tmp_path, "noheader.eph", header=_GOOD_HEADER[:-1], rows=[])
    with pytest.raises(EphemerisFormatError, match="no column header line found"):
        read_ephemeris(path)


@pytest.mark.unit
def test_ephemeris_from_states_rejects_a_wrong_shape():
    with pytest.raises(EphemerisFormatError, match=r"states must have shape \(3, 6\)"):
        ephemeris_from_states(
            np.array([0.0, 1.0, 2.0]),
            np.zeros((3, 5)),
            epoch=EPOCH,
            frame=ReferenceFrame.GCRF_APPROX,
        )


@pytest.mark.unit
def test_ephemeris_from_states_rejects_an_empty_time_array():
    with pytest.raises(EphemerisFormatError, match="non-empty 1-D array"):
        ephemeris_from_states(
            np.array([]), np.zeros((0, 6)), epoch=EPOCH, frame=ReferenceFrame.GCRF_APPROX
        )


@pytest.mark.unit
def test_ephemeris_arrays_are_read_only():
    ephemeris = _reference_ephemeris(duration_s=600.0)
    with pytest.raises(ValueError, match="read-only"):
        ephemeris.states[0, 0] = 0.0


@pytest.mark.unit
def test_provenance_defaults_to_unknown_and_is_not_silently_upgraded(tmp_path: Path):
    header = [line for line in _GOOD_HEADER if not line.startswith("# provenance")]
    path = _write_variant(tmp_path, "noprov.eph", header=header, rows=_good_rows())
    assert read_ephemeris(path).provenance is Provenance.UNKNOWN


# ======================================================================================
# Interpolation
# ======================================================================================


@pytest.mark.unit
def test_interpolation_reproduces_the_samples_it_was_built_from():
    reference = _reference_ephemeris(duration_s=3600.0)
    got = interpolate_states(reference.times_s, reference.states, reference.times_s)
    np.testing.assert_allclose(got, reference.states, atol=1.0e-6)


@pytest.mark.unit
def test_measured_interpolation_error_is_far_below_any_reported_difference(capsys):
    """Measure the interpolation error against truth, not against an estimate.

    The analytic oracle gives exact states at the half-step times, so this is the true
    interpolation error, not a proxy. It is also compared against
    ``estimate_interpolation_error_m``, which is what the report carries and which must be
    available on a real tool export where no truth exists.
    """
    coarse = _reference_ephemeris(duration_s=7200.0, step_s=60.0)
    fine = _reference_ephemeris(duration_s=7200.0, step_s=30.0)
    midpoints = 0.5 * (coarse.times_s[:-1] + coarse.times_s[1:])
    truth = kepler_propagate(STATE_CIRCULAR, midpoints)

    measured = {}
    for method, points in (
        (InterpolationMethod.LAGRANGE, DEFAULT_LAGRANGE_POINTS),
        (InterpolationMethod.LAGRANGE, 4),
        (InterpolationMethod.HERMITE, 0),
    ):
        got = interpolate_states(
            coarse.times_s,
            coarse.states,
            midpoints,
            method=method,
            points=points or DEFAULT_LAGRANGE_POINTS,
        )
        measured[(method.value, points)] = float(
            np.max(np.linalg.norm(got[:, :3] - truth[:, :3], axis=1))
        )

    # Mid-span only. The worst-case above is set by the four intervals at each end, where the
    # Lagrange window cannot be centred and the node product is an order of magnitude larger.
    got_interior = interpolate_states(coarse.times_s, coarse.states, midpoints[4:-4])
    interior = float(np.max(np.linalg.norm(got_interior[:, :3] - truth[4:-4, :3], axis=1)))

    fine_mid = 0.5 * (fine.times_s[:-1] + fine.times_s[1:])
    fine_truth = kepler_propagate(STATE_CIRCULAR, fine_mid)
    got_fine = interpolate_states(fine.times_s, fine.states, fine_mid)
    fine_worst = float(np.max(np.linalg.norm(got_fine[:, :3] - fine_truth[:, :3], axis=1)))

    estimated = estimate_interpolation_error_m(coarse)
    lagrange8 = measured[("lagrange", DEFAULT_LAGRANGE_POINTS)]
    with capsys.disabled():
        print(
            f"\n[measured] interpolation error against the analytic truth, LEO:"
            f"\n    60 s spacing: Lagrange-8 {lagrange8:.3e} m worst / "
            f"{interior:.3e} m mid-span, "
            f"Lagrange-4 {measured[('lagrange', 4)]:.3e} m, "
            f"Hermite-cubic {measured[('hermite', 0)]:.3e} m"
            f"\n    30 s spacing: Lagrange-8 {fine_worst:.3e} m worst"
            f"\n    Richardson estimate (no truth needed): {estimated:.3e} m at 60 s"
        )

    # At 30 s spacing the interpolation error is ~1.5e-7 m, three decades below the ~8.4e-5 m
    # difference the harness reports between the propagator and the analytic reference. That
    # is the margin the headline comparison relies on, and the reason the runbook asks for a
    # 30 s export. At 60 s the two are within a factor of two of each other, which is exactly
    # what ``interpolation_is_negligible`` exists to catch.
    assert fine_worst < 1.0e-6
    assert measured[("lagrange", DEFAULT_LAGRANGE_POINTS)] < 1.0e-4
    # End windows dominate the worst case; mid-span is an order of magnitude better.
    assert interior < 0.2 * measured[("lagrange", DEFAULT_LAGRANGE_POINTS)]
    # Order check: 8-point Lagrange beats both fourth-order schemes by four decades.
    assert measured[("lagrange", DEFAULT_LAGRANGE_POINTS)] < 1.0e-3 * measured[("hermite", 0)]
    # Between the two fourth-order schemes, cubic Hermite wins by about 15x -- measured
    # 0.37 m against 5.5 m. Both are O(h^4); Hermite's error constant is h^4/384 because it
    # is given the slopes, while 4-point Lagrange has to infer them from positions alone.
    # A cubic that knows the velocities is not the same cubic as one that does not.
    assert measured[("hermite", 0)] < measured[("lagrange", 4)]
    assert measured[("lagrange", 4)] / measured[("hermite", 0)] > 5.0
    # The Richardson estimate tracks the measured worst case within a factor of two, so the
    # number the report carries is usable on a real export where no truth exists.
    assert 0.5 < estimated / measured[("lagrange", DEFAULT_LAGRANGE_POINTS)] < 2.0


@pytest.mark.unit
def test_interpolation_error_falls_at_the_expected_order(capsys):
    """Convergence behaviour, not a hand-picked threshold: halving h must gain ~2**8."""
    fine = _reference_ephemeris(duration_s=7200.0, step_s=30.0)
    coarse = _reference_ephemeris(duration_s=7200.0, step_s=60.0)
    fine_error = estimate_interpolation_error_m(fine)
    coarse_error = estimate_interpolation_error_m(coarse)
    ratio = coarse_error / fine_error
    with capsys.disabled():
        print(
            f"\n[measured] interpolation error halving the step 60 s -> 30 s: "
            f"{coarse_error:.3e} m -> {fine_error:.3e} m, ratio {ratio:.1f} "
            f"(O(h^8) predicts 2**8 = 256)"
        )
    assert fine_error < coarse_error
    # Eight-point Lagrange is O(h^8), so halving h must gain a factor of 2**8 = 256. Bracket
    # it within a factor of two either side: anything outside that band means the scheme is
    # not achieving its nominal order and the error estimate in the report is not trustworthy.
    assert 128.0 < ratio < 512.0


@pytest.mark.unit
def test_extrapolation_is_refused():
    reference = _reference_ephemeris(duration_s=3600.0)
    with pytest.raises(InterpolationRangeError, match="extrapolation is refused"):
        interpolate_states(reference.times_s, reference.states, np.array([3660.0]))
    with pytest.raises(InterpolationRangeError, match="extrapolation is refused"):
        interpolate_states(reference.times_s, reference.states, np.array([-1.0]))


@pytest.mark.unit
def test_interpolation_rejects_an_invalid_window_size():
    reference = _reference_ephemeris(duration_s=3600.0)
    with pytest.raises(ValueError, match=r"points must be even and in \[2, 16\]"):
        interpolate_states(reference.times_s, reference.states, np.array([30.0]), points=7)
    with pytest.raises(ValueError, match=r"points must be even and in \[2, 16\]"):
        interpolate_states(reference.times_s, reference.states, np.array([30.0]), points=18)


@pytest.mark.unit
def test_interpolation_rejects_a_table_shorter_than_the_window():
    short = _reference_ephemeris(duration_s=180.0)
    with pytest.raises(ValueError, match="needs at least 8 samples"):
        interpolate_states(short.times_s, short.states, np.array([30.0]))


@pytest.mark.unit
def test_interpolation_rejects_non_finite_queries():
    reference = _reference_ephemeris(duration_s=3600.0)
    with pytest.raises(ValueError, match="query_s must be finite"):
        interpolate_states(reference.times_s, reference.states, np.array([math.nan]))


@pytest.mark.unit
def test_hermite_needs_two_samples():
    single = ephemeris_from_states(
        np.array([0.0]),
        STATE_CIRCULAR.reshape(1, 6),
        epoch=EPOCH,
        frame=ReferenceFrame.GCRF_APPROX,
    )
    with pytest.raises(ValueError, match="Hermite interpolation needs at least 2 samples"):
        interpolate_states(
            single.times_s,
            single.states,
            np.array([0.0]),
            method=InterpolationMethod.HERMITE,
        )


@pytest.mark.unit
def test_interpolation_error_estimate_rejects_a_short_ephemeris():
    short = _reference_ephemeris(duration_s=300.0)
    with pytest.raises(ValueError, match="needs at least 17 samples"):
        estimate_interpolation_error_m(short)


# ======================================================================================
# Alignment: the refusals
# ======================================================================================


@pytest.mark.unit
def test_mismatched_date_dependent_frame_always_raises():
    internal = _internal_ephemeris(duration_s=3600.0)
    external = _reference_ephemeris(duration_s=3600.0, frame=ReferenceFrame.ITRF)
    with pytest.raises(FrameMismatchError, match="at least one is date-dependent"):
        check_alignment(internal, external, allow_approximate_frame_tie=True)


@pytest.mark.unit
def test_mismatched_inertial_frames_raise_unless_explicitly_permitted():
    internal = _internal_ephemeris(duration_s=3600.0)
    external = _reference_ephemeris(duration_s=3600.0, frame=ReferenceFrame.EME2000)
    with pytest.raises(FrameMismatchError, match=r"Pass allow_approximate_frame_tie=True"):
        check_alignment(internal, external)

    # Complement: with the flag it succeeds, and the cost is on the record.
    report = check_alignment(internal, external, allow_approximate_frame_tie=True)
    assert report.frame_tie == "approximate_identity"
    assert report.frame_tie_error_m > 0.0


@pytest.mark.unit
def test_compare_refuses_a_frame_mismatch_it_was_not_told_to_accept():
    internal = _internal_ephemeris(duration_s=3600.0)
    external = _reference_ephemeris(duration_s=3600.0, frame=ReferenceFrame.EME2000)
    with pytest.raises(FrameMismatchError):
        compare_ephemerides(internal, external)


@pytest.mark.unit
def test_mismatched_time_scales_raise_without_a_supplied_offset():
    internal = _internal_ephemeris(duration_s=3600.0)
    utc_epoch = Epoch(EPOCH.mjd_day, EPOCH.seconds_of_day, TimeScale.UTC)
    external = _reference_ephemeris(duration_s=3600.0, epoch=utc_epoch)
    with pytest.raises(EpochMismatchError, match=r"time scales differ.*along-track"):
        check_alignment(internal, external)

    # Complement: supplied explicitly, it aligns and records what was applied.
    report = check_alignment(internal, external, time_scale_offset_s=37.0)
    assert report.time_scale_offset_s == pytest.approx(37.0)
    assert report.epoch_offset_s == pytest.approx(37.0)


@pytest.mark.unit
def test_tai_and_tt_align_automatically_with_the_exact_offset():
    internal = _internal_ephemeris(duration_s=7200.0)
    tt_epoch = EPOCH.in_scale(TimeScale.TT)
    external = _reference_ephemeris(duration_s=7200.0, epoch=tt_epoch)
    report = check_alignment(internal, external)
    assert report.time_scale is TimeScale.TAI
    assert report.time_scale_offset_s == pytest.approx(-32.184, abs=1e-12)
    assert report.epoch_offset_s == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_non_overlapping_arcs_raise():
    internal = _internal_ephemeris(duration_s=3600.0)
    external = _reference_ephemeris(duration_s=3600.0, epoch=EPOCH.shifted(2.0 * 86400.0))
    with pytest.raises(EpochMismatchError, match="the two arcs do not overlap"):
        check_alignment(internal, external)


@pytest.mark.unit
def test_align_ephemeris_shifts_times_onto_a_new_epoch():
    reference = _reference_ephemeris(duration_s=3600.0)
    later = EPOCH.shifted(600.0)
    shifted = align_ephemeris(reference, to_epoch=later, to_frame=ReferenceFrame.GCRF_APPROX)
    np.testing.assert_allclose(shifted.times_s, reference.times_s - 600.0, atol=1e-9)
    np.testing.assert_array_equal(shifted.states, reference.states)
    # Alignment must not launder provenance.
    assert shifted.provenance is reference.provenance


@pytest.mark.unit
def test_align_ephemeris_refuses_an_impossible_frame_change():
    reference = _reference_ephemeris(duration_s=3600.0)
    with pytest.raises(FrameMismatchError, match="no rotation from GCRF_APPROX to ITRF"):
        align_ephemeris(reference, to_epoch=EPOCH, to_frame=ReferenceFrame.ITRF)
    with pytest.raises(FrameMismatchError, match="allow_approximate_frame_tie=True"):
        align_ephemeris(reference, to_epoch=EPOCH, to_frame=ReferenceFrame.EME2000)


# ======================================================================================
# Comparison
# ======================================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_comparison_of_propagator_against_the_analytic_reference(capsys, tmp_path: Path):
    """The end-to-end harness run, on the only genuine oracle available here.

    Marked slow: measured 40 s, the largest single test in this file (the rest of the file
    is 87 s together). The propagator-versus-oracle *number* is not lost from the fast job --
    ``test_propagate_two_body_agrees_with_the_analytic_kepler_reference`` measures the same
    agreement in 11 s without the file round trip and the RIC decomposition.

    The analytic reference is tabulated at 30 s and the propagator output at 60 s, so the
    external side is interpolated with a measured error of ~1.5e-7 m -- three decades below
    the difference being reported. Exporting the reference at 60 s instead would put the
    interpolation error within a factor of two of the signal, which is why the GMAT runbook
    asks for a 30 s cadence.
    """
    internal = _internal_ephemeris(duration_s=86400.0, step_s=60.0)
    external_path = write_ephemeris(
        tmp_path / "kepler.eph", _reference_ephemeris(duration_s=86400.0, step_s=30.0)
    )
    external = read_ephemeris(external_path)

    report = compare_ephemerides(internal, external, label="two-body-vs-kepler-24h")
    with capsys.disabled():
        print(
            f"\n[measured] harness comparison over 24 h: "
            f"max |dr| = {report.position_max_m:.4e} m, RMS = {report.position_rms_m:.4e} m; "
            f"radial {report.radial_m.max_abs:.3e} m, "
            f"along-track {report.along_track_m.max_abs:.3e} m, "
            f"cross-track {report.cross_track_m.max_abs:.3e} m; "
            f"interpolation error {report.interpolation_error_m:.3e} m "
            f"(margin {report.interpolation_margin:.3g}x); "
            f"frame budget {report.frame_tie_error_m:.2f} m"
        )
    assert report.num_points == internal.size
    assert report.position_max_m < 1.0e-3
    assert report.interpolation_is_negligible
    assert report.interpolation_margin > 100.0
    # The difference is far below the frame-approximation budget, so no dynamics conclusion
    # may be drawn from it -- and the report says so rather than leaving it to the reader.
    assert report.difference_within_frame_budget
    assert report.external_provenance is Provenance.SYNTHETIC_REFERENCE
    assert report.is_external_tool_validated is False


@pytest.mark.unit
def test_a_ten_metre_along_track_bias_is_attributed_to_along_track():
    """The whole point of the RIC breakdown: a known bias lands on the right axis."""
    internal = _internal_ephemeris(duration_s=7200.0)
    biased = internal.states.copy()
    for index in range(internal.size):
        rotation, _ = hill_basis(internal.states[index, :3], internal.states[index, 3:])
        biased[index, :3] += rotation.T @ np.array([0.0, 10.0, 0.0])
    external = ephemeris_from_states(
        internal.times_s,
        biased,
        epoch=EPOCH,
        frame=ReferenceFrame.GCRF_APPROX,
        provenance=Provenance.SYNTHETIC_REFERENCE,
        source="internal + 10 m along-track",
    )

    report = compare_ephemerides(internal, external, label="along-track-bias")
    assert report.along_track_m.mean == pytest.approx(10.0, abs=1e-6)
    assert report.along_track_m.max_abs == pytest.approx(10.0, abs=1e-6)
    assert report.radial_m.max_abs < 1.0e-6
    assert report.cross_track_m.max_abs < 1.0e-6
    assert report.position_max_m == pytest.approx(10.0, abs=1e-6)


@pytest.mark.unit
def test_a_ten_metre_radial_bias_is_attributed_to_radial():
    """Complement to the along-track case: the axes are not interchangeable labels."""
    internal = _internal_ephemeris(duration_s=7200.0)
    biased = internal.states.copy()
    for index in range(internal.size):
        rotation, _ = hill_basis(internal.states[index, :3], internal.states[index, 3:])
        biased[index, :3] += rotation.T @ np.array([10.0, 0.0, 0.0])
    external = ephemeris_from_states(
        internal.times_s, biased, epoch=EPOCH, frame=ReferenceFrame.GCRF_APPROX
    )
    report = compare_ephemerides(internal, external)
    assert report.radial_m.mean == pytest.approx(10.0, abs=1e-6)
    assert report.along_track_m.max_abs < 1.0e-6
    assert report.cross_track_m.max_abs < 1.0e-6


@pytest.mark.unit
def test_a_ten_metre_cross_track_bias_is_attributed_to_cross_track():
    internal = _internal_ephemeris(duration_s=7200.0)
    biased = internal.states.copy()
    for index in range(internal.size):
        rotation, _ = hill_basis(internal.states[index, :3], internal.states[index, 3:])
        biased[index, :3] += rotation.T @ np.array([0.0, 0.0, 10.0])
    external = ephemeris_from_states(
        internal.times_s, biased, epoch=EPOCH, frame=ReferenceFrame.GCRF_APPROX
    )
    report = compare_ephemerides(internal, external)
    assert report.cross_track_m.mean == pytest.approx(10.0, abs=1e-6)
    assert report.radial_m.max_abs < 1.0e-6
    assert report.along_track_m.max_abs < 1.0e-6


@pytest.mark.unit
def test_a_secular_along_track_drift_shows_in_the_signed_mean_not_only_the_magnitude():
    """A total magnitude alone cannot tell drift from noise; the signed mean can."""
    internal = _internal_ephemeris(duration_s=7200.0)
    drifted = internal.states.copy()
    rate_m_s = 1.0e-3
    for index in range(internal.size):
        rotation, _ = hill_basis(internal.states[index, :3], internal.states[index, 3:])
        offset = rate_m_s * float(internal.times_s[index])
        drifted[index, :3] += rotation.T @ np.array([0.0, offset, 0.0])
    external = ephemeris_from_states(
        internal.times_s, drifted, epoch=EPOCH, frame=ReferenceFrame.GCRF_APPROX
    )
    report = compare_ephemerides(internal, external)
    expected_mean = rate_m_s * float(np.mean(internal.times_s))
    assert report.along_track_m.mean == pytest.approx(expected_mean, rel=1e-9)
    assert report.along_track_m.max_abs == pytest.approx(rate_m_s * 7200.0, rel=1e-9)
    # The bias is one-sided: mean and max are the same sign and the same order.
    assert report.along_track_m.mean > 0.4 * report.along_track_m.max_abs


@pytest.mark.unit
def test_an_epoch_offset_appears_as_along_track_at_the_predicted_size():
    """A one-second epoch error produces ~7.7 km along-track, exactly as advertised."""
    internal = _internal_ephemeris(duration_s=7200.0)
    # States that are truly at t + 1 s, tagged as if they were at t: a one-second mis-tag,
    # which is what a wrong epoch header or a TAI/UTC slip produces.
    mistagged = ephemeris_from_states(
        internal.times_s,
        kepler_propagate(STATE_CIRCULAR, internal.times_s + 1.0),
        epoch=EPOCH,
        frame=ReferenceFrame.GCRF_APPROX,
        provenance=Provenance.SYNTHETIC_REFERENCE,
        source="mis-tagged by 1 s",
    )
    report = compare_ephemerides(internal, mistagged, times_s=internal.times_s[10:-10])
    predicted = along_track_error_from_time_offset_m(1.0)
    assert report.along_track_m.max_abs == pytest.approx(predicted, rel=0.02)
    assert report.along_track_m.max_abs > 20.0 * report.radial_m.max_abs


@pytest.mark.unit
def test_rotating_frame_velocity_flag_changes_the_velocity_breakdown():
    """The transport term is real and worth ~n per metre; the flag is not cosmetic."""
    internal = _internal_ephemeris(duration_s=3600.0)
    biased = internal.states.copy()
    for index in range(internal.size):
        rotation, _ = hill_basis(internal.states[index, :3], internal.states[index, 3:])
        biased[index, :3] += rotation.T @ np.array([0.0, 100.0, 0.0])
    external = ephemeris_from_states(
        internal.times_s, biased, epoch=EPOCH, frame=ReferenceFrame.GCRF_APPROX
    )
    inertial = compare_ephemerides(internal, external)
    rotating = compare_ephemerides(internal, external, rotating_frame_velocity=True)
    n = math.sqrt(MU_EARTH_M3_S2 / A_M**3)
    assert inertial.velocity_max_m_s < 1.0e-9
    # omega x dr with |dr| = 100 m radial-perpendicular gives n * 100 m/s.
    assert rotating.velocity_max_m_s == pytest.approx(n * 100.0, rel=0.05)


@pytest.mark.unit
def test_report_flags_a_comparison_whose_interpolation_error_is_not_negligible():
    """A method whose error is comparable to the signal must not read as clean."""
    internal = _internal_ephemeris(duration_s=7200.0, step_s=60.0)
    external = _reference_ephemeris(duration_s=7200.0, step_s=60.0)
    grid = 0.5 * (internal.times_s[:-1] + internal.times_s[1:])
    honest = compare_ephemerides(internal, external, times_s=grid)
    sloppy = compare_ephemerides(
        internal, external, times_s=grid, method=InterpolationMethod.HERMITE
    )
    assert sloppy.interpolation_error_m > honest.interpolation_error_m * 1.0e3
    assert honest.interpolation_is_negligible is False or honest.interpolation_margin > 0.0
    # Cubic Hermite at 60 s is ~0.4 m, which swamps the millimetre-scale real difference,
    # so the report must refuse to call itself clean.
    assert sloppy.interpolation_is_negligible is False
    assert sloppy.interpolation_margin < 100.0


@pytest.mark.unit
def test_report_serialises_to_json_carrying_provenance_and_the_full_breakdown(tmp_path: Path):
    internal = _internal_ephemeris(duration_s=7200.0)
    external = _reference_ephemeris(duration_s=7200.0, provenance=Provenance.SYNTHETIC_REFERENCE)
    report = compare_ephemerides(internal, external, label="json-contract")
    path = write_comparison_report(tmp_path / "comparison.json", report)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "rpo-comparison/1.0"
    assert payload["label"] == "json-contract"
    assert payload["provenance"]["external_provenance"] == "synthetic_reference"
    assert payload["provenance"]["is_external_tool_validated"] is False
    assert payload["alignment"]["frame"] == "GCRF_APPROX"
    assert payload["alignment"]["time_scale"] == "TAI"
    for axis in ("radial", "along_track", "cross_track"):
        for statistic in ("max_abs", "rms", "mean"):
            assert isinstance(payload["position_m"][axis][statistic], float)
            assert isinstance(payload["velocity_m_s"][axis][statistic], float)
    assert payload["grid"]["interpolation_error_m"] >= 0.0
    assert isinstance(payload["interpretation"]["difference_within_frame_budget"], bool)


@pytest.mark.unit
def test_tool_run_provenance_is_the_only_thing_that_sets_the_validated_flag():
    internal = _internal_ephemeris(duration_s=3600.0)
    for provenance, expected in (
        (Provenance.TOOL_RUN, True),
        (Provenance.SYNTHETIC_REFERENCE, False),
        (Provenance.UNKNOWN, False),
    ):
        external = _reference_ephemeris(duration_s=3600.0, provenance=provenance)
        report = compare_ephemerides(internal, external)
        assert report.is_external_tool_validated is expected
        assert report.external_provenance is provenance
        assert json.loads(report.to_json())["provenance"]["external_provenance"] == (
            provenance.value
        )


@pytest.mark.unit
def test_report_json_refuses_non_finite_values():
    """``allow_nan=False``: a NaN must not be written as the non-standard ``NaN`` token."""
    internal = _internal_ephemeris(duration_s=3600.0)
    external = _reference_ephemeris(duration_s=3600.0)
    report = compare_ephemerides(internal, external)
    poisoned = ComparisonReport(
        **{
            **{
                field: getattr(report, field)
                for field in ComparisonReport.__dataclass_fields__
                if field != "position_max_m"
            },
            "position_max_m": math.nan,
        }
    )
    with pytest.raises(ValueError, match="Out of range float"):
        poisoned.to_json()


@pytest.mark.unit
def test_comparison_with_a_disjoint_explicit_grid_raises():
    internal = _internal_ephemeris(duration_s=3600.0)
    external = _reference_ephemeris(duration_s=3600.0)
    with pytest.raises(InterpolationRangeError, match="extrapolation is refused"):
        compare_ephemerides(internal, external, times_s=np.array([7200.0]))


@pytest.mark.unit
def test_comparison_grid_that_intersects_no_internal_sample_raises():
    internal = _internal_ephemeris(duration_s=3600.0, step_s=60.0)
    # A short external arc lying strictly between two internal samples: the arcs overlap, so
    # alignment succeeds, but no internal sample falls inside the overlap.
    external = _reference_ephemeris(duration_s=30.0, step_s=3.0, epoch=EPOCH.shifted(1210.0))
    with pytest.raises(ValueError, match="the comparison grid is empty"):
        compare_ephemerides(internal, external)
