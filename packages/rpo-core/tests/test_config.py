"""Scenario configuration, hashing, and run-directory provenance.

Two things distinguish these tests from schema tests. First, the derived orbital quantities
are checked against closed-form Kepler expressions, not against a stored number this code
produced earlier. Second, every cross-field validator is tested as a *knife edge*: the
rejection is paired with the nearest scenario that must still be accepted. A rejection test
on its own would pass just as happily if the validator rejected everything.
"""

import json
import math
import os
import re
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from rpo_core.config import (
    CW_ERROR_BUDGET_FRACTION_OF_KOZ,
    MIN_ALTITUDE_M,
    PERIOD_MULTIPLE_REL_TOL,
    ConstraintConfig,
    HoldPointConfig,
    IntegratorConfig,
    ManeuverConfig,
    OrbitConfig,
    ScenarioConfig,
    ScenarioConfigError,
    _git_provenance,
    config_hash,
    create_run_directory,
    load_scenario,
)
from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from rpo_core.relative.cw import DEFAULT_FEASIBILITY_TOL_M
from rpo_core.relative.nonlinear import CW_ERROR_COEFFICIENT, CW_ERROR_SAFETY_FACTOR

# packages/rpo-core/tests/test_config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = REPO_ROOT / "configs" / "vbar_baseline.yaml"


def _baseline_raw() -> dict[str, Any]:
    """Return the shipped baseline scenario as a plain dict."""
    loaded = yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _mutated(**overrides: Any) -> dict[str, Any]:
    """Return the baseline dict with ``a__b__c=value`` style overrides applied."""
    raw = _baseline_raw()
    for dotted, value in overrides.items():
        keys = dotted.split("__")
        node: Any = raw
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return raw


def _build(raw: dict[str, Any]) -> ScenarioConfig:
    """Validate a scenario dict with the CW-envelope warning muted.

    The warning has dedicated tests below. Muting it here stops an assertion about, say,
    transfer times from depending on whether the scenario also happens to sit inside the
    linearisation envelope.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ScenarioConfig.model_validate(raw)


BASELINE = _build(_baseline_raw())
PERIOD_S = BASELINE.orbit.orbital_period_s


# --- The shipped scenario -----------------------------------------------------------------


@pytest.mark.unit
def test_shipped_baseline_loads_and_validates():
    scenario = load_scenario(BASELINE_PATH)
    assert scenario.name == "vbar_baseline"
    assert scenario.seed == 42


@pytest.mark.unit
def test_shipped_baseline_is_the_documented_mvp_scenario():
    """The numbers in docs/cw_validity.md refer to this file; drift breaks that reference."""
    scenario = load_scenario(BASELINE_PATH)
    assert scenario.orbit.altitude_m == 420.0e3
    assert scenario.orbit.inclination_deg == pytest.approx(51.6)
    assert scenario.start_hold_point.position_hill_m == (0.0, -1000.0, 0.0)
    assert scenario.target_hold_point.position_hill_m == (0.0, -250.0, 0.0)
    assert scenario.constraints.keep_out_sphere_radius_m == 200.0
    assert scenario.constraints.max_closing_velocity_m_s == pytest.approx(0.1)
    assert scenario.tof_periods == pytest.approx(0.5)
    # Chaser trails the target: docs/conventions.md fixes that as negative y.
    assert scenario.start_hold_point.position_hill_m[1] < 0.0


@pytest.mark.unit
def test_missing_file_raises_with_the_path():
    with pytest.raises(ScenarioConfigError, match="cannot read scenario file"):
        load_scenario(REPO_ROOT / "configs" / "does_not_exist.yaml")


@pytest.mark.unit
def test_malformed_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match="not valid YAML"):
        load_scenario(bad)


@pytest.mark.unit
def test_yaml_that_is_not_a_mapping_raises(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match="mapping at the top level"):
        load_scenario(bad)


@pytest.mark.unit
def test_validation_failure_names_the_field_and_the_value(tmp_path):
    bad = tmp_path / "low.yaml"
    bad.write_text(yaml.safe_dump(_mutated(orbit__altitude_m=100.0e3)), encoding="utf-8")
    with pytest.raises(ScenarioConfigError) as excinfo:
        load_scenario(bad)
    message = str(excinfo.value)
    assert "orbit.altitude_m" in message
    assert "100,000" in message or "100000" in message


# --- extra="forbid" -----------------------------------------------------------------------


@pytest.mark.unit
def test_typo_at_the_top_level_is_rejected():
    raw = _baseline_raw()
    raw["sed"] = 42  # meant "seed"
    with pytest.raises(ValidationError, match="sed"):
        ScenarioConfig.model_validate(raw)


@pytest.mark.unit
def test_typo_in_a_nested_block_is_rejected():
    raw = _baseline_raw()
    raw["orbit"]["inclincation_deg"] = 51.6
    with pytest.raises(ValidationError, match="inclincation_deg"):
        ScenarioConfig.model_validate(raw)


@pytest.mark.unit
def test_the_same_scenario_without_the_typo_is_accepted():
    """Complement: proves the rejection above is about the unknown key, nothing else."""
    assert _build(_baseline_raw()).name == "vbar_baseline"


# --- Derived orbital quantities -----------------------------------------------------------


@pytest.mark.unit
def test_semi_major_axis_is_radius_plus_altitude():
    orbit = OrbitConfig(altitude_m=420.0e3, inclination_deg=51.6)
    assert orbit.semi_major_axis_m == R_EARTH_EQUATORIAL_M + 420.0e3


@pytest.mark.unit
@pytest.mark.parametrize("altitude_m", [150.0e3, 420.0e3, 800.0e3, 35_786.0e3])
def test_mean_motion_and_period_match_closed_form(altitude_m):
    orbit = OrbitConfig(altitude_m=altitude_m, inclination_deg=0.0)
    a = R_EARTH_EQUATORIAL_M + altitude_m
    assert orbit.mean_motion_rad_s == pytest.approx(math.sqrt(MU_EARTH_M3_S2 / a**3), rel=1e-15)
    assert orbit.orbital_period_s == pytest.approx(
        2.0 * math.pi / orbit.mean_motion_rad_s, rel=1e-15
    )
    # n*T = 2*pi is the definition; it catches a swapped or stale property.
    assert orbit.mean_motion_rad_s * orbit.orbital_period_s == pytest.approx(
        2.0 * math.pi, rel=1e-14
    )


@pytest.mark.unit
def test_geostationary_altitude_gives_a_sidereal_day():
    """A limiting case with an independently known answer: 35 786 km -> 23 h 56 min."""
    orbit = OrbitConfig(altitude_m=35_786.0e3, inclination_deg=0.0)
    assert orbit.orbital_period_s == pytest.approx(86_164.0, rel=2e-4)


@pytest.mark.unit
def test_period_grows_with_altitude():
    """Complement to the closed-form check: the period must actually depend on altitude."""
    low = OrbitConfig(altitude_m=420.0e3, inclination_deg=0.0)
    high = OrbitConfig(altitude_m=840.0e3, inclination_deg=0.0)
    assert high.orbital_period_s > low.orbital_period_s * 1.05


@pytest.mark.unit
def test_hold_point_radius_is_the_euclidean_norm():
    point = HoldPointConfig(name="p", position_hill_m=(3.0, 4.0, 12.0))
    assert point.radius_m == pytest.approx(13.0, rel=1e-15)


# --- Field-level validation ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("altitude_m", [149_999.0, 100.0e3, 0.0, -1.0])
def test_reentry_altitude_is_rejected(altitude_m):
    """Literal altitudes, not MIN_ALTITUDE_M.

    A test written against the constant it is supposed to pin would keep passing if
    somebody moved the floor to 300 km.
    """
    with pytest.raises(ValidationError, match="reentry"):
        OrbitConfig(altitude_m=altitude_m, inclination_deg=51.6)


@pytest.mark.unit
@pytest.mark.parametrize("altitude_m", [150_000.0, 150_001.0, 420.0e3])
def test_the_altitude_floor_itself_is_accepted(altitude_m):
    """Knife edge: 149 999 m fails, 150 000 m does not."""
    assert OrbitConfig(altitude_m=altitude_m, inclination_deg=51.6).altitude_m == altitude_m
    assert MIN_ALTITUDE_M == 150.0e3, "the floor these literals encode has moved"


@pytest.mark.unit
@pytest.mark.parametrize("inclination_deg", [-0.001, 180.001, 360.0])
def test_out_of_range_inclination_is_rejected(inclination_deg):
    with pytest.raises(ValidationError, match=r"inclination_deg"):
        OrbitConfig(altitude_m=420.0e3, inclination_deg=inclination_deg)


@pytest.mark.unit
@pytest.mark.parametrize("inclination_deg", [0.0, 90.0, 180.0])
def test_inclination_endpoints_are_accepted(inclination_deg):
    assert OrbitConfig(altitude_m=420.0e3, inclination_deg=inclination_deg).inclination_deg == (
        inclination_deg
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_altitude_is_rejected(value):
    """NaN passes every ordering comparison silently; it has to die at parse time."""
    with pytest.raises(ValidationError):
        OrbitConfig(altitude_m=value, inclination_deg=51.6)


@pytest.mark.unit
def test_integrator_defaults_are_the_documented_reference_tolerances():
    integrator = IntegratorConfig()
    assert integrator.method == "DOP853"
    assert integrator.rtol == 1.0e-12
    assert integrator.atol == 1.0e-12


@pytest.mark.unit
def test_unknown_integrator_method_is_rejected_naming_the_value():
    with pytest.raises(ValidationError, match="Euler"):
        IntegratorConfig(method="Euler")


@pytest.mark.unit
@pytest.mark.parametrize("tolerance", [0.0, -1e-12])
def test_non_positive_integrator_tolerance_is_rejected(tolerance):
    with pytest.raises(ValidationError, match="rtol"):
        IntegratorConfig(rtol=tolerance)


def _constraints(**overrides: Any) -> dict[str, Any]:
    """Return a valid constraints block with overrides applied."""
    block: dict[str, Any] = {
        "keep_out_sphere_radius_m": 200.0,
        "approach_ellipsoid_semi_axes_m": [2000.0, 4000.0, 2000.0],
        "approach_cone_half_angle_deg": 10.0,
        "approach_cone_activation_range_m": 1000.0,
        "max_closing_velocity_m_s": 0.1,
        "max_closing_velocity_activation_range_m": 250.0,
    }
    block.update(overrides)
    return block


@pytest.mark.unit
def test_non_positive_keep_out_radius_is_rejected():
    with pytest.raises(ValidationError, match="keep_out_sphere_radius_m"):
        ConstraintConfig.model_validate(_constraints(keep_out_sphere_radius_m=0.0))


@pytest.mark.unit
def test_degenerate_approach_ellipsoid_is_rejected():
    with pytest.raises(ValidationError, match="strictly positive"):
        ConstraintConfig.model_validate(
            _constraints(approach_ellipsoid_semi_axes_m=[2000.0, 0.0, 2000.0])
        )


@pytest.mark.unit
def test_approach_ellipsoid_inside_the_keep_out_sphere_is_rejected():
    with pytest.raises(ValidationError, match="must enclose the keep-out sphere"):
        ConstraintConfig.model_validate(
            _constraints(
                keep_out_sphere_radius_m=200.0,
                approach_ellipsoid_semi_axes_m=[199.999, 4000.0, 2000.0],
            )
        )


@pytest.mark.unit
def test_approach_ellipsoid_exactly_touching_the_keep_out_sphere_is_accepted():
    """Knife edge on the containment check: equality is contact, not intersection."""
    constraints = ConstraintConfig.model_validate(
        _constraints(
            keep_out_sphere_radius_m=200.0,
            approach_ellipsoid_semi_axes_m=[200.0, 4000.0, 2000.0],
        )
    )
    assert min(constraints.approach_ellipsoid_semi_axes_m) == 200.0


@pytest.mark.unit
@pytest.mark.parametrize("half_angle_deg", [0.0, 90.0, 120.0])
def test_out_of_range_cone_half_angle_is_rejected(half_angle_deg):
    with pytest.raises(ValidationError, match="approach_cone_half_angle_deg"):
        ConstraintConfig.model_validate(_constraints(approach_cone_half_angle_deg=half_angle_deg))


# --- Time of flight, either spelling ------------------------------------------------------


@pytest.mark.unit
def test_time_of_flight_in_seconds_and_in_periods_resolve_identically():
    seconds = _build(_mutated(maneuver={"tof_s": 0.4 * PERIOD_S}))
    periods = _build(_mutated(maneuver={"tof_periods": 0.4}))
    assert seconds.tof_s == pytest.approx(periods.tof_s, rel=1e-15)


@pytest.mark.unit
def test_specifying_both_spellings_of_time_of_flight_is_rejected():
    with pytest.raises(ValidationError, match="two spellings"):
        ScenarioConfig.model_validate(_mutated(maneuver={"tof_s": 2789.0, "tof_periods": 0.5}))


@pytest.mark.unit
def test_specifying_neither_spelling_of_time_of_flight_is_rejected():
    with pytest.raises(ValidationError, match="neither tof_s nor tof_periods"):
        ScenarioConfig.model_validate(_mutated(maneuver={}))


@pytest.mark.unit
def test_resolve_tof_s_rejects_a_non_positive_period():
    maneuver = ManeuverConfig(tof_periods=0.5)
    with pytest.raises(ScenarioConfigError, match="finite positive period"):
        maneuver.resolve_tof_s(0.0)


# --- Cross-field validator 1: hold points versus the keep-out sphere -----------------------


@pytest.mark.unit
@pytest.mark.parametrize("field", ["start_hold_point", "target_hold_point"])
def test_hold_point_inside_the_keep_out_sphere_is_rejected(field):
    inside = {"name": "inside_koz", "position_hill_m": [0.0, -199.0, 0.0]}
    with pytest.raises(ValidationError, match="inside the keep_out_sphere"):
        ScenarioConfig.model_validate(_mutated(**{field: inside}))


@pytest.mark.unit
def test_hold_point_just_outside_the_keep_out_sphere_is_accepted():
    """Knife edge: 199 m fails, 201 m passes. The check is a radius, not a blanket ban."""
    outside = {"name": "just_outside_koz", "position_hill_m": [0.0, -201.0, 0.0]}
    scenario = _build(_mutated(target_hold_point=outside))
    assert scenario.target_hold_point.radius_m == pytest.approx(201.0)


@pytest.mark.unit
def test_the_keep_out_check_uses_the_radius_not_a_single_axis():
    """A point at (150, -150, 0) is 212 m out: inside on every axis, outside on radius."""
    diagonal = {"name": "diagonal", "position_hill_m": [150.0, -150.0, 0.0]}
    scenario = _build(_mutated(target_hold_point=diagonal))
    assert scenario.target_hold_point.radius_m == pytest.approx(150.0 * math.sqrt(2.0))


# --- Cross-field validator 2: integer multiples of the orbital period ----------------------


@pytest.mark.unit
@pytest.mark.parametrize("tof_periods", [1.0, 2.0, 3.0])
def test_whole_period_time_of_flight_is_rejected(tof_periods):
    with pytest.raises(ValidationError, match="in-plane two-impulse"):
        ScenarioConfig.model_validate(_mutated(maneuver={"tof_periods": tof_periods}))


@pytest.mark.unit
def test_whole_period_rejection_names_the_singularity_and_the_numbers():
    with pytest.raises(ValidationError) as excinfo:
        ScenarioConfig.model_validate(_mutated(maneuver={"tof_periods": 1.0}))
    message = str(excinfo.value)
    assert "vanishes at integer multiples of the period" in message
    assert "1.000000000 orbital periods" in message


@pytest.mark.unit
def test_time_of_flight_just_off_a_whole_period_is_accepted():
    """Knife edge: 1.0 periods is singular, 1.0001 is a perfectly ordinary transfer."""
    scenario = _build(_mutated(maneuver={"tof_periods": 1.0 + 1.0e-4}))
    assert scenario.tof_periods == pytest.approx(1.0 + 1.0e-4)


@pytest.mark.unit
def test_the_period_multiple_band_has_the_documented_width():
    """Two-sided, and written in literals rather than in PERIOD_MULTIPLE_REL_TOL.

    Expressing the band edges as multiples of the constant would make this test move
    whenever the constant moved -- it would pass at a band of 1e-12 or 1e-2 alike, which
    is no test at all. 1e-7 must reject and 1e-5 must accept, pinning the width to the
    documented 1e-6 from both sides. That width is itself anchored: relative/cw.py
    measures the solve losing rank within ~3e-8 of an exact multiple.
    """
    with pytest.raises(ValidationError, match="in-plane two-impulse"):
        ScenarioConfig.model_validate(_mutated(maneuver={"tof_periods": 1.0 + 1.0e-7}))
    assert _build(_mutated(maneuver={"tof_periods": 1.0 + 1.0e-5})).tof_periods > 1.0
    assert PERIOD_MULTIPLE_REL_TOL == 1.0e-6, "the band these literals bracket has moved"


@pytest.mark.unit
def test_half_period_transfer_without_cross_track_motion_is_accepted():
    """The baseline. A single 3x3 conditioning check would wrongly reject this."""
    assert BASELINE.tof_periods == pytest.approx(0.5)


# --- Cross-field validator 3: half-period multiples with cross-track motion ----------------


@pytest.mark.unit
@pytest.mark.parametrize("tof_periods", [0.5, 1.5, 2.5])
def test_cross_track_change_at_a_half_period_multiple_is_rejected(tof_periods):
    moved = {"name": "off_plane", "position_hill_m": [0.0, -250.0, 40.0]}
    with pytest.raises(ValidationError, match="rank-deficient"):
        ScenarioConfig.model_validate(
            _mutated(target_hold_point=moved, maneuver={"tof_periods": tof_periods})
        )


@pytest.mark.unit
def test_cross_track_change_away_from_a_half_period_multiple_is_accepted():
    """Knife edge: the same cross-track move at 0.4 periods is well posed."""
    moved = {"name": "off_plane", "position_hill_m": [0.0, -250.0, 40.0]}
    scenario = _build(_mutated(target_hold_point=moved, maneuver={"tof_periods": 0.4}))
    assert scenario.target_hold_point.position_hill_m[2] == 40.0


@pytest.mark.unit
def test_half_period_transfer_is_accepted_when_cross_track_position_does_not_change():
    """Complement in the other direction: it is the z motion, not the timing, that fails."""
    both_off_plane = 40.0
    scenario = _build(
        _mutated(
            start_hold_point={"name": "a", "position_hill_m": [0.0, -1000.0, both_off_plane]},
            target_hold_point={"name": "b", "position_hill_m": [0.0, -250.0, both_off_plane]},
        )
    )
    assert scenario.tof_periods == pytest.approx(0.5)


@pytest.mark.unit
def test_cross_track_change_below_the_cw_feasibility_tolerance_is_accepted():
    """CW itself calls a sub-tolerance z request already satisfied; so does this check."""
    negligible = {"name": "b", "position_hill_m": [0.0, -250.0, 0.1 * DEFAULT_FEASIBILITY_TOL_M]}
    scenario = _build(_mutated(target_hold_point=negligible))
    assert scenario.tof_periods == pytest.approx(0.5)


@pytest.mark.unit
def test_cross_track_change_above_the_cw_feasibility_tolerance_is_rejected():
    """Knife edge on that tolerance: ten times the threshold is a real cross-track request."""
    real = {"name": "b", "position_hill_m": [0.0, -250.0, 10.0 * DEFAULT_FEASIBILITY_TOL_M]}
    with pytest.raises(ValidationError, match="rank-deficient"):
        ScenarioConfig.model_validate(_mutated(target_hold_point=real))


# --- Cross-field validator 4: the measured CW validity envelope ----------------------------


def _warning_separation_threshold_m(scenario: ScenarioConfig) -> float:
    """Return the separation at which the estimated CW error exactly meets the budget.

    Inverts the CONSERVATIVE bound the config actually guards on --
    err_bound = safety_factor * 6*pi * rho**2 / r * n_orbits -- not the central estimate.
    Inverting the wrong one would put the knife edge in the wrong place and the test would
    pass while measuring nothing. Anchored to docs/cw_validity.md, not a hand-picked number.
    """
    tolerance_m = CW_ERROR_BUDGET_FRACTION_OF_KOZ * scenario.constraints.keep_out_sphere_radius_m
    return math.sqrt(
        tolerance_m
        * scenario.orbit.semi_major_axis_m
        / (CW_ERROR_SAFETY_FACTOR * CW_ERROR_COEFFICIENT * scenario.tof_periods)
    )


@pytest.mark.unit
def test_the_baseline_scenario_does_not_warn():
    """docs/cw_validity.md puts the MVP hop at ~1.5 m against a 2 m budget: inside, barely."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ScenarioConfig.model_validate(_baseline_raw())


@pytest.mark.unit
def test_a_far_range_scenario_warns_and_names_both_numbers():
    far = {"name": "far", "position_hill_m": [0.0, -10_000.0, 0.0]}
    with pytest.warns(UserWarning, match="Clohessy-Wiltshire validity envelope") as record:
        ScenarioConfig.model_validate(_mutated(start_hold_point=far))
    message = str(record[0].message)
    # Both the bound and the budget it breached must appear, per docs/cw_validity.md
    # far-range row. Literals, not expressions over the constants: a test written in terms
    # of the constant it checks shrinks along with that constant and stays green.
    # 10 km over half an orbit: 1.5 * 6*pi * 1e8 / 6798137 * 0.5 = 208.0 m, against a
    # budget of 2.5% * 200 m = 5.0 m.
    assert "10,000.0 m separation" in message
    assert "5.000 m budget" in message
    assert re.search(r"error is 2[0-9]{2}\.\d+ m", message), message


@pytest.mark.unit
def test_the_budget_and_safety_factor_still_equal_what_the_literals_encode():
    """Companion to the literal assertions above — catches a constant drifting silently."""
    assert CW_ERROR_BUDGET_FRACTION_OF_KOZ == 0.025
    assert CW_ERROR_SAFETY_FACTOR == 1.5


@pytest.mark.unit
def test_the_warning_threshold_is_the_measured_error_law():
    """Knife edge: 1 % either side of the computed threshold flips the warning."""
    threshold_m = _warning_separation_threshold_m(BASELINE)
    # Sanity: the threshold must sit between the baseline separation and far range.
    assert 1000.0 < threshold_m < 10_000.0

    quiet = {"name": "quiet", "position_hill_m": [0.0, -0.99 * threshold_m, 0.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ScenarioConfig.model_validate(_mutated(start_hold_point=quiet))

    loud = {"name": "loud", "position_hill_m": [0.0, -1.01 * threshold_m, 0.0]}
    with pytest.warns(UserWarning, match="validity envelope"):
        ScenarioConfig.model_validate(_mutated(start_hold_point=loud))


@pytest.mark.unit
def test_the_envelope_warning_does_not_reject():
    """It is a modelling judgement, not a malformed input. The object must still be built."""
    far = {"name": "far", "position_hill_m": [0.0, -10_000.0, 0.0]}
    with pytest.warns(UserWarning):
        scenario = ScenarioConfig.model_validate(_mutated(start_hold_point=far))
    assert scenario.max_separation_m == pytest.approx(10_000.0)


# --- Hashing ------------------------------------------------------------------------------


@pytest.mark.unit
def test_identical_scenarios_hash_identically():
    assert config_hash(_build(_baseline_raw())) == config_hash(_build(_baseline_raw()))


@pytest.mark.unit
def test_the_hash_is_a_short_lowercase_hex_digest():
    assert re.fullmatch(r"[0-9a-f]{12}", config_hash(BASELINE))


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": 43},
        {"name": "vbar_baseline_2"},
        {"description": "a different description"},
        {"orbit__altitude_m": 420_001.0},
        {"orbit__inclination_deg": 51.7},
        {"constraints__keep_out_sphere_radius_m": 201.0},
        {"integrator__rtol": 1.0e-10},
        {"integrator__method": "Radau"},
        {"maneuver": {"tof_periods": 0.4}},
        {"start_hold_point": {"name": "other", "position_hill_m": [0.0, -1000.0, 0.0]}},
        {"target_hold_point": {"name": "vbar_minus_250", "position_hill_m": [0.0, -251.0, 0.0]}},
    ],
)
def test_changing_any_field_changes_the_hash(overrides):
    """Every field is part of the identity; a field the hash ignores is a silent collision."""
    assert config_hash(_build(_mutated(**overrides))) != config_hash(BASELINE)


@pytest.mark.unit
def test_the_hash_survives_a_yaml_round_trip(tmp_path):
    dumped = tmp_path / "round_trip.yaml"
    dumped.write_text(yaml.safe_dump(BASELINE.model_dump(mode="json")), encoding="utf-8")
    assert config_hash(load_scenario(dumped)) == config_hash(BASELINE)


@pytest.mark.unit
def test_the_hash_survives_a_json_round_trip():
    revalidated = ScenarioConfig.model_validate(json.loads(BASELINE.model_dump_json()))
    assert config_hash(revalidated) == config_hash(BASELINE)


@pytest.mark.unit
def test_the_hash_ignores_the_order_keys_were_written_in():
    reversed_keys = dict(reversed(list(_baseline_raw().items())))
    assert config_hash(_build(reversed_keys)) == config_hash(BASELINE)


@pytest.mark.unit
def test_hash_length_is_configurable_and_a_prefix_of_the_full_digest():
    full = config_hash(BASELINE, length=64)
    assert full.startswith(config_hash(BASELINE))
    with pytest.raises(ScenarioConfigError, match=r"\[1, 64\]"):
        config_hash(BASELINE, length=65)


@pytest.mark.integration
@pytest.mark.parametrize("hash_seed", ["0", "1", "424242"])
def test_the_hash_is_identical_in_a_fresh_process_under_a_different_hash_seed(hash_seed):
    """The reason config_hash may not use built-in hash(): that one is salted per process."""
    script = (
        "from rpo_core.config import config_hash, load_scenario;"
        f"print(config_hash(load_scenario({str(BASELINE_PATH)!r})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == config_hash(BASELINE)


# --- Run directories and provenance -------------------------------------------------------

EXPECTED_PROVENANCE_KEYS = {
    "config",
    "config_hash",
    "created_utc",
    "git_commit",
    "git_dirty",
    "package_versions",
    "python_version",
    "seed",
}


def _provenance(run_dir: Path) -> dict[str, Any]:
    """Return the provenance record written into ``run_dir``."""
    loaded = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.unit
def test_run_directory_is_named_by_hash_and_seed(tmp_path):
    run_dir = create_run_directory(BASELINE, 7, tmp_path)
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert run_dir.name == f"{config_hash(BASELINE)}-7"


@pytest.mark.unit
def test_run_directory_defaults_the_seed_to_the_scenario_seed(tmp_path):
    assert create_run_directory(BASELINE, base_dir=tmp_path).name.endswith("-42")


@pytest.mark.unit
def test_different_seeds_give_different_run_directories(tmp_path):
    first = create_run_directory(BASELINE, 1, tmp_path)
    second = create_run_directory(BASELINE, 2, tmp_path)
    assert first != second
    assert first.name.split("-")[0] == second.name.split("-")[0]


@pytest.mark.unit
def test_a_negative_seed_is_rejected(tmp_path):
    with pytest.raises(ScenarioConfigError, match="non-negative"):
        create_run_directory(BASELINE, -1, tmp_path)


@pytest.mark.unit
def test_provenance_carries_everything_needed_to_rebuild_the_run(tmp_path):
    record = _provenance(create_run_directory(BASELINE, 42, tmp_path))
    assert set(record) == EXPECTED_PROVENANCE_KEYS
    assert record["seed"] == 42
    assert record["config_hash"] == config_hash(BASELINE)
    assert record["python_version"] == ".".join(str(part) for part in sys.version_info[:3])
    for package in ("numpy", "scipy", "pydantic", "pyyaml", "rpo-core"):
        assert record["package_versions"][package] != "unknown", package


@pytest.mark.unit
def test_the_serialised_config_in_provenance_reproduces_the_scenario(tmp_path):
    """The provenance must be sufficient on its own, not a summary of the config."""
    record = _provenance(create_run_directory(BASELINE, 42, tmp_path))
    rebuilt = _build(record["config"])
    assert rebuilt == BASELINE
    assert config_hash(rebuilt) == record["config_hash"]


@pytest.mark.unit
def test_provenance_timestamp_is_utc_iso_8601(tmp_path):
    record = _provenance(create_run_directory(BASELINE, 42, tmp_path))
    stamp = datetime.fromisoformat(record["created_utc"])
    assert stamp.tzinfo is not None
    assert stamp.utcoffset().total_seconds() == 0.0
    assert abs((datetime.now(UTC) - stamp).total_seconds()) < 300.0


@pytest.mark.unit
def test_the_timestamp_does_not_feed_the_hash(tmp_path):
    """Two runs of one scenario share a directory: only the provenance record is refreshed."""
    first = create_run_directory(BASELINE, 42, tmp_path)
    first_record = _provenance(first)
    second = create_run_directory(BASELINE, 42, tmp_path)
    second_record = _provenance(second)

    assert second == first
    assert list(tmp_path.iterdir()) == [first]
    assert second_record["config_hash"] == first_record["config_hash"] == config_hash(BASELINE)
    # The timestamp is recorded, and the hash is unmoved by it in either direction.
    assert second_record["created_utc"] >= first_record["created_utc"]
    hashed_payload = json.dumps(first_record["config"], sort_keys=True)
    assert "created_utc" not in hashed_payload


@pytest.mark.unit
def test_re_running_leaves_existing_result_files_alone(tmp_path):
    run_dir = create_run_directory(BASELINE, 42, tmp_path)
    (run_dir / "trajectory.npz").write_bytes(b"results")
    create_run_directory(BASELINE, 42, tmp_path)
    assert (run_dir / "trajectory.npz").read_bytes() == b"results"


@pytest.mark.unit
def test_git_commit_falls_back_to_uncommitted_in_a_repository_with_no_commits(tmp_path):
    repo = tmp_path / "fresh"
    repo.mkdir()
    if subprocess.run(["git", "init", "-q", str(repo)], check=False).returncode != 0:
        pytest.skip("git is not available")
    commit, dirty = _git_provenance(repo)
    assert commit == "uncommitted"
    assert dirty is False


@pytest.mark.unit
def test_git_commit_is_unknown_outside_a_repository(tmp_path):
    """Distinct from "uncommitted": we did not look at a repository at all."""
    outside = tmp_path / "plain"
    outside.mkdir()
    commit, dirty = _git_provenance(outside)
    assert (commit, dirty) == ("unknown", None)


@pytest.mark.unit
def test_recorded_git_commit_is_a_sha_or_a_documented_fallback(tmp_path):
    record = _provenance(create_run_directory(BASELINE, 42, tmp_path))
    assert re.fullmatch(r"[0-9a-f]{40}|uncommitted|unknown", record["git_commit"])
    assert record["git_dirty"] in (True, False, None)
