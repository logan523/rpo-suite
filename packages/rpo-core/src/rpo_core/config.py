r"""Scenario configuration, content-addressed hashing, and reproducible run directories.

A result that cannot be regenerated is an anecdote. This module is the boundary where a
hand-written YAML file becomes a validated, frozen object, and where every number needed
to rebuild a run is written down next to that run's output.

The equations
-------------
Derived quantities for a circular target orbit at altitude :math:`h` above the WGS-84
equatorial radius :math:`R_\oplus`:

.. math::

    a = R_\oplus + h, \qquad n = \sqrt{\mu / a^3}, \qquad T = 2\pi / n

Time of flight may be written as seconds or as a fraction of :math:`T`. The two forms
resolve to the same :math:`t_f` and therefore to the same trajectory; only one may be
given, because two disagreeing spellings of the same quantity is a bug waiting to happen.

Four cross-field validators encode facts that live in the dynamics rather than in the
schema. Three are exact, one is measured:

1. A hold point inside the keep-out sphere is a scenario-definition error. There is no
   trajectory to plan from inside the volume the trajectory exists to avoid.

2. The in-plane two-impulse Clohessy-Wiltshire solve inverts the position-from-velocity
   block, whose in-plane 2x2 determinant is

   .. math:: \det \Phi_{rv}^{xy} = \frac{8 - 8\cos\tau - 3\tau\sin\tau}{n^2},
             \qquad \tau = n t_f,

   which vanishes at :math:`\tau = 2\pi k` -- integer multiples of the orbital period.
   See :class:`rpo_core.exceptions.SingularTransferTimeError`.

3. The cross-track term of the same block is :math:`\sin\tau / n`, which vanishes at
   :math:`\tau = \pi k` -- integer multiples of the *half* period. At those times
   :math:`z(t_f)` is pinned to :math:`z_0 \cos\tau` regardless of the impulse applied, so
   a manoeuvre that must change cross-track position is unsolvable while one that does not
   -- the half-period V-bar hop, the baseline scenario in ``configs/`` -- is perfectly
   well posed. This is why the check is conditional on the manoeuvre actually moving in z.

4. CW linearisation error is measured, not assumed, and obeys

   .. math:: \varepsilon_{1\ \mathrm{orbit}} = 6\pi \rho^2 / r

   (``docs/cw_validity.md``). A scenario whose estimated error eats a meaningful fraction
   of the keep-out sphere gets a warning, not a rejection: CW being imprecise is a
   modelling judgement for the user to make, not a malformed input.

Validity
--------
The orbit model here is *circular and Earth-centred*: altitude is measured from the WGS-84
equatorial radius and the semi-major axis is the orbit radius. Eccentricity, a different
central body, and orbital elements beyond altitude/inclination are out of scope and are
rejected as unknown keys rather than silently ignored.

The CW envelope check estimates the scenario separation as the larger of the two hold-point
radii. A transfer arc bulges away from the chord joining its endpoints, so this slightly
*under*-estimates the true maximum separation; the linear-in-time scaling inside
:func:`~rpo_core.relative.nonlinear.estimated_cw_error_m` over-estimates. Neither is a
correction to apply -- it is an order-of-magnitude budget for deciding whether CW is the
right model at all.

Units are SI (metres, seconds, radians) with one deliberate exception: angles are degrees
in the configuration files, because this is the I/O boundary and the conversion belongs
here rather than in the numerics. Field names carry the unit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .constants import R_EARTH_EQUATORIAL_M
from .constants import mean_motion_rad_s as _mean_motion_rad_s
from .constants import orbital_period_s as _orbital_period_s
from .exceptions import RpoCoreError
from .relative.cw import DEFAULT_FEASIBILITY_TOL_M
from .relative.nonlinear import conservative_cw_error_bound_m

__all__ = [
    "CONFIG_HASH_LENGTH",
    "CW_ERROR_BUDGET_FRACTION_OF_KOZ",
    "DEFAULT_RUNS_DIR",
    "MIN_ALTITUDE_M",
    "PERIOD_MULTIPLE_REL_TOL",
    "ConstraintConfig",
    "HoldPointConfig",
    "IntegratorConfig",
    "ManeuverConfig",
    "OrbitConfig",
    "ScenarioConfig",
    "ScenarioConfigError",
    "config_hash",
    "create_run_directory",
    "load_scenario",
]


class ScenarioConfigError(RpoCoreError, ValueError):
    """Raised when a scenario is malformed, unreadable, or physically self-contradictory.

    Subclasses :class:`ValueError` so that it can be raised from inside a Pydantic
    validator: Pydantic collects ``ValueError`` subclasses into a
    :class:`pydantic.ValidationError` and preserves the message, so a direct
    ``ScenarioConfig(...)`` call surfaces a ``ValidationError`` carrying this text while
    :func:`load_scenario` re-raises the whole thing as this type.
    """


#: Lowest altitude (m) a scenario may specify.
#:
#: Below roughly 150 km, drag dominates and the orbit is not maintainable: a "circular
#: orbit" there is a reentry trajectory, and every CW quantity derived from it is
#: meaningless. The floor is a modelling boundary, not a safety margin.
MIN_ALTITUDE_M: float = 150.0e3

#: Relative half-width of the band around ``k*T`` (and ``k*T/2``) that is rejected.
#:
#: Chosen from the measured behaviour of the solve rather than from feel. The in-plane
#: condition number grows only as ``~3 / (1 - t/T)`` (see
#: ``SINGULARITY_CONDITION_LIMIT`` in :mod:`rpo_core.relative.cw`), so the solver's own
#: backstop trips within ``~3e-8`` of an exact multiple. Rejecting at ``1e-6`` gives about
#: 30x headroom above the measured failure band while costing nothing real -- no scenario
#: needs a time of flight pinned to within a microsecond-per-second of a period multiple,
#: and a value that close is almost always a user who meant to type the multiple exactly.
PERIOD_MULTIPLE_REL_TOL: float = 1.0e-6

#: Fraction of the keep-out-sphere radius allowed as CW linearisation error before warning.
#:
#: ``docs/cw_validity.md`` works its error budget at 2.5 % of the 200 m keep-out sphere,
#: i.e. 5 m. The MVP V-bar hop measures 1.455 m of linearisation error with a conservative
#: bound of 2.08 m, so it sits at 42 % of budget. An earlier 1 % / 2 m budget was rejected:
#: the conservative bound exceeded it and the flagship scenario warned about itself, while
#: its measured error was well inside. The budget therefore
#: sits deliberately close to the baseline: it is meant to fire on the next scenario out,
#: not to be comfortably slack.
CW_ERROR_BUDGET_FRACTION_OF_KOZ: float = 0.025

#: Number of hex characters retained from the SHA-256 digest in :func:`config_hash`.
#:
#: 12 hex characters is 48 bits. At the scale of a study -- thousands of scenarios, not
#: billions -- collision probability is negligible, and the digest still fits in a
#: directory name a human can read out loud.
CONFIG_HASH_LENGTH: int = 12

#: Default parent directory for run directories, relative to the current working directory.
DEFAULT_RUNS_DIR: Path = Path("results/runs")

# frozen: a scenario that mutates after validation has escaped every check below, and its
# hash would no longer describe it. extra="forbid": an unknown key in a scenario file is a
# typo the user needs told about, not a field to ignore. allow_inf_nan=False: NaN passes
# every ordering comparison silently, so it must be rejected at parse time or not at all.
_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class OrbitConfig(BaseModel):
    """Circular target orbit, specified by altitude above the WGS-84 equatorial radius."""

    model_config = _MODEL_CONFIG

    altitude_m: float = Field(description="Altitude above the WGS-84 equatorial radius, metres.")
    inclination_deg: float = Field(description="Orbit inclination, degrees, in [0, 180].")

    @field_validator("altitude_m")
    @classmethod
    def _reject_reentry_altitudes(cls, value: float) -> float:
        """Reject altitudes in the uncontrolled-reentry regime."""
        if value < MIN_ALTITUDE_M:
            raise ScenarioConfigError(
                f"altitude_m={value:,.1f} m is below the {MIN_ALTITUDE_M:,.0f} m floor. "
                "Below that altitude drag dominates and the orbit is a reentry "
                "trajectory, not a circular orbit CW can be linearised about."
            )
        return value

    @field_validator("inclination_deg")
    @classmethod
    def _reject_out_of_range_inclination(cls, value: float) -> float:
        """Reject inclinations outside the physically defined range."""
        if not 0.0 <= value <= 180.0:
            raise ScenarioConfigError(
                f"inclination_deg={value!r} is outside [0, 180]. Inclination is the angle "
                "between the orbit normal and the pole; retrograde orbits are values "
                "above 90 deg, not negative ones."
            )
        return value

    @property
    def semi_major_axis_m(self) -> float:
        """Return the orbit radius, metres. Equal to the semi-major axis for a circle."""
        return R_EARTH_EQUATORIAL_M + self.altitude_m

    @property
    def mean_motion_rad_s(self) -> float:
        """Return the Keplerian mean motion ``n = sqrt(mu / a**3)``, rad/s."""
        return _mean_motion_rad_s(self.semi_major_axis_m)

    @property
    def orbital_period_s(self) -> float:
        """Return the Keplerian orbital period ``T = 2*pi / n``, seconds."""
        return _orbital_period_s(self.semi_major_axis_m)


class HoldPointConfig(BaseModel):
    """A named station-keeping position in the target's Hill frame.

    Ordering is ``[x, y, z]`` = radial-outward (R-bar), along-track (V-bar), positive orbit
    normal, per ``docs/conventions.md``. A chaser trailing the target has negative ``y``.
    """

    model_config = _MODEL_CONFIG

    name: str = Field(min_length=1, description="Human-readable label, used in messages.")
    position_hill_m: tuple[float, float, float] = Field(
        description="Hill-frame position [x, y, z], metres."
    )

    @property
    def radius_m(self) -> float:
        """Return the separation from the target, metres."""
        return math.hypot(*self.position_hill_m)


class ConstraintConfig(BaseModel):
    """Geometric and kinematic approach constraints.

    Activation ranges are separate from the limits themselves because these constraints are
    range-gated: a closing-velocity ceiling that applied at 10 km would forbid the approach
    it exists to protect.
    """

    model_config = _MODEL_CONFIG

    keep_out_sphere_radius_m: float = Field(
        gt=0.0, description="Radius of the sphere about the target that must never be entered."
    )
    approach_ellipsoid_semi_axes_m: tuple[float, float, float] = Field(
        description="Semi-axes [x, y, z] of the approach ellipsoid in the Hill frame, metres."
    )
    approach_cone_half_angle_deg: float = Field(
        gt=0.0, lt=90.0, description="Half-angle of the approach cone, degrees."
    )
    approach_cone_activation_range_m: float = Field(
        gt=0.0, description="Range at or below which the approach cone is enforced, metres."
    )
    max_closing_velocity_m_s: float = Field(
        gt=0.0, description="Maximum permitted range-rate towards the target, m/s."
    )
    max_closing_velocity_activation_range_m: float = Field(
        gt=0.0, description="Range at or below which the closing-velocity limit applies, metres."
    )

    @field_validator("approach_ellipsoid_semi_axes_m")
    @classmethod
    def _reject_non_positive_semi_axes(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Reject a degenerate ellipsoid."""
        if any(axis <= 0.0 for axis in value):
            raise ScenarioConfigError(
                f"approach_ellipsoid_semi_axes_m={value!r} must be strictly positive in all "
                "three axes; a zero or negative semi-axis is a degenerate volume."
            )
        return value

    @model_validator(mode="after")
    def _ellipsoid_must_contain_keep_out_sphere(self) -> Self:
        """Reject an approach corridor smaller than the volume it is supposed to surround."""
        smallest = min(self.approach_ellipsoid_semi_axes_m)
        if smallest < self.keep_out_sphere_radius_m:
            raise ScenarioConfigError(
                f"approach_ellipsoid_semi_axes_m={self.approach_ellipsoid_semi_axes_m!r} has a "
                f"smallest semi-axis of {smallest:,.1f} m, inside the "
                f"keep_out_sphere_radius_m={self.keep_out_sphere_radius_m:,.1f} m sphere. The "
                "approach ellipsoid is the volume within which the approach is controlled and "
                "must enclose the keep-out sphere, otherwise the two constraints contradict."
            )
        return self


class IntegratorConfig(BaseModel):
    """Numerical integration settings.

    Tolerances are configuration, never module constants (``docs/conventions.md``): any
    quoted numerical result has to survive a tolerance sweep, which is only possible if the
    tolerance is a field somebody can sweep.
    """

    model_config = _MODEL_CONFIG

    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = Field(
        default="DOP853", description="scipy.integrate.solve_ivp method name."
    )
    rtol: float = Field(default=1.0e-12, gt=0.0, description="Relative tolerance.")
    atol: float = Field(default=1.0e-12, gt=0.0, description="Absolute tolerance, m and m/s.")


class ManeuverConfig(BaseModel):
    """Two-impulse transfer specification.

    The time of flight is given *either* as an explicit number of seconds *or* as a
    fraction of the orbital period, never both: two spellings of one quantity can disagree,
    and there is no defensible rule for which one wins.
    """

    model_config = _MODEL_CONFIG

    tof_s: float | None = Field(
        default=None, gt=0.0, description="Time of flight in seconds, if given directly."
    )
    tof_periods: float | None = Field(
        default=None, gt=0.0, description="Time of flight in orbital periods, if given that way."
    )

    @model_validator(mode="after")
    def _exactly_one_time_of_flight(self) -> Self:
        """Require exactly one of the two spellings of time of flight."""
        if self.tof_s is not None and self.tof_periods is not None:
            raise ScenarioConfigError(
                f"manoeuvre specifies both tof_s={self.tof_s!r} and "
                f"tof_periods={self.tof_periods!r}. Give exactly one: they are two spellings "
                "of the same quantity and can disagree."
            )
        if self.tof_s is None and self.tof_periods is None:
            raise ScenarioConfigError(
                "manoeuvre specifies neither tof_s nor tof_periods; exactly one is required."
            )
        return self

    def resolve_tof_s(self, orbital_period_s: float) -> float:
        """Return the time of flight in seconds, whichever way it was written.

        Parameters
        ----------
        orbital_period_s
            Target orbital period, seconds. Must be finite and strictly positive.

        Returns
        -------
        float
            Time of flight, seconds.

        Raises
        ------
        ScenarioConfigError
            If ``orbital_period_s`` is not a finite positive time.

        """
        if not math.isfinite(orbital_period_s) or orbital_period_s <= 0.0:
            raise ScenarioConfigError(
                f"orbital_period_s must be a finite positive period, got {orbital_period_s!r}"
            )
        if self.tof_s is not None:
            return self.tof_s
        if self.tof_periods is None:  # pragma: no cover - forbidden by the model validator
            raise ScenarioConfigError("manoeuvre has neither tof_s nor tof_periods")
        return self.tof_periods * orbital_period_s


class ScenarioConfig(BaseModel):
    """A complete, self-consistent rendezvous scenario.

    Everything a run needs and nothing that varies between runs: no paths, no timestamps,
    no machine identity. That is what makes :func:`config_hash` a stable identity for the
    scenario rather than for the invocation.
    """

    model_config = _MODEL_CONFIG

    name: str = Field(min_length=1, description="Short scenario identifier.")
    description: str = Field(default="", description="Free text; part of the hashed identity.")
    orbit: OrbitConfig
    start_hold_point: HoldPointConfig
    target_hold_point: HoldPointConfig
    constraints: ConstraintConfig
    maneuver: ManeuverConfig
    integrator: IntegratorConfig = Field(default_factory=IntegratorConfig)
    seed: int = Field(ge=0, description="Seed for every stochastic entry point in the run.")

    @property
    def tof_s(self) -> float:
        """Return the resolved time of flight, seconds."""
        return self.maneuver.resolve_tof_s(self.orbit.orbital_period_s)

    @property
    def tof_periods(self) -> float:
        """Return the resolved time of flight in orbital periods."""
        return self.tof_s / self.orbit.orbital_period_s

    @property
    def max_separation_m(self) -> float:
        """Return the larger hold-point radius, metres.

        A lower bound on the true maximum separation: the coasting arc between two hold
        points bulges away from the chord joining them.
        """
        return max(self.start_hold_point.radius_m, self.target_hold_point.radius_m)

    @model_validator(mode="after")
    def _hold_points_outside_keep_out_sphere(self) -> Self:
        """Reject hold points inside the volume the trajectory exists to avoid."""
        radius = self.constraints.keep_out_sphere_radius_m
        for field, point in (
            ("start_hold_point", self.start_hold_point),
            ("target_hold_point", self.target_hold_point),
        ):
            if point.radius_m <= radius:
                raise ScenarioConfigError(
                    f"{field} {point.name!r} at {point.position_hill_m!r} m is "
                    f"{point.radius_m:,.3f} m from the target, inside the "
                    f"keep_out_sphere_radius_m={radius:,.3f} m sphere. A hold point inside the "
                    "keep-out zone is a scenario-definition error: there is no trajectory to "
                    "plan from inside the volume the plan exists to avoid."
                )
        return self

    @model_validator(mode="after")
    def _time_of_flight_is_not_a_period_multiple(self) -> Self:
        """Reject transfer times where the in-plane two-impulse solve loses rank."""
        period_s = self.orbit.orbital_period_s
        tof_s = self.tof_s
        multiple = round(tof_s / period_s)
        if multiple >= 1 and abs(tof_s - multiple * period_s) <= PERIOD_MULTIPLE_REL_TOL * period_s:
            raise ScenarioConfigError(
                f"time of flight {tof_s:,.6f} s is {tof_s / period_s:.9f} orbital periods, "
                f"within {PERIOD_MULTIPLE_REL_TOL:.1e} relative of the integer multiple "
                f"{multiple}. The in-plane two-impulse Clohessy-Wiltshire solve inverts "
                "Phi_rv, whose in-plane determinant (8 - 8*cos(tau) - 3*tau*sin(tau)) / n**2 "
                "vanishes at integer multiples of the period, so the departure impulse is "
                f"undefined there (orbital period {period_s:,.3f} s). Choose a different time "
                "of flight or add an intermediate manoeuvre."
            )
        return self

    @model_validator(mode="after")
    def _cross_track_change_is_not_at_a_half_period(self) -> Self:
        """Reject half-period transfers that are asked to move the chaser cross-track."""
        delta_z_m = abs(
            self.target_hold_point.position_hill_m[2] - self.start_hold_point.position_hill_m[2]
        )
        # Below the tolerance the CW solve itself treats the request as already satisfied,
        # so there is nothing rank-deficient about it. Same threshold, same reason.
        if delta_z_m <= DEFAULT_FEASIBILITY_TOL_M:
            return self

        half_period_s = 0.5 * self.orbit.orbital_period_s
        tof_s = self.tof_s
        multiple = round(tof_s / half_period_s)
        if (
            multiple >= 1
            and abs(tof_s - multiple * half_period_s) <= PERIOD_MULTIPLE_REL_TOL * half_period_s
        ):
            raise ScenarioConfigError(
                f"the manoeuvre changes cross-track position by {delta_z_m:,.3f} m, but the "
                f"time of flight {tof_s:,.6f} s is {tof_s / half_period_s:.9f} half-periods, "
                f"within {PERIOD_MULTIPLE_REL_TOL:.1e} relative of the integer multiple "
                f"{multiple}. Cross-track targeting is rank-deficient there: the cross-track "
                "term of Phi_rv is sin(tau)/n, so z(t_f) is pinned to z_0*cos(tau) whatever "
                "impulse is applied. A half-period transfer that does *not* move in z -- the "
                "V-bar hop -- is well posed; this one is not."
            )
        return self

    @model_validator(mode="after")
    def _warn_outside_cw_validity_envelope(self) -> Self:
        """Warn when linearisation error eats a meaningful fraction of the keep-out sphere."""
        tolerance_m = CW_ERROR_BUDGET_FRACTION_OF_KOZ * self.constraints.keep_out_sphere_radius_m
        separation_m = self.max_separation_m
        n_orbits = self.tof_periods
        estimate_m = conservative_cw_error_bound_m(
            separation_m, self.orbit.semi_major_axis_m, n_orbits
        )
        if estimate_m > tolerance_m:
            warnings.warn(
                f"scenario {self.name!r} is outside the measured Clohessy-Wiltshire validity "
                f"envelope: at {separation_m:,.1f} m separation over {n_orbits:.3f} orbits the "
                f"estimated linearisation error is {estimate_m:,.3f} m, which exceeds the "
                f"{tolerance_m:,.3f} m budget "
                f"({CW_ERROR_BUDGET_FRACTION_OF_KOZ:.0%} of the "
                f"{self.constraints.keep_out_sphere_radius_m:,.1f} m keep-out sphere). This is "
                "a modelling judgement, not a malformed scenario -- but any metre-level claim "
                "about keep-out clearance here must be checked against "
                "rpo_core.relative.nonlinear rather than asserted from CW.",
                UserWarning,
                stacklevel=2,
            )
        return self


def config_hash(config: BaseModel, *, length: int = CONFIG_HASH_LENGTH) -> str:
    """Return a short, stable hex digest of a configuration's canonical serialisation.

    Deterministic across processes and interpreter restarts. Python's built-in :func:`hash`
    is seeded per process by ``PYTHONHASHSEED`` and would give a different answer on every
    run, which is exactly the failure mode a run identity must not have.

    The digest covers the JSON-mode dump of the model with sorted keys: every field, no
    derived properties (they are functions of the fields), and nothing about the
    invocation -- no timestamp, no path, no host, no library version. Two identical
    scenarios hash identically whether they were typed out twice or round-tripped through
    a file.

    Parameters
    ----------
    config
        Any validated model in this module.
    length
        Number of hex characters to retain. Defaults to :data:`CONFIG_HASH_LENGTH`.

    Returns
    -------
    str
        Lowercase hex digest, ``length`` characters.

    Raises
    ------
    ScenarioConfigError
        If ``length`` is not in ``[1, 64]``.

    """
    if not 1 <= length <= 64:
        raise ScenarioConfigError(f"length must be in [1, 64] hex characters, got {length!r}")
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _describe_validation_errors(error: ValidationError) -> str:
    """Render a Pydantic validation failure as one line per offending field and value."""
    lines: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "<scenario>"
        offending = repr(detail.get("input"))
        if len(offending) > 120:
            offending = offending[:117] + "..."
        lines.append(f"  {location}: {detail['msg']} (got {offending})")
    return "\n".join(lines)


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML file containing a single top-level mapping.

    Returns
    -------
    ScenarioConfig
        The validated scenario.

    Raises
    ------
    ScenarioConfigError
        If the file cannot be read, is not valid YAML, is not a mapping, or fails
        validation. The message names each offending field and the value it received.

    """
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioConfigError(f"cannot read scenario file {resolved}: {exc}") from exc

    try:
        raw: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioConfigError(f"{resolved} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ScenarioConfigError(
            f"{resolved} must contain a YAML mapping at the top level, got "
            f"{type(raw).__name__} ({raw!r:.80})"
        )

    try:
        return ScenarioConfig.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioConfigError(
            f"{resolved} is not a valid scenario:\n{_describe_validation_errors(exc)}"
        ) from exc


def _package_version(distribution: str) -> str:
    """Return an installed distribution's version, or ``"unknown"`` if it is absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _run_git(args: list[str], repo_dir: Path) -> tuple[int, str]:
    """Run a git command in ``repo_dir``, returning ``(returncode, stripped stdout)``."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return completed.returncode, completed.stdout.strip()


def _git_provenance(repo_dir: Path) -> tuple[str, bool | None]:
    """Return ``(commit, dirty)`` for the working tree containing ``repo_dir``.

    ``commit`` is the full SHA, ``"uncommitted"`` when the repository exists but has no
    commits yet (the current state of this repository), or ``"unknown"`` when there is no
    repository or no git binary. ``dirty`` is ``None`` when it could not be determined --
    reporting a clean tree we did not actually inspect would make the provenance lie.
    """
    inside_code, inside_out = _run_git(["rev-parse", "--is-inside-work-tree"], repo_dir)
    if inside_code != 0 or inside_out != "true":
        return "unknown", None

    status_code, status_out = _run_git(["status", "--porcelain"], repo_dir)
    dirty = bool(status_out) if status_code == 0 else None

    head_code, head_out = _run_git(["rev-parse", "HEAD"], repo_dir)
    if head_code != 0 or not head_out:
        return "uncommitted", dirty
    return head_out, dirty


def create_run_directory(
    config: ScenarioConfig,
    seed: int | None = None,
    base_dir: str | Path = DEFAULT_RUNS_DIR,
    variant: str | None = None,
) -> Path:
    """Create ``<base_dir>/<config_hash>-<seed>/`` and write ``provenance.json`` into it.

    The directory name is content-addressed: the same scenario at the same seed always
    lands in the same place, which is the point. Re-running is therefore idempotent in the
    path and refreshes ``provenance.json`` (its timestamp and git state may legitimately
    have changed); existing result files are left alone.

    The timestamp is recorded in the provenance and deliberately excluded from the hash. A
    timestamped hash would make every run unique, which is the opposite of reproducible.

    ``variant`` distinguishes runs that share a scenario but differ in a *run option* rather
    than in the scenario itself. Without it, planning the same scenario with and without
    nonlinear correction resolves to one directory and the second run silently overwrites the
    first -- two different results, one path, no warning. Any option that changes the numbers
    must appear here.

    Parameters
    ----------
    config
        The validated scenario. Serialised in full into the provenance record.
    seed
        Random seed for this run. Defaults to ``config.seed``. Must be non-negative.
    variant
        Optional discriminator appended to the directory name, for runs that share a
        scenario but differ in a run option that changes the numbers (for example planning
        with and without nonlinear correction). Omit for the canonical run.
    base_dir
        Parent directory. Created if missing. Relative paths resolve against the current
        working directory.

    Returns
    -------
    pathlib.Path
        The created run directory.

    Raises
    ------
    ScenarioConfigError
        If ``seed`` is negative, or the directory or provenance file cannot be written.

    """
    effective_seed = config.seed if seed is None else int(seed)
    if effective_seed < 0:
        raise ScenarioConfigError(f"seed must be non-negative, got {effective_seed!r}")

    digest = config_hash(config)
    suffix = f"-{variant}" if variant else ""
    run_dir = Path(base_dir) / f"{digest}-{effective_seed}{suffix}"

    commit, dirty = _git_provenance(Path(__file__).resolve().parent)
    provenance: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "config_hash": digest,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "package_versions": {
            "numpy": _package_version("numpy"),
            "pydantic": _package_version("pydantic"),
            "pyyaml": _package_version("pyyaml"),
            "rpo-core": _package_version("rpo-core"),
            "scipy": _package_version("scipy"),
        },
        "python_version": platform.python_version(),
        "seed": effective_seed,
    }

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ScenarioConfigError(f"cannot create run directory {run_dir}: {exc}") from exc
    return run_dir
