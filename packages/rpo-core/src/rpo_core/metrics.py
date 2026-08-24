r"""Mission metrics for a planned relative trajectory, and their JSON record.

The equations
-------------
Let the plan be a sampled Hill-frame trajectory :math:`(t_i, \boldsymbol{x}_i)` for
:math:`i = 0 \ldots N-1` with :math:`\boldsymbol{x} = [\boldsymbol{\rho},
\dot{\boldsymbol{\rho}}]`, together with a set of impulses :math:`\Delta\boldsymbol{v}_k`
applied at times :math:`\tau_k`. Every headline number is one of the following.

**Delta-v budget.** The impulses are summed in *magnitude*, not as vectors:

.. math:: \Delta v_\text{total} = \sum_k \lVert \Delta\boldsymbol{v}_k \rVert.

Two opposed impulses of 0.2 m/s cost 0.4 m/s of propellant, not zero, which is why the
per-burn breakdown is carried alongside the total rather than being derivable from it.

**Time of flight.** :math:`t_f = t_{N-1} - t_0`, and :math:`t_f / T` orbital periods.

**Terminal errors** against the commanded terminal state
:math:`[\boldsymbol{\rho}^\star, \dot{\boldsymbol{\rho}}^\star]`:

.. math::

    e_r = \lVert \boldsymbol{\rho}_{N-1} - \boldsymbol{\rho}^\star \rVert, \qquad
    e_v = \lVert \dot{\boldsymbol{\rho}}_{N-1} - \dot{\boldsymbol{\rho}}^\star \rVert.

The comparison is made at the **last sample**. For a two-impulse plan that sample must
already carry the arrival impulse, otherwise :math:`e_v` reports the pre-burn coast
velocity -- a number that looks like a terminal error and is not one. This module cannot
detect the difference and therefore states the requirement rather than guessing.

**Keep-out minimum.** Both the sampled minimum :math:`\min_i \lVert\boldsymbol{\rho}_i\rVert`
and the sub-sample refined minimum come from :mod:`rpo_core.constraints` and are recorded
**separately**. Collapsing them into one "minimum distance" throws away the only signal
that says the grid was too coarse to resolve the closest approach; see
:class:`~rpo_core.constraints.KeepOutSphereResult`.

**Closing velocity.** :math:`v_c = -(\boldsymbol{\rho} \cdot \dot{\boldsymbol{\rho}}) /
\lVert\boldsymbol{\rho}\rVert`, positive when approaching, maximised over the samples
inside the limit's activation range, with the time at which the maximum occurred.

**Clohessy-Wiltshire linearisation error.** The bound, from
:func:`~rpo_core.relative.conservative_cw_error_bound_m`:

.. math:: \varepsilon_\text{bound} = 1.5 \cdot 6\pi \rho_\text{max}^2 / r \cdot N_\text{orb}.

Never :func:`~rpo_core.relative.estimated_cw_error_m`. The central estimate scales
linearly in elapsed orbits and is measurably **optimistic** between roughly 0.4 and 1.0
orbits (``docs/cw_validity.md``), which is exactly where a half-orbit V-bar hop operates.
A metric that under-warns in the flagship regime is worse than no metric.

Why the sampled series lives on the metrics object
--------------------------------------------------
:class:`TrajectoryMetrics` carries the range, range-rate and violation-mask series, not
only the scalar summaries. That is deliberate and it is the whole contract with
:mod:`rpo_core.plotting`: **every number a figure displays is read from a field here, and
the plots recompute nothing.** A figure and ``metrics.json`` that draw from one source
cannot disagree; a figure that recomputes its own range history can, and will, on the day
someone changes a definition in one place. The cost is a ``metrics.json`` that grows
linearly with sample count: measured at 177 bytes per sample on the baseline scenario
(2.6 kB at 2 samples, 45 kB at 241, 179 kB at 1001), which is the right trade for a file
whose job is to be the record. A Monte Carlo campaign retaining thousands of runs should
retain the scalars and drop :attr:`TrajectoryMetrics.series`, not shrink the record here.

Validity
--------
Everything here describes the trajectory that was handed in. Sampled quantities inherit the
sampling: the maximum closing velocity is the maximum *over samples*, and unlike the
keep-out minimum it is not sub-sample refined, so a coarse grid under-reports it. The CW
bound uses the largest *sampled* separation, which is itself a lower bound on the
continuous maximum, so the bound is conservative in its safety factor and mildly optimistic
in its input; it is an order-of-magnitude budget for deciding whether CW is the right model
at all, not a correction to apply to a result.

Constraint counts are counts of *samples*, inherited unchanged from
:class:`~rpo_core.constraints.SafetyReport`: one continuous breach spanning forty samples
counts as forty. ``first_violation_time_s`` is likewise a sampled time, so a keep-out breach
detected only by refinement does not appear in it. Read ``all_constraints_satisfied`` to
decide whether the plan is safe.

Units are SI: metres, seconds, radians, metres per second.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import CW_ERROR_BUDGET_FRACTION_OF_KOZ, ScenarioConfig, config_hash
from .constraints import (
    ApproachCorridor,
    ConstraintResult,
    SafetyReport,
    range_rate_m_s,
    separation_m,
)
from .exceptions import RpoCoreError
from .relative import conservative_cw_error_bound_m

__all__ = [
    "METRICS_FILENAME",
    "METRICS_SCHEMA_VERSION",
    "Burn",
    "BurnMetrics",
    "HoldPointMetrics",
    "MetricsError",
    "TrajectoryMetrics",
    "TrajectorySeries",
    "compute_metrics",
    "read_metrics",
    "write_metrics",
]

#: Name of the file :func:`write_metrics` creates inside a run directory.
METRICS_FILENAME: str = "metrics.json"

#: Version of the :class:`TrajectoryMetrics` field layout written into every record.
#:
#: Bumped whenever a field is added, removed or reinterpreted. :func:`read_metrics` refuses
#: a file it does not recognise rather than reconstructing a dataclass out of the fields
#: that happen to match, which would silently default away the ones that do not.
METRICS_SCHEMA_VERSION: int = 1


class MetricsError(RpoCoreError, ValueError):
    """Raised when metrics cannot be computed, written, or read back.

    Covers a malformed trajectory, a burn placed outside the trajectory's time span, a
    safety report that does not describe the trajectory it was paired with, a non-finite
    value reaching serialisation, and an unreadable or foreign ``metrics.json``. A metrics
    record is the artefact a result is quoted from; producing a plausible-looking one from
    inputs that do not fit together is the failure this type exists to prevent.
    """


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Burn:
    """One impulsive manoeuvre in the plan.

    Parameters
    ----------
    label
        Short human-readable name, used as the bar label in the delta-v budget figure.
        Must be non-empty: an unlabelled bar in a printed budget is not traceable.
    time_s
        Epoch of the impulse, seconds on the trajectory's own time base. Must be finite.
    delta_v_hill_m_s
        Impulse ``[dvx, dvy, dvz]`` in the target Hill frame, m/s. Must be finite. Any
        three-element sequence is accepted and stored as a tuple of floats.

    Raises
    ------
    MetricsError
        If the label is empty, the time is non-finite, or the impulse is not three finite
        components.

    """

    label: str
    time_s: float
    delta_v_hill_m_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate and canonicalise the burn where the offending value is still in scope."""
        if not str(self.label).strip():
            raise MetricsError(f"burn label must be a non-empty name, got {self.label!r}")
        time_s = float(self.time_s)
        if not math.isfinite(time_s):
            raise MetricsError(
                f"burn {self.label!r} has non-finite time_s={self.time_s!r}; a burn epoch "
                "must be a real time on the trajectory's time base"
            )
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(
            self, "delta_v_hill_m_s", _unit_free_triple(self.delta_v_hill_m_s, "delta_v_hill_m_s")
        )

    @property
    def magnitude_m_s(self) -> float:
        """Return the impulse magnitude ``|dv|``, m/s."""
        return float(math.sqrt(sum(component * component for component in self.delta_v_hill_m_s)))


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BurnMetrics:
    """One burn, located on the trajectory it was applied to.

    Attributes
    ----------
    label, time_s, delta_v_hill_m_s
        As supplied on the :class:`Burn`.
    magnitude_m_s
        ``|dv|``, m/s. The contribution this burn makes to the total.
    sample_index
        Index of the trajectory sample nearest ``time_s``. The burn is *drawn* at this
        sample, so a burn between samples is marked at the nearer one rather than at an
        interpolated point that exists on no propagated state.
    position_hill_m
        Hill-frame position at ``sample_index``, metres.

    """

    label: str
    time_s: float
    delta_v_hill_m_s: tuple[float, float, float]
    magnitude_m_s: float
    sample_index: int
    position_hill_m: tuple[float, float, float]


@dataclass(frozen=True)
class HoldPointMetrics:
    """A named hold point carried through from the scenario, for figure annotation."""

    name: str
    position_hill_m: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectorySeries:
    """The per-sample history every figure draws from.

    Attributes
    ----------
    times_s
        Sample times, seconds.
    position_hill_m
        Hill-frame positions ``[x, y, z]``, metres, one triple per sample.
    range_m
        ``|rho|`` at each sample, metres.
    range_rate_m_s
        ``d|rho|/dt`` at each sample, m/s. Negative means closing; the closing velocity is
        its negation.
    closing_velocity_violating
        Per-sample mask, ``True`` where the sample is inside the closing-velocity
        activation range **and** exceeds the limit. Carried as data rather than
        recomputed by the plotting layer so that the shaded spans in a figure and the
        violation count in the JSON are the same decision.

    """

    times_s: tuple[float, ...]
    position_hill_m: tuple[tuple[float, float, float], ...]
    range_m: tuple[float, ...]
    range_rate_m_s: tuple[float, ...]
    closing_velocity_violating: tuple[bool, ...]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.times_s)


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Every number quoted about one planned trajectory.

    Frozen, and built only by :func:`compute_metrics`. The plotting layer treats this as
    read-only input and derives nothing of its own from the trajectory.

    Attributes
    ----------
    schema_version
        :data:`METRICS_SCHEMA_VERSION` at the time of writing.
    scenario_name, config_hash, seed
        Run identity. ``config_hash`` and ``seed`` name the run directory
        (:func:`rpo_core.config.create_run_directory`) and appear in every figure footer,
        so a printed figure traces back to the configuration that produced it.
    total_delta_v_m_s, burns
        Sum of impulse magnitudes, m/s, and the per-burn breakdown.
    time_of_flight_s, time_of_flight_periods
        Span of the trajectory, seconds and orbital periods of the target orbit.
    commanded_terminal_state_hill, achieved_terminal_state_hill
        Six-element Hill states, m and m/s: what was asked for, and the last sample.
    terminal_position_error_m, terminal_velocity_error_m_s
        Norms of the difference between the two.
    keep_out_radius_m
        Sphere radius from the scenario, metres.
    min_koz_range_sampled_m, min_koz_range_refined_m
        Minimum ``|rho|``, as sampled and after sub-sample refinement, metres. Kept apart
        on purpose; see the module docstring.
    min_koz_clearance_sampled_m, min_koz_clearance_refined_m
        The same two minima less ``keep_out_radius_m``. Negative means inside the sphere.
    min_koz_time_sampled_s, min_koz_time_refined_s
        Times of the two minima, seconds.
    koz_refinement_applied
        ``False`` when the sampled minimum sat on an endpoint or the local fit did not
        describe a strict minimum, in which case the refined fields repeat the sampled
        ones and the closest approach is unresolved rather than confirmed.
    closing_velocity_limit_m_s, closing_velocity_activation_range_m
        The limit from the scenario and the range at or below which it applies.
    max_closing_velocity_m_s, max_closing_velocity_time_s
        Largest closing velocity over samples inside the activation range, m/s, and when.
        Both ``None`` when no sample was inside the activation range -- "never enforced",
        which is a different statement from "enforced and zero".
    corridor_half_angle_rad, corridor_activation_range_m, corridor_axis_hill
        Approach-cone geometry from the scenario, used to draw the corridor wedge.
    max_corridor_angle_rad
        Largest angle from the cone axis over active samples, radians, or ``None`` when the
        corridor was not evaluated or never active.
    max_ellipsoid_quadratic_form
        Largest value of ``(x/a)^2 + (y/b)^2 + (z/c)^2``; above 1 is outside. ``None`` when
        the ellipsoid was not evaluated.
    constraint_violation_count, first_violation_time_s, all_constraints_satisfied
        Straight from the :class:`~rpo_core.constraints.SafetyReport`. The count is of
        samples across all constraints, and the time is a sampled time.
    cw_error_bound_m, cw_error_budget_m, cw_error_separation_m, cw_error_n_orbits
        Conservative CW linearisation error bound, the budget it is judged against
        (``CW_ERROR_BUDGET_FRACTION_OF_KOZ`` of the keep-out radius), and the two inputs
        that produced the bound.
    cw_within_budget
        ``cw_error_bound_m <= cw_error_budget_m``.
    hold_points
        Scenario hold points, for figure annotation.
    series
        Per-sample history; see :class:`TrajectorySeries`.

    """

    schema_version: int
    scenario_name: str
    config_hash: str
    seed: int

    total_delta_v_m_s: float
    burns: tuple[BurnMetrics, ...]

    time_of_flight_s: float
    time_of_flight_periods: float

    commanded_terminal_state_hill: tuple[float, float, float, float, float, float]
    achieved_terminal_state_hill: tuple[float, float, float, float, float, float]
    terminal_position_error_m: float
    terminal_velocity_error_m_s: float

    keep_out_radius_m: float
    min_koz_range_sampled_m: float
    min_koz_range_refined_m: float
    min_koz_clearance_sampled_m: float
    min_koz_clearance_refined_m: float
    min_koz_time_sampled_s: float
    min_koz_time_refined_s: float
    koz_refinement_applied: bool

    closing_velocity_limit_m_s: float
    closing_velocity_activation_range_m: float
    max_closing_velocity_m_s: float | None
    max_closing_velocity_time_s: float | None

    corridor_half_angle_rad: float
    corridor_activation_range_m: float
    corridor_axis_hill: tuple[float, float, float]
    max_corridor_angle_rad: float | None
    max_ellipsoid_quadratic_form: float | None

    constraint_violation_count: int
    first_violation_time_s: float | None
    all_constraints_satisfied: bool

    cw_error_bound_m: float
    cw_error_budget_m: float
    cw_error_separation_m: float
    cw_error_n_orbits: float
    cw_within_budget: bool

    hold_points: tuple[HoldPointMetrics, ...]
    series: TrajectorySeries

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of every field, with tuples flattened to lists.

        Floats are left as Python floats; :func:`write_metrics` serialises them at ``repr``
        precision, which is shortest-round-trip in CPython and therefore exact.

        Returns
        -------
        dict
            Keys are the dataclass field names, in declaration order.

        Raises
        ------
        MetricsError
            If any float in the record is non-finite. ``nan`` and ``inf`` have no standard
            JSON spelling, and a record that silently emits them is one that cannot be read
            back by anything but Python.

        """
        return {
            "schema_version": self.schema_version,
            "scenario_name": self.scenario_name,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "total_delta_v_m_s": _finite(self.total_delta_v_m_s, "total_delta_v_m_s"),
            "burns": [
                {
                    "label": burn.label,
                    "time_s": _finite(burn.time_s, "burns[].time_s"),
                    "delta_v_hill_m_s": _finite_list(
                        burn.delta_v_hill_m_s, "burns[].delta_v_hill_m_s"
                    ),
                    "magnitude_m_s": _finite(burn.magnitude_m_s, "burns[].magnitude_m_s"),
                    "sample_index": burn.sample_index,
                    "position_hill_m": _finite_list(
                        burn.position_hill_m, "burns[].position_hill_m"
                    ),
                }
                for burn in self.burns
            ],
            "time_of_flight_s": _finite(self.time_of_flight_s, "time_of_flight_s"),
            "time_of_flight_periods": _finite(
                self.time_of_flight_periods, "time_of_flight_periods"
            ),
            "commanded_terminal_state_hill": _finite_list(
                self.commanded_terminal_state_hill, "commanded_terminal_state_hill"
            ),
            "achieved_terminal_state_hill": _finite_list(
                self.achieved_terminal_state_hill, "achieved_terminal_state_hill"
            ),
            "terminal_position_error_m": _finite(
                self.terminal_position_error_m, "terminal_position_error_m"
            ),
            "terminal_velocity_error_m_s": _finite(
                self.terminal_velocity_error_m_s, "terminal_velocity_error_m_s"
            ),
            "keep_out_radius_m": _finite(self.keep_out_radius_m, "keep_out_radius_m"),
            "min_koz_range_sampled_m": _finite(
                self.min_koz_range_sampled_m, "min_koz_range_sampled_m"
            ),
            "min_koz_range_refined_m": _finite(
                self.min_koz_range_refined_m, "min_koz_range_refined_m"
            ),
            "min_koz_clearance_sampled_m": _finite(
                self.min_koz_clearance_sampled_m, "min_koz_clearance_sampled_m"
            ),
            "min_koz_clearance_refined_m": _finite(
                self.min_koz_clearance_refined_m, "min_koz_clearance_refined_m"
            ),
            "min_koz_time_sampled_s": _finite(
                self.min_koz_time_sampled_s, "min_koz_time_sampled_s"
            ),
            "min_koz_time_refined_s": _finite(
                self.min_koz_time_refined_s, "min_koz_time_refined_s"
            ),
            "koz_refinement_applied": self.koz_refinement_applied,
            "closing_velocity_limit_m_s": _finite(
                self.closing_velocity_limit_m_s, "closing_velocity_limit_m_s"
            ),
            "closing_velocity_activation_range_m": _finite(
                self.closing_velocity_activation_range_m, "closing_velocity_activation_range_m"
            ),
            "max_closing_velocity_m_s": _finite_or_none(
                self.max_closing_velocity_m_s, "max_closing_velocity_m_s"
            ),
            "max_closing_velocity_time_s": _finite_or_none(
                self.max_closing_velocity_time_s, "max_closing_velocity_time_s"
            ),
            "corridor_half_angle_rad": _finite(
                self.corridor_half_angle_rad, "corridor_half_angle_rad"
            ),
            "corridor_activation_range_m": _finite(
                self.corridor_activation_range_m, "corridor_activation_range_m"
            ),
            "corridor_axis_hill": _finite_list(self.corridor_axis_hill, "corridor_axis_hill"),
            "max_corridor_angle_rad": _finite_or_none(
                self.max_corridor_angle_rad, "max_corridor_angle_rad"
            ),
            "max_ellipsoid_quadratic_form": _finite_or_none(
                self.max_ellipsoid_quadratic_form, "max_ellipsoid_quadratic_form"
            ),
            "constraint_violation_count": self.constraint_violation_count,
            "first_violation_time_s": _finite_or_none(
                self.first_violation_time_s, "first_violation_time_s"
            ),
            "all_constraints_satisfied": self.all_constraints_satisfied,
            "cw_error_bound_m": _finite(self.cw_error_bound_m, "cw_error_bound_m"),
            "cw_error_budget_m": _finite(self.cw_error_budget_m, "cw_error_budget_m"),
            "cw_error_separation_m": _finite(self.cw_error_separation_m, "cw_error_separation_m"),
            "cw_error_n_orbits": _finite(self.cw_error_n_orbits, "cw_error_n_orbits"),
            "cw_within_budget": self.cw_within_budget,
            "hold_points": [
                {
                    "name": point.name,
                    "position_hill_m": _finite_list(
                        point.position_hill_m, "hold_points[].position_hill_m"
                    ),
                }
                for point in self.hold_points
            ],
            "series": {
                "times_s": _finite_list(self.series.times_s, "series.times_s"),
                "position_hill_m": [
                    _finite_list(triple, "series.position_hill_m")
                    for triple in self.series.position_hill_m
                ],
                "range_m": _finite_list(self.series.range_m, "series.range_m"),
                "range_rate_m_s": _finite_list(self.series.range_rate_m_s, "series.range_rate_m_s"),
                "closing_velocity_violating": [
                    bool(flag) for flag in self.series.closing_velocity_violating
                ],
            },
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> TrajectoryMetrics:
        """Rebuild a :class:`TrajectoryMetrics` from :meth:`to_json_dict` output.

        Parameters
        ----------
        payload
            A decoded ``metrics.json`` body.

        Returns
        -------
        TrajectoryMetrics
            Bitwise identical in every float to the record that was written, because
            CPython's ``repr`` for floats is shortest-round-trip.

        Raises
        ------
        MetricsError
            If ``schema_version`` is absent or is not :data:`METRICS_SCHEMA_VERSION`, or a
            field is missing or has the wrong shape.

        """
        version = payload.get("schema_version")
        if version != METRICS_SCHEMA_VERSION:
            raise MetricsError(
                f"metrics schema_version {version!r} is not the supported "
                f"{METRICS_SCHEMA_VERSION!r}. Refusing to rebuild a record whose field "
                "layout this version does not know, because the fields it does not "
                "recognise would be silently defaulted away."
            )
        try:
            series = payload["series"]
            return cls(
                schema_version=int(payload["schema_version"]),
                scenario_name=str(payload["scenario_name"]),
                config_hash=str(payload["config_hash"]),
                seed=int(payload["seed"]),
                total_delta_v_m_s=float(payload["total_delta_v_m_s"]),
                burns=tuple(
                    BurnMetrics(
                        label=str(burn["label"]),
                        time_s=float(burn["time_s"]),
                        delta_v_hill_m_s=_triple(burn["delta_v_hill_m_s"]),
                        magnitude_m_s=float(burn["magnitude_m_s"]),
                        sample_index=int(burn["sample_index"]),
                        position_hill_m=_triple(burn["position_hill_m"]),
                    )
                    for burn in payload["burns"]
                ),
                time_of_flight_s=float(payload["time_of_flight_s"]),
                time_of_flight_periods=float(payload["time_of_flight_periods"]),
                commanded_terminal_state_hill=_sextuple(payload["commanded_terminal_state_hill"]),
                achieved_terminal_state_hill=_sextuple(payload["achieved_terminal_state_hill"]),
                terminal_position_error_m=float(payload["terminal_position_error_m"]),
                terminal_velocity_error_m_s=float(payload["terminal_velocity_error_m_s"]),
                keep_out_radius_m=float(payload["keep_out_radius_m"]),
                min_koz_range_sampled_m=float(payload["min_koz_range_sampled_m"]),
                min_koz_range_refined_m=float(payload["min_koz_range_refined_m"]),
                min_koz_clearance_sampled_m=float(payload["min_koz_clearance_sampled_m"]),
                min_koz_clearance_refined_m=float(payload["min_koz_clearance_refined_m"]),
                min_koz_time_sampled_s=float(payload["min_koz_time_sampled_s"]),
                min_koz_time_refined_s=float(payload["min_koz_time_refined_s"]),
                koz_refinement_applied=bool(payload["koz_refinement_applied"]),
                closing_velocity_limit_m_s=float(payload["closing_velocity_limit_m_s"]),
                closing_velocity_activation_range_m=float(
                    payload["closing_velocity_activation_range_m"]
                ),
                max_closing_velocity_m_s=_optional_float(payload["max_closing_velocity_m_s"]),
                max_closing_velocity_time_s=_optional_float(payload["max_closing_velocity_time_s"]),
                corridor_half_angle_rad=float(payload["corridor_half_angle_rad"]),
                corridor_activation_range_m=float(payload["corridor_activation_range_m"]),
                corridor_axis_hill=_triple(payload["corridor_axis_hill"]),
                max_corridor_angle_rad=_optional_float(payload["max_corridor_angle_rad"]),
                max_ellipsoid_quadratic_form=_optional_float(
                    payload["max_ellipsoid_quadratic_form"]
                ),
                constraint_violation_count=int(payload["constraint_violation_count"]),
                first_violation_time_s=_optional_float(payload["first_violation_time_s"]),
                all_constraints_satisfied=bool(payload["all_constraints_satisfied"]),
                cw_error_bound_m=float(payload["cw_error_bound_m"]),
                cw_error_budget_m=float(payload["cw_error_budget_m"]),
                cw_error_separation_m=float(payload["cw_error_separation_m"]),
                cw_error_n_orbits=float(payload["cw_error_n_orbits"]),
                cw_within_budget=bool(payload["cw_within_budget"]),
                hold_points=tuple(
                    HoldPointMetrics(
                        name=str(point["name"]),
                        position_hill_m=_triple(point["position_hill_m"]),
                    )
                    for point in payload["hold_points"]
                ),
                series=TrajectorySeries(
                    times_s=tuple(float(value) for value in series["times_s"]),
                    position_hill_m=tuple(_triple(triple) for triple in series["position_hill_m"]),
                    range_m=tuple(float(value) for value in series["range_m"]),
                    range_rate_m_s=tuple(float(value) for value in series["range_rate_m_s"]),
                    closing_velocity_violating=tuple(
                        bool(flag) for flag in series["closing_velocity_violating"]
                    ),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MetricsError(f"metrics record is malformed: {exc}") from exc


# --------------------------------------------------------------------------------------
# Coercion and validation helpers
# --------------------------------------------------------------------------------------


def _unit_free_triple(value: Iterable[float], name: str) -> tuple[float, float, float]:
    """Return ``value`` as a validated three-tuple of finite floats."""
    try:
        components = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"{name} must be three finite numbers, got {value!r}") from exc
    if len(components) != 3:
        raise MetricsError(f"{name} must have three entries, got {len(components)}: {value!r}")
    if not all(math.isfinite(item) for item in components):
        raise MetricsError(f"{name} must be finite, got {value!r}")
    return (components[0], components[1], components[2])


def _triple(value: Any) -> tuple[float, float, float]:
    """Return a decoded JSON array as a three-tuple of floats."""
    components = [float(item) for item in value]
    if len(components) != 3:
        raise ValueError(f"expected three components, got {len(components)}")
    return (components[0], components[1], components[2])


def _sextuple(value: Any) -> tuple[float, float, float, float, float, float]:
    """Return a decoded JSON array as a six-tuple of floats."""
    components = [float(item) for item in value]
    if len(components) != 6:
        raise ValueError(f"expected six components, got {len(components)}")
    return (
        components[0],
        components[1],
        components[2],
        components[3],
        components[4],
        components[5],
    )


def _optional_float(value: Any) -> float | None:
    """Return ``None`` for JSON null, otherwise a float."""
    return None if value is None else float(value)


def _finite(value: float, name: str) -> float:
    """Return ``value`` if it is finite, otherwise raise."""
    number = float(value)
    if not math.isfinite(number):
        raise MetricsError(
            f"{name}={value!r} is not finite. NaN and infinity have no standard JSON "
            "spelling, so a record carrying them is unreadable outside Python; a metric "
            "that is undefined is recorded as null, never as NaN."
        )
    return number


def _finite_or_none(value: float | None, name: str) -> float | None:
    """Return ``None`` unchanged, otherwise the value if finite."""
    return None if value is None else _finite(value, name)


def _finite_list(values: Iterable[float], name: str) -> list[float]:
    """Return ``values`` as a list of floats, raising on the first non-finite entry."""
    return [_finite(value, name) for value in values]


def _nan_to_none(value: float) -> float | None:
    """Return ``None`` for a ``nan`` sentinel, otherwise the float.

    :mod:`rpo_core.constraints` reports ``nan`` worst-case fields for a constraint that had
    no active samples, precisely so that "never enforced" is distinguishable from "enforced
    and fine". That distinction survives into the record as ``null``; it would not survive
    as a fabricated zero.
    """
    return None if math.isnan(value) else float(value)


def _validate_trajectory(
    times_s: npt.ArrayLike, states_hill: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the trajectory as validated float arrays, or raise :class:`MetricsError`.

    Deliberately duplicates the shape and monotonicity checks in
    :mod:`rpo_core.constraints` rather than importing its private validator, so that a
    caller who computes metrics without ever calling ``evaluate_constraints`` still gets
    the same rejection, reported as a metrics error.
    """
    times = np.asarray(times_s, dtype=np.float64)
    states = np.asarray(states_hill, dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != 6:
        raise MetricsError(
            f"states_hill must have shape (N, 6) as [x, y, z, xdot, ydot, zdot], got shape "
            f"{states.shape}"
        )
    if times.ndim != 1:
        raise MetricsError(f"times_s must have shape (N,), got shape {times.shape}")
    if times.size < 2:
        raise MetricsError(
            f"a trajectory needs at least two samples to have a time of flight, got {times.size}"
        )
    if times.size != states.shape[0]:
        raise MetricsError(
            f"times_s and states_hill must have the same length, got {times.size} times and "
            f"{states.shape[0]} states"
        )
    if not np.all(np.isfinite(times)):
        index = int(np.flatnonzero(~np.isfinite(times))[0])
        raise MetricsError(
            f"times_s must be finite: first non-finite entry at index {index}, value "
            f"{float(times[index])!r}"
        )
    if not np.all(np.isfinite(states)):
        index = int(np.flatnonzero(~np.all(np.isfinite(states), axis=1))[0])
        raise MetricsError(
            f"states_hill must be finite: first non-finite entry at row {index}, value "
            f"{states[index].tolist()!r}"
        )
    steps = np.diff(times)
    bad = np.flatnonzero(steps <= 0.0)
    if bad.size > 0:
        index = int(bad[0])
        raise MetricsError(
            f"times_s must be strictly increasing: times_s[{index}]={float(times[index])!r} is "
            f"not less than times_s[{index + 1}]={float(times[index + 1])!r} (step "
            f"{float(steps[index])!r} s). A time base that runs backwards makes "
            "'first violation time' meaningless while still looking like an answer."
        )
    return times, states


def _check_fits(result: ConstraintResult | None, n_samples: int) -> None:
    """Raise if a constraint result cannot have come from a trajectory of this length.

    A report computed from a *different* trajectory is the failure this catches: its
    ``worst_index`` will not sit inside the sample range it is being paired with. Cheap,
    and it turns a silent mismatch into a message naming both lengths.
    """
    if result is None:
        return
    if not -1 <= result.worst_index < n_samples:
        raise MetricsError(
            f"safety report does not describe this trajectory: its {result.name!r} result has "
            f"worst_index={result.worst_index} but the trajectory has {n_samples} samples. "
            "The report and the trajectory must come from the same run."
        )


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


def compute_metrics(
    config: ScenarioConfig,
    times_s: npt.ArrayLike,
    states_hill: npt.ArrayLike,
    burns: Sequence[Burn],
    report: SafetyReport,
    *,
    commanded_terminal_state_hill: npt.ArrayLike,
    corridor_axis_hill: npt.ArrayLike = ApproachCorridor.axis_hill,
    seed: int | None = None,
) -> TrajectoryMetrics:
    """Reduce a planned trajectory and its safety report to a :class:`TrajectoryMetrics`.

    Parameters
    ----------
    config
        The validated scenario. Supplies the run identity, the constraint geometry drawn on
        the figures, and the orbit used to convert time of flight into orbital periods.
    times_s
        Shape (N,) sample times, seconds, strictly increasing, ``N >= 2``.
    states_hill
        Shape (N, 6) Hill-frame states ``[x, y, z, xdot, ydot, zdot]``, m and m/s. The
        **final sample must already carry the arrival impulse** if the plan has one;
        otherwise ``terminal_velocity_error_m_s`` reports the coast velocity. See the module
        docstring.
    burns
        The impulses in the plan, in any order. May be empty: a natural-motion
        circumnavigation is a legitimate plan with a zero delta-v budget.
    report
        Constraint outcome for *this* trajectory. Must carry at least the keep-out sphere
        and the closing-velocity results, which back two of the required metrics.
    commanded_terminal_state_hill
        Shape (6,) Hill state the plan was targeting, m and m/s.
    corridor_axis_hill
        Unit-length approach-cone axis in Hill coordinates, pointing from the target toward
        the chaser. Defaults to the same ``(0, -1, 0)`` V-bar-from-behind axis that
        :class:`~rpo_core.constraints.ApproachCorridor` defaults to. Carried here only so
        the corridor wedge can be drawn; it is **not** validated against the corridor the
        report was computed with, because the report does not carry its own geometry. Pass
        the axis you evaluated with.
    seed
        Run seed. Defaults to ``config.seed``.

    Returns
    -------
    TrajectoryMetrics
        Frozen, and the sole source of every number the plotting layer draws.

    Raises
    ------
    MetricsError
        If the trajectory is malformed (wrong shape, fewer than two samples, non-finite, or
        not strictly increasing in time); if a burn falls outside the trajectory's time
        span; if ``commanded_terminal_state_hill`` is not six finite components; if the
        report omits the keep-out or closing-velocity result, or describes a different
        trajectory.

    """
    times, states = _validate_trajectory(times_s, states_hill)
    n_samples = times.size

    commanded = np.asarray(commanded_terminal_state_hill, dtype=np.float64)
    if commanded.shape != (6,):
        raise MetricsError(
            f"commanded_terminal_state_hill must have shape (6,) as "
            f"[x, y, z, xdot, ydot, zdot], got shape {commanded.shape}"
        )
    if not np.all(np.isfinite(commanded)):
        raise MetricsError(
            f"commanded_terminal_state_hill must be finite, got {commanded.tolist()!r}"
        )

    axis = _unit_free_triple(np.asarray(corridor_axis_hill).tolist(), "corridor_axis_hill")

    keep_out = report.keep_out
    if keep_out is None:
        raise MetricsError(
            "safety report has no keep_out result, but minimum keep-out-zone distance is a "
            "required metric. Evaluate the keep-out sphere before computing metrics rather "
            "than recording a plan whose closest approach is unknown."
        )
    closing = report.closing_velocity
    if closing is None:
        raise MetricsError(
            "safety report has no closing_velocity result, but maximum closing velocity is "
            "a required metric. Evaluate the closing-velocity limit before computing "
            "metrics."
        )
    corridor = report.corridor
    ellipsoid = report.ellipsoid
    for result in report.results:
        _check_fits(result, n_samples)

    # Sampled series. Computed once, here, and read by every figure: the plotting layer
    # deriving its own range history is exactly how a figure and a JSON file start to
    # disagree.
    ranges = separation_m(states)
    rates = range_rate_m_s(states)
    constraint_config = config.constraints
    active = ranges <= constraint_config.max_closing_velocity_activation_range_m
    violating = active & (-rates > constraint_config.max_closing_velocity_m_s)
    if int(np.count_nonzero(violating)) != closing.n_violating_samples:
        raise MetricsError(
            f"the closing-velocity result was computed with a different limit than the "
            f"scenario's: the report counts {closing.n_violating_samples} violating samples "
            f"but scenario limits (max {constraint_config.max_closing_velocity_m_s!r} m/s "
            f"inside {constraint_config.max_closing_velocity_activation_range_m!r} m) give "
            f"{int(np.count_nonzero(violating))}. The shaded spans in a figure and the count "
            "in the record would disagree."
        )

    burn_metrics = tuple(
        _locate_burn(burn, times, states, index) for index, burn in enumerate(burns)
    )
    total_delta_v = float(sum(burn.magnitude_m_s for burn in burn_metrics))

    tof_s = float(times[-1] - times[0])
    period_s = config.orbit.orbital_period_s
    n_orbits = tof_s / period_s

    achieved = states[-1]
    terminal_position_error = float(np.linalg.norm(achieved[:3] - commanded[:3]))
    terminal_velocity_error = float(np.linalg.norm(achieved[3:] - commanded[3:]))

    keep_out_radius = constraint_config.keep_out_sphere_radius_m
    sampled_min_range = keep_out.sampled_min_range_m
    refined_min_range = keep_out.refined_min_range_m

    # The bound, never the central estimate: the linear-in-time law is optimistic between
    # ~0.4 and 1.0 orbits, which is the regime of the flagship half-orbit V-bar hop.
    # The separation is the largest *sampled* range rather than config.max_separation_m,
    # because the transfer arc bulges away from the chord joining the hold points and the
    # hold-point radius therefore understates the excursion the linearisation has to cover.
    separation = float(ranges.max())
    bound_m = conservative_cw_error_bound_m(separation, config.orbit.semi_major_axis_m, n_orbits)
    budget_m = CW_ERROR_BUDGET_FRACTION_OF_KOZ * keep_out_radius

    return TrajectoryMetrics(
        schema_version=METRICS_SCHEMA_VERSION,
        scenario_name=config.name,
        config_hash=config_hash(config),
        seed=config.seed if seed is None else int(seed),
        total_delta_v_m_s=total_delta_v,
        burns=burn_metrics,
        time_of_flight_s=tof_s,
        time_of_flight_periods=n_orbits,
        commanded_terminal_state_hill=_sextuple(commanded.tolist()),
        achieved_terminal_state_hill=_sextuple(achieved.tolist()),
        terminal_position_error_m=terminal_position_error,
        terminal_velocity_error_m_s=terminal_velocity_error,
        keep_out_radius_m=keep_out_radius,
        min_koz_range_sampled_m=sampled_min_range,
        min_koz_range_refined_m=refined_min_range,
        min_koz_clearance_sampled_m=sampled_min_range - keep_out_radius,
        min_koz_clearance_refined_m=refined_min_range - keep_out_radius,
        min_koz_time_sampled_s=keep_out.worst_time_s,
        min_koz_time_refined_s=keep_out.refined_time_s,
        koz_refinement_applied=keep_out.refinement_applied,
        closing_velocity_limit_m_s=constraint_config.max_closing_velocity_m_s,
        closing_velocity_activation_range_m=(
            constraint_config.max_closing_velocity_activation_range_m
        ),
        max_closing_velocity_m_s=_nan_to_none(closing.worst_value),
        max_closing_velocity_time_s=_nan_to_none(closing.worst_time_s),
        corridor_half_angle_rad=math.radians(constraint_config.approach_cone_half_angle_deg),
        corridor_activation_range_m=constraint_config.approach_cone_activation_range_m,
        corridor_axis_hill=axis,
        max_corridor_angle_rad=None if corridor is None else _nan_to_none(corridor.worst_value),
        max_ellipsoid_quadratic_form=(
            None if ellipsoid is None else _nan_to_none(ellipsoid.worst_value)
        ),
        constraint_violation_count=report.total_violating_samples,
        first_violation_time_s=report.first_violation_time_s,
        all_constraints_satisfied=report.all_satisfied,
        cw_error_bound_m=bound_m,
        cw_error_budget_m=budget_m,
        cw_error_separation_m=separation,
        cw_error_n_orbits=n_orbits,
        cw_within_budget=bound_m <= budget_m,
        hold_points=(
            HoldPointMetrics(
                name=config.start_hold_point.name,
                position_hill_m=config.start_hold_point.position_hill_m,
            ),
            HoldPointMetrics(
                name=config.target_hold_point.name,
                position_hill_m=config.target_hold_point.position_hill_m,
            ),
        ),
        series=TrajectorySeries(
            times_s=tuple(float(value) for value in times),
            position_hill_m=tuple(
                (float(row[0]), float(row[1]), float(row[2])) for row in states[:, :3]
            ),
            range_m=tuple(float(value) for value in ranges),
            range_rate_m_s=tuple(float(value) for value in rates),
            closing_velocity_violating=tuple(bool(flag) for flag in violating),
        ),
    )


def _locate_burn(
    burn: Burn,
    times: npt.NDArray[np.float64],
    states: npt.NDArray[np.float64],
    ordinal: int,
) -> BurnMetrics:
    """Return ``burn`` enriched with the trajectory sample it is drawn at."""
    if not times[0] <= burn.time_s <= times[-1]:
        raise MetricsError(
            f"burn {ordinal} ({burn.label!r}) is at time_s={burn.time_s!r} s, outside the "
            f"trajectory span [{float(times[0])!r}, {float(times[-1])!r}] s. A burn the "
            "trajectory does not cover cannot be placed on it, and a delta-v budget that "
            "includes it would describe a different plan."
        )
    index = int(np.argmin(np.abs(times - burn.time_s)))
    position = states[index, :3]
    return BurnMetrics(
        label=burn.label,
        time_s=burn.time_s,
        delta_v_hill_m_s=burn.delta_v_hill_m_s,
        magnitude_m_s=burn.magnitude_m_s,
        sample_index=index,
        position_hill_m=(float(position[0]), float(position[1]), float(position[2])),
    )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def write_metrics(path: str | Path, metrics: TrajectoryMetrics) -> Path:
    """Write ``metrics.json`` into a run directory.

    Floats are serialised by :mod:`json`, which uses ``float.__repr__`` -- the shortest
    decimal string that reads back as the identical double. The record therefore round-trips
    bitwise, and a downstream table quoting from it is quoting the number that was computed
    rather than a rounding of it.

    Parameters
    ----------
    path
        Run directory, typically from :func:`rpo_core.config.create_run_directory`. Created
        if missing.
    metrics
        The record to write.

    Returns
    -------
    pathlib.Path
        Path of the written ``metrics.json``.

    Raises
    ------
    MetricsError
        If a metric is non-finite, or the directory or file cannot be written.

    """
    payload = metrics.to_json_dict()
    directory = Path(path)
    destination = directory / METRICS_FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise MetricsError(f"cannot write {destination}: {exc}") from exc
    return destination


def read_metrics(path: str | Path) -> TrajectoryMetrics:
    """Read ``metrics.json`` back from a run directory.

    Parameters
    ----------
    path
        Run directory containing ``metrics.json``.

    Returns
    -------
    TrajectoryMetrics
        Equal, field for field and bit for bit in every float, to the record written.

    Raises
    ------
    MetricsError
        If the file cannot be read, is not JSON, is not a JSON object, or does not match
        :data:`METRICS_SCHEMA_VERSION`.

    """
    source = Path(path) / METRICS_FILENAME
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise MetricsError(f"cannot read {source}: {exc}") from exc
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetricsError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MetricsError(
            f"{source} must contain a JSON object at the top level, got {type(payload).__name__}"
        )
    return TrajectoryMetrics.from_json_dict(payload)
