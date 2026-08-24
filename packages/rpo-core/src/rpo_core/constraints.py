r"""Proximity-operations safety constraints on a sampled Hill-frame trajectory.

The equations
-------------
Let :math:`\boldsymbol{\rho}(t) = [x, y, z]` be the chaser position relative to the target
and :math:`\dot{\boldsymbol{\rho}}(t) = [\dot{x}, \dot{y}, \dot{z}]` its rotating-frame
derivative, both in the target-centred Hill frame of ``docs/conventions.md`` (x radial
outward / R-bar, y along-track / V-bar, z the positive orbit normal; a trailing chaser sits
at negative y). Four constraints are evaluated sample by sample.

**Keep-out sphere** of radius :math:`R`. The signed clearance is

.. math:: c(t) = \lVert\boldsymbol{\rho}(t)\rVert - R,

violated when :math:`c < 0`. Exactly zero is tangency, and tangency is *not* a violation.

**Approach ellipsoid** with semi-axes :math:`(a, b, c)` along Hill x/y/z. Containment is
the quadratic form

.. math:: q(t) = (x/a)^2 + (y/b)^2 + (z/c)^2,

inside when :math:`q \le 1`. This is a *containment* constraint: the chaser must stay
inside, and every sample is checked. There is deliberately no activation range -- if a
mission concept only requires containment after an approach gate, slice the trajectory
before calling rather than asking this function to guess where the gate was.

**Approach corridor cone** of half-angle :math:`\alpha` about a unit axis
:math:`\hat{u}` expressed in the Hill frame (a V-bar approach from behind uses
:math:`\hat{u} = [0, -1, 0]`, pointing from the target toward the trailing chaser):

.. math:: \theta(t) = \arccos\!\left(
    \frac{\boldsymbol{\rho} \cdot \hat{u}}{\lVert\boldsymbol{\rho}\rVert}\right),

inside when :math:`\theta \le \alpha`. The *decision* is taken on
:math:`\cos\theta \ge \cos\alpha`, which is equivalent because cosine is strictly
decreasing on :math:`[0, \pi]` and is far better conditioned: ``arccos`` amplifies
relative error by :math:`1/\sin\theta`, so a point placed exactly on the boundary of a
10-degree cone comes back from the arccos round trip 2.2e-16 rad *outside* it. The angle
is still what gets reported, because radians are what an operator reads. Enforced **only**
while
:math:`\lVert\boldsymbol{\rho}\rVert \le d_\text{act}`: a corridor is a terminal-approach
concept and has no meaning at 50 km, so scoring far-field samples against it would
manufacture violations that no operator would accept.

**Closing velocity.** The range rate is the projection of relative velocity onto the
line of sight,

.. math:: \dot{r}(t) = \frac{\boldsymbol{\rho} \cdot \dot{\boldsymbol{\rho}}}
                            {\lVert\boldsymbol{\rho}\rVert},

and the closing velocity is :math:`v_c = -\dot{r}`, positive when approaching. Violated
when :math:`v_c > v_\text{max}` inside the activation range. Two consequences follow
directly from the projection and are what the tests check: a purely radial approach at
speed :math:`s` closes at exactly :math:`s`, and a circular relative orbit at constant
range closes at exactly zero, because :math:`\boldsymbol{\rho} \perp
\dot{\boldsymbol{\rho}}` there.

Discrete sampling under-reports violations
------------------------------------------
The true minimum of :math:`\lVert\boldsymbol{\rho}\rVert` almost never lands on a sample.
A coarse trajectory can therefore report positive clearance while the continuous path
actually breached the sphere, and no amount of care in the per-sample arithmetic fixes
that. This module does two things about it rather than pretending the sampled minimum is
the true one:

1. It refines the minimum locally by fitting a parabola in :math:`t` through the discrete
   minimum of the **squared** range and its two neighbours. Squared range is the right
   quantity: for constant relative velocity
   :math:`\lVert\boldsymbol{\rho}_0 + \dot{\boldsymbol{\rho}} t\rVert^2` is *exactly* a
   quadratic in :math:`t`, so the fit recovers the true closest approach to machine
   precision for a rectilinear pass, and is the leading-order local model otherwise.
2. It reports the sampled and refined minima **separately**
   (:attr:`KeepOutSphereResult.worst_value` versus
   :attr:`KeepOutSphereResult.refined_clearance_m`) and judges
   :attr:`ConstraintResult.satisfied` on the refined value. A keep-out result can
   therefore have ``n_violating_samples == 0`` and ``satisfied is False``. That is
   intended: it means the grid missed the breach.

Validity
--------
The refinement is **local**. It sharpens the neighbourhood of the sampled minimum and
cannot see a breach that occurs entirely between two widely spaced samples elsewhere on
the arc, nor a second, deeper minimum that the grid aliased away. Nothing here is a
substitute for sampling the trajectory finely enough that the parabola is a good local
model; the refined number is a correction, not a guarantee. If the discrete minimum falls
on the first or last sample there is no bracketing triple, no refinement is applied, and
:attr:`KeepOutSphereResult.refinement_applied` is ``False`` -- an endpoint minimum usually
means the trajectory was truncated while still approaching, which is a sampling problem,
not a geometry result.

The corridor angle and the range rate are both undefined at
:math:`\lVert\boldsymbol{\rho}\rVert = 0`. Rather than divide by zero this module defines
both as zero within :data:`DEFAULT_ZERO_RANGE_TOL_M` of the target and leaves that case to
the keep-out sphere, which is the constraint that actually has something to say about a
chaser sitting on the target.

Units are SI: metres, seconds, radians, metres per second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .exceptions import RpoCoreError

#: Separation below which the line-of-sight direction is treated as undefined, so the
#: corridor angle and the range rate are reported as zero instead of dividing by ~0. One
#: nanometre is far below any physically meaningful RPO separation.
DEFAULT_ZERO_RANGE_TOL_M: float = 1.0e-9

#: Smallest Euclidean norm accepted for a corridor axis before normalisation. A vector
#: this short carries no reliable direction.
DEFAULT_AXIS_NORM_TOL: float = 1.0e-12


class ConstraintDefinitionError(RpoCoreError, ValueError):
    """Raised when a constraint is specified with geometry that cannot be evaluated.

    A non-positive keep-out radius, a half-angle outside ``(0, pi)``, a zero-length
    corridor axis. These are definition errors, not trajectory errors: the constraint
    itself has no meaning, so no trajectory could be scored against it.
    """


class TrajectorySamplingError(RpoCoreError, ValueError):
    """Raised when the sampled trajectory is malformed.

    Empty, mismatched ``times_s``/``states_hill`` lengths, non-finite entries, or times
    that are not strictly increasing. A constraint report computed from a trajectory whose
    time base runs backwards is worse than no report at all, because "first violation
    time" would be meaningless while still looking like an answer.
    """


# --------------------------------------------------------------------------------------
# Constraint definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeepOutSphere:
    """A spherical exclusion zone centred on the target.

    Parameters
    ----------
    radius_m
        Sphere radius, metres. Must be finite and strictly positive.

    """

    radius_m: float

    def __post_init__(self) -> None:
        """Validate the radius at construction, where the offending value is in scope."""
        radius = float(self.radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ConstraintDefinitionError(
                f"radius_m must be a finite positive keep-out radius, got {self.radius_m!r}"
            )
        object.__setattr__(self, "radius_m", radius)


@dataclass(frozen=True)
class ApproachEllipsoid:
    """An ellipsoidal containment region with semi-axes along the Hill axes.

    Parameters
    ----------
    semi_axes_m
        ``(a, b, c)`` semi-axes along Hill x, y, z in metres. Each must be finite and
        strictly positive. Any three-element sequence is accepted and stored as a tuple
        of floats.

    """

    semi_axes_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate and canonicalise the semi-axes at construction."""
        try:
            axes = [float(value) for value in self.semi_axes_m]
        except (TypeError, ValueError) as exc:
            raise ConstraintDefinitionError(
                f"semi_axes_m must be three finite positive lengths, got {self.semi_axes_m!r}"
            ) from exc
        if len(axes) != 3:
            raise ConstraintDefinitionError(
                f"semi_axes_m must have three entries (a, b, c), got {len(axes)}: "
                f"{self.semi_axes_m!r}"
            )
        for name, value in zip("abc", axes, strict=True):
            if not math.isfinite(value) or value <= 0.0:
                raise ConstraintDefinitionError(
                    f"semi_axes_m semi-axis {name!r} must be finite and positive, got {value!r} "
                    f"(from {self.semi_axes_m!r})"
                )
        object.__setattr__(self, "semi_axes_m", (axes[0], axes[1], axes[2]))


@dataclass(frozen=True)
class ApproachCorridor:
    """A cone the chaser must stay inside once it is close enough to be on approach.

    Parameters
    ----------
    half_angle_rad
        Cone half-angle, radians. Must lie strictly inside ``(0, pi)``: zero is a line
        that no real trajectory lies on, and ``pi`` is all of space, so both endpoints
        describe a constraint that is not a constraint.
    activation_range_m
        The cone is enforced only while ``|rho| <= activation_range_m``. Must be positive;
        pass ``math.inf`` for a corridor that is always active. Required rather than
        defaulted, because an always-on corridor is almost never what a mission means and
        a silent default would hide that choice.
    axis_hill
        Cone axis in Hill coordinates, pointing from the target *toward* the chaser.
        Defaults to ``(0, -1, 0)``, the V-bar approach from behind. Normalised on
        construction, so any non-zero length is accepted and the stored value is the unit
        vector; a norm at or below :data:`DEFAULT_AXIS_NORM_TOL` raises.

    """

    half_angle_rad: float
    activation_range_m: float
    axis_hill: tuple[float, float, float] = (0.0, -1.0, 0.0)

    def __post_init__(self) -> None:
        """Validate the cone geometry and store the axis as a unit vector."""
        half_angle = float(self.half_angle_rad)
        if not math.isfinite(half_angle) or not (0.0 < half_angle < math.pi):
            raise ConstraintDefinitionError(
                f"half_angle_rad must lie strictly inside (0, pi) radians, got "
                f"{self.half_angle_rad!r}"
            )
        object.__setattr__(self, "half_angle_rad", half_angle)
        object.__setattr__(
            self, "activation_range_m", _validate_activation_range(self.activation_range_m)
        )
        object.__setattr__(self, "axis_hill", _unit_axis(self.axis_hill))


@dataclass(frozen=True)
class ClosingVelocityLimit:
    """An upper bound on the rate at which the chaser may close on the target.

    Parameters
    ----------
    max_closing_speed_m_s
        Maximum permitted closing velocity, m/s. Must be finite and non-negative. Zero
        is accepted and meaningful (no net approach permitted), but note that at exactly
        zero the comparison is decided at round-off level: a numerically circular relative
        orbit produces closing velocities of order 1e-16 m/s that will register as
        violations. Prefer a small positive limit if that matters.
    activation_range_m
        The limit is enforced only while ``|rho| <= activation_range_m``. Must be
        positive; pass ``math.inf`` to enforce everywhere.

    """

    max_closing_speed_m_s: float
    activation_range_m: float

    def __post_init__(self) -> None:
        """Validate the speed limit and activation range at construction."""
        limit = float(self.max_closing_speed_m_s)
        if not math.isfinite(limit) or limit < 0.0:
            raise ConstraintDefinitionError(
                "max_closing_speed_m_s must be a finite non-negative speed, got "
                f"{self.max_closing_speed_m_s!r}"
            )
        object.__setattr__(self, "max_closing_speed_m_s", limit)
        object.__setattr__(
            self, "activation_range_m", _validate_activation_range(self.activation_range_m)
        )


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintResult:
    """Outcome of one constraint evaluated over a whole sampled trajectory.

    Attributes
    ----------
    name
        Stable identifier of the constraint, e.g. ``"keep_out_sphere"``.
    satisfied
        ``True`` when no violation was found. For the keep-out sphere this is judged on
        the *refined* clearance, so it can be ``False`` while
        ``n_violating_samples == 0``; see :class:`KeepOutSphereResult`.
    worst_value
        Worst per-sample value of the constraint metric, in that constraint's own units:
        minimum clearance in metres, maximum ellipsoid quadratic form (dimensionless),
        maximum cone angle in radians, maximum closing velocity in m/s. ``nan`` when the
        constraint had no active samples at all.
    worst_time_s
        Time of ``worst_value``, seconds. ``nan`` when there were no active samples.
    worst_index
        Index of ``worst_value`` into the sampled trajectory. ``-1`` when there were no
        active samples.
    n_violating_samples
        Number of samples that violate the constraint. Counts *samples*, not intervals or
        excursions: a single continuous breach spanning four samples counts as four.
    first_violation_time_s
        Time of the earliest violating sample, or ``None`` if there is none.

    """

    name: str
    satisfied: bool
    worst_value: float
    worst_time_s: float
    worst_index: int
    n_violating_samples: int
    first_violation_time_s: float | None


@dataclass(frozen=True)
class KeepOutSphereResult(ConstraintResult):
    """Keep-out sphere outcome, carrying both the sampled and the refined minimum.

    ``worst_value`` is the minimum clearance *as sampled*; ``refined_clearance_m`` is the
    parabola-refined estimate of the continuous minimum, and ``satisfied`` is judged on
    the refined value. The two are reported separately on purpose so that a caller can see
    when the grid was too coarse to resolve the closest approach.

    Attributes
    ----------
    sampled_min_range_m
        ``|rho|`` at the discrete minimum, metres. Equals ``worst_value + radius_m``.
    refined_min_range_m
        Parabola-refined estimate of the continuous minimum of ``|rho|``, metres. Never
        larger than ``sampled_min_range_m``: the vertex of an upward parabola through
        three points lies at or below the middle point.
    refined_time_s
        Time of the refined minimum, seconds. Lies strictly between the neighbours of the
        sampled minimum when refinement was applied.
    refined_clearance_m
        ``refined_min_range_m - radius_m``.
    refinement_applied
        ``False`` when the discrete minimum sits on an endpoint (no bracketing triple) or
        the three points do not describe a strict minimum, in which case the refined
        fields simply repeat the sampled ones.

    """

    sampled_min_range_m: float
    refined_min_range_m: float
    refined_time_s: float
    refined_clearance_m: float
    refinement_applied: bool


@dataclass(frozen=True)
class SafetyReport:
    """Aggregate outcome of every constraint that was requested.

    Attributes
    ----------
    keep_out, ellipsoid, corridor, closing_velocity
        Per-constraint results, or ``None`` for a constraint that was not requested.
    total_violating_samples
        Sum of ``n_violating_samples`` across all evaluated constraints. A single sample
        that breaks three constraints contributes three.
    first_violation_time_s
        Earliest violating sample time across all constraints, or ``None`` if there is
        none. Note this is a *sampled* time: a keep-out breach detected only by refinement
        does not appear here, because it did not happen at a sample. Check
        :attr:`all_satisfied`, not this field, to decide whether the trajectory is safe.

    """

    keep_out: KeepOutSphereResult | None
    ellipsoid: ConstraintResult | None
    corridor: ConstraintResult | None
    closing_velocity: ConstraintResult | None
    total_violating_samples: int
    first_violation_time_s: float | None

    @property
    def results(self) -> tuple[ConstraintResult, ...]:
        """Return the constraints that were actually evaluated, in declaration order."""
        candidates = (self.keep_out, self.ellipsoid, self.corridor, self.closing_velocity)
        return tuple(result for result in candidates if result is not None)

    @property
    def all_satisfied(self) -> bool:
        """Return whether every evaluated constraint was satisfied."""
        return all(result.satisfied for result in self.results)


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------


def _validate_activation_range(activation_range_m: float) -> float:
    """Return a validated activation range, allowing ``+inf`` for always-on."""
    activation = float(activation_range_m)
    if math.isnan(activation) or activation <= 0.0:
        raise ConstraintDefinitionError(
            "activation_range_m must be a positive range (math.inf for always active), got "
            f"{activation_range_m!r}"
        )
    return activation


def _unit_axis(axis_hill: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return ``axis_hill`` normalised to unit length, raising if it has no direction."""
    try:
        components = [float(value) for value in axis_hill]
    except (TypeError, ValueError) as exc:
        raise ConstraintDefinitionError(
            f"axis_hill must be three finite components, got {axis_hill!r}"
        ) from exc
    if len(components) != 3:
        raise ConstraintDefinitionError(
            f"axis_hill must have three entries, got {len(components)}: {axis_hill!r}"
        )
    if not all(math.isfinite(value) for value in components):
        raise ConstraintDefinitionError(f"axis_hill must be finite, got {axis_hill!r}")
    norm = math.sqrt(sum(value * value for value in components))
    if norm <= DEFAULT_AXIS_NORM_TOL:
        raise ConstraintDefinitionError(
            f"axis_hill must have non-zero length to define a direction, got {axis_hill!r} "
            f"with norm {norm!r} <= {DEFAULT_AXIS_NORM_TOL!r}"
        )
    return (components[0] / norm, components[1] / norm, components[2] / norm)


def _validate_states(states_hill: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the state history as a validated (N, 6) float array."""
    states = np.asarray(states_hill, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6:
        raise TrajectorySamplingError(
            f"states_hill must have shape (N, 6) as [x, y, z, xdot, ydot, zdot], got "
            f"shape {states.shape}"
        )
    if states.shape[0] == 0:
        raise TrajectorySamplingError(
            "states_hill is empty: a trajectory needs at least one sample, got 0"
        )
    non_finite_states = np.flatnonzero(~np.all(np.isfinite(states), axis=1))
    if non_finite_states.size > 0:
        index = int(non_finite_states[0])
        raise TrajectorySamplingError(
            f"states_hill must be finite: first non-finite entry at row {index}, value "
            f"{states[index].tolist()!r}"
        )
    return states


def _validate_trajectory(
    times_s: npt.ArrayLike, states_hill: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the trajectory as validated float arrays, or raise TrajectorySamplingError."""
    times = np.asarray(times_s, dtype=np.float64)
    states = _validate_states(states_hill)

    if times.ndim != 1:
        raise TrajectorySamplingError(
            f"times_s must be one-dimensional with shape (N,), got shape {times.shape}"
        )
    if times.size == 0:
        raise TrajectorySamplingError(
            "times_s is empty: a trajectory needs at least one sample, got 0"
        )
    if times.size != states.shape[0]:
        raise TrajectorySamplingError(
            f"times_s and states_hill must have the same length, got {times.size} times and "
            f"{states.shape[0]} states"
        )

    non_finite_times = np.flatnonzero(~np.isfinite(times))
    if non_finite_times.size > 0:
        index = int(non_finite_times[0])
        raise TrajectorySamplingError(
            f"times_s must be finite: first non-finite entry at index {index}, value "
            f"{float(times[index])!r}"
        )
    if times.size > 1:
        steps = np.diff(times)
        bad_steps = np.flatnonzero(steps <= 0.0)
        if bad_steps.size > 0:
            index = int(bad_steps[0])
            raise TrajectorySamplingError(
                f"times_s must be strictly increasing: times_s[{index}]="
                f"{float(times[index])!r} is not less than times_s[{index + 1}]="
                f"{float(times[index + 1])!r} (step {float(steps[index])!r} s)"
            )

    return times, states


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


def separation_m(states_hill: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``|rho|`` at each sample, metres.

    Parameters
    ----------
    states_hill
        Shape (N, 6): ``[x, y, z, xdot, ydot, zdot]`` in the Hill frame, m and m/s.

    Returns
    -------
    numpy.ndarray
        Shape (N,), metres.

    Raises
    ------
    TrajectorySamplingError
        If ``states_hill`` is not (N, 6) or contains non-finite entries.

    """
    states = _validate_states(states_hill)
    result: npt.NDArray[np.float64] = np.linalg.norm(states[:, :3], axis=1)
    return result


def range_rate_m_s(
    states_hill: npt.ArrayLike, *, zero_range_tol_m: float = DEFAULT_ZERO_RANGE_TOL_M
) -> npt.NDArray[np.float64]:
    """Return ``d|rho|/dt = (rho . rhodot) / |rho|`` at each sample, m/s.

    Negative means closing. Where ``|rho| <= zero_range_tol_m`` the line of sight has no
    direction and the range rate is defined as zero rather than dividing by ~0; see the
    module docstring.

    Parameters
    ----------
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s.
    zero_range_tol_m
        Separation below which the range rate is defined as zero.

    Returns
    -------
    numpy.ndarray
        Shape (N,), m/s.

    Raises
    ------
    TrajectorySamplingError
        If ``states_hill`` is not (N, 6) or contains non-finite entries.

    """
    states = _validate_states(states_hill)
    ranges = np.linalg.norm(states[:, :3], axis=1)
    projection = np.sum(states[:, :3] * states[:, 3:], axis=1)
    safe = ranges > float(zero_range_tol_m)
    result: npt.NDArray[np.float64] = np.zeros_like(ranges)
    np.divide(projection, ranges, out=result, where=safe)
    return result


def _parabolic_vertex(
    s_prev: float, s_next: float, y_prev: float, y_mid: float, y_next: float
) -> tuple[float, float] | None:
    """Return the ``(offset, value)`` vertex of the parabola through three points.

    Abscissae are given as offsets from the middle sample, so ``s_prev < 0 < s_next``.
    Returns ``None`` when the three points do not describe a strict minimum, or when the
    vertex falls outside the bracket -- in either case there is nothing to refine and the
    caller must keep the sampled value.
    """
    slope_prev = (y_prev - y_mid) / s_prev
    slope_next = (y_next - y_mid) / s_next
    curvature = (slope_next - slope_prev) / (s_next - s_prev)
    if not math.isfinite(curvature) or curvature <= 0.0:
        return None
    slope = slope_prev - curvature * s_prev
    offset = -slope / (2.0 * curvature)
    if not math.isfinite(offset) or offset < s_prev or offset > s_next:
        return None
    return offset, curvature * offset * offset + slope * offset + y_mid


def _summarise(
    name: str,
    values: npt.NDArray[np.float64],
    active: npt.NDArray[np.bool_],
    violating: npt.NDArray[np.bool_],
    times: npt.NDArray[np.float64],
    *,
    worst_is_max: bool,
) -> ConstraintResult:
    """Reduce per-sample metric values and a violation mask to a :class:`ConstraintResult`.

    ``active`` selects the samples the constraint applies to. With no active samples the
    constraint is vacuously satisfied and the worst-case fields are ``nan``/``-1`` rather
    than a fabricated zero, so a caller can tell "never enforced" from "enforced and fine".
    """
    active_indices = np.flatnonzero(active)
    n_violating = int(np.count_nonzero(violating))
    violating_indices = np.flatnonzero(violating)
    first_violation = float(times[violating_indices[0]]) if n_violating > 0 else None

    if active_indices.size == 0:
        return ConstraintResult(
            name=name,
            satisfied=True,
            worst_value=math.nan,
            worst_time_s=math.nan,
            worst_index=-1,
            n_violating_samples=0,
            first_violation_time_s=None,
        )

    active_values = values[active_indices]
    offset = int(np.argmax(active_values) if worst_is_max else np.argmin(active_values))
    worst_index = int(active_indices[offset])
    return ConstraintResult(
        name=name,
        satisfied=n_violating == 0,
        worst_value=float(values[worst_index]),
        worst_time_s=float(times[worst_index]),
        worst_index=worst_index,
        n_violating_samples=n_violating,
        first_violation_time_s=first_violation,
    )


# --------------------------------------------------------------------------------------
# Individual constraint evaluations
# --------------------------------------------------------------------------------------


def evaluate_keep_out_sphere(
    times_s: npt.ArrayLike, states_hill: npt.ArrayLike, keep_out: KeepOutSphere
) -> KeepOutSphereResult:
    """Evaluate a spherical keep-out zone, with parabolic refinement of the minimum.

    Parameters
    ----------
    times_s
        Shape (N,) sample times, seconds, strictly increasing.
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s.
    keep_out
        The sphere to evaluate.

    Returns
    -------
    KeepOutSphereResult
        ``worst_value`` is the minimum *sampled* clearance in metres (negative inside the
        sphere), ``n_violating_samples`` counts samples with clearance ``< 0``, and
        ``satisfied`` is judged on the refined clearance -- so it can be ``False`` with
        zero violating samples when the grid stepped over the breach.

    Raises
    ------
    TrajectorySamplingError
        If the trajectory is empty, mismatched, non-finite, or not strictly increasing
        in time.

    """
    times, states = _validate_trajectory(times_s, states_hill)
    ranges = np.linalg.norm(states[:, :3], axis=1)
    clearance = ranges - keep_out.radius_m

    base = _summarise(
        "keep_out_sphere",
        clearance,
        np.ones(times.size, dtype=np.bool_),
        clearance < 0.0,
        times,
        worst_is_max=False,
    )

    index = base.worst_index
    sampled_min_range = float(ranges[index])
    refined_range = sampled_min_range
    refined_time = float(times[index])
    refinement_applied = False

    # Refine on the *squared* range: it is exactly quadratic in t for constant relative
    # velocity, so a rectilinear pass is recovered to machine precision.
    if 0 < index < times.size - 1:
        squared = ranges * ranges
        vertex = _parabolic_vertex(
            float(times[index - 1] - times[index]),
            float(times[index + 1] - times[index]),
            float(squared[index - 1]),
            float(squared[index]),
            float(squared[index + 1]),
        )
        if vertex is not None:
            offset, value = vertex
            refined_time = float(times[index]) + offset
            refined_range = math.sqrt(max(value, 0.0))
            refinement_applied = True

    refined_clearance = refined_range - keep_out.radius_m
    return KeepOutSphereResult(
        name=base.name,
        satisfied=min(base.worst_value, refined_clearance) >= 0.0,
        worst_value=base.worst_value,
        worst_time_s=base.worst_time_s,
        worst_index=base.worst_index,
        n_violating_samples=base.n_violating_samples,
        first_violation_time_s=base.first_violation_time_s,
        sampled_min_range_m=sampled_min_range,
        refined_min_range_m=refined_range,
        refined_time_s=refined_time,
        refined_clearance_m=refined_clearance,
        refinement_applied=refinement_applied,
    )


def evaluate_approach_ellipsoid(
    times_s: npt.ArrayLike, states_hill: npt.ArrayLike, ellipsoid: ApproachEllipsoid
) -> ConstraintResult:
    """Evaluate containment inside an approach ellipsoid aligned with the Hill axes.

    Parameters
    ----------
    times_s
        Shape (N,) sample times, seconds, strictly increasing.
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s.
    ellipsoid
        The containment region.

    Returns
    -------
    ConstraintResult
        ``worst_value`` is the maximum of the dimensionless quadratic form
        ``(x/a)^2 + (y/b)^2 + (z/c)^2``; values ``> 1`` are outside and count as
        violations, and exactly ``1`` is on the surface and is not a violation.

    Raises
    ------
    TrajectorySamplingError
        If the trajectory is malformed. See :func:`_validate_trajectory`.

    """
    times, states = _validate_trajectory(times_s, states_hill)
    axes = np.asarray(ellipsoid.semi_axes_m, dtype=np.float64)
    quadratic_form: npt.NDArray[np.float64] = np.sum((states[:, :3] / axes) ** 2, axis=1)
    return _summarise(
        "approach_ellipsoid",
        quadratic_form,
        np.ones(times.size, dtype=np.bool_),
        quadratic_form > 1.0,
        times,
        worst_is_max=True,
    )


def evaluate_approach_corridor(
    times_s: npt.ArrayLike,
    states_hill: npt.ArrayLike,
    corridor: ApproachCorridor,
    *,
    zero_range_tol_m: float = DEFAULT_ZERO_RANGE_TOL_M,
) -> ConstraintResult:
    """Evaluate the approach-corridor cone, enforced only inside the activation range.

    Parameters
    ----------
    times_s
        Shape (N,) sample times, seconds, strictly increasing.
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s.
    corridor
        Cone half-angle, activation range and unit axis.
    zero_range_tol_m
        Separation below which the cone angle is defined as zero; see the module
        docstring.

    Returns
    -------
    ConstraintResult
        ``worst_value`` is the maximum angle from the cone axis **over active samples
        only**, radians; angles greater than the half-angle are violations, and exactly
        the half-angle is on the boundary and is not (the comparison is made on cosines
        so that the boundary case survives round-off; see the module docstring). If no
        sample is inside the activation range the result is vacuously satisfied with
        ``nan`` worst-case fields and ``worst_index == -1``.

    Raises
    ------
    TrajectorySamplingError
        If the trajectory is malformed. See :func:`_validate_trajectory`.

    """
    times, states = _validate_trajectory(times_s, states_hill)
    axis = np.asarray(corridor.axis_hill, dtype=np.float64)
    ranges = np.linalg.norm(states[:, :3], axis=1)

    tol = float(zero_range_tol_m)
    directed = ranges > tol
    cosine = np.ones_like(ranges)
    np.divide(states[:, :3] @ axis, ranges, out=cosine, where=directed)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle: npt.NDArray[np.float64] = np.arccos(cosine)

    # Decide on the cosine, report the angle. The two are equivalent in exact arithmetic
    # because cos is strictly decreasing on [0, pi], but arccos has an infinite derivative
    # at its endpoint and amplifies relative error by 1/sin(theta) -- a factor of ~57 at a
    # 1 degree corridor and ~570 at 0.1 degrees, precisely where a tight corridor lives.
    # Deciding on the angle instead flags a point constructed exactly on the boundary as a
    # violation, by 2.2e-16 rad; see test_cone_boundary_point_is_not_a_violation.
    active = ranges <= corridor.activation_range_m
    violating = active & (cosine < math.cos(corridor.half_angle_rad))
    return _summarise("approach_corridor", angle, active, violating, times, worst_is_max=True)


def evaluate_closing_velocity(
    times_s: npt.ArrayLike,
    states_hill: npt.ArrayLike,
    limit: ClosingVelocityLimit,
    *,
    zero_range_tol_m: float = DEFAULT_ZERO_RANGE_TOL_M,
) -> ConstraintResult:
    """Evaluate the closing-velocity limit, enforced only inside the activation range.

    Parameters
    ----------
    times_s
        Shape (N,) sample times, seconds, strictly increasing.
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s.
    limit
        Maximum closing speed and activation range.
    zero_range_tol_m
        Separation below which the range rate is defined as zero.

    Returns
    -------
    ConstraintResult
        ``worst_value`` is the maximum closing velocity ``-(rho . rhodot)/|rho|`` **over
        active samples only**, m/s; positive means approaching, and values greater than
        the limit are violations. Vacuously satisfied with ``nan`` worst-case fields if
        no sample is inside the activation range.

    Raises
    ------
    TrajectorySamplingError
        If the trajectory is malformed. See :func:`_validate_trajectory`.

    """
    times, states = _validate_trajectory(times_s, states_hill)
    ranges = np.linalg.norm(states[:, :3], axis=1)
    closing = -range_rate_m_s(states, zero_range_tol_m=zero_range_tol_m)

    active = ranges <= limit.activation_range_m
    violating = active & (closing > limit.max_closing_speed_m_s)
    return _summarise("closing_velocity", closing, active, violating, times, worst_is_max=True)


# --------------------------------------------------------------------------------------
# Aggregate report
# --------------------------------------------------------------------------------------


def evaluate_constraints(
    times_s: npt.ArrayLike,
    states_hill: npt.ArrayLike,
    *,
    keep_out: KeepOutSphere | None = None,
    ellipsoid: ApproachEllipsoid | None = None,
    corridor: ApproachCorridor | None = None,
    closing_velocity: ClosingVelocityLimit | None = None,
    zero_range_tol_m: float = DEFAULT_ZERO_RANGE_TOL_M,
) -> SafetyReport:
    """Evaluate every requested constraint over one trajectory and aggregate the outcome.

    Parameters
    ----------
    times_s
        Shape (N,) sample times, seconds, strictly increasing.
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s.
    keep_out, ellipsoid, corridor, closing_velocity
        Constraints to evaluate. ``None`` skips one. At least one must be given.
    zero_range_tol_m
        Separation below which line-of-sight quantities are defined as zero.

    Returns
    -------
    SafetyReport
        Per-constraint results plus the total violating-sample count and the earliest
        violating sample time across all constraints.

    Raises
    ------
    ConstraintDefinitionError
        If no constraint was supplied. An empty report that answers "satisfied" is the
        plausible-looking wrong answer this package exists to avoid.
    TrajectorySamplingError
        If the trajectory is malformed. See :func:`_validate_trajectory`.

    """
    if keep_out is None and ellipsoid is None and corridor is None and closing_velocity is None:
        raise ConstraintDefinitionError(
            "evaluate_constraints requires at least one constraint; all of keep_out, "
            "ellipsoid, corridor and closing_velocity were None, which would report a "
            "vacuously safe trajectory"
        )

    # Validate once up front so a malformed trajectory fails before any partial work.
    _validate_trajectory(times_s, states_hill)

    keep_out_result = (
        None if keep_out is None else evaluate_keep_out_sphere(times_s, states_hill, keep_out)
    )
    ellipsoid_result = (
        None if ellipsoid is None else evaluate_approach_ellipsoid(times_s, states_hill, ellipsoid)
    )
    corridor_result = (
        None
        if corridor is None
        else evaluate_approach_corridor(
            times_s, states_hill, corridor, zero_range_tol_m=zero_range_tol_m
        )
    )
    closing_result = (
        None
        if closing_velocity is None
        else evaluate_closing_velocity(
            times_s, states_hill, closing_velocity, zero_range_tol_m=zero_range_tol_m
        )
    )

    results = tuple(
        result
        for result in (keep_out_result, ellipsoid_result, corridor_result, closing_result)
        if result is not None
    )
    total = sum(result.n_violating_samples for result in results)
    first_times = [
        result.first_violation_time_s
        for result in results
        if result.first_violation_time_s is not None
    ]

    return SafetyReport(
        keep_out=keep_out_result,
        ellipsoid=ellipsoid_result,
        corridor=corridor_result,
        closing_velocity=closing_result,
        total_violating_samples=total,
        first_violation_time_s=min(first_times) if first_times else None,
    )


__all__ = [
    "DEFAULT_AXIS_NORM_TOL",
    "DEFAULT_ZERO_RANGE_TOL_M",
    "ApproachCorridor",
    "ApproachEllipsoid",
    "ClosingVelocityLimit",
    "ConstraintDefinitionError",
    "ConstraintResult",
    "KeepOutSphere",
    "KeepOutSphereResult",
    "SafetyReport",
    "TrajectorySamplingError",
    "evaluate_approach_corridor",
    "evaluate_approach_ellipsoid",
    "evaluate_closing_velocity",
    "evaluate_constraints",
    "evaluate_keep_out_sphere",
    "range_rate_m_s",
    "separation_m",
]
