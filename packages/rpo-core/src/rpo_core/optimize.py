r"""The Δv-versus-time-of-flight trade, and the table that makes a comparison honest.

Why this module exists
----------------------
Two jobs, and they are the same job.

The first is the trade study SRS F-2.7 and F-7.3 ask for: Δv as a function of time of
flight, its minimum, and the non-dominated set. The second is SRS F-7.4 -- run all three
baselines on one scenario and put them in one table. They belong together because a Δv
number without a time next to it, and a time without a validity flag next to it, are both
ways of winning an argument rather than settling one.

The equations
-------------
**The curve.** For the Clohessy-Wiltshire two-impulse transfer the objective is
:math:`\lVert \Delta v_1 \rVert + \lVert \Delta v_2 \rVert` with both impulses from
:math:`v_0^+ = \Phi_{rv}^{-1}(\rho_f - \Phi_{rr}\rho_0)`. It is not smooth everywhere:
:math:`\Phi_{rv}` loses rank at two *different* families of transfer time, because in-plane
and cross-track motion decouple exactly (``docs/project1/math-model.md`` M4):

.. math::

    \det \Phi_{rv}^{(2\times2)} = \frac{8 - 8\cos\tau - 3\tau\sin\tau}{n^2}
      \;\to\; 0 \quad \text{at } \tau = 2\pi k \quad\text{(whole periods, in-plane)}

    \Phi_{rv}[2,2] = \frac{\sin\tau}{n}
      \;\to\; 0 \quad \text{at } \tau = \pi k \quad\text{(half periods, cross-track)}

**Whether the singularity bites is a property of the transfer, not only of the time.** At
:math:`\tau = 2\pi k` the in-plane block collapses to rank one with range
:math:`\operatorname{span}\{\hat{e}_y\}`, so a transfer whose position shortfall
:math:`\rho_f - \Phi_{rr}\rho_0` has no radial component is perfectly well posed there. That
is exactly the flagship V-bar hop: measured, its Δv at 0.9999999 periods is a sane
0.0896 m/s, while a transfer with radial and cross-track content asks for **529 374 m/s** at
the same time of flight. Excluding a fixed neighbourhood of :math:`kT` from every sweep
would therefore throw away good samples for one problem while keeping absurd ones for
another.

So the exclusion is measured per sample, by the guards :mod:`rpo_core.relative.cw` already
owns and already tests: the in-plane condition number ``max_condition`` and the cross-track
``cross_track_sin_tol``. Measured amplification for a 3-D transfer at a 420 km circular
target (initial ``[100, -1000, 50]`` m, terminal ``[-30, -250, -20]`` m, both at rest):

=================  ==================  =============
:math:`1 - t/T`    in-plane cond.      total Δv (m/s)
=================  ==================  =============
1.0e-01            3.25e+01            0.639
1.0e-02            3.01e+02            5.31
3.0e-03            1.00e+03            17.6
3.0e-04            1.00e+04            176
1.0e-05            3.00e+05            5 294
1.0e-07            3.00e+07            529 374
=================  ==================  =============

The condition number tracks the documented ``~3/(1 - t/T)`` law exactly, and the Δv tracks
it linearly. :data:`SWEEP_MAX_CONDITION` cuts at 1e4, i.e. at 3e-04 of a period and 176 m/s
of amplified impulse -- real physics, but three orders above the well-conditioned value and
not a design point anyone flies. :data:`SWEEP_CROSS_TRACK_SIN_TOL` cuts the other family at
a matched 215 m/s. Excluded samples are **recorded with their reason**, never dropped
silently and never replaced by ``nan``: a ``nan`` in a Δv column is one ``np.min`` away from
becoming a reported optimum.

**The minimum.** ``scipy.optimize.minimize_scalar(method="bounded")`` on each
singularity-free sub-interval, best over sub-intervals. Bounded Brent is used rather than a
gradient method because the objective is cheap, non-smooth at known points, and has no
analytic derivative worth writing. Non-convergence raises: the last iterate of a
non-converged bounded search is an arbitrary point of the bracket that looks exactly like an
answer.

**Dominance.** :math:`a` dominates :math:`b` iff :math:`a` is no worse on both objectives
and strictly better on at least one -- the standard weak-dominance relation, written out
here rather than imported so it can be tested against hand-built point sets. Under it,
identical points do not dominate each other; :func:`pareto_front` therefore dedupes
explicitly rather than returning a front with repeated entries.

Validity
--------
Nothing in this module decides validity; it reports what
:mod:`rpo_core.baselines` measured. :meth:`BaselineComparison.render_table` prints a model-
premise column for every row and the sentence behind each verdict, and it has **no "best"
column**: ranking by Δv while a method sits outside its own envelope is the misleading
benchmark this whole package exists to prevent.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from .baselines import (
    DEFAULT_CW_TOLERANCE_M,
    DEFAULT_DRIFT_REVOLUTIONS,
    BaselineResult,
    Method,
    PhasingGeometryError,
    RendezvousProblem,
    cw_two_impulse_baseline,
    lambert_baseline,
    phasing_baseline,
)
from .exceptions import InfeasibleTransferError, RpoCoreError, SingularTransferTimeError
from .frames import relative_state_hill_to_eci
from .lambert import LambertConvergenceError, solve_lambert
from .propagate import DEFAULT_ATOL, DEFAULT_RTOL, propagate_two_body
from .relative.cw import two_impulse_transfer

__all__ = [
    "DEFAULT_INTERVAL_GUARD_FRACTION",
    "DEFAULT_MINIMISER_MAX_ITERATIONS",
    "SWEEPABLE_METHODS",
    "SWEEP_CROSS_TRACK_SIN_TOL",
    "SWEEP_MAX_CONDITION",
    "BaselineComparison",
    "DeltaVConvergenceError",
    "DeltaVMinimum",
    "DeltaVSweep",
    "ExcludedTof",
    "NoRegularIntervalError",
    "OptimisationError",
    "compare_baselines",
    "delta_v_vs_tof",
    "dominates",
    "minimise_delta_v",
    "pareto_front",
    "phasing_delta_v_vs_tof",
    "regular_intervals_s",
    "singular_transfer_times_s",
]

#: In-plane condition-number ceiling for a Δv-versus-TOF sweep.
#:
#: Far tighter than :data:`rpo_core.relative.cw.SINGULARITY_CONDITION_LIMIT` (1e8), which is
#: an exact-singularity backstop and correctly lets 0.99999 periods through. A *sweep* needs
#: a different cut: it is looking for design points, and a sample whose impulse is amplified
#: 1e4-fold by proximity to a whole-period transfer time is not one. Measured on the 3-D
#: reference transfer in the module docstring, cond = 1e4 sits at 3.0e-04 of a period and
#: 176 m/s of Δv against 0.64 m/s for a well-conditioned time of flight. Pass
#: ``max_condition=SINGULARITY_CONDITION_LIMIT`` to recover the raw divergence.
SWEEP_MAX_CONDITION: float = 1.0e4

#: ``|sin(n * tof)|`` floor for the cross-track solve during a sweep.
#:
#: The cross-track amplification is exactly ``1/|sin(tau)|``; measured on the same transfer,
#: the largest Δv this cut admits is 215 m/s, against the 176 m/s the in-plane cut admits.
#: Chosen so the two families are excluded at comparable *amplification* rather than at
#: comparable *time*, since they diverge at different rates: 3.0e-04 of a period from a whole
#: period against 9.5e-05 of a half period from a half period.
SWEEP_CROSS_TRACK_SIN_TOL: float = 3.0e-04

#: Fraction of an orbital period trimmed off each end of a singularity-free sub-interval
#: before the minimiser is let loose on it.
#:
#: The bounded Brent search evaluates arbitrarily close to its bounds, so a bound placed
#: exactly on ``k*T/2`` would hand the objective a singular transfer time. 1e-03 of a period
#: (5.6 s in LEO) clears the widest sweep exclusion band -- 3.0e-04 of a period, from
#: :data:`SWEEP_MAX_CONDITION` -- with 3.3x headroom.
DEFAULT_INTERVAL_GUARD_FRACTION: float = 1.0e-03

#: Iteration cap handed to ``minimize_scalar``. SciPy's own default is 500; measured on the
#: reference sweeps, bounded Brent converges in 12-16 evaluations, so 100 is 6x headroom and
#: still turns a pathological objective into a raise rather than a hang.
DEFAULT_MINIMISER_MAX_ITERATIONS: int = 100

#: Methods whose time of flight is an *input* and can therefore be swept directly.
#:
#: The phasing baseline is absent on purpose: its time of flight is an output, fixed by the
#: phase it must deliver. Sweeping it means sweeping the drift revolution count, which is
#: what :func:`phasing_delta_v_vs_tof` does.
SWEEPABLE_METHODS: tuple[Method, ...] = (Method.CW_TWO_IMPULSE, Method.LAMBERT)


class OptimisationError(RpoCoreError):
    """Base class for failures in the Δv-versus-time-of-flight machinery."""


class DeltaVConvergenceError(OptimisationError, RuntimeError):
    """Raised when the bounded minimiser did not converge on a sub-interval.

    Carries the interval, the iteration count, and SciPy's own message. Deliberately not
    recoverable into a partial answer: the last iterate of a non-converged bounded Brent
    search is a point of the bracket chosen by where the iteration happened to stop, and it
    is indistinguishable from a genuine minimum by inspection. Returning it would put a
    number into a Δv budget that no one can audit.
    """

    def __init__(
        self, message: str, *, interval_s: tuple[float, float], iterations: int, detail: str
    ) -> None:
        """Record the sub-interval, the iteration count, and the solver's message."""
        super().__init__(message)
        self.interval_s = interval_s
        self.iterations = iterations
        self.detail = detail


class NoRegularIntervalError(OptimisationError, ValueError):
    """Raised when a requested search range contains no singularity-free sub-interval.

    Distinct from :class:`DeltaVConvergenceError`: the minimiser never ran, because every
    part of the range sits inside the guard band around a singular transfer time. Carries
    the singular times that consumed the range.
    """

    def __init__(
        self, message: str, *, singular_times_s: tuple[float, ...], guard_s: float
    ) -> None:
        """Record the singular times inside the range and the guard half-width, seconds."""
        super().__init__(message)
        self.singular_times_s = singular_times_s
        self.guard_s = guard_s


@dataclass(frozen=True)
class ExcludedTof:
    """One time of flight the sweep refused to report, and why.

    Attributes
    ----------
    tof_s
        The excluded time of flight, seconds. ``nan`` when the sweep parameter is not
        itself a time of flight -- :func:`phasing_delta_v_vs_tof` sweeps drift revolutions,
        and the profile that would have *produced* a time of flight is the one that failed,
        so there is no time to report and inventing one would be worse than ``nan``. The
        refused parameter value is named in ``reason``.
    reason
        Human-readable cause, carrying the numbers that motivated it -- the condition
        number, the ``|sin(tau)|``, or the solver's own message.

    """

    tof_s: float
    reason: str


@dataclass(frozen=True, eq=False)
class DeltaVSweep:
    """Total Δv against time of flight, with the excluded samples kept rather than hidden.

    Attributes
    ----------
    method
        Which baseline produced the curve.
    tof_s
        Times of flight that produced a usable Δv, seconds, ascending. Shape (N,).
    delta_v_m_s
        Total Δv at each of those times, m/s. Shape (N,). Guaranteed finite.
    excluded
        Samples that were refused, in the order they were requested. Non-empty is normal
        and informative, not a failure.

    """

    method: Method
    tof_s: npt.NDArray[np.float64]
    delta_v_m_s: npt.NDArray[np.float64]
    excluded: tuple[ExcludedTof, ...]

    @property
    def minimum(self) -> tuple[float, float]:
        """Return ``(tof_s, delta_v_m_s)`` at the cheapest retained sample.

        Raises
        ------
        ValueError
            If every sample was excluded, so there is no minimum to report. Returning
            ``(nan, nan)`` here is precisely the silent pollution this module exists to
            avoid.

        """
        if self.tof_s.size == 0:
            raise ValueError(
                f"sweep for method {self.method.value!r} retained no samples: all "
                f"{len(self.excluded)} times of flight were excluded, so it has no minimum"
            )
        index = int(np.argmin(self.delta_v_m_s))
        return float(self.tof_s[index]), float(self.delta_v_m_s[index])

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        """Return retained samples as ``(delta_v_m_s, tof_s)`` pairs for :func:`pareto_front`."""
        return tuple(
            (float(dv), float(t)) for dv, t in zip(self.delta_v_m_s, self.tof_s, strict=True)
        )


@dataclass(frozen=True)
class DeltaVMinimum:
    """A converged Δv minimum over a bounded interval.

    Attributes
    ----------
    tof_s
        Minimising time of flight, seconds.
    delta_v_m_s
        Total Δv there, m/s.
    interval_s
        The singularity-free sub-interval the minimum was found in. Reported because a
        range straddling a singular transfer time is searched piecewise, and knowing which
        piece won is the difference between a global minimum and a local one.
    function_evaluations
        Objective evaluations across all sub-intervals.
    sub_intervals
        How many singularity-free sub-intervals the range was split into.

    """

    tof_s: float
    delta_v_m_s: float
    interval_s: tuple[float, float]
    function_evaluations: int
    sub_intervals: int


def _positive_float(value: float, name: str) -> float:
    """Return ``value`` as a validated finite strictly-positive float."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}")
    return number


def _cw_delta_v(
    problem: RendezvousProblem,
    tof_s: float,
    *,
    max_condition: float,
    cross_track_sin_tol: float,
) -> float:
    """Total CW two-impulse Δv at one time of flight, m/s. Raises at singular times."""
    dv1, dv2 = two_impulse_transfer(
        problem.n_rad_s,
        problem.r0_hill_m,
        problem.v0_hill_m_s,
        problem.rf_hill_m,
        problem.vf_hill_m_s,
        tof_s,
        max_condition=max_condition,
        cross_track_sin_tol=cross_track_sin_tol,
    )
    return float(np.linalg.norm(dv1) + np.linalg.norm(dv2))


def _lambert_delta_v(
    problem: RendezvousProblem,
    tof_s: float,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Total Lambert Δv at one time of flight, m/s.

    Re-propagates the target for each time of flight rather than interpolating a stored
    trajectory: the arrival point moves with the target, and interpolating it would put an
    interpolation error into a Δv the sweep then reports as physics.
    """
    mu = problem.mu_m3_s2
    chaser0 = problem.chaser_state0_eci
    target_f = propagate_two_body(
        problem.target_state_eci, np.array([0.0, tof_s]), mu, rtol=rtol, atol=atol
    )[-1]
    r_chaser_f, v_chaser_f = relative_state_hill_to_eci(
        target_f[:3], target_f[3:], np.concatenate((problem.rf_hill_m, problem.vf_hill_m_s))
    )
    v_depart, v_arrive = solve_lambert(chaser0[:3], r_chaser_f, tof_s, mu)
    return float(np.linalg.norm(v_depart - chaser0[3:]) + np.linalg.norm(v_chaser_f - v_arrive))


def _delta_v_at(
    problem: RendezvousProblem,
    tof_s: float,
    method: Method,
    *,
    max_condition: float,
    cross_track_sin_tol: float,
) -> float:
    """Dispatch one Δv evaluation. Raises the underlying typed error at singular times."""
    if method is Method.CW_TWO_IMPULSE:
        return _cw_delta_v(
            problem,
            tof_s,
            max_condition=max_condition,
            cross_track_sin_tol=cross_track_sin_tol,
        )
    if method is Method.LAMBERT:
        return _lambert_delta_v(problem, tof_s)
    raise ValueError(
        f"method {method.value!r} has no time of flight to sweep; sweepable methods are "
        f"{[m.value for m in SWEEPABLE_METHODS]}. The phasing baseline's time of flight is "
        "an output -- use phasing_delta_v_vs_tof to sweep it."
    )


def delta_v_vs_tof(
    problem: RendezvousProblem,
    tof_values_s: npt.ArrayLike,
    *,
    method: Method = Method.CW_TWO_IMPULSE,
    max_condition: float = SWEEP_MAX_CONDITION,
    cross_track_sin_tol: float = SWEEP_CROSS_TRACK_SIN_TOL,
) -> DeltaVSweep:
    """Sweep time of flight and return total Δv at each, excluding what cannot be reported.

    A sample is excluded when the underlying solve refuses it -- a
    :class:`~rpo_core.exceptions.SingularTransferTimeError` from the in-plane block, an
    :class:`~rpo_core.exceptions.InfeasibleTransferError` from the cross-track block, or a
    :class:`~rpo_core.lambert.LambertConvergenceError`. Nothing is coerced to ``nan`` and
    nothing is dropped without a recorded reason; see :class:`ExcludedTof`.

    The guards are the ones :mod:`rpo_core.relative.cw` already owns, tightened for sweep
    use. That matters: whether a whole-period transfer time is singular depends on whether
    the transfer excites the rank-deficient direction, so a time-based exclusion rule would
    be wrong for one problem in exactly the cases it is right for another. See the module
    docstring for the measurement.

    Parameters
    ----------
    problem
        The scenario. ``problem.tof_s`` is ignored -- ``tof_values_s`` replaces it.
    tof_values_s
        Times of flight to evaluate, seconds. Must be 1-D, finite and strictly positive.
    method
        One of :data:`SWEEPABLE_METHODS`.
    max_condition, cross_track_sin_tol
        In-plane and cross-track guards, forwarded to
        :func:`~rpo_core.relative.cw.two_impulse_transfer`. Ignored for Lambert, which has
        no such block.

    Returns
    -------
    DeltaVSweep
        Retained samples, their Δv, and every exclusion with its reason.

    Raises
    ------
    ValueError
        If ``tof_values_s`` is not a finite, strictly positive 1-D array, or ``method`` is
        not sweepable.

    """
    values = np.asarray(tof_values_s, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"tof_values_s must be a non-empty 1-D array, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("tof_values_s must be finite")
    if np.any(values <= 0.0):
        raise ValueError(
            f"tof_values_s must be strictly positive, got minimum {float(values.min())!r}"
        )
    if method not in SWEEPABLE_METHODS:
        raise ValueError(
            f"method {method.value!r} has no time of flight to sweep; sweepable methods are "
            f"{[m.value for m in SWEEPABLE_METHODS]}. The phasing baseline's time of flight "
            "is an output -- use phasing_delta_v_vs_tof to sweep it."
        )

    kept_tof: list[float] = []
    kept_dv: list[float] = []
    excluded: list[ExcludedTof] = []
    for raw in values:
        tof = float(raw)
        try:
            delta_v = _delta_v_at(
                problem,
                tof,
                method,
                max_condition=max_condition,
                cross_track_sin_tol=cross_track_sin_tol,
            )
        except (
            SingularTransferTimeError,
            InfeasibleTransferError,
            LambertConvergenceError,
        ) as error:
            excluded.append(ExcludedTof(tof, f"{type(error).__name__}: {error}"))
            continue
        if not math.isfinite(delta_v):  # pragma: no cover - defence in depth
            excluded.append(
                ExcludedTof(tof, f"non-finite delta-v {delta_v!r} from an accepted solve")
            )
            continue
        kept_tof.append(tof)
        kept_dv.append(delta_v)

    order = np.argsort(np.asarray(kept_tof, dtype=np.float64), kind="stable")
    return DeltaVSweep(
        method=method,
        tof_s=np.asarray(kept_tof, dtype=np.float64)[order],
        delta_v_m_s=np.asarray(kept_dv, dtype=np.float64)[order],
        excluded=tuple(excluded),
    )


def phasing_delta_v_vs_tof(
    problem: RendezvousProblem,
    drift_revolutions_values: npt.ArrayLike,
    **phasing_kwargs: float,
) -> DeltaVSweep:
    """Sweep the phasing baseline's *drift revolutions* and report Δv against the TOF it buys.

    The phasing profile has no commanded time of flight to sweep: once the phase to deliver
    is fixed, the drift radius follows from the revolution count and the time of flight
    follows from both. Sweeping ``k`` is therefore the honest parameterisation of the same
    trade, and it is monotone -- more revolutions means a smaller radius offset, less Δv and
    more time -- which makes these points the cheap-and-slow end of a Pareto front that CW
    and Lambert cannot reach.

    Parameters
    ----------
    problem
        The scenario.
    drift_revolutions_values
        Revolution counts to evaluate. Need not be integers. Finite and non-negative.
    **phasing_kwargs
        Forwarded to :func:`~rpo_core.baselines.phasing_baseline`.

    Returns
    -------
    DeltaVSweep
        ``tof_s`` holds the *derived* time of flight of each solved profile, ascending.
        Revolution counts whose phase cannot be delivered are recorded as exclusions.

    Raises
    ------
    ValueError
        If ``drift_revolutions_values`` is not a finite, non-negative 1-D array.

    """
    values = np.asarray(drift_revolutions_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"drift_revolutions_values must be a non-empty 1-D array, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("drift_revolutions_values must be finite and non-negative")

    kept_tof: list[float] = []
    kept_dv: list[float] = []
    excluded: list[ExcludedTof] = []
    for raw in values:
        revolutions = float(raw)
        try:
            result = phasing_baseline(
                problem,
                drift_revolutions=revolutions,
                **phasing_kwargs,  # type: ignore[arg-type]
            )
        except PhasingGeometryError as error:
            excluded.append(ExcludedTof(math.nan, f"drift_revolutions={revolutions:g}: {error}"))
            continue
        kept_tof.append(result.tof_s)
        kept_dv.append(result.total_delta_v_m_s)

    order = np.argsort(np.asarray(kept_tof, dtype=np.float64), kind="stable")
    return DeltaVSweep(
        method=Method.PHASING,
        tof_s=np.asarray(kept_tof, dtype=np.float64)[order],
        delta_v_m_s=np.asarray(kept_dv, dtype=np.float64)[order],
        excluded=tuple(excluded),
    )


def singular_transfer_times_s(
    problem: RendezvousProblem, tof_lo_s: float, tof_hi_s: float
) -> tuple[float, ...]:
    """Return every CW singular transfer time inside ``[tof_lo_s, tof_hi_s]``, ascending.

    The union of the two families: whole periods ``k*T`` (in-plane rank loss) and half
    periods ``k*T/2`` (cross-track rank loss). Since the whole periods are the even members
    of the half-period family, the union is simply ``k*T/2`` for integer ``k >= 1``.

    These are the times at which the solve *can* be singular. Whether it *is* depends on
    whether the transfer excites the rank-deficient direction -- see the module docstring --
    which is why :func:`delta_v_vs_tof` tests each sample rather than excluding by time, and
    why :func:`regular_intervals_s` (which must place bounds, not test points) is
    deliberately conservative and splits at all of them.

    Raises
    ------
    ValueError
        If the bounds are not finite with ``0 < tof_lo_s < tof_hi_s``.

    """
    lo = _positive_float(tof_lo_s, "tof_lo_s")
    hi = _positive_float(tof_hi_s, "tof_hi_s")
    if hi <= lo:
        raise ValueError(f"tof_hi_s must exceed tof_lo_s, got {tof_lo_s!r} and {tof_hi_s!r}")
    half_period = 0.5 * problem.period_s
    first = max(1, math.ceil(lo / half_period))
    last = math.floor(hi / half_period)
    return tuple(k * half_period for k in range(first, last + 1))


def regular_intervals_s(
    problem: RendezvousProblem,
    tof_lo_s: float,
    tof_hi_s: float,
    *,
    guard_fraction: float = DEFAULT_INTERVAL_GUARD_FRACTION,
) -> tuple[tuple[float, float], ...]:
    """Split ``[tof_lo_s, tof_hi_s]`` into singularity-free sub-intervals, guard included.

    Each returned interval is trimmed by ``guard_fraction * period`` away from any singular
    transfer time so a bounded search cannot evaluate the objective on one. Intervals that
    the trimming collapses are dropped.

    Parameters
    ----------
    problem
        The scenario; supplies the orbital period.
    tof_lo_s, tof_hi_s
        Search range, seconds, with ``0 < lo < hi``.
    guard_fraction
        Trim half-width as a fraction of the orbital period. See
        :data:`DEFAULT_INTERVAL_GUARD_FRACTION`.

    Returns
    -------
    tuple
        ``(lo, hi)`` pairs, ascending and disjoint. Possibly empty.

    Raises
    ------
    ValueError
        On a malformed range or a non-positive ``guard_fraction``.

    """
    lo = _positive_float(tof_lo_s, "tof_lo_s")
    hi = _positive_float(tof_hi_s, "tof_hi_s")
    if hi <= lo:
        raise ValueError(f"tof_hi_s must exceed tof_lo_s, got {tof_lo_s!r} and {tof_hi_s!r}")
    guard = _positive_float(guard_fraction, "guard_fraction") * problem.period_s

    singular = singular_transfer_times_s(problem, lo, hi)
    edges = [lo, *singular, hi]
    intervals: list[tuple[float, float]] = []
    for index in range(len(edges) - 1):
        start = edges[index] + (guard if index > 0 else 0.0)
        stop = edges[index + 1] - (guard if index + 1 < len(edges) - 1 else 0.0)
        if stop > start:
            intervals.append((start, stop))
    return tuple(intervals)


def minimise_delta_v(
    problem: RendezvousProblem,
    tof_lo_s: float,
    tof_hi_s: float,
    *,
    method: Method = Method.CW_TWO_IMPULSE,
    guard_fraction: float = DEFAULT_INTERVAL_GUARD_FRACTION,
    xatol_s: float = 1.0e-3,
    max_iterations: int = DEFAULT_MINIMISER_MAX_ITERATIONS,
    max_condition: float = SWEEP_MAX_CONDITION,
    cross_track_sin_tol: float = SWEEP_CROSS_TRACK_SIN_TOL,
) -> DeltaVMinimum:
    """Find the Δv-optimal time of flight over a bounded range.

    The range is split at every CW singular transfer time (:func:`regular_intervals_s`) and
    ``scipy.optimize.minimize_scalar(method="bounded")`` is run on each piece, because a
    single bounded search across a pole would be minimising a function it cannot evaluate
    in the middle. The best sub-interval minimum is returned, and which one it was is
    reported.

    **Non-convergence raises.** If any sub-interval search reports failure, the whole call
    raises :class:`DeltaVConvergenceError` rather than falling back to the successful pieces
    or to the last iterate: a Δv budget assembled from a partly-converged optimisation is
    worse than no optimisation, because it looks the same.

    Parameters
    ----------
    problem
        The scenario. ``problem.tof_s`` is ignored.
    tof_lo_s, tof_hi_s
        Search range, seconds.
    method
        One of :data:`SWEEPABLE_METHODS`.
    guard_fraction
        Trim applied around each singular time; see :data:`DEFAULT_INTERVAL_GUARD_FRACTION`.
    xatol_s
        Absolute convergence tolerance on the time of flight, seconds. One millisecond
        against a 5578 s period is 1.8e-07 of an orbit; measured on the reference V-bar
        sweep the resulting Δv agrees with a 400-point brute-force grid to 1.8e-07 m/s.
    max_iterations
        Cap handed to SciPy. Exceeding it is a failure, not a result.
    max_condition, cross_track_sin_tol
        Guards forwarded to the CW solve.

    Returns
    -------
    DeltaVMinimum
        The converged minimum and where it was found.

    Raises
    ------
    NoRegularIntervalError
        If the guard bands consume the entire range.
    DeltaVConvergenceError
        If any sub-interval search fails to converge.
    ValueError
        On a malformed range, a non-sweepable method, or non-positive tolerances.
    SingularTransferTimeError, InfeasibleTransferError, LambertConvergenceError
        If the objective is singular *inside* a guarded interval. Propagated rather than
        absorbed: it means the guard model is wrong for this problem, which the caller
        needs to know.

    """
    lo = _positive_float(tof_lo_s, "tof_lo_s")
    hi = _positive_float(tof_hi_s, "tof_hi_s")
    if hi <= lo:
        raise ValueError(f"tof_hi_s must exceed tof_lo_s, got {tof_lo_s!r} and {tof_hi_s!r}")
    _positive_float(xatol_s, "xatol_s")
    if method not in SWEEPABLE_METHODS:
        raise ValueError(
            f"method {method.value!r} has no time of flight to optimise; optimisable methods "
            f"are {[m.value for m in SWEEPABLE_METHODS]}"
        )
    iteration_cap = int(max_iterations)
    if iteration_cap < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")

    intervals = regular_intervals_s(problem, lo, hi, guard_fraction=guard_fraction)
    if not intervals:
        singular = singular_transfer_times_s(problem, lo, hi)
        raise NoRegularIntervalError(
            f"no singularity-free sub-interval survives inside [{lo:.6g}, {hi:.6g}] s: "
            f"{len(singular)} singular transfer time(s) at {[round(t, 3) for t in singular]} s "
            f"with a guard half-width of {guard_fraction * problem.period_s:.6g} s consume "
            "the whole range. Widen the range or reduce guard_fraction.",
            singular_times_s=singular,
            guard_s=guard_fraction * problem.period_s,
        )

    def objective(tof: float) -> float:
        return _delta_v_at(
            problem,
            float(tof),
            method,
            max_condition=max_condition,
            cross_track_sin_tol=cross_track_sin_tol,
        )

    candidates: list[tuple[float, float, tuple[float, float]]] = []
    evaluations = 0
    for interval in intervals:
        result = minimize_scalar(
            objective,
            bounds=interval,
            method="bounded",
            options={"xatol": xatol_s, "maxiter": iteration_cap},
        )
        evaluations += int(result.nfev)
        if not bool(result.success):
            raise DeltaVConvergenceError(
                f"bounded minimisation of {method.value!r} Δv did not converge on "
                f"[{interval[0]:.6g}, {interval[1]:.6g}] s after {int(result.nit)} "
                f"iterations (cap {iteration_cap}): {result.message}. No best-effort iterate "
                "is returned -- the last iterate of a failed bounded search is a point of "
                "the bracket, not a minimum.",
                interval_s=interval,
                iterations=int(result.nit),
                detail=str(result.message),
            )
        candidates.append((float(result.fun), float(result.x), interval))

    best_delta_v, best_tof, best_interval = min(candidates)
    return DeltaVMinimum(
        tof_s=best_tof,
        delta_v_m_s=best_delta_v,
        interval_s=best_interval,
        function_evaluations=evaluations,
        sub_intervals=len(intervals),
    )


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return whether ``a`` Pareto-dominates ``b`` for ``(delta_v_m_s, tof_s)``, both minimised.

    ``a`` dominates ``b`` iff it is no worse on both objectives **and** strictly better on
    at least one. Written out rather than imported so it can be tested directly against
    hand-built point sets, including the two cases that separate weak from strict dominance:

    * A tie on one objective with a strict improvement on the other **is** dominance --
      ``(1.0, 100.0)`` dominates ``(1.0, 200.0)``.
    * A tie on both is **not** -- identical points do not dominate each other, which is why
      :func:`pareto_front` dedupes rather than relying on the relation to do it.

    Parameters
    ----------
    a, b
        ``(delta_v_m_s, tof_s)`` pairs. Both entries must be finite.

    Raises
    ------
    ValueError
        If either point is not a finite pair.

    """
    ax, ay = _point(a, "a")
    bx, by = _point(b, "b")
    return ax <= bx and ay <= by and (ax < bx or ay < by)


def _point(value: Sequence[float], name: str) -> tuple[float, float]:
    """Return ``value`` as a validated finite ``(delta_v, tof)`` pair."""
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"{name} must be a (delta_v_m_s, tof_s) pair, got {value!r}")
    first, second = float(items[0]), float(items[1])
    if not (math.isfinite(first) and math.isfinite(second)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return first, second


def pareto_front(
    points: Iterable[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Return the non-dominated ``(delta_v_m_s, tof_s)`` set, ascending in time of flight.

    Both objectives are minimised. Exact duplicates are collapsed to one entry: under
    :func:`dominates` identical points do not dominate each other, so leaving them in would
    return a "front" with repeated members, and a front is a set.

    Parameters
    ----------
    points
        Iterable of ``(delta_v_m_s, tof_s)`` pairs. May be empty.

    Returns
    -------
    tuple
        The non-dominated pairs, sorted by ``tof_s`` then ``delta_v_m_s``. Empty input
        gives an empty front; a single point is its own front.

    Raises
    ------
    ValueError
        If any point is not a finite pair of numbers.

    """
    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for index, raw in enumerate(points):
        point = _point(raw, f"points[{index}]")
        if point not in seen:
            seen.add(point)
            unique.append(point)

    front = [
        candidate
        for candidate in unique
        if not any(dominates(other, candidate) for other in unique if other != candidate)
    ]
    return tuple(sorted(front, key=lambda p: (p[1], p[0])))


@dataclass(frozen=True, eq=False)
class BaselineComparison:
    """All baselines on one scenario, plus the renderer that keeps the table honest.

    Attributes
    ----------
    problem
        The scenario every result was produced from -- one problem, three methods, which is
        the whole point.
    results
        One :class:`~rpo_core.baselines.BaselineResult` per method, in table order.
    cw_tolerance_m
        The position-error budget the CW envelope was judged against.

    """

    problem: RendezvousProblem
    results: tuple[BaselineResult, ...]
    cw_tolerance_m: float

    def by_method(self, method: Method) -> BaselineResult:
        """Return the result for ``method``.

        Raises
        ------
        KeyError
            If that method was not run for this comparison.

        """
        for result in self.results:
            if result.method is method:
                return result
        raise KeyError(
            f"{method.value!r} was not run for this comparison; available: "
            f"{[r.method.value for r in self.results]}"
        )

    @property
    def cheapest_valid(self) -> BaselineResult | None:
        """Return the lowest-Δv result **whose model premise holds**, or ``None``.

        Not a "best" column and deliberately not part of the rendered table: it is here so
        a caller who wants a single answer has to ask for one that has already been filtered
        by validity, rather than reading a ranking off a table that has no idea which rows
        can be trusted.
        """
        valid = [result for result in self.results if result.is_valid]
        if not valid:
            return None
        return min(valid, key=lambda result: result.total_delta_v_m_s)

    def render_table(self) -> str:
        """Render the comparison as a plain-text table suitable for a README.

        Every row carries its model-premise verdict and every verdict is spelled out
        underneath. There is no "best" column: a Δv ranking that ignores validity is the
        misleading benchmark this package exists to prevent, and a column is exactly how one
        gets published.
        """
        return _render(self)


def _format_error(value: float) -> str:
    """Format a terminal error so metres and nanometres are legible in one column."""
    if value == 0.0:
        return "0"
    return f"{value:.3e}"


_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("Method", "", 26),
    ("Burns", "", 6),
    ("Total dv", "(m/s)", 11),
    ("TOF", "(s)", 11),
    ("TOF", "(orbits)", 9),
    ("Term. pos", "err (m)", 11),
    ("Term. vel", "err (m/s)", 11),
    ("Min sep", "(m)", 10),
    ("Model", "premise", 8),
)


def _render(comparison: BaselineComparison) -> str:
    """Build the plain-text comparison table. See :meth:`BaselineComparison.render_table`."""
    problem = comparison.problem
    period_s = problem.period_s

    # The method column is left-aligned in the body, so its heading is too; every numeric
    # column is right-aligned so the decimal points line up under their own heading.
    def _head(index: int, text: str) -> str:
        width = _COLUMNS[index][2]
        return text.ljust(width) if index == 0 else text.rjust(width)

    header_1 = "  ".join(_head(i, name) for i, (name, _, _) in enumerate(_COLUMNS)).rstrip()
    header_2 = "  ".join(_head(i, unit) for i, (_, unit, _) in enumerate(_COLUMNS)).rstrip()
    rule = "-" * max(len(header_1), sum(width + 2 for _, _, width in _COLUMNS) - 2)

    lines = [
        "Rendezvous baseline comparison -- one problem, "
        f"{len(comparison.results)} methods, identical scoring",
        f"  Target      : r = {problem.orbit_radius_m:,.0f} m, "
        f"T = {period_s:,.1f} s, n = {problem.n_rad_s:.6e} rad/s",
        f"  Chaser      : {_vector(problem.r0_hill_m)} m  ->  "
        f"{_vector(problem.rf_hill_m)} m  (Hill frame)",
        f"  Separation  : {problem.separation_m:,.1f} m; "
        f"commanded TOF for fixed-time methods {problem.tof_s:,.1f} s "
        f"({problem.tof_s / period_s:.3f} orbits)",
        "  Scoring     : terminal errors measured under nonlinear two-body dynamics for "
        "every method",
        "",
        header_1,
        header_2,
        rule,
    ]

    for result in comparison.results:
        cells = (
            result.method.label.ljust(_COLUMNS[0][2]),
            f"{result.burn_count:d}".rjust(_COLUMNS[1][2]),
            f"{result.total_delta_v_m_s:.6f}".rjust(_COLUMNS[2][2]),
            f"{result.tof_s:,.1f}".rjust(_COLUMNS[3][2]),
            f"{result.tof_s / period_s:.3f}".rjust(_COLUMNS[4][2]),
            _format_error(result.terminal_position_error_m).rjust(_COLUMNS[5][2]),
            _format_error(result.terminal_velocity_error_m_s).rjust(_COLUMNS[6][2]),
            f"{result.min_separation_m:,.1f}".rjust(_COLUMNS[7][2]),
            result.validity.label.rjust(_COLUMNS[8][2]),
        )
        lines.append("  ".join(cells).rstrip())

    lines.append(rule)
    lines.append(
        f"Model premise, judged against a {comparison.cw_tolerance_m:,.3g} m position-error budget:"
    )
    for result in comparison.results:
        lines.append(f"  [{result.validity.label}] {result.method.label}:")
        lines.extend(f"      {line}" for line in _wrap(result.validity_detail, 92))
    lines.extend(
        [
            "",
            "Model premise is not accuracy. It says whether each method's own assumptions hold",
            "here; the terminal-error columns say how close it got. A method can be VALID and",
            "still miss by metres, or INVALID and still be a usable first guess.",
            "",
            'There is deliberately no "best" column. Ranking these rows by delta-v alone would',
            "recommend a method that is outside its own envelope, which is the failure mode this",
            "table exists to prevent.",
        ]
    )
    return "\n".join(lines)


def _vector(values: npt.NDArray[np.float64]) -> str:
    """Render a Hill-frame position as a compact bracketed triple."""
    return "[" + ", ".join(f"{float(v):,.1f}" for v in values) + "]"


def _wrap(text: str, width: int) -> list[str]:
    """Wrap ``text`` to ``width`` characters on whitespace, never mid-word."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def compare_baselines(
    problem: RendezvousProblem,
    *,
    cw_tolerance_m: float = DEFAULT_CW_TOLERANCE_M,
    drift_revolutions: float = DEFAULT_DRIFT_REVOLUTIONS,
    include_corrected: bool = True,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> BaselineComparison:
    """Run every baseline on one scenario and return a structured, renderable comparison.

    This is SRS F-7.4. All results come from the same :class:`RendezvousProblem` and are
    scored by the same nonlinear flight, so the columns are comparable by construction
    rather than by care.

    Parameters
    ----------
    problem
        The scenario.
    cw_tolerance_m
        Position-error budget the CW linearisation envelope is judged against. Defaults to
        the 5 m figure ``docs/cw_validity.md`` derives from the 200 m keep-out sphere.
    drift_revolutions
        Revolutions on the drift orbit for the phasing baseline.
    include_corrected
        Also run the CW solve with nonlinear differential correction. Adds a fourth row and
        roughly 30 ms.
    rtol, atol
        Integrator tolerances used for every scoring flight.

    Returns
    -------
    BaselineComparison
        Results in table order: phasing, Lambert, CW, and optionally corrected CW.

    Raises
    ------
    PhasingGeometryError
        If the phasing profile cannot deliver the required phase.
    LambertConvergenceError, DegenerateGeometryError
        From the Lambert solve.
    SingularTransferTimeError, InfeasibleTransferError
        If ``problem.tof_s`` is a singular CW transfer time for this transfer.

    """
    results = [
        phasing_baseline(
            problem,
            drift_revolutions=drift_revolutions,
            cw_tolerance_m=cw_tolerance_m,
            rtol=rtol,
            atol=atol,
        ),
        lambert_baseline(problem, cw_tolerance_m=cw_tolerance_m, rtol=rtol, atol=atol),
        cw_two_impulse_baseline(problem, cw_tolerance_m=cw_tolerance_m, rtol=rtol, atol=atol),
    ]
    if include_corrected:
        results.append(
            cw_two_impulse_baseline(
                problem, correct=True, cw_tolerance_m=cw_tolerance_m, rtol=rtol, atol=atol
            )
        )
    return BaselineComparison(
        problem=problem, results=tuple(results), cw_tolerance_m=float(cw_tolerance_m)
    )
