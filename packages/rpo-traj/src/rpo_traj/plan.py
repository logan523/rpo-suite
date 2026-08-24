r"""The mission workflow: a validated scenario in, a complete run directory out.

This module is the composition layer. It owns no numerics of its own -- every quantity it
reports is produced by :mod:`rpo_core` -- and its whole job is to wire those pieces together
in the one order that makes the result defensible, then hand back an object. It never
prints: presentation is :mod:`rpo_traj.cli`'s problem, and a library that writes to stdout
cannot be called twice in a loop without making a mess.

The equations
-------------
The scenario fixes a circular target orbit of radius :math:`a = R_\oplus + h` and mean
motion :math:`n = \sqrt{\mu/a^3}`, two Hill-frame hold points
:math:`\boldsymbol{\rho}_0, \boldsymbol{\rho}_f` (both at rest in the rotating frame), and a
time of flight :math:`t_f`.

**Step 1, the linear solve.** :func:`~rpo_core.relative.cw.two_impulse_transfer` inverts the
Clohessy-Wiltshire position-from-velocity block to give the impulse pair

.. math::

    \Delta v_1 = \Phi_{rv}^{-1}\!\left(\boldsymbol{\rho}_f
                 - \Phi_{rr}\boldsymbol{\rho}_0\right) - \dot{\boldsymbol{\rho}}_0,
    \qquad
    \Delta v_2 = \dot{\boldsymbol{\rho}}_f
                 - \left(\Phi_{vr}\boldsymbol{\rho}_0 + \Phi_{vv}\, v_0^+\right).

For the flagship half-period V-bar hop this collapses to two equal purely radial impulses of
:math:`n\,\Delta y/4` each, so :math:`\Delta v_\text{total} = n\,\Delta y / 2`
(``docs/project1/math-model.md``, model M4). That closed form is what ``test_plan.py``
checks, and it holds to the last bit.

**Step 2, the correction.** :math:`\Delta v_1` is exact for CW and CW is wrong by a measured
amount. Flown through the nonlinear reference of
:func:`~rpo_core.relative.nonlinear.propagate_relative_nonlinear`, the baseline's raw CW
impulse misses its arrival point by **1.225271 m**;
:func:`~rpo_core.targeting.correct_two_impulse_transfer` shoots on :math:`\Delta v_1` until
the *nonlinear* arrival lands, reaching 5.52e-07 m in one Newton step -- an improvement of
2.2e+06. Both behaviours ship: ``correct=False`` is not a degraded mode but the measurement
that says how much the correction was worth.

**Step 3, the trajectory.** Whichever impulse pair was chosen, the arc is sampled under the
**nonlinear** dynamics, never under CW. Evaluating safety constraints on the same linear
model that designed the burn would be marking your own homework: the corridor breach and
the closing-velocity breach would be reported from a trajectory that is known to be a metre
off the one that would actually be flown.

Sample density
--------------
:data:`DEFAULT_SAMPLE_COUNT` is 2001, chosen by measurement (uniform spacing 1.394 s over
the baseline's 2789.111 s hop). Two quantities set it, and they fail in different ways:

**Minimum keep-out clearance -- a resolution threshold.** Under the *uncorrected* CW impulse
the arc's closest approach is not its endpoint. It dips 1.074e-03 m below the terminal range
1.81 s before arrival, a minimum only 3.6 s wide. Sampling coarser than that misses it
entirely and reports the endpoint, high by 1.074e-03 m; sub-sample refinement cannot help,
because with the discrete minimum sitting on an endpoint there is no bracketing triple to
refine. Measured, the flip is sharp: N = 760 (spacing 3.675 s) reports 51.151194 m with
``refinement_applied=False``, N = 780 (spacing 3.580 s) reports 51.150120 m with refinement
applied. 2001 samples put 2.6 samples inside the window, a 2.6x margin on spacing, and the
residual error is 1.4e-10 m against an N = 40001 reference.

**Maximum closing velocity -- a supremum at an activation boundary, which never fully
converges.** Closing velocity decreases monotonically through the final approach, so the
limit is not enforced at a peak: it is enforced from the moment the range crosses
``max_closing_velocity_activation_range_m``, and the reported maximum is simply the first
sample inside that range. The exact supremum is the value at the crossing itself
(0.2312233 m/s corrected, 0.2288959 m/s raw, both found by bisection), and a uniform grid
approaches it from below at :math:`O(1/N)` -- never at it, and not monotonically, since the
error depends on where the nearest grid point happens to fall. The error is bounded by the
sample spacing times the slope there, measured at 4.4e-04 m/s per second:

.. math:: \varepsilon \;\lesssim\; 4.4\times10^{-4} \cdot \frac{t_f}{N-1}
          \;=\; \frac{1.23}{N-1}\ \text{m/s}.

At N = 2001 that bound is 6.1e-04 m/s and the measured error is 5.2e-04 m/s -- **0.5 % of
the 0.1 m/s limit the number is judged against**, which is the criterion the count was
chosen to meet. Buying another decade costs a factor of ten in samples for a change no
reader would act on.

Measured convergence, uncorrected baseline, against an N = 40001 reference (keep-out) and
the bisected crossing value (closing velocity):

======  ==========================  =========================
``N``   keep-out clearance err (m)  closing-velocity err (m/s)
======  ==========================  =========================
101     1.074e-03                   -5.271e-03
241     1.074e-03                   -3.184e-03
751     1.074e-03                   -2.883e-04
1009    6.790e-11                   -4.746e-04
2003    -1.965e-10                  -4.660e-04
9001    9.521e-11                   -1.411e-05
======  ==========================  =========================

The keep-out column is a step, not a taper: that is the signature of a threshold, and the
reason a convergence claim here has to be made on a non-nested sweep. Halving a grid that
already resolves the dip changes nothing, which reads as convergence whether or not the
coarse grid ever found the feature.

Validity
--------
Everything downstream of :mod:`rpo_core.relative.nonlinear` is two-body: no J2, no drag, no
third bodies, no solar radiation pressure. Impulses are instantaneous, so no finite-burn
attitude or gravity-loss modelling applies. What ``correct=True`` removes is CW's
linearisation error and nothing else -- it does not make the trajectory more physical than
the propagator underneath it.

The target's orbital plane is placed with the epoch radius along inertial :math:`+x` and the
velocity inclined out of the equator by ``inclination_deg``. In a spherical field that
choice is arbitrary: relative motion depends on the orbit's size and shape, not its
orientation, and ``test_plan.py`` pins that invariance rather than assuming it. Inclination
therefore does **not** influence any number this module reports, and will stop being inert
the moment J2 arrives (F-1.6).

``seed`` is recorded and propagated but changes no number here, because nothing in a
two-impulse plan is stochastic. It names the run directory and lands in ``provenance.json``
so that the Monte Carlo campaigns of SRS 2.5 slot into the same identity scheme rather than
inventing a second one.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rpo_core.config import (
    DEFAULT_RUNS_DIR,
    OrbitConfig,
    ScenarioConfig,
    create_run_directory,
)
from rpo_core.constants import MU_EARTH_M3_S2
from rpo_core.constraints import (
    ApproachCorridor,
    ApproachEllipsoid,
    ClosingVelocityLimit,
    KeepOutSphere,
    SafetyReport,
    evaluate_constraints,
)
from rpo_core.exceptions import RpoCoreError
from rpo_core.metrics import Burn, TrajectoryMetrics, compute_metrics, write_metrics
from rpo_core.relative.cw import two_impulse_transfer
from rpo_core.relative.nonlinear import propagate_relative_nonlinear
from rpo_core.targeting import DEFAULT_TOLERANCE_M, correct_two_impulse_transfer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rpo_core.targeting import CorrectedTransfer

__all__ = [
    "CORRIDOR_AXIS_HILL",
    "DEFAULT_SAMPLE_COUNT",
    "MIN_SAMPLE_COUNT",
    "PlanningError",
    "RendezvousPlan",
    "plan_rendezvous",
    "target_state_eci",
]


class PlanningError(RpoCoreError, ValueError):
    """Raised when a planning request is malformed before any numerics are attempted.

    Deliberately narrow. Failures *inside* the numerics -- a singular transfer time, a
    diverging differential correction, an integrator that gives up -- keep their own typed
    exceptions from :mod:`rpo_core.exceptions` and :mod:`rpo_core.targeting` and travel out
    of :func:`plan_rendezvous` untouched. Rewrapping them here would replace a message that
    names the condition number, the residual history, or the failing step with one that says
    "planning failed", which is the diagnosis thrown away rather than reported.
    """


#: Trajectory samples per plan, uniform on ``[0, tof_s]``.
#:
#: Measured, not chosen for roundness. See the module docstring's "Sample density" section
#: for the sweep: 2001 resolves the 3.6 s keep-out dip that a coarser grid steps over
#: entirely (threshold measured at spacing 3.58 s, i.e. N = 780, so this carries 2.6x
#: margin), and holds the closing-velocity discretisation error to 5.2e-04 m/s, which is
#: 0.5 % of the 0.1 m/s limit that number is judged against.
DEFAULT_SAMPLE_COUNT: int = 2001

#: Fewest samples :func:`plan_rendezvous` will accept.
#:
#: Three, not two. :mod:`rpo_core.metrics` needs two to have a time span at all, but the
#: keep-out sub-sample refinement needs a bracketing *triple* around the discrete minimum
#: (F-4.6), and with two samples every minimum is an endpoint. Two samples would therefore
#: produce a report whose refined minimum silently equals its sampled one for a structural
#: reason rather than a physical one.
MIN_SAMPLE_COUNT: int = 3

#: Approach-corridor axis in Hill coordinates: from the target *toward* a trailing chaser.
#:
#: ``(0, -1, 0)`` is V-bar-from-behind, the geometry every scenario in ``configs/`` uses and
#: the default :class:`~rpo_core.constraints.ApproachCorridor` carries. It is named here
#: rather than defaulted implicitly because the same value has to reach two places that do
#: not talk to each other -- the corridor that decides violations and the metrics record
#: that draws the wedge -- and a corridor evaluated about one axis and drawn about another
#: is a figure that disagrees with its own JSON. Scenario-configurable approach geometry is
#: F-3.4, and is not in the MVP.
CORRIDOR_AXIS_HILL: tuple[float, float, float] = (0.0, -1.0, 0.0)

_DEPARTURE_BURN_LABEL = "depart"
_ARRIVAL_BURN_LABEL = "arrive"


@dataclass(frozen=True, eq=False)
class RendezvousPlan:
    """Everything one planning run produced, in memory and on disk.

    ``eq=False`` because two of the fields are numpy arrays: a generated ``__eq__`` would
    return an array from an ``==`` comparison and raise on the truth test, which is a
    confusing failure a long way from its cause.

    Attributes
    ----------
    config
        The validated scenario that was planned.
    seed
        Seed this run was recorded under. Equal to ``config.seed`` unless overridden.
    corrected
        Whether differential correction was applied. When ``False`` the plan flies the raw
        Clohessy-Wiltshire impulses through the nonlinear dynamics, which is the honest
        measurement of the linearisation error rather than a broken run.
    n_samples
        Number of trajectory samples. See :data:`DEFAULT_SAMPLE_COUNT`.
    times_s
        Shape (N,), seconds from the departure impulse, uniform on ``[0, tof_s]``.
    states_hill
        Shape (N, 6) Hill-frame states, m and m/s. The **final sample carries the arrival
        impulse**, which is what makes ``terminal_velocity_error_m_s`` a terminal error and
        not the pre-burn coast velocity.
    burns
        Departure and arrival impulses, in time order.
    report
        The constraint outcome for exactly the trajectory in ``states_hill``.
    metrics
        The full metrics record, identical to the contents of ``metrics_path``.
    run_dir, metrics_path
        The content-addressed run directory and the ``metrics.json`` inside it.
    figure_paths
        Figure name to written path, keyed as in ``rpo_core.plotting.FIGURE_FILENAMES``.
        Empty when plots were not requested or matplotlib is absent.
    plots_skipped_reason
        Why ``figure_paths`` is empty, or ``None`` when figures were written. Carried rather
        than logged so a caller can decide whether a silent skip is acceptable.
    raw_cw_terminal_miss_m
        Terminal position miss of the **uncorrected** CW impulse under nonlinear dynamics,
        metres. Populated in both modes: when correcting it is the shooting solver's own
        first residual, and when not it is what the run achieved. It is the baseline the
        correction is measured against, and it costs nothing extra either way.
    corrected_terminal_miss_m
        Terminal position miss actually achieved, metres, or ``None`` when ``corrected`` is
        false. ``raw_cw_terminal_miss_m / corrected_terminal_miss_m`` is the improvement.
    targeting_iterations
        Accepted Newton steps, or ``None`` when ``corrected`` is false.
    cw_total_delta_v_m_s
        Delta-v budget of the uncorrected CW solution, m/s. Reported next to the achieved
        budget so the correction's cost is visible: it is not free, merely cheap.

    """

    config: ScenarioConfig
    seed: int
    corrected: bool
    n_samples: int
    times_s: npt.NDArray[np.float64]
    states_hill: npt.NDArray[np.float64]
    burns: tuple[Burn, ...]
    report: SafetyReport
    metrics: TrajectoryMetrics
    run_dir: Path
    metrics_path: Path
    raw_cw_terminal_miss_m: float
    corrected_terminal_miss_m: float | None
    targeting_iterations: int | None
    cw_total_delta_v_m_s: float
    figure_paths: Mapping[str, Path] = field(default_factory=dict)
    plots_skipped_reason: str | None = None

    @property
    def all_constraints_satisfied(self) -> bool:
        """Return whether every evaluated constraint was satisfied.

        Read off :attr:`metrics`, not :attr:`report`, on purpose: this is the value the CLI
        turns into an exit code, and it must be the value that was written to disk. If the
        two ever disagreed, the record would be the thing a reader has and the exit code
        would be the thing they trusted.
        """
        return self.metrics.all_constraints_satisfied

    @property
    def violation_count(self) -> int:
        """Return the total number of violating samples across all constraints."""
        return self.metrics.constraint_violation_count


def target_state_eci(
    orbit: OrbitConfig, mu_m3_s2: float = MU_EARTH_M3_S2
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Return the target's inertial state at the epoch for a circular orbit configuration.

    The epoch radius is placed along inertial :math:`+x` and the velocity is inclined out of
    the equatorial plane by ``inclination_deg``:

    .. math::

        oldsymbol{r} = (a, 0, 0), \qquad
        oldsymbol{v} = \sqrt{\mu/a}\,(0, \cos i, \sin i).

    The right ascension and argument of latitude are both taken as zero. In a spherical
    gravity field that is not an approximation but a choice of coordinates: relative motion
    depends on the orbit's size and shape, never on where in inertial space it is pointed.
    ``test_plan.py`` measures that invariance instead of asserting it.

    Parameters
    ----------
    orbit
        Validated circular-orbit configuration.
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Defaults to Earth.

    Returns
    -------
    tuple of numpy.ndarray
        ``(r_eci_m, v_eci_m_s)``, shape (3,) each, metres and metres per second.

    Raises
    ------
    PlanningError
        If ``mu_m3_s2`` is not finite and strictly positive.

    """
    mu = float(mu_m3_s2)
    if not math.isfinite(mu) or mu <= 0.0:
        raise PlanningError(
            f"mu_m3_s2 must be a finite positive gravitational parameter, got {mu_m3_s2!r}"
        )
    radius_m = orbit.semi_major_axis_m
    speed_m_s = math.sqrt(mu / radius_m)
    inclination_rad = math.radians(orbit.inclination_deg)
    r_eci_m = np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    v_eci_m_s = speed_m_s * np.array(
        [0.0, math.cos(inclination_rad), math.sin(inclination_rad)], dtype=np.float64
    )
    return r_eci_m, v_eci_m_s


def _validate_sample_count(n_samples: int) -> int:
    """Return ``n_samples`` as a validated integer at or above :data:`MIN_SAMPLE_COUNT`."""
    try:
        count = int(n_samples)
    except (TypeError, ValueError) as exc:
        raise PlanningError(
            f"n_samples must be an integer number of trajectory samples, got {n_samples!r}"
        ) from exc
    if count != n_samples:
        raise PlanningError(
            f"n_samples must be a whole number of samples, got {n_samples!r}; a fractional "
            "sample count silently truncates and changes the reported extrema"
        )
    if count < MIN_SAMPLE_COUNT:
        raise PlanningError(
            f"n_samples={count} is below MIN_SAMPLE_COUNT={MIN_SAMPLE_COUNT}. Sub-sample "
            "refinement of the closest approach needs a bracketing triple, so with fewer "
            "than three samples the refined keep-out minimum would equal the sampled one "
            "for a structural reason and the report would understate a breach (F-4.6). See "
            f"DEFAULT_SAMPLE_COUNT={DEFAULT_SAMPLE_COUNT} for the measured working value."
        )
    return count


def _validate_seed(seed: int | None, config: ScenarioConfig) -> int:
    """Return the effective run seed, rejecting a negative override early."""
    if seed is None:
        return config.seed
    try:
        effective = int(seed)
    except (TypeError, ValueError) as exc:
        raise PlanningError(f"seed must be a non-negative integer, got {seed!r}") from exc
    if effective < 0:
        raise PlanningError(
            f"seed must be non-negative, got {effective!r}. The seed names the run directory "
            "and is recorded in provenance.json; rejecting it here rather than after the "
            "solve keeps a typo from costing a propagation."
        )
    return effective


def _solve_impulses(
    config: ScenarioConfig,
    r_target_eci_m: npt.NDArray[np.float64],
    v_target_eci_m_s: npt.NDArray[np.float64],
    *,
    correct: bool,
    targeting_tolerance_m: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    CorrectedTransfer | None,
]:
    """Return ``(dv1, dv2, cw_dv1, cw_dv2, correction)`` for the scenario.

    The CW pair is always solved, whether or not it is the pair that flies: it is the
    baseline the correction is quoted against, and it is the same solve the corrector would
    seed itself from, so computing it here costs nothing.
    """
    r0_hill_m = np.asarray(config.start_hold_point.position_hill_m, dtype=np.float64)
    rf_hill_m = np.asarray(config.target_hold_point.position_hill_m, dtype=np.float64)
    v0_hill_m_s = np.zeros(3, dtype=np.float64)
    vf_hill_m_s = np.zeros(3, dtype=np.float64)
    tof_s = config.tof_s
    n_rad_s = config.orbit.mean_motion_rad_s

    cw_dv1, cw_dv2 = two_impulse_transfer(
        n_rad_s, r0_hill_m, v0_hill_m_s, rf_hill_m, vf_hill_m_s, tof_s
    )
    if not correct:
        return cw_dv1, cw_dv2, cw_dv1, cw_dv2, None

    correction = correct_two_impulse_transfer(
        r_target_eci_m,
        v_target_eci_m_s,
        r0_hill_m,
        v0_hill_m_s,
        rf_hill_m,
        vf_hill_m_s,
        tof_s,
        n_rad_s=n_rad_s,
        tolerance_m=targeting_tolerance_m,
        rtol=config.integrator.rtol,
        atol=config.integrator.atol,
    )
    return (
        correction.dv1_hill_m_s,
        correction.dv2_hill_m_s,
        correction.cw_dv1_hill_m_s,
        correction.cw_dv2_hill_m_s,
        correction,
    )


def _sample_trajectory(
    config: ScenarioConfig,
    r_target_eci_m: npt.NDArray[np.float64],
    v_target_eci_m_s: npt.NDArray[np.float64],
    dv1_hill_m_s: npt.NDArray[np.float64],
    dv2_hill_m_s: npt.NDArray[np.float64],
    n_samples: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(times_s, states_hill)`` for the coast, with the arrival impulse applied.

    The arrival impulse is added to the **final sample only**. Every sample before it is a
    coast state, which is what the constraints must see; the last one has to carry the burn
    or the terminal velocity error reports the pre-burn closing velocity and looks like a
    manoeuvre that failed to stop (see the :mod:`rpo_core.metrics` module docstring).
    """
    r0_hill_m = np.asarray(config.start_hold_point.position_hill_m, dtype=np.float64)
    times_s = np.linspace(0.0, config.tof_s, n_samples, dtype=np.float64)
    initial_state_hill = np.concatenate((r0_hill_m, dv1_hill_m_s))
    states_hill = propagate_relative_nonlinear(
        r_target_eci_m,
        v_target_eci_m_s,
        initial_state_hill,
        times_s,
        rtol=config.integrator.rtol,
        atol=config.integrator.atol,
    )
    states_hill[-1, 3:] += dv2_hill_m_s
    return times_s, states_hill


def _evaluate(
    config: ScenarioConfig,
    times_s: npt.NDArray[np.float64],
    states_hill: npt.NDArray[np.float64],
) -> SafetyReport:
    """Evaluate all four scenario constraints over the sampled trajectory.

    All four are always requested. Passing ``None`` for one would drop it from the report
    and from ``total_violating_samples`` without leaving a trace that it was ever meant to
    be checked, which reads in the output as a constraint that passed.
    """
    constraints = config.constraints
    return evaluate_constraints(
        times_s,
        states_hill,
        keep_out=KeepOutSphere(radius_m=constraints.keep_out_sphere_radius_m),
        ellipsoid=ApproachEllipsoid(semi_axes_m=constraints.approach_ellipsoid_semi_axes_m),
        corridor=ApproachCorridor(
            half_angle_rad=math.radians(constraints.approach_cone_half_angle_deg),
            activation_range_m=constraints.approach_cone_activation_range_m,
            axis_hill=CORRIDOR_AXIS_HILL,
        ),
        closing_velocity=ClosingVelocityLimit(
            max_closing_speed_m_s=constraints.max_closing_velocity_m_s,
            activation_range_m=constraints.max_closing_velocity_activation_range_m,
        ),
    )


def _write_figures(metrics: TrajectoryMetrics, run_dir: Path) -> tuple[dict[str, Path], str | None]:
    """Render the figure suite, or report why it was skipped.

    The import of :mod:`rpo_core.plotting` is deliberately local. That module selects the
    ``Agg`` matplotlib backend at import time, a process-global side effect, and matplotlib
    is an optional extra; importing it at module scope would make ``--no-plots`` pull in the
    entire plotting stack it was asked to avoid. ``test_cli.py`` asserts ``matplotlib`` is
    absent from ``sys.modules`` after a ``--no-plots`` run, which is what keeps this honest.
    """
    try:
        from rpo_core.plotting import plot_all
    except ImportError as exc:
        return {}, (
            f"matplotlib is not installed ({exc}); install the 'viz' extra "
            "(uv run --extra viz ...) to render figures"
        )
    return plot_all(metrics, run_dir), None


def plan_rendezvous(
    config: ScenarioConfig,
    *,
    seed: int | None = None,
    correct: bool = True,
    n_samples: int = DEFAULT_SAMPLE_COUNT,
    base_dir: str | Path = DEFAULT_RUNS_DIR,
    make_plots: bool = True,
    targeting_tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> RendezvousPlan:
    """Plan a two-impulse rendezvous, evaluate it, and write its run directory.

    The full workflow, in the order that makes the result defensible: solve the transfer,
    optionally correct it onto nonlinear dynamics, sample the resulting arc, evaluate every
    constraint against **those** samples, reduce to metrics, and only then write anything to
    disk. Nothing is printed; :mod:`rpo_traj.cli` renders the returned object.

    A violated constraint is a **result**, not a failure. This function returns normally
    with ``all_constraints_satisfied`` false and every output written. The shipped
    ``configs/vbar_baseline.yaml`` is exactly that case, and deliberately so: a half-period
    two-impulse V-bar hop bulges radially by ``dy/4`` regardless of altitude or transfer
    time, which for a 750 m hop is 187.5 m and reaches 20.56 deg against a 10 deg corridor.
    That is geometry (``docs/project1/math-model.md``, model M4 corollary), so the MVP's job
    is to report it, not to widen the cone until it passes.

    Parameters
    ----------
    config
        A validated scenario. Load one with :func:`rpo_core.config.load_scenario`.
    seed
        Run seed. Defaults to ``config.seed``. Recorded in ``provenance.json`` and used in
        the run directory name; it changes no number in a two-impulse plan, because nothing
        here is stochastic.
    correct
        Apply differential correction onto nonlinear dynamics. Default ``True``. With
        ``False`` the raw CW impulses fly, and ``terminal_position_error_m`` reports the
        linearisation error directly -- 1.225271 m on the baseline scenario.
    n_samples
        Trajectory samples, uniform on ``[0, tof_s]``. See :data:`DEFAULT_SAMPLE_COUNT` and
        the module docstring for the convergence sweep that fixes the default.
    base_dir
        Parent directory for the run directory. The run lands in
        ``<base_dir>/<config_hash>-<seed>/``, which is content-addressed, so re-running the
        same scenario at the same seed overwrites in place rather than accumulating.
    make_plots
        Render the figure suite into the run directory. Requires matplotlib; when it is
        missing this is a recorded skip in ``plots_skipped_reason``, not an error, because
        the numerics do not depend on the plotting stack (N-3).
    targeting_tolerance_m
        Convergence tolerance on terminal position miss for the correction, metres. Ignored
        when ``correct`` is false. Below roughly 1e-08 m the nonlinear oracle's own noise
        floor makes the request unsatisfiable and
        :class:`~rpo_core.targeting.TargetingConvergenceError` is raised, correctly.

    Returns
    -------
    RendezvousPlan
        The trajectory, the constraint report, the metrics record, and the paths written.

    Raises
    ------
    PlanningError
        If ``n_samples`` is not a whole number at or above :data:`MIN_SAMPLE_COUNT`, or
        ``seed`` is negative.
    SingularTransferTimeError
        From the CW solve, if the time of flight is at an integer number of orbital periods.
        ``ScenarioConfig`` rejects those up front, so reaching this means a hand-built
        config bypassed :func:`~rpo_core.config.load_scenario`.
    InfeasibleTransferError
        If a cross-track change is requested at a half-period transfer time.
    TargetingConvergenceError, IllConditionedJacobianError
        From the differential correction. Propagated unchanged: the residual history and
        condition number they carry are the diagnosis.
    PropagationError
        If the nonlinear propagation fails.
    MetricsError, ScenarioConfigError, PlottingError
        From writing the metrics record, the run directory, or the figures.

    Examples
    --------
    >>> import tempfile
    >>> from rpo_core.config import load_scenario
    >>> config = load_scenario("configs/vbar_baseline.yaml")  # doctest: +SKIP
    >>> with tempfile.TemporaryDirectory() as tmp:  # doctest: +SKIP
    ...     plan = plan_rendezvous(config, base_dir=tmp, make_plots=False)
    ...     plan.all_constraints_satisfied
    False

    """
    count = _validate_sample_count(n_samples)
    effective_seed = _validate_seed(seed, config)

    r_target_eci_m, v_target_eci_m_s = target_state_eci(config.orbit)

    dv1_hill_m_s, dv2_hill_m_s, cw_dv1, cw_dv2, correction = _solve_impulses(
        config,
        r_target_eci_m,
        v_target_eci_m_s,
        correct=correct,
        targeting_tolerance_m=targeting_tolerance_m,
    )

    times_s, states_hill = _sample_trajectory(
        config, r_target_eci_m, v_target_eci_m_s, dv1_hill_m_s, dv2_hill_m_s, count
    )

    burns = (
        Burn(
            label=_DEPARTURE_BURN_LABEL,
            time_s=0.0,
            delta_v_hill_m_s=(
                float(dv1_hill_m_s[0]),
                float(dv1_hill_m_s[1]),
                float(dv1_hill_m_s[2]),
            ),
        ),
        Burn(
            label=_ARRIVAL_BURN_LABEL,
            time_s=config.tof_s,
            delta_v_hill_m_s=(
                float(dv2_hill_m_s[0]),
                float(dv2_hill_m_s[1]),
                float(dv2_hill_m_s[2]),
            ),
        ),
    )

    report = _evaluate(config, times_s, states_hill)

    commanded_terminal_state_hill = np.concatenate(
        (np.asarray(config.target_hold_point.position_hill_m, dtype=np.float64), np.zeros(3))
    )
    metrics = compute_metrics(
        config,
        times_s,
        states_hill,
        burns,
        report,
        commanded_terminal_state_hill=commanded_terminal_state_hill,
        corridor_axis_hill=CORRIDOR_AXIS_HILL,
        seed=effective_seed,
    )

    # 'correct' changes the numbers, so it must change the path. The run directory is
    # content-addressed on the SCENARIO, and correction is a run option rather than part of
    # the scenario, so without a variant both modes resolve to one directory and the second
    # run silently overwrites the first -- two different results, one path, no warning.
    run_dir = create_run_directory(
        config, effective_seed, base_dir, variant=None if correct else "raw"
    )
    metrics_path = write_metrics(run_dir, metrics)

    figure_paths: dict[str, Path] = {}
    plots_skipped_reason: str | None = None
    if make_plots:
        figure_paths, plots_skipped_reason = _write_figures(metrics, run_dir)
    else:
        plots_skipped_reason = "figures not requested"

    # The raw-CW miss is free in both modes and is the only number that says what the
    # correction bought. When correcting, the shooting solver's first residual *is* that
    # miss by construction. When not, the flown trajectory is the raw CW one, so its
    # achieved terminal position error is the same quantity.
    raw_cw_terminal_miss_m = (
        correction.initial_residual_m
        if correction is not None
        else metrics.terminal_position_error_m
    )
    cw_total_delta_v_m_s = float(np.linalg.norm(cw_dv1) + np.linalg.norm(cw_dv2))

    return RendezvousPlan(
        config=config,
        seed=effective_seed,
        corrected=correct,
        n_samples=count,
        times_s=times_s,
        states_hill=states_hill,
        burns=burns,
        report=report,
        metrics=metrics,
        run_dir=run_dir,
        metrics_path=metrics_path,
        raw_cw_terminal_miss_m=raw_cw_terminal_miss_m,
        corrected_terminal_miss_m=None if correction is None else correction.final_residual_m,
        targeting_iterations=None if correction is None else correction.iterations,
        cw_total_delta_v_m_s=cw_total_delta_v_m_s,
        figure_paths=figure_paths,
        plots_skipped_reason=plots_skipped_reason,
    )
