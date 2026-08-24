r"""Dispersed rendezvous campaigns: navigation error, burn error, honest rates.

This module closes SRS F-5.1 ... F-5.5. It is a composition layer in the same sense as
:mod:`rpo_traj.plan`: it owns no numerics of its own. The randomness and the failure
accounting come from :mod:`rpo_core.montecarlo`, the estimation-error model and the
plan-from-estimate/fly-against-truth split from :mod:`rpo_core.navigation`, the dynamics
from :mod:`rpo_core.relative`, and the safety verdict from :mod:`rpo_core.constraints`.
What this module adds is the wiring, the retention policy, and the reduction to a report
that separates outcomes which must never be merged.

What one run does
-----------------
1. **Disperse the truth.** The chaser is not where the scenario says it is: the true
   initial relative state is the nominal hold point plus a delivery-dispersion draw.
2. **Estimate it, wrongly.** The onboard estimate is truth plus a per-run navigation bias
   plus per-estimate white noise (:class:`~rpo_core.navigation.NavigationErrorModel`).
3. **Plan from the estimate.** The two-impulse Clohessy-Wiltshire solve sees the estimate
   and nothing else.
4. **Execute imperfectly.** Each commanded impulse is scaled and tilted by a
   :class:`~rpo_core.montecarlo.MagnitudePointingDispersion` draw, under the normative
   perpendicular-axis pointing convention of model M9.
5. **Fly against truth.** The executed departure impulse is applied to the *true* state and
   propagated under the nonlinear relative dynamics -- the same dynamics
   :mod:`rpo_traj.plan` flies, for the same reason: evaluating safety on the linear model
   that designed the burn is marking your own homework.
6. **Score it.** All four scenario constraints are evaluated on the flown samples, and a
   fixed set of **scalar** metrics is retained.

Three outcomes, never merged
----------------------------
A campaign has three per-run outcomes and they mean different things:

* **Failed** -- the run raised. No trajectory exists, so nothing about it is known. Counted
  in the denominator of every rate; see :meth:`rpo_core.montecarlo.CampaignResults.success_rate`.
* **Completed but violated** -- the run flew and broke a constraint or missed by more than
  the terminal tolerance. This is a *result*, and often the interesting one.
* **Succeeded** -- flew, satisfied every constraint, and arrived inside tolerance.

:attr:`DispersedCampaignReport.n_failed` and
:attr:`DispersedCampaignReport.n_completed_but_violated` are separate fields for exactly this
reason. Adding them together would let a campaign improve its apparent violation rate by
crashing more often.

Rates are reported with **Wilson score intervals**, taken from
:func:`rpo_core.montecarlo.wilson_interval` rather than re-derived here. A dispersion
campaign is run to establish a rate near 0 % or 100 % -- a keep-out breach probability, a
success rate -- and the normal (Wald) approximation returns a zero-width interval at exactly
those ends. See that function's docstring for the numbers.

Breach probabilities and the failed-run denominator
---------------------------------------------------
A breach rate is reported over **all** runs, failures included. A run that raised did not
breach, but it also did not demonstrate that it would not have, so counting it as a
non-breach makes the reported probability a **lower bound** whenever there are failures.
That is why :attr:`DispersedCampaignReport.n_failed` sits next to the breach rates and why
:meth:`DispersedCampaignReport.to_dict` emits ``"breach_rate_is_lower_bound"``. The
alternative -- conditioning on completion -- silently answers a different question.

The half-period cross-track trap
--------------------------------
The shipped baseline is a **half-period** V-bar hop, and at an integer multiple of the half
period the cross-track subproblem is rank-deficient: ``Phi_rv[2,2] = sin(tau)/n`` vanishes,
so ``z(t_f)`` is pinned to ``cos(tau) * z_0 = -z_0`` whatever impulse is applied
(``docs/project1/math-model.md``, model M4). The nominal scenario has ``z_0 = 0`` exactly, so
the planner never meets this. A *dispersed* campaign meets it in **every run**: the delivery
and navigation dispersions are three-dimensional, so the estimated ``z_0`` is essentially
never zero, and :func:`~rpo_core.relative.cw.two_impulse_transfer` would raise
:class:`~rpo_core.exceptions.InfeasibleTransferError` on all of them at its default
``feasibility_tol_m`` of 1e-6 m. A campaign reporting 100 % failure would be arithmetically
correct and completely useless.

The resolution here is to pass ``cross_track_tolerance_m`` (default ``math.inf``) into the
solve, which accepts the request and applies no cross-track impulse -- *which is the correct
physics*: at a half period there is no impulse that changes ``z``. The residual is not
hidden, it is reported: the terminal cross-track miss is exactly ``|z_0|`` of the **true**
initial state, and :attr:`RunOutcome.terminal_position_error_m` carries it. Because that
floor is structural rather than a guidance shortfall,
:attr:`RunOutcome.terminal_in_plane_position_error_m` is retained alongside it, so the
controllable part of the miss can be read on its own.

Setting ``cross_track_tolerance_m`` back to a small value restores the raising behaviour,
and the campaign then reports a wall of :class:`~rpo_core.exceptions.InfeasibleTransferError`
in ``failure_counts_by_type``. Both behaviours are available; neither is silent.

What dominates the baseline, measured
-------------------------------------
Worth stating here because it is counter-intuitive and it is what the campaign is for:
**delivery dispersion is nearly free and navigation error is expensive.** The terminal state
error of a two-impulse plan is exactly ``-Phi @ e`` in the *estimation* error ``e``
(``rpo_core.navigation``, model M9), so an initial-state dispersion the navigation solution
can see costs nothing in-plane -- the guidance simply plans from where the chaser actually
is. Measured on the shipped baseline with the delivery dispersion alone at 5 m one-sigma:
worst in-plane terminal miss **6.4e-11 m** over 60 runs.

Navigation error does not cancel, and the half-period transfer amplifies it. At
``tau = pi`` the along-track position-from-velocity coefficient is
``Phi_rv[1,1] = (4 sin tau - 3 tau)/n = -3 pi / n = -8367 s``, so **1 mm/s of velocity
*knowledge* error is 8.4 m of along-track terminal miss**. With the representative
navigation model of :meth:`DispersionSettings.realistic` the predicted terminal miss is
``sqrt(diag(Phi P_nav Phi.T)) = (23.9, 62.7, 2.2) m``, and the arrival hold point sits only
50 m outside the 200 m keep-out sphere. The keep-out breach probability that follows is a
*minority but not a small one*, and it is driven by navigation, not by delivery or by burn
execution. A study that modelled navigation error as a smaller effect than delivery
dispersion -- which is the intuitive ordering -- would get this exactly backwards.

Memory, measured
----------------
The planner's ``metrics.json`` is ~394 KB at its default 2001 samples because the per-sample
series is in the record. Retaining that per run would make a 1000-run campaign ~400 MB. This
module therefore retains **scalars only**: :class:`RunOutcome` has no array field, and the
campaign is run with ``keep_results=False``, so no trajectory survives the run that produced
it. The (N, 6) state history exists inside one run, is scored, and is dropped.

**Measured, 1000 runs at the default 401 samples** on an idle 2026 laptop, by recursive
``sys.getsizeof`` over the returned report plus ``tracemalloc``:

====================================  ====================  =========================
Quantity                              ``dynamics="cw"``     ``dynamics="nonlinear"``
====================================  ====================  =========================
Wall time                             **19.9 s** (20 ms/run)  **387 s** (387 ms/run)
Retained report                       298.1 KiB (305 B/run)   296.9 KiB (304 B/run)
Peak traced allocation                2.04 MiB                2.08 MiB
``to_dict()`` as JSON                 7.7 KiB                 7.7 KiB
====================================  ====================  =========================

So a 1000-run campaign at the deliverable fidelity costs **about six and a half minutes and
about 300 KiB**. The 304 bytes per run are that run's dispersion samples (a
:class:`~rpo_core.montecarlo.BurnExecutionSample` and two 6-vectors) plus 16 retained
floats; they do **not** grow with ``n_samples`` at all -- see
``test_the_retained_metrics_do_not_grow_with_the_sample_count``. Against the ~400 MB the
per-sample series would have cost, that is a factor of ~1350.

The nonlinear cost is two ``solve_ivp`` integrations per run at ``rtol = atol = 1e-12`` and
scales with ``n_samples`` only through the number of dense-output evaluations. The linear
cost is dominated by a Python loop over :func:`~rpo_core.relative.cw.propagate_cw` -- one
6x6 STM construction per output sample -- so ``dynamics="cw"`` is O(n_samples) in
interpreter overhead and is the obvious thing to vectorise if screening throughput ever
becomes the constraint.

Validity
--------
* Two-body dynamics throughout: no J2, no drag, no third bodies. Impulses are instantaneous.
* Guidance is **open loop after departure**: the arrival impulse is the one commanded at the
  planning epoch. No mid-course re-navigation. That is the conservative reading of a
  two-impulse plan and it is what makes the terminal-error closed form in
  :mod:`rpo_core.navigation` exact.
* The plan is **not** differentially corrected onto nonlinear dynamics. The correction of
  :func:`rpo_core.targeting.correct_two_impulse_transfer` costs several nonlinear
  propagations per run, and -- more importantly -- correcting against the *truth* propagator
  would be planning on truth, the exact error this module's structure exists to prevent.
  Correcting against the estimate is legitimate and is left as future work; its absence
  shows up as a bias of about 1.2 m in the terminal miss (``rpo_traj.plan``, measured),
  which is small against the dispersion-driven spread but is not zero.
* The dispersion magnitudes in :meth:`DispersionSettings.realistic` are representative
  order-of-magnitude values for a small chaser in LEO, chosen so the campaign exercises the
  right regime. They are **not** flight requirements and carry no authority; see
  ``DISCLAIMER.md``.
* Runs are independent by construction (the substream scheme of
  :mod:`rpo_core.montecarlo`), and nothing here carries state between runs.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from rpo_core.config import ScenarioConfig
from rpo_core.constraints import (
    DEFAULT_ZERO_RANGE_TOL_M,
    ApproachCorridor,
    ApproachEllipsoid,
    ClosingVelocityLimit,
    KeepOutSphere,
    evaluate_constraints,
    range_rate_m_s,
)
from rpo_core.exceptions import RpoCoreError
from rpo_core.montecarlo import (
    DEFAULT_CONFIDENCE,
    DEFAULT_PERCENTILES,
    BurnExecutionSample,
    CampaignSummary,
    Dispersion,
    DispersionSample,
    MagnitudePointingDispersion,
    MetricSummary,
    ProportionEstimate,
    VectorNormalDispersion,
    proportion_estimate,
    run_campaign,
)
from rpo_core.navigation import (
    STATE_DIMENSION,
    NavigationErrorModel,
    cw_truth_propagator,
    plan_from_estimate,
    validate_covariance,
)
from rpo_core.relative.nonlinear import propagate_relative_nonlinear

from .plan import CORRIDOR_AXIS_HILL, target_state_eci

__all__ = [
    "BURN_EXECUTION_KEY",
    "DEFAULT_CAMPAIGN_SAMPLE_COUNT",
    "DEFAULT_TERMINAL_TOLERANCE_M",
    "INITIAL_STATE_KEY",
    "NAVIGATION_BIAS_KEY",
    "CampaignSetupError",
    "DispersedCampaignReport",
    "DispersionSettings",
    "RunOutcome",
    "SensitivityPoint",
    "run_dispersed_rendezvous",
    "sensitivity_sweep",
]


class CampaignSetupError(RpoCoreError, ValueError):
    """Raised when a campaign request is malformed before any run is executed.

    Deliberately narrow, in the same spirit as :class:`rpo_traj.plan.PlanningError`.
    Failures *inside* a run -- a singular transfer time, an infeasible cross-track request,
    an integrator that gives up -- are recorded per run by
    :func:`rpo_core.montecarlo.run_campaign` and reported in ``failure_counts_by_type``,
    never rewrapped. Invalid run counts, seeds and confidence levels are validated by the
    Monte Carlo harness itself and raise
    :class:`~rpo_core.montecarlo.CampaignConfigurationError`; this type covers what only a
    campaign over a *scenario* can get wrong.
    """


#: Trajectory samples per dispersed run, uniform on ``[0, tof_s]``.
#:
#: Lower than :data:`rpo_traj.plan.DEFAULT_SAMPLE_COUNT` (2001), and the reason is a change of
#: question, not a lowering of standards. The planner resolves a *single* trajectory's
#: closest approach to sub-millimetre, where a 1.07e-03 m keep-out dip 3.6 s wide decides the
#: reported clearance. A campaign asks what fraction of a thousand dispersed trajectories
#: breach a 200 m sphere, and the run-to-run spread of the minimum separation under realistic
#: dispersions is metres, four orders of magnitude above the discretisation error that
#: motivated 2001. Sampling finer buys precision on a quantity whose uncertainty is dominated
#: by the dispersion, at linear cost in the propagation, which is the entire cost of a
#: campaign. See ``test_the_sample_count_does_not_move_the_campaign_rates`` for the measured
#: check that 401 and 2001 give the same rates.
DEFAULT_CAMPAIGN_SAMPLE_COUNT: int = 401

#: Terminal position miss at or below which a completed run counts as a success, metres.
#:
#: 10 m against a 250 m arrival hold point is 4 % of the standoff. It is a reporting
#: threshold, not a requirement: pass ``terminal_tolerance_m`` to use another, and read the
#: terminal-error percentiles rather than the pass/fail count if the threshold is the thing
#: in question.
DEFAULT_TERMINAL_TOLERANCE_M: float = 10.0

#: Dispersion names, fixed because they name the random substreams.
#:
#: :func:`rpo_core.montecarlo.draw_samples` addresses each dispersion's substream by hashing
#: its *name*, so renaming one of these changes every historical campaign's samples for the
#: same seed. They are constants rather than string literals for that reason.
BURN_EXECUTION_KEY: str = "burn_execution"
INITIAL_STATE_KEY: str = "initial_state"
NAVIGATION_BIAS_KEY: str = "navigation_bias"

_CONSTRAINT_NAMES: tuple[str, ...] = (
    "keep_out_sphere",
    "approach_ellipsoid",
    "approach_corridor",
    "closing_velocity",
)


# --------------------------------------------------------------------------------------
# Dispersion settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersionSettings:
    """What varies from run to run, and by how much.

    Defaults to **no dispersion at all**, so ``DispersionSettings()`` is the control case
    that must reproduce :func:`rpo_traj.plan.plan_rendezvous`'s nominal manoeuvre exactly.
    Use :meth:`realistic` for the representative set.

    Attributes
    ----------
    burn_execution
        Magnitude scale-factor and pointing dispersion applied to **both** impulses, or
        ``None`` for perfect execution. The two impulses draw from the same distribution but
        are separate draws, because a thruster's error is not repeatable between burns.
    navigation
        Estimation-error model. Its bias term becomes a declared campaign dispersion (drawn
        once per run by the harness); its white-noise term is drawn from the run's own
        generator at each estimate.
    initial_state_covariance
        Shape (6, 6) SPD covariance of the **true** initial relative state about the nominal
        start hold point, or ``None`` for exact delivery. This is where the chaser actually
        is, as opposed to where it is believed to be.

    Raises
    ------
    CovarianceDefinitionError
        If ``initial_state_covariance`` is not a 6x6 symmetric positive-definite matrix.

    """

    burn_execution: MagnitudePointingDispersion | None = None
    navigation: NavigationErrorModel = field(default_factory=NavigationErrorModel)
    initial_state_covariance: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate the initial-state covariance in place."""
        if self.initial_state_covariance is not None:
            object.__setattr__(
                self,
                "initial_state_covariance",
                validate_covariance(
                    self.initial_state_covariance,
                    name="initial_state_covariance",
                    dimension=STATE_DIMENSION,
                ),
            )

    @classmethod
    def realistic(cls) -> DispersionSettings:
        r"""Return the representative dispersion set used for the headline campaign.

        Order-of-magnitude values for a small chaser doing terminal proximity operations in
        LEO, chosen so the campaign exercises the regime the constraints were written for.
        **Not flight requirements** (``DISCLAIMER.md``):

        =========================  ========  =============================================
        Quantity                   1-sigma   Why this order
        =========================  ========  =============================================
        Burn magnitude             2 %       Scale-factor error of a small RCS thruster.
        Burn pointing              1 deg     Thruster misalignment plus attitude knowledge.
        Delivery position          5 m       Where the previous phase actually left it.
        Delivery velocity          0.01 m/s  Residual rate at the hold point.
        Navigation noise, position 2 m       Relative RF/optical nav at ~1 km.
        Navigation noise, velocity 0.005 m/s Differenced over a filter interval.
        Navigation bias, position  1 m       Un-modelled offset, constant over a manoeuvre.
        Navigation bias, velocity  0.001 m/s Same, on the rate channel.
        =========================  ========  =============================================

        The bias is deliberately *smaller* than the noise. That is the interesting case: a
        bias smaller than the noise is invisible in a single estimate and is the part that
        survives every amount of filtering, so a campaign that models it as extra white
        noise would report a manoeuvre that gets arbitrarily accurate with a better filter.

        All three covariances are diagonal. A correlated navigation covariance is perfectly
        supported (:class:`~rpo_core.montecarlo.VectorNormalDispersion` samples through a
        Cholesky factor); diagonal is used here because inventing plausible-looking
        correlations for a public example would be fabricating a filter that does not exist.

        Returns
        -------
        DispersionSettings

        """
        return cls(
            burn_execution=MagnitudePointingDispersion(
                sigma_magnitude=0.02, sigma_pointing_rad=math.radians(1.0)
            ),
            navigation=NavigationErrorModel(
                noise_covariance=_diagonal_state_covariance(2.0, 5.0e-3),
                bias_covariance=_diagonal_state_covariance(1.0, 1.0e-3),
            ),
            initial_state_covariance=_diagonal_state_covariance(5.0, 1.0e-2),
        )

    def scaled(
        self,
        *,
        burn_execution: float = 1.0,
        navigation: float = 1.0,
        initial_state: float = 1.0,
    ) -> DispersionSettings:
        """Return a copy with each dispersion family's **sigmas** scaled by a factor.

        Covariances scale by the square, sigmas by the factor, so ``scaled(navigation=2.0)``
        doubles every navigation one-sigma. A factor of exactly zero removes the term
        (``None``) rather than producing a zero covariance, which would not be positive
        definite and would be rejected -- the same reason
        :class:`~rpo_core.navigation.NavigationErrorModel` expresses "no error" by omission.

        Parameters
        ----------
        burn_execution, navigation, initial_state
            Non-negative scale factors on the corresponding one-sigmas.

        Returns
        -------
        DispersionSettings

        Raises
        ------
        CampaignSetupError
            If any factor is negative or not finite.

        """
        for name, value in (
            ("burn_execution", burn_execution),
            ("navigation", navigation),
            ("initial_state", initial_state),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise CampaignSetupError(
                    f"{name} scale must be a finite non-negative factor on the one-sigma, got "
                    f"{value!r}. A negative factor is not a smaller dispersion; it samples "
                    "identically to its absolute value and hides the sign error."
                )
        return DispersionSettings(
            burn_execution=(
                None
                if self.burn_execution is None or burn_execution == 0.0
                else MagnitudePointingDispersion(
                    sigma_magnitude=self.burn_execution.sigma_magnitude * burn_execution,
                    sigma_pointing_rad=self.burn_execution.sigma_pointing_rad * burn_execution,
                )
            ),
            navigation=NavigationErrorModel(
                noise_covariance=_scale_covariance(self.navigation.noise_covariance, navigation),
                bias_covariance=_scale_covariance(self.navigation.bias_covariance, navigation),
            ),
            initial_state_covariance=_scale_covariance(
                self.initial_state_covariance, initial_state
            ),
        )

    def dispersions(self) -> dict[str, Dispersion]:
        """Return the declared campaign dispersions, one draw per run each.

        The navigation **bias** appears here and the navigation **noise** does not, and that
        asymmetry is the model, not an oversight: a campaign dispersion is by definition
        drawn once per run from that run's substream, which is exactly what a bias is. The
        noise is drawn per estimate from the run's own generator.
        """
        declared: dict[str, Dispersion] = {}
        if self.burn_execution is not None:
            declared[BURN_EXECUTION_KEY] = self.burn_execution
        if self.initial_state_covariance is not None:
            declared[INITIAL_STATE_KEY] = VectorNormalDispersion(
                mean=np.zeros(STATE_DIMENSION), covariance=self.initial_state_covariance
            )
        bias = self.navigation.bias_dispersion()
        if bias is not None:
            declared[NAVIGATION_BIAS_KEY] = bias
        return declared

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of every dispersion family."""
        return {
            BURN_EXECUTION_KEY: (
                None if self.burn_execution is None else self.burn_execution.describe()
            ),
            "navigation": self.navigation.describe(),
            INITIAL_STATE_KEY: (
                None
                if self.initial_state_covariance is None
                else [[float(v) for v in row] for row in self.initial_state_covariance]
            ),
        }


def _diagonal_state_covariance(
    sigma_position_m: float, sigma_velocity_m_s: float
) -> npt.NDArray[np.float64]:
    """Return a diagonal 6x6 relative-state covariance from two one-sigmas."""
    return np.diag(
        np.array(
            [
                sigma_position_m**2,
                sigma_position_m**2,
                sigma_position_m**2,
                sigma_velocity_m_s**2,
                sigma_velocity_m_s**2,
                sigma_velocity_m_s**2,
            ],
            dtype=np.float64,
        )
    )


def _scale_covariance(
    covariance: npt.NDArray[np.float64] | None, sigma_scale: float
) -> npt.NDArray[np.float64] | None:
    """Return ``covariance`` with its one-sigmas scaled, or ``None`` at a zero factor."""
    if covariance is None or sigma_scale == 0.0:
        return None
    scaled: npt.NDArray[np.float64] = covariance * (float(sigma_scale) ** 2)
    return scaled


# --------------------------------------------------------------------------------------
# Per-run outcome
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    """Everything one dispersed run contributes -- scalars only, by construction.

    There is no array field on this class and there is not meant to be. The (N, 6) state
    history and the (N,) time base exist inside :func:`_execute_dispersed_run`, are scored
    against every constraint, and are dropped when it returns. See the module docstring's
    memory section for the arithmetic that makes that non-negotiable at 1000 runs.

    Attributes
    ----------
    terminal_position_error_m, terminal_velocity_error_m_s
        Norms of the miss against the commanded terminal state, after the arrival impulse.
    terminal_in_plane_position_error_m
        Miss in the Hill x-y plane only. Reported separately because at a half-period
        transfer time the cross-track component is structurally uncontrollable, so the full
        norm carries a floor that no guidance can remove; see the module docstring.
    total_delta_v_m_s
        Sum of the **executed** impulse magnitudes -- what was actually spent.
    commanded_delta_v_m_s
        Sum of the commanded magnitudes. The difference from ``total_delta_v_m_s`` is the
        magnitude component of the burn execution error, and is the reason both are kept.
    min_separation_m
        Smallest sampled range from the target.
    keep_out_clearance_m
        Parabola-refined minimum clearance against the keep-out sphere, metres, negative
        inside. Refined rather than sampled, matching
        :class:`~rpo_core.constraints.KeepOutSphereResult`'s own verdict.
    max_closing_velocity_m_s, max_corridor_angle_rad
        Worst values over the **whole** arc, not gated by the constraints' activation
        ranges. Ungated on purpose: the gated worst value is ``nan`` when no sample enters
        the activation range, and a ``nan`` metric fails the run in
        :func:`~rpo_core.montecarlo.execute_run`, which would record "this constraint never
        applied" as "this trajectory failed". The gated *verdicts* are the ``*_satisfied``
        flags below, which are correct and always defined.
    max_ellipsoid_quadratic_form
        Largest value of ``(x/a)^2 + (y/b)^2 + (z/c)^2``; above 1.0 is outside.
    n_violating_samples
        Total violating samples across all four constraints.
    keep_out_satisfied, ellipsoid_satisfied, corridor_satisfied, closing_velocity_satisfied
        Per-constraint verdicts, activation ranges honoured.
    all_constraints_satisfied
        Conjunction of the four.

    """

    terminal_position_error_m: float
    terminal_in_plane_position_error_m: float
    terminal_velocity_error_m_s: float
    total_delta_v_m_s: float
    commanded_delta_v_m_s: float
    min_separation_m: float
    keep_out_clearance_m: float
    max_closing_velocity_m_s: float
    max_corridor_angle_rad: float
    max_ellipsoid_quadratic_form: float
    n_violating_samples: int
    keep_out_satisfied: bool
    ellipsoid_satisfied: bool
    corridor_satisfied: bool
    closing_velocity_satisfied: bool
    all_constraints_satisfied: bool


#: Retained scalar metrics: name to extractor. This mapping *is* the retention policy.
#:
#: Booleans are retained as 0.0/1.0 so that their mean is the rate and their sum the count,
#: which is what :class:`DispersedCampaignReport` reads them back as.
_RETAIN: Mapping[str, Callable[[RunOutcome], float]] = {
    "terminal_position_error_m": lambda o: o.terminal_position_error_m,
    "terminal_in_plane_position_error_m": lambda o: o.terminal_in_plane_position_error_m,
    "terminal_velocity_error_m_s": lambda o: o.terminal_velocity_error_m_s,
    "total_delta_v_m_s": lambda o: o.total_delta_v_m_s,
    "commanded_delta_v_m_s": lambda o: o.commanded_delta_v_m_s,
    "min_separation_m": lambda o: o.min_separation_m,
    "keep_out_clearance_m": lambda o: o.keep_out_clearance_m,
    "max_closing_velocity_m_s": lambda o: o.max_closing_velocity_m_s,
    "max_corridor_angle_rad": lambda o: o.max_corridor_angle_rad,
    "max_ellipsoid_quadratic_form": lambda o: o.max_ellipsoid_quadratic_form,
    "n_violating_samples": lambda o: float(o.n_violating_samples),
    "keep_out_sphere_breached": lambda o: float(not o.keep_out_satisfied),
    "approach_ellipsoid_breached": lambda o: float(not o.ellipsoid_satisfied),
    "approach_corridor_breached": lambda o: float(not o.corridor_satisfied),
    "closing_velocity_breached": lambda o: float(not o.closing_velocity_satisfied),
    "all_constraints_satisfied": lambda o: float(o.all_constraints_satisfied),
}

_BREACH_METRIC: Mapping[str, str] = {
    "keep_out_sphere": "keep_out_sphere_breached",
    "approach_ellipsoid": "approach_ellipsoid_breached",
    "approach_corridor": "approach_corridor_breached",
    "closing_velocity": "closing_velocity_breached",
}


# --------------------------------------------------------------------------------------
# Nominal scenario, frozen for the run function
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Nominal:
    """The undispersed scenario, pre-resolved once so no run repeats the setup."""

    n_rad_s: float
    times_s: npt.NDArray[np.float64]
    nominal_initial_state_hill: npt.NDArray[np.float64]
    commanded_terminal_state_hill: npt.NDArray[np.float64]
    corridor_axis_hill: npt.NDArray[np.float64]
    settings: DispersionSettings
    propagate_fn: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]
    cross_track_tolerance_m: float
    keep_out: KeepOutSphere
    ellipsoid: ApproachEllipsoid
    corridor: ApproachCorridor
    closing_velocity: ClosingVelocityLimit


def _nonlinear_propagator(
    r_target_eci_m: npt.NDArray[np.float64],
    v_target_eci_m_s: npt.NDArray[np.float64],
    *,
    rtol: float,
    atol: float,
) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Return a truth propagator flying the nonlinear relative dynamics."""

    def _propagate(
        initial_state_hill: npt.NDArray[np.float64], times_s: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Propagate one dispersed relative state onto the campaign's output grid."""
        return propagate_relative_nonlinear(
            r_target_eci_m,
            v_target_eci_m_s,
            initial_state_hill,
            times_s,
            rtol=rtol,
            atol=atol,
        )

    return _propagate


def _execute_dispersed_run(
    nominal: _Nominal,
    samples: Mapping[str, DispersionSample],
    rng: np.random.Generator,
) -> RunOutcome:
    """Fly one dispersed run and reduce it to scalars.

    The order is load-bearing and is the order of the module docstring: disperse truth,
    estimate it, plan from the estimate, execute imperfectly, fly against truth, score.
    """
    truth_state_hill = nominal.nominal_initial_state_hill.copy()
    initial_offset = samples.get(INITIAL_STATE_KEY)
    if initial_offset is not None:
        truth_state_hill = truth_state_hill + np.asarray(initial_offset, dtype=np.float64)

    bias_sample = samples.get(NAVIGATION_BIAS_KEY)
    solution = (
        nominal.settings.navigation.begin_run(rng)
        if bias_sample is None
        else nominal.settings.navigation.with_bias(np.asarray(bias_sample, dtype=np.float64))
    )
    estimated_state_hill = solution.estimate(truth_state_hill, rng)

    burn_sample = samples.get(BURN_EXECUTION_KEY)
    execute_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None
    if burn_sample is not None:
        assert isinstance(burn_sample, BurnExecutionSample)
        execute_fn = burn_sample.apply

    transfer = plan_from_estimate(
        nominal.n_rad_s,
        nominal.times_s,
        estimated_state_hill=estimated_state_hill,
        truth_state_hill=truth_state_hill,
        commanded_terminal_state_hill=nominal.commanded_terminal_state_hill,
        propagate_fn=nominal.propagate_fn,
        execute_fn=execute_fn,
        cross_track_feasibility_tol_m=nominal.cross_track_tolerance_m,
    )

    states = transfer.states_hill
    report = evaluate_constraints(
        nominal.times_s,
        states,
        keep_out=nominal.keep_out,
        ellipsoid=nominal.ellipsoid,
        corridor=nominal.corridor,
        closing_velocity=nominal.closing_velocity,
    )
    assert report.keep_out is not None
    assert report.ellipsoid is not None
    assert report.corridor is not None
    assert report.closing_velocity is not None

    ranges = np.linalg.norm(states[:, :3], axis=1)
    closing = -range_rate_m_s(states)
    directed = ranges > DEFAULT_ZERO_RANGE_TOL_M
    cosine = np.ones_like(ranges)
    np.divide(states[:, :3] @ nominal.corridor_axis_hill, ranges, out=cosine, where=directed)
    corridor_angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    terminal_error = states[-1] - nominal.commanded_terminal_state_hill
    return RunOutcome(
        terminal_position_error_m=transfer.terminal_position_error_m,
        terminal_in_plane_position_error_m=float(np.linalg.norm(terminal_error[:2])),
        terminal_velocity_error_m_s=transfer.terminal_velocity_error_m_s,
        total_delta_v_m_s=transfer.total_delta_v_m_s,
        commanded_delta_v_m_s=float(
            np.linalg.norm(transfer.dv1_commanded_hill_m_s)
            + np.linalg.norm(transfer.dv2_commanded_hill_m_s)
        ),
        min_separation_m=float(np.min(ranges)),
        keep_out_clearance_m=float(report.keep_out.refined_clearance_m),
        max_closing_velocity_m_s=float(np.max(closing)),
        max_corridor_angle_rad=float(np.max(corridor_angle)),
        max_ellipsoid_quadratic_form=float(report.ellipsoid.worst_value),
        n_violating_samples=int(report.total_violating_samples),
        keep_out_satisfied=bool(report.keep_out.satisfied),
        ellipsoid_satisfied=bool(report.ellipsoid.satisfied),
        corridor_satisfied=bool(report.corridor.satisfied),
        closing_velocity_satisfied=bool(report.closing_velocity.satisfied),
        all_constraints_satisfied=bool(report.all_satisfied),
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersedCampaignReport:
    """A dispersed campaign reduced to rates, percentiles and counts.

    Attributes
    ----------
    seed, n_runs
        Campaign identity and size.
    n_failed
        Runs that **raised**. No trajectory exists for these.
    n_completed
        ``n_runs - n_failed``.
    n_succeeded
        Runs that completed, satisfied every constraint, and arrived within
        ``terminal_tolerance_m``.
    n_completed_but_violated
        ``n_completed - n_succeeded``. Kept as its own field, never folded into
        ``n_failed``: a run that flew and broke the corridor is a *result*, and a run that
        raised is a missing measurement. Merging them destroys the distinction that decides
        whether the next thing to fix is the trajectory or the code.
    success_rate, completion_rate
        Wilson score interval estimates over all ``n_runs``.
    breach_rates
        Constraint name to Wilson interval estimate, over all ``n_runs``. A lower bound when
        ``n_failed > 0``; see the module docstring.
    metrics
        Retained scalar metric name to :class:`~rpo_core.montecarlo.MetricSummary`, including
        the requested percentiles.
    per_run_metrics
        Metric name to the raw per-run values, in run-index order over the **completed**
        runs. Scalars only; 1000 runs is ~130 KB. Kept so a caller can re-percentile,
        cross-plot, or check run-for-run reproducibility without re-running the campaign.
    completed_indices
        Run indices the rows of ``per_run_metrics`` belong to, so a failed run in the middle
        does not silently shift the alignment.
    failure_counts_by_type
        Exception class name to count.
    terminal_tolerance_m, confidence
        The thresholds this report was reduced at, carried so the numbers can be read
        without the call that produced them.
    dispersions
        The dispersion families used, JSON-serialisable.
    summary
        The underlying :class:`~rpo_core.montecarlo.CampaignSummary`, which carries the seed,
        the verbatim dispersion definitions and the numpy/bit-generator provenance needed to
        reconstruct the campaign.

    """

    seed: int
    n_runs: int
    n_failed: int
    n_completed: int
    n_succeeded: int
    n_completed_but_violated: int
    success_rate: ProportionEstimate
    completion_rate: ProportionEstimate
    breach_rates: Mapping[str, ProportionEstimate]
    metrics: Mapping[str, MetricSummary]
    per_run_metrics: Mapping[str, npt.NDArray[np.float64]]
    completed_indices: npt.NDArray[np.int64]
    failure_counts_by_type: Mapping[str, int]
    terminal_tolerance_m: float
    confidence: float
    dispersions: Mapping[str, Any]
    summary: CampaignSummary

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form, without the per-run arrays.

        ``per_run_metrics`` is excluded: it is a working array, not a result, and emitting a
        thousand-element list per metric would put the thing this module refuses to retain
        per run straight back into the record.
        """
        return {
            "seed": self.seed,
            "n_runs": self.n_runs,
            "n_failed": self.n_failed,
            "n_completed": self.n_completed,
            "n_succeeded": self.n_succeeded,
            "n_completed_but_violated": self.n_completed_but_violated,
            "success_rate": self.success_rate.describe(),
            "completion_rate": self.completion_rate.describe(),
            "breach_rates": {
                name: estimate.describe() for name, estimate in self.breach_rates.items()
            },
            "breach_rate_is_lower_bound": self.n_failed > 0,
            "breach_rate_denominator": "all_runs_including_failures",
            "metrics": {name: summary.describe() for name, summary in self.metrics.items()},
            "failure_counts_by_type": dict(self.failure_counts_by_type),
            "terminal_tolerance_m": self.terminal_tolerance_m,
            "confidence": self.confidence,
            "dispersions": dict(self.dispersions),
            "numpy_version": self.summary.numpy_version,
            "bit_generator": self.summary.bit_generator,
        }


# --------------------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------------------


def run_dispersed_rendezvous(
    config: ScenarioConfig,
    n_runs: int,
    seed: int,
    *,
    settings: DispersionSettings | None = None,
    n_samples: int = DEFAULT_CAMPAIGN_SAMPLE_COUNT,
    dynamics: Literal["nonlinear", "cw"] = "nonlinear",
    propagate_fn: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]
    | None = None,
    terminal_tolerance_m: float = DEFAULT_TERMINAL_TOLERANCE_M,
    cross_track_tolerance_m: float = math.inf,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    confidence: float = DEFAULT_CONFIDENCE,
    map_fn: Callable[..., Any] | None = None,
) -> DispersedCampaignReport:
    """Run a dispersed rendezvous campaign and reduce it to a report.

    Parameters
    ----------
    config
        A validated scenario, as :func:`rpo_traj.plan.plan_rendezvous` takes.
    n_runs
        Number of runs, strictly positive.
    seed
        Non-negative campaign seed. Run ``i`` depends on ``(seed, i)`` alone, so the first
        100 runs of a 1000-run campaign are bitwise identical to a 100-run campaign at the
        same seed. That property belongs to :func:`rpo_core.montecarlo.run_campaign`; this
        function is tested for not breaking it.
    settings
        What varies and by how much. Defaults to :meth:`DispersionSettings.realistic`. Pass
        ``DispersionSettings()`` for the zero-dispersion control.
    n_samples
        Trajectory samples per run. See :data:`DEFAULT_CAMPAIGN_SAMPLE_COUNT`.
    dynamics
        ``"nonlinear"`` (default) flies
        :func:`~rpo_core.relative.nonlinear.propagate_relative_nonlinear`, the same truth
        model :mod:`rpo_traj.plan` uses. ``"cw"`` flies the linear
        Clohessy-Wiltshire STM, which is one to two orders of magnitude cheaper and is the
        right choice for screening a dispersion set before committing to a long campaign --
        but it evaluates the safety constraints on the same model that designed the burn, so
        no reported rate from it is a deliverable.
    propagate_fn
        ``(initial_state_hill, times_s) -> (N, 6)``. Overrides ``dynamics`` entirely.
        Provided so a caller can supply a perturbed or higher-fidelity truth model, and so a
        test can inject a fault and check that a raising run is reported as a failure.
    terminal_tolerance_m
        Terminal position miss at or below which a completed run counts as a success.
    cross_track_tolerance_m
        Passed to the CW solve as ``feasibility_tol_m``. Defaults to ``math.inf``: at a
        half-period transfer time cross-track is uncontrollable, so the default accepts the
        request, applies no cross-track impulse, and lets the residual appear in the
        terminal error. Lower it to have those runs raise instead. **Read the module
        docstring's "half-period cross-track trap" section before changing this.**
    percentiles
        Percentile levels reported for every retained metric.
    confidence
        Two-sided confidence level for every interval.
    map_fn
        Passed through to :func:`rpo_core.montecarlo.run_campaign` to parallelise. Results
        are re-sorted by index and the harness verifies no run was dropped or duplicated.
        **A process pool will not work as written**: the truth propagator is a closure over
        the target's inertial state, so the nominal is not picklable. A thread pool does
        work and is not as useless as it sounds, because the cost is inside
        ``scipy.integrate.solve_ivp``, which releases the GIL for the compiled part of the
        step. Making the nominal picklable is a small change and a real improvement; it is
        not made here because it would trade the closure for a module-level factory and a
        dispatch table, and the campaign is not yet the bottleneck it would be justified by.

    Returns
    -------
    DispersedCampaignReport

    Raises
    ------
    CampaignSetupError
        If ``n_samples`` is below 3 (the keep-out refinement needs a bracketing triple, as
        :data:`rpo_traj.plan.MIN_SAMPLE_COUNT` explains), ``terminal_tolerance_m`` is
        negative or not finite, or ``dynamics`` is not a recognised model.
    CampaignConfigurationError
        From the Monte Carlo harness, for an invalid ``n_runs``, ``seed`` or ``confidence``.

    Examples
    --------
    >>> from rpo_core.config import load_scenario
    >>> config = load_scenario("configs/vbar_baseline.yaml")  # doctest: +SKIP
    >>> report = run_dispersed_rendezvous(  # doctest: +SKIP
    ...     config, 200, seed=42, dynamics="cw")
    >>> report.n_failed + report.n_succeeded + report.n_completed_but_violated  # doctest: +SKIP
    200

    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 3:
        raise CampaignSetupError(
            f"n_samples must be an int of at least 3, got {n_samples!r}. Sub-sample "
            "refinement of the closest approach needs a bracketing triple, so with fewer "
            "than three samples every keep-out minimum is an endpoint and the refined "
            "clearance silently equals the sampled one for a structural reason."
        )
    tolerance = float(terminal_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise CampaignSetupError(
            f"terminal_tolerance_m must be a finite non-negative miss distance, got "
            f"{terminal_tolerance_m!r}"
        )
    if dynamics not in ("nonlinear", "cw"):
        raise CampaignSetupError(
            f"dynamics must be 'nonlinear' or 'cw', got {dynamics!r}. Pass propagate_fn to "
            "supply a truth model this module does not know about."
        )

    effective_settings = DispersionSettings.realistic() if settings is None else settings
    nominal = _build_nominal(
        config,
        effective_settings,
        n_samples=n_samples,
        dynamics=dynamics,
        propagate_fn=propagate_fn,
        cross_track_tolerance_m=cross_track_tolerance_m,
    )

    def _success(outcome: RunOutcome) -> bool:
        """Return whether a completed run both stayed safe and arrived in tolerance."""
        return outcome.all_constraints_satisfied and outcome.terminal_position_error_m <= tolerance

    results = run_campaign(
        nominal,
        effective_settings.dispersions(),
        _execute_dispersed_run,
        n_runs,
        seed,
        retain=_RETAIN,
        success_fn=_success,
        keep_results=False,
        map_fn=map_fn,
    )

    summary = results.summary(percentiles=percentiles, confidence=confidence)
    per_run = {name: results.metric_values(name) for name in _RETAIN}
    completed = np.array(
        [record.index for record in results.records if not record.failed], dtype=np.int64
    )
    # The breach flags are retained as exactly 0.0 or 1.0, so the sum is an integer count
    # that happens to be stored as a float; round() rather than int() so that a hypothetical
    # 0.9999999 would be caught by the strict equality check below instead of truncated to 0.
    breach_rates: dict[str, ProportionEstimate] = {}
    for name in _CONSTRAINT_NAMES:
        total = float(np.sum(per_run[_BREACH_METRIC[name]]))
        count = round(total)
        if abs(total - count) > 0.0:
            raise CampaignSetupError(
                f"breach flag {_BREACH_METRIC[name]!r} summed to {total!r}, which is not an "
                "integer count. The flag is retained as 0.0 or 1.0 per run, so a fractional "
                "total means the retention policy and this reduction disagree."
            )
        breach_rates[name] = proportion_estimate(count, results.n_runs, confidence=confidence)

    return DispersedCampaignReport(
        seed=results.seed,
        n_runs=results.n_runs,
        n_failed=results.n_failures,
        n_completed=results.n_runs - results.n_failures,
        n_succeeded=results.n_successes,
        n_completed_but_violated=results.n_runs - results.n_failures - results.n_successes,
        success_rate=results.success_rate(confidence=confidence),
        completion_rate=results.completion_rate(confidence=confidence),
        breach_rates=breach_rates,
        metrics=summary.metrics,
        per_run_metrics=per_run,
        completed_indices=completed,
        failure_counts_by_type=summary.failure_counts_by_type(),
        terminal_tolerance_m=tolerance,
        confidence=float(confidence),
        dispersions=effective_settings.describe(),
        summary=summary,
    )


def _build_nominal(
    config: ScenarioConfig,
    settings: DispersionSettings,
    *,
    n_samples: int,
    dynamics: str,
    propagate_fn: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]
    | None,
    cross_track_tolerance_m: float,
) -> _Nominal:
    """Resolve the scenario once, so no run repeats the setup.

    The constraint objects, the time grid and the target's inertial state are identical in
    every run by definition -- they are the *nominal* -- so building them per run would cost
    a thousand redundant validations and, worse, would let a future edit disperse one of
    them by accident.
    """
    r_target_eci_m, v_target_eci_m_s = target_state_eci(config.orbit)
    times_s = np.linspace(0.0, config.tof_s, n_samples, dtype=np.float64)

    truth_propagator = propagate_fn
    if truth_propagator is None:
        if dynamics == "cw":
            truth_propagator = cw_truth_propagator(config.orbit.mean_motion_rad_s)
        else:
            truth_propagator = _nonlinear_propagator(
                r_target_eci_m,
                v_target_eci_m_s,
                rtol=config.integrator.rtol,
                atol=config.integrator.atol,
            )

    constraints = config.constraints
    return _Nominal(
        n_rad_s=config.orbit.mean_motion_rad_s,
        times_s=times_s,
        nominal_initial_state_hill=np.concatenate(
            (
                np.asarray(config.start_hold_point.position_hill_m, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        ),
        commanded_terminal_state_hill=np.concatenate(
            (
                np.asarray(config.target_hold_point.position_hill_m, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        ),
        corridor_axis_hill=np.asarray(CORRIDOR_AXIS_HILL, dtype=np.float64),
        settings=settings,
        propagate_fn=truth_propagator,
        cross_track_tolerance_m=float(cross_track_tolerance_m),
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


# --------------------------------------------------------------------------------------
# Sensitivity (F-5.5)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityPoint:
    """One campaign in a sensitivity sweep: which family was scaled, by how much, result.

    Attributes
    ----------
    source
        ``"burn_execution"``, ``"navigation"`` or ``"initial_state"``.
    scale
        Factor applied to that family's one-sigmas. ``0.0`` removes the family entirely and
        ``1.0`` is the baseline setting.
    report
        The campaign report at that point.

    """

    source: str
    scale: float
    report: DispersedCampaignReport


#: Families :func:`sensitivity_sweep` can scale.
_SENSITIVITY_SOURCES: tuple[str, ...] = ("burn_execution", "navigation", "initial_state")


def sensitivity_sweep(
    config: ScenarioConfig,
    n_runs: int,
    seed: int,
    *,
    sources: Sequence[str] = ("burn_execution", "navigation"),
    scales: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    settings: DispersionSettings | None = None,
    **kwargs: Any,
) -> tuple[SensitivityPoint, ...]:
    """Scale one dispersion family at a time and report the effect (SRS F-5.5).

    Every point uses the **same seed**. That is common random numbers, and it is deliberate:
    with a shared seed the difference between two points is the effect of the scaling, not
    the difference between two draws. Using a fresh seed per point would add sampling noise
    of the same order as the effect being measured at small ``n_runs``, which is the usual
    way a sensitivity study concludes that nothing matters.

    Note the substream scheme makes this stronger than it sounds: run ``i``'s burn draw
    comes from a substream keyed on ``(seed, i, "burn_execution")``, so scaling the
    *navigation* dispersion leaves every run's burn draw bitwise unchanged rather than
    merely identically distributed.

    Parameters
    ----------
    config
        Scenario to sweep.
    n_runs
        Runs **per point**. The total cost is ``len(sources) * len(scales) * n_runs``.
    seed
        Shared campaign seed.
    sources
        Families to scale, from ``"burn_execution"``, ``"navigation"``, ``"initial_state"``.
    scales
        Factors on the one-sigmas.
    settings
        Baseline settings the factors multiply. Defaults to
        :meth:`DispersionSettings.realistic`.
    **kwargs
        Forwarded to :func:`run_dispersed_rendezvous`.

    Returns
    -------
    tuple of SensitivityPoint
        In ``sources`` x ``scales`` order.

    Raises
    ------
    CampaignSetupError
        If a source name is not recognised or a scale is negative.

    """
    base = DispersionSettings.realistic() if settings is None else settings
    unknown = [name for name in sources if name not in _SENSITIVITY_SOURCES]
    if unknown:
        raise CampaignSetupError(
            f"unknown sensitivity source(s) {unknown}; known sources are "
            f"{list(_SENSITIVITY_SOURCES)}"
        )

    points: list[SensitivityPoint] = []
    for source in sources:
        for raw_scale in scales:
            scale = float(raw_scale)
            if source == "burn_execution":
                scaled = base.scaled(burn_execution=scale)
            elif source == "navigation":
                scaled = base.scaled(navigation=scale)
            else:
                scaled = base.scaled(initial_state=scale)
            points.append(
                SensitivityPoint(
                    source=source,
                    scale=scale,
                    report=run_dispersed_rendezvous(
                        config, n_runs, seed, settings=scaled, **kwargs
                    ),
                )
            )
    return tuple(points)
