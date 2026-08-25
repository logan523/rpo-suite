r"""Differential correction of a two-impulse transfer onto nonlinear relative dynamics.

Why this module exists
----------------------
:func:`rpo_core.relative.cw.two_impulse_transfer` is *exact* for the Clohessy-Wiltshire
model, and the CW model is wrong by a measured amount: the one-orbit position error is
``6*pi * rho**2 / r`` (see ``docs/cw_validity.md``), which is 2.77 m at 1 km separation and
277 m at 10 km. A burn planned against the linear model therefore does not arrive where it
was aimed once the real dynamics are applied. This module closes that gap by shooting: it
takes the CW impulse as a first guess and corrects it until the trajectory *actually*
arrives, with arrival evaluated by
:func:`rpo_core.relative.nonlinear.propagate_relative_nonlinear`.

Measured on the flagship half-period V-bar hop (420 km circular target, chaser hopping from
``-rho`` to ``-rho/4`` along V-bar), terminal position miss under nonlinear dynamics:

===========  ==================  ================
Separation   Raw CW miss (m)     Corrected (m)
===========  ==================  ================
100 m        1.225e-02           < 1e-6
1 km         1.225e+00           < 1e-6
10 km        1.225e+02           < 1e-6
===========  ==================  ================

The equations
-------------
Let :math:`\rho_0` be the initial Hill-frame relative position, :math:`v_0` the initial
relative velocity, and :math:`\Delta v_1` the departure impulse. Write
:math:`\mathcal{N}(\Delta v_1)` for the nonlinear flow: propagate target and chaser
independently under two-body motion from the state implied by
:math:`(\rho_0,\; v_0 + \Delta v_1)`, then difference them in the target's Hill frame at
``tof_s``. The shooting residual is the terminal **position** miss

.. math::

    F(\Delta v_1) \;=\; \big[\mathcal{N}(\Delta v_1)\big]_{\text{pos}} \;-\; \rho_f ,

three equations in the three unknowns :math:`\Delta v_1`. Newton's method needs the
sensitivity :math:`J = \partial F / \partial \Delta v_1`, estimated column-wise by forward
differences with step :math:`h`,

.. math::

    J_{:,i} \;\approx\; \frac{F(\Delta v_1 + h\,e_i) - F(\Delta v_1)}{h},

and iterates :math:`\Delta v_1 \leftarrow \Delta v_1 + \lambda\, s` with
:math:`s = -J^{-1} F`. The arrival impulse is then recomputed against the *achieved*
nonlinear arrival velocity rather than the CW prediction,

.. math::

    \Delta v_2 \;=\; \dot\rho_f - \big[\mathcal{N}(\Delta v_1)\big]_{\text{vel}} .

Recomputing :math:`\Delta v_2` is not cosmetic: at 10 km separation the CW arrival impulse
is wrong by 4.21e-03 m/s out of 3.09 m/s (1.4e-3 relative), which is a real residual drift.

Choice of finite-difference step
--------------------------------
``DEFAULT_FD_STEP_M_S = 1e-4`` m/s, chosen from measurement, not from feel. See that
constant's documentation for the noise-floor sweep that fixes it and for why forward
differences are used rather than central.

Rank structure, and the trap it hides
-------------------------------------
:math:`J` inherits CW's block structure: in-plane (x, y) and cross-track (z) decouple to
within :math:`O(\rho/r)`, and they lose rank at *different* transfer times -- in-plane at
whole orbital periods, cross-track at half periods. A single 3x3 condition-number test does
**not** catch the cross-track case: at exactly half a period the measured
:math:`\mathrm{cond}(J)` is 1.0e6, comfortably below
:data:`rpo_core.relative.cw.SINGULARITY_CONDITION_LIMIT` (1e8), yet the 3x3 Newton step it
produces asks for a cross-track impulse of **-52 km/s**. The nonzero :math:`J_{22}` there is
pure nonlinear coupling leaking into a structurally singular entry. This module therefore
checks the in-plane block and the cross-track entry separately, exactly as
:func:`~rpo_core.relative.cw.two_impulse_transfer` does, and for the same reason: a single
3x3 test would either reject the perfectly well-posed half-period V-bar hop or -- worse, as
here -- accept it and return garbage.

Validity
--------
The oracle is two-body only: no J2, no drag, no third bodies, no finite-burn modelling.
Impulses are instantaneous. The CW seed and the derived mean motion assume a near-circular
target; for an eccentric target pass ``n_rad_s`` explicitly or supply ``dv1_guess_m_s``.
Convergence is to the *nonlinear two-body* arrival, so what this module removes is CW's
linearisation error and nothing else -- it does not make the trajectory more physical than
the propagator underneath it.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._validate import as_vector
from .constants import MU_EARTH_M3_S2
from .exceptions import InfeasibleTransferError, RpoCoreError
from .propagate import DEFAULT_ATOL, DEFAULT_RTOL
from .relative.cw import (
    DEFAULT_FEASIBILITY_TOL_M,
    SINGULARITY_CONDITION_LIMIT,
    two_impulse_transfer,
)
from .relative.nonlinear import propagate_relative_nonlinear

#: Forward-difference step (m/s) used to estimate the shooting Jacobian.
#:
#: **Measured, not guessed.** The residual map is bitwise deterministic (identical inputs
#: reproduce identical output exactly), so the "noise" that limits a finite difference is
#: not run-to-run scatter but the non-smooth dependence of an adaptive-step integrator on
#: its initial condition. Two independent measurements of that floor, on a 420 km circular
#: target with DOP853 at ``rtol = atol = 1e-12``:
#:
#: * Tightening the integrator tolerance by 10x and 100x moves the terminal position by
#:   2.5e-08 to 1.5e-07 m. That is the accuracy floor of the oracle itself.
#: * Sweeping ``h`` and comparing the recovered Jacobian against the analytic CW
#:   ``Phi_rv``, the relative error sits on a flat plateau of 1.09e-04 (at 1 km separation)
#:   for every ``h`` from 1e-2 down to 1e-6, then degrades: 2.6e-04 at ``h = 1e-7`` and
#:   3.6e-03 at ``h = 1e-8`` as the difference drowns in the 1e-7 m floor. Above the
#:   plateau, truncation appears: 1.23e-04 at ``h = 1e-1``.
#:
#: The plateau therefore spans ``1e-6 <= h <= 1e-2`` and ``1e-4`` is its geometric centre,
#: two decades from the truncation knee and two from the roundoff knee. The plateau value
#: itself is *not* finite-difference error -- it is the genuine
#: nonlinear-versus-CW difference, and it scales with separation exactly as a linearisation
#: error should (1.09e-04 at 1 km, 1.09e-03 at 10 km), which is what confirms the sweep is
#: measuring physics rather than arithmetic.
#:
#: **Forward, not central.** At ``h = 1e-4`` forward and central differences agree to all
#: printed digits (1.085e-04 versus 1.085e-04 at 1 km; 1.088e-03 versus 1.088e-03 at 10 km).
#: Central differences only widen the usable plateau by one decade at the bottom end, which
#: buys nothing when the operating point is four decades above it, and they double the
#: Jacobian cost from 3 to 6 propagations per iteration -- 24 ms to 48 ms at the measured
#: 8 ms per evaluation. Paying 2x for an improvement that does not register is not a
#: robustness margin, it is a slower answer.
DEFAULT_FD_STEP_M_S: float = 1.0e-4

#: Default convergence tolerance on terminal position miss, metres.
#:
#: The measured floor is ~5e-09 m (the tightest residual reachable before the line search
#: stalls, at 1 km separation over 0.4 periods), so 1 mm carries roughly five decades of
#: headroom while sitting far below any operational requirement -- the reference scenario's
#: keep-out sphere is 200 m and its error budget 5 m. Ask for less than ~1e-8 m and the
#: solver will correctly raise :class:`TargetingConvergenceError` rather than pretend.
DEFAULT_TOLERANCE_M: float = 1.0e-3

#: Default cap on Newton iterations before :class:`TargetingConvergenceError` is raised.
#:
#: Generous on purpose. Measured iteration counts from the CW seed: 1 at 100 m, 1 at 1 km,
#: 2 at 10 km, 3 at 100 km, 4 at 500 km, and 9 at a frankly unphysical 5000 km separation
#: (a separation comparable to the orbit radius itself). 25 is far above anything the
#: module's real envelope needs, so hitting it means the problem is wrong, not slow.
DEFAULT_MAX_ITERATIONS: int = 25

#: ``|J[2,2]| / ||J_in-plane||_2`` below which cross-track targeting is rank-deficient.
#:
#: Measured on both sides. At transfer times that are *not* half-period multiples the ratio
#: never drops below **3.3e-03** (its tightest value, at 0.99 periods; it is 8.2e-02 at 0.4
#: periods). At exactly half a period the ratio collapses to **9.9e-08** at 100 m
#: separation, rising to **8.2e-05** at 100 km as nonlinear coupling leaks into the
#: structurally-zero entry. 5e-4 is the balanced split: 6.1x above the worst rank-deficient
#: case measured and 6.6x below the tightest healthy one.
DEFAULT_CROSS_TRACK_RANK_TOL: float = 5.0e-4

#: Smallest line-search fraction tried before the step is declared stalled (2**-12).
#:
#: A step this heavily damped is no longer a Newton step; if it still does not reduce the
#: residual, the iteration has reached the noise floor and no further progress exists.
MIN_LINE_SEARCH_FRACTION: float = 2.0**-12


class TargetingConvergenceError(RpoCoreError, RuntimeError):
    """Raised when differential correction did not reach the requested tolerance.

    Carries the iteration count, the final residual in metres, and the full residual
    history, so a caller can distinguish "asked for a tolerance below the integrator's
    noise floor" (history flattens out just above the request) from "the shooting problem
    is diverging" (history grows) without attaching a debugger.

    Deliberately not recoverable into a partial answer. The last iterate of a non-converged
    shooting solve is a delta-v that looks entirely reasonable and misses the target by an
    unbounded amount -- returning it is the precise failure mode this package exists to
    prevent, and it is worse here than in a pure solver because the number flows onward into
    a burn plan.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        residual_m: float,
        residual_history_m: tuple[float, ...],
    ) -> None:
        """Record the iteration count, final residual (m), and full residual history (m)."""
        super().__init__(message)
        self.iterations = iterations
        self.residual_m = residual_m
        self.residual_history_m = residual_history_m


class IllConditionedJacobianError(RpoCoreError, ValueError):
    """Raised when the shooting Jacobian cannot be inverted to a trustworthy step.

    Distinct from :class:`TargetingConvergenceError`: the iteration did not fail to
    converge, it could not legitimately take a step at all. Carries the measured condition
    number and the iteration at which the Jacobian went bad, because a Jacobian that is
    well-conditioned at iteration 0 and singular at iteration 4 is a different diagnosis
    -- the iterate has wandered -- from one that is singular on the first evaluation, which
    means the requested transfer time is degenerate.
    """

    def __init__(self, message: str, *, condition_number: float, iteration: int) -> None:
        """Record the measured condition number and the iteration that produced it."""
        super().__init__(message)
        self.condition_number = condition_number
        self.iteration = iteration


@dataclass(frozen=True, eq=False)
class CorrectedTransfer:
    """Result of :func:`correct_two_impulse_transfer`.

    Attributes
    ----------
    dv1_hill_m_s, dv2_hill_m_s
        The corrected departure and arrival impulses, shape (3,), m/s, Hill frame.
    arrival_state_hill
        Shape (6,): the state the coast *actually* delivers under nonlinear dynamics,
        immediately **before** ``dv2_hill_m_s`` is applied. Its position block is the
        achieved terminal position, and its miss against the commanded position is
        ``residual_history_m[-1]``.
    terminal_state_hill
        Shape (6,): ``arrival_state_hill`` with ``dv2_hill_m_s`` applied. Its velocity block
        equals the commanded terminal velocity by construction -- that is what ``dv2`` is
        for -- so this is the achieved terminal state of the manoeuvre as a whole.
    residual_history_m
        Terminal position miss at every accepted iterate, metres. ``[0]`` is the miss of the
        raw CW guess (or of ``dv1_guess_m_s``) and ``[-1]`` the converged miss, so
        ``residual_history_m[0] / residual_history_m[-1]`` is the improvement this module
        bought. Non-increasing whenever ``damping`` is true.
    iterations
        Number of accepted Newton steps, equal to ``len(residual_history_m) - 1``. Zero
        means the initial guess already met ``tolerance_m``.
    cw_dv1_hill_m_s, cw_dv2_hill_m_s
        The uncorrected CW impulses used as the initial guess, retained so the caller can
        report the correction rather than recomputing it.

    """

    dv1_hill_m_s: npt.NDArray[np.float64]
    dv2_hill_m_s: npt.NDArray[np.float64]
    arrival_state_hill: npt.NDArray[np.float64]
    terminal_state_hill: npt.NDArray[np.float64]
    residual_history_m: tuple[float, ...]
    iterations: int
    cw_dv1_hill_m_s: npt.NDArray[np.float64]
    cw_dv2_hill_m_s: npt.NDArray[np.float64]

    @property
    def final_residual_m(self) -> float:
        """Terminal position miss actually achieved, metres."""
        return self.residual_history_m[-1]

    @property
    def initial_residual_m(self) -> float:
        """Terminal position miss of the uncorrected initial guess, metres."""
        return self.residual_history_m[0]

    @property
    def dv1_correction_m_s(self) -> float:
        """Magnitude of the departure-impulse correction ``|dv1 - dv1_cw|``, m/s.

        Scales as the square of the separation, mirroring the ``rho**2`` CW error law:
        measured 1.15e-08, 1.15e-06, 1.15e-04 and 1.15e-02 m/s at 10 m, 100 m, 1 km and
        10 km. Near zero here is the correct answer at small separation, not a no-op bug.
        """
        return float(np.linalg.norm(self.dv1_hill_m_s - self.cw_dv1_hill_m_s))


def _positive_float(value: float, name: str) -> float:
    """Return ``value`` as a validated finite strictly-positive float."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}")
    return number


def correct_two_impulse_transfer(
    r_target0_eci_m: npt.ArrayLike,
    v_target0_eci_m_s: npt.ArrayLike,
    r0_hill_m: npt.ArrayLike,
    v0_hill_m_s: npt.ArrayLike,
    rf_hill_m: npt.ArrayLike,
    vf_hill_m_s: npt.ArrayLike,
    tof_s: float,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    n_rad_s: float | None = None,
    dv1_guess_m_s: npt.ArrayLike | None = None,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    fd_step_m_s: float = DEFAULT_FD_STEP_M_S,
    damping: bool = True,
    max_condition: float = SINGULARITY_CONDITION_LIMIT,
    cross_track_rank_tol: float = DEFAULT_CROSS_TRACK_RANK_TOL,
    feasibility_tol_m: float = DEFAULT_FEASIBILITY_TOL_M,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> CorrectedTransfer:
    """Correct a two-impulse CW transfer so it arrives under nonlinear relative dynamics.

    Seeds from :func:`~rpo_core.relative.cw.two_impulse_transfer`, then Newton-iterates on
    the departure impulse until the terminal position miss measured by
    :func:`~rpo_core.relative.nonlinear.propagate_relative_nonlinear` falls below
    ``tolerance_m``. The arrival impulse is recomputed against the achieved nonlinear
    arrival velocity.

    Parameters
    ----------
    r_target0_eci_m, v_target0_eci_m_s
        Target inertial state at the epoch, shape (3,) each, metres and m/s.
    r0_hill_m, v0_hill_m_s
        Initial relative position (m) and velocity (m/s) in the target Hill frame,
        shape (3,) each.
    rf_hill_m, vf_hill_m_s
        Commanded terminal relative position (m) and velocity (m/s), shape (3,) each.
    tof_s
        Time of flight, seconds. Must be finite and strictly positive.
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Defaults to Earth.
    n_rad_s
        Mean motion for the CW seed, rad/s. Defaults to ``sqrt(mu / |r_target0|**3)``,
        which is the exact mean motion for a circular target and the natural
        osculating value otherwise. Supply it explicitly for an eccentric target.
    dv1_guess_m_s
        Departure impulse to start from, shape (3,), m/s. Defaults to the CW solution.
        Supplying it bypasses the CW solve entirely, which is how a caller warm-starts from
        a neighbouring solution, and the only way to attempt a transfer time at which CW
        itself is singular.
    tolerance_m
        Convergence tolerance on terminal position miss, metres. See
        :data:`DEFAULT_TOLERANCE_M`; the measured floor is ~5e-09 m.
    max_iterations
        Cap on Newton steps. See :data:`DEFAULT_MAX_ITERATIONS`.
    fd_step_m_s
        Forward-difference step for the Jacobian, m/s. See :data:`DEFAULT_FD_STEP_M_S` for
        the noise-floor measurement that fixes the default.
    damping
        When true (the default), backtrack the Newton step by halving until the residual
        actually decreases, which makes ``residual_history_m`` monotone **by construction**
        rather than by luck. Measured cost when it is not needed: one extra propagation per
        iteration (~8 ms), since full step length is accepted on the first trial throughout
        the module's real envelope. Measured benefit where it is: undamped Newton diverges
        at 5000 km separation (residual reaching 2.6e+09 m and never converging in 40
        iterations) where the damped iteration converges monotonically in 9.
    max_condition
        Condition-number limit on the in-plane block, identifying the exact singularity at
        whole-period transfer times. Defaults to the CW module's own limit.
    cross_track_rank_tol
        Relative cross-track sensitivity below which cross-track targeting is treated as
        rank-deficient. See :data:`DEFAULT_CROSS_TRACK_RANK_TOL`.
    feasibility_tol_m
        Cross-track position tolerance for accepting a rank-deficient request, metres.
    rtol, atol
        Integrator tolerances for the nonlinear oracle.

    Returns
    -------
    CorrectedTransfer
        Corrected impulses, achieved terminal state, residual history and iteration count.

    Raises
    ------
    TargetingConvergenceError
        If ``tolerance_m`` was not met within ``max_iterations``, or if the damped step
        stalled because no fraction down to :data:`MIN_LINE_SEARCH_FRACTION` reduced the
        residual. No best-effort iterate is returned in either case.
    IllConditionedJacobianError
        If the in-plane block of the shooting Jacobian is singular or ill-conditioned,
        i.e. the transfer time is at or near a whole number of orbital periods.
    InfeasibleTransferError
        If cross-track targeting is rank-deficient (transfer time at a half-period multiple)
        *and* the commanded terminal cross-track position is not the one the coast is pinned
        to. Also raised by the CW seed for the same reason.
    SingularTransferTimeError
        From the CW seed, when the in-plane CW solve is singular at ``tof_s``. Supply
        ``dv1_guess_m_s`` to attempt such a transfer time anyway.
    PropagationError
        If the nonlinear oracle fails, which is what a commanded terminal state that would
        drive the chaser into the central body looks like. Deliberately not translated:
        the integrator's own message localises the failure better than a rewrite would.
    ValueError
        On malformed or non-finite input, non-positive ``tof_s``, ``tolerance_m``,
        ``fd_step_m_s`` or ``mu_m3_s2``, or ``max_iterations < 1``.

    Examples
    --------
    A half-period V-bar hop at 10 km, where CW misses by 122 m:

    >>> import math
    >>> import numpy as np
    >>> from rpo_core.constants import (
    ...     MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M, orbital_period_s)
    >>> a = R_EARTH_EQUATORIAL_M + 420e3
    >>> v_c = math.sqrt(MU_EARTH_M3_S2 / a)
    >>> result = correct_two_impulse_transfer(
    ...     [a, 0.0, 0.0], [0.0, v_c, 0.0],
    ...     [0.0, -10_000.0, 0.0], [0.0, 0.0, 0.0],
    ...     [0.0, -2_500.0, 0.0], [0.0, 0.0, 0.0],
    ...     0.5 * orbital_period_s(a))
    >>> bool(result.initial_residual_m > 100.0)
    True
    >>> bool(result.final_residual_m < 1e-3)
    True

    """
    r_t0 = as_vector(r_target0_eci_m, "r_target0_eci_m")
    v_t0 = as_vector(v_target0_eci_m_s, "v_target0_eci_m_s")
    r0 = as_vector(r0_hill_m, "r0_hill_m")
    v0 = as_vector(v0_hill_m_s, "v0_hill_m_s")
    rf = as_vector(rf_hill_m, "rf_hill_m")
    vf = as_vector(vf_hill_m_s, "vf_hill_m_s")

    tof = _positive_float(tof_s, "tof_s")
    mu = _positive_float(mu_m3_s2, "mu_m3_s2")
    tolerance = _positive_float(tolerance_m, "tolerance_m")
    step_size = _positive_float(fd_step_m_s, "fd_step_m_s")
    _positive_float(cross_track_rank_tol, "cross_track_rank_tol")

    iteration_cap = int(max_iterations)
    if iteration_cap < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")

    target_radius = float(np.linalg.norm(r_t0))
    if target_radius <= 0.0:
        raise ValueError("r_target0_eci_m must be a nonzero position vector")
    mean_motion = (
        math.sqrt(mu / target_radius**3) if n_rad_s is None else _positive_float(n_rad_s, "n_rad_s")
    )

    # The CW seed is computed even when dv1_guess_m_s overrides it: the caller gets the
    # uncorrected impulses back for comparison, and the correction is only meaningful as a
    # delta against them. When CW itself is singular that comparison is unavailable, which
    # is precisely the case dv1_guess_m_s exists to serve, so the failure is absorbed there
    # and nowhere else.
    if dv1_guess_m_s is None:
        cw_dv1, cw_dv2 = two_impulse_transfer(mean_motion, r0, v0, rf, vf, tof)
        dv1 = cw_dv1.copy()
    else:
        dv1 = as_vector(dv1_guess_m_s, "dv1_guess_m_s").copy()
        try:
            cw_dv1, cw_dv2 = two_impulse_transfer(mean_motion, r0, v0, rf, vf, tof)
        except RpoCoreError:
            cw_dv1 = np.full(3, np.nan, dtype=np.float64)
            cw_dv2 = np.full(3, np.nan, dtype=np.float64)

    times = np.array([0.0, tof], dtype=np.float64)

    def arrival(departure_dv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the Hill-frame state at ``tof`` for a given departure impulse."""
        initial = np.concatenate((r0, v0 + departure_dv))
        trajectory = propagate_relative_nonlinear(
            r_t0, v_t0, initial, times, mu, rtol=rtol, atol=atol
        )
        return np.asarray(trajectory[-1], dtype=np.float64)

    state = arrival(dv1)
    residual = state[:3] - rf
    residual_norm = float(np.linalg.norm(residual))
    history: list[float] = [residual_norm]

    for iteration in range(iteration_cap):
        if residual_norm <= tolerance:
            break

        # --- Jacobian by forward differences ---------------------------------------
        jacobian = np.empty((3, 3), dtype=np.float64)
        for axis in range(3):
            perturbed = dv1.copy()
            perturbed[axis] += step_size
            jacobian[:, axis] = (arrival(perturbed)[:3] - state[:3]) / step_size

        # --- Rank checks, in-plane and cross-track separately ----------------------
        # Both blocks are checked every iteration rather than once up front: the iterate
        # moves, and a Jacobian that was healthy at the seed can degrade as it does.
        in_plane = jacobian[np.ix_([0, 1], [0, 1])]
        condition = float(np.linalg.cond(in_plane))
        if not math.isfinite(condition) or condition > max_condition:
            periods = tof * mean_motion / (2.0 * math.pi)
            raise IllConditionedJacobianError(
                "in-plane shooting Jacobian is singular or ill-conditioned at "
                f"tof_s={tof:.6g} s (= {periods:.6f} orbital periods): condition number "
                f"{condition:.3e} exceeds max_condition={max_condition:.3e}, at Newton "
                f"iteration {iteration}. In-plane targeting loses rank at integer multiples "
                "of the orbital period; choose a different time of flight or add an "
                "intermediate manoeuvre.",
                condition_number=condition,
                iteration=iteration,
            )

        in_plane_scale = float(np.linalg.norm(in_plane, 2))
        cross_track_ratio = abs(float(jacobian[2, 2])) / in_plane_scale
        if cross_track_ratio < cross_track_rank_tol:
            # Structurally rank-deficient in z. Whatever nonzero value J[2,2] carries here
            # is nonlinear coupling, not sensitivity: inverting it produced a -52 km/s
            # cross-track impulse in the measurement that motivated this branch. Either the
            # commanded cross-track position is already met, or it is unreachable.
            if abs(float(residual[2])) > feasibility_tol_m:
                half_periods = tof * mean_motion / math.pi
                raise InfeasibleTransferError(
                    "cross-track shooting is rank-deficient at "
                    f"tof_s={tof:.6g} s (= {half_periods:.6f} half-periods): the relative "
                    f"cross-track sensitivity |J[2,2]|/||J_in-plane|| is "
                    f"{cross_track_ratio:.3e}, below cross_track_rank_tol="
                    f"{cross_track_rank_tol:.3e}, while the commanded terminal cross-track "
                    f"position is still missed by {float(residual[2]):.6g} m (> "
                    f"feasibility_tol_m={feasibility_tol_m:.3e} m). Cross-track position "
                    "cannot be changed at integer multiples of the half period; choose a "
                    "different time of flight."
                )
            step = np.zeros(3, dtype=np.float64)
            step[[0, 1]] = -np.linalg.solve(in_plane, residual[[0, 1]])
        else:
            full_condition = float(np.linalg.cond(jacobian))
            if not math.isfinite(full_condition) or full_condition > max_condition:
                raise IllConditionedJacobianError(
                    "shooting Jacobian is singular or ill-conditioned at "
                    f"tof_s={tof:.6g} s: condition number {full_condition:.3e} exceeds "
                    f"max_condition={max_condition:.3e}, at Newton iteration {iteration}, "
                    "with the in-plane block healthy "
                    f"(condition {condition:.3e}) and cross-track sensitivity "
                    f"{cross_track_ratio:.3e}.",
                    condition_number=full_condition,
                    iteration=iteration,
                )
            step = np.asarray(-np.linalg.solve(jacobian, residual), dtype=np.float64)

        if not np.all(np.isfinite(step)):  # pragma: no cover - defensive
            raise IllConditionedJacobianError(
                f"shooting Jacobian produced a non-finite Newton step {step!r} at "
                f"iteration {iteration}",
                condition_number=condition,
                iteration=iteration,
            )

        # --- Step acceptance -------------------------------------------------------
        if damping:
            fraction = 1.0
            accepted = False
            while fraction >= MIN_LINE_SEARCH_FRACTION:
                trial_dv1 = dv1 + fraction * step
                trial_state = arrival(trial_dv1)
                trial_residual = trial_state[:3] - rf
                trial_norm = float(np.linalg.norm(trial_residual))
                if trial_norm < residual_norm:
                    accepted = True
                    break
                fraction *= 0.5
            if not accepted:
                raise TargetingConvergenceError(
                    "differential correction stalled at "
                    f"tof_s={tof:.6g} s (= {tof * mean_motion / (2.0 * math.pi):.6f} "
                    "orbital periods): no damped Newton step down to "
                    f"{MIN_LINE_SEARCH_FRACTION:.3e} of full length reduced the terminal "
                    f"position miss below its current {residual_norm:.6g} m, after "
                    f"{iteration} iterations against a tolerance of {tolerance:.6g} m. "
                    "The iteration has reached the noise floor of the nonlinear oracle "
                    f"(~5e-09 m at rtol={rtol:.1e}); ask for a tolerance it can deliver. "
                    "The last iterate is not returned: it misses by an amount this "
                    "solver cannot certify.",
                    iterations=iteration,
                    residual_m=residual_norm,
                    residual_history_m=tuple(history),
                )
            dv1 = trial_dv1
            state = trial_state
            residual = trial_residual
            residual_norm = trial_norm
        else:
            dv1 = dv1 + step
            state = arrival(dv1)
            residual = state[:3] - rf
            residual_norm = float(np.linalg.norm(residual))
        history.append(residual_norm)

    if residual_norm > tolerance:
        raise TargetingConvergenceError(
            f"differential correction did not converge: after {len(history) - 1} "
            f"iterations the terminal position miss is {residual_norm:.6g} m against a "
            f"tolerance of {tolerance:.6g} m (max_iterations={iteration_cap}). The initial "
            f"guess missed by {history[0]:.6g} m. Returning this iterate would hand back a "
            "delta-v that looks reasonable and does not arrive.",
            iterations=len(history) - 1,
            residual_m=residual_norm,
            residual_history_m=tuple(history),
        )

    # Recompute the arrival impulse against the velocity the nonlinear coast actually
    # delivers. Reusing the CW dv2 here is wrong by 1.4e-3 relative at 10 km separation
    # (4.21e-03 m/s out of 3.09 m/s), which is a real post-manoeuvre drift.
    dv2 = vf - state[3:]
    terminal_state = np.concatenate((state[:3], state[3:] + dv2))

    return CorrectedTransfer(
        dv1_hill_m_s=dv1,
        dv2_hill_m_s=dv2,
        arrival_state_hill=state,
        terminal_state_hill=terminal_state,
        residual_history_m=tuple(history),
        iterations=len(history) - 1,
        cw_dv1_hill_m_s=cw_dv1,
        cw_dv2_hill_m_s=cw_dv2,
    )


def raw_cw_terminal_miss_m(
    r_target0_eci_m: npt.ArrayLike,
    v_target0_eci_m_s: npt.ArrayLike,
    r0_hill_m: npt.ArrayLike,
    v0_hill_m_s: npt.ArrayLike,
    rf_hill_m: npt.ArrayLike,
    vf_hill_m_s: npt.ArrayLike,
    tof_s: float,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    n_rad_s: float | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Return how far the *uncorrected* CW impulse misses under nonlinear dynamics, metres.

    The baseline this module is measured against. Flies the CW departure impulse through
    :func:`~rpo_core.relative.nonlinear.propagate_relative_nonlinear` and reports the
    terminal position miss, with no correction applied.

    Parameters are as for :func:`correct_two_impulse_transfer`.

    Returns
    -------
    float
        Terminal position miss of the raw CW solution, metres.

    Raises
    ------
    SingularTransferTimeError, InfeasibleTransferError
        From the underlying CW solve.
    PropagationError
        If the nonlinear propagation fails.
    ValueError
        On malformed or non-finite input.

    """
    r_t0 = as_vector(r_target0_eci_m, "r_target0_eci_m")
    v_t0 = as_vector(v_target0_eci_m_s, "v_target0_eci_m_s")
    r0 = as_vector(r0_hill_m, "r0_hill_m")
    v0 = as_vector(v0_hill_m_s, "v0_hill_m_s")
    rf = as_vector(rf_hill_m, "rf_hill_m")
    vf = as_vector(vf_hill_m_s, "vf_hill_m_s")
    tof = _positive_float(tof_s, "tof_s")
    mu = _positive_float(mu_m3_s2, "mu_m3_s2")

    target_radius = float(np.linalg.norm(r_t0))
    if target_radius <= 0.0:
        raise ValueError("r_target0_eci_m must be a nonzero position vector")
    mean_motion = (
        math.sqrt(mu / target_radius**3) if n_rad_s is None else _positive_float(n_rad_s, "n_rad_s")
    )

    dv1, _ = two_impulse_transfer(mean_motion, r0, v0, rf, vf, tof)
    initial = np.concatenate((r0, v0 + dv1))
    trajectory = propagate_relative_nonlinear(
        r_t0, v_t0, initial, np.array([0.0, tof]), mu, rtol=rtol, atol=atol
    )
    return float(np.linalg.norm(trajectory[-1, :3] - rf))
