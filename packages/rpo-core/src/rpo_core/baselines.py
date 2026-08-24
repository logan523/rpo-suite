r"""Three rendezvous strategies behind one problem statement and one result type.

Why this module exists
----------------------
The governing rule of this repository is that no algorithm is called superior without a
quantitative comparison on identical scenarios (SRS F-7.5). A rule like that is only
enforceable if the alternatives are *runnable*, from the *same* problem statement, into the
*same* result type. Otherwise the comparison degrades into three separately-tuned demos with
three different definitions of "arrived", which is how misleading benchmark tables get made.

So all three baselines here take a :class:`RendezvousProblem` and return a
:class:`BaselineResult`, and -- this is the load-bearing part -- **every one of them has its
terminal error measured by the same nonlinear two-body flight**, by
:func:`_fly_impulse_schedule`. Scoring a linear method under the linear model that produced
it would flatter it by construction: the Clohessy-Wiltshire two-impulse solve reproduces its
own commanded terminal state to 1e-9 m and misses by 73 m under real dynamics at 10 km
separation. Only the second number is a comparison.

The equations
-------------
**Phasing / Hohmann (F-7.1).** The classical far-range approach. Transfer the chaser to a
drift orbit of radius :math:`r_d`, coast there for :math:`k` revolutions while the
along-track phase builds at the differential rate, then transfer back onto the target's
radius :math:`r_t`. Each transfer is a Hohmann pair,

.. math::

    \Delta v_1 &= \sqrt{\mu/r_1}\left(\sqrt{\tfrac{2 r_2}{r_1 + r_2}} - 1\right) \\
    \Delta v_2 &= \sqrt{\mu/r_2}\left(1 - \sqrt{\tfrac{2 r_1}{r_1 + r_2}}\right) \\
    t_h        &= \pi \sqrt{\tfrac{((r_1 + r_2)/2)^3}{\mu}}

and the profile costs four burns. The drift radius is *not* a free parameter: it is fixed by
the phase the manoeuvre has to deliver. Over the whole profile the chaser sweeps exactly
:math:`2\pi (k + 1)` radians of true anomaly -- one revolution split between the two
half-transfers plus :math:`k` on the drift orbit -- while the target sweeps
:math:`n_t \cdot t_{\text{TOF}}`. The relative phase delivered is therefore

.. math::

    \Delta\theta(r_d) \;=\; 2\pi (k + 1) \;-\; n_t \big(t_h(r_1, r_d)
                        \;+\; k\,T(r_d) \;+\; t_h(r_d, r_t)\big),

one scalar equation in the one unknown :math:`r_d`, solved by ``scipy.optimize.brentq``.
:math:`\Delta\theta` is monotone decreasing in :math:`r_d` (a higher drift orbit is slower and
loses forward phase), so the root is unique inside the bracket and a failure to bracket is a
genuine statement about reachability rather than a solver artefact.

**Lambert (F-7.2).** Solve the fixed-time two-point boundary value problem between the two
*absolute* positions with :func:`rpo_core.lambert.solve_lambert`: depart from where the
chaser is now, arrive where the commanded terminal relative state will be once the target has
propagated for ``tof_s``. Two burns, exact two-body, no linearisation, and no free
parameters at all beyond the time of flight.

**CW two-impulse (F-7.3, seed).** :func:`rpo_core.relative.cw.two_impulse_transfer`, with an
optional differential correction by
:func:`rpo_core.targeting.correct_two_impulse_transfer` that re-aims the departure impulse
until the trajectory arrives under nonlinear dynamics.

Measured on the reference scenario (420 km circular target, chaser hopping from -10 km to
-2.5 km on V-bar, 0.4 orbits commanded for the fixed-time methods):

===========================  ==========  ===========  ==============
Method                       Δv (m/s)    TOF (s)      Term. miss (m)
===========================  ==========  ===========  ==============
Hohmann phasing              0.275       22 311.9     22.5
Lambert                      6.177       2 231.3      4.5e-06
CW two-impulse               6.168       2 231.3      73.4
CW two-impulse + correction  6.177       2 231.3      2.5e-04
===========================  ==========  ===========  ==============

That is the whole trade in four rows: phasing is 22x cheaper and 10x slower; Lambert and CW
cost the same to three significant figures and differ by seven orders of magnitude in where
they actually put the vehicle.

Validity
--------
Every result carries a :class:`Validity` flag and the sentence that justifies it, and
:func:`rpo_core.optimize.BaselineComparison.render_table` prints both. The flag answers
*is this method's own modelling premise satisfied in this regime* -- **not** *is it
accurate*, which is what the terminal-error columns are for. The two are separate questions
and collapsing them is how a table ends up recommending a method that cannot be trusted.

* **CW** is a linearisation with a measured envelope (``docs/cw_validity.md``): one-orbit
  position error ``6*pi*rho**2/r``, guarded by
  :func:`rpo_core.relative.conservative_cw_error_bound_m`. At 10 km separation the bound is
  277 m over one orbit against a 5 m budget, so CW is flagged **INVALID** there and VALID at
  250 m. A table that shows CW winning on Δv at a separation where it is invalid is worse
  than no table.
* **Phasing** assumes coplanar, near-circular, at-rest endpoints. It is flagged INVALID when
  the problem asks for a cross-track change, when either endpoint carries relative velocity,
  or when the chaser's initial osculating eccentricity breaks the circular premise.
* **Lambert** is exact for two-body motion and makes no linearisation, so on modelling
  grounds it is always VALID. Degenerate *geometry* is not silently absorbed into the flag:
  :func:`~rpo_core.lambert.solve_lambert` raises, and the exception propagates.

What every baseline here still neglects: J2, drag, third bodies, finite burn duration,
navigation and execution error. The comparison is between manoeuvre-design methods under
identical two-body dynamics, not a flight-fidelity claim.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from .constants import MU_EARTH_M3_S2
from .constraints import separation_m
from .elements import eccentricity_vector
from .exceptions import RpoCoreError
from .frames import hill_basis, relative_state_eci_to_hill, relative_state_hill_to_eci
from .lambert import solve_lambert
from .propagate import DEFAULT_ATOL, DEFAULT_RTOL, propagate_two_body, specific_energy_j_kg
from .relative.cw import two_impulse_transfer
from .relative.nonlinear import conservative_cw_error_bound_m
from .targeting import correct_two_impulse_transfer

__all__ = [
    "DEFAULT_CIRCULAR_ECCENTRICITY_LIMIT",
    "DEFAULT_CW_TOLERANCE_M",
    "DEFAULT_DRIFT_REVOLUTIONS",
    "DEFAULT_MAX_DRIFT_RADIUS_FRACTION",
    "DEFAULT_REST_TOL_M_S",
    "DEFAULT_TRAJECTORY_SAMPLES",
    "BaselineError",
    "BaselineResult",
    "Method",
    "PhasingGeometryError",
    "RendezvousProblem",
    "Validity",
    "cw_two_impulse_baseline",
    "hohmann_delta_v_m_s",
    "hohmann_transfer_time_s",
    "lambert_baseline",
    "phasing_baseline",
]

#: Position-error budget, metres, against which a method's modelling premise is judged.
#:
#: 2.5 % of the reference scenario's 200 m keep-out sphere, taken from ``docs/cw_validity.md``
#: -- which sets it there deliberately, after an earlier 1 % / 2 m budget made the flagship
#: half-orbit hop warn about itself (conservative bound 2.08 m, measured error 1.455 m). The
#: guard exists to catch CW being used at 10 km, where the bound is 277 m over one orbit, not
#: to police 2.00 against 2.08 m.
DEFAULT_CW_TOLERANCE_M: float = 5.0

#: Revolutions spent on the drift orbit by :func:`phasing_baseline`, default.
#:
#: Three is a design choice, not a measurement, and it is the single knob that sets where the
#: phasing baseline sits on the Δv-versus-time trade: more revolutions means a smaller drift
#: radius offset, which means less Δv and more time. Measured on the reference 10 km scenario,
#: k = 1 costs 0.6197 m/s over 2.000 orbits and k = 9 costs 0.1118 m/s over 10.000 orbits.
#:
#: The underlying law is on the drift *radius*, not on Δv: measured,
#: ``drift_radius_offset_m * (k + 1/2)`` is constant to 1.4e-05 relative across k = 1..12
#: (-803.14 m at 10 km separation, -79.651 m at 1 km). The half is the one revolution the two
#: half-transfers share, and they sit at half the offset. Δv does **not** inherit a clean
#: 1/(k+1) law from it, because the return transfer targets the target's radius while the
#: outbound one departs from the chaser's, which is 29.4 m higher in semi-major axis at 10 km
#: -- an additive term that does not shrink with k. Measured ``Δv * (k+1)`` accordingly runs
#: 1.239, 1.135, 1.100, 1.086, 1.118 for k = 1, 2, 3, 5, 9: not constant, and not even
#: monotone. Sweep it with :func:`rpo_core.optimize.phasing_delta_v_vs_tof` rather than
#: reasoning about it from a scaling law that does not hold.
DEFAULT_DRIFT_REVOLUTIONS: float = 3.0

#: Half-width of the drift-radius bracket handed to ``brentq``, as a fraction of the target
#: orbit radius. 5 % of a 420 km LEO is ±340 km, which spans every phase a sane profile can
#: need (±1.4 rad over three revolutions, i.e. ±9600 km of along-track) while staying inside
#: the regime where the two-body model is the only thing being neglected. The reference
#: scenario solves at an offset of -229 m, four orders of magnitude inside the bracket.
DEFAULT_MAX_DRIFT_RADIUS_FRACTION: float = 0.05

#: Relative speed, m/s, below which a Hill-frame endpoint counts as "at rest" for the
#: phasing baseline's circular-endpoint premise. One millimetre per second is three orders
#: below the smallest impulse any of these baselines plans (6.4e-03 m/s at 250 m separation).
DEFAULT_REST_TOL_M_S: float = 1.0e-3

#: Osculating eccentricity above which the chaser's initial orbit is no longer "near
#: circular" for the phasing baseline. Measured on the reference scenarios, a chaser placed
#: at a pure along-track Hill offset at rest has e = 3.2e-09 at 250 m, 3.2e-08 at 1 km and
#: 3.2e-06 at 10 km -- the limit clears the worst of those by three orders of magnitude while
#: still rejecting an orbit whose apsides differ by more than ~14 km at LEO.
DEFAULT_CIRCULAR_ECCENTRICITY_LIMIT: float = 1.0e-3

#: Samples used to report minimum separation along a baseline's relative trajectory.
DEFAULT_TRAJECTORY_SAMPLES: int = 201

_TWO_PI: float = 2.0 * math.pi

#: Impulse rule: maps a pre-burn inertial state to the impulse to apply, m/s, ECI.
#:
#: A callable rather than a stored vector because the phasing profile's burns are *tangential
#: to the chaser's velocity at the burn epoch*, which is not known until the coast before it
#: has been flown. Passing precomputed vectors would silently freeze the epoch-0 direction
#: into all four burns.
ImpulseRule = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


class BaselineError(RpoCoreError):
    """Base class for failures specific to the rendezvous baselines."""


class PhasingGeometryError(BaselineError, ValueError):
    """Raised when no drift radius inside the bracket delivers the required phase.

    Carries the requested along-track phase change and the range the bracket can actually
    deliver, so the caller can widen the bracket, change the revolution count, or conclude
    that the manoeuvre is not a phasing problem -- without a debugger.
    """

    def __init__(
        self,
        message: str,
        *,
        required_phase_rad: float,
        achievable_phase_rad: tuple[float, float],
    ) -> None:
        """Record the required phase and the achievable interval, both radians."""
        super().__init__(message)
        self.required_phase_rad = required_phase_rad
        self.achievable_phase_rad = achievable_phase_rad


class Method(StrEnum):
    """Which rendezvous strategy produced a :class:`BaselineResult`."""

    PHASING = "phasing"
    LAMBERT = "lambert"
    CW_TWO_IMPULSE = "cw-two-impulse"
    CW_CORRECTED = "cw-corrected"

    @property
    def label(self) -> str:
        """Human-readable name for the comparison table."""
        return _METHOD_LABELS[self]


_METHOD_LABELS: dict[Method, str] = {
    Method.PHASING: "Hohmann phasing",
    Method.LAMBERT: "Lambert direct",
    Method.CW_TWO_IMPULSE: "CW two-impulse",
    Method.CW_CORRECTED: "CW + nonlinear correction",
}


class Validity(Enum):
    """Whether a method's own modelling premise holds for the scenario it was run on.

    Deliberately *not* a statement about accuracy: a method can be VALID and still miss by
    metres (phasing), or INVALID and still produce a usable first guess (CW at 10 km). The
    terminal-error fields answer the accuracy question; this answers whether the numbers in
    the other columns mean what they appear to mean.
    """

    VALID = "valid"
    INVALID = "invalid"

    @property
    def label(self) -> str:
        """Upper-case name for the comparison table."""
        return self.name


def _vec3(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``value`` as a validated finite shape-(3,) float64 array."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array!r}")
    return array


def _positive_float(value: float, name: str) -> float:
    """Return ``value`` as a validated finite strictly-positive float."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}")
    return number


def hohmann_delta_v_m_s(
    r1_m: float, r2_m: float, mu_m3_s2: float = MU_EARTH_M3_S2
) -> tuple[float, float]:
    r"""Return the two **signed** impulses of a Hohmann transfer between circular orbits.

    .. math::

        \Delta v_1 = \sqrt{\mu/r_1}\left(\sqrt{\tfrac{2 r_2}{r_1 + r_2}} - 1\right),
        \qquad
        \Delta v_2 = \sqrt{\mu/r_2}\left(1 - \sqrt{\tfrac{2 r_1}{r_1 + r_2}}\right)

    Signed, not absolute: the sign is the direction along the velocity vector, and a caller
    flying the profile needs it. Take :func:`abs` for the Δv budget.

    Parameters
    ----------
    r1_m, r2_m
        Departure and arrival circular-orbit radii, metres. Both strictly positive.
    mu_m3_s2
        Gravitational parameter, m^3/s^2.

    Returns
    -------
    tuple of float
        ``(dv_departure, dv_arrival)``, m/s. Both negative for a transfer inwards
        (``r2 < r1``), both positive outwards, both exactly zero when ``r1 == r2``.

    Raises
    ------
    ValueError
        If either radius or ``mu_m3_s2`` is not finite and strictly positive.

    Examples
    --------
    >>> dv1, dv2 = hohmann_delta_v_m_s(7.0e6, 7.0e6)
    >>> (dv1, dv2)
    (0.0, 0.0)

    """
    r1 = _positive_float(r1_m, "r1_m")
    r2 = _positive_float(r2_m, "r2_m")
    mu = _positive_float(mu_m3_s2, "mu_m3_s2")
    total = r1 + r2
    dv1 = math.sqrt(mu / r1) * (math.sqrt(2.0 * r2 / total) - 1.0)
    dv2 = math.sqrt(mu / r2) * (1.0 - math.sqrt(2.0 * r1 / total))
    return dv1, dv2


def hohmann_transfer_time_s(r1_m: float, r2_m: float, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    r"""Return the Hohmann transfer time, half the period of the transfer ellipse.

    :math:`t_h = \pi \sqrt{a_h^3/\mu}` with :math:`a_h = (r_1 + r_2)/2`. Degenerates to
    exactly half a circular period when ``r1 == r2``, which is the limiting case the tests
    use to pin it.

    Raises
    ------
    ValueError
        If either radius or ``mu_m3_s2`` is not finite and strictly positive.

    """
    r1 = _positive_float(r1_m, "r1_m")
    r2 = _positive_float(r2_m, "r2_m")
    mu = _positive_float(mu_m3_s2, "mu_m3_s2")
    return math.pi * math.sqrt((0.5 * (r1 + r2)) ** 3 / mu)


@dataclass(frozen=True, eq=False)
class RendezvousProblem:
    """One rendezvous, stated once, so three methods can be compared on it.

    All three baselines consume exactly this and nothing else, which is what makes the
    comparison apples to apples. Method-specific knobs (drift revolutions, correction
    tolerance) are keyword arguments to the individual baselines, never fields here: a knob
    that lived on the problem would let one method quietly re-specify the scenario.

    Attributes
    ----------
    r_target0_eci_m, v_target0_eci_m_s
        Target inertial state at the epoch, shape (3,) each, metres and m/s.
    r0_hill_m, v0_hill_m_s
        Initial chaser relative state in the target's Hill frame, shape (3,) each.
    rf_hill_m, vf_hill_m_s
        Commanded terminal relative state, shape (3,) each.
    tof_s
        Commanded time of flight for the **fixed-time** methods (Lambert, CW), seconds.
        The phasing baseline does not consume it: its time of flight is an *output*, fixed
        by the phase it has to deliver and the revolution count it is allowed. That
        asymmetry is the trade the comparison exists to expose, not a flaw in the statement.
    mu_m3_s2
        Gravitational parameter, m^3/s^2.

    Raises
    ------
    ValueError
        On malformed or non-finite vectors, non-positive ``tof_s`` or ``mu_m3_s2``, or a
        target state with zero radius or speed.

    """

    r_target0_eci_m: npt.NDArray[np.float64]
    v_target0_eci_m_s: npt.NDArray[np.float64]
    r0_hill_m: npt.NDArray[np.float64]
    v0_hill_m_s: npt.NDArray[np.float64]
    rf_hill_m: npt.NDArray[np.float64]
    vf_hill_m_s: npt.NDArray[np.float64]
    tof_s: float
    mu_m3_s2: float = MU_EARTH_M3_S2

    def __post_init__(self) -> None:
        """Coerce every vector to a validated shape-(3,) float64 array and check scalars."""
        for name in (
            "r_target0_eci_m",
            "v_target0_eci_m_s",
            "r0_hill_m",
            "v0_hill_m_s",
            "rf_hill_m",
            "vf_hill_m_s",
        ):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))
        object.__setattr__(self, "tof_s", _positive_float(self.tof_s, "tof_s"))
        object.__setattr__(self, "mu_m3_s2", _positive_float(self.mu_m3_s2, "mu_m3_s2"))
        if float(np.linalg.norm(self.r_target0_eci_m)) <= 0.0:
            raise ValueError("r_target0_eci_m must be a non-zero position vector")
        if float(np.linalg.norm(self.v_target0_eci_m_s)) <= 0.0:
            raise ValueError("v_target0_eci_m_s must be a non-zero velocity vector")

    @property
    def target_state_eci(self) -> npt.NDArray[np.float64]:
        """Target inertial state ``[r(3), v(3)]``, shape (6,)."""
        return np.concatenate((self.r_target0_eci_m, self.v_target0_eci_m_s))

    @property
    def orbit_radius_m(self) -> float:
        """Target orbit radius at the epoch, metres."""
        return float(np.linalg.norm(self.r_target0_eci_m))

    @property
    def n_rad_s(self) -> float:
        """Target mean motion from its osculating semi-major axis, rad/s.

        Taken from specific energy rather than from ``|r|``: the two agree exactly for a
        circular target and differ for an eccentric one, and the mean motion that governs
        both CW and the phase budget is the one belonging to the *orbit*, not to the
        instantaneous radius.
        """
        return math.sqrt(self.mu_m3_s2 / self.target_semi_major_axis_m**3)

    @property
    def target_semi_major_axis_m(self) -> float:
        """Target osculating semi-major axis, metres, from ``a = -mu / (2 * energy)``."""
        energy = specific_energy_j_kg(self.target_state_eci, self.mu_m3_s2)
        if energy >= 0.0:
            raise ValueError(
                f"target orbit is not closed: specific energy {energy:.6g} J/kg >= 0, so "
                "it has no semi-major axis and no mean motion"
            )
        return -self.mu_m3_s2 / (2.0 * energy)

    @property
    def period_s(self) -> float:
        """Target orbital period, seconds."""
        return _TWO_PI / self.n_rad_s

    @property
    def separation_m(self) -> float:
        """Largest chaser-target separation the problem spans, metres.

        The max of the two endpoints, not the initial value: the CW validity envelope is
        quadratic in separation and must be judged on the worst point of the transfer, and
        for an outbound manoeuvre that is the terminal one.
        """
        return max(float(np.linalg.norm(self.r0_hill_m)), float(np.linalg.norm(self.rf_hill_m)))

    @property
    def chaser_state0_eci(self) -> npt.NDArray[np.float64]:
        """Chaser inertial state at the epoch implied by the initial relative state."""
        r_c, v_c = relative_state_hill_to_eci(
            self.r_target0_eci_m,
            self.v_target0_eci_m_s,
            np.concatenate((self.r0_hill_m, self.v0_hill_m_s)),
        )
        return np.concatenate((r_c, v_c))


@dataclass(frozen=True, eq=False)
class BaselineResult:
    """What every baseline returns, so the columns of the comparison mean one thing each.

    Attributes
    ----------
    method
        Which strategy produced this.
    total_delta_v_m_s
        Sum of ``burn_delta_v_m_s``. Held equal to the sum by construction, and tested.
    burn_delta_v_m_s
        Magnitude of each impulse in flight order, m/s. Length is the burn count: 4 for
        phasing, 2 for Lambert and CW.
    burn_times_s
        Epoch of each impulse, seconds from the problem epoch, same order.
    tof_s
        Time of flight actually flown. An *input* for Lambert and CW, an *output* for
        phasing -- see :attr:`RendezvousProblem.tof_s`.
    terminal_position_error_m, terminal_velocity_error_m_s
        Miss against the commanded terminal relative state, **measured under nonlinear
        two-body dynamics** by flying the returned impulses. Never evaluated under the
        model that designed them.
    min_separation_m
        Smallest chaser-target separation reached along the flown trajectory, metres. A Δv
        table that ignores whether the trajectory passes through the target is its own kind
        of misleading benchmark.
    validity
        Whether the method's modelling premise holds here. See :class:`Validity`.
    validity_detail
        The sentence that justifies ``validity``, carrying the numbers that decided it.
    cw_error_bound_m
        Conservative CW linearisation bound for this scenario, metres, from
        :func:`rpo_core.relative.conservative_cw_error_bound_m`. Reported for every method,
        not just CW, because it is the yardstick that says how far the linear model can be
        trusted here at all.
    detail
        Free-form method-specific notes (drift radius, correction iterations, ...).

    """

    method: Method
    total_delta_v_m_s: float
    burn_delta_v_m_s: tuple[float, ...]
    burn_times_s: tuple[float, ...]
    tof_s: float
    terminal_position_error_m: float
    terminal_velocity_error_m_s: float
    min_separation_m: float
    validity: Validity
    validity_detail: str
    cw_error_bound_m: float
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True when the method's modelling premise holds for this scenario."""
        return self.validity is Validity.VALID

    @property
    def burn_count(self) -> int:
        """Number of impulses in the profile."""
        return len(self.burn_delta_v_m_s)


def _rotation_at(
    target_state: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the ECI->Hill rotation for a target state, shape (3, 3)."""
    rotation, _ = hill_basis(target_state[:3], target_state[3:])
    return rotation


def _fly_impulse_schedule(
    problem: RendezvousProblem,
    schedule: Sequence[tuple[float, ImpulseRule]],
    tof_s: float,
    *,
    samples: int = DEFAULT_TRAJECTORY_SAMPLES,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> tuple[tuple[float, ...], float, float, float]:
    """Fly an impulse schedule under two-body dynamics and measure the terminal miss.

    This is the single scoring function all three baselines share. Giving each method its
    own would let each grade its own homework; sharing one is what makes the terminal-error
    column comparable across rows.

    Parameters
    ----------
    problem
        The scenario. Supplies the chaser's initial inertial state and the commanded
        terminal relative state.
    schedule
        ``(epoch_s, rule)`` pairs in non-decreasing epoch order, each epoch in
        ``[0, tof_s]``. ``rule`` maps the pre-burn inertial state to the impulse to apply.
    tof_s
        Total time of flight, seconds.
    samples
        Points at which the relative trajectory is sampled for ``min_separation_m``.
    rtol, atol
        Integrator tolerances.

    Returns
    -------
    tuple
        ``(burn_magnitudes_m_s, terminal_position_error_m, terminal_velocity_error_m_s,
        min_separation_m)``.

    Raises
    ------
    ValueError
        If the schedule is empty, out of order, or has an epoch outside ``[0, tof_s]``.
    PropagationError
        If any leg of the propagation fails; deliberately not translated.

    """
    if not schedule:
        raise ValueError("schedule must contain at least one impulse")
    epochs = [float(epoch) for epoch, _ in schedule]
    if any(not math.isfinite(e) or e < 0.0 or e > tof_s for e in epochs):
        raise ValueError(f"every impulse epoch must lie in [0, {tof_s:.6g}] s, got {epochs!r}")
    if any(b < a for a, b in itertools.pairwise(epochs)):
        raise ValueError(f"impulse epochs must be non-decreasing, got {epochs!r}")
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples!r}")

    mu = problem.mu_m3_s2
    # Output grid contains every burn epoch by construction, so each coast leg ends exactly
    # on the burn that closes it and no leg needs an extra propagation to reach it.
    times = np.unique(np.concatenate((np.linspace(0.0, tof_s, samples), np.array(epochs))))
    # One target propagation for the whole span; the Hill frame is recomputed from the
    # propagated target state at every sample, never frozen at the epoch.
    target = propagate_two_body(problem.target_state_eci, times, mu, rtol=rtol, atol=atol)

    state = problem.chaser_state0_eci
    chaser_samples = np.empty((times.size, 6), dtype=np.float64)
    chaser_samples[0] = state
    magnitudes: list[float] = []
    t_ref = 0.0
    for epoch, rule in schedule:
        if epoch > t_ref:
            mask = (times > t_ref) & (times <= epoch)
            leg = propagate_two_body(
                state,
                np.concatenate((np.zeros(1), times[mask] - t_ref)),
                mu,
                rtol=rtol,
                atol=atol,
            )
            chaser_samples[mask] = leg[1:]
            state = leg[-1]
            t_ref = epoch
        impulse = _vec3(rule(state), "impulse")
        state = np.concatenate((state[:3], state[3:] + impulse))
        magnitudes.append(float(np.linalg.norm(impulse)))
    if tof_s > t_ref:
        mask = times > t_ref
        leg = propagate_two_body(
            state,
            np.concatenate((np.zeros(1), times[mask] - t_ref)),
            mu,
            rtol=rtol,
            atol=atol,
        )
        chaser_samples[mask] = leg[1:]
        state = leg[-1]
    # Samples that coincide with a burn epoch carry the *pre*-burn velocity, which is
    # correct for a position-based separation metric and would be wrong for a range-rate
    # one. The terminal sample is the exception and is overwritten with the post-burn state,
    # because that is the state the manoeuvre actually delivers.
    chaser_samples[-1] = state

    relative = np.empty((times.size, 6), dtype=np.float64)
    for index in range(times.size):
        relative[index] = relative_state_eci_to_hill(
            target[index, :3],
            target[index, 3:],
            chaser_samples[index, :3],
            chaser_samples[index, 3:],
        )

    terminal = relative[-1]
    position_error = float(np.linalg.norm(terminal[:3] - problem.rf_hill_m))
    velocity_error = float(np.linalg.norm(terminal[3:] - problem.vf_hill_m_s))
    minimum_separation = float(np.min(separation_m(relative)))
    return tuple(magnitudes), position_error, velocity_error, minimum_separation


def _cw_validity(
    problem: RendezvousProblem, tof_s: float, tolerance_m: float
) -> tuple[Validity, str, float]:
    """Return the CW validity verdict, its justification, and the bound in metres."""
    n_orbits = tof_s / problem.period_s
    bound = conservative_cw_error_bound_m(problem.separation_m, problem.orbit_radius_m, n_orbits)
    if bound <= tolerance_m:
        detail = (
            f"conservative CW linearisation bound {bound:,.3g} m over {n_orbits:.3f} "
            f"orbits at {problem.separation_m:,.0f} m separation is within the "
            f"{tolerance_m:,.3g} m budget"
        )
        return Validity.VALID, detail, bound
    detail = (
        f"conservative CW linearisation bound {bound:,.4g} m over {n_orbits:.3f} orbits at "
        f"{problem.separation_m:,.0f} m separation EXCEEDS the {tolerance_m:,.3g} m budget "
        f"({bound / tolerance_m:,.1f}x) -- CW is outside its measured envelope here; use "
        "Lambert or a nonlinear correction"
    )
    return Validity.INVALID, detail, bound


def _hill_impulse_rule(
    target_state: npt.NDArray[np.float64], dv_hill_m_s: npt.NDArray[np.float64]
) -> ImpulseRule:
    """Return a rule applying a fixed Hill-frame impulse, rotated into ECI.

    An impulse changes velocity without moving the vehicle, so the transport-theorem
    ``omega x dr`` term contributes nothing and the map is the plain rotation transpose.
    Getting that wrong would add an offset proportional to separation -- ~0.1 m/s per km in
    LEO, the same order as the manoeuvres themselves.
    """
    dv_eci = _rotation_at(target_state).T @ dv_hill_m_s

    def rule(_state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return dv_eci

    return rule


def _fixed_impulse_rule(dv_eci_m_s: npt.NDArray[np.float64]) -> ImpulseRule:
    """Return a rule applying a fixed ECI impulse."""

    def rule(_state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return dv_eci_m_s

    return rule


def _tangential_impulse_rule(magnitude_m_s: float) -> ImpulseRule:
    """Return a rule applying a signed impulse along the chaser's current velocity."""

    def rule(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        speed = float(np.linalg.norm(state[3:]))
        if speed <= 0.0:  # pragma: no cover - unreachable for a closed orbit
            raise ValueError("cannot apply a tangential impulse to a state at rest")
        direction: npt.NDArray[np.float64] = state[3:] / speed
        return magnitude_m_s * direction

    return rule


def _phasing_validity(
    problem: RendezvousProblem,
    *,
    rest_tol_m_s: float,
    cross_track_tol_m: float,
    eccentricity_limit: float,
) -> tuple[Validity, str, float]:
    """Return the phasing baseline's premise verdict, justification, and chaser ``e``."""
    chaser_e = float(
        np.linalg.norm(eccentricity_vector(problem.chaser_state0_eci, problem.mu_m3_s2))
    )
    speed0 = float(np.linalg.norm(problem.v0_hill_m_s))
    speedf = float(np.linalg.norm(problem.vf_hill_m_s))
    cross_track = abs(float(problem.rf_hill_m[2] - problem.r0_hill_m[2]))

    breaches: list[str] = []
    if cross_track > cross_track_tol_m:
        breaches.append(
            f"requires a {cross_track:,.3g} m cross-track change, which a coplanar "
            f"Hohmann profile cannot deliver (tolerance {cross_track_tol_m:,.3g} m)"
        )
    if max(speed0, speedf) > rest_tol_m_s:
        breaches.append(
            f"endpoint relative speeds ({speed0:.3g}, {speedf:.3g}) m/s exceed the "
            f"{rest_tol_m_s:.3g} m/s at-rest tolerance the circular-endpoint model assumes"
        )
    if chaser_e > eccentricity_limit:
        breaches.append(
            f"chaser initial osculating eccentricity {chaser_e:.3g} exceeds the "
            f"{eccentricity_limit:.3g} near-circular limit the Hohmann closed form assumes"
        )
    if breaches:
        return Validity.INVALID, "; ".join(breaches), chaser_e
    detail = (
        f"coplanar, at-rest endpoints on a near-circular chaser orbit "
        f"(e = {chaser_e:.3g}); the Hohmann closed form applies"
    )
    return Validity.VALID, detail, chaser_e


def phasing_baseline(
    problem: RendezvousProblem,
    *,
    drift_revolutions: float = DEFAULT_DRIFT_REVOLUTIONS,
    max_drift_radius_fraction: float = DEFAULT_MAX_DRIFT_RADIUS_FRACTION,
    cw_tolerance_m: float = DEFAULT_CW_TOLERANCE_M,
    rest_tol_m_s: float = DEFAULT_REST_TOL_M_S,
    cross_track_tol_m: float = 1.0,
    eccentricity_limit: float = DEFAULT_CIRCULAR_ECCENTRICITY_LIMIT,
    samples: int = DEFAULT_TRAJECTORY_SAMPLES,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> BaselineResult:
    r"""Fly the classical phasing profile: drift orbit, wait for the phase, come back.

    Four burns. Hohmann out from the chaser's osculating semi-major axis to a drift radius
    :math:`r_d`, :math:`k` revolutions of coasting while the along-track phase builds at the
    differential rate, Hohmann back onto the target's radius. Δv comes from
    :func:`hohmann_delta_v_m_s`; the time of flight is dominated by the drift and is an
    *output*, so ``problem.tof_s`` is deliberately not consulted.

    :math:`r_d` is solved, not chosen. Over the profile the chaser sweeps exactly
    :math:`2\pi(k+1)` while the target sweeps :math:`n_t \cdot t_{\text{TOF}}(r_d)`, and the
    difference must equal the requested along-track phase change
    :math:`(y_f - y_0)/r_t`. That is one monotone scalar equation, closed by
    ``scipy.optimize.brentq``.

    Why the chaser's *semi-major axis* rather than its radius seeds the first Hohmann: a
    chaser placed at a pure along-track Hill offset at rest is not on the target's circular
    orbit. At 10 km it sits 29.4 m higher in semi-major axis, and 29.4 m of ``a`` is
    :math:`3\pi \cdot 29.4 = 277` m of along-track drift per orbit -- which is exactly the
    measured CW error law ``6*pi*rho**2/r`` at that separation, arriving from the other
    direction. Seeding from the radius instead leaves that drift unmodelled and the measured
    terminal miss grows from 22.5 m to 831.8 m.

    Parameters
    ----------
    problem
        The scenario.
    drift_revolutions
        Revolutions :math:`k` spent on the drift orbit. Need not be an integer -- the drift
        orbit is circular, so the return transfer can depart from any point of it -- which
        is what lets :func:`rpo_core.optimize.phasing_delta_v_vs_tof` sweep it continuously.
    max_drift_radius_fraction
        Half-width of the ``brentq`` bracket as a fraction of the target orbit radius.
    cw_tolerance_m
        Budget used to report the CW envelope alongside every method. Does not affect the
        phasing verdict, which rests on its own premise.
    rest_tol_m_s, cross_track_tol_m, eccentricity_limit
        Thresholds for the circular-endpoint premise; see :func:`_phasing_validity`.
    samples, rtol, atol
        Trajectory sampling and integrator tolerances for the nonlinear scoring flight.

    Returns
    -------
    BaselineResult
        Four burns, the derived time of flight, and the terminal miss measured under
        nonlinear two-body dynamics.

    Raises
    ------
    PhasingGeometryError
        If no drift radius inside the bracket delivers the required phase.
    ValueError
        If ``drift_revolutions`` is negative or non-finite, or the bracket fraction is not
        in ``(0, 1)``.
    PropagationError
        If the scoring flight fails.

    """
    revolutions = float(drift_revolutions)
    if not math.isfinite(revolutions) or revolutions < 0.0:
        raise ValueError(f"drift_revolutions must be finite and >= 0, got {drift_revolutions!r}")
    fraction = float(max_drift_radius_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            f"max_drift_radius_fraction must lie in (0, 1), got {max_drift_radius_fraction!r}"
        )

    mu = problem.mu_m3_s2
    r_target = problem.target_semi_major_axis_m
    n_target = problem.n_rad_s
    chaser_energy = specific_energy_j_kg(problem.chaser_state0_eci, mu)
    if chaser_energy >= 0.0:
        raise ValueError(
            f"chaser orbit is not closed: specific energy {chaser_energy:.6g} J/kg >= 0"
        )
    a_chaser = -mu / (2.0 * chaser_energy)

    required_phase_rad = float(problem.rf_hill_m[1] - problem.r0_hill_m[1]) / r_target

    def legs(drift_radius_m: float) -> tuple[float, float, float]:
        out = hohmann_transfer_time_s(a_chaser, drift_radius_m, mu)
        drift = revolutions * _TWO_PI * math.sqrt(drift_radius_m**3 / mu)
        back = hohmann_transfer_time_s(drift_radius_m, r_target, mu)
        return out, drift, back

    def phase_residual(drift_radius_m: float) -> float:
        return (
            _TWO_PI * (revolutions + 1.0)
            - n_target * sum(legs(drift_radius_m))
            - required_phase_rad
        )

    lower = r_target * (1.0 - fraction)
    upper = r_target * (1.0 + fraction)
    residual_lo = phase_residual(lower)
    residual_hi = phase_residual(upper)
    if residual_lo * residual_hi > 0.0:
        # residual = deliverable - required, so the deliverable interval is
        # [required + residual_hi, required + residual_lo] (residual decreases with radius).
        achievable = (required_phase_rad + residual_hi, required_phase_rad + residual_lo)
        raise PhasingGeometryError(
            f"no drift radius within {fraction:.1%} of the target radius delivers the "
            f"required along-track phase change of {required_phase_rad:.6g} rad "
            f"({required_phase_rad * r_target:,.1f} m) in {revolutions:g} drift "
            f"revolutions: the bracket spans [{achievable[0]:.6g}, {achievable[1]:.6g}] rad. "
            "Widen max_drift_radius_fraction, change drift_revolutions, or use a "
            "fixed-time method.",
            required_phase_rad=required_phase_rad,
            achievable_phase_rad=achievable,
        )

    drift_radius_m = float(brentq(phase_residual, lower, upper, xtol=1.0e-9, rtol=1.0e-15))
    out_s, drift_s, back_s = legs(drift_radius_m)
    tof_s = out_s + drift_s + back_s

    dv_out1, dv_out2 = hohmann_delta_v_m_s(a_chaser, drift_radius_m, mu)
    dv_back1, dv_back2 = hohmann_delta_v_m_s(drift_radius_m, r_target, mu)
    schedule: list[tuple[float, ImpulseRule]] = [
        (0.0, _tangential_impulse_rule(dv_out1)),
        (out_s, _tangential_impulse_rule(dv_out2)),
        (out_s + drift_s, _tangential_impulse_rule(dv_back1)),
        (tof_s, _tangential_impulse_rule(dv_back2)),
    ]
    burns, position_error, velocity_error, minimum = _fly_impulse_schedule(
        problem, schedule, tof_s, samples=samples, rtol=rtol, atol=atol
    )

    validity, detail, chaser_e = _phasing_validity(
        problem,
        rest_tol_m_s=rest_tol_m_s,
        cross_track_tol_m=cross_track_tol_m,
        eccentricity_limit=eccentricity_limit,
    )
    _, _, cw_bound = _cw_validity(problem, tof_s, cw_tolerance_m)
    return BaselineResult(
        method=Method.PHASING,
        total_delta_v_m_s=float(sum(burns)),
        burn_delta_v_m_s=burns,
        burn_times_s=(0.0, out_s, out_s + drift_s, tof_s),
        tof_s=tof_s,
        terminal_position_error_m=position_error,
        terminal_velocity_error_m_s=velocity_error,
        min_separation_m=minimum,
        validity=validity,
        validity_detail=detail,
        cw_error_bound_m=cw_bound,
        detail={
            "drift_radius_m": drift_radius_m,
            "drift_radius_offset_m": drift_radius_m - r_target,
            "drift_revolutions": revolutions,
            "required_phase_rad": required_phase_rad,
            "chaser_eccentricity": chaser_e,
            "chaser_semi_major_axis_offset_m": a_chaser - r_target,
        },
    )


def lambert_baseline(
    problem: RendezvousProblem,
    *,
    cw_tolerance_m: float = DEFAULT_CW_TOLERANCE_M,
    prograde: bool = True,
    revolutions: int = 0,
    samples: int = DEFAULT_TRAJECTORY_SAMPLES,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> BaselineResult:
    """Direct fixed-time transfer between the two absolute positions.

    Departure position: where the chaser is at the epoch. Arrival position: where the
    commanded terminal relative state will be, once the target has been propagated for
    ``problem.tof_s`` -- **propagated**, not the epoch position offset by ``rf_hill_m``,
    because the target moves 6 000 km in 0.4 of a LEO orbit and rotating the offset into the
    epoch frame instead would aim the transfer at empty space.

    Two burns: :func:`~rpo_core.lambert.solve_lambert` supplies the departure and arrival
    velocities on the connecting conic, and the impulses are the differences against the
    chaser's actual departure velocity and the commanded arrival velocity. No linearisation
    anywhere, so the terminal miss is integrator noise -- measured 4.5e-06 m on the
    reference scenario, consistent with the 7.09e-04 m worst case the Lambert module reports
    across its own 20-case sweep.

    Parameters
    ----------
    problem
        The scenario. ``problem.tof_s`` is the transfer time.
    cw_tolerance_m
        Budget for the CW envelope reported alongside every method. Does not gate Lambert,
        which makes no linearisation.
    prograde, revolutions
        Passed through to :func:`~rpo_core.lambert.solve_lambert`.
    samples, rtol, atol
        Trajectory sampling and integrator tolerances for the nonlinear scoring flight.

    Returns
    -------
    BaselineResult
        Two burns at ``0`` and ``tof_s``.

    Raises
    ------
    DegenerateGeometryError, InfeasibleTransferError, LambertConvergenceError
        Propagated from :func:`~rpo_core.lambert.solve_lambert`. Degenerate geometry is a
        real statement about the transfer, not something to absorb into a validity flag.
    PropagationError
        If the target propagation or the scoring flight fails.

    """
    tof_s = problem.tof_s
    mu = problem.mu_m3_s2
    chaser0 = problem.chaser_state0_eci
    target_f = propagate_two_body(
        problem.target_state_eci, np.array([0.0, tof_s]), mu, rtol=rtol, atol=atol
    )[-1]
    r_chaser_f, v_chaser_f = relative_state_hill_to_eci(
        target_f[:3], target_f[3:], np.concatenate((problem.rf_hill_m, problem.vf_hill_m_s))
    )

    v_depart, v_arrive = solve_lambert(
        chaser0[:3], r_chaser_f, tof_s, mu, prograde=prograde, revolutions=revolutions
    )
    dv1_eci = v_depart - chaser0[3:]
    dv2_eci = v_chaser_f - v_arrive

    schedule: list[tuple[float, ImpulseRule]] = [
        (0.0, _fixed_impulse_rule(dv1_eci)),
        (tof_s, _fixed_impulse_rule(dv2_eci)),
    ]
    burns, position_error, velocity_error, minimum = _fly_impulse_schedule(
        problem, schedule, tof_s, samples=samples, rtol=rtol, atol=atol
    )

    _, _, cw_bound = _cw_validity(problem, tof_s, cw_tolerance_m)
    detail = (
        "universal-variable Lambert is exact for two-body motion: no linearisation, so no "
        f"separation envelope. For scale, CW's bound on this scenario is {cw_bound:,.4g} m"
    )
    return BaselineResult(
        method=Method.LAMBERT,
        total_delta_v_m_s=float(sum(burns)),
        burn_delta_v_m_s=burns,
        burn_times_s=(0.0, tof_s),
        tof_s=tof_s,
        terminal_position_error_m=position_error,
        terminal_velocity_error_m_s=velocity_error,
        min_separation_m=minimum,
        validity=Validity.VALID,
        validity_detail=detail,
        cw_error_bound_m=cw_bound,
        detail={"revolutions": float(revolutions)},
    )


def cw_two_impulse_baseline(
    problem: RendezvousProblem,
    *,
    correct: bool = False,
    cw_tolerance_m: float = DEFAULT_CW_TOLERANCE_M,
    samples: int = DEFAULT_TRAJECTORY_SAMPLES,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    **correction_kwargs: float,
) -> BaselineResult:
    """Two-impulse Clohessy-Wiltshire transfer, optionally corrected onto real dynamics.

    With ``correct=False`` this is :func:`~rpo_core.relative.cw.two_impulse_transfer` flown
    open loop: the impulses are exact for the *linear* model and the flight measures how
    much that costs. Measured on the reference 10 km scenario, 73.4 m of terminal miss --
    against a commanded terminal state the linear model reproduces to 1e-9 m. That gap is
    the entire reason this module scores every method under nonlinear dynamics.

    With ``correct=True`` the departure impulse is re-aimed by
    :func:`~rpo_core.targeting.correct_two_impulse_transfer` until the trajectory arrives
    under nonlinear dynamics: 73.4 m becomes 2.5e-04 m for 0.14 % more Δv.

    The validity flag reports the **CW** envelope in both cases, because CW is what designed
    the burn. The corrected variant's ``validity_detail`` says so explicitly rather than
    quietly upgrading itself: the correction fixes the arrival, not the model.

    Parameters
    ----------
    problem
        The scenario. ``problem.tof_s`` is the transfer time.
    correct
        Run the nonlinear differential correction on the departure impulse.
    cw_tolerance_m
        Position-error budget the CW linearisation bound is judged against.
    samples, rtol, atol
        Trajectory sampling and integrator tolerances for the nonlinear scoring flight.
    **correction_kwargs
        Forwarded to :func:`~rpo_core.targeting.correct_two_impulse_transfer` when
        ``correct`` is true (``tolerance_m``, ``fd_step_m_s``, ...).

    Returns
    -------
    BaselineResult
        Two burns at ``0`` and ``tof_s``.

    Raises
    ------
    SingularTransferTimeError
        At a transfer time that is an integer number of orbital periods, when the transfer
        excites the rank-deficient in-plane direction.
    InfeasibleTransferError
        At a half-period transfer time when a cross-track change is requested.
    TargetingConvergenceError, IllConditionedJacobianError
        From the differential correction when ``correct`` is true.
    ValueError
        If ``correct`` is false and ``correction_kwargs`` were supplied, which would
        otherwise be silently ignored.

    """
    if not correct and correction_kwargs:
        raise ValueError(
            f"correction_kwargs {sorted(correction_kwargs)} were supplied with correct=False, "
            "where they would be silently ignored; pass correct=True or drop them"
        )

    tof_s = problem.tof_s
    n_rad_s = problem.n_rad_s
    if correct:
        corrected = correct_two_impulse_transfer(
            problem.r_target0_eci_m,
            problem.v_target0_eci_m_s,
            problem.r0_hill_m,
            problem.v0_hill_m_s,
            problem.rf_hill_m,
            problem.vf_hill_m_s,
            tof_s,
            problem.mu_m3_s2,
            n_rad_s=n_rad_s,
            rtol=rtol,
            atol=atol,
            **correction_kwargs,  # type: ignore[arg-type]
        )
        dv1_hill = corrected.dv1_hill_m_s
        dv2_hill = corrected.dv2_hill_m_s
        method = Method.CW_CORRECTED
        extra = {
            "iterations": float(corrected.iterations),
            "initial_residual_m": corrected.initial_residual_m,
            "final_residual_m": corrected.final_residual_m,
            "dv1_correction_m_s": corrected.dv1_correction_m_s,
        }
    else:
        dv1_hill, dv2_hill = two_impulse_transfer(
            n_rad_s,
            problem.r0_hill_m,
            problem.v0_hill_m_s,
            problem.rf_hill_m,
            problem.vf_hill_m_s,
            tof_s,
        )
        method = Method.CW_TWO_IMPULSE
        extra = {}

    target_f = propagate_two_body(
        problem.target_state_eci, np.array([0.0, tof_s]), problem.mu_m3_s2, rtol=rtol, atol=atol
    )[-1]
    schedule: list[tuple[float, ImpulseRule]] = [
        (0.0, _hill_impulse_rule(problem.target_state_eci, dv1_hill)),
        (tof_s, _hill_impulse_rule(target_f, dv2_hill)),
    ]
    burns, position_error, velocity_error, minimum = _fly_impulse_schedule(
        problem, schedule, tof_s, samples=samples, rtol=rtol, atol=atol
    )

    validity, detail, cw_bound = _cw_validity(problem, tof_s, cw_tolerance_m)
    if correct:
        detail = (
            f"{detail}. The differential correction re-aims the departure impulse onto "
            "nonlinear dynamics, so the arrival is not governed by this bound -- but the "
            "seed still is, and the flag reports the seed's model"
        )
    return BaselineResult(
        method=method,
        total_delta_v_m_s=float(sum(burns)),
        burn_delta_v_m_s=burns,
        burn_times_s=(0.0, tof_s),
        tof_s=tof_s,
        terminal_position_error_m=position_error,
        terminal_velocity_error_m_s=velocity_error,
        min_separation_m=minimum,
        validity=validity,
        validity_detail=detail,
        cw_error_bound_m=cw_bound,
        detail=extra,
    )
