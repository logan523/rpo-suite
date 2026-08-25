r"""Finite-burn (continuous-thrust) manoeuvre modelling -- model M8.

Every other manoeuvre in this suite is impulsive: a velocity discontinuity applied at a
point in time. A real thruster produces a bounded force for a finite time, during which the
vehicle keeps falling, keeps rotating relative to inertial space, and gets lighter. This
module integrates that, and -- more usefully -- measures how wrong the impulsive assumption
is for a given thrust level, so a mission designer can decide whether to care.

The equations
-------------
The state is seven-dimensional, :math:`[\mathbf{r}, \mathbf{v}, m]`, and mass is a
*propagated* state rather than a post-hoc correction:

.. math::

    \ddot{\mathbf{r}} &= -\frac{\mu}{\lVert\mathbf{r}\rVert^{3}} \mathbf{r}
                         \;+\; \frac{F}{m}\, \hat{\mathbf{u}} \\
    \dot{m}           &= -\frac{F}{g_0 I_{sp}}

with :math:`F` the thrust magnitude (N), :math:`I_{sp}` the specific impulse (s),
:math:`\hat{\mathbf{u}}` the thrust direction, and

.. math::

    g_0 = 9.80665\ \mathrm{m/s^2} \quad \text{(exact, by definition)} .

:data:`STANDARD_GRAVITY_M_S2` is a **defined constant, not a measurement**: the 3rd CGPM
(1901) fixed standard gravity at exactly 9.80665 m/s^2, and ISO 80000-3 carries the same
value. It is not the local gravitational acceleration anywhere in particular, and it has no
uncertainty. It appears here only because the aerospace industry defines specific impulse in
seconds -- :math:`I_{sp} = c / g_0` for exhaust velocity :math:`c` -- so :math:`g_0` is a
pure unit-conversion factor between seconds and metres per second. Nothing physical in this
module depends on Earth's actual gravity field beyond ``mu_m3_s2``.

Since :math:`\dot{m}` is constant while the thruster is on, mass is linear in time and the
integral of the thrust acceleration has the closed form that gives Tsiolkovsky's equation,

.. math::

    \Delta v_{\text{ideal}} = g_0 I_{sp} \ln\!\left(\frac{m_0}{m_f}\right) ,

which is *not* used to advance the state -- it is the independent oracle the propagated mass
and the free-space velocity change are checked against.

Thrust direction policies
-------------------------
:math:`\hat{\mathbf{u}}` is not a constant of the problem; it depends on what the attitude
control system is told to do. Two policies are provided and they are not interchangeable:

``ThrustDirection.INERTIAL_FIXED`` (**the default**)
    The direction is frozen in the inertial frame at ignition. This is what a vehicle that
    holds inertial attitude through the burn actually does, and it is the policy that makes
    the finite burn the direct analogue of an impulse: an impulsive :math:`\Delta v` *is* a
    fixed inertial vector. Comparing a Hill-fixed finite burn against an impulse would mix
    the finite-duration effect with an attitude-profile effect and report their sum as
    "finite-burn loss", which is why the default is the inertial one.

``ThrustDirection.HILL_FIXED``
    The direction is fixed in the vehicle's own instantaneous LVLH frame (``docs/conventions.md``:
    x radial-outward, y along-track, z positive orbit normal) and therefore rotates with it
    at roughly the orbital rate. This is what a burn commanded as "5 cm/s along V-bar" means
    if the vehicle holds LVLH attitude, and it is the natural policy for a low-thrust
    proximity manoeuvre whose duration is a meaningful fraction of an orbit.

For a burn lasting a quarter of an orbit the two differ by hundreds of metres in terminal
position (measured in ``test_finite_burn.py``); for a burn of a few seconds they agree to
well under a millimetre. The frame used by ``HILL_FIXED`` is the *burning vehicle's own*
osculating LVLH frame, not a separate target's -- this module propagates one vehicle, and
introducing a second would make the direction depend on a state this function never sees.

Integration, and why the burn is a separate segment
---------------------------------------------------
The right-hand side is discontinuous at ignition and at cut-off. Handing that discontinuity
to an adaptive integrator and asking for ``rtol = 1e-12`` invites it to hunt: the step
controller repeatedly rejects steps that straddle the jump, and the error estimate is
meaningless across it. :func:`propagate_with_finite_burn` therefore splits the interval at
the burn boundaries and integrates each segment separately, so every integration sees a
smooth right-hand side.

Coast segments are integrated by :func:`rpo_core.propagate.propagate_two_body` itself -- not
by a copy of it with a zero-thrust branch. Mass is exactly constant on a coast (that is not
an approximation, :math:`\dot m = 0` when :math:`F = 0`), so carrying a seventh state
through a coast would only add a component to the integrator's error norm and change its
step selection for no physical reason. The consequence is a limiting case with no wiggle
room: a propagation whose burn window does not intersect the requested interval returns
**bit-for-bit** what ``propagate_two_body`` returns, and the test suite asserts exactly
that.

Note that a thrust of exactly zero is rejected at construction. A thruster specified to
produce no thrust is a specification error, not a manoeuvre, and it also makes the
"duration from commanded delta-v" branch infinite. The realisable zero-thrust limit is the
one above: a burn that is never lit within the propagation window.

What "finite-burn loss" means here
----------------------------------
:func:`finite_burn_loss` reports two different numbers and they answer different questions.

* ``terminal_position_offset_m`` / ``terminal_velocity_offset_m_s`` -- where the vehicle
  actually is, versus where the impulsive plan said it would be. The velocity offset is
  also the size of the trim impulse that would null the velocity error.
* ``extra_delta_v_m_s`` -- ideal (Tsiolkovsky) :math:`\Delta v` minus the velocity change
  actually achieved relative to an unpowered coast from the same initial state. This is the
  classical gravity/steering loss: propellant spent that did not turn into useful velocity
  change, because the thrust direction was held fixed while the local geometry rotated
  underneath it.

  **This one is a short-burn diagnostic and it has a measured range of validity.** Comparing
  a powered arc against an unpowered one only isolates the propulsive loss while the two
  arcs are still close. Measured for the reference case below, it grows as :math:`t_b^{2}`
  (log-log slope ``-2.000`` in thrust) while the burn is under about 9 % of an orbit, peaks
  near 9 %, and **changes sign between 9 % and 12 %**: past that the difference between the
  powered and coasting orbits is dominated by orbital mechanics rather than by thruster
  efficiency, and the number stops meaning "loss". It is reported unclamped, because
  silently flooring it at zero would hide exactly the regime in which the caller should stop
  trusting it.

The position offset depends strongly on **where the reference impulse is placed**, and that
is a choice, not a fact. With the impulse at ignition (:data:`ImpulseEpoch.IGNITION`, the
default, and what most impulsive planning tools implicitly assume) the leading error is
purely kinematic: a constant acceleration delivers only half the position gain that an
equal impulse at the start of the same interval does, so

.. math::

    \lVert \Delta \mathbf{r} \rVert \;\approx\; \tfrac{1}{2}\, \Delta v \, t_b
    \;\propto\; F^{-1} ,

first order in burn duration. Placing the reference impulse at the delta-v-weighted centroid
of the burn (:data:`ImpulseEpoch.CENTROID`) matches the burn's first moment as well as its
zeroth, which cancels both the constant and the linear term and leaves the second-moment
mismatch at **third** order -- measured log-log slope -3.00 against thrust, versus -1.00 for
an ignition-placed impulse. The centroid is not the midpoint: with constant mass flow the
acceleration rises through the burn, and

.. math::

    \bar{t} = t_b \,\frac{\ln(m_0/m_f) - (1 - m_f/m_0)}{(1 - m_f/m_0)\ln(m_0/m_f)}
            \;\approx\; \frac{t_b}{2}\left(1 + \frac{1}{6}\frac{\Delta v}{g_0 I_{sp}}\right).

Both convergence orders are measured in the test suite; the difference between them is the
single most useful practical result in this module, because it says the bulk of the apparent
"finite-burn error" in a mission plan is bookkeeping that costs nothing to remove.

Validity
--------
Point-mass gravity only: no J2, no drag, no third bodies (compose with
:mod:`rpo_core.perturbations` if those matter -- this module deliberately does not, so that
the impulsive comparison isolates the finite-burn effect). Thrust magnitude is constant
through the burn: no throttling, no blow-down pressure decay, no start-up or shut-down
transient, and no minimum impulse bit. Specific impulse is constant, so there is no
efficiency variation with chamber pressure or duty cycle. Attitude is assumed to track the
commanded direction exactly and instantaneously -- there is no slew time and no pointing
error (see model M9 for the latter). Mass is a single lumped scalar: no centre-of-mass shift,
no slosh, no thrust-vector misalignment torque. The vehicle is a point, so there is no
attitude state at all.

Units are SI: metres, seconds, radians, kilograms, newtons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from ._validate import as_vector, validate_positive
from .constants import MU_EARTH_M3_S2
from .exceptions import PropagationError, RpoCoreError
from .frames import hill_basis
from .propagate import DEFAULT_ATOL, DEFAULT_RTOL, propagate_two_body, two_body_derivative

#: Standard gravity, m/s^2. **Exact by definition** (3rd CGPM, 1901; ISO 80000-3), not a
#: measured local value and not Earth's mean surface gravity. It enters this module purely
#: as the conversion between specific impulse expressed in seconds and exhaust velocity in
#: metres per second, ``c = g0 * Isp``. Writing 9.81 here would be a 0.03 % error in every
#: mass-flow rate and every Tsiolkovsky delta-v the module produces.
STANDARD_GRAVITY_M_S2: float = 9.80665


class ThrustDirection(Enum):
    """Which frame the commanded thrust direction is held fixed in.

    See the module docstring for why the choice is load-bearing and why
    :attr:`INERTIAL_FIXED` is the default.
    """

    #: Direction frozen in the inertial frame at ignition. The commanded vector is
    #: interpreted in ECI components.
    INERTIAL_FIXED = "inertial_fixed"

    #: Direction fixed in the vehicle's own instantaneous LVLH frame, so it rotates with
    #: that frame. The commanded vector is interpreted in Hill components
    #: ``[radial, along-track, cross-track]``.
    HILL_FIXED = "hill_fixed"


class ImpulseEpoch(Enum):
    """Where the reference impulse is placed when comparing against a finite burn."""

    #: At ignition -- the implicit assumption of an impulsive planning tool that schedules a
    #: manoeuvre "at" a time. Leading position error is first order in burn duration.
    IGNITION = "ignition"

    #: At the delta-v-weighted centroid of the burn, which cancels the first-order term.
    #: Slightly later than the midpoint because the acceleration rises as mass falls.
    CENTROID = "centroid"


class PropellantExhaustedError(RpoCoreError, ValueError):
    """Raised when a burn would consume the entire vehicle mass.

    Integrating past this point produces a negative mass, and ``F/m`` with negative ``m``
    is a thruster that pushes backwards and accelerates without limit. The integrator will
    not complain -- it will return a beautifully converged trajectory for a vehicle that
    stopped existing partway through the burn -- so the condition is checked rather than
    discovered.

    Note this is the *total* mass, not the propellant mass: this module carries a single
    lumped mass and has no dry-mass field, so exhaustion here means "lighter than nothing",
    which is strictly weaker than the real constraint "lighter than the empty vehicle". A
    caller who knows their dry mass should check against it themselves.
    """


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------


def _validate_non_negative(value: float, name: str) -> float:
    """Return ``value`` as a validated finite, non-negative float."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return number


# --------------------------------------------------------------------------------------
# Tsiolkovsky
# --------------------------------------------------------------------------------------


def equivalent_impulsive_delta_v(
    initial_mass_kg: float, final_mass_kg: float, specific_impulse_s: float
) -> float:
    r"""Return the ideal delta-v of a burn from its mass ratio, m/s.

    Tsiolkovsky's rocket equation,
    :math:`\Delta v = g_0 I_{sp} \ln(m_0 / m_f)`. This is the velocity change the burn would
    produce in field-free space with the thrust direction held fixed -- i.e. the propellant
    actually spent, expressed as velocity. In a gravity field the *achieved* velocity change
    is smaller; the difference is what :func:`finite_burn_loss` reports as
    ``extra_delta_v_m_s``.

    Parameters
    ----------
    initial_mass_kg
        Wet mass at ignition, kg. Must be finite and strictly positive.
    final_mass_kg
        Mass at cut-off, kg. Must be finite, strictly positive, and no greater than
        ``initial_mass_kg``.
    specific_impulse_s
        Specific impulse, seconds. Must be finite and strictly positive.

    Returns
    -------
    float
        Ideal delta-v, m/s. Zero when the masses are equal.

    Raises
    ------
    ValueError
        On non-finite or non-positive input, or if ``final_mass_kg > initial_mass_kg``,
        which describes a vehicle that gained mass during a burn.

    Examples
    --------
    >>> round(equivalent_impulsive_delta_v(200.0, 199.0, 220.0), 6)
    10.813162

    """
    m0 = validate_positive(initial_mass_kg, "initial_mass_kg")
    mf = validate_positive(final_mass_kg, "final_mass_kg")
    isp = validate_positive(specific_impulse_s, "specific_impulse_s")
    if mf > m0:
        raise ValueError(
            f"final_mass_kg = {mf!r} kg exceeds initial_mass_kg = {m0!r} kg; a burn cannot "
            "increase the vehicle mass, and Tsiolkovsky would return a negative delta-v"
        )
    return isp * STANDARD_GRAVITY_M_S2 * math.log(m0 / mf)


def _delta_v_centroid_fraction(mass_ratio_burnt: float) -> float:
    """Return the delta-v-weighted centroid of a burn as a fraction of its duration.

    ``mass_ratio_burnt`` is ``x = 1 - m_f/m_0``, the fraction of the initial mass consumed.
    The exact result is ``(-x - ln(1-x)) / (x * (-ln(1-x)))``, which for the tiny mass
    ratios typical of an RPO burn (``x ~ 1e-4``) suffers catastrophic cancellation in the
    numerator: ``-x - ln(1-x)`` is ``x**2/2`` to leading order, so eight of sixteen digits
    are lost. Below ``x = 1e-3`` the numerator is therefore evaluated from its series
    instead, which is exact to machine precision there.

    The value tends to ``1/2 + x/12`` as ``x -> 0``: the centroid sits slightly *after* the
    midpoint because ``F/m`` grows as propellant leaves.
    """
    x = mass_ratio_burnt
    log_ratio = -math.log1p(-x)  # = ln(m0/mf) = delta_v / (g0 * Isp)
    if x < 1.0e-3:
        numerator = x * x * (0.5 + x / 3.0 + x * x / 4.0 + x**3 / 5.0)
    else:
        numerator = log_ratio - x
    return numerator / (x * log_ratio)


# --------------------------------------------------------------------------------------
# The burn specification
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FiniteBurn:
    r"""A constant-thrust, constant-:math:`I_{sp}` manoeuvre.

    Immutable, and fully validated at construction: every derived quantity below is a pure
    function of these fields, so a ``FiniteBurn`` that exists is one that can be flown.

    The burn length is specified **either** by ``duration_s`` **or** by
    ``commanded_delta_v_m_s``, never both and never neither. Both are legitimate ways to
    describe the same burn and they are related by Tsiolkovsky, but accepting both at once
    would let a caller supply a pair that disagree, and there is no defensible rule for
    which one then wins.

    Parameters
    ----------
    thrust_n
        Thrust magnitude, newtons. Finite and strictly positive; see the module docstring
        for why exactly zero is rejected rather than treated as a coast.
    specific_impulse_s
        Specific impulse, seconds. Finite and strictly positive.
    initial_mass_kg
        Vehicle mass at ignition, kg. Finite and strictly positive.
    direction_unit
        Commanded thrust direction, shape (3,). Normalised at construction, so magnitude is
        ignored -- the magnitude of the manoeuvre lives in ``thrust_n`` and the burn length,
        not here. Interpreted in **ECI** components for
        :attr:`ThrustDirection.INERTIAL_FIXED` and in **Hill** components
        ``[radial, along-track, cross-track]`` for :attr:`ThrustDirection.HILL_FIXED`.
    start_time_s
        Ignition time, seconds after the epoch of the state being propagated. Finite and
        non-negative.
    duration_s
        Burn duration, seconds. Finite and strictly positive if given.
    commanded_delta_v_m_s
        Ideal (Tsiolkovsky) delta-v the burn is sized to deliver, m/s. Finite and strictly
        positive if given.
    direction_policy
        Which frame ``direction_unit`` is held fixed in. Defaults to
        :attr:`ThrustDirection.INERTIAL_FIXED`.

    Raises
    ------
    ValueError
        Non-finite, non-positive, or negative input; a malformed or zero direction vector;
        or a specification that gives both or neither of ``duration_s`` and
        ``commanded_delta_v_m_s``.
    PropellantExhaustedError
        If ``duration_s`` is long enough to consume the entire vehicle mass. Only reachable
        on the duration branch: a burn sized by commanded delta-v has
        ``m_f = m_0 e^{-\Delta v / c} > 0`` for every finite delta-v.

    Examples
    --------
    A 22 N monopropellant thruster on a 200 kg servicer, sized for 0.2 m/s along V-bar:

    >>> burn = FiniteBurn(
    ...     thrust_n=22.0,
    ...     specific_impulse_s=220.0,
    ...     initial_mass_kg=200.0,
    ...     direction_unit=[0.0, 1.0, 0.0],
    ...     commanded_delta_v_m_s=0.2,
    ... )
    >>> round(burn.burn_duration_s, 4)
    1.8188

    """

    thrust_n: float
    specific_impulse_s: float
    initial_mass_kg: float
    direction_unit: npt.NDArray[np.float64]
    start_time_s: float = 0.0
    duration_s: float | None = None
    commanded_delta_v_m_s: float | None = None
    direction_policy: ThrustDirection = ThrustDirection.INERTIAL_FIXED

    def __post_init__(self) -> None:
        """Coerce and validate every field; see the class docstring for the raise paths."""
        set_field = object.__setattr__
        set_field(self, "thrust_n", validate_positive(self.thrust_n, "thrust_n"))
        set_field(
            self,
            "specific_impulse_s",
            validate_positive(self.specific_impulse_s, "specific_impulse_s"),
        )
        set_field(
            self, "initial_mass_kg", validate_positive(self.initial_mass_kg, "initial_mass_kg")
        )
        set_field(self, "start_time_s", _validate_non_negative(self.start_time_s, "start_time_s"))

        direction = as_vector(self.direction_unit, "direction_unit")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError(
                "direction_unit has zero magnitude; a thrust vector needs a direction, and "
                "there is no sensible default for where a thruster points"
            )
        set_field(self, "direction_unit", direction / norm)

        if not isinstance(self.direction_policy, ThrustDirection):
            raise ValueError(
                f"direction_policy must be a ThrustDirection, got {self.direction_policy!r}"
            )

        if (self.duration_s is None) == (self.commanded_delta_v_m_s is None):
            raise ValueError(
                "specify exactly one of duration_s and commanded_delta_v_m_s, got "
                f"duration_s={self.duration_s!r}, "
                f"commanded_delta_v_m_s={self.commanded_delta_v_m_s!r}; the two are related "
                "by Tsiolkovsky, so supplying both admits a pair that disagree"
            )
        if self.duration_s is not None:
            set_field(self, "duration_s", validate_positive(self.duration_s, "duration_s"))
        if self.commanded_delta_v_m_s is not None:
            set_field(
                self,
                "commanded_delta_v_m_s",
                validate_positive(self.commanded_delta_v_m_s, "commanded_delta_v_m_s"),
            )

        propellant = self.mass_flow_rate_kg_s * self.burn_duration_s
        if propellant >= self.initial_mass_kg:
            raise PropellantExhaustedError(
                f"a {self.burn_duration_s:.6g} s burn at {self.thrust_n:.6g} N and "
                f"Isp = {self.specific_impulse_s:.6g} s consumes {propellant:.6g} kg, but the "
                f"vehicle mass is only {self.initial_mass_kg:.6g} kg (mass flow "
                f"{self.mass_flow_rate_kg_s:.6g} kg/s); the burn runs out of vehicle after "
                f"{self.initial_mass_kg / self.mass_flow_rate_kg_s:.6g} s"
            )

    @property
    def exhaust_velocity_m_s(self) -> float:
        """Effective exhaust velocity ``c = g0 * Isp``, m/s."""
        return STANDARD_GRAVITY_M_S2 * self.specific_impulse_s

    @property
    def mass_flow_rate_kg_s(self) -> float:
        """Propellant mass flow ``F / c``, kg/s. Constant, because ``F`` and ``Isp`` are."""
        return self.thrust_n / self.exhaust_velocity_m_s

    @property
    def burn_duration_s(self) -> float:
        r"""Burn duration, seconds.

        Given directly when the burn was specified by ``duration_s``. When it was specified
        by a commanded delta-v, inverted from Tsiolkovsky with constant mass flow:
        :math:`t_b = (m_0 / \dot m)\left(1 - e^{-\Delta v / c}\right)`, written with
        ``expm1`` so that the tiny mass ratios of an RPO burn keep full precision.
        """
        if self.duration_s is not None:
            return self.duration_s
        if self.commanded_delta_v_m_s is None:  # pragma: no cover - __post_init__ forbids it
            raise ValueError("FiniteBurn has neither duration_s nor commanded_delta_v_m_s")
        burnt_fraction = -math.expm1(-self.commanded_delta_v_m_s / self.exhaust_velocity_m_s)
        return self.initial_mass_kg * burnt_fraction / self.mass_flow_rate_kg_s

    @property
    def end_time_s(self) -> float:
        """Cut-off time, seconds after the propagation epoch."""
        return self.start_time_s + self.burn_duration_s

    @property
    def propellant_mass_kg(self) -> float:
        """Propellant consumed over the whole burn, kg."""
        return self.mass_flow_rate_kg_s * self.burn_duration_s

    @property
    def final_mass_kg(self) -> float:
        """Vehicle mass at cut-off, kg. Strictly positive by construction."""
        return self.initial_mass_kg - self.propellant_mass_kg

    @property
    def ideal_delta_v_m_s(self) -> float:
        """Tsiolkovsky delta-v for the propellant this burn spends, m/s.

        Equal to ``commanded_delta_v_m_s`` to round-off when the burn was specified that
        way; computed from the mass ratio when it was specified by duration. This is the
        quantity :func:`finite_burn_loss` compares the achieved velocity change against.
        """
        return equivalent_impulsive_delta_v(
            self.initial_mass_kg, self.final_mass_kg, self.specific_impulse_s
        )

    @property
    def delta_v_centroid_time_s(self) -> float:
        """Delta-v-weighted centroid of the burn, seconds after the propagation epoch.

        The epoch at which a single impulse best represents this burn; see
        :func:`_delta_v_centroid_fraction` and the module docstring.
        """
        burnt_fraction = self.propellant_mass_kg / self.initial_mass_kg
        return self.start_time_s + self.burn_duration_s * _delta_v_centroid_fraction(burnt_fraction)


# --------------------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------------------


def thrust_unit_eci(
    state: npt.NDArray[np.float64],
    direction_unit: npt.NDArray[np.float64],
    direction_policy: ThrustDirection,
) -> npt.NDArray[np.float64]:
    """Return the thrust direction in ECI components for the current state.

    Parameters
    ----------
    state
        Vehicle state ``[r(3), v(3), ...]``; only the first six entries are read.
    direction_unit
        Commanded unit direction, in ECI for :attr:`ThrustDirection.INERTIAL_FIXED` and in
        Hill components for :attr:`ThrustDirection.HILL_FIXED`.
    direction_policy
        Which of those two it is.

    Returns
    -------
    numpy.ndarray
        Unit vector in ECI, shape (3,).

    Raises
    ------
    DegenerateGeometryError
        For :attr:`ThrustDirection.HILL_FIXED` only, if the state has no defined LVLH frame
        (zero position or velocity, or a purely radial trajectory).

    Notes
    -----
    The Hill branch calls :func:`rpo_core.frames.hill_basis` rather than rebuilding the
    triad inline. That costs a little input validation per integrator stage, and buys the
    guarantee that this module cannot drift away from the locked frame convention -- a
    sign error in a duplicated ``y_hat = z_hat x x_hat`` would point a V-bar burn backwards
    and still integrate perfectly happily.

    """
    if direction_policy is ThrustDirection.INERTIAL_FIXED:
        return direction_unit
    rotation_eci_to_hill, _ = hill_basis(state[:3], state[3:6])
    return np.asarray(rotation_eci_to_hill.T @ direction_unit, dtype=np.float64)


def finite_burn_derivative(
    _t: float,
    state: npt.NDArray[np.float64],
    mu_m3_s2: float,
    thrust_n: float,
    mass_flow_rate_kg_s: float,
    direction_unit: npt.NDArray[np.float64],
    direction_policy: ThrustDirection,
) -> npt.NDArray[np.float64]:
    """Return ``d/dt [r, v, m]`` for two-body motion under constant thrust.

    Signature matches what ``solve_ivp`` expects. ``_t`` is unused: with constant thrust and
    constant mass flow the dynamics are autonomous, and the *only* thing that makes a finite
    burn time-dependent -- the falling mass -- is carried in the state where the integrator
    can see it, not evaluated from the clock. That is the difference between mass as a
    propagated state and mass as a post-hoc correction, and it is why the Tsiolkovsky check
    in the test suite is a real check.

    Raises
    ------
    PropellantExhaustedError
        If the propagated mass has reached zero or gone negative. A backstop:
        :class:`FiniteBurn` proves at construction that this cannot happen over the
        commanded duration, so reaching it means the caller drove this function directly or
        the burn window was extended behind the spec's back.
    PropagationError
        Through :func:`rpo_core.propagate.two_body_derivative`, if the trajectory reaches
        ``|r| = 0``.

    """
    mass_kg = float(state[6])
    if mass_kg <= 0.0:
        raise PropellantExhaustedError(
            f"vehicle mass reached {mass_kg:.6g} kg during the burn; F/m for non-positive m "
            "is a thruster that pushes backwards with unbounded acceleration, so the "
            "integration is stopped rather than continued into nonsense"
        )
    derivative6 = two_body_derivative(_t, state[:6], mu_m3_s2)
    unit_eci = thrust_unit_eci(state, direction_unit, direction_policy)
    derivative6[3:] += (thrust_n / mass_kg) * unit_eci
    return np.concatenate((derivative6, (-mass_flow_rate_kg_s,)))


# --------------------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------------------


def _validate_times(times_s: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce and validate an output-time schedule.

    Stricter than :func:`rpo_core.propagate.propagate_two_body` in exactly one respect, and
    deliberately: that function documents "non-decreasing" but a repeated output time is
    rejected downstream by ``solve_ivp`` with ``Values in t_eval are not properly sorted``,
    which names neither the offending value nor the argument the caller passed. Requiring
    strictly increasing times here changes nothing about which schedules succeed -- it only
    moves the rejection to where the numbers are still in scope.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"times_s must be a non-empty 1-D array, got shape {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_s must be finite")
    if times[0] != 0.0:
        raise ValueError(f"times_s must start at 0.0, got {times[0]!r}")
    steps = np.diff(times)
    if times.size > 1 and np.any(steps <= 0.0):
        first_bad = int(np.argmin(steps > 0.0))
        raise ValueError(
            "times_s must be strictly increasing after the first entry; "
            f"times_s[{first_bad}] = {times[first_bad]!r} is followed by "
            f"times_s[{first_bad + 1}] = {times[first_bad + 1]!r}"
        )
    return times


def _validate_state6(state0_eci: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce and validate an initial inertial state of shape (6,)."""
    state0 = np.asarray(state0_eci, dtype=np.float64)
    if state0.shape != (6,):
        raise ValueError(f"state0_eci must have shape (6,), got {state0.shape}")
    if not np.all(np.isfinite(state0)):
        raise ValueError(f"state0_eci must be finite, got {state0!r}")
    return state0


def _segment_schedule(
    eval_times_rel: npt.NDArray[np.float64], span_s: float
) -> tuple[npt.NDArray[np.float64], int]:
    """Build a segment-local time schedule that starts at 0 and ends at ``span_s``.

    Returns the schedule and the index at which the caller's requested times begin inside
    it. When the requested times already start at 0 and end at ``span_s`` the schedule is
    returned unchanged, which is what makes the pure-coast case bit-for-bit identical to a
    direct ``propagate_two_body`` call: same array, same integrator arguments, same bits.
    """
    parts: list[npt.NDArray[np.float64]] = []
    offset = 0
    if eval_times_rel.size == 0 or eval_times_rel[0] != 0.0:
        parts.append(np.zeros(1, dtype=np.float64))
        offset = 1
    parts.append(eval_times_rel)
    if eval_times_rel.size == 0 or eval_times_rel[-1] != span_s:
        parts.append(np.array([span_s], dtype=np.float64))
    return (parts[0] if len(parts) == 1 else np.concatenate(parts)), offset


def propagate_with_finite_burn(
    state0_eci: npt.ArrayLike,
    times_s: npt.ArrayLike,
    burn: FiniteBurn,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> npt.NDArray[np.float64]:
    """Propagate ``[r, v, m]`` through a coast, a finite burn, and a coast.

    The interval is split at the burn boundaries and each piece is integrated separately, so
    no integration ever straddles the thrust discontinuity. Coast pieces are handed to
    :func:`rpo_core.propagate.propagate_two_body` unchanged; only the burn piece carries the
    seventh (mass) state. See the module docstring for why.

    Parameters
    ----------
    state0_eci
        Initial inertial state ``[r(3), v(3)]``, metres and m/s, shape (6,). Mass comes from
        ``burn.initial_mass_kg``: it is a property of the vehicle, not of its trajectory.
    times_s
        Output times, seconds from the epoch of ``state0_eci``. Must start at 0.0 and be
        strictly increasing thereafter; see :func:`_validate_times` for why repeated times
        are rejected here rather than deep inside ``solve_ivp``.
    burn
        The manoeuvre. A burn whose window ``[start_time_s, end_time_s]`` lies entirely
        outside ``[0, times_s[-1]]`` is the zero-thrust limiting case and returns exactly
        what ``propagate_two_body`` returns, bit for bit, with a constant mass column.
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Finite and non-negative; **zero is accepted and
        means field-free space**, which is the configuration in which the propagated mass
        and velocity change can be checked against Tsiolkovsky with no gravity term to
        confound them.
    rtol, atol
        Integrator tolerances, applied to every segment. Exposed for the same reason as in
        :func:`rpo_core.propagate.propagate_two_body`: a quoted number that has not survived
        a tolerance sweep is an integrator setting, not a result.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times_s), 7)`` -- ``[r(3), v(3), m]`` at each requested time.

    Raises
    ------
    PropagationError
        If any segment's integrator fails or returns fewer states than requested. Never a
        truncated trajectory.
    PropellantExhaustedError
        If the propagated mass reaches zero. Unreachable for a well-formed
        :class:`FiniteBurn`, which proves the opposite at construction.
    ValueError
        Malformed or non-finite state, malformed time schedule, or negative ``mu_m3_s2``.

    Examples
    --------
    >>> import numpy as np
    >>> from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
    >>> a = R_EARTH_EQUATORIAL_M + 420.0e3
    >>> speed = np.sqrt(MU_EARTH_M3_S2 / a)
    >>> state = [a, 0.0, 0.0, 0.0, speed, 0.0]
    >>> burn = FiniteBurn(22.0, 220.0, 200.0, [0.0, 1.0, 0.0], commanded_delta_v_m_s=0.2)
    >>> out = propagate_with_finite_burn(state, [0.0, 60.0], burn)
    >>> out.shape
    (2, 7)
    >>> bool(out[-1, 6] < 200.0)
    True

    """
    state0 = _validate_state6(state0_eci)
    times = _validate_times(times_s)
    if not isinstance(burn, FiniteBurn):
        raise ValueError(f"burn must be a FiniteBurn, got {type(burn).__name__}")
    mu = float(mu_m3_s2)
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError(
            f"mu_m3_s2 must be finite and >= 0, got {mu_m3_s2!r}; zero is permitted and "
            "means field-free space"
        )

    final_time_s = float(times[-1])
    if times.size == 1:
        # Strictly-increasing times means this is the only way to be asked for zero elapsed
        # time. Mirrors the same early return in ``propagate_two_body``.
        return np.concatenate((state0, (burn.initial_mass_kg,))).reshape(1, 7)

    # Segment edges: the propagation bounds plus whichever burn boundaries fall strictly
    # inside them. Sorting a set collapses the cases where ignition coincides with an
    # endpoint, so no segment is ever empty.
    edges = {0.0, final_time_s}
    for boundary in (burn.start_time_s, burn.end_time_s):
        if 0.0 < boundary < final_time_s:
            edges.add(boundary)
    edge_list = sorted(edges)
    edge_array = np.asarray(edge_list, dtype=np.float64)

    # side="right" puts a requested time that lands exactly on an interior edge into the
    # later segment. The state is continuous there, so the choice is cosmetic -- but it must
    # be made consistently or a time could be evaluated twice or not at all.
    segment_of_time = np.clip(
        np.searchsorted(edge_array, times, side="right") - 1, 0, len(edge_list) - 2
    )

    output = np.empty((times.size, 7), dtype=np.float64)
    state6 = state0
    mass_kg = burn.initial_mass_kg

    for index in range(len(edge_list) - 1):
        segment_start_s = edge_list[index]
        segment_end_s = edge_list[index + 1]
        midpoint_s = 0.5 * (segment_start_s + segment_end_s)
        thrusting = burn.start_time_s <= midpoint_s <= burn.end_time_s

        rows = np.flatnonzero(segment_of_time == index)
        eval_rel = times[rows] - segment_start_s
        schedule, offset = _segment_schedule(eval_rel, segment_end_s - segment_start_s)

        if thrusting:
            states7 = _integrate_burn_segment(
                np.concatenate((state6, (mass_kg,))), schedule, burn, mu, rtol, atol
            )
            output[rows] = states7[offset : offset + rows.size]
            state6 = states7[-1, :6]
            mass_kg = float(states7[-1, 6])
        else:
            states6 = propagate_two_body(state6, schedule, mu, rtol=rtol, atol=atol)
            output[rows, :6] = states6[offset : offset + rows.size]
            output[rows, 6] = mass_kg
            state6 = states6[-1]

    return output


def _integrate_burn_segment(
    state7: npt.NDArray[np.float64],
    schedule_s: npt.NDArray[np.float64],
    burn: FiniteBurn,
    mu_m3_s2: float,
    rtol: float,
    atol: float,
) -> npt.NDArray[np.float64]:
    """Integrate one powered segment, returning states of shape ``(len(schedule_s), 7)``."""
    if schedule_s.size == 1:
        return state7.reshape(1, 7).copy()
    solution = solve_ivp(
        finite_burn_derivative,
        (0.0, float(schedule_s[-1])),
        state7,
        method="DOP853",
        t_eval=schedule_s,
        args=(
            mu_m3_s2,
            burn.thrust_n,
            burn.mass_flow_rate_kg_s,
            burn.direction_unit,
            burn.direction_policy,
        ),
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise PropagationError(
            "finite-burn propagation failed at t = "
            f"{solution.t[-1] if solution.t.size else 0.0:.6g} s of "
            f"{float(schedule_s[-1]):.6g} s of powered flight "
            f"(thrust {burn.thrust_n:.6g} N, Isp {burn.specific_impulse_s:.6g} s, "
            f"policy {burn.direction_policy.value}): {solution.message}"
        )
    if solution.y.shape[1] != schedule_s.size:
        raise PropagationError(
            f"integrator returned {solution.y.shape[1]} states for {schedule_s.size} "
            "requested times during the burn; the trajectory is incomplete"
        )
    return np.ascontiguousarray(solution.y.T)


# --------------------------------------------------------------------------------------
# The headline: finite-burn loss
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteBurnLoss:
    """What a finite burn costs relative to the impulsive manoeuvre it approximates.

    Every field is a measured consequence of two propagations, not a fitted or assumed
    quantity. See :func:`finite_burn_loss` and the module docstring for the definitions;
    the two that matter are ``terminal_position_offset_m`` (where you end up versus where
    the impulsive plan said) and ``extra_delta_v_m_s`` (propellant spent that did not become
    useful velocity change).
    """

    #: Distance between the finite-burn and impulsive positions at ``comparison_time_s``, m.
    terminal_position_offset_m: float
    #: Speed difference at ``comparison_time_s``, m/s. Also the size of the trim impulse
    #: that would put the finite-burn vehicle on the impulsive velocity.
    terminal_velocity_offset_m_s: float
    #: Tsiolkovsky delta-v for the propellant spent, m/s -- what the burn "cost".
    ideal_delta_v_m_s: float
    #: Velocity change actually achieved relative to an unpowered coast from the same
    #: initial state, m/s -- what the burn "bought".
    effective_delta_v_m_s: float
    #: ``ideal_delta_v_m_s - effective_delta_v_m_s``, m/s. The gravity/steering loss.
    #: Meaningful only while the burn is short compared with an orbit -- it changes sign
    #: somewhere between 9 % and 12 % of an orbital period, and is reported unclamped so
    #: that the breakdown is visible rather than hidden. See the module docstring.
    extra_delta_v_m_s: float
    #: Epoch at which the two trajectories were compared, s after the propagation epoch.
    comparison_time_s: float
    #: Epoch at which the reference impulse was applied, s after the propagation epoch.
    impulse_time_s: float
    #: Burn duration, s.
    burn_duration_s: float
    #: Propellant consumed, kg.
    propellant_mass_kg: float


def finite_burn_loss(
    state0_eci: npt.ArrayLike,
    burn: FiniteBurn,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    total_time_s: float | None = None,
    impulse_epoch: ImpulseEpoch = ImpulseEpoch.IGNITION,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> FiniteBurnLoss:
    """Measure the error the impulsive assumption introduces for a given thrust level.

    Runs three propagations from the same initial state and differences them:

    1. the finite burn, via :func:`propagate_with_finite_burn`;
    2. the impulsive reference -- coast to ``impulse_epoch``, add
       ``burn.ideal_delta_v_m_s`` instantaneously in the commanded direction, coast on;
    3. an unpowered coast, which is what the achieved velocity change is measured against.

    Parameters
    ----------
    state0_eci
        Initial inertial state ``[r(3), v(3)]``, shape (6,).
    burn
        The manoeuvre. Its ``ideal_delta_v_m_s`` is the commanded impulsive delta-v the
        reference uses, so a burn specified by duration and one specified by delta-v are
        handled identically.
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Finite and non-negative.
    total_time_s
        Epoch at which the two trajectories are compared, seconds after the propagation
        epoch. Defaults to cut-off (``burn.end_time_s``), which is where the comparison is
        cleanest: after that, the two trajectories keep separating under their small
        velocity difference, so a later epoch measures that divergence as well as the burn.
        Must not be earlier than cut-off -- mid-burn there is no "outcome" to compare.
    impulse_epoch
        Where to put the reference impulse. :attr:`ImpulseEpoch.IGNITION` (default)
        reproduces what an impulsive planning tool assumes; :attr:`ImpulseEpoch.CENTROID`
        places it at the burn's delta-v centroid and cancels the leading kinematic term.
        The difference between the two is large and is the module's most actionable result.
    rtol, atol
        Integrator tolerances for all three propagations.

    Returns
    -------
    FiniteBurnLoss

    Raises
    ------
    ValueError
        Malformed state, negative ``mu_m3_s2``, a ``total_time_s`` earlier than cut-off, or
        an ``impulse_epoch`` that is not an :class:`ImpulseEpoch`.
    PropagationError
        If any of the three propagations fails.

    Notes
    -----
    For a :attr:`ThrustDirection.HILL_FIXED` burn the reference impulse is applied in the
    Hill frame of the *coasting* reference trajectory at ``impulse_epoch``. That is the
    consistent choice -- the reference is the impulsive vehicle, so it is the impulsive
    vehicle's frame that defines "along V-bar" -- and it is also the only one available
    without already having solved the powered problem.

    """
    state0 = _validate_state6(state0_eci)
    if not isinstance(burn, FiniteBurn):
        raise ValueError(f"burn must be a FiniteBurn, got {type(burn).__name__}")
    if not isinstance(impulse_epoch, ImpulseEpoch):
        raise ValueError(f"impulse_epoch must be an ImpulseEpoch, got {impulse_epoch!r}")
    mu = float(mu_m3_s2)
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu_m3_s2 must be finite and >= 0, got {mu_m3_s2!r}")

    comparison_time_s = burn.end_time_s if total_time_s is None else float(total_time_s)
    if not math.isfinite(comparison_time_s) or comparison_time_s < burn.end_time_s:
        raise ValueError(
            f"total_time_s must be finite and >= the burn cut-off time "
            f"{burn.end_time_s:.6g} s, got {total_time_s!r}; comparing part-way through a "
            "burn measures an unfinished manoeuvre against a completed impulse"
        )

    ideal_delta_v_m_s = burn.ideal_delta_v_m_s
    impulse_time_s = (
        burn.start_time_s
        if impulse_epoch is ImpulseEpoch.IGNITION
        else burn.delta_v_centroid_time_s
    )

    finite_state = propagate_with_finite_burn(
        state0, (0.0, comparison_time_s), burn, mu, rtol=rtol, atol=atol
    )[-1]
    coast_state = propagate_two_body(state0, (0.0, comparison_time_s), mu, rtol=rtol, atol=atol)[-1]

    state_at_impulse = (
        state0
        if impulse_time_s == 0.0
        else propagate_two_body(state0, (0.0, impulse_time_s), mu, rtol=rtol, atol=atol)[-1]
    )
    delta_v_eci = ideal_delta_v_m_s * thrust_unit_eci(
        state_at_impulse, burn.direction_unit, burn.direction_policy
    )
    kicked = state_at_impulse.copy()
    kicked[3:] += delta_v_eci
    remaining_s = comparison_time_s - impulse_time_s
    impulsive_state = (
        kicked
        if remaining_s == 0.0
        else propagate_two_body(kicked, (0.0, remaining_s), mu, rtol=rtol, atol=atol)[-1]
    )

    effective_delta_v_m_s = float(np.linalg.norm(finite_state[3:6] - coast_state[3:]))
    return FiniteBurnLoss(
        terminal_position_offset_m=float(np.linalg.norm(finite_state[:3] - impulsive_state[:3])),
        terminal_velocity_offset_m_s=float(np.linalg.norm(finite_state[3:6] - impulsive_state[3:])),
        ideal_delta_v_m_s=ideal_delta_v_m_s,
        effective_delta_v_m_s=effective_delta_v_m_s,
        extra_delta_v_m_s=ideal_delta_v_m_s - effective_delta_v_m_s,
        comparison_time_s=comparison_time_s,
        impulse_time_s=impulse_time_s,
        burn_duration_s=burn.burn_duration_s,
        propellant_mass_kg=burn.propellant_mass_kg,
    )
