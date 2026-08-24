r"""Lambert's problem: the fixed-time two-point boundary value problem of two-body motion.

Given two position vectors :math:`\mathbf{r}_1`, :math:`\mathbf{r}_2` about the same point
mass and a time of flight :math:`\Delta t`, find the conic arc that joins them in exactly
that time, and with it the two terminal velocities. This is the targeting primitive that
sits under every phasing burn, every intercept, and every free-flight leg of a rendezvous
profile.

Which formulation, and why
--------------------------
This module implements the **universal-variable** formulation (Bate-Mueller-White;
Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., sec. 7.6; Curtis,
*Orbital Mechanics for Engineering Students*, 3rd ed., sec. 5.3) rather than Izzo's method.
Izzo's is faster and better conditioned near the minimum-energy arc, and if this were a
Monte Carlo inner loop that would decide it. It is not; the decision here is testability:

* Every intermediate quantity has a closed form a test can assert against on its own. The
  Stumpff functions have both a series and an elementary closed form that must agree; the
  time equation is one scalar equation in one scalar unknown, and its root is bracketed, so
  "converged" is a checkable property rather than a reported one. Izzo's :math:`x` variable,
  its :math:`T(x)` series near :math:`x = 1`, and its Householder iterate have no comparably
  simple independent oracle -- an error in them would have to be caught by the same
  end-to-end test that catches everything else.
* One equation covers elliptic, parabolic and hyperbolic arcs with no branch on conic type.
  The conic type falls out of the sign of the converged :math:`z` instead of being decided
  in advance.

The equations
-------------
Write :math:`r_1 = \lVert \mathbf{r}_1 \rVert`, :math:`r_2 = \lVert \mathbf{r}_2 \rVert`,
and let :math:`\Delta\theta` be the transfer angle swept from :math:`\mathbf{r}_1` to
:math:`\mathbf{r}_2` in the direction of motion, so :math:`\Delta\theta \in (0, 2\pi)`.

The whole geometry of the problem enters through the single constant

.. math::

    A = \sin\Delta\theta \sqrt{\frac{r_1 r_2}{1 - \cos\Delta\theta}}
      = \sqrt{2 r_1 r_2}\, \cos\!\left(\frac{\Delta\theta}{2}\right),

the two forms being identical via :math:`1 - \cos\Delta\theta = 2\sin^2(\Delta\theta/2)`.
The half-angle form is what the code evaluates: the first form differences two numbers that
both tend to 1 as :math:`\Delta\theta \to 0` and loses precision exactly where transfers get
interesting. Note :math:`A > 0` for :math:`\Delta\theta < \pi`, :math:`A < 0` for
:math:`\Delta\theta > \pi`, and :math:`A = 0` at :math:`\Delta\theta = \pi` -- which is the
analytic reason the 180-degree transfer is degenerate rather than merely awkward, since
:math:`A` divides the terminal velocities below.

The Stumpff functions of the universal anomaly parameter :math:`z` are

.. math::

    C(z) = \sum_{k=0}^{\infty} \frac{(-z)^k}{(2k+2)!}, \qquad
    S(z) = \sum_{k=0}^{\infty} \frac{(-z)^k}{(2k+3)!},

with the elementary closed forms

.. math::

    C(z) = \frac{1 - \cos\sqrt{z}}{z},\quad
    S(z) = \frac{\sqrt{z} - \sin\sqrt{z}}{z^{3/2}} \qquad (z > 0)

and the hyperbolic counterparts for :math:`z < 0`. Here :math:`z = \chi^2 / a`: for an
ellipse :math:`z = \Delta E^2`, the square of the change in eccentric anomaly; for a
hyperbola :math:`z = -\Delta H^2`; :math:`z = 0` is the parabolic arc.

With

.. math::

    y(z) = r_1 + r_2 + A\,\frac{z S(z) - 1}{\sqrt{C(z)}}, \qquad
    \chi(z) = \sqrt{\frac{y(z)}{C(z)}},

the time of flight along the arc labelled by :math:`z` is

.. math::

    \sqrt{\mu}\, \Delta t(z) = \chi(z)^3 S(z) + A \sqrt{y(z)}.

That is the Lambert time equation. Solving :math:`\Delta t(z) = \Delta t` for :math:`z` is
the entire numerical content of this module. Everything after it is closed form: the
Lagrange coefficients

.. math::

    f = 1 - \frac{y}{r_1}, \qquad g = A\sqrt{\frac{y}{\mu}}, \qquad
    \dot{g} = 1 - \frac{y}{r_2}

give the terminal velocities directly,

.. math::

    \mathbf{v}_1 = \frac{1}{g}\left(\mathbf{r}_2 - f\,\mathbf{r}_1\right), \qquad
    \mathbf{v}_2 = \frac{1}{g}\left(\dot{g}\,\mathbf{r}_2 - \mathbf{r}_1\right).

Multiple revolutions
--------------------
Because :math:`z = \Delta E^2`, an arc that completes :math:`N` full revolutions before
arriving has :math:`\Delta E \in (2N\pi,\, 2(N+1)\pi)`, that is

.. math::

    z \in \left((2N\pi)^2,\ (2(N+1)\pi)^2\right).

At both ends of that interval :math:`C(z) \to 0`, so :math:`\chi \to \infty` and
:math:`\Delta t(z) \to \infty`. In between, :math:`\Delta t(z)` has a single interior
minimum. Hence for :math:`N \ge 1` there are **exactly two** solutions whenever the
requested time of flight exceeds that minimum, one on each side of it, and **none** below
it. Both are implemented (see ``branch``); a request below the minimum raises
:class:`~rpo_core.exceptions.InfeasibleTransferError` carrying the minimum time it missed.

For :math:`N = 0` the domain is :math:`z < 4\pi^2` and :math:`\Delta t(z)` is monotonically
increasing, so the solution is unique.

Direction of motion
-------------------
``prograde=True`` selects the arc whose specific angular momentum has a **positive ECI
z-component** -- inclination below 90 degrees, motion counter-clockwise seen from :math:`+z`.
Operationally: with :math:`\Delta\theta_0 = \arccos(\hat{r}_1 \cdot \hat{r}_2) \in [0, \pi]`,

* prograde takes :math:`\Delta\theta = \Delta\theta_0` when
  :math:`(\mathbf{r}_1 \times \mathbf{r}_2)_z > 0` and :math:`2\pi - \Delta\theta_0`
  otherwise;
* retrograde is the complement.

The two therefore trace opposite arcs of different planes and give genuinely different
velocities, never a sign flip of the same answer.

Validity
--------
Point-mass two-body only: no J2, no drag, no third bodies, no relativity, no finite-burn
modelling. It is the same model :func:`rpo_core.propagate.propagate_two_body` integrates,
which is what makes that function a legitimate independent oracle for this one.

The frame is the pseudo-inertial ECI of ``docs/conventions.md``. Because the
prograde/retrograde convention is tied to that frame's z axis, a transfer whose plane
**contains** the z axis (a polar transfer, :math:`h_z = 0`) has no prograde/retrograde
distinction at all: the sign of a rounding error would otherwise decide between two
completely different arcs. That case is rejected rather than silently resolved.

No check is made that the solved arc clears the central body. A mathematically valid
Lambert solution can pass through the Earth, and callers doing intercept design must screen
periapsis themselves.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from .constants import MU_EARTH_M3_S2
from .exceptions import DegenerateGeometryError, InfeasibleTransferError, RpoCoreError

#: Relative tolerance on the time-of-flight residual, ``|dt(z) - tof| <= rtol * tof``.
#:
#: Measured: the residual function is a sum of terms of order ``tof`` evaluated in double
#: precision, so its own noise floor is ~1e-15 relative. The bracketed solver reaches 1e-12
#: relative in four to thirteen iterations across every case in the test suite, which is
#: three decades of headroom above the noise floor and roughly six decades tighter than any
#: physically meaningful targeting requirement.
DEFAULT_TIME_RTOL: float = 1.0e-12

#: Iteration cap for the bracketed root solve. Exceeding it raises rather than returning.
DEFAULT_MAX_ITERATIONS: int = 100

#: ``|r1 x r2| / (r1 r2) = |sin(dtheta)|`` below which the two positions are treated as
#: collinear. At 1e-8 the transfer plane normal is still resolved to ~8 significant digits;
#: below it the cross product is dominated by cancellation in the input coordinates.
DEFAULT_COLLINEAR_TOL: float = 1.0e-8

#: ``|h_z| / |h|`` below which the transfer plane is treated as polar and the
#: prograde/retrograde choice as undefined. Deliberately tight: this rejects only the case
#: where the sign of ``h_z`` is numerical noise, not merely a high-inclination transfer.
DEFAULT_POLAR_TOL: float = 1.0e-12

#: Radius (m) at or below which a position vector is treated as degenerate. A point one
#: metre from a point mass is not a trajectory; it is a modelling error upstream.
DEFAULT_MIN_RADIUS_M: float = 1.0

#: ``|z| <= _SERIES_THRESHOLD`` uses the Stumpff power series instead of the closed form.
#: The closed forms differ two quantities that both tend to their leading term as
#: ``z -> 0``; at ``z = 1`` the series needs eight terms for full double precision, so
#: there is no reason to push the threshold lower.
_SERIES_THRESHOLD: float = 1.0

_TWO_PI: float = 2.0 * math.pi
_FOUR_PI_SQUARED: float = 4.0 * math.pi**2

#: Golden-section ratio, used to locate the multi-revolution minimum-time arc.
_INV_PHI: float = 0.5 * (math.sqrt(5.0) - 1.0)


class LambertConvergenceError(RpoCoreError, RuntimeError):
    """Raised when the Lambert time equation could not be solved to tolerance.

    Carries the iteration count, the final time-of-flight residual in seconds, and the
    requested time of flight, so the caller can tell "asked for something unreachable" from
    "asked for a tolerance the arithmetic cannot deliver" without attaching a debugger.

    This is deliberately not recoverable into a partial answer. The last iterate of a
    non-converged Newton-like solve is a velocity vector that looks entirely plausible and
    is wrong by an unbounded amount; returning it is the exact failure mode this package
    exists to prevent.
    """

    def __init__(self, message: str, *, iterations: int, residual_s: float, tof_s: float) -> None:
        """Record the iteration count, final residual (s), and requested time of flight (s)."""
        super().__init__(message)
        self.iterations = iterations
        self.residual_s = residual_s
        self.tof_s = tof_s


def _stumpff_series(z: float, offset: int) -> float:
    """Sum ``sum_k (-z)**k / (2k + offset)!`` for small ``|z|``.

    ``offset = 2`` gives ``C(z)``, ``offset = 3`` gives ``S(z)``. Successive terms are
    formed by recurrence rather than by evaluating factorials, which keeps the whole sum in
    the range of the leading term.
    """
    total = 1.0 / float(math.factorial(offset))
    term = total
    for k in range(1, 26):
        index = 2 * k + offset
        term *= -z / float(index * (index - 1))
        total += term
        if abs(term) <= 1e-18 * abs(total):
            break
    return total


def stumpff_c(z: float) -> float:
    r"""Return the Stumpff function :math:`C(z)`.

    Parameters
    ----------
    z
        Universal anomaly parameter, dimensionless. ``z > 0`` elliptic, ``z = 0`` parabolic,
        ``z < 0`` hyperbolic.

    Returns
    -------
    float
        :math:`(1 - \cos\sqrt{z})/z` for ``z > 0``, :math:`(\cosh\sqrt{-z} - 1)/(-z)` for
        ``z < 0``, and ``0.5`` at ``z = 0``. Evaluated by power series for ``|z| <= 1``,
        where the closed forms would cancel.

    """
    if abs(z) <= _SERIES_THRESHOLD:
        return _stumpff_series(z, 2)
    if z > 0.0:
        return (1.0 - math.cos(math.sqrt(z))) / z
    return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)


def stumpff_s(z: float) -> float:
    r"""Return the Stumpff function :math:`S(z)`.

    Parameters
    ----------
    z
        Universal anomaly parameter, dimensionless.

    Returns
    -------
    float
        :math:`(\sqrt{z} - \sin\sqrt{z})/z^{3/2}` for ``z > 0``,
        :math:`(\sinh\sqrt{-z} - \sqrt{-z})/(-z)^{3/2}` for ``z < 0``, and ``1/6`` at
        ``z = 0``. Evaluated by power series for ``|z| <= 1``.

    """
    if abs(z) <= _SERIES_THRESHOLD:
        return _stumpff_series(z, 3)
    if z > 0.0:
        root = math.sqrt(z)
        return (root - math.sin(root)) / root**3
    root = math.sqrt(-z)
    return (math.sinh(root) - root) / root**3


def _positive_float(value: float, name: str) -> float:
    """Return ``value`` as a validated finite positive float."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}")
    return number


def _position(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``value`` as a validated finite shape-(3,) float array."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array!r}")
    return array


def transfer_angle_rad(
    r1_eci_m: npt.ArrayLike,
    r2_eci_m: npt.ArrayLike,
    *,
    prograde: bool = True,
    collinear_tol: float = DEFAULT_COLLINEAR_TOL,
    polar_tol: float = DEFAULT_POLAR_TOL,
    min_radius_m: float = DEFAULT_MIN_RADIUS_M,
) -> float:
    r"""Return the swept transfer angle :math:`\Delta\theta \in (0, 2\pi)`, radians.

    Exposed separately from :func:`solve_lambert` because the direction-of-motion convention
    is the single most error-prone line in a Lambert implementation and deserves to be
    testable without solving anything.

    Parameters
    ----------
    r1_eci_m, r2_eci_m
        Departure and arrival position vectors in the inertial frame, metres, shape (3,).
    prograde
        ``True`` selects the arc with positive ECI z-component of angular momentum. See the
        module docstring.
    collinear_tol
        ``|sin(dtheta)|`` below which the two positions are collinear.
    polar_tol
        ``|h_z| / |h|`` below which prograde/retrograde is undefined.
    min_radius_m
        Radius at or below which a position vector is degenerate.

    Returns
    -------
    float
        Transfer angle in radians, strictly inside ``(0, 2*pi)``.

    Raises
    ------
    DegenerateGeometryError
        If either radius is at or below ``min_radius_m``; if the positions are collinear
        (transfer angle 0, pi, or 2*pi); or if the transfer plane is polar so that the
        prograde/retrograde choice is undefined. The measured quantity is named in every
        message.
    ValueError
        On malformed or non-finite input.

    """
    r1 = _position(r1_eci_m, "r1_eci_m")
    r2 = _position(r2_eci_m, "r2_eci_m")
    floor_m = _positive_float(min_radius_m, "min_radius_m")

    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    if r1_norm <= floor_m or r2_norm <= floor_m:
        raise DegenerateGeometryError(
            f"position vectors must be clear of the central body: |r1| = {r1_norm:.6g} m, "
            f"|r2| = {r2_norm:.6g} m, min_radius_m = {floor_m:.6g} m. No conic arc is "
            "defined through the singularity at the focus."
        )

    cross = np.cross(r1, r2)
    cross_norm = float(np.linalg.norm(cross))
    cos_theta = float(np.dot(r1, r2)) / (r1_norm * r2_norm)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    sin_theta = cross_norm / (r1_norm * r2_norm)

    if sin_theta < collinear_tol:
        if cos_theta < 0.0:
            raise DegenerateGeometryError(
                "r1 and r2 are antiparallel (transfer angle pi): "
                f"|r1 x r2| / (|r1| |r2|) = {sin_theta:.3e} < collinear_tol = "
                f"{collinear_tol:.3e} with cos(dtheta) = {cos_theta:.6f}. Infinitely many "
                "planes contain both vectors, so the transfer plane -- and with it the "
                "solution -- is undefined. Perturb one endpoint off the line, or solve the "
                "nearby 179.9-degree transfer and take the limit."
            )
        raise DegenerateGeometryError(
            "r1 and r2 are parallel (transfer angle 0 or 2*pi): "
            f"|r1 x r2| / (|r1| |r2|) = {sin_theta:.3e} < collinear_tol = "
            f"{collinear_tol:.3e} with cos(dtheta) = {cos_theta:.6f}. There is no swept "
            "arc to solve for."
        )

    normal_z_fraction = abs(float(cross[2])) / cross_norm
    if normal_z_fraction < polar_tol:
        raise DegenerateGeometryError(
            "the transfer plane contains the inertial z axis (polar transfer): "
            f"|h_z| / |h| = {normal_z_fraction:.3e} < polar_tol = {polar_tol:.3e}. The "
            "prograde/retrograde convention is defined by the sign of h_z, so at h_z = 0 "
            "it selects between two entirely different arcs on the strength of a rounding "
            "error. Specify the transfer by an explicit target plane instead."
        )

    theta = math.acos(cos_theta)
    is_direct = float(cross[2]) > 0.0
    if bool(prograde) != is_direct:
        theta = _TWO_PI - theta
    return theta


def _time_of_flight_s(
    z: float, a_geom_m: float, radius_sum_m: float, sqrt_mu: float
) -> float | None:
    """Return ``dt(z)`` in seconds, or ``None`` where the arc is not defined.

    ``None`` means ``y(z) <= 0``: no real arc of that universal anomaly connects the two
    points. It is returned rather than raised because the bracketing search probes outside
    the domain on purpose.
    """
    c = stumpff_c(z)
    if c <= 0.0:
        return None
    s = stumpff_s(z)
    y = radius_sum_m + a_geom_m * (z * s - 1.0) / math.sqrt(c)
    if y <= 0.0:
        return None
    chi = math.sqrt(y / c)
    return (chi**3 * s + a_geom_m * math.sqrt(y)) / sqrt_mu


def _bracketed_root(
    residual: Callable[[float], float | None],
    z_a: float,
    f_a: float,
    z_b: float,
    f_b: float,
    *,
    tol_s: float,
    max_iterations: int,
) -> tuple[float, int, float]:
    """Solve ``residual(z) = 0`` inside a sign-changing bracket, by the Illinois method.

    Regula falsi with the Illinois modification: superlinear like a secant iteration but,
    unlike Newton, it can never leave the bracket, so a converged answer here is converged
    by construction rather than by a reported flag. The derivative of the Lambert time
    equation exists in closed form and would be slightly faster, but making correctness
    depend on transcribing it correctly buys nothing -- the bracket is the guarantee.

    Returns ``(z, iterations, residual_s)``. The caller decides what a residual above
    tolerance means; this function does not raise.
    """
    a, fa = z_a, f_a
    b, fb = z_b, f_b
    z = 0.5 * (a + b)
    fz = math.inf
    iterations = 0
    retained = ""

    for iterations in range(1, max_iterations + 1):
        denominator = fb - fa
        z = b - fb * (b - a) / denominator if denominator != 0.0 else 0.5 * (a + b)
        if not min(a, b) < z < max(a, b):
            z = 0.5 * (a + b)

        probed = residual(z)
        if probed is None:  # pragma: no cover - the domain is convex inside a bracket
            z = 0.5 * (a + b)
            probed = residual(z)
            if probed is None:
                return z, iterations, math.inf
        fz = probed

        if abs(fz) <= tol_s:
            return z, iterations, fz

        # The Illinois halving applies only when the *same* endpoint is retained twice in
        # a row, which is the stagnation regula falsi actually suffers from. Halving on
        # every step instead corrupts both stored function values symmetrically and
        # collapses the whole method to bisection -- it still converges, so nothing fails
        # loudly; it just quietly costs three times the iterations.
        if (fz > 0.0) == (fa > 0.0):
            a, fa = z, fz
            if retained == "b":
                fb *= 0.5
            retained = "b"
        else:
            b, fb = z, fz
            if retained == "a":
                fa *= 0.5
            retained = "a"

        # The bracket has collapsed to adjacent floats without meeting tolerance: the
        # requested tolerance is below what the arithmetic can resolve. Report, don't fudge.
        if abs(b - a) <= 4.0 * math.ulp(max(abs(a), abs(b), 1.0)):
            return z, iterations, fz

    return z, iterations, fz


_NO_BRACKET: tuple[float, float, float, float] = (math.nan, math.nan, math.nan, math.nan)


def _walk_towards(
    anchor: float,
    limit: float,
    residual: Callable[[float], float | None],
    anchor_residual: float,
) -> tuple[float, float, float, float]:
    """Halve the gap from ``anchor`` towards ``limit`` until the residual changes sign.

    Trial ``k`` sits at ``anchor + (limit - anchor) * (1 - 2**-k)``, so consecutive trials
    close on ``limit`` geometrically. Returning the *last two* trials rather than
    ``(anchor, limit)`` is what makes the subsequent root solve fast: the returned bracket
    has width ``|limit - anchor| * 2**-k`` instead of the full interval, and the time
    equation is violently nonlinear near ``limit`` (where ``dt -> infinity``), so a wide
    bracket makes regula falsi stall against the singular end.

    Returns ``(z_lo, r_lo, z_hi, r_hi)`` ordered by ``z``, or NaNs if no sign change was
    found before the trials ran off the domain.
    """
    inner, inner_residual = anchor, anchor_residual
    for k in range(1, 61):
        trial = anchor + (limit - anchor) * (1.0 - 0.5**k)
        value = residual(trial)
        if value is None:
            continue
        if (value > 0.0) != (inner_residual > 0.0):
            if inner <= trial:
                return inner, inner_residual, trial, value
            return trial, value, inner, inner_residual
        inner, inner_residual = trial, value
    return _NO_BRACKET


def _bracket_single_revolution(
    residual: Callable[[float], float | None],
) -> tuple[float, float, float, float]:
    """Return a tight sign-changing bracket for the zero-revolution solution.

    ``dt(z)`` increases monotonically on ``z < 4*pi**2``, from 0 at the lower edge of the
    domain (where ``y(z) -> 0``) to infinity as ``z -> 4*pi**2``. So the sign of the
    residual at ``z = 0`` says which way to look: up towards ``4*pi**2`` for a slow
    transfer, down into the hyperbolic branch for a fast one.
    """
    at_origin = residual(0.0)
    if at_origin is None:  # pragma: no cover - requires y(0) == 0 exactly
        return _NO_BRACKET

    if at_origin < 0.0:
        return _walk_towards(0.0, _FOUR_PI_SQUARED, residual, at_origin)

    # Descend into z < 0. The domain's lower edge is where dt -> 0, so the residual is
    # guaranteed to change sign there; doubling the step finds it in a few evaluations and
    # halving recovers from a step that overshot the edge into y <= 0.
    upper, upper_residual = 0.0, at_origin
    step = 1.0
    for _ in range(400):
        trial = upper - step
        value = residual(trial)
        if value is None:
            step *= 0.5
            if step <= 1e-14:
                return _NO_BRACKET
            continue
        if value <= 0.0:
            return trial, value, upper, upper_residual
        upper, upper_residual = trial, value
        step *= 2.0
    return _NO_BRACKET


def _multi_revolution_interval(
    revolutions: int, tof_of: Callable[[float], float | None]
) -> tuple[float, float]:
    """Return usable interior endpoints of ``((2*N*pi)**2, (2*(N+1)*pi)**2)``.

    ``dt`` diverges at both ends because ``C(z) -> 0`` there, so the endpoints only need to
    be inside the domain and finite; the inset is grown from a very small value only if the
    evaluation there is unusable, which happens when ``y(z)`` leaves the domain rather than
    when the mathematics fails.
    """
    z_start = (_TWO_PI * revolutions) ** 2
    z_end = (_TWO_PI * (revolutions + 1)) ** 2
    span = z_end - z_start

    def usable_edge(sign: float) -> float:
        inset = 1.0e-9
        for _ in range(40):
            candidate = (z_start + inset * span) if sign > 0.0 else (z_end - inset * span)
            value = tof_of(candidate)
            if value is not None and math.isfinite(value) and value > 0.0:
                return candidate
            inset *= 2.0
        return math.nan

    return usable_edge(1.0), usable_edge(-1.0)


def _minimum_time_z(
    z_lower: float, z_upper: float, tof_of: Callable[[float], float | None]
) -> tuple[float, float]:
    """Locate the interior minimum of ``dt(z)`` by golden-section search.

    Derivative-free on purpose: ``d(dt)/dz`` changes sign steeply near the minimum-time arc,
    and a bracketing search that only ever compares function values cannot be walked out of
    the interval by a bad slope estimate.
    """

    def value_at(z: float) -> float:
        result = tof_of(z)
        return math.inf if result is None else result

    a, b = z_lower, z_upper
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = value_at(c), value_at(d)
    tolerance = 1.0e-12 * (z_upper - z_lower)
    for _ in range(300):
        if abs(b - a) <= tolerance:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = value_at(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = value_at(d)
    z_star = 0.5 * (a + b)
    return z_star, value_at(z_star)


def minimum_multirev_time_s(
    r1_eci_m: npt.ArrayLike,
    r2_eci_m: npt.ArrayLike,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    prograde: bool = True,
    revolutions: int = 1,
    collinear_tol: float = DEFAULT_COLLINEAR_TOL,
    polar_tol: float = DEFAULT_POLAR_TOL,
    min_radius_m: float = DEFAULT_MIN_RADIUS_M,
) -> float:
    """Return the shortest time of flight, seconds, admitting an ``N``-revolution transfer.

    Below this time no ``N``-revolution arc connects the two points; at exactly this time
    the two branches coincide; above it there are two. Useful for sizing a search before
    calling :func:`solve_lambert`, and it is the number quoted when ``solve_lambert`` refuses
    an infeasible multi-revolution request.

    Parameters
    ----------
    r1_eci_m, r2_eci_m
        Departure and arrival position vectors, metres, shape (3,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2.
    prograde
        Direction of motion; see the module docstring.
    revolutions
        Number of complete revolutions, must be >= 1.
    collinear_tol, polar_tol, min_radius_m
        Geometry-degeneracy thresholds, passed to :func:`transfer_angle_rad`.

    Returns
    -------
    float
        Minimum time of flight in seconds.

    Raises
    ------
    ValueError
        If ``revolutions < 1``, or on malformed input.
    DegenerateGeometryError
        As :func:`transfer_angle_rad`.

    """
    revs = int(revolutions)
    if revs < 1:
        raise ValueError(f"revolutions must be >= 1 for a minimum-time query, got {revolutions!r}")
    mu = _positive_float(mu_m3_s2, "mu_m3_s2")
    r1 = _position(r1_eci_m, "r1_eci_m")
    r2 = _position(r2_eci_m, "r2_eci_m")
    theta = transfer_angle_rad(
        r1,
        r2,
        prograde=prograde,
        collinear_tol=collinear_tol,
        polar_tol=polar_tol,
        min_radius_m=min_radius_m,
    )
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    a_geom = math.sqrt(2.0 * r1_norm * r2_norm) * math.cos(0.5 * theta)
    radius_sum = r1_norm + r2_norm
    sqrt_mu = math.sqrt(mu)

    def tof_of(z: float) -> float | None:
        return _time_of_flight_s(z, a_geom, radius_sum, sqrt_mu)

    lower, upper = _multi_revolution_interval(revs, tof_of)
    if not (math.isfinite(lower) and math.isfinite(upper)):  # pragma: no cover - defensive
        raise LambertConvergenceError(
            f"could not bracket the {revs}-revolution branch interval for this geometry",
            iterations=0,
            residual_s=math.inf,
            tof_s=math.nan,
        )
    return _minimum_time_z(lower, upper, tof_of)[1]


def solve_lambert(
    r1_eci_m: npt.ArrayLike,
    r2_eci_m: npt.ArrayLike,
    tof_s: float,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    prograde: bool = True,
    revolutions: int = 0,
    branch: Literal["low", "high"] = "low",
    time_rtol: float = DEFAULT_TIME_RTOL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    collinear_tol: float = DEFAULT_COLLINEAR_TOL,
    polar_tol: float = DEFAULT_POLAR_TOL,
    min_radius_m: float = DEFAULT_MIN_RADIUS_M,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Solve Lambert's problem: the conic arc joining two positions in a fixed time.

    Parameters
    ----------
    r1_eci_m, r2_eci_m
        Departure and arrival position vectors in the inertial frame, metres, shape (3,).
    tof_s
        Time of flight, seconds. Must be finite and strictly positive: the two-point
        boundary value problem has no content at ``tof = 0``, and a negative time of flight
        is a caller-side sign error, not a backwards transfer (swap the endpoints for that).
    mu_m3_s2
        Gravitational parameter of the central body, m^3/s^2. Defaults to Earth.
    prograde
        ``True`` selects the arc whose angular momentum has positive ECI z-component
        (inclination < 90 deg); ``False`` the retrograde arc. These are different transfers
        in different planes, not two signs of one answer.
    revolutions
        Number of complete revolutions ``N >= 0`` before arrival. ``N = 0`` has a unique
        solution; ``N >= 1`` has two, selected by ``branch``, and none at all below the
        minimum time returned by :func:`minimum_multirev_time_s`.
    branch
        Which of the two multi-revolution solutions to return. ``"low"`` is the root at
        smaller universal anomaly ``z`` than the minimum-time arc, ``"high"`` the root at
        larger ``z``. Ignored when ``revolutions == 0``.
    time_rtol
        Convergence tolerance on the time-of-flight residual, relative to ``tof_s``.
    max_iterations
        Cap on root-solve iterations before raising :class:`LambertConvergenceError`.
    collinear_tol, polar_tol, min_radius_m
        Geometry-degeneracy thresholds; see :func:`transfer_angle_rad`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(v1_eci_m_s, v2_eci_m_s)``, each shape (3,) in metres per second: the velocity at
        ``r1_eci_m`` on departure and at ``r2_eci_m`` on arrival.

    Raises
    ------
    DegenerateGeometryError
        Near-zero position vectors; a collinear pair (transfer angle 0, pi, or 2*pi); or a
        polar transfer plane for which prograde/retrograde is undefined. Each message names
        the measured quantity and the threshold it failed.
    InfeasibleTransferError
        If ``revolutions >= 1`` and ``tof_s`` is below the minimum time for that number of
        revolutions. The minimum is computed and reported.
    LambertConvergenceError
        If the time equation could not be bracketed, or was bracketed but not solved to
        ``time_rtol`` within ``max_iterations``. No best-effort iterate is returned.
    ValueError
        On malformed or non-finite input, non-positive ``tof_s`` or ``mu_m3_s2``, negative
        ``revolutions``, or an unrecognised ``branch``.

    Examples
    --------
    A quarter-orbit transfer in a 420 km circular LEO, checked against the circular speed:

    >>> import numpy as np
    >>> from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M, orbital_period_s
    >>> a = R_EARTH_EQUATORIAL_M + 420e3
    >>> v_circ = np.sqrt(MU_EARTH_M3_S2 / a)
    >>> v1, v2 = solve_lambert([a, 0.0, 0.0], [0.0, a, 0.0], 0.25 * orbital_period_s(a))
    >>> bool(abs(np.linalg.norm(v1) - v_circ) < 1e-6)
    True

    """
    tof = _positive_float(tof_s, "tof_s")
    mu = _positive_float(mu_m3_s2, "mu_m3_s2")
    if not isinstance(revolutions, (int, np.integer)) or isinstance(revolutions, bool):
        raise ValueError(f"revolutions must be an integer, got {revolutions!r}")
    revs = int(revolutions)
    if revs < 0:
        raise ValueError(f"revolutions must be >= 0, got {revolutions!r}")
    if branch not in ("low", "high"):
        raise ValueError(f"branch must be 'low' or 'high', got {branch!r}")
    iteration_cap = int(max_iterations)
    if iteration_cap < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
    tolerance_s = _positive_float(time_rtol, "time_rtol") * tof

    r1 = _position(r1_eci_m, "r1_eci_m")
    r2 = _position(r2_eci_m, "r2_eci_m")
    theta = transfer_angle_rad(
        r1,
        r2,
        prograde=prograde,
        collinear_tol=collinear_tol,
        polar_tol=polar_tol,
        min_radius_m=min_radius_m,
    )

    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    # A = sqrt(2 r1 r2) cos(dtheta/2), the half-angle form of the usual
    # sin(dtheta) sqrt(r1 r2 / (1 - cos dtheta)); identical, but free of the 0/0 that the
    # textbook form develops as dtheta -> 0.
    a_geom = math.sqrt(2.0 * r1_norm * r2_norm) * math.cos(0.5 * theta)
    radius_sum = r1_norm + r2_norm
    sqrt_mu = math.sqrt(mu)

    def tof_of(z: float) -> float | None:
        return _time_of_flight_s(z, a_geom, radius_sum, sqrt_mu)

    def residual_of(z: float) -> float | None:
        value = tof_of(z)
        return None if value is None else value - tof

    if revs == 0:
        z_lo, r_lo, z_hi, r_hi = _bracket_single_revolution(residual_of)
        if not math.isfinite(z_lo):
            raise LambertConvergenceError(
                f"could not bracket a zero-revolution solution for tof_s={tof:.6g} s with "
                f"transfer angle {math.degrees(theta):.4f} deg, |r1|={r1_norm:.6g} m, "
                f"|r2|={r2_norm:.6g} m. The time equation was evaluated across the whole "
                "domain z < 4*pi**2 without straddling the requested time.",
                iterations=0,
                residual_s=math.inf,
                tof_s=tof,
            )
    else:
        lower, upper = _multi_revolution_interval(revs, tof_of)
        if not (math.isfinite(lower) and math.isfinite(upper)):  # pragma: no cover
            raise LambertConvergenceError(
                f"could not bracket the {revs}-revolution interval "
                f"z in (({2 * revs}*pi)**2, ({2 * (revs + 1)}*pi)**2) for "
                f"tof_s={tof:.6g} s",
                iterations=0,
                residual_s=math.inf,
                tof_s=tof,
            )
        z_star, tof_min = _minimum_time_z(lower, upper, tof_of)
        if tof < tof_min:
            raise InfeasibleTransferError(
                f"no {revs}-revolution transfer exists at tof_s={tof:.6g} s: the "
                f"minimum-time {revs}-revolution arc between these points takes "
                f"{tof_min:.6g} s (short by {tof_min - tof:.6g} s). Allow more time, "
                f"reduce revolutions, or accept the {revs - 1}-revolution solution."
            )
        # Both branches are anchored at the minimum-time arc, where the residual is <= 0,
        # and walk outwards towards the interval edge where dt diverges.
        z_lo, r_lo, z_hi, r_hi = _walk_towards(
            z_star, lower if branch == "low" else upper, residual_of, tof_min - tof
        )
        if not math.isfinite(z_lo):  # pragma: no cover - defensive
            raise LambertConvergenceError(
                f"could not bracket the {branch} {revs}-revolution branch at "
                f"tof_s={tof:.6g} s (minimum-time arc {tof_min:.6g} s at z={z_star:.6g})",
                iterations=0,
                residual_s=math.inf,
                tof_s=tof,
            )

    if (r_lo > 0.0) == (r_hi > 0.0):  # pragma: no cover - defensive
        raise LambertConvergenceError(
            f"time-equation bracket does not straddle tof_s={tof:.6g} s: residuals "
            f"{r_lo:.6g} s and {r_hi:.6g} s at z = {z_lo:.6g} and {z_hi:.6g}",
            iterations=0,
            residual_s=min(abs(r_lo), abs(r_hi)),
            tof_s=tof,
        )

    z, iterations, residual = _bracketed_root(
        residual_of, z_lo, r_lo, z_hi, r_hi, tol_s=tolerance_s, max_iterations=iteration_cap
    )
    if not math.isfinite(residual) or abs(residual) > tolerance_s:
        raise LambertConvergenceError(
            f"Lambert time equation did not converge: after {iterations} iterations the "
            f"time-of-flight residual is {residual:.6g} s against a tolerance of "
            f"{tolerance_s:.6g} s (time_rtol={time_rtol:.3e} on tof_s={tof:.6g} s), at "
            f"z = {z:.10g}. Returning this iterate would hand back a velocity that looks "
            "reasonable and is not.",
            iterations=iterations,
            residual_s=residual,
            tof_s=tof,
        )

    c = stumpff_c(z)
    s = stumpff_s(z)
    y = radius_sum + a_geom * (z * s - 1.0) / math.sqrt(c)

    f = 1.0 - y / r1_norm
    g = a_geom * math.sqrt(y / mu)
    g_dot = 1.0 - y / r2_norm
    if g == 0.0:  # pragma: no cover - excluded upstream by the collinearity check
        raise DegenerateGeometryError(
            f"Lagrange coefficient g vanished (A = {a_geom:.6g}, y = {y:.6g}); the "
            "terminal velocities are undefined for this geometry"
        )

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):  # pragma: no cover
        raise LambertConvergenceError(
            f"Lambert solve produced non-finite velocities at z = {z:.10g}: "
            f"v1 = {v1!r}, v2 = {v2!r}",
            iterations=iterations,
            residual_s=residual,
            tof_s=tof,
        )
    return v1, v2
