"""Typed exceptions for rpo_core.

Every numerical routine that can fail raises one of these rather than returning a
silently-wrong value. A targeting solve that returns a 10 km/s delta-v because a matrix
was singular is far more dangerous than one that refuses to answer.
"""

from __future__ import annotations


class RpoCoreError(Exception):
    """Base class for all rpo_core errors."""


class DegenerateGeometryError(RpoCoreError, ValueError):
    """Raised when an orbital state is geometrically degenerate.

    Examples: zero position or velocity vector, or a radius/velocity pair that is
    exactly parallel (zero specific angular momentum, i.e. a purely radial trajectory),
    for which no orbit plane -- and therefore no LVLH frame -- is defined.
    """


class SingularTransferTimeError(RpoCoreError, ValueError):
    """Raised when the CW position-from-velocity block is singular at the requested TOF.

    The two-impulse Clohessy-Wiltshire solve inverts ``Phi_rv``. That block loses rank at
    specific transfer times, and the two subproblems fail at *different* times because
    in-plane and cross-track motion decouple exactly:

    * In-plane (x, y): the 2x2 determinant is ``(8 - 8*cos(tau) - 3*tau*sin(tau)) / n**2``,
      which vanishes at ``tau = 2*pi*k`` -- integer multiples of the orbital period.
    * Cross-track (z): the scalar term is ``sin(tau) / n``, which vanishes at
      ``tau = pi*k`` -- integer multiples of *half* the orbital period.

    The half-period case is the common trap: it is a perfectly good in-plane transfer time
    (and the natural choice for a V-bar hop) while being completely unable to change
    cross-track position. A single 3x3 conditioning check would reject that valid planar
    transfer; this package therefore checks the two subproblems separately.
    """


class InfeasibleTransferError(RpoCoreError, ValueError):
    """Raised when a requested terminal state cannot be reached at the requested TOF.

    Distinct from :class:`SingularTransferTimeError`: the solve is not ill-conditioned,
    the target is simply unreachable. Occurs for cross-track at half-period transfer times,
    where ``z(t_f)`` is pinned to ``cos(tau) * z_0`` regardless of the applied impulse.
    """


class PropagationError(RpoCoreError, RuntimeError):
    """Raised when a numerical propagation fails to complete.

    The integrator reporting failure is information, not noise. Returning a partial or
    last-known-good trajectory here would let a silently truncated run flow into a metrics
    table as if it were a completed one.
    """
