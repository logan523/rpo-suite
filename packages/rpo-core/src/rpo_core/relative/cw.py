r"""Clohessy-Wiltshire (Hill) linearised relative motion.

The equations
-------------
For a chaser near a target on a **circular** orbit of mean motion :math:`n`, linearising
the difference of the two two-body accelerations about the target and expressing the
result in the rotating Hill frame (x radial-outward, y along-track, z cross-track) gives
the Clohessy-Wiltshire equations:

.. math::

    \ddot{x} - 3 n^2 x - 2 n \dot{y} &= 0 \\
    \ddot{y} + 2 n \dot{x}           &= 0 \\
    \ddot{z} + n^2 z                 &= 0

Three things are worth reading off directly, because they are what the tests check:

1. **Cross-track decouples.** ``z`` is a simple harmonic oscillator at exactly the orbital
   frequency, independent of the in-plane motion.
2. **The in-plane motion has a secular term.** Integrating the ``y`` equation gives
   :math:`\dot{y} + 2 n x = \text{const}`, and the along-track drift rate is
   :math:`-3 n (2 n x_0 + \dot{y}_0) t`. It vanishes -- giving a closed, repeating relative
   orbit -- only when :math:`\dot{y}_0 = -2 n x_0`. This is the *drift-free* (or
   energy-matching) condition, and it is the single most useful sanity check on a Hill-frame
   implementation: get a sign wrong anywhere and the closed ellipse stops closing.
3. **A purely radial impulse produces a closed 2:1 ellipse.** From rest at the origin, an
   impulse :math:`\Delta v_x` gives :math:`x = (\Delta v_x / n)\sin nt` and
   :math:`y = -(2\Delta v_x / n)(1 - \cos nt)`: an ellipse twice as long along-track as it
   is tall radially, which returns to the origin after exactly one orbital period.

Validity
--------
This is a *linearisation* about a *circular* orbit. Its error grows with separation and
with elapsed time, and it carries no J2, no drag, and no eccentricity. It is the right
model for terminal proximity operations over a fraction of an orbit and the wrong model
for far-range phasing. The quantitative error envelope is measured, not assumed -- see the
CW-versus-nonlinear study in ``docs/`` -- and nothing in this package silently extrapolates
beyond it.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ..exceptions import InfeasibleTransferError, SingularTransferTimeError

#: Condition-number limit that identifies the *exact* in-plane singularity, and nothing
#: softer than that.
#:
#: Measured behaviour: the in-plane condition number grows only as ``~3 / (1 - t/T)``, so
#: at 0.99999 orbital periods it is ~3e5 -- three hundred times below this limit -- and the
#: solved impulse is still a well-scaled 0.045 m/s for a representative V-bar hop. In other
#: words this is a backstop against the singular case where the solve returns genuine
#: garbage, **not** a general accuracy envelope. Transfer times close to ``k*T`` are
#: numerically fine here; the limit only trips within ~3e-8 of an exact period multiple.
#: See ``test_condition_number_grows_slowly_near_the_singularity``.
SINGULARITY_CONDITION_LIMIT: float = 1.0e8

#: ``|sin(n*dt)|`` below which the cross-track solve is treated as rank-deficient.
DEFAULT_CROSS_TRACK_SIN_TOL: float = 1.0e-8

#: Absolute position tolerance (m) for deciding whether a rank-deficient cross-track
#: request is nonetheless satisfiable.
DEFAULT_FEASIBILITY_TOL_M: float = 1.0e-6


def _validate_mean_motion(n_rad_s: float) -> float:
    """Return ``n_rad_s`` as a validated positive float."""
    n = float(n_rad_s)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"n_rad_s must be a finite positive mean motion, got {n_rad_s!r}")
    return n


def cw_dynamics_matrix(n_rad_s: float) -> npt.NDArray[np.float64]:
    """Return the 6x6 continuous-time CW plant matrix ``A`` for ``xdot = A x``.

    State ordering is ``[x, y, z, xdot, ydot, zdot]``. Provided both for independent
    verification of :func:`cw_stm` by numerical integration and for later use by the
    LQR/MPC baselines.
    """
    n = _validate_mean_motion(n_rad_s)
    a = np.zeros((6, 6), dtype=np.float64)
    a[0, 3] = 1.0
    a[1, 4] = 1.0
    a[2, 5] = 1.0
    a[3, 0] = 3.0 * n**2
    a[3, 4] = 2.0 * n
    a[4, 3] = -2.0 * n
    a[5, 2] = -(n**2)
    return a


def cw_stm(n_rad_s: float, dt_s: float) -> npt.NDArray[np.float64]:
    """Return the closed-form 6x6 Clohessy-Wiltshire state transition matrix.

    Parameters
    ----------
    n_rad_s
        Target mean motion, rad/s. Must be finite and strictly positive. Note this is the
        **target's** mean motion; using the chaser's is a common and silent error.
    dt_s
        Elapsed time, seconds. May be negative (the STM is defined for propagation
        backwards in time).

    Returns
    -------
    numpy.ndarray
        Shape (6, 6). ``state(t0 + dt) = Phi @ state(t0)`` with state ordering
        ``[x, y, z, xdot, ydot, zdot]`` in metres and metres per second.

    Notes
    -----
    Mixed units: the position-from-velocity block has units of seconds and the
    velocity-from-position block units of 1/s. Do not take a norm of the raw matrix and
    expect it to mean anything.

    """
    n = _validate_mean_motion(n_rad_s)
    dt = float(dt_s)
    if not math.isfinite(dt):
        raise ValueError(f"dt_s must be finite, got {dt_s!r}")

    tau = n * dt
    s = math.sin(tau)
    c = math.cos(tau)

    phi = np.zeros((6, 6), dtype=np.float64)

    # Position from initial position.
    phi[0, 0] = 4.0 - 3.0 * c
    phi[1, 0] = 6.0 * (s - tau)
    phi[1, 1] = 1.0
    phi[2, 2] = c

    # Position from initial velocity (units: seconds).
    phi[0, 3] = s / n
    phi[0, 4] = 2.0 * (1.0 - c) / n
    phi[1, 3] = -2.0 * (1.0 - c) / n
    phi[1, 4] = (4.0 * s - 3.0 * tau) / n
    phi[2, 5] = s / n

    # Velocity from initial position (units: 1/s).
    phi[3, 0] = 3.0 * n * s
    phi[4, 0] = -6.0 * n * (1.0 - c)
    phi[5, 2] = -n * s

    # Velocity from initial velocity (dimensionless).
    phi[3, 3] = c
    phi[3, 4] = 2.0 * s
    phi[4, 3] = -2.0 * s
    phi[4, 4] = 4.0 * c - 3.0
    phi[5, 5] = c

    return phi


def propagate_cw(n_rad_s: float, state_hill: npt.ArrayLike, dt_s: float) -> npt.NDArray[np.float64]:
    """Propagate a Hill-frame relative state by ``dt_s`` using the CW STM.

    Parameters
    ----------
    n_rad_s
        Target mean motion, rad/s.
    state_hill
        Shape (6,): ``[x, y, z, xdot, ydot, zdot]``, metres and m/s.
    dt_s
        Elapsed time, seconds.

    """
    state = np.asarray(state_hill, dtype=np.float64)
    if state.shape != (6,):
        raise ValueError(f"state_hill must have shape (6,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"state_hill must be finite, got {state!r}")
    return cw_stm(n_rad_s, dt_s) @ state


def two_impulse_transfer(
    n_rad_s: float,
    r0_hill_m: npt.ArrayLike,
    v0_hill_m_s: npt.ArrayLike,
    rf_hill_m: npt.ArrayLike,
    vf_hill_m_s: npt.ArrayLike,
    tof_s: float,
    *,
    max_condition: float = SINGULARITY_CONDITION_LIMIT,
    cross_track_sin_tol: float = DEFAULT_CROSS_TRACK_SIN_TOL,
    feasibility_tol_m: float = DEFAULT_FEASIBILITY_TOL_M,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Solve the two-impulse CW rendezvous problem.

    Given an initial relative state, a desired terminal relative state, and a fixed time of
    flight, find the departure impulse :math:`\Delta v_1` that places the chaser on a
    coasting arc arriving at the target position, and the arrival impulse
    :math:`\Delta v_2` that matches the target velocity.

    Partitioning the STM into 3x3 blocks,

    .. math::

        \begin{bmatrix} r_f \\ v_f^- \end{bmatrix} =
        \begin{bmatrix} \Phi_{rr} & \Phi_{rv} \\ \Phi_{vr} & \Phi_{vv} \end{bmatrix}
        \begin{bmatrix} r_0 \\ v_0^+ \end{bmatrix},

    the post-burn departure velocity follows from
    :math:`v_0^+ = \Phi_{rv}^{-1}(r_f - \Phi_{rr} r_0)`, giving
    :math:`\Delta v_1 = v_0^+ - v_0` and
    :math:`\Delta v_2 = v_f - (\Phi_{vr} r_0 + \Phi_{vv} v_0^+)`.

    The in-plane and cross-track subproblems are solved **separately** because they
    decouple exactly and become singular at different times: in-plane at integer multiples
    of the orbital period, cross-track at integer multiples of the *half* period. Applying
    a single 3x3 conditioning test would reject the half-period V-bar hop, which is a
    perfectly well-posed planar transfer and the natural baseline manoeuvre.

    Parameters
    ----------
    n_rad_s
        Target mean motion, rad/s.
    r0_hill_m, v0_hill_m_s
        Initial relative position (m) and velocity (m/s), shape (3,) each.
    rf_hill_m, vf_hill_m_s
        Desired terminal relative position (m) and velocity (m/s), shape (3,) each.
    tof_s
        Time of flight, seconds. Must be strictly positive.
    max_condition
        Condition-number limit identifying the exact in-plane singularity. See
        :data:`SINGULARITY_CONDITION_LIMIT` -- this is not a general accuracy envelope.
    cross_track_sin_tol
        ``|sin(n*tof)|`` below which cross-track targeting is rank-deficient.
    feasibility_tol_m
        Position tolerance for accepting a rank-deficient cross-track request.

    Returns
    -------
    tuple of numpy.ndarray
        ``(dv1, dv2)``, each shape (3,) in m/s, expressed in the Hill frame.

    Raises
    ------
    SingularTransferTimeError
        If the in-plane block is ill-conditioned at ``tof_s`` (transfer time at or near an
        integer number of orbital periods). The condition number is reported.
    InfeasibleTransferError
        If cross-track targeting is rank-deficient *and* the requested terminal cross-track
        position is not the one the coast is pinned to.
    ValueError
        On malformed or non-finite inputs, or non-positive ``tof_s``.

    Examples
    --------
    A half-period V-bar hop from -1000 m to -250 m, both at rest in the Hill frame:

    >>> import numpy as np
    >>> from rpo_core.constants import mean_motion_rad_s, orbital_period_s
    >>> a = 6378137.0 + 420e3
    >>> n = mean_motion_rad_s(a)
    >>> dv1, dv2 = two_impulse_transfer(
    ...     n, [0.0, -1000.0, 0.0], [0.0, 0.0, 0.0],
    ...     [0.0, -250.0, 0.0], [0.0, 0.0, 0.0], 0.5 * orbital_period_s(a))
    >>> bool(np.linalg.norm(dv1) > 0.0)
    True

    """
    n = _validate_mean_motion(n_rad_s)
    tof = float(tof_s)
    if not math.isfinite(tof) or tof <= 0.0:
        raise ValueError(f"tof_s must be a finite positive time of flight, got {tof_s!r}")

    def _vec(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite, got {array!r}")
        return array

    r0 = _vec(r0_hill_m, "r0_hill_m")
    v0 = _vec(v0_hill_m_s, "v0_hill_m_s")
    rf = _vec(rf_hill_m, "rf_hill_m")
    vf = _vec(vf_hill_m_s, "vf_hill_m_s")

    phi = cw_stm(n, tof)
    phi_rr = phi[:3, :3]
    phi_rv = phi[:3, 3:]
    phi_vr = phi[3:, :3]
    phi_vv = phi[3:, 3:]

    # Position shortfall that the departure velocity must make up.
    delta = rf - phi_rr @ r0

    v0_plus = np.empty(3, dtype=np.float64)

    # --- In-plane (x, y) ---------------------------------------------------------------
    in_plane = phi_rv[np.ix_([0, 1], [0, 1])]
    condition = float(np.linalg.cond(in_plane))
    if not math.isfinite(condition) or condition > max_condition:
        periods = tof * n / (2.0 * math.pi)
        raise SingularTransferTimeError(
            "in-plane Phi_rv block is singular or ill-conditioned at "
            f"tof_s={tof:.6g} s (= {periods:.6f} orbital periods): condition number "
            f"{condition:.3e} exceeds max_condition={max_condition:.3e}. The in-plane "
            "two-impulse solve loses rank at integer multiples of the orbital period; "
            "choose a different time of flight or add an intermediate manoeuvre."
        )
    v0_plus[[0, 1]] = np.linalg.solve(in_plane, delta[[0, 1]])

    # --- Cross-track (z) ---------------------------------------------------------------
    sin_tau = math.sin(n * tof)
    if abs(sin_tau) < cross_track_sin_tol:
        # z(t_f) is pinned to cos(tau) * z_0 regardless of the impulse. Either the request
        # already matches that value -- in which case the minimum-norm answer is to leave
        # zdot alone -- or it is unreachable at this time of flight.
        if abs(delta[2]) > feasibility_tol_m:
            half_periods = tof * n / math.pi
            raise InfeasibleTransferError(
                "cross-track targeting is rank-deficient at "
                f"tof_s={tof:.6g} s (= {half_periods:.6f} half-periods, "
                f"|sin(n*tof)|={abs(sin_tau):.3e}): z(t_f) is pinned to "
                f"{float(phi_rr[2, 2] * r0[2]):.6g} m, but "
                f"{float(rf[2]):.6g} m was requested "
                f"(shortfall {float(delta[2]):.6g} m > feasibility_tol_m="
                f"{feasibility_tol_m:.3e} m). Cross-track position cannot be changed at "
                "integer multiples of the half period; choose a different time of flight."
            )
        v0_plus[2] = v0[2]
    else:
        v0_plus[2] = delta[2] / phi_rv[2, 2]

    dv1 = v0_plus - v0
    vf_minus = phi_vr @ r0 + phi_vv @ v0_plus
    dv2 = vf - vf_minus

    if not (np.all(np.isfinite(dv1)) and np.all(np.isfinite(dv2))):  # pragma: no cover
        raise SingularTransferTimeError(
            f"two-impulse solve produced non-finite delta-v at tof_s={tof:.6g} s: "
            f"dv1={dv1!r}, dv2={dv2!r}"
        )
    return dv1, dv2
