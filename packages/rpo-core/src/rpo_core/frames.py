"""ECI <-> Hill (LVLH) frame transformations.

Frame conventions (locked; see ``docs/conventions.md``)
------------------------------------------------------
**Inertial frame.** An Earth-centred pseudo-inertial frame, taken as GCRF-approximated.
Precession, nutation, polar motion, and frame-tie corrections are neglected. This is an
explicit modelling choice, not an oversight; the induced error is quantified in
``docs/frames.md`` rather than assumed negligible.

**Hill / LVLH frame.** Origin at the *target* centre of mass, rotating with the target:

===== ================= ==================================================
Axis  Name              Definition
===== ================= ==================================================
``x`` radial (R-bar)    ``r_hat`` -- along the target radius, positive
                        *away from* the central body
``z`` cross-track       ``h_hat`` -- along the specific angular momentum
                        ``r x v``, i.e. the *positive* orbit normal
``y`` along-track       ``z_hat x x_hat`` -- completes the right-handed
                        set; for a circular orbit this is ``v_hat``
                        (V-bar, positive along the velocity)
===== ================= ==================================================

The triad is right-handed with ``x_hat x y_hat = z_hat``. Note that ``z`` is the
*positive* orbit normal: since ``x_hat = r_hat`` and ``y_hat ~ v_hat``, the cross product
``r_hat x v_hat`` is by definition ``h_hat``.

``y`` is built as ``z_hat x x_hat`` rather than taken directly from ``v_hat`` so that the
triad is exactly orthonormal even for an eccentric orbit, where ``v`` is not perpendicular
to ``r``. For a circular orbit the two definitions coincide.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._validate import as_vector
from .exceptions import DegenerateGeometryError

Vec3 = npt.NDArray[np.float64]
Mat3 = npt.NDArray[np.float64]

#: Relative tolerance below which a cross product is treated as degenerate.
_DEGENERATE_REL_TOL: float = 1e-12


def hill_basis(r_eci_m: npt.ArrayLike, v_eci_m_s: npt.ArrayLike) -> tuple[Mat3, Vec3]:
    """Return the ECI-to-Hill rotation matrix and the frame angular velocity.

    Parameters
    ----------
    r_eci_m
        Target inertial position, metres, shape (3,).
    v_eci_m_s
        Target inertial velocity, metres per second, shape (3,).

    Returns
    -------
    rotation_eci_to_hill
        3x3 orthonormal matrix whose *rows* are ``x_hat``, ``y_hat``, ``z_hat`` expressed
        in ECI. A free vector transforms as ``a_hill = rotation @ a_eci``.
    omega_eci_rad_s
        Angular velocity of the Hill frame with respect to ECI, expressed in **ECI**
        components, rad/s. Equal to ``h / |r|**2``.

    Raises
    ------
    DegenerateGeometryError
        If the position or velocity is (near) zero, or if ``r`` and ``v`` are (near)
        parallel, in which case the orbit plane -- and hence the frame -- is undefined.

    Notes
    -----
    ``omega`` is returned in ECI components because that is the form needed by the
    transport theorem when differencing inertial velocities; rotate it with ``rotation``
    if Hill components are wanted.

    """
    r = as_vector(r_eci_m, "r_eci_m")
    v = as_vector(v_eci_m_s, "v_eci_m_s")

    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm == 0.0:
        raise DegenerateGeometryError("r_eci_m has zero magnitude; no radial direction exists")
    if v_norm == 0.0:
        raise DegenerateGeometryError("v_eci_m_s has zero magnitude; no orbit plane exists")

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    # |r x v| = |r||v|sin(angle); a vanishing ratio means r and v are parallel, i.e. a
    # purely radial (rectilinear) trajectory with no orbit plane.
    if h_norm <= _DEGENERATE_REL_TOL * r_norm * v_norm:
        raise DegenerateGeometryError(
            "r_eci_m and v_eci_m_s are parallel to within "
            f"{_DEGENERATE_REL_TOL:g} (|r x v| / (|r||v|) = {h_norm / (r_norm * v_norm):.3e}); "
            "specific angular momentum is zero and the LVLH frame is undefined"
        )

    x_hat = r / r_norm
    z_hat = h / h_norm
    y_hat = np.cross(z_hat, x_hat)

    rotation_eci_to_hill = np.vstack((x_hat, y_hat, z_hat))
    omega_eci_rad_s = h / r_norm**2
    return rotation_eci_to_hill, omega_eci_rad_s


def relative_state_eci_to_hill(
    r_target_eci_m: npt.ArrayLike,
    v_target_eci_m_s: npt.ArrayLike,
    r_chaser_eci_m: npt.ArrayLike,
    v_chaser_eci_m_s: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Map an absolute chaser/target ECI pair to a relative state in the target Hill frame.

    Applies the transport theorem: the Hill-frame relative velocity is the *rotating-frame*
    derivative, ``R @ (dv_eci - omega x dr_eci)``, not simply the rotated inertial
    difference. Omitting the ``omega x dr`` term is a classic error that produces a
    velocity offset proportional to separation -- roughly 0.1 m/s per km in LEO, which is
    the same order as the manoeuvres being designed.

    Returns
    -------
    numpy.ndarray
        Shape (6,): ``[x, y, z, xdot, ydot, zdot]`` in metres and metres per second.

    """
    r_t = as_vector(r_target_eci_m, "r_target_eci_m")
    v_t = as_vector(v_target_eci_m_s, "v_target_eci_m_s")
    r_c = as_vector(r_chaser_eci_m, "r_chaser_eci_m")
    v_c = as_vector(v_chaser_eci_m_s, "v_chaser_eci_m_s")

    rotation, omega_eci = hill_basis(r_t, v_t)
    dr_eci = r_c - r_t
    dv_eci = v_c - v_t

    r_hill = rotation @ dr_eci
    v_hill = rotation @ (dv_eci - np.cross(omega_eci, dr_eci))
    return np.concatenate((r_hill, v_hill))


def relative_state_hill_to_eci(
    r_target_eci_m: npt.ArrayLike,
    v_target_eci_m_s: npt.ArrayLike,
    relative_state_hill: npt.ArrayLike,
) -> tuple[Vec3, Vec3]:
    """Invert :func:`relative_state_eci_to_hill`, returning absolute chaser ECI state.

    Returns
    -------
    tuple
        ``(r_chaser_eci_m, v_chaser_eci_m_s)``, each shape (3,).

    """
    state = np.asarray(relative_state_hill, dtype=np.float64)
    if state.shape != (6,):
        raise ValueError(f"relative_state_hill must have shape (6,), got {state.shape}")

    r_t = as_vector(r_target_eci_m, "r_target_eci_m")
    v_t = as_vector(v_target_eci_m_s, "v_target_eci_m_s")
    rotation, omega_eci = hill_basis(r_t, v_t)

    dr_eci = rotation.T @ state[:3]
    dv_eci = rotation.T @ state[3:] + np.cross(omega_eci, dr_eci)
    return r_t + dr_eci, v_t + dv_eci
