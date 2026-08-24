r"""Classical (Keplerian) orbital elements and their conversion to and from ECI states.

The equations
-------------
For a closed two-body orbit, the inertial state :math:`[\mathbf{r}, \mathbf{v}]` and the six
classical elements :math:`(a, e, i, \Omega, \omega, \nu)` carry the same information. The
forward map (RV2COE) is built from three vectors:

.. math::

    \mathbf{h} &= \mathbf{r} \times \mathbf{v}
        &&\text{(specific angular momentum, normal to the orbit plane)} \\
    \mathbf{n} &= \hat{\mathbf{z}} \times \mathbf{h}
        &&\text{(node vector, along the ascending node)} \\
    \mathbf{e} &= \frac{(v^{2} - \mu/r)\,\mathbf{r} - (\mathbf{r}\cdot\mathbf{v})\,\mathbf{v}}{\mu}
        &&\text{(eccentricity vector, pointing at periapsis)}

with the size fixed by the specific energy,

.. math::

    \xi = \frac{v^{2}}{2} - \frac{\mu}{r},
    \qquad a = -\frac{\mu}{2\xi},
    \qquad p = a(1 - e^{2}) = \frac{h^{2}}{\mu}.

The four angles are each an inverse cosine plus a *quadrant test*, and the quadrant tests
are the whole difficulty of this routine:

.. math::

    i      &= \arccos(h_z / h)                              &&\in [0, \pi] \\
    \Omega &= \arccos(n_x / n),      &&\ n_y < 0 \Rightarrow \Omega = 2\pi - \Omega \\
    \omega &= \arccos(\mathbf{n}\cdot\mathbf{e} / (n e)),
               &&\ e_z < 0 \Rightarrow \omega = 2\pi - \omega \\
    \nu    &= \arccos(\mathbf{e}\cdot\mathbf{r} / (e r)),
               &&\ \mathbf{r}\cdot\mathbf{v} < 0 \Rightarrow \nu = 2\pi - \nu

Inclination needs no quadrant test because it is defined on :math:`[0, \pi]`; the other
three do, and dropping one produces an answer that is right in half the sky and reflected
in the other half. A round-trip test alone does not always catch that, which is why
:mod:`tests.test_elements` checks all four quadrants against an independently constructed
rotation.

The inverse map (COE2RV) builds the state in the perifocal frame, where periapsis is the
:math:`+x` axis and the orbit is planar,

.. math::

    \mathbf{r}_{pf} = \frac{p}{1 + e\cos\nu}
        \begin{bmatrix}\cos\nu \\ \sin\nu \\ 0\end{bmatrix},
    \qquad
    \mathbf{v}_{pf} = \sqrt{\frac{\mu}{p}}
        \begin{bmatrix}-\sin\nu \\ e + \cos\nu \\ 0\end{bmatrix},

and rotates it into the inertial frame with the 3-1-3 sequence
:math:`R_3(-\Omega)\,R_1(-i)\,R_3(-\omega)` -- equivalently the active rotation
:math:`R_z(\Omega)\,R_x(i)\,R_z(\omega)`.

Singular cases
--------------
Three of the six elements are **not defined** for certain orbit geometries, and this module
refuses to invent a value for them:

* **Circular** (:math:`e < ` :data:`CIRCULAR_ECCENTRICITY_TOL`): there is no periapsis, so
  :math:`\omega` and :math:`\nu` are undefined. The eccentricity *vector* is numerical
  noise and its direction is meaningless.
* **Equatorial** (:math:`\sin i < ` :data:`EQUATORIAL_SINE_TOL`, covering both
  :math:`i \approx 0` and :math:`i \approx \pi`): the orbit plane and the equator do not
  intersect in a line, so there is no ascending node and :math:`\Omega` is undefined.
* **Circular equatorial**: all three of :math:`\Omega, \omega, \nu` are undefined.

Returning ``0.0`` or ``nan`` in these cases would be a silent lie: ``0.0`` is
indistinguishable from a genuine measurement, and ``nan`` propagates into a covariance or a
plot without saying why. :func:`cartesian_to_classical` instead raises
:class:`UndefinedOrbitalElementError`, and the well-defined replacement angles are
available as separate functions -- :func:`argument_of_latitude_rad`,
:func:`true_longitude_rad`, :func:`longitude_of_periapsis_rad` -- which is the same
substitution the standard references make.

Validity
--------
**Closed orbits only.** Parabolic and hyperbolic states (:math:`e \ge 1`) are rejected with
:class:`NonClosedOrbitError` rather than converted. The formula :math:`a = -\mu/(2\xi)`
happily returns a *negative* semi-major axis for a hyperbola, and that negative number
flows downstream into :func:`~rpo_core.constants.mean_motion_rad_s`, period calculations,
and any element-space propagation as a domain error at best and a plausible wrong answer at
worst.

Pure two-body geometry: these are *osculating* elements. Under any perturbation (J2, drag,
third bodies) they are the elements of the instantaneously tangent Keplerian orbit and they
change with time; nothing here models that. Units are SI -- metres, seconds, radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .constants import MU_EARTH_M3_S2
from .exceptions import DegenerateGeometryError, RpoCoreError

#: Eccentricity below which the orbit is treated as circular and the periapsis direction as
#: undefined.
#:
#: Chosen from what the number *means* physically rather than from floating-point noise. At
#: ``e = 1e-8`` on a 7000 km orbit the difference between apoapsis and periapsis radius is
#: ``2ae ~ 0.14 m`` -- well below the accuracy of any real orbit determination, so a
#: "periapsis" located at that eccentricity is an artefact of the fit, not a feature of the
#: orbit. The floating-point floor is far lower (the eccentricity vector is resolvable to
#: ``~1e-15`` in double precision), so this threshold leaves roughly seven decades of margin
#: above the noise: when the code says the orbit is circular, it is circular for a physical
#: reason and not because the arithmetic ran out of digits.
CIRCULAR_ECCENTRICITY_TOL: float = 1.0e-8

#: ``sin(i)`` below which the orbit is treated as equatorial and the line of nodes as
#: undefined. Tested on the *sine* so that it covers retrograde equatorial orbits
#: (``i ~ pi``) with the same threshold as prograde ones (``i ~ 0``).
#:
#: Same reasoning as :data:`CIRCULAR_ECCENTRICITY_TOL`: at ``sin(i) = 1e-8`` the maximum
#: out-of-plane excursion of a 7000 km orbit is ``a sin(i) ~ 7 cm``. The node vector
#: ``n = z_hat x h`` has magnitude ``h sin(i)``, so this is exactly a relative test on
#: ``|n| / |h|`` and inherits no scale from the orbit size.
EQUATORIAL_SINE_TOL: float = 1.0e-8

#: Largest eccentricity accepted as a closed orbit.
#:
#: Not exactly 1.0: at ``e = 1 - 1e-12`` the semi-major axis is already ``~5e11`` times the
#: periapsis radius and ``a = -mu/(2 xi)`` is computed from a difference that has lost
#: twelve of its sixteen significant digits. Anything at or above this is reported as
#: non-closed rather than converted into an ``a`` that is mostly rounding error.
MAX_CLOSED_ECCENTRICITY: float = 1.0 - 1.0e-12

#: Relative size below which the equatorial-plane projection of a direction vector is
#: treated as absent, making its right ascension undefined.
#:
#: Only reachable in one geometry: a polar orbit whose position (or periapsis) lies exactly
#: over a pole, where ``atan2(0, 0)`` would silently return ``0.0``.
_POLAR_PROJECTION_REL_TOL: float = 1.0e-12

#: ``|r x v| / (|r||v|)`` below which the trajectory is treated as rectilinear -- ``r`` and
#: ``v`` parallel, no orbit plane. Matches the threshold used by :mod:`rpo_core.frames`, so
#: a state that has no LVLH frame also has no elements, rather than the two modules
#: disagreeing about where degeneracy starts.
_RECTILINEAR_REL_TOL: float = 1.0e-12

_TWO_PI: float = 2.0 * math.pi


class UndefinedOrbitalElementError(RpoCoreError, ValueError):
    """Raised when a requested element does not exist for the given orbit geometry.

    This is a statement about the *orbit*, not about the arithmetic. A circular orbit has
    no periapsis, so its argument of periapsis and true anomaly are not small or uncertain
    -- they do not exist. An equatorial orbit has no line of nodes, so it has no right
    ascension of the ascending node.

    The alternative to raising is to return ``0.0`` (indistinguishable from a real value)
    or ``nan`` (propagates silently into means, covariances, and plots). Both are worse:
    the caller's next step is usually to difference two element sets or to feed them to a
    controller, and either sink will happily consume a fabricated angle.

    Attributes
    ----------
    undefined_elements
        Names of the elements that are undefined, in the order they appear in
        :class:`ClassicalElements`.

    """

    def __init__(self, message: str, undefined_elements: tuple[str, ...] = ()) -> None:
        """Store the message and the names of the elements that do not exist."""
        super().__init__(message)
        self.undefined_elements = undefined_elements


class NonClosedOrbitError(RpoCoreError, ValueError):
    """Raised when a state or element set describes a parabolic or hyperbolic orbit.

    This module covers closed (elliptical and circular) orbits only. For ``e >= 1`` the
    specific energy is non-negative and ``a = -mu / (2 xi)`` is negative or infinite. A
    negative semi-major axis is a perfectly conventional way to write a hyperbola, but it
    is *not* what the rest of this package expects: ``mean_motion_rad_s`` rejects it,
    ``orbital_period_s`` is meaningless for it, and any element-space propagation built on
    Kepler's equation would need the hyperbolic form instead. Returning it here would push
    that confusion downstream to whichever consumer notices first.
    """


@dataclass(frozen=True, slots=True)
class ClassicalElements:
    """Classical (Keplerian) orbital elements of a closed two-body orbit.

    All values are SI: metres and radians. Instances are immutable, so an element set can
    be shared without a defensive copy.

    Attributes
    ----------
    semi_major_axis_m
        Semi-major axis ``a``, metres. Strictly positive: closed orbits only.
    eccentricity
        Eccentricity ``e``, dimensionless, in ``[0, 1)``.
    inclination_rad
        Inclination ``i``, radians, in ``[0, pi]``. Always defined.
    raan_rad
        Right ascension of the ascending node ``Omega``, radians. Undefined for equatorial
        orbits; :func:`cartesian_to_classical` raises rather than fabricating one.
    arg_periapsis_rad
        Argument of periapsis ``omega``, radians. Undefined for circular orbits.
    true_anomaly_rad
        True anomaly ``nu``, radians. Undefined for circular orbits.

    Notes
    -----
    Angles produced by :func:`cartesian_to_classical` are wrapped to ``[0, 2*pi)`` (except
    inclination, on ``[0, pi]``). Angles *accepted* by :func:`classical_to_cartesian` need
    not be: the trigonometric functions are periodic and an unwrapped angle is not an error.

    Raises
    ------
    ValueError
        On non-finite values, non-positive semi-major axis, or negative eccentricity.
    NonClosedOrbitError
        If ``eccentricity >= 1``.

    """

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_periapsis_rad: float
    true_anomaly_rad: float

    def __post_init__(self) -> None:
        """Reject element sets that do not describe a closed orbit."""
        values = (
            self.semi_major_axis_m,
            self.eccentricity,
            self.inclination_rad,
            self.raan_rad,
            self.arg_periapsis_rad,
            self.true_anomaly_rad,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"all elements must be finite, got {values!r}")
        if self.semi_major_axis_m <= 0.0:
            raise ValueError(
                "semi_major_axis_m must be > 0 for a closed orbit, got "
                f"{self.semi_major_axis_m!r} m"
            )
        if self.eccentricity < 0.0:
            raise ValueError(f"eccentricity must be >= 0, got {self.eccentricity!r}")
        if self.eccentricity >= MAX_CLOSED_ECCENTRICITY:
            raise NonClosedOrbitError(
                f"eccentricity {self.eccentricity!r} is at or above "
                f"MAX_CLOSED_ECCENTRICITY={MAX_CLOSED_ECCENTRICITY!r}; this module covers "
                "closed (elliptical and circular) orbits only"
            )


def _wrap_two_pi(angle_rad: float) -> float:
    """Wrap an angle to ``[0, 2*pi)``.

    ``2*pi - x`` for a tiny ``x`` rounds to exactly ``2*pi`` in double precision, which
    would leak an out-of-range angle out of an otherwise correct quadrant test.
    """
    wrapped = math.fmod(angle_rad, _TWO_PI)
    if wrapped < 0.0:
        wrapped += _TWO_PI
    return 0.0 if wrapped >= _TWO_PI else wrapped


def _angle_about_axis(
    from_vec: npt.NDArray[np.float64],
    to_vec: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
) -> float:
    """Return the angle from ``from_vec`` to ``to_vec`` about ``axis``, on ``[0, 2*pi)``.

    All three vectors may have any magnitude; only directions matter. ``from_vec`` and
    ``to_vec`` are assumed perpendicular to ``axis`` (true for every pair used here: the
    node, eccentricity and position vectors all lie in the orbit plane, whose normal is
    ``h``).

    Written as ``atan2`` rather than ``arccos`` plus a sign test. The two are algebraically
    identical -- ``arccos`` of the parallel component, reflected when the perpendicular
    component is negative -- but they are not numerically identical. ``arccos`` near an
    argument of +/-1 has a square-root singularity in its derivative, so an angle within
    ``eps`` of 0 or ``pi`` is recovered with absolute error ``~sqrt(eps) = 1.5e-8`` rad. The
    ``atan2`` form has no such loss anywhere on the circle, and gets the quadrant from the
    sign of the perpendicular component rather than from a separate branch that is easy to
    omit.
    """
    axis_norm = float(np.linalg.norm(axis))
    parallel = float(np.dot(from_vec, to_vec))
    perpendicular = float(np.dot(np.cross(axis, from_vec), to_vec)) / axis_norm
    return _wrap_two_pi(math.atan2(perpendicular, parallel))


def _as_state6(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``value`` to a finite float64 inertial state of shape (6,)."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array!r}")
    return array


def _validate_mu(mu_m3_s2: float, name: str = "mu_m3_s2") -> float:
    """Return ``mu_m3_s2`` as a validated finite positive float."""
    mu = float(mu_m3_s2)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"{name} must be a finite positive gravitational parameter, got {mu!r}")
    return mu


def _orbit_vectors(
    state_eci: npt.ArrayLike, mu_m3_s2: float, name: str
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """Return ``(r, v, h, e_vec, mu)`` after validating the state and rejecting bad geometry.

    Every public entry point in this module starts here, so the validation branches -- shape,
    finiteness, positive ``mu``, zero radius, zero angular momentum, non-closed orbit -- are
    enforced identically no matter which element is being asked for.
    """
    state = _as_state6(state_eci, name)
    mu = _validate_mu(mu_m3_s2)

    r = state[:3]
    v = state[3:]
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm == 0.0:
        raise DegenerateGeometryError(
            f"{name} has zero position magnitude; the orbit is undefined at the central body "
            "singularity"
        )

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    # |r x v| = |r||v| sin(angle). A vanishing ratio is a rectilinear (purely radial)
    # trajectory: no orbit plane exists, so neither does an inclination or a node line.
    if v_norm == 0.0 or h_norm <= _RECTILINEAR_REL_TOL * r_norm * v_norm:
        raise DegenerateGeometryError(
            f"{name} has (near-)zero specific angular momentum "
            f"(|r x v| = {h_norm:.6g} m^2/s, |r||v| = {r_norm * v_norm:.6g} m^2/s); the "
            "trajectory is rectilinear and has no orbit plane"
        )

    e_vec = ((v_norm**2 - mu / r_norm) * r - float(np.dot(r, v)) * v) / mu
    e = float(np.linalg.norm(e_vec))
    if e >= MAX_CLOSED_ECCENTRICITY:
        xi = 0.5 * v_norm**2 - mu / r_norm
        raise NonClosedOrbitError(
            f"eccentricity {e:.12g} is at or above MAX_CLOSED_ECCENTRICITY="
            f"{MAX_CLOSED_ECCENTRICITY!r} (specific energy {xi:.6g} J/kg, |r| = {r_norm:.6g} m, "
            f"|v| = {v_norm:.6g} m/s): the orbit is parabolic or hyperbolic and this module "
            "covers closed orbits only. a = -mu/(2*xi) would be negative or infinite here."
        )
    return r, v, h, e_vec, mu


def eccentricity_vector(
    state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2
) -> npt.NDArray[np.float64]:
    """Return the eccentricity vector ``e``, dimensionless, pointing at periapsis.

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, metres and metres per second, shape (6,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2.

    Returns
    -------
    numpy.ndarray
        Shape (3,). Magnitude is the eccentricity; direction is from the central body
        towards periapsis. For a circular orbit the magnitude is (numerically) zero and the
        **direction is meaningless** -- see :data:`CIRCULAR_ECCENTRICITY_TOL`.

    Raises
    ------
    DegenerateGeometryError
        Zero radius, or a rectilinear trajectory.
    NonClosedOrbitError
        If the state is parabolic or hyperbolic.
    ValueError
        Wrong shape, non-finite input, or non-positive ``mu_m3_s2``.

    Notes
    -----
    Exposed separately from :func:`cartesian_to_classical` because it is defined for every
    closed orbit, including the circular ones for which the *angles* are not. That makes it
    the honest way to ask "how circular is this orbit?" without catching an exception.

    """
    _, _, _, e_vec, _ = _orbit_vectors(state_eci, mu_m3_s2, "state_eci")
    return e_vec


def inclination_rad(state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return the inclination ``i = arccos(h_z / |h|)``, radians, on ``[0, pi]``.

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, metres and metres per second, shape (6,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Used only for the closed-orbit check.

    Returns
    -------
    float
        Inclination in radians. ``i < pi/2`` is prograde, ``i > pi/2`` retrograde.

    Raises
    ------
    DegenerateGeometryError
        Zero radius, or a rectilinear trajectory.
    NonClosedOrbitError
        If the state is parabolic or hyperbolic.
    ValueError
        Wrong shape, non-finite input, or non-positive ``mu_m3_s2``.

    Notes
    -----
    Never undefined for a state that has an orbit plane at all, and needs no quadrant test:
    inclination lives on ``[0, pi]``, which is exactly the range of ``arccos``. It is
    therefore available even for the circular and equatorial orbits whose other angles are
    not.

    """
    _, _, h, _, _ = _orbit_vectors(state_eci, mu_m3_s2, "state_eci")
    # atan2(|h_xy|, h_z) rather than arccos(h_z / |h|): identical in exact arithmetic, but
    # arccos loses half the significant digits when i is near 0 or pi, which is precisely
    # the near-equatorial regime this accessor exists to serve.
    return math.atan2(math.hypot(float(h[0]), float(h[1])), float(h[2]))


def cartesian_to_classical(
    state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2
) -> ClassicalElements:
    """Convert an inertial Cartesian state to classical orbital elements (RV2COE).

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, metres and metres per second, shape (6,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Defaults to Earth.

    Returns
    -------
    ClassicalElements
        Osculating elements. ``inclination_rad`` is on ``[0, pi]``; ``raan_rad``,
        ``arg_periapsis_rad`` and ``true_anomaly_rad`` are wrapped to ``[0, 2*pi)``.

    Raises
    ------
    UndefinedOrbitalElementError
        If the orbit is circular (no periapsis, so ``omega`` and ``nu`` do not exist),
        equatorial (no line of nodes, so ``Omega`` does not exist), or both. The message
        names the missing elements, reports the measured ``e`` and ``sin(i)`` against the
        thresholds that tripped, and points at the replacement angle.
    NonClosedOrbitError
        If ``e >= 1``: parabolic and hyperbolic states are rejected, not converted.
    DegenerateGeometryError
        Zero radius, or a rectilinear trajectory with no orbit plane.
    ValueError
        Wrong shape, non-finite input, or non-positive ``mu_m3_s2``.

    Examples
    --------
    >>> import numpy as np
    >>> from rpo_core.constants import MU_EARTH_M3_S2
    >>> a = 7000e3
    >>> state = classical_to_cartesian(
    ...     ClassicalElements(a, 0.01, 0.9, 0.5, 1.2, 2.0), MU_EARTH_M3_S2)
    >>> elements = cartesian_to_classical(state, MU_EARTH_M3_S2)
    >>> bool(np.isclose(elements.semi_major_axis_m, a))
    True

    """
    r, v, h, e_vec, mu = _orbit_vectors(state_eci, mu_m3_s2, "state_eci")

    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    h_norm = float(np.linalg.norm(h))
    e = float(np.linalg.norm(e_vec))

    xi = 0.5 * v_norm**2 - mu / r_norm
    a = -mu / (2.0 * xi)

    # n = z_hat x h, so |n| = |h| sin(i): the equatorial test below is a relative test on
    # sin(i) and carries no dependence on orbit size.
    n_vec = np.array([-float(h[1]), float(h[0]), 0.0])
    n_norm = float(np.linalg.norm(n_vec))
    sin_inclination = n_norm / h_norm
    inclination = math.atan2(n_norm, float(h[2]))

    is_circular = e < CIRCULAR_ECCENTRICITY_TOL
    is_equatorial = sin_inclination < EQUATORIAL_SINE_TOL
    if is_circular or is_equatorial:
        raise _undefined_element_error(e, sin_inclination, is_circular, is_equatorial)

    # n lies in the equatorial plane, so its quadrant is fixed by the sign of n_y.
    raan = _wrap_two_pi(math.atan2(float(n_vec[1]), float(n_vec[0])))
    # Both measured in the orbit plane, about h, in the direction of motion. The
    # perpendicular components that set the quadrants are e_z / sin(i) for omega and
    # (r . v) * p / (h e) for nu -- i.e. exactly the textbook sign tests on e_z and r . v,
    # arrived at without a branch.
    arg_periapsis = _angle_about_axis(n_vec, e_vec, h)
    true_anomaly = _angle_about_axis(e_vec, r, h)

    return ClassicalElements(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_rad=inclination,
        raan_rad=raan,
        arg_periapsis_rad=arg_periapsis,
        true_anomaly_rad=true_anomaly,
    )


def _undefined_element_error(
    e: float, sin_inclination: float, is_circular: bool, is_equatorial: bool
) -> UndefinedOrbitalElementError:
    """Build the :class:`UndefinedOrbitalElementError` for a circular and/or equatorial orbit.

    Kept separate so the three message variants sit next to each other and stay consistent;
    each names the missing elements, the measured quantity that tripped the threshold, and
    the replacement function.
    """
    measured = (
        f"e = {e:.6g} (tol {CIRCULAR_ECCENTRICITY_TOL:g}), "
        f"sin(i) = {sin_inclination:.6g} (tol {EQUATORIAL_SINE_TOL:g})"
    )
    if is_circular and is_equatorial:
        return UndefinedOrbitalElementError(
            "raan_rad, arg_periapsis_rad and true_anomaly_rad are undefined for a circular "
            f"equatorial orbit [{measured}]: there is no periapsis to measure from and no "
            "line of nodes to measure it against. Use true_longitude_rad(), which is "
            "defined for this geometry.",
            ("raan_rad", "arg_periapsis_rad", "true_anomaly_rad"),
        )
    if is_circular:
        return UndefinedOrbitalElementError(
            "arg_periapsis_rad and true_anomaly_rad are undefined for a circular orbit "
            f"[{measured}]: the orbit has no periapsis, so the eccentricity vector is "
            "numerical noise and its direction carries no information. Use "
            "argument_of_latitude_rad(), which measures from the ascending node instead.",
            ("arg_periapsis_rad", "true_anomaly_rad"),
        )
    return UndefinedOrbitalElementError(
        f"raan_rad is undefined for an equatorial orbit [{measured}]: the orbit plane and "
        "the equator coincide, so they share no line of nodes for the ascending node to lie "
        "on. Use longitude_of_periapsis_rad(), which measures periapsis from the inertial "
        "x-axis directly.",
        ("raan_rad",),
    )


def classical_to_cartesian(
    elements: ClassicalElements, mu_m3_s2: float = MU_EARTH_M3_S2
) -> npt.NDArray[np.float64]:
    """Convert classical orbital elements to an inertial Cartesian state (COE2RV).

    Parameters
    ----------
    elements
        Classical elements. Validated on construction; see :class:`ClassicalElements`.
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Defaults to Earth.

    Returns
    -------
    numpy.ndarray
        Shape (6,): ``[r(3), v(3)]`` in metres and metres per second.

    Raises
    ------
    ValueError
        If ``elements`` is not a :class:`ClassicalElements`, or ``mu_m3_s2`` is not finite
        and positive.

    Notes
    -----
    No singular cases exist in this direction. A circular or equatorial element set maps to
    a perfectly good state; it is only the *inverse* that cannot recover which angle was
    which. The asymmetry is real, not an implementation limitation.

    """
    if not isinstance(elements, ClassicalElements):
        raise ValueError(f"elements must be a ClassicalElements, got {type(elements).__name__}")
    mu = _validate_mu(mu_m3_s2)

    a = elements.semi_major_axis_m
    e = elements.eccentricity
    nu = elements.true_anomaly_rad

    semi_latus_rectum = a * (1.0 - e**2)
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)

    radius = semi_latus_rectum / (1.0 + e * cos_nu)
    r_perifocal = radius * np.array([cos_nu, sin_nu, 0.0])
    v_perifocal = math.sqrt(mu / semi_latus_rectum) * np.array([-sin_nu, e + cos_nu, 0.0])

    rotation = _perifocal_to_eci(
        elements.raan_rad, elements.inclination_rad, elements.arg_periapsis_rad
    )
    return np.concatenate((rotation @ r_perifocal, rotation @ v_perifocal))


def _perifocal_to_eci(
    raan_rad: float, inclination_rad_: float, arg_periapsis_rad: float
) -> npt.NDArray[np.float64]:
    """Return the 3-1-3 rotation ``R_z(Omega) R_x(i) R_z(omega)`` mapping perifocal to ECI.

    Written out rather than composed from three matrix products: the composition is the
    same arithmetic with three extra temporaries, and the expanded form is what the
    standard references print, which makes it directly checkable against them.
    """
    c_raan, s_raan = math.cos(raan_rad), math.sin(raan_rad)
    c_inc, s_inc = math.cos(inclination_rad_), math.sin(inclination_rad_)
    c_argp, s_argp = math.cos(arg_periapsis_rad), math.sin(arg_periapsis_rad)

    return np.array(
        [
            [
                c_raan * c_argp - s_raan * s_argp * c_inc,
                -c_raan * s_argp - s_raan * c_argp * c_inc,
                s_raan * s_inc,
            ],
            [
                s_raan * c_argp + c_raan * s_argp * c_inc,
                -s_raan * s_argp + c_raan * c_argp * c_inc,
                -c_raan * s_inc,
            ],
            [s_argp * s_inc, c_argp * s_inc, c_inc],
        ]
    )


def argument_of_latitude_rad(state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return the argument of latitude ``u``, radians, on ``[0, 2*pi)``.

    ``u`` is the angle from the ascending node to the position vector, measured in the
    orbit plane in the direction of motion. For a non-circular orbit it equals
    ``omega + nu``; unlike that sum it survives ``e -> 0``, which makes it the standard
    replacement element for circular inclined orbits.

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, metres and metres per second, shape (6,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Used only for the closed-orbit check.

    Returns
    -------
    float
        Argument of latitude in radians, wrapped to ``[0, 2*pi)``.

    Raises
    ------
    UndefinedOrbitalElementError
        If the orbit is equatorial: there is no ascending node to measure from. Use
        :func:`true_longitude_rad` instead.
    NonClosedOrbitError
        If the state is parabolic or hyperbolic.
    DegenerateGeometryError
        Zero radius, or a rectilinear trajectory.
    ValueError
        Wrong shape, non-finite input, or non-positive ``mu_m3_s2``.

    Notes
    -----
    The quadrant test is ``r_z``: the spacecraft is north of the equator on the half of the
    orbit between the ascending and descending nodes, i.e. ``0 < u < pi``. This holds for
    retrograde orbits too, because ``u`` is measured in the direction of motion and the
    node vector flips with the sense of ``h``.

    """
    r, _, h, _, _ = _orbit_vectors(state_eci, mu_m3_s2, "state_eci")
    h_norm = float(np.linalg.norm(h))
    n_vec = np.array([-float(h[1]), float(h[0]), 0.0])
    n_norm = float(np.linalg.norm(n_vec))

    if n_norm / h_norm < EQUATORIAL_SINE_TOL:
        raise UndefinedOrbitalElementError(
            "argument of latitude is undefined for an equatorial orbit "
            f"[sin(i) = {n_norm / h_norm:.6g} < EQUATORIAL_SINE_TOL="
            f"{EQUATORIAL_SINE_TOL:g}]: there is no line of nodes to measure from. Use "
            "true_longitude_rad(), which is defined for any orbit.",
            ("argument_of_latitude",),
        )

    return _angle_about_axis(n_vec, r, h)


def true_longitude_rad(state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return the true longitude ``lambda``, radians, on ``[0, 2*pi)``.

    The right ascension of the position vector: the angle in the equatorial plane from the
    inertial x-axis to the projection of ``r``, ``atan2(r_y, r_x)``. It needs no periapsis
    and no line of nodes, which makes it the replacement element for circular equatorial
    orbits -- the one geometry in which *three* of the six classical elements vanish at
    once.

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, metres and metres per second, shape (6,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2. Used only for the closed-orbit check.

    Returns
    -------
    float
        True longitude in radians, wrapped to ``[0, 2*pi)``.

    Raises
    ------
    DegenerateGeometryError
        If the position vector lies along the polar axis, where its equatorial projection
        vanishes and its right ascension is undefined. Also zero radius or a rectilinear
        trajectory.
    NonClosedOrbitError
        If the state is parabolic or hyperbolic.
    ValueError
        Wrong shape, non-finite input, or non-positive ``mu_m3_s2``.

    Notes
    -----
    For an **equatorial** orbit this is exactly the classical ``lambda_true = Omega + omega
    + nu``, which is how the standard references define it and the only case in which they
    use it. For an inclined orbit the two differ -- ``Omega + omega + nu`` is a dogleg sum
    of angles measured in two different planes, whereas this function always returns a
    right ascension. The uniform definition is deliberate: it is the quantity that is
    actually well defined for every orbit, which is what the docstring promises.

    A polar orbit passing directly over a pole is the one geometry where even this fails,
    and it raises rather than returning the ``0.0`` that ``atan2(0, 0)`` would hand back.

    """
    r, _, _, _, _ = _orbit_vectors(state_eci, mu_m3_s2, "state_eci")
    return _right_ascension_rad(r, "position vector", "true longitude")


def longitude_of_periapsis_rad(state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return the longitude of periapsis ``varpi``, radians, on ``[0, 2*pi)``.

    The right ascension of the eccentricity vector: the angle in the equatorial plane from
    the inertial x-axis to the projection of the periapsis direction, ``atan2(e_y, e_x)``.
    It needs no line of nodes, which makes it the replacement element for **elliptical
    equatorial** orbits, where ``Omega`` -- and therefore ``omega`` measured from it -- does
    not exist.

    Parameters
    ----------
    state_eci
        Inertial state ``[r(3), v(3)]``, metres and metres per second, shape (6,).
    mu_m3_s2
        Gravitational parameter, m^3/s^2.

    Returns
    -------
    float
        Longitude of periapsis in radians, wrapped to ``[0, 2*pi)``.

    Raises
    ------
    UndefinedOrbitalElementError
        If the orbit is circular: there is no periapsis. Use :func:`true_longitude_rad`.
    DegenerateGeometryError
        If periapsis lies along the polar axis (a polar orbit with periapsis over a pole),
        where the equatorial projection of ``e`` vanishes. Also zero radius or a
        rectilinear trajectory.
    NonClosedOrbitError
        If the state is parabolic or hyperbolic.
    ValueError
        Wrong shape, non-finite input, or non-positive ``mu_m3_s2``.

    Notes
    -----
    For an **equatorial** orbit this equals the classical ``varpi = Omega + omega`` (the
    only case in which the references use it). For an inclined orbit it does not: the
    classical ``varpi`` is a dogleg sum across two planes, while this is a right ascension,
    consistently with :func:`true_longitude_rad`. When the orbit is inclined, ``Omega`` and
    ``omega`` are both individually available from :func:`cartesian_to_classical`, so
    nothing is lost by that choice.

    """
    _, _, _, e_vec, _ = _orbit_vectors(state_eci, mu_m3_s2, "state_eci")
    e = float(np.linalg.norm(e_vec))
    if e < CIRCULAR_ECCENTRICITY_TOL:
        raise UndefinedOrbitalElementError(
            "longitude of periapsis is undefined for a circular orbit "
            f"[e = {e:.6g} < CIRCULAR_ECCENTRICITY_TOL={CIRCULAR_ECCENTRICITY_TOL:g}]: the "
            "orbit has no periapsis. Use true_longitude_rad(), which is defined for any "
            "orbit.",
            ("longitude_of_periapsis",),
        )
    return _right_ascension_rad(e_vec, "eccentricity vector", "longitude of periapsis")


def _right_ascension_rad(
    vector: npt.NDArray[np.float64], vector_name: str, element_name: str
) -> float:
    """Return ``atan2(y, x)`` wrapped to ``[0, 2*pi)``, refusing a polar-axis vector.

    ``atan2(0.0, 0.0)`` returns ``0.0`` without complaint, which is precisely the kind of
    plausible-looking wrong answer this package does not emit.
    """
    norm = float(np.linalg.norm(vector))
    equatorial_projection = math.hypot(float(vector[0]), float(vector[1]))
    if equatorial_projection <= _POLAR_PROJECTION_REL_TOL * norm:
        raise DegenerateGeometryError(
            f"the {vector_name} lies along the polar axis "
            f"(equatorial projection {equatorial_projection:.6g} of magnitude {norm:.6g}); "
            f"its right ascension, and therefore the {element_name}, is undefined"
        )
    return _wrap_two_pi(math.atan2(float(vector[1]), float(vector[0])))
