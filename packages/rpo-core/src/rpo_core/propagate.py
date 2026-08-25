r"""Numerical propagation of absolute two-body motion.

The equations
-------------
Restricted two-body motion of a massless spacecraft about a point-mass central body:

.. math::

    \ddot{\mathbf{r}} = -\frac{\mu}{\lVert\mathbf{r}\rVert^{3}} \mathbf{r}

Written as a first-order system in the state :math:`[\mathbf{r}, \mathbf{v}]`, this is what
:func:`propagate_two_body` integrates. No J2, no drag, no third bodies, no relativity: this
is the model that the Clohessy-Wiltshire equations are a linearisation *of*, which is
exactly why it is the right oracle for measuring CW's error.

Two conserved quantities make this model self-checking, and the test suite uses both:

* **Specific orbital energy** :math:`\varepsilon = v^{2}/2 - \mu/r`
* **Specific angular momentum** :math:`\mathbf{h} = \mathbf{r} \times \mathbf{v}`

An integrator that drifts in either is not converged, whatever its reported tolerance says.

Integrator and tolerances
-------------------------
``scipy.integrate.solve_ivp`` with ``DOP853`` (8th-order explicit Runge-Kutta), default
``rtol = atol = 1e-12``. Absolute tolerance is in metres and metres per second, which for
LEO states of order 1e7 m means relative tolerance dominates. Both are arguments, not
constants: any result quoted from this module should survive a tolerance sweep first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from ._validate import as_state6, as_states_n6, validate_time_grid
from .constants import MU_EARTH_M3_S2
from .exceptions import PropagationError

DEFAULT_RTOL: float = 1.0e-12
DEFAULT_ATOL: float = 1.0e-12


def two_body_derivative(
    _t: float, state: npt.NDArray[np.float64], mu_m3_s2: float
) -> npt.NDArray[np.float64]:
    """Return ``d/dt [r, v]`` for restricted two-body motion.

    Signature matches what ``solve_ivp`` expects. ``_t`` is unused: the dynamics are
    autonomous.
    """
    r = state[:3]
    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise PropagationError("trajectory reached the central body singularity (|r| = 0)")
    return np.concatenate((state[3:], -mu_m3_s2 * r / r_norm**3))


def specific_energy_j_kg(state_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2) -> float:
    """Return specific orbital energy ``v**2/2 - mu/r``, J/kg. Conserved under two-body.

    Accepts a single state of shape (6,). For stacked states use
    :func:`specific_energy_batch_j_kg` -- passing an (N, 6) array here raises rather than
    silently collapsing it into one wrong number.
    """
    state = as_state6(state_eci, "state_eci", batch_hint=True)
    r_norm = float(np.linalg.norm(state[:3]))
    v_norm = float(np.linalg.norm(state[3:]))
    return 0.5 * v_norm**2 - mu_m3_s2 / r_norm


def specific_energy_batch_j_kg(
    states_eci: npt.ArrayLike, mu_m3_s2: float = MU_EARTH_M3_S2
) -> npt.NDArray[np.float64]:
    """Return specific orbital energy for each of N stacked states, shape (N,), J/kg."""
    states = as_states_n6(states_eci)
    r_norm = np.linalg.norm(states[:, :3], axis=1)
    v_norm = np.linalg.norm(states[:, 3:], axis=1)
    return np.asarray(0.5 * v_norm**2 - mu_m3_s2 / r_norm, dtype=np.float64)


def specific_angular_momentum(state_eci: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return specific angular momentum ``r x v``, m^2/s. Conserved under two-body.

    Accepts a single state of shape (6,); see
    :func:`specific_angular_momentum_batch` for stacked states.
    """
    state = as_state6(state_eci, "state_eci", batch_hint=True)
    return np.cross(state[:3], state[3:])


def specific_angular_momentum_batch(states_eci: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return specific angular momentum for each of N stacked states, shape (N, 3)."""
    states = as_states_n6(states_eci)
    return np.cross(states[:, :3], states[:, 3:])


def propagate_two_body(
    state0_eci: npt.ArrayLike,
    times_s: npt.ArrayLike,
    mu_m3_s2: float = MU_EARTH_M3_S2,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> npt.NDArray[np.float64]:
    """Propagate an inertial state over a schedule of output times.

    Parameters
    ----------
    state0_eci
        Initial state ``[r(3), v(3)]`` in metres and metres per second, shape (6,).
    times_s
        Output times, seconds, relative to the epoch of ``state0_eci``. Must be
        **strictly increasing** and begin at 0.0. A single-element schedule is allowed and
        returns the initial state unchanged.
    mu_m3_s2
        Gravitational parameter, m^3/s^2.
    rtol, atol
        Integrator tolerances. Exposed deliberately: quoting a result without a tolerance
        sweep behind it is how integrator settings get mistaken for physics.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times_s), 6)`` -- one inertial state per requested time.

    Raises
    ------
    PropagationError
        If the integrator fails to reach the final time. The failure is surfaced, never
        swallowed into a truncated trajectory.
    ValueError
        On malformed input, non-finite input, or a non-monotonic time schedule.

    """
    state0 = np.asarray(state0_eci, dtype=np.float64)
    if state0.shape != (6,):
        raise ValueError(f"state0_eci must have shape (6,), got {state0.shape}")
    if not np.all(np.isfinite(state0)):
        raise ValueError(f"state0_eci must be finite, got {state0!r}")

    times = validate_time_grid(times_s, "times_s", min_size=1)

    if times.size == 1:
        return state0.reshape(1, 6).copy()

    solution = solve_ivp(
        two_body_derivative,
        (0.0, float(times[-1])),
        state0,
        method="DOP853",
        t_eval=times,
        args=(mu_m3_s2,),
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise PropagationError(
            "two-body propagation failed at t = "
            f"{solution.t[-1] if solution.t.size else 0.0:.6g} s of "
            f"{float(times[-1]):.6g} s requested: {solution.message}"
        )
    if solution.y.shape[1] != times.size:
        raise PropagationError(
            f"integrator returned {solution.y.shape[1]} states for {times.size} requested "
            "times; the trajectory is incomplete"
        )
    return np.ascontiguousarray(solution.y.T)
