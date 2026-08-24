r"""Navigation error, linear covariance propagation, and guidance under estimation error.

Model M9 in ``docs/project1/math-model.md``. :mod:`rpo_core.montecarlo` owns *how randomness
is generated and accounted for*; this module owns *what the randomness means for the
navigation solution*, and it deliberately owns three things that are easy to conflate:

1. An estimation-error model in which the per-run **bias** and the per-estimate **white
   noise** are separate objects with separate covariances.
2. The linear covariance propagation :math:`P^+ = \Phi P \Phi^\mathsf{T} + Q`, with
   :math:`\Phi` the Clohessy-Wiltshire state transition matrix of
   :func:`rpo_core.relative.cw.cw_stm`.
3. A guidance helper whose signature makes it impossible to plan on truth by accident.

The equations
-------------
**Estimation error.** The estimated relative state is

.. math:: \hat{x} = x + b + w, \qquad
          b \sim \mathcal{N}(0, P_b)\ \text{drawn once per run}, \qquad
          w \sim \mathcal{N}(0, P_w)\ \text{drawn per estimate}.

Both terms are zero-mean and both have the same marginal shape, so a *single* estimate
cannot tell them apart: :math:`b + w \sim \mathcal{N}(0, P_b + P_w)` whatever the split.
They separate the moment an estimate is formed from more than one look. Averaging
:math:`m` independent estimates gives

.. math:: \operatorname{Cov}\!\left(\frac{1}{m}\sum_{k=1}^{m}(b + w_k)\right)
          = P_b + \frac{P_w}{m},

so the white part averages down as :math:`1/m` and the bias part does not average down at
all. A model that redraws the bias every step reports :math:`(P_b + P_w)/m`, which tends to
zero -- it predicts that navigation error can be filtered away, which is exactly the error a
dispersion study exists to avoid making. That is why :class:`NavigationSolution` holds the
bias as a frozen field drawn once by :meth:`NavigationErrorModel.begin_run`: redrawing it
per estimate is not an option the API offers.

**Linear covariance propagation.** For a linear flow :math:`x^+ = \Phi x` with additive
process noise of covariance :math:`Q`,

.. math:: P^+ = \Phi P \Phi^\mathsf{T} + Q .

Two structural properties follow and are used as tests rather than assumed. The CW plant
matrix has zero trace, so :math:`\det \Phi = \exp \int \operatorname{tr} A \, dt = 1`, hence
with :math:`Q = 0` the propagation **preserves the determinant** of the covariance -- a
conservation law that :math:`\Phi P \Phi` or :math:`\Phi^\mathsf{T} P \Phi` would not
satisfy in general. And :math:`\Phi(-\Delta t) = \Phi(\Delta t)^{-1}`, so a forward-backward
round trip returns :math:`P` exactly.

**Guidance under estimation error.** Plan the two-impulse transfer from the *estimate*, fly
it against *truth*, and the terminal error has a closed form. Writing
:math:`e = \hat{x} - x` for the estimation error at the planning epoch, the departure
impulse is :math:`\Delta v_1 = \Phi_{rv}^{-1}(r_f - \Phi_{rr}\hat{r}) - \hat{v}` and the
truth arrives at

.. math:: r(t_f) = r_f - (\Phi_{rr} e_r + \Phi_{rv} e_v), \qquad
          \dot{r}(t_f) = \dot{r}_f - (\Phi_{vr} e_r + \Phi_{vv} e_v),

that is, **the terminal state error is exactly** :math:`-\Phi(t_f)\, e`. Every term of the
plan cancels except the estimation error, propagated. Two consequences are worth stating
because they are what the tests check:

* With :math:`e = 0` the terminal state is the commanded one to machine precision. Zero
  dispersion reproduces the nominal plan exactly.
* The terminal-error covariance is :math:`\Phi P_e \Phi^\mathsf{T}` -- the same linear
  covariance propagation as above, which ties the guidance result to the covariance result
  rather than leaving them as two unrelated claims.

Both hold under CW dynamics with perfect execution. Under the nonlinear dynamics of
:mod:`rpo_core.relative.nonlinear` they hold to the linearisation error, which is measured,
not assumed.

Validity
--------
* The closed forms above are exact for a **linear** truth flow. :func:`plan_from_estimate`
  takes the truth propagator as an argument precisely so that the linear case can be used as
  an oracle and the nonlinear case as the deliverable; nothing here assumes they agree.
* Guidance is **open loop after the planning epoch**: the arrival impulse is the one
  commanded at departure, not a re-planned one. That is the conservative reading of a
  two-impulse plan and it is what the SRS baseline describes. A mid-course re-plan would
  reduce the terminal error and would need its own model of how often the navigation
  solution is refreshed.
* The linear covariance propagation is exact for CW and is an approximation for the
  nonlinear relative dynamics, degrading with separation exactly as the CW model itself
  does (``docs/cw_validity.md``). It is not a substitute for an unscented or Monte Carlo
  propagation at large dispersion; the small-dispersion agreement is measured in
  ``test_navigation.py`` and the large-dispersion *disagreement* is measured next to it.
* Covariances are required strictly positive definite, matching
  :class:`rpo_core.montecarlo.VectorNormalDispersion`. A zero-variance component is
  expressed by omitting the term (``None``), not by a singular matrix.
* Process noise :math:`Q` is permitted to be positive *semi*-definite, because
  :math:`Q = 0` (no process noise) is the ordinary case and a legitimate one.
* **Chaining is bounded by conditioning, and the bound is not large.** The CW along-track
  secular drift stretches the covariance faster than any other direction, so the condition
  number of a chained :math:`\Phi P \Phi^\mathsf{T}` grows quadratically in elapsed time.
  Measured, starting from ``diag(25, 25, 25, 1e-4, 1e-4, 1e-4)`` and stepping a half period
  at a time: ``cond(P) = 7.1e15`` after 20 steps, ``1.1e17`` after 40, and at **43 steps**
  round-off drives the smallest eigenvalue negative (``-1.8e-10`` against a largest of
  ``2.8e+07``) and :func:`validate_covariance` refuses the next step. That refusal is
  correct -- the matrix genuinely has no Cholesky factor by then, so anything sampled from
  it would be fiction -- but it means a *long-horizon* covariance study needs a square-root
  (UD or Cholesky-factor) formulation rather than this one. Twenty-odd orbits is the
  practical ceiling here, which is far beyond any proximity-operations manoeuvre and far
  short of a mission-lifetime uncertainty budget. See
  ``test_a_long_chain_eventually_loses_positive_definiteness``.

Units are SI: metres, seconds, radians. Covariances carry squared units, so a relative-state
covariance mixes m^2, m^2/s and m^2/s^2 blocks; do not take a norm of one and expect it to
mean anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .exceptions import RpoCoreError
from .montecarlo import VectorNormalDispersion
from .relative.cw import DEFAULT_FEASIBILITY_TOL_M, cw_stm, propagate_cw, two_impulse_transfer

__all__ = [
    "DEFAULT_SYMMETRY_RTOL",
    "STATE_DIMENSION",
    "CovarianceDefinitionError",
    "GuidanceDefinitionError",
    "GuidedTransfer",
    "NavigationErrorModel",
    "NavigationModelError",
    "NavigationSolution",
    "cw_truth_propagator",
    "plan_from_estimate",
    "propagate_covariance",
    "terminal_error_covariance",
    "validate_covariance",
]


#: Dimension of a Hill-frame relative state ``[x, y, z, xdot, ydot, zdot]``.
STATE_DIMENSION: int = 6

#: Relative tolerance on ``max |C - C.T|`` before a matrix is rejected as non-symmetric.
#:
#: Not zero, unlike :class:`rpo_core.montecarlo.VectorNormalDispersion`, and the difference
#: is deliberate. That class validates a covariance a *human* assembled, where any asymmetry
#: is an assembly error. This module validates covariances that come back out of
#: :math:`\Phi P \Phi^\mathsf{T}`, which is symmetric in exact arithmetic and asymmetric at
#: the rounding level in floating point. **Measured** over 200 chained half-period
#: propagations of a representative relative-state covariance
#: (``diag(25, 25, 25, 1e-4, 1e-4, 1e-4)``): the bare product
#: :math:`\Phi P \Phi^\mathsf{T}` reaches ``max |C - C.T| / max |C| = 1.045e-15``, while
#: re-symmetrising each step -- which :func:`propagate_covariance` does -- holds it at
#: **exactly 0.0**. 1e-10 therefore accepts the bare product with ~1e5 headroom over its
#: measured worst case, while still rejecting a matrix whose off-diagonal entries were typed
#: in inconsistently, which is wrong in the third significant figure, not the fifteenth. See
#: ``test_chained_propagation_stays_symmetric``.
DEFAULT_SYMMETRY_RTOL: float = 1.0e-10


class NavigationModelError(RpoCoreError):
    """Base class for every error raised by :mod:`rpo_core.navigation`."""


class CovarianceDefinitionError(NavigationModelError, ValueError):
    """Raised when a covariance matrix cannot describe a realisable Gaussian.

    Wrong shape, a non-finite entry, asymmetry beyond :data:`DEFAULT_SYMMETRY_RTOL`, or a
    non-positive eigenvalue. The message carries the offending number -- the asymmetry, or
    the smallest eigenvalue and the largest for scale -- because "not positive definite" on
    its own does not say which direction went negative or by how much.
    """


class GuidanceDefinitionError(NavigationModelError, ValueError):
    """Raised when a guidance request is malformed before any dynamics are evaluated.

    A time grid that does not start at zero, fewer than two output times, a non-finite
    state, a truth propagator that returned the wrong shape. Failures *inside* the
    numerics -- a singular transfer time, an integrator that gives up -- keep their own
    typed exceptions and travel out of :func:`plan_from_estimate` untouched, because the
    condition number or residual they carry is the diagnosis.
    """


# --------------------------------------------------------------------------------------
# Covariance validation and propagation
# --------------------------------------------------------------------------------------


def validate_covariance(
    covariance: npt.ArrayLike,
    *,
    name: str = "covariance",
    dimension: int | None = None,
    require_positive_definite: bool = True,
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
) -> npt.NDArray[np.float64]:
    """Return ``covariance`` as a validated, exactly symmetric float array.

    The returned matrix is ``(C + C.T) / 2``. Symmetrising *after* the asymmetry has been
    measured and accepted is the honest order: symmetrising first would make the check
    unfalsifiable, and not symmetrising at all lets round-off accumulate across a chain of
    propagations until an exact-symmetry consumer rejects it.

    Parameters
    ----------
    covariance
        Square matrix, shape ``(k, k)``.
    name
        Name used in error messages, so a caller with several covariances can tell which
        one was rejected.
    dimension
        Required ``k``, or ``None`` to accept any square matrix.
    require_positive_definite
        Reject a non-positive smallest eigenvalue. Set ``False`` for a process-noise matrix,
        where positive *semi*-definiteness is the correct requirement and ``Q = 0`` is the
        ordinary case.
    symmetry_rtol
        Relative tolerance on ``max |C - C.T|``, scaled by ``max(1, max |C|)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(k, k)``, exactly symmetric.

    Raises
    ------
    CovarianceDefinitionError
        If the matrix is not square, has the wrong dimension, contains a non-finite entry,
        is asymmetric beyond ``symmetry_rtol``, or violates the requested definiteness.

    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise CovarianceDefinitionError(
            f"{name} must be a square 2-D matrix, got shape {matrix.shape}"
        )
    if dimension is not None and matrix.shape[0] != dimension:
        raise CovarianceDefinitionError(
            f"{name} must have shape ({dimension}, {dimension}) to describe a "
            f"{dimension}-element state, got {matrix.shape}"
        )
    if matrix.size == 0:
        raise CovarianceDefinitionError(f"{name} must be non-empty, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise CovarianceDefinitionError(
            f"{name} must be finite; a non-finite entry propagates through every eigenvalue, "
            f"Cholesky factor and sample drawn from it. Got {matrix!r}"
        )

    scale = max(1.0, float(np.max(np.abs(matrix))))
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > symmetry_rtol * scale:
        raise CovarianceDefinitionError(
            f"{name} is not symmetric: max |C - C.T| = {asymmetry:.6e}, which exceeds "
            f"symmetry_rtol={symmetry_rtol:.3e} times the matrix scale {scale:.6e} "
            f"(= {symmetry_rtol * scale:.6e}). A covariance is symmetric by definition, so an "
            "asymmetry this large means the entries were assembled wrongly; symmetrising it "
            "silently would hide that."
        )

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    if require_positive_definite and smallest <= 0.0:
        raise CovarianceDefinitionError(
            f"{name} is not positive definite: smallest eigenvalue {smallest:.6e} "
            f"(largest {largest:.6e}). A non-positive eigenvalue means a direction with zero "
            "or negative variance, for which no Cholesky factor and no realisable Gaussian "
            "exists. Express an exactly-certain component by omitting the term, not by a "
            "singular matrix."
        )
    if not require_positive_definite and smallest < -symmetry_rtol * scale:
        raise CovarianceDefinitionError(
            f"{name} is not positive semi-definite: smallest eigenvalue {smallest:.6e} "
            f"(largest {largest:.6e}), below the round-off floor "
            f"{-symmetry_rtol * scale:.6e}. A negative-variance direction in a process-noise "
            "matrix would shrink the propagated covariance, reporting more confidence after "
            "propagating than before."
        )
    return symmetric


def propagate_covariance(
    covariance: npt.ArrayLike,
    n_rad_s: float,
    dt_s: float,
    *,
    process_noise: npt.ArrayLike | None = None,
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
) -> npt.NDArray[np.float64]:
    r"""Propagate a relative-state covariance one step: ``P⁺ = Phi P Phi.T + Q``.

    :math:`\Phi` is the closed-form Clohessy-Wiltshire state transition matrix of
    :func:`rpo_core.relative.cw.cw_stm`, so this is the linear covariance propagation of
    model M9 and nothing more: it says how an *initial* uncertainty is stretched by the
    relative dynamics, and it knows nothing about measurements.

    Parameters
    ----------
    covariance
        Shape (6, 6) symmetric positive-definite covariance of
        ``[x, y, z, xdot, ydot, zdot]``, in m^2 / (m^2/s) / (m^2/s^2) blocks.
    n_rad_s
        Target mean motion, rad/s. Strictly positive.
    dt_s
        Step, seconds. May be negative: the STM is defined backwards in time, and
        ``propagate_covariance(propagate_covariance(P, n, dt), n, -dt)`` returns ``P``.
    process_noise
        Shape (6, 6) symmetric positive *semi*-definite ``Q``, or ``None`` for no process
        noise. ``None`` and a zero matrix are numerically identical; ``None`` is cheaper and
        says so.
    symmetry_rtol
        Relative symmetry tolerance passed to :func:`validate_covariance`.

    Returns
    -------
    numpy.ndarray
        Shape (6, 6), exactly symmetric.

    Raises
    ------
    CovarianceDefinitionError
        If ``covariance`` is not a 6x6 symmetric positive-definite matrix, or
        ``process_noise`` is not 6x6 symmetric positive semi-definite.
    ValueError
        From :func:`~rpo_core.relative.cw.cw_stm` if ``n_rad_s`` is not a finite positive
        mean motion or ``dt_s`` is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from rpo_core.constants import mean_motion_rad_s
    >>> n = mean_motion_rad_s(6378137.0 + 420e3)
    >>> p = np.diag([4.0, 4.0, 4.0, 1e-4, 1e-4, 1e-4])
    >>> propagated = propagate_covariance(p, n, 600.0)
    >>> bool(np.allclose(np.linalg.det(propagated), np.linalg.det(p), rtol=1e-9))
    True

    """
    p = validate_covariance(
        covariance,
        name="covariance",
        dimension=STATE_DIMENSION,
        symmetry_rtol=symmetry_rtol,
    )
    phi = cw_stm(n_rad_s, dt_s)
    propagated = phi @ p @ phi.T
    if process_noise is not None:
        propagated = propagated + validate_covariance(
            process_noise,
            name="process_noise",
            dimension=STATE_DIMENSION,
            require_positive_definite=False,
            symmetry_rtol=symmetry_rtol,
        )
    # Re-symmetrise: the product is symmetric in exact arithmetic, and leaving the
    # rounding-level asymmetry in place lets it compound over a chain of steps.
    result: npt.NDArray[np.float64] = 0.5 * (propagated + propagated.T)
    return result


# --------------------------------------------------------------------------------------
# Estimation error
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NavigationSolution:
    """One run's navigation filter: a bias fixed for the run, noise redrawn per estimate.

    Obtained from :meth:`NavigationErrorModel.begin_run` or
    :meth:`NavigationErrorModel.with_bias`, never constructed with a bias that changes.
    ``bias_hill`` is a frozen field of a frozen dataclass precisely so that "redraw the bias
    each step" is not something a caller can do by accident: to get a different bias you
    have to start a different run, which is what a different bias means.

    Attributes
    ----------
    model
        The model this solution was drawn from.
    bias_hill
        Shape (6,) constant estimation bias for this run, in m and m/s. Zero when the model
        declares no bias term.

    """

    model: NavigationErrorModel
    bias_hill: npt.NDArray[np.float64]

    def estimate(
        self, truth_state_hill: npt.ArrayLike, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Return the estimated relative state for one look at the truth.

        Parameters
        ----------
        truth_state_hill
            Shape (6,) true Hill-frame relative state, m and m/s.
        rng
            Explicit generator for the white-noise draw (``docs/conventions.md``: no global
            RNG). Not consumed at all when the model declares no noise term, so a
            bias-only model is bitwise reproducible without it.

        Returns
        -------
        numpy.ndarray
            Shape (6,), ``truth + bias + noise``.

        Raises
        ------
        GuidanceDefinitionError
            If ``truth_state_hill`` is not a finite 6-vector.

        """
        truth = _validate_state(truth_state_hill, "truth_state_hill")
        estimate = truth + self.bias_hill
        noise = self.model.draw_noise(rng)
        if noise is not None:
            estimate = estimate + noise
        return estimate


@dataclass(frozen=True)
class NavigationErrorModel:
    """Estimation-error model: a per-run bias plus per-estimate white noise.

    Both terms are optional and both default to absent, so the zero-navigation-error case
    -- the limiting case every dispersion study needs as its control -- is
    ``NavigationErrorModel()`` rather than a hand-built matrix of zeros (which would not be
    positive definite and would be rejected).

    Sampling is delegated to :class:`rpo_core.montecarlo.VectorNormalDispersion`, so the
    Cholesky factorisation, the validation, and the number of standard normals consumed per
    draw are the ones the Monte Carlo harness already documents and tests. This class adds
    only the bias/noise distinction.

    Attributes
    ----------
    noise_covariance
        Shape (6, 6) SPD covariance of the per-estimate white noise, or ``None``.
    bias_covariance
        Shape (6, 6) SPD covariance from which the per-run constant bias is drawn, or
        ``None``.

    Raises
    ------
    CovarianceDefinitionError
        If either covariance is given and is not a 6x6 symmetric positive-definite matrix.

    """

    noise_covariance: npt.NDArray[np.float64] | None = None
    bias_covariance: npt.NDArray[np.float64] | None = None
    _noise: VectorNormalDispersion | None = field(init=False, repr=False, compare=False)
    _bias: VectorNormalDispersion | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate both covariances and build their samplers."""
        object.__setattr__(self, "_noise", self._build("noise_covariance"))
        object.__setattr__(self, "_bias", self._build("bias_covariance"))

    def _build(self, attribute: str) -> VectorNormalDispersion | None:
        """Validate one covariance attribute in place and return its sampler."""
        value = getattr(self, attribute)
        if value is None:
            return None
        matrix = validate_covariance(value, name=attribute, dimension=STATE_DIMENSION)
        object.__setattr__(self, attribute, matrix)
        # The matrix is exactly symmetric coming out of validate_covariance, which is what
        # VectorNormalDispersion's stricter zero-tolerance symmetry check requires.
        return VectorNormalDispersion(mean=np.zeros(STATE_DIMENSION), covariance=matrix)

    @property
    def total_covariance(self) -> npt.NDArray[np.float64]:
        """Return ``P_bias + P_noise``, the covariance of a *single* estimate's error.

        This is the quantity a single look measures, and it is the reason bias and noise
        cannot be separated from one estimate: every split with the same sum produces the
        same distribution here. They separate only when several estimates are combined.
        """
        total = np.zeros((STATE_DIMENSION, STATE_DIMENSION), dtype=np.float64)
        if self.noise_covariance is not None:
            total = total + self.noise_covariance
        if self.bias_covariance is not None:
            total = total + self.bias_covariance
        return total

    def averaged_covariance(self, n_estimates: int) -> npt.NDArray[np.float64]:
        """Return ``P_bias + P_noise / m``, the error covariance of an ``m``-look average.

        The closed form the bias-versus-white-noise test is judged against: the white part
        averages down, the bias part does not.

        Parameters
        ----------
        n_estimates
            Number ``m`` of independent estimates averaged. Strictly positive.

        Returns
        -------
        numpy.ndarray
            Shape (6, 6).

        Raises
        ------
        CovarianceDefinitionError
            If ``n_estimates`` is not a positive integer.

        """
        if isinstance(n_estimates, bool) or not isinstance(n_estimates, int) or n_estimates < 1:
            raise CovarianceDefinitionError(
                f"n_estimates must be a positive int number of averaged looks, got {n_estimates!r}"
            )
        total = np.zeros((STATE_DIMENSION, STATE_DIMENSION), dtype=np.float64)
        if self.bias_covariance is not None:
            total = total + self.bias_covariance
        if self.noise_covariance is not None:
            total = total + self.noise_covariance / float(n_estimates)
        return total

    def bias_dispersion(self) -> VectorNormalDispersion | None:
        """Return the bias as a :class:`~rpo_core.montecarlo.Dispersion`, or ``None``.

        Handing the bias to :func:`rpo_core.montecarlo.run_campaign` as a declared
        dispersion is what makes "once per run" structural in a campaign: the harness draws
        every dispersion exactly once per run from that run's own substream. The white noise
        is *not* a dispersion, because it is drawn many times per run; it comes from the
        run's generator instead.
        """
        return self._bias

    def draw_noise(self, rng: np.random.Generator) -> npt.NDArray[np.float64] | None:
        """Return one white-noise draw, shape (6,), or ``None`` if the model has no noise."""
        if self._noise is None:
            return None
        return self._noise.sample(rng)

    def begin_run(self, rng: np.random.Generator) -> NavigationSolution:
        """Draw this run's constant bias and return the resulting per-run solution.

        Call **once per run**. Calling it per estimate converts the bias into white noise of
        the same marginal covariance, which is the modelling error this class exists to make
        difficult; see the module docstring.

        Parameters
        ----------
        rng
            Explicit generator for the bias draw.

        Returns
        -------
        NavigationSolution

        """
        if self._bias is None:
            return NavigationSolution(self, np.zeros(STATE_DIMENSION, dtype=np.float64))
        return NavigationSolution(self, self._bias.sample(rng))

    def with_bias(self, bias_hill: npt.ArrayLike) -> NavigationSolution:
        """Return a per-run solution using an externally drawn bias.

        Used when the bias is supplied by a Monte Carlo campaign's dispersion machinery
        (:meth:`bias_dispersion`) rather than drawn here, so that the campaign's substream
        scheme -- not this class -- owns reproducibility.

        Parameters
        ----------
        bias_hill
            Shape (6,) bias, m and m/s.

        Returns
        -------
        NavigationSolution

        Raises
        ------
        GuidanceDefinitionError
            If ``bias_hill`` is not a finite 6-vector.

        """
        return NavigationSolution(self, _validate_state(bias_hill, "bias_hill"))

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, mirroring ``Dispersion.describe``."""
        return {
            "kind": "navigation_error",
            "noise_covariance": (
                None
                if self.noise_covariance is None
                else [[float(v) for v in row] for row in self.noise_covariance]
            ),
            "bias_covariance": (
                None
                if self.bias_covariance is None
                else [[float(v) for v in row] for row in self.bias_covariance]
            ),
        }


# --------------------------------------------------------------------------------------
# Guidance under estimation error
# --------------------------------------------------------------------------------------


def _validate_state(state: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``state`` as a validated finite 6-vector."""
    array = np.asarray(state, dtype=np.float64)
    if array.shape != (STATE_DIMENSION,):
        raise GuidanceDefinitionError(
            f"{name} must have shape ({STATE_DIMENSION},) as "
            f"[x, y, z, xdot, ydot, zdot], got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise GuidanceDefinitionError(f"{name} must be finite, got {array!r}")
    return array


def _validate_times(times_s: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``times_s`` as a validated grid starting at zero and strictly increasing."""
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise GuidanceDefinitionError(
            f"times_s must be a 1-D grid of at least two output times, got shape {times.shape}"
        )
    if not np.all(np.isfinite(times)):
        raise GuidanceDefinitionError(f"times_s must be finite, got {times!r}")
    if times[0] != 0.0:
        raise GuidanceDefinitionError(
            f"times_s must start at 0.0 s, the epoch of the departure impulse, got "
            f"{float(times[0])!r}. A grid starting elsewhere would silently shift the burn."
        )
    steps = np.diff(times)
    bad = np.flatnonzero(steps <= 0.0)
    if bad.size > 0:
        index = int(bad[0])
        raise GuidanceDefinitionError(
            f"times_s must be strictly increasing: times_s[{index}]={float(times[index])!r} is "
            f"not below times_s[{index + 1}]={float(times[index + 1])!r}"
        )
    return times


def cw_truth_propagator(
    n_rad_s: float,
) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Return a truth propagator that flies the **linear** CW dynamics.

    The oracle case for :func:`plan_from_estimate`: with this propagator the terminal error
    equals ``-Phi @ e`` exactly, so a test can assert a closed form rather than a tolerance.
    Pass a nonlinear propagator instead when the deliverable is a trajectory rather than a
    proof.

    Parameters
    ----------
    n_rad_s
        Target mean motion, rad/s.

    Returns
    -------
    callable
        ``(initial_state_hill, times_s) -> states_hill`` of shape ``(len(times_s), 6)``.

    """

    def _propagate(
        initial_state_hill: npt.NDArray[np.float64], times_s: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Propagate one relative state onto the requested grid under CW."""
        return np.array(
            [propagate_cw(n_rad_s, initial_state_hill, float(t)) for t in times_s],
            dtype=np.float64,
        )

    return _propagate


@dataclass(frozen=True, eq=False)
class GuidedTransfer:
    """One manoeuvre planned from an estimate and flown against truth.

    ``eq=False`` because several fields are numpy arrays: a generated ``__eq__`` would
    return an array from ``==`` and raise on the truth test, a confusing failure a long way
    from its cause.

    Attributes
    ----------
    times_s
        Shape (N,), seconds from the departure impulse, as supplied.
    states_hill
        Shape (N, 6) **truth** Hill-frame states. The final sample carries the arrival
        impulse, so ``terminal_velocity_error_m_s`` is a terminal error and not the pre-burn
        coast velocity.
    estimated_initial_state_hill, truth_initial_state_hill
        The two states the plan and the flight were built from. Kept side by side because
        the whole point of this function is that they are different.
    dv1_commanded_hill_m_s, dv2_commanded_hill_m_s
        Impulses the guidance asked for, m/s. Functions of the *estimate*.
    dv1_executed_hill_m_s, dv2_executed_hill_m_s
        Impulses actually applied to truth, m/s. Equal to the commanded pair when no
        execution-error model was supplied.
    terminal_position_error_m, terminal_velocity_error_m_s
        Norms of the difference between the achieved terminal state and the commanded one.
    total_delta_v_m_s
        ``|dv1_executed| + |dv2_executed|``: what the propellant budget actually paid, not
        what the plan asked for.

    """

    times_s: npt.NDArray[np.float64]
    states_hill: npt.NDArray[np.float64]
    estimated_initial_state_hill: npt.NDArray[np.float64]
    truth_initial_state_hill: npt.NDArray[np.float64]
    dv1_commanded_hill_m_s: npt.NDArray[np.float64]
    dv2_commanded_hill_m_s: npt.NDArray[np.float64]
    dv1_executed_hill_m_s: npt.NDArray[np.float64]
    dv2_executed_hill_m_s: npt.NDArray[np.float64]
    terminal_position_error_m: float
    terminal_velocity_error_m_s: float
    total_delta_v_m_s: float

    @property
    def terminal_state_hill(self) -> npt.NDArray[np.float64]:
        """Return the achieved terminal state, arrival impulse included."""
        terminal: npt.NDArray[np.float64] = self.states_hill[-1]
        return terminal

    @property
    def estimation_error_hill(self) -> npt.NDArray[np.float64]:
        """Return ``estimate - truth`` at the planning epoch, m and m/s."""
        error: npt.NDArray[np.float64] = (
            self.estimated_initial_state_hill - self.truth_initial_state_hill
        )
        return error


def plan_from_estimate(
    n_rad_s: float,
    times_s: npt.ArrayLike,
    *,
    estimated_state_hill: npt.ArrayLike,
    truth_state_hill: npt.ArrayLike,
    commanded_terminal_state_hill: npt.ArrayLike,
    propagate_fn: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    execute_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
    cross_track_feasibility_tol_m: float = DEFAULT_FEASIBILITY_TOL_M,
) -> GuidedTransfer:
    r"""Plan a two-impulse transfer from the **estimate** and fly it against **truth**.

    This is where navigation error costs propellant and miss distance, and the signature is
    built so that the two states cannot be transposed by accident: both are keyword-only and
    both are named for what they are. Planning on truth -- the classic way to make a
    dispersion study report a reassuring answer -- requires deliberately passing the same
    array twice.

    The sequence is: solve
    :func:`~rpo_core.relative.cw.two_impulse_transfer` from ``estimated_state_hill`` to
    ``commanded_terminal_state_hill``; pass both impulses through ``execute_fn`` to apply
    burn execution error; add the executed departure impulse to the **truth** velocity;
    propagate truth over ``times_s`` with ``propagate_fn``; add the executed arrival impulse
    to the final sample.

    With a linear (CW) ``propagate_fn`` and no execution error the achieved terminal state
    error is exactly :math:`-\Phi(t_f)\,e`, with :math:`e` the estimation error; see the
    module docstring.

    Parameters
    ----------
    n_rad_s
        Target mean motion, rad/s.
    times_s
        Shape (N,) output grid, seconds, starting at 0.0 and strictly increasing. The last
        entry is the time of flight.
    estimated_state_hill
        Shape (6,) relative state **the guidance believes it has**. The plan is a function
        of this and of nothing else.
    truth_state_hill
        Shape (6,) relative state that is actually flown.
    commanded_terminal_state_hill
        Shape (6,) terminal state the plan targets, m and m/s.
    propagate_fn
        ``(initial_state_hill, times_s) -> (N, 6)`` truth dynamics. Use
        :func:`cw_truth_propagator` for the linear oracle, or a closure over
        :func:`rpo_core.relative.nonlinear.propagate_relative_nonlinear` for the real thing.
    execute_fn
        ``dv_commanded -> dv_executed``, m/s. Typically
        :meth:`rpo_core.montecarlo.BurnExecutionSample.apply`. ``None`` means perfect
        execution.
    cross_track_feasibility_tol_m
        Passed through to :func:`~rpo_core.relative.cw.two_impulse_transfer`. Raised above
        its default when the transfer time is a half-period multiple and the *dispersed*
        cross-track offset is therefore structurally uncorrectable; see the
        ``rpo_traj.campaign`` module docstring, which is where that trap actually bites.

    Returns
    -------
    GuidedTransfer

    Raises
    ------
    GuidanceDefinitionError
        If a state is not a finite 6-vector, ``times_s`` is malformed, or ``propagate_fn``
        returned something other than an ``(N, 6)`` finite array.
    SingularTransferTimeError, InfeasibleTransferError
        From the underlying CW solve, unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from rpo_core.constants import mean_motion_rad_s, orbital_period_s
    >>> a = 6378137.0 + 420e3
    >>> n = mean_motion_rad_s(a)
    >>> times = np.linspace(0.0, 0.3 * orbital_period_s(a), 5)
    >>> truth = np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    >>> result = plan_from_estimate(
    ...     n, times,
    ...     estimated_state_hill=truth,
    ...     truth_state_hill=truth,
    ...     commanded_terminal_state_hill=np.array([0.0, -250.0, 0.0, 0.0, 0.0, 0.0]),
    ...     propagate_fn=cw_truth_propagator(n))
    >>> bool(result.terminal_position_error_m < 1e-9)
    True

    """
    times = _validate_times(times_s)
    estimate = _validate_state(estimated_state_hill, "estimated_state_hill")
    truth = _validate_state(truth_state_hill, "truth_state_hill")
    commanded = _validate_state(commanded_terminal_state_hill, "commanded_terminal_state_hill")
    tof_s = float(times[-1])

    dv1_commanded, dv2_commanded = two_impulse_transfer(
        n_rad_s,
        estimate[:3],
        estimate[3:],
        commanded[:3],
        commanded[3:],
        tof_s,
        feasibility_tol_m=cross_track_feasibility_tol_m,
    )

    dv1_executed = dv1_commanded if execute_fn is None else np.asarray(execute_fn(dv1_commanded))
    dv2_executed = dv2_commanded if execute_fn is None else np.asarray(execute_fn(dv2_commanded))
    dv1_executed = _validate_impulse(dv1_executed, "dv1_executed")
    dv2_executed = _validate_impulse(dv2_executed, "dv2_executed")

    # The departure impulse is applied to TRUTH, not to the estimate. This single line is
    # the difference between a dispersion study and a self-congratulatory one.
    departure_state = np.concatenate((truth[:3], truth[3:] + dv1_executed))
    states = np.asarray(propagate_fn(departure_state, times), dtype=np.float64)
    if states.shape != (times.size, STATE_DIMENSION):
        raise GuidanceDefinitionError(
            f"propagate_fn must return an array of shape ({times.size}, {STATE_DIMENSION}) "
            f"for {times.size} output times, got {states.shape}"
        )
    if not np.all(np.isfinite(states)):
        raise GuidanceDefinitionError(
            "propagate_fn returned a non-finite state; a trajectory containing NaN would "
            "flow into every constraint check and percentile as a silently-passing value"
        )

    states = states.copy()
    states[-1, 3:] += dv2_executed

    terminal_error = states[-1] - commanded
    return GuidedTransfer(
        times_s=times,
        states_hill=states,
        estimated_initial_state_hill=estimate,
        truth_initial_state_hill=truth,
        dv1_commanded_hill_m_s=dv1_commanded,
        dv2_commanded_hill_m_s=dv2_commanded,
        dv1_executed_hill_m_s=dv1_executed,
        dv2_executed_hill_m_s=dv2_executed,
        terminal_position_error_m=float(np.linalg.norm(terminal_error[:3])),
        terminal_velocity_error_m_s=float(np.linalg.norm(terminal_error[3:])),
        total_delta_v_m_s=float(np.linalg.norm(dv1_executed) + np.linalg.norm(dv2_executed)),
    )


def _validate_impulse(impulse: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``impulse`` as a validated finite 3-vector."""
    array = np.asarray(impulse, dtype=np.float64)
    if array.shape != (3,):
        raise GuidanceDefinitionError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GuidanceDefinitionError(f"{name} must be finite, got {array!r}")
    return array


def terminal_error_covariance(
    covariance: npt.ArrayLike,
    n_rad_s: float,
    tof_s: float,
    *,
    symmetry_rtol: float = DEFAULT_SYMMETRY_RTOL,
) -> npt.NDArray[np.float64]:
    r"""Return the terminal-state error covariance ``Phi P Phi.T`` for guided flight.

    A named alias for :func:`propagate_covariance` with no process noise, provided because
    the *meaning* is different and worth being able to read off a call site: here ``P`` is
    the covariance of the **estimation error at the planning epoch**, and the result is the
    covariance of the **terminal miss**, by the closed form
    :math:`\delta x(t_f) = -\Phi(t_f) e` derived in the module docstring. The sign does not
    survive into a covariance, which is exactly why the closed form is worth stating
    separately from the second moment.

    Parameters
    ----------
    covariance
        Shape (6, 6) SPD estimation-error covariance at the planning epoch.
    n_rad_s
        Target mean motion, rad/s.
    tof_s
        Time of flight, seconds.
    symmetry_rtol
        Relative symmetry tolerance passed to :func:`validate_covariance`.

    Returns
    -------
    numpy.ndarray
        Shape (6, 6) covariance of the terminal state error.

    Raises
    ------
    CovarianceDefinitionError
        If ``covariance`` is not 6x6 symmetric positive definite.

    """
    return propagate_covariance(covariance, n_rad_s, tof_s, symmetry_rtol=symmetry_rtol)
