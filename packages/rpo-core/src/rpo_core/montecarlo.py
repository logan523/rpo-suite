r"""Monte Carlo dispersion campaigns: seeded substreams, honest failure accounting.

Model M9 in ``docs/project1/math-model.md``. This module is the *harness*, not a physics
model: it decides how randomness is generated, how a campaign survives a run that raises,
and how a batch of runs becomes a defensible number. Projects 2 and 3 consume it, so the
interface is the deliverable.

Four concerns are kept apart on purpose, following the decomposition used by Basilisk's
Monte Carlo controller:

1. **Nominal configuration** -- the undispersed scenario. Opaque to this module; it is
   whatever ``run_fn`` understands.
2. **Dispersion definitions** -- declarative, frozen, JSON-serialisable descriptions of
   what varies and how (:class:`Dispersion` and its subclasses).
3. **Per-run execution** -- :func:`execute_run`, a pure function of
   ``(nominal, dispersions, seed, index)``. It does not know a campaign exists.
4. **Retention and analysis** -- named scalar extractors (``retain``) and the summary
   statistics in :class:`CampaignSummary`.

The equations
-------------
**Correlated Gaussian sampling.** For covariance :math:`C = L L^\mathsf{T}` (lower
Cholesky) and :math:`z \sim \mathcal{N}(0, I)`,

.. math:: x = \mu + L z, \qquad \operatorname{Cov}(x) = L \, \mathbb{E}[z z^\mathsf{T}] \,
          L^\mathsf{T} = C .

The factorisation exists iff :math:`C` is symmetric positive definite, which is why a
non-positive-definite covariance is rejected at construction rather than at sample time.

**Burn execution error** (:class:`MagnitudePointingDispersion`). A commanded impulse
:math:`\Delta v` is executed as

.. math:: \Delta v' = (1 + \delta)\, R(\hat{n}(\psi), \theta)\, \Delta v, \qquad
          \delta \sim \mathcal{N}(0, \sigma_\mathrm{mag}^2), \quad
          \theta \sim \mathcal{N}(0, \sigma_\mathrm{point}^2), \quad
          \psi \sim \mathcal{U}[0, 2\pi),

where :math:`\hat{n}(\psi)` is a unit vector in the plane perpendicular to
:math:`\Delta v`, at azimuth :math:`\psi`, and :math:`R` is the Rodrigues rotation

.. math:: R(\hat{n}, \theta) v = v \cos\theta + (\hat{n} \times v) \sin\theta +
          \hat{n} (\hat{n} \cdot v)(1 - \cos\theta).

Because :math:`R` is orthogonal, this decomposition is *exact*:

.. math:: |\Delta v'| = |1 + \delta| \, |\Delta v|, \qquad
          \angle(\Delta v', \Delta v) = |\theta| .

Magnitude error and pointing error are therefore statistically independent by
construction, and each parameter means exactly what its name says.

**Why not additive component-wise noise.** The common shortcut
:math:`\Delta v' = \Delta v + \varepsilon`, :math:`\varepsilon \sim \mathcal{N}(0,
\sigma^2 I)`, is physically wrong for a burn. It couples the two error sources: the
induced magnitude error is :math:`\approx \varepsilon \cdot \hat{v}`, so magnitude and
direction errors are driven by the same draw and are not independent. Worse, the induced
*relative* magnitude error is :math:`\sigma / |\Delta v|` and the induced pointing error
is :math:`\sigma / |\Delta v|` radians, so both scale as the inverse of the burn size: a
0.01 m/s trim burn acquires a 100 % magnitude error and a radian of pointing error from
the same :math:`\sigma` that is realistic for a 1 m/s burn. Real thrusters do not behave
that way -- a scale-factor error and a misalignment are properties of the *thruster*, not
of the commanded impulse -- so a campaign built on additive noise reports a false
sensitivity to small burns. See ``test_component_wise_noise_couples_magnitude_and_pointing``
for the complement test that shows the two models are measurably different.

**Wilson score interval.** For :math:`k` successes in :math:`n` trials, the Wilson
interval is the set of :math:`p` satisfying :math:`|k/n - p| \le z\sqrt{p(1-p)/n}`,
i.e. the roots of a quadratic in :math:`p`:

.. math:: p_{\pm} = \frac{1}{n + z^2}\left[ k + \frac{z^2}{2} \pm
          z\sqrt{\frac{k(n-k)}{n} + \frac{z^2}{4}} \right].

The normal ("Wald") approximation :math:`\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}` is used
instead of this almost everywhere and is badly wrong exactly where safety-critical rates
live: at :math:`k = 0` or :math:`k = n` it returns a zero-width interval, and near those
ends it returns bounds outside :math:`[0, 1]`. The Wilson interval is bounded in
:math:`[0, 1]` for every :math:`k`, degenerates gracefully (:math:`k=0` gives
:math:`[0,\, z^2/(n+z^2)]`), and has far better coverage for small :math:`n`.

**Convergence bounds used by the tests.** The sample mean of :math:`N` draws has standard
error :math:`\sigma/\sqrt{N}`, and the sample covariance entry :math:`\hat{C}_{ij}` has
standard error :math:`\sqrt{(C_{ii}C_{jj} + C_{ij}^2)/N}`. Test tolerances are computed
from these, at a stated confidence level, rather than picked by feel.

Validity
--------
* Runs are assumed **independent and identically distributed**. The substream scheme
  guarantees independence of the *inputs*; nothing here detects a ``run_fn`` that carries
  state between calls, and such a function makes every statistic on this page meaningless.
* ``1 + delta`` is **not clipped at zero**. For :math:`\sigma_\mathrm{mag} \gtrsim 0.3` a
  Gaussian scale factor produces retrograde burns at a non-negligible rate. Clipping would
  silently bias the mean of the very quantity the campaign is measuring, so the model is
  left honest and the regime is declared out of scope instead.
* The pointing model treats :math:`\theta` as an unwrapped Gaussian angle. It is exact for
  any :math:`\theta`, but the Gaussian *model* is only meaningful for
  :math:`\sigma_\mathrm{point} \ll 1` rad; beyond that, angles wrap and the distribution on
  the sphere is not what the parameter suggests.
* Covariances must be strictly positive definite. A positive *semi*-definite covariance
  (an exactly-certain component) is rejected, not degenerately sampled. That is a
  deliberate choice: a zero-variance direction is more often a user error than an intent.
* The confidence intervals are frequentist and assume the runs are Bernoulli trials with a
  fixed underlying probability. They quantify sampling noise only; they say nothing about
  whether the dispersions themselves are the right ones.

Units are SI (metres, seconds, radians). Dispersion sigmas carry the units of the quantity
they disperse, except :attr:`MagnitudePointingDispersion.sigma_magnitude`, which is a
dimensionless fraction.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .exceptions import DegenerateGeometryError, RpoCoreError

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_PERCENTILES",
    "BurnExecutionSample",
    "CampaignConfigurationError",
    "CampaignResults",
    "CampaignSummary",
    "Dispersion",
    "DispersionDefinitionError",
    "MagnitudePointingDispersion",
    "MetricSummary",
    "MonteCarloError",
    "NormalDispersion",
    "ProportionEstimate",
    "RunFailure",
    "RunRecord",
    "UniformDispersion",
    "VectorNormalDispersion",
    "dispersion_from_dict",
    "draw_samples",
    "execute_run",
    "proportion_estimate",
    "run_campaign",
    "summarise_metric",
    "wilson_interval",
]


_ResultT = TypeVar("_ResultT")
_NominalT = TypeVar("_NominalT")


class MonteCarloError(RpoCoreError):
    """Base class for every error raised by :mod:`rpo_core.montecarlo`."""


class DispersionDefinitionError(MonteCarloError, ValueError):
    """Raised when a dispersion is malformed or statistically impossible.

    Examples: a negative standard deviation, a uniform range with ``low >= high``, a
    non-symmetric covariance, or a covariance that is not positive definite (and therefore
    has no Cholesky factor and describes no realisable Gaussian).
    """


class CampaignConfigurationError(MonteCarloError, ValueError):
    """Raised when a campaign is asked for something it cannot deliver reproducibly.

    Examples: a non-positive run count, a seed that is not a non-negative integer, a
    confidence level outside ``(0, 1)``, or a user-supplied ``map_fn`` that returned a
    different set of run indices from the one it was given -- which would mean runs were
    silently dropped or duplicated, the exact failure this module exists to prevent.
    """


#: Percentiles reported for every retained metric unless overridden.
#:
#: 1/5/50/95/99 rather than quartiles: a dispersion campaign is run to find out what
#: happens in the tails. The median is included because a mean alone hides a bimodal
#: outcome (e.g. half the runs abort and half do not).
DEFAULT_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 50.0, 95.0, 99.0)

#: Default two-sided confidence level for interval estimates.
DEFAULT_CONFIDENCE: float = 0.95

#: Bytes of BLAKE2b digest used to turn a dispersion name into a spawn key.
#:
#: 8 bytes = 64 bits. Two dispersion names in one campaign colliding would give them the
#: same random substream; at the scale of tens of names per campaign the probability is
#: ~1e-17 and the alternative (positional keys) has the far worse property that adding a
#: dispersion perturbs every other dispersion's stream.
_NAME_DIGEST_BYTES: int = 8


# --------------------------------------------------------------------------------------
# Sample types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BurnExecutionSample:
    """One draw of burn execution error, in a form that does not yet know the burn.

    Holding the draw separately from its application is what keeps the dispersion
    *definition* free of the nominal delta-v. The rotation axis is defined by an azimuth
    in the plane perpendicular to the commanded impulse, so the plane -- and therefore the
    nominal vector -- is only needed at :meth:`apply` time.

    Attributes
    ----------
    scale
        Magnitude scale factor ``1 + delta``. Dimensionless. Values below zero are
        possible and are not clipped; see the module Validity section.
    tilt_rad
        Signed rotation angle about the perpendicular axis, radians. The angle between the
        executed and commanded impulse is exactly ``abs(tilt_rad)``.
    azimuth_rad
        Orientation of the rotation axis within the plane perpendicular to the commanded
        impulse, radians in ``[0, 2*pi)``.

    """

    scale: float
    tilt_rad: float
    azimuth_rad: float

    def apply(self, dv_nominal_m_s: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the executed impulse for a commanded impulse.

        Parameters
        ----------
        dv_nominal_m_s
            Commanded impulse, shape (3,), m/s. Must be finite and non-zero.

        Returns
        -------
        numpy.ndarray
            Shape (3,), m/s. Its norm is ``abs(scale)`` times the commanded norm and its
            angle from the commanded direction is ``abs(tilt_rad)``, both exactly.

        Raises
        ------
        ValueError
            If ``dv_nominal_m_s`` is not a finite 3-vector.
        DegenerateGeometryError
            If the commanded impulse is the zero vector: it has no direction, so there is
            no pointing error to apply and no plane in which to place the rotation axis.

        """
        dv = np.asarray(dv_nominal_m_s, dtype=np.float64)
        if dv.shape != (3,):
            raise ValueError(f"dv_nominal_m_s must have shape (3,), got {dv.shape}")
        if not np.all(np.isfinite(dv)):
            raise ValueError(f"dv_nominal_m_s must be finite, got {dv!r}")
        norm = float(np.linalg.norm(dv))
        if norm == 0.0:
            raise DegenerateGeometryError(
                "dv_nominal_m_s is the zero vector, which has no direction: a pointing "
                "error is a rotation of a direction, and there is none to rotate. A burn "
                "of zero magnitude should be omitted from the manoeuvre plan rather than "
                "dispersed."
            )
        u = dv / norm
        e1, e2 = _perpendicular_basis(u)
        axis = math.cos(self.azimuth_rad) * e1 + math.sin(self.azimuth_rad) * e2
        rotated = _rodrigues(u, axis, self.tilt_rad)
        return self.scale * norm * rotated


#: What a single dispersion contributes to a run's sample map.
DispersionSample = float | npt.NDArray[np.float64] | BurnExecutionSample


def _perpendicular_basis(
    u: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return an orthonormal pair spanning the plane perpendicular to unit vector ``u``.

    The seed vector is the coordinate axis along which ``u`` is *smallest*, which bounds
    the cross product away from zero: the worst case is ``u`` at equal angles to all three
    axes, where the cross product still has norm ``sqrt(2/3)``. Seeding with a fixed axis
    instead would fail whenever ``u`` happened to be parallel to it.
    """
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(u)))] = 1.0
    e1 = np.cross(u, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def _rodrigues(
    v: npt.NDArray[np.float64], axis: npt.NDArray[np.float64], angle_rad: float
) -> npt.NDArray[np.float64]:
    """Rotate ``v`` about the unit vector ``axis`` by ``angle_rad`` (Rodrigues' formula).

    The full formula is used rather than the two-term form valid for ``axis . v == 0``, so
    that a rounding-level component of ``axis`` along ``v`` degrades the result smoothly
    instead of quietly breaking norm preservation.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


# --------------------------------------------------------------------------------------
# Dispersion definitions
# --------------------------------------------------------------------------------------


class Dispersion(ABC):
    """Declarative description of one dispersed quantity.

    A dispersion knows how to draw itself from a generator and how to describe itself as
    JSON. It does **not** know the nominal scenario, how many runs there are, or what the
    sample will be used for. That separation is what lets the same definition be reused
    across projects and recorded verbatim in a campaign summary.

    Subclasses are frozen dataclasses that validate in ``__post_init__``. Plain
    dataclasses are used rather than the Pydantic models of :mod:`rpo_core.config`
    because Pydantic collects ``ValueError`` subclasses raised inside validation into a
    ``ValidationError``, which would bury :class:`DispersionDefinitionError` -- and the
    smallest eigenvalue it reports -- one exception deep. Pydantic earns its place at the
    YAML boundary; dispersions are constructed in Python by other numerics code, where a
    directly catchable typed exception is worth more than schema coercion.
    """

    #: Tag written to, and dispatched on by, the JSON representation.
    kind: str = "dispersion"

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> DispersionSample:
        """Draw one sample from an explicit generator.

        Parameters
        ----------
        rng
            Generator to draw from. Never the global RNG; see ``docs/conventions.md``.

        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, including the ``kind`` tag.

        Round-trips through :func:`dispersion_from_dict`, so a campaign summary carries
        enough information to rebuild the campaign that produced it.
        """

    def __eq__(self, other: object) -> bool:
        """Compare two dispersions by type and by their JSON description.

        Defined here, with ``eq=False`` on every subclass, because the dataclass-generated
        ``__eq__`` compares fields with ``==`` and a numpy covariance returns an *array*
        from that comparison, so ``VectorNormalDispersion(...) == other`` would raise
        "truth value of an array is ambiguous" instead of answering the question.
        """
        if type(self) is not type(other):
            return NotImplemented
        assert isinstance(other, Dispersion)
        return self.describe() == other.describe()

    def __hash__(self) -> int:
        """Hash the canonical JSON description, consistent with :meth:`__eq__`."""
        return hash(json.dumps(self.describe(), sort_keys=True))


def _check_sigma(value: float, name: str, kind: str) -> float:
    """Return ``value`` as a validated finite non-negative standard deviation."""
    sigma = float(value)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise DispersionDefinitionError(
            f"{kind}: {name}={value!r} must be a finite non-negative standard deviation. "
            "A negative sigma is not a wider distribution; it is a sign error that would "
            "sample identically to its absolute value and hide itself."
        )
    return sigma


@dataclass(frozen=True, eq=False)
class NormalDispersion(Dispersion):
    """Scalar Gaussian ``N(mean, sigma**2)``.

    Attributes
    ----------
    mean
        Distribution mean, in the units of the dispersed quantity.
    sigma
        Standard deviation, same units. ``sigma == 0`` is allowed and yields a constant;
        it is the useful way to switch one dispersion off without changing the campaign
        structure.

    Raises
    ------
    DispersionDefinitionError
        If ``mean`` is not finite, or ``sigma`` is negative or not finite.

    """

    kind: str = field(default="normal", init=False, repr=False)
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Validate the distribution parameters."""
        if not math.isfinite(float(self.mean)):
            raise DispersionDefinitionError(f"normal: mean={self.mean!r} must be finite")
        _check_sigma(self.sigma, "sigma", "normal")

    def sample(self, rng: np.random.Generator) -> float:
        """Return one draw from ``N(mean, sigma**2)``."""
        return float(rng.normal(self.mean, self.sigma))

    def describe(self) -> dict[str, Any]:
        """Return the JSON description of this dispersion."""
        return {"kind": self.kind, "mean": float(self.mean), "sigma": float(self.sigma)}


@dataclass(frozen=True, eq=False)
class UniformDispersion(Dispersion):
    """Scalar uniform on the half-open interval ``[low, high)``.

    Attributes
    ----------
    low, high
        Interval bounds, in the units of the dispersed quantity. ``low`` must be strictly
        below ``high``.

    Raises
    ------
    DispersionDefinitionError
        If a bound is not finite or ``low >= high``.

    """

    kind: str = field(default="uniform", init=False, repr=False)
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        """Validate the interval."""
        low, high = float(self.low), float(self.high)
        if not (math.isfinite(low) and math.isfinite(high)):
            raise DispersionDefinitionError(
                f"uniform: bounds must be finite, got low={self.low!r}, high={self.high!r}"
            )
        if low >= high:
            raise DispersionDefinitionError(
                f"uniform: low={low!r} must be strictly below high={high!r}. An empty or "
                "point interval is a degenerate distribution; use NormalDispersion with "
                "sigma=0 if a constant is what was meant."
            )

    def sample(self, rng: np.random.Generator) -> float:
        """Return one draw from ``U[low, high)``."""
        return float(rng.uniform(self.low, self.high))

    def describe(self) -> dict[str, Any]:
        """Return the JSON description of this dispersion."""
        return {"kind": self.kind, "low": float(self.low), "high": float(self.high)}


@dataclass(frozen=True, eq=False)
class VectorNormalDispersion(Dispersion):
    """Correlated multivariate Gaussian, sampled through a Cholesky factor.

    ``x = mean + L @ z`` with ``L L.T == covariance`` and ``z ~ N(0, I)``. Sampling
    through the factor rather than drawing each component independently is the whole
    point: the off-diagonal terms are what make a navigation-error dispersion describe a
    real error ellipsoid instead of an axis-aligned box.

    Attributes
    ----------
    mean
        Shape ``(k,)``, in the units of the dispersed quantity.
    covariance
        Shape ``(k, k)``, symmetric positive definite, in squared units.

    Raises
    ------
    DispersionDefinitionError
        If the shapes disagree, an entry is non-finite, the covariance is not symmetric,
        or the covariance is not positive definite. The last case reports the smallest
        eigenvalue, which is the number that says how badly it fails and in which
        direction the variance went negative.

    """

    kind: str = field(default="vector_normal", init=False, repr=False)
    mean: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))
    covariance: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(1))
    _cholesky: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Coerce to float arrays and prove the covariance admits a Cholesky factor."""
        mean = np.asarray(self.mean, dtype=np.float64)
        cov = np.asarray(self.covariance, dtype=np.float64)
        if mean.ndim != 1 or mean.size == 0:
            raise DispersionDefinitionError(
                f"vector_normal: mean must be a non-empty 1-D array, got shape {mean.shape}"
            )
        if cov.shape != (mean.size, mean.size):
            raise DispersionDefinitionError(
                f"vector_normal: covariance must have shape {(mean.size, mean.size)} to "
                f"match mean of length {mean.size}, got {cov.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            raise DispersionDefinitionError(
                f"vector_normal: mean and covariance must be finite, got mean={mean!r}, "
                f"covariance={cov!r}"
            )
        asymmetry = float(np.max(np.abs(cov - cov.T))) if cov.size else 0.0
        if asymmetry > 0.0:
            raise DispersionDefinitionError(
                f"vector_normal: covariance is not symmetric (max |C - C.T| = "
                f"{asymmetry:.6e}). A covariance matrix is symmetric by definition, so an "
                "asymmetric one means the entries were assembled wrongly; symmetrising it "
                "here would hide that."
            )
        eigenvalues = np.linalg.eigvalsh(cov)
        smallest = float(eigenvalues[0])
        if smallest <= 0.0:
            raise DispersionDefinitionError(
                f"vector_normal: covariance is not positive definite -- its smallest "
                f"eigenvalue is {smallest:.6e} (largest {float(eigenvalues[-1]):.6e}). A "
                "non-positive eigenvalue means a direction with zero or negative variance, "
                "for which no Cholesky factor and no realisable Gaussian exists."
            )
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - eigenvalue check precedes
            raise DispersionDefinitionError(
                f"vector_normal: Cholesky factorisation failed despite a smallest "
                f"eigenvalue of {smallest:.6e}: {exc}"
            ) from exc
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "_cholesky", chol)

    @property
    def dimension(self) -> int:
        """Return the dimension ``k`` of the sampled vector."""
        return int(self.mean.size)

    @property
    def cholesky(self) -> npt.NDArray[np.float64]:
        """Return the lower Cholesky factor ``L`` with ``L @ L.T == covariance``."""
        return self._cholesky

    def sample(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Return one correlated draw, shape ``(dimension,)``.

        Implemented as ``mean + L @ z`` rather than ``rng.multivariate_normal`` so that
        the factorisation is the one this class validated, and so the draw consumes a
        fixed, documented number of standard normals.
        """
        z = rng.standard_normal(self.dimension)
        drawn: npt.NDArray[np.float64] = self.mean + self.cholesky @ z
        return drawn

    def describe(self) -> dict[str, Any]:
        """Return the JSON description of this dispersion."""
        return {
            "kind": self.kind,
            "mean": [float(v) for v in self.mean],
            "covariance": [[float(v) for v in row] for row in self.covariance],
        }


@dataclass(frozen=True, eq=False)
class MagnitudePointingDispersion(Dispersion):
    """Burn execution error: a magnitude scale factor and an independent pointing tilt.

    The executed impulse is ``(1 + delta) * R(axis, theta) @ dv``, with
    ``delta ~ N(0, sigma_magnitude**2)``, ``theta ~ N(0, sigma_pointing_rad**2)``, and the
    rotation axis uniformly distributed in azimuth *within the plane perpendicular to the
    commanded impulse*. See the module docstring for the equations and for why additive
    component-wise noise is the wrong model.

    **Why the axis is perpendicular rather than uniform on the sphere.** Both are
    "uniformly random axis" models, but they mean different things. Rotating by ``theta``
    about an axis at angle ``phi`` from the impulse turns the impulse by only
    ``alpha`` with ``cos(alpha) = cos(phi)**2 + sin(phi)**2 * cos(theta)``, so with a
    full-sphere axis the *realised* pointing error is strictly smaller than ``theta`` and
    ``sigma_pointing_rad`` would not be the standard deviation of anything observable.
    Restricting the axis to the perpendicular plane -- uniform in azimuth, which is the
    physically meaningful uniformity for a thruster misalignment -- makes the realised
    half-cone angle exactly ``abs(theta)``, so the parameter is checkable against a closed
    form and means what it says.

    Attributes
    ----------
    sigma_magnitude
        Standard deviation of the *fractional* magnitude error. Dimensionless: 0.01 is a
        1 % one-sigma scale-factor error.
    sigma_pointing_rad
        Standard deviation of the tilt angle, radians.

    Raises
    ------
    DispersionDefinitionError
        If either sigma is negative or non-finite.

    """

    kind: str = field(default="magnitude_pointing", init=False, repr=False)
    sigma_magnitude: float = 0.0
    sigma_pointing_rad: float = 0.0

    def __post_init__(self) -> None:
        """Validate both sigmas."""
        _check_sigma(self.sigma_magnitude, "sigma_magnitude", "magnitude_pointing")
        _check_sigma(self.sigma_pointing_rad, "sigma_pointing_rad", "magnitude_pointing")

    def sample(self, rng: np.random.Generator) -> BurnExecutionSample:
        """Return one burn-execution draw.

        Draw order is fixed (scale, tilt, azimuth) because changing it would change every
        historical campaign's samples for the same seed.
        """
        scale = 1.0 + float(rng.normal(0.0, self.sigma_magnitude))
        tilt = float(rng.normal(0.0, self.sigma_pointing_rad))
        azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
        return BurnExecutionSample(scale=scale, tilt_rad=tilt, azimuth_rad=azimuth)

    def describe(self) -> dict[str, Any]:
        """Return the JSON description of this dispersion."""
        return {
            "kind": self.kind,
            "sigma_magnitude": float(self.sigma_magnitude),
            "sigma_pointing_rad": float(self.sigma_pointing_rad),
        }


_DISPERSION_TYPES: dict[str, type[Dispersion]] = {
    "normal": NormalDispersion,
    "uniform": UniformDispersion,
    "vector_normal": VectorNormalDispersion,
    "magnitude_pointing": MagnitudePointingDispersion,
}


def dispersion_from_dict(payload: Mapping[str, Any]) -> Dispersion:
    """Rebuild a dispersion from its :meth:`Dispersion.describe` output.

    The ``kind`` tag is the discriminator: the class hierarchy is a tagged union on the
    wire even though it is polymorphic in memory. This is what makes a campaign summary
    self-sufficient -- the JSON it writes is enough to reconstruct the campaign.

    Parameters
    ----------
    payload
        Mapping containing a ``kind`` key and that kind's fields.

    Returns
    -------
    Dispersion
        A validated dispersion of the tagged type.

    Raises
    ------
    DispersionDefinitionError
        If ``kind`` is missing or unknown, or the fields do not match the type.

    """
    try:
        kind = payload["kind"]
    except KeyError as exc:
        raise DispersionDefinitionError(
            f"dispersion payload has no 'kind' tag; keys were {sorted(payload)}"
        ) from exc
    cls = _DISPERSION_TYPES.get(str(kind))
    if cls is None:
        raise DispersionDefinitionError(
            f"unknown dispersion kind {kind!r}; known kinds are {sorted(_DISPERSION_TYPES)}"
        )
    fields = {k: v for k, v in payload.items() if k != "kind"}
    try:
        return cls(**fields)
    except TypeError as exc:
        raise DispersionDefinitionError(
            f"dispersion kind {kind!r} does not accept fields {sorted(fields)}: {exc}"
        ) from exc


# --------------------------------------------------------------------------------------
# Seeding: independent substreams
# --------------------------------------------------------------------------------------


def _validate_seed(seed: int) -> int:
    """Return ``seed`` as a validated non-negative integer."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CampaignConfigurationError(
            f"seed must be a non-negative int so that it can be written to, and read back "
            f"from, a JSON summary; got {seed!r} of type {type(seed).__name__}"
        )
    if seed < 0:
        raise CampaignConfigurationError(
            f"seed must be a non-negative int, got {seed}. numpy's SeedSequence rejects "
            "negative entropy, and a seed that cannot round-trip through JSON cannot "
            "reproduce the campaign it labels."
        )
    return seed


def _run_seed_sequence(seed: int, index: int) -> np.random.SeedSequence:
    """Return the :class:`numpy.random.SeedSequence` belonging to run ``index``.

    Built as the ``index``-th child of ``SeedSequence(seed)``. ``SeedSequence.spawn``
    numbers its children from zero on a freshly constructed parent, so child ``i`` depends
    only on ``(seed, i)`` -- not on how many children were requested, in what order they
    were consumed, or on which process consumed them. Constructing the child directly from
    its spawn key, rather than calling ``spawn(n)`` and indexing, makes that independence
    structural instead of incidental: there is no ``n`` in this function to depend on.
    """
    if index < 0:
        raise CampaignConfigurationError(f"run index must be non-negative, got {index}")
    return np.random.SeedSequence(entropy=seed, spawn_key=(index,))


def _named_substream(parent: np.random.SeedSequence, label: str) -> np.random.SeedSequence:
    """Return a substream of ``parent`` addressed by ``label`` rather than by position.

    A positional scheme (``parent.spawn(k)``, dispersion ``j`` takes child ``j``) has a
    nasty property: adding or removing one dispersion shifts every later dispersion onto a
    different stream, so an unrelated edit silently changes the samples of variables that
    did not change. Hashing the name into the spawn key means each dispersion's stream
    depends only on ``(seed, run index, its own name)``.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=_NAME_DIGEST_BYTES).digest()
    key = int.from_bytes(digest, "big")
    return np.random.SeedSequence(entropy=parent.entropy, spawn_key=(*parent.spawn_key, key))


def draw_samples(
    dispersions: Mapping[str, Dispersion], *, seed: int, index: int
) -> dict[str, DispersionSample]:
    """Draw the sample map for one run, independently of every other run.

    Parameters
    ----------
    dispersions
        Named dispersion definitions.
    seed
        Campaign seed.
    index
        Zero-based run index.

    Returns
    -------
    dict
        Name to sample, iterated in sorted-name order. The order is cosmetic: each
        dispersion draws from its own name-addressed substream, so the value of one
        dispersion does not depend on which others are present or on dict insertion order.

    Raises
    ------
    CampaignConfigurationError
        If ``seed`` or ``index`` is invalid.

    """
    parent = _run_seed_sequence(_validate_seed(seed), index)
    return {
        name: dispersions[name].sample(np.random.default_rng(_named_substream(parent, name)))
        for name in sorted(dispersions)
    }


# --------------------------------------------------------------------------------------
# Per-run execution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunFailure:
    """A run that raised, recorded rather than discarded.

    Attributes
    ----------
    index
        Zero-based index of the run that failed.
    exception_type
        Class name of the exception.
    message
        ``str`` of the exception.
    traceback_text
        Formatted traceback, kept in memory for debugging. It is deliberately *not*
        written to the JSON summary: tracebacks carry absolute paths and line numbers of
        the machine that ran the campaign, which would make the summary differ between
        machines for the same seed.
    stage
        Where it failed: ``"sample"``, ``"run"``, ``"retain"``, or ``"success"``. A
        campaign whose failures are all in ``retain`` has a broken metric extractor, not a
        fragile trajectory, and the distinction is invisible without this field.

    """

    index: int
    exception_type: str
    message: str
    traceback_text: str
    stage: str

    def describe(self) -> dict[str, Any]:
        """Return the JSON description, without the machine-specific traceback."""
        return {
            "index": self.index,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class RunRecord(Generic[_ResultT]):
    """Everything retained about one run, successful or not.

    Attributes
    ----------
    index
        Zero-based run index. Records are always aggregated in index order regardless of
        execution order.
    samples
        The dispersion samples this run was given. Reproducible from ``(seed, index)``,
        but kept so that a failure can be reproduced without re-deriving them.
    metrics
        Retained scalar metrics. Empty for a failed run -- not zero-filled, because a zero
        would flow into a percentile as if it were an observation.
    result
        The raw ``run_fn`` return value, or ``None`` if the campaign was run with
        ``keep_results=False`` (the default) or the run failed.
    success
        ``True`` only if the run completed *and* satisfied the success criterion.
    failure
        The recorded exception, or ``None``.

    """

    index: int
    samples: Mapping[str, DispersionSample]
    metrics: Mapping[str, float]
    result: _ResultT | None
    success: bool
    failure: RunFailure | None

    @property
    def failed(self) -> bool:
        """Return whether this run raised."""
        return self.failure is not None


def execute_run(
    nominal: _NominalT,
    dispersions: Mapping[str, Dispersion],
    run_fn: Callable[[_NominalT, Mapping[str, DispersionSample], np.random.Generator], _ResultT],
    *,
    seed: int,
    index: int,
    retain: Mapping[str, Callable[[_ResultT], float]] | None = None,
    success_fn: Callable[[_ResultT], bool] | None = None,
    keep_result: bool = False,
) -> RunRecord[_ResultT]:
    """Execute one run and return its record. Never raises for a failing run.

    This is the whole of concern 3. It is a pure function of its arguments: two calls with
    the same ``(nominal, dispersions, seed, index)`` produce the same samples, and -- for a
    deterministic ``run_fn`` -- the same metrics. Nothing about the campaign it belongs to
    is visible from here, which is what makes a campaign resumable: run 4711 of a failed
    10 000-run campaign can be re-executed on its own and will be the identical run.

    ``run_fn`` receives its own generator, drawn from the run's substream and independent
    of every dispersion's substream, for randomness the declared dispersions do not cover
    (per-step navigation noise, for instance). Consuming it does not perturb the samples.

    Parameters
    ----------
    nominal
        The undispersed scenario, passed through untouched.
    dispersions
        Named dispersion definitions.
    run_fn
        ``(nominal, samples, rng) -> result``.
    seed
        Campaign seed.
    index
        Zero-based run index.
    retain
        Named scalar extractors applied to the result. An extractor that raises, or
        returns a non-finite value, fails the run rather than poisoning the statistics.
    success_fn
        Predicate on the result. Defaults to "the run completed".
    keep_result
        Whether to retain the raw result on the record.

    Returns
    -------
    RunRecord
        With ``failure`` set if any stage raised.

    Raises
    ------
    CampaignConfigurationError
        Only for invalid ``seed`` or ``index``. Failures *inside* the run are recorded,
        not raised.

    """
    parent = _run_seed_sequence(_validate_seed(seed), index)
    run_rng = np.random.default_rng(parent)
    samples: dict[str, DispersionSample] = {}

    def _record_failure(stage: str, exc: Exception) -> RunRecord[_ResultT]:
        return RunRecord(
            index=index,
            samples=samples,
            metrics={},
            result=None,
            success=False,
            failure=RunFailure(
                index=index,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback_text="".join(traceback.format_exception(exc)),
                stage=stage,
            ),
        )

    # Only `Exception` is caught, never `BaseException`: KeyboardInterrupt and SystemExit
    # mean the operator wants the campaign to stop, and swallowing them into a "failed
    # run" would turn Ctrl-C into a campaign of 10 000 recorded failures.
    try:
        samples = draw_samples(dispersions, seed=seed, index=index)
    except Exception as exc:
        return _record_failure("sample", exc)

    try:
        result = run_fn(nominal, samples, run_rng)
    except Exception as exc:
        return _record_failure("run", exc)

    metrics: dict[str, float] = {}
    for name, extractor in (retain or {}).items():
        try:
            value = float(extractor(result))
        except Exception as exc:
            return _record_failure("retain", exc)
        if not math.isfinite(value):
            return _record_failure(
                "retain",
                ValueError(
                    f"retained metric {name!r} is {value!r} for run {index}. A non-finite "
                    "metric would propagate through every mean and percentile as NaN; the "
                    "run is recorded as a failure instead."
                ),
            )
        metrics[name] = value

    try:
        success = True if success_fn is None else bool(success_fn(result))
    except Exception as exc:
        return _record_failure("success", exc)

    return RunRecord(
        index=index,
        samples=samples,
        metrics=metrics,
        result=result if keep_result else None,
        success=success,
        failure=None,
    )


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def _validate_confidence(confidence: float) -> float:
    """Return ``confidence`` as a validated level strictly inside ``(0, 1)``."""
    level = float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise CampaignConfigurationError(
            f"confidence must lie strictly inside (0, 1), got {confidence!r}. A level of 1 "
            "would demand an infinitely wide interval and 0 a zero-width one."
        )
    return level


def wilson_interval(
    successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    r"""Return the two-sided Wilson score interval for a binomial proportion.

    .. math:: p_{\pm} = \frac{1}{n + z^2}\left[ k + \frac{z^2}{2} \pm
              z\sqrt{\frac{k(n-k)}{n} + \frac{z^2}{4}} \right]

    Wilson rather than the normal (Wald) approximation because a dispersion campaign is
    usually run to establish a rate that is *close to* 0 % or 100 % -- a collision
    probability, a constraint-violation rate -- and that is exactly where Wald fails. At
    ``k = 0`` Wald reports the interval ``[0, 0]``: zero failures observed in ten runs
    "proves" a zero failure rate. Wilson reports ``[0, z**2/(n + z**2)]``, which for
    ``n = 10`` at 95 % is ``[0, 0.2775]`` -- an honest statement that ten runs are
    compatible with a 27 % failure rate. Wald also produces bounds outside ``[0, 1]`` for
    small ``k`` (at ``k = 1, n = 10`` its lower bound is ``-0.086``); Wilson cannot,
    because its bounds are the roots of the score equation and lie in ``[0, 1]`` by
    construction.

    Parameters
    ----------
    successes
        Number of successes ``k``, ``0 <= k <= n``.
    trials
        Number of trials ``n``, strictly positive.
    confidence
        Two-sided confidence level in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(lower, upper)``, both in ``[0, 1]``.

    Raises
    ------
    CampaignConfigurationError
        If ``trials`` is not positive, ``successes`` is outside ``[0, trials]``, either is
        not an integer, or ``confidence`` is outside ``(0, 1)``.

    Examples
    --------
    >>> lower, upper = wilson_interval(0, 10)
    >>> round(lower, 4), round(upper, 4)
    (0.0, 0.2775)

    """
    level = _validate_confidence(confidence)
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise CampaignConfigurationError(
            f"trials must be a positive int, got {trials!r}. A proportion over zero trials "
            "is undefined, and reporting one as 0/0 = 0 would be the worst possible answer."
        )
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise CampaignConfigurationError(f"successes must be an int, got {successes!r}")
    if not 0 <= successes <= trials:
        raise CampaignConfigurationError(f"successes={successes} must lie in [0, trials={trials}]")

    n = float(trials)
    k = float(successes)
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    z2 = z * z
    denominator = n + z2
    centre = (k + z2 / 2.0) / denominator
    half_width = z / denominator * math.sqrt(k * (n - k) / n + z2 / 4.0)
    lower = centre - half_width
    upper = centre + half_width
    # The endpoints are exact, not rounded: at k = 0 the half width equals the centre, so
    # the lower bound is analytically zero, and k = n is its mirror image. Floating-point
    # evaluation of the two terms lands an ulp away from that, and an interval reported as
    # [0, 0.2775327] but stored as [-1e-17, ...] fails an "is the rate zero?" check for
    # reasons that have nothing to do with the statistics. The clip on the other end is a
    # rounding guard only; the analytic bounds are inside [0, 1] for every admissible pair.
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    return (max(0.0, lower), min(1.0, upper))


@dataclass(frozen=True)
class ProportionEstimate:
    """A rate with the interval that says how much of it is sampling noise.

    Attributes
    ----------
    successes, trials
        Counts. ``trials`` is *every* run in the campaign, including those that raised;
        see :meth:`CampaignResults.success_rate`.
    point
        ``successes / trials``.
    lower, upper
        Wilson score bounds at ``confidence``.
    confidence
        Two-sided confidence level.

    """

    successes: int
    trials: int
    point: float
    lower: float
    upper: float
    confidence: float

    def describe(self) -> dict[str, Any]:
        """Return the JSON description of this estimate."""
        return {
            "successes": self.successes,
            "trials": self.trials,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "interval": "wilson_score",
        }


def proportion_estimate(
    successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> ProportionEstimate:
    """Return a :class:`ProportionEstimate` with Wilson score bounds."""
    lower, upper = wilson_interval(successes, trials, confidence=confidence)
    return ProportionEstimate(
        successes=successes,
        trials=trials,
        point=successes / trials,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


@dataclass(frozen=True)
class MetricSummary:
    """Summary statistics of one retained scalar metric.

    ``None`` rather than ``NaN`` for undefined entries: NaN is not representable in
    strict JSON, and it compares false against everything, so a NaN mean silently passes
    an "is it below the limit?" check. ``None`` fails loudly.

    Attributes
    ----------
    name
        Metric name, as given in ``retain``.
    count
        Number of *completed* runs contributing. Failed runs contribute nothing.
    mean, std
        Sample mean and sample standard deviation (``ddof=1``, the unbiased estimator).
        ``std`` is ``None`` for ``count < 2``, where it is undefined rather than zero.
    minimum, maximum
        Extremes, ``None`` for ``count == 0``.
    percentiles
        Percentile level (0-100) to value, by linear interpolation between order
        statistics (``numpy.percentile`` default). Empty for ``count == 0``.

    """

    name: str
    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    percentiles: Mapping[float, float]

    def describe(self) -> dict[str, Any]:
        """Return the JSON description of this metric summary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "percentiles": {f"p{q:g}": v for q, v in self.percentiles.items()},
            "percentile_method": "linear",
            "std_ddof": 1,
        }


def summarise_metric(
    name: str,
    values: npt.ArrayLike,
    *,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> MetricSummary:
    """Summarise one metric's samples.

    Parameters
    ----------
    name
        Metric name, carried into the summary.
    values
        1-D array of observations from completed runs.
    percentiles
        Levels in ``[0, 100]``.

    Returns
    -------
    MetricSummary

    Raises
    ------
    CampaignConfigurationError
        If ``values`` is not 1-D, contains a non-finite entry, or a percentile level lies
        outside ``[0, 100]``.

    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if np.asarray(values, dtype=np.float64).ndim > 1:
        raise CampaignConfigurationError(
            f"metric {name!r}: values must be 1-D, got shape "
            f"{np.asarray(values, dtype=np.float64).shape}"
        )
    if array.size and not np.all(np.isfinite(array)):
        raise CampaignConfigurationError(
            f"metric {name!r}: values must all be finite; a non-finite observation would "
            "propagate silently through the mean and every percentile"
        )
    for q in percentiles:
        if not 0.0 <= float(q) <= 100.0:
            raise CampaignConfigurationError(
                f"metric {name!r}: percentile level {q!r} is outside [0, 100]"
            )

    count = int(array.size)
    if count == 0:
        return MetricSummary(name, 0, None, None, None, None, {})
    return MetricSummary(
        name=name,
        count=count,
        mean=float(np.mean(array)),
        std=float(np.std(array, ddof=1)) if count > 1 else None,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        percentiles={
            float(q): float(np.percentile(array, float(q), method="linear")) for q in percentiles
        },
    )


# --------------------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignSummary:
    """A campaign reduced to numbers, and reproducible from them.

    The JSON produced by :meth:`to_json` carries the seed, the run count, the failure
    count and breakdown, and the verbatim dispersion definitions. That is everything
    needed to reconstruct the campaign: feed the ``dispersions`` block back through
    :func:`dispersion_from_dict` and re-run with the same seed.

    ``numpy_version`` and ``bit_generator`` are recorded because they are the parts of the
    reproducibility contract this module does not own. Identical seeds under a different
    default bit generator give different samples, and a summary that did not say which one
    it used would be quietly unreproducible.
    """

    seed: int
    n_runs: int
    n_failures: int
    n_successes: int
    success_rate: ProportionEstimate
    completion_rate: ProportionEstimate
    dispersions: Mapping[str, dict[str, Any]]
    metrics: Mapping[str, MetricSummary]
    failures: Sequence[RunFailure]
    numpy_version: str = field(default_factory=lambda: str(np.__version__))
    bit_generator: str = "PCG64"

    def failure_counts_by_type(self) -> dict[str, int]:
        """Return how many failures each exception type accounts for."""
        counts: dict[str, int] = {}
        for failure in self.failures:
            counts[failure.exception_type] = counts.get(failure.exception_type, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of this summary."""
        return {
            "seed": self.seed,
            "n_runs": self.n_runs,
            "n_successes": self.n_successes,
            "n_failures": self.n_failures,
            "success_rate": self.success_rate.describe(),
            "completion_rate": self.completion_rate.describe(),
            "dispersions": {name: dict(spec) for name, spec in self.dispersions.items()},
            "metrics": {name: m.describe() for name, m in self.metrics.items()},
            "failures": [f.describe() for f in self.failures],
            "failure_counts_by_type": self.failure_counts_by_type(),
            "numpy_version": self.numpy_version,
            "bit_generator": self.bit_generator,
            "substream_scheme": "seed_sequence_spawn_key",
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the summary as a JSON string.

        ``allow_nan=False``: a NaN would be emitted as the non-standard literal ``NaN``,
        which most JSON parsers reject. Every undefined statistic is already ``None``.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class CampaignResults(Generic[_ResultT]):
    """The records of one campaign, plus the analysis built on them.

    Attributes
    ----------
    seed
        Campaign seed.
    records
        One :class:`RunRecord` per run, in index order, including failures.
    dispersions
        The definitions used, retained verbatim for the summary.
    retained
        Names of the metrics that were extracted.

    """

    seed: int
    records: tuple[RunRecord[_ResultT], ...]
    dispersions: Mapping[str, Dispersion]
    retained: tuple[str, ...]

    @property
    def n_runs(self) -> int:
        """Return the total number of runs, failures included."""
        return len(self.records)

    @property
    def failures(self) -> tuple[RunFailure, ...]:
        """Return the recorded failures, in run-index order."""
        return tuple(r.failure for r in self.records if r.failure is not None)

    @property
    def n_failures(self) -> int:
        """Return the number of runs that raised."""
        return len(self.failures)

    @property
    def n_successes(self) -> int:
        """Return the number of runs that completed *and* met the success criterion."""
        return sum(1 for r in self.records if r.success)

    def metric_values(self, name: str) -> npt.NDArray[np.float64]:
        """Return the observations of one metric, from completed runs only.

        Raises
        ------
        CampaignConfigurationError
            If ``name`` was not retained.

        """
        if name not in self.retained:
            raise CampaignConfigurationError(
                f"metric {name!r} was not retained; retained metrics are {list(self.retained)}"
            )
        return np.array(
            [r.metrics[name] for r in self.records if not r.failed and name in r.metrics],
            dtype=np.float64,
        )

    def success_rate(self, *, confidence: float = DEFAULT_CONFIDENCE) -> ProportionEstimate:
        """Return the success rate over **all** runs, with a Wilson interval.

        The denominator is ``n_runs``, not the number of runs that completed. A campaign
        in which 30 % of runs raised and every survivor met its criterion has a success
        rate of 70 %, not 100 %. Dividing by the survivors would let a campaign improve
        its reported reliability by crashing more often, which is the single most
        dangerous number this module could produce.
        """
        return proportion_estimate(self.n_successes, self.n_runs, confidence=confidence)

    def completion_rate(self, *, confidence: float = DEFAULT_CONFIDENCE) -> ProportionEstimate:
        """Return the fraction of runs that did not raise, with a Wilson interval.

        Reported alongside the success rate so the two causes of a low rate -- the
        scenario failing its criterion versus the code failing to run -- are never
        confused for one another.
        """
        return proportion_estimate(
            self.n_runs - self.n_failures, self.n_runs, confidence=confidence
        )

    def summary(
        self,
        *,
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> CampaignSummary:
        """Reduce the campaign to a serialisable summary."""
        return CampaignSummary(
            seed=self.seed,
            n_runs=self.n_runs,
            n_failures=self.n_failures,
            n_successes=self.n_successes,
            success_rate=self.success_rate(confidence=confidence),
            completion_rate=self.completion_rate(confidence=confidence),
            dispersions={name: d.describe() for name, d in sorted(self.dispersions.items())},
            metrics={
                name: summarise_metric(name, self.metric_values(name), percentiles=percentiles)
                for name in self.retained
            },
            failures=self.failures,
        )


def run_campaign(
    nominal: _NominalT,
    dispersions: Mapping[str, Dispersion],
    run_fn: Callable[[_NominalT, Mapping[str, DispersionSample], np.random.Generator], _ResultT],
    n_runs: int,
    seed: int,
    *,
    retain: Mapping[str, Callable[[_ResultT], float]] | None = None,
    success_fn: Callable[[_ResultT], bool] | None = None,
    keep_results: bool = False,
    map_fn: Callable[
        [Callable[[int], RunRecord[_ResultT]], Sequence[int]], Iterable[RunRecord[_ResultT]]
    ]
    | None = None,
) -> CampaignResults[_ResultT]:
    """Run a seeded dispersion campaign.

    Determinism is the hard requirement, and it is structural rather than incidental.
    Run ``i`` draws from ``SeedSequence(seed, spawn_key=(i,))``, constructed directly from
    its index; ``n_runs`` appears nowhere in the derivation. Consequently the first 100
    runs of a 1000-run campaign are bitwise identical to a 100-run campaign with the same
    seed, and any execution order -- sequential, reversed, or a process pool -- gives the
    same records. That is what makes a campaign resumable: an interrupted 10 000-run study
    can be finished by executing only the missing indices.

    Failing runs are recorded, never dropped and never silently retried. They keep their
    place in the denominator of :meth:`CampaignResults.success_rate`.

    Parameters
    ----------
    nominal
        The undispersed scenario. Opaque to this function.
    dispersions
        Named dispersion definitions.
    run_fn
        ``(nominal, samples, rng) -> result``. Should be a pure function of its arguments;
        state carried between calls invalidates every statistic computed here.
    n_runs
        Number of runs, strictly positive.
    seed
        Non-negative integer campaign seed, recorded in the summary.
    retain
        Named scalar extractors. Anything not extracted here is gone unless
        ``keep_results`` is set.
    success_fn
        Predicate on the result. Defaults to "the run completed without raising".
    keep_results
        Retain the raw ``run_fn`` return values on the records. Defaults to ``False``: a
        10 000-run campaign holding a trajectory per run is a memory footgun, and the
        retained metrics are the intended durable product.
    map_fn
        Applies a per-index function over the run indices. Defaults to :func:`map`. Supply
        an executor's ``map`` to parallelise; results are re-sorted by index, so the
        mapping need not preserve order.

    Returns
    -------
    CampaignResults

    Raises
    ------
    CampaignConfigurationError
        If ``n_runs`` is not a positive int, ``seed`` is not a non-negative int, or
        ``map_fn`` returns a different set of run indices from the one it was given.

    Examples
    --------
    >>> import numpy as np
    >>> disp = {"dv": NormalDispersion(mean=1.0, sigma=0.1)}
    >>> def run(nominal, samples, rng):
    ...     return {"dv": float(samples["dv"])}
    >>> results = run_campaign(
    ...     None, disp, run, 200, seed=7, retain={"dv": lambda r: r["dv"]})
    >>> results.n_failures
    0
    >>> bool(abs(results.summary().metrics["dv"].mean - 1.0) < 0.05)
    True

    """
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs <= 0:
        raise CampaignConfigurationError(
            f"n_runs must be a positive int, got {n_runs!r}. An empty campaign has no "
            "proportion to estimate and no metrics to summarise."
        )
    validated_seed = _validate_seed(seed)
    indices = tuple(range(n_runs))

    def _one(index: int) -> RunRecord[_ResultT]:
        return execute_run(
            nominal,
            dispersions,
            run_fn,
            seed=validated_seed,
            index=index,
            retain=retain,
            success_fn=success_fn,
            keep_result=keep_results,
        )

    mapper = map_fn if map_fn is not None else map
    records = sorted(mapper(_one, indices), key=lambda r: r.index)

    # A map_fn that drops or duplicates work is the same silent-loss failure this module
    # exists to prevent, one level up. Check rather than trust.
    produced = [r.index for r in records]
    if produced != list(indices):
        raise CampaignConfigurationError(
            f"map_fn returned {len(produced)} records with indices "
            f"{produced[:8]}{'...' if len(produced) > 8 else ''}, but {n_runs} runs with "
            f"indices 0..{n_runs - 1} were requested. Runs were dropped or duplicated; a "
            "campaign that silently loses runs reports a success rate over the wrong "
            "denominator."
        )

    return CampaignResults(
        seed=validated_seed,
        records=tuple(records),
        dispersions=dict(dispersions),
        retained=tuple(retain or {}),
    )
