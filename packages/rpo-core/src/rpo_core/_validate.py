"""Shared input validation for the numerics core.

Private module. Every public entry point in ``rpo_core`` coerces and checks its arguments
before doing arithmetic, and before this module existed each of them did so with its own
copy of the same few checks: fifteen implementations across eleven files, seven of them
near-identical.

**Why these take an ``error_type`` rather than raising a single shared exception.** The
duplication was not purely accidental. Each module raises its *own* typed error --
``ConstraintDefinitionError``, ``GuidanceDefinitionError``, ``MetricsError``,
``CampaignConfigurationError`` -- so that a caller can catch precisely what it means to
handle. Collapsing those into one shared ``ValueError`` would remove real information and
break every ``pytest.raises`` that names a specific type. So the shared helpers carry the
*logic*, and each caller keeps its own exception class by passing ``error_type``.

The default is ``ValueError`` because that is what an unqualified bad-argument failure is,
and because ``RpoCoreError`` subclasses in this package derive from it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "as_state6",
    "as_states_n6",
    "as_vector",
    "validate_positive",
    "validate_seed",
    "validate_time_grid",
]


def as_vector(
    value: npt.ArrayLike,
    name: str,
    *,
    size: int = 3,
    error_type: type[Exception] = ValueError,
) -> npt.NDArray[np.float64]:
    """Coerce ``value`` to a finite float64 array of shape ``(size,)``.

    Raises
    ------
    error_type
        If the shape is wrong or any element is non-finite. The message names the argument
        and the offending value, so a caller can act without a debugger.

    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise error_type(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise error_type(f"{name} must be finite, got {array!r}")
    return array


def as_state6(
    value: npt.ArrayLike,
    name: str = "state",
    *,
    batch_hint: bool = False,
    error_type: type[Exception] = ValueError,
) -> npt.NDArray[np.float64]:
    """Coerce ``value`` to a finite float64 state of shape (6,).

    Deliberately strict about shape. An earlier version of ``propagate`` indexed with an
    ellipsis, advertising batch support that ``numpy.linalg.norm`` did not honour: a stacked
    (N, 6) input was flattened into a single norm, so three identical LEO states returned
    +54 MJ/kg instead of -29.3 MJ/kg. The sign flip made a bound orbit read as hyperbolic.
    Use :func:`as_states_n6` for stacked states.

    Parameters
    ----------
    value
        The candidate state.
    name
        Argument name, used in error messages.
    batch_hint
        Whether to suggest a batch variant in the error message. Only true for callers that
        actually expose one; suggesting a function that does not exist is worse than silence.
    error_type
        Exception class to raise, so each caller keeps its own typed error.

    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,):
        hint = "; use the batch variant for stacked (N, 6) states" if batch_hint else ""
        raise error_type(f"{name} must have shape (6,), got {array.shape}{hint}")
    if not np.all(np.isfinite(array)):
        raise error_type(f"{name} must be finite, got {array!r}")
    return array


def as_states_n6(
    value: npt.ArrayLike,
    name: str = "states",
    *,
    error_type: type[Exception] = ValueError,
) -> npt.NDArray[np.float64]:
    """Coerce ``value`` to a finite float64 array of shape (N, 6)."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6:
        raise error_type(f"{name} must have shape (N, 6), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise error_type(f"{name} must be finite")
    return array


def validate_positive(
    value: float,
    name: str,
    *,
    error_type: type[Exception] = ValueError,
) -> float:
    """Return ``value`` as a float, requiring it to be finite and strictly positive."""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise error_type(f"{name} must be finite and > 0, got {value!r}")
    return result


def validate_seed(
    seed: int | None,
    name: str = "seed",
    *,
    error_type: type[Exception] = ValueError,
) -> int | None:
    """Return ``seed`` as a non-negative int, or ``None`` if unset."""
    if seed is None:
        return None
    result = int(seed)
    if result < 0:
        raise error_type(f"{name} must be non-negative, got {seed!r}")
    return result


def validate_time_grid(
    times: npt.ArrayLike,
    name: str = "times_s",
    *,
    require_zero_start: bool = True,
    min_size: int = 1,
    error_type: type[Exception] = ValueError,
) -> npt.NDArray[np.float64]:
    """Coerce ``times`` to a finite, strictly increasing 1-D grid.

    **Strictly** increasing, not merely non-decreasing. ``scipy.integrate.solve_ivp``
    requires strict monotonicity, and an earlier version that validated ``diff >= 0`` let a
    repeated output time through, where it surfaced as scipy's own "Values in ``t_eval`` are
    not properly sorted" -- an error naming neither the offending argument nor the offending
    value -- while ``[0.0, 0.0]`` raised an ``AttributeError`` about a list having no shape.

    Parameters
    ----------
    times
        Output times, seconds.
    name
        Argument name, used in error messages so a caller can locate the fault.
    require_zero_start
        Whether the grid must begin at exactly 0.0. True for propagators, whose times are
        relative to an epoch; False for callers that pass absolute times.
    min_size
        Minimum number of output times. Propagators accept 1 (return the initial state);
        guidance grids need at least 2.
    error_type
        Exception class to raise, so each caller keeps its own typed error.

    """
    array = np.asarray(times, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise error_type(f"{name} must be a non-empty 1-D array, got shape {array.shape}")
    if array.size < min_size:
        raise error_type(
            f"{name} must contain at least {min_size} times, got {array.size} (shape {array.shape})"
        )
    if not np.all(np.isfinite(array)):
        raise error_type(f"{name} must be finite")
    if require_zero_start and array[0] != 0.0:
        raise error_type(f"{name} must start at 0.0, got {array[0]!r}")
    steps = np.diff(array)
    bad = np.flatnonzero(steps <= 0.0)
    if bad.size > 0:
        index = int(bad[0])
        raise error_type(
            f"{name} must be strictly increasing, but {name}[{index}] = {array[index]!r} is "
            f"followed by {name}[{index + 1}] = {array[index + 1]!r}"
        )
    return array
