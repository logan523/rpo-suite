r"""External-tool validation: ephemeris ingest, frame/time alignment, and comparison.

This module is the honest half of requirement group F-6 (``docs/project1/srs.md`` §2.6): it
ingests a time-tagged state ephemeris produced by *some other tool*, proves that both sides
are expressed in the same frame at the same instants, and reports the difference in a form
that names the physics. It does **not** drive STK, Astrogator, PySTK or GMAT, and it never
claims that it did. Whether the external side came from a real tool run or from a synthetic
reference is carried explicitly in :class:`Provenance` and stamped into every report; a
comparison record that cannot answer that question is worse than no record at all.

The equations
-------------
**Interpolation onto common epochs.** Two tools rarely tabulate the same instants, so one
side must be interpolated. Two schemes are provided.

*Lagrange* (default), on a window of :math:`k` samples surrounding the query time, evaluated
by Neville's recurrence

.. math::

    P_{i,j}(t) = \frac{(t - t_{i+j})\,P_{i,j-1}(t) - (t - t_{i})\,P_{i+1,j-1}(t)}
                      {t_{i} - t_{i+j}},

with truncation error

.. math::

    \lVert \varepsilon \rVert \;\sim\;
    \frac{\lVert \mathrm{d}^{k}\mathbf{r}/\mathrm{d}t^{k} \rVert}{k!}
    \prod_{i=0}^{k-1} \lvert t - t_i \rvert
    \;=\; \mathcal{O}(h^{k}).

For near-circular motion :math:`\mathrm{d}^{k}\mathbf{r}/\mathrm{d}t^{k} \sim n^{k} r`, so
with :math:`k = 8` and LEO values (:math:`n = 1.13 \times 10^{-3}` rad/s,
:math:`r = 6.8 \times 10^{6}` m) the error falls as :math:`h^{8}`. **Measured against the
analytic Kepler truth** in ``tests/test_validation.py``: :math:`3.9 \times 10^{-5}` m
worst-case at :math:`h = 60` s (set by the four intervals at each end, where the window
cannot be centred and the node product is an order of magnitude larger),
:math:`3 \times 10^{-6}` m mid-span, and :math:`1.5 \times 10^{-7}` m at :math:`h = 30` s --
the halving gains a measured factor of 255 against the predicted :math:`2^{8} = 256`. That
:math:`h^{8}` slope is why 8 points is the default and why the GMAT runbook asks for a 30 s
export rather than 60 s: at 60 s the interpolation error is within a factor of two of the
propagator difference being reported, and at 30 s it is three decades below it. None of this
is taken on faith; see the "measured, not assumed" paragraph below.

*Cubic Hermite*, which uses the tabulated velocity as well as position,

.. math::

    \mathbf{r}(t) = h_{00}\mathbf{r}_0 + h_{10} h \mathbf{v}_0
                  + h_{01}\mathbf{r}_1 + h_{11} h \mathbf{v}_1,
    \qquad s = (t - t_0)/h,

which is only :math:`\mathcal{O}(h^{4})` and therefore four decades *worse* than 8-point
Lagrange at practical ephemeris spacing (measured: 0.37 m at :math:`h = 60` s in LEO,
against :math:`3.9 \times 10^{-5}` m). It is offered because it is exactly
consistent between position and velocity, which matters if the ingested ephemeris is coarse
and only qualitative agreement is wanted.

**Measured, not assumed.** :func:`estimate_interpolation_error_m` decimates the ephemeris to
:math:`2h`, interpolates back onto the samples that were removed, and Richardson-extrapolates
the observed error down to the true spacing by :math:`\varepsilon_h \approx
\varepsilon_{2h} / 2^{p}`. This needs no truth model and no knowledge of the dynamics, so it
works on a real tool export. Every :class:`ComparisonReport` carries the resulting number and
refuses to call itself clean when the interpolation error is not comfortably below the
difference being reported.

**Radial / along-track / cross-track breakdown.** With the *internal* state defining the
target, :func:`rpo_core.frames.hill_basis` gives the rotation :math:`R` whose rows are
:math:`\hat{x} = \hat{r}`, :math:`\hat{z} = \hat{h}`, :math:`\hat{y} = \hat{z}\times\hat{x}`
(``docs/conventions.md``). The difference is decomposed as

.. math::

    \delta\mathbf{r}_{\text{RIC}} = R\,(\mathbf{r}_{\text{ext}} - \mathbf{r}_{\text{int}}),
    \qquad
    \delta\mathbf{v}_{\text{RIC}} = R\,(\mathbf{v}_{\text{ext}} - \mathbf{v}_{\text{int}}).

Reporting only :math:`\lVert\delta\mathbf{r}\rVert` hides the physics. A difference in
semi-major axis, mean motion, or epoch appears first and largest in **along-track**, growing
secularly; a difference in force model or in the frame's pole appears in **cross-track**; a
genuine radial disagreement in an otherwise-matching orbit usually means the two sides are
not the same orbit at all. The signed mean of each component is reported alongside the
magnitudes precisely so that a secular bias is distinguishable from noise.

The velocity difference is deliberately the **rotated inertial** difference, not the
transport-theorem rotating-frame derivative that ``docs/conventions.md`` mandates for a
physical *relative state*. The two are different objects: a comparison residual is an error
vector, and adding :math:`-\boldsymbol{\omega} \times \delta\mathbf{r}` would fold position
error into the velocity metric at roughly :math:`1.1 \times 10^{-3}` m/s per metre of
position difference in LEO, making a pure position error read as a velocity error. Callers
who want the rotating-frame quantity pass ``rotating_frame_velocity=True`` and get it, with
that coupling documented.

**Epoch sensitivity.** An unresolved epoch offset is the single cheapest way to manufacture a
large fake disagreement:

.. math:: \lvert \delta \mathbf{r}_{\text{along-track}} \rvert \approx \lvert \Delta t \rvert\, v .

At LEO orbital speed 7.66 km/s, one second of epoch error is 7.66 km, and the 37-second
TAI-UTC difference is 283 km. This module therefore refuses to guess a time-scale offset it
does not know exactly; see :func:`along_track_error_from_time_offset_m`.

**Frame-approximation budget.** This repository's inertial frame is GCRF-*approximated*:
precession, nutation, polar motion, and the frame-tie are neglected
(``docs/conventions.md``). Against a tool that models them, the induced position difference
is estimated as

.. math::

    \delta r \;\approx\; r \left[
        \dot{\theta}_{p}\, \Delta t
        + \min(\dot{\theta}_{n}\,\Delta t,\ \theta_{n,\max})
        + \theta_{\text{bias}}
    \right],

with general precession in right ascension :math:`\dot{\theta}_{p} = 50.29''`/Julian year,
a nutation drift bound :math:`\dot{\theta}_{n} \approx 0.15''`/day (the fastest-moving
principal terms) saturating at the :math:`17.2''` amplitude of the 18.6-year term, and a
constant :math:`23` mas frame-bias term where the external frame is EME2000/J2000 rather than
GCRF. **Over one day in LEO this is 9.5 m** (4.54 m of precession, 4.94 m of nutation, at
:math:`r = 6.798 \times 10^{6}` m), and 66 m over a week. Both terms grow while the nutation
bound is unsaturated, so the rate is about 9.5 m/day for the first 115 days and 4.54 m/day
(precession alone) after that.
That is far larger than the integrator agreement this suite achieves internally -- measured
:math:`8.4 \times 10^{-5}` m over 24 h against an independent analytic solution -- so for any
comparison spanning more than a few hours the frame approximation -- not the propagator --
is expected to dominate the disagreement. Every report carries this budget
(:func:`estimate_frame_tie_error_m`) so that a difference smaller than it is not misread as
a dynamics result.

Validity
--------
* **Assumes** the ingested file follows the documented ``rpo-ephemeris/1.0`` contract
  (see :func:`read_ephemeris`). Nothing here sniffs or guesses a foreign format; an
  unrecognised header is an error, never a best-effort parse.
* **Assumes** both sides describe the same central body and the same object. Nothing checks
  that, and nothing can.
* **Neglects** precession, nutation, polar motion and the frame tie, inheriting the
  repository convention. The magnitude is estimated, not assumed negligible, and the estimate
  is an order-of-magnitude bound from published rates -- it is *not* a substitute for a
  rigorous IAU-2006/2000A implementation.
* **Refuses** rather than approximates for any date-dependent frame (ITRF, TEME, MOD, TOD):
  converting out of those requires exactly the Earth-orientation model that is neglected.
* **Refuses** rather than approximates for time-scale conversions that need a leap-second
  table (UTC, UT1) or a relativistic series (TDB). Only TAI-TT is applied, and only because
  it is exact by definition.
* The velocity-column consistency check is a **gross-blunder detector** (unit factors,
  swapped or sign-flipped columns), not a precision check; its threshold is set from
  measurement, see :func:`read_ephemeris`.
* Nothing in this module has been exercised against a real STK, Astrogator or GMAT run on
  this machine. See ``optional_stk/README.md``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from .constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from .exceptions import RpoCoreError
from .frames import hill_basis

__all__ = [
    "ARCSEC_RAD",
    "DEFAULT_INTERPOLATION_MARGIN",
    "DEFAULT_LAGRANGE_POINTS",
    "DEFAULT_MAX_STEP_RATIO",
    "DEFAULT_VELOCITY_REL_TOL",
    "EPHEMERIS_FORMAT",
    "REQUIRED_COLUMNS",
    "REQUIRED_HEADER_KEYS",
    "AlignmentReport",
    "ComparisonReport",
    "ComponentStatistics",
    "Ephemeris",
    "EphemerisFormatError",
    "EphemerisGapError",
    "EphemerisTimeError",
    "EphemerisUnitError",
    "Epoch",
    "EpochMismatchError",
    "FrameMismatchError",
    "InterpolationMethod",
    "InterpolationRangeError",
    "Provenance",
    "ReferenceFrame",
    "TimeScale",
    "ValidationError",
    "align_ephemeris",
    "along_track_error_from_time_offset_m",
    "check_alignment",
    "compare_ephemerides",
    "ephemeris_from_states",
    "estimate_frame_tie_error_m",
    "estimate_interpolation_error_m",
    "interpolate_states",
    "read_ephemeris",
    "write_comparison_report",
    "write_ephemeris",
]


# --------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------


class ValidationError(RpoCoreError):
    """Base class for every failure raised by the external-validation layer.

    Subclasses ``RpoCoreError`` so that a caller catching the package base class still
    catches ingest and alignment failures. Nothing in this module degrades to a best-effort
    answer: a comparison that cannot be trusted is refused, because a bogus validation
    number is more expensive than a missing one.
    """


class EphemerisFormatError(ValidationError, ValueError):
    """Raised when an ephemeris file does not follow the documented column contract.

    Covers a missing or malformed header, missing required columns, a row with the wrong
    number of fields, and non-finite values. A malformed row is never skipped: silently
    dropping one line turns a truncated export into a clean-looking comparison over a
    shorter arc.
    """


class EphemerisTimeError(ValidationError, ValueError):
    """Raised when the time column is not strictly increasing.

    Duplicated or reversed epochs are treated identically to any other corruption. A
    duplicate breaks every interpolation window (zero denominator in Neville's recurrence);
    a reversal usually means two exports were concatenated.
    """


class EphemerisGapError(EphemerisTimeError):
    """Raised when the sample spacing jumps, i.e. the ephemeris has a dropout.

    A gap is not merely inconvenient. Interpolating across one silently reports the error of
    a much coarser grid while the report still quotes the nominal step, and the arc that was
    lost is exactly the arc a reader assumes was compared.
    """


class EphemerisUnitError(ValidationError, ValueError):
    """Raised for an unknown unit token, or for data inconsistent with the declared units.

    The metre/kilometre confusion is the classic ephemeris-comparison blunder and it is
    worth a factor of 1000, so the declared unit is checked against the magnitudes actually
    present rather than trusted.
    """


class FrameMismatchError(ValidationError, ValueError):
    """Raised when two ephemerides are not in the same frame and no honest tie exists.

    Comparing across frames without a rotation is how a several-kilometre "validation
    disagreement" gets published. Where a tie is only approximate it must be requested
    explicitly and its magnitude is recorded in the report.
    """


class EpochMismatchError(ValidationError, ValueError):
    """Raised when two ephemerides cannot be resolved to a common absolute instant.

    Either the time scales differ and the offset is not exactly known (UTC and UT1 need a
    leap-second/EOP table; TDB needs a relativistic series), or the arcs do not overlap
    after alignment. The message states the along-track cost of the offset in metres,
    because that is the number that decides whether it matters.
    """


class InterpolationRangeError(ValidationError, ValueError):
    """Raised when a query time falls outside the tabulated span.

    Extrapolation is refused. A Lagrange window extrapolated by even one step diverges far
    faster than it interpolates, and the resulting difference looks like a physics result.
    """


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class TimeScale(Enum):
    """Time scale in which an epoch is expressed.

    Only the TAI-TT relationship is applied automatically, because it is exact by
    definition (``TT = TAI + 32.184 s``). UTC and UT1 need a leap-second / Earth-orientation
    table and TDB needs a relativistic series; this package ships none of those, so those
    conversions must be supplied by the caller or the comparison is refused.
    """

    TAI = "TAI"
    TT = "TT"
    TDB = "TDB"
    UTC = "UTC"
    UT1 = "UT1"


class ReferenceFrame(Enum):
    """Declared reference frame of an ephemeris.

    ``GCRF_APPROX`` is this repository's own frame: GCRF-approximated with precession,
    nutation, polar motion and frame-tie neglected. It is a *different* frame from GCRF and
    is named as such so that the approximation cannot be lost in a comparison.
    """

    GCRF_APPROX = "GCRF_APPROX"
    GCRF = "GCRF"
    ICRF = "ICRF"
    EME2000 = "EME2000"
    TOD = "TOD"
    MOD = "MOD"
    TEME = "TEME"
    ITRF = "ITRF"


class Provenance(Enum):
    """Where an ephemeris came from.

    ``TOOL_RUN`` means a licensed external tool was actually executed and produced the file.
    ``SYNTHETIC_REFERENCE`` means the file was generated inside this project (for example by
    an analytic Kepler solution) and is therefore *not* independent-tool validation, however
    useful it is. ``UNKNOWN`` is the default for a file that did not declare, and it is never
    silently upgraded.
    """

    TOOL_RUN = "tool_run"
    SYNTHETIC_REFERENCE = "synthetic_reference"
    UNKNOWN = "unknown"


class InterpolationMethod(Enum):
    """Scheme used to place one ephemeris onto another's epochs."""

    LAGRANGE = "lagrange"
    HERMITE = "hermite"


# --------------------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------------------

#: Radians per arcsecond.
ARCSEC_RAD: Final[float] = math.pi / (180.0 * 3600.0)

#: Format token that must appear in an ephemeris header.
EPHEMERIS_FORMAT: Final[str] = "rpo-ephemeris/1.0"

#: Header keys every ephemeris must declare. An ephemeris without a stated epoch, frame and
#: time scale is uncomparable, so its absence is an error rather than a default.
REQUIRED_HEADER_KEYS: Final[tuple[str, ...]] = (
    "format",
    "frame",
    "time_scale",
    "epoch",
    "position_unit",
    "velocity_unit",
)

#: Column names every ephemeris must provide. Matched by *name*, not position, so a
#: reordered export is read correctly and a missing column is named in the error.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("t", "x", "y", "z", "vx", "vy", "vz")

#: Accepted position units and their factor to metres.
_POSITION_UNITS: Final[dict[str, float]] = {"m": 1.0, "km": 1.0e3}

#: Accepted velocity units and their factor to metres per second.
_VELOCITY_UNITS: Final[dict[str, float]] = {"m/s": 1.0, "km/s": 1.0e3, "m/sec": 1.0}

#: Exactly-known time-scale offsets, seconds to add to convert ``from`` into ``to``.
_EXACT_SCALE_OFFSETS_S: Final[dict[tuple[TimeScale, TimeScale], float]] = {
    (TimeScale.TAI, TimeScale.TT): 32.184,
    (TimeScale.TT, TimeScale.TAI): -32.184,
}

#: Frames whose orientation is identical to ``GCRF_APPROX`` up to the neglected corrections.
#: Anything not listed here is date-dependent and cannot be rotated without the very
#: Earth-orientation model this repository declines to implement.
_APPROXIMATE_TIE_FRAMES: Final[frozenset[ReferenceFrame]] = frozenset(
    {ReferenceFrame.GCRF_APPROX, ReferenceFrame.GCRF, ReferenceFrame.ICRF, ReferenceFrame.EME2000}
)

#: Constant frame-bias angle between EME2000/J2000 and the GCRF pole+origin, radians.
#: ~23 mas from the IAU-2006 bias terms (xi0 = -16.617 mas, eta0 = -6.819 mas,
#: dalpha0 = -14.6 mas); constant in time, ~0.76 m at LEO radius.
_FRAME_BIAS_RAD: Final[float] = 23.0e-3 * ARCSEC_RAD

#: General precession in right ascension, 50.2879 arcsec per Julian year, rad/s.
_PRECESSION_RATE_RAD_S: Final[float] = 50.2879 * ARCSEC_RAD / (365.25 * 86400.0)

#: Bound on the rate of change of nutation, ~0.15 arcsec/day. Dominated by the 13.66-day
#: term (0.227 arcsec amplitude) and the semiannual term (1.32 arcsec, 182.6 days).
_NUTATION_RATE_RAD_S: Final[float] = 0.15 * ARCSEC_RAD / 86400.0

#: Amplitude of the 18.6-year principal nutation term; the drift estimate saturates here.
_NUTATION_AMPLITUDE_RAD: Final[float] = 17.2 * ARCSEC_RAD

#: Reference radius for frame-error and epoch-error estimates: the SRS baseline 420 km LEO.
_REFERENCE_RADIUS_M: Final[float] = R_EARTH_EQUATORIAL_M + 420.0e3

#: Reference speed for epoch-error estimates: circular speed at ``_REFERENCE_RADIUS_M``.
_REFERENCE_SPEED_M_S: Final[float] = math.sqrt(MU_EARTH_M3_S2 / _REFERENCE_RADIUS_M)

#: Plausible range of geocentric radius, metres. Wide enough for LEO through cislunar,
#: narrow enough that a metre/kilometre confusion (a factor of 1000) cannot hide inside it.
_PLAUSIBLE_RADIUS_M: Final[tuple[float, float]] = (1.0e6, 1.0e9)

#: Default Lagrange window size. See the module docstring for why 8 and not 4 or 16.
DEFAULT_LAGRANGE_POINTS: Final[int] = 8

#: Default ratio of largest to median step above which the ephemeris is declared gapped.
DEFAULT_MAX_STEP_RATIO: Final[float] = 1.5

#: Relative tolerance for the velocity-column consistency check. Measured: at 60 s spacing
#: in LEO a correct ephemeris disagrees with its own central difference by ~7.6e-4 relative
#: (the O(h^2 n^3 r) truncation of the difference itself), so 1e-2 leaves a factor of ~13 of
#: headroom while still catching a 1000x unit error, a swapped column pair, or a sign flip.
DEFAULT_VELOCITY_REL_TOL: Final[float] = 1.0e-2

#: Minimum ratio of reported difference to estimated interpolation error for a comparison to
#: call itself interpolation-clean. Two decades: below that, the number being reported is
#: contaminated by the machinery producing it.
DEFAULT_INTERPOLATION_MARGIN: Final[float] = 100.0

_SECONDS_PER_DAY: Final[float] = 86400.0
_MJD_AT_UNIX_EPOCH: Final[int] = 40587

_ISO_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<year>[+-]?\d{4,})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"[T ](?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}(?:\.\d+)?)Z?$"
)


# --------------------------------------------------------------------------------------
# Calendar arithmetic (pure integer; no timezone database, no ambiguity)
# --------------------------------------------------------------------------------------


def _days_from_civil(year: int, month: int, day: int) -> int:
    """Return days since 1970-01-01 for a proleptic Gregorian date (Hinnant's algorithm)."""
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _civil_from_days(days: int) -> tuple[int, int, int]:
    """Invert :func:`_days_from_civil`, returning ``(year, month, day)``."""
    z = days + 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    return y + (1 if m <= 2 else 0), m, d


# --------------------------------------------------------------------------------------
# Epoch
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Epoch:
    """An absolute instant, stored as an integer day plus seconds of day.

    Two-part storage is deliberate. A modern MJD stored as a single double resolves to about
    1 microsecond, which at LEO speed is 7.7 mm of along-track -- small, but it is a
    resolution limit hiding inside a comparison whose whole purpose is to attribute
    millimetres. Splitting the day off keeps epoch arithmetic accurate to ~1.5e-11 s.

    Attributes
    ----------
    mjd_day
        Integer Modified Julian Day number.
    seconds_of_day
        Seconds elapsed in that day, in ``[0, 86400)``.
    scale
        Time scale the epoch is expressed in.

    Raises
    ------
    ValueError
        If ``seconds_of_day`` is non-finite or outside ``[0, 86400)``.

    """

    mjd_day: int
    seconds_of_day: float
    scale: TimeScale

    def __post_init__(self) -> None:
        """Reject a seconds-of-day value that is not a normalised, finite time of day."""
        if not math.isfinite(self.seconds_of_day):
            raise ValueError(f"seconds_of_day must be finite, got {self.seconds_of_day!r}")
        if not 0.0 <= self.seconds_of_day < _SECONDS_PER_DAY:
            raise ValueError(
                "seconds_of_day must lie in [0, 86400), got "
                f"{self.seconds_of_day!r}; construct via Epoch.from_mjd or Epoch.shifted, "
                "which normalise"
            )

    @classmethod
    def from_mjd(cls, mjd: float, scale: TimeScale) -> Epoch:
        """Build an epoch from a (possibly fractional) Modified Julian Date.

        Parameters
        ----------
        mjd
            Modified Julian Date.
        scale
            Time scale of ``mjd``.

        Returns
        -------
        Epoch
            Normalised epoch.

        Raises
        ------
        ValueError
            If ``mjd`` is not finite.

        """
        if not math.isfinite(mjd):
            raise ValueError(f"mjd must be finite, got {mjd!r}")
        day = math.floor(mjd)
        return cls(int(day), (mjd - day) * _SECONDS_PER_DAY, scale)

    @classmethod
    def from_iso(cls, text: str, scale: TimeScale) -> Epoch:
        """Parse ``YYYY-MM-DDThh:mm:ss[.fff]`` into an epoch.

        No timezone handling and no leap-second handling: the string is a reading on the
        stated ``scale``, and a trailing ``Z`` is accepted but means nothing beyond "no
        offset attached". A seconds field of 60 (a leap second) is rejected rather than
        wrapped, because wrapping it would move the epoch by a second, i.e. 7.7 km.

        Parameters
        ----------
        text
            Timestamp string.
        scale
            Time scale the timestamp is a reading on.

        Returns
        -------
        Epoch
            Parsed epoch.

        Raises
        ------
        ValueError
            If the string does not match the accepted pattern or contains an out-of-range
            field.

        """
        match = _ISO_PATTERN.match(text.strip())
        if match is None:
            raise ValueError(
                f"epoch {text!r} is not of the form YYYY-MM-DDThh:mm:ss[.fff]; "
                "an ambiguous timestamp is not parsed on a guess"
            )
        year = int(match["year"])
        month = int(match["month"])
        day = int(match["day"])
        hour = int(match["hour"])
        minute = int(match["minute"])
        second = float(match["second"])
        if not 1 <= month <= 12:
            raise ValueError(f"epoch {text!r} has month {month}, expected 1-12")
        if not 1 <= day <= 31:
            raise ValueError(f"epoch {text!r} has day {day}, expected 1-31")
        if hour > 23 or minute > 59 or second >= 60.0:
            raise ValueError(
                f"epoch {text!r} has an out-of-range time field "
                f"({hour:02d}:{minute:02d}:{second:09.6f}); a leap second is not wrapped "
                "silently because one second is ~7.7 km of along-track in LEO"
            )
        return cls(
            _days_from_civil(year, month, day) + _MJD_AT_UNIX_EPOCH,
            hour * 3600.0 + minute * 60.0 + second,
            scale,
        )

    def to_iso(self) -> str:
        """Return the epoch as ``YYYY-MM-DDThh:mm:ss.ssssss`` (no scale suffix)."""
        year, month, day = _civil_from_days(self.mjd_day - _MJD_AT_UNIX_EPOCH)
        total = self.seconds_of_day
        hour = int(total // 3600.0)
        minute = int((total - hour * 3600.0) // 60.0)
        second = total - hour * 3600.0 - minute * 60.0
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:09.6f}"

    def mjd(self) -> float:
        """Return the epoch as a single float MJD (lossy at the microsecond level)."""
        return self.mjd_day + self.seconds_of_day / _SECONDS_PER_DAY

    def shifted(self, seconds: float) -> Epoch:
        """Return this epoch advanced by ``seconds``, renormalising the day.

        Raises
        ------
        ValueError
            If ``seconds`` is not finite.

        """
        if not math.isfinite(seconds):
            raise ValueError(f"seconds must be finite, got {seconds!r}")
        total = self.seconds_of_day + seconds
        whole_days = math.floor(total / _SECONDS_PER_DAY)
        return Epoch(
            self.mjd_day + int(whole_days),
            total - whole_days * _SECONDS_PER_DAY,
            self.scale,
        )

    def offset_s(self, other: Epoch) -> float:
        """Return ``self - other`` in seconds. Both must be on the same time scale.

        Raises
        ------
        EpochMismatchError
            If the two epochs are on different time scales. Differencing across scales
            without conversion is the mistake this whole module exists to prevent.

        """
        if self.scale is not other.scale:
            raise EpochMismatchError(
                f"cannot difference epochs on different time scales "
                f"({self.scale.value} and {other.scale.value}); convert with "
                "Epoch.in_scale first"
            )
        return (self.mjd_day - other.mjd_day) * _SECONDS_PER_DAY + (
            self.seconds_of_day - other.seconds_of_day
        )

    def in_scale(self, target: TimeScale, *, offset_s: float | None = None) -> Epoch:
        """Convert this epoch to ``target``.

        Only ``TAI <-> TT`` is applied automatically (exactly 32.184 s, by definition). Any
        other pairing requires ``offset_s`` from the caller, who is then on record as having
        supplied it.

        Parameters
        ----------
        target
            Time scale to convert to.
        offset_s
            Seconds to add to this epoch's reading to obtain the ``target`` reading. Ignored
            when the pairing is exactly known.

        Returns
        -------
        Epoch
            Epoch on ``target``.

        Raises
        ------
        EpochMismatchError
            If the conversion is neither exactly known nor supplied.

        """
        if self.scale is target:
            return self
        exact = _EXACT_SCALE_OFFSETS_S.get((self.scale, target))
        if exact is not None:
            return replace(self.shifted(exact), scale=target)
        if offset_s is not None:
            return replace(self.shifted(offset_s), scale=target)
        raise EpochMismatchError(
            f"no exact offset is defined from {self.scale.value} to {target.value}: "
            "UTC and UT1 need a leap-second/Earth-orientation table and TDB needs a "
            "relativistic series, neither of which this package ships. Supply "
            "time_scale_offset_s explicitly -- guessing 37 s of TAI-UTC wrongly is "
            f"{along_track_error_from_time_offset_m(37.0):.3e} m of along-track difference"
        )


# --------------------------------------------------------------------------------------
# Diagnostic helpers
# --------------------------------------------------------------------------------------


def along_track_error_from_time_offset_m(
    offset_s: float, speed_m_s: float = _REFERENCE_SPEED_M_S
) -> float:
    """Return the along-track position difference produced by an epoch offset.

    To first order a time-tag error simply slides the spacecraft along its own track, so the
    difference is ``|offset| * speed``. This is the number that decides whether an epoch
    discrepancy matters: at LEO speed one second is 7.66 km, which is why this module refuses
    to guess time-scale offsets.

    Parameters
    ----------
    offset_s
        Epoch offset, seconds.
    speed_m_s
        Orbital speed, metres per second. Defaults to circular speed at the SRS baseline
        420 km LEO altitude.

    Returns
    -------
    float
        Along-track difference in metres.

    Raises
    ------
    ValueError
        If either argument is non-finite, or ``speed_m_s`` is negative.

    """
    if not math.isfinite(offset_s) or not math.isfinite(speed_m_s):
        raise ValueError(f"offset_s and speed_m_s must be finite, got {offset_s!r}, {speed_m_s!r}")
    if speed_m_s < 0.0:
        raise ValueError(f"speed_m_s must be >= 0, got {speed_m_s!r}")
    return abs(offset_s) * speed_m_s


def estimate_frame_tie_error_m(
    elapsed_s: float,
    radius_m: float = _REFERENCE_RADIUS_M,
    *,
    include_frame_bias: bool = False,
) -> float:
    """Estimate the position difference caused by this repository's GCRF approximation.

    ``docs/conventions.md`` neglects precession, nutation, polar motion and the frame tie.
    Against a tool that models them, the two frames rotate apart, and a rotation ``theta``
    displaces a point at radius ``r`` by ``r * theta``. The estimate sums a secular
    precession term, a nutation term that saturates at the amplitude of the 18.6-year
    principal term, and optionally the constant EME2000-to-GCRF frame bias.

    **Over one day at LEO radius this returns 9.5 m**, of which 4.54 m is precession and
    4.94 m nutation, and 66 m over a week. The rate is about 9.5 m/day while both terms
    grow; the nutation bound saturates after 17.2/0.15 = 115 days, leaving 4.54 m/day of
    precession from then on. Polar motion is
    excluded on purpose: it enters the terrestrial-to-inertial tie only, so it contributes
    nothing to an inertial-to-inertial comparison unless the external states were routed
    through an Earth-fixed frame, in which case up to 0.3 arcsec (about 10 m) must be added.

    This is an order-of-magnitude budget from published rates, **not** a rigorous
    IAU-2006/2000A difference. Its purpose is to stop a difference smaller than the frame
    approximation from being reported as a dynamics result.

    Parameters
    ----------
    elapsed_s
        Time since the frames were coincident (in practice, the ephemeris epoch), seconds.
    radius_m
        Geocentric radius, metres.
    include_frame_bias
        Add the constant ~23 mas EME2000/J2000-to-GCRF bias (about 0.76 m at LEO).

    Returns
    -------
    float
        Estimated position difference, metres.

    Raises
    ------
    ValueError
        If ``elapsed_s`` is non-finite or negative, or ``radius_m`` is non-finite or
        non-positive.

    """
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError(f"elapsed_s must be finite and >= 0, got {elapsed_s!r}")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError(f"radius_m must be finite and > 0, got {radius_m!r}")
    theta = _PRECESSION_RATE_RAD_S * elapsed_s
    theta += min(_NUTATION_RATE_RAD_S * elapsed_s, _NUTATION_AMPLITUDE_RAD)
    if include_frame_bias:
        theta += _FRAME_BIAS_RAD
    return theta * radius_m


# --------------------------------------------------------------------------------------
# Ephemeris
# --------------------------------------------------------------------------------------


def _validate_arrays(
    times_s: npt.NDArray[np.float64],
    states: npt.NDArray[np.float64],
    *,
    max_step_ratio: float | None,
    origin: str,
) -> None:
    """Run every structural check an ephemeris must pass, raising on the first failure."""
    if times_s.ndim != 1 or times_s.size == 0:
        raise EphemerisFormatError(
            f"{origin}: times must be a non-empty 1-D array, got shape {times_s.shape}"
        )
    if states.ndim != 2 or states.shape != (times_s.size, 6):
        raise EphemerisFormatError(
            f"{origin}: states must have shape ({times_s.size}, 6), got {states.shape}"
        )
    if not np.all(np.isfinite(times_s)):
        bad = int(np.flatnonzero(~np.isfinite(times_s))[0])
        raise EphemerisFormatError(f"{origin}: time at index {bad} is not finite")
    if not np.all(np.isfinite(states)):
        row = int(np.flatnonzero(~np.all(np.isfinite(states), axis=1))[0])
        raise EphemerisFormatError(
            f"{origin}: state at index {row} (t = {times_s[row]!r} s) is not finite"
        )

    steps = np.diff(times_s)
    if steps.size and np.any(steps <= 0.0):
        bad = int(np.flatnonzero(steps <= 0.0)[0])
        raise EphemerisTimeError(
            f"{origin}: times must be strictly increasing, but t[{bad}] = {times_s[bad]!r} s "
            f"is followed by t[{bad + 1}] = {times_s[bad + 1]!r} s "
            f"(step {steps[bad]!r} s)"
        )
    if max_step_ratio is not None and steps.size >= 2:
        median_step = float(np.median(steps))
        largest = float(np.max(steps))
        if largest > max_step_ratio * median_step:
            bad = int(np.argmax(steps))
            raise EphemerisGapError(
                f"{origin}: sample spacing jumps at index {bad}: step {largest:.6g} s is "
                f"{largest / median_step:.3f}x the median {median_step:.6g} s (limit "
                f"{max_step_ratio:g}x). The arc from t = {times_s[bad]!r} s to "
                f"{times_s[bad + 1]!r} s is missing; interpolating across it would report "
                "the accuracy of the nominal step over a gap that does not have it"
            )


def _check_unit_plausibility(states: npt.NDArray[np.float64], declared: str, origin: str) -> None:
    """Reject data whose magnitudes contradict the declared position unit."""
    radii = np.linalg.norm(states[:, :3], axis=1)
    smallest = float(np.min(radii))
    largest = float(np.max(radii))
    low, high = _PLAUSIBLE_RADIUS_M
    if smallest < low or largest > high:
        raise EphemerisUnitError(
            f"{origin}: geocentric radius spans {smallest:.6g} to {largest:.6g} m after "
            f"applying the declared position_unit {declared!r}, outside the plausible range "
            f"{low:.3g} to {high:.3g} m. The declared unit is almost certainly wrong by a "
            "factor of 1000 (m vs km), which is the classic ephemeris-comparison blunder"
        )


def _check_velocity_consistency(
    times_s: npt.NDArray[np.float64],
    states: npt.NDArray[np.float64],
    *,
    rel_tol: float,
    origin: str,
) -> None:
    """Cross-check the velocity columns against a central difference of the position columns.

    A gross-blunder detector, not a precision check: it catches unit factors, swapped or
    sign-flipped columns, and position/velocity columns in the wrong order, all of which
    produce a disagreement of order 100 % or more.
    """
    if times_s.size < 3:
        return
    dt = times_s[2:] - times_s[:-2]
    finite_difference = (states[2:, :3] - states[:-2, :3]) / dt[:, None]
    tabulated = states[1:-1, 3:]
    speed = np.linalg.norm(tabulated, axis=1)
    scale = np.maximum(speed, np.finfo(np.float64).tiny)
    residual = np.linalg.norm(finite_difference - tabulated, axis=1) / scale
    worst = int(np.argmax(residual))
    if float(residual[worst]) > rel_tol:
        raise EphemerisUnitError(
            f"{origin}: the velocity columns disagree with a central difference of the "
            f"position columns by {float(residual[worst]):.4g} relative at index "
            f"{worst + 1} (t = {times_s[worst + 1]!r} s), above the {rel_tol:g} limit. "
            "That size of disagreement means a unit factor, a swapped or sign-flipped "
            "column, or position and velocity blocks in the wrong order -- not truncation"
        )


@dataclass(frozen=True, slots=True)
class Ephemeris:
    """A time-tagged state history with everything needed to compare it against another.

    An ephemeris that does not know its own frame, epoch, time scale and provenance cannot
    be compared honestly, so all four are required members rather than optional metadata.

    Attributes
    ----------
    epoch
        Absolute epoch that ``times_s`` is measured from.
    times_s
        Strictly increasing seconds from ``epoch``, shape ``(N,)``.
    states
        Inertial states ``[r(3), v(3)]`` in metres and metres per second, shape ``(N, 6)``.
        Always SI: unit conversion happens at ingest, never here.
    frame
        Declared reference frame.
    provenance
        Whether these states came from a real external tool run or from a synthetic
        reference generated inside this project.
    source
        Free text naming the producer, e.g. ``"GMAT R2022a EphemerisFile"``.
    metadata
        Any additional header keys read from the file, verbatim.

    """

    epoch: Epoch
    times_s: npt.NDArray[np.float64]
    states: npt.NDArray[np.float64]
    frame: ReferenceFrame
    provenance: Provenance
    source: str
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze the arrays so a shared ephemeris cannot be mutated behind a caller's back."""
        times = np.ascontiguousarray(self.times_s, dtype=np.float64)
        states = np.ascontiguousarray(self.states, dtype=np.float64)
        times.flags.writeable = False
        states.flags.writeable = False
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def size(self) -> int:
        """Number of tabulated states."""
        return int(self.times_s.size)

    @property
    def span_s(self) -> float:
        """Duration covered, seconds."""
        return float(self.times_s[-1] - self.times_s[0])

    @property
    def median_step_s(self) -> float:
        """Median sample spacing, seconds. ``nan`` for a single-sample ephemeris."""
        if self.times_s.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.times_s)))

    def epoch_of(self, index: int) -> Epoch:
        """Return the absolute epoch of sample ``index``."""
        return self.epoch.shifted(float(self.times_s[index]))


def ephemeris_from_states(
    times_s: npt.ArrayLike,
    states: npt.ArrayLike,
    *,
    epoch: Epoch,
    frame: ReferenceFrame,
    provenance: Provenance = Provenance.UNKNOWN,
    source: str = "",
    metadata: Mapping[str, str] | None = None,
    max_step_ratio: float | None = DEFAULT_MAX_STEP_RATIO,
) -> Ephemeris:
    """Build an :class:`Ephemeris` from arrays, applying the same checks as :func:`read_ephemeris`.

    Parameters
    ----------
    times_s
        Seconds from ``epoch``, shape ``(N,)``, strictly increasing.
    states
        Inertial states ``[r(3), v(3)]`` in metres and metres per second, shape ``(N, 6)``.
    epoch
        Absolute epoch of ``times_s[0] == 0``.
    frame
        Declared reference frame.
    provenance
        Where the states came from. Defaults to ``UNKNOWN`` and is never silently upgraded.
    source
        Free text naming the producer.
    metadata
        Extra key/value pairs to carry through to any written file.
    max_step_ratio
        Largest permitted ratio of a step to the median step before the ephemeris is
        declared gapped. ``None`` disables the check for a deliberately non-uniform grid.

    Returns
    -------
    Ephemeris
        Validated ephemeris.

    Raises
    ------
    EphemerisFormatError
        On a wrong shape or a non-finite value.
    EphemerisTimeError
        If times are not strictly increasing.
    EphemerisGapError
        If the sample spacing jumps.

    """
    times = np.asarray(times_s, dtype=np.float64)
    array = np.asarray(states, dtype=np.float64)
    _validate_arrays(times, array, max_step_ratio=max_step_ratio, origin="ephemeris_from_states")
    return Ephemeris(
        epoch=epoch,
        times_s=times,
        states=array,
        frame=frame,
        provenance=provenance,
        source=source,
        metadata=dict(metadata or {}),
    )


def _split_fields(line: str) -> list[str]:
    """Split a data line on commas if present, otherwise on whitespace."""
    if "," in line:
        return [field.strip() for field in line.split(",")]
    return line.split()


def read_ephemeris(
    path: str | Path,
    *,
    max_step_ratio: float | None = DEFAULT_MAX_STEP_RATIO,
    check_units: bool = True,
    velocity_rel_tol: float | None = DEFAULT_VELOCITY_REL_TOL,
) -> Ephemeris:
    r"""Read a state ephemeris in the ``rpo-ephemeris/1.0`` contract.

    The contract, in full::

        # format: rpo-ephemeris/1.0
        # frame: GCRF_APPROX
        # time_scale: TAI
        # epoch: 2026-03-01T00:00:00.000000
        # position_unit: m
        # velocity_unit: m/s
        # provenance: tool_run
        # source: GMAT R2022a EphemerisFile
        t,x,y,z,vx,vy,vz
        0.0,6798137.0,0.0,0.0,0.0,4739.7,5981.3
        60.0,...

    Comment lines begin with ``#`` and carry ``key: value`` pairs. Every key in
    :data:`REQUIRED_HEADER_KEYS` must be present; ``provenance`` and ``source`` are optional
    and default to ``unknown`` and the empty string. The first non-comment line names the
    columns, which are matched **by name** so a reordered export still reads correctly and a
    missing column is named in the error. Rows may be comma- or whitespace-separated.
    Position may be declared ``m`` or ``km``, velocity ``m/s`` or ``km/s``; conversion to SI
    happens here, at the I/O boundary, and nowhere else.

    Every check below is a refusal, not a warning, because the failure mode this module
    exists to prevent is a plausible-looking comparison built on a misread file:

    * missing header key, missing column, or a row with the wrong field count;
    * a non-finite value anywhere;
    * times that are not strictly increasing (a duplicate breaks every interpolation
      window; a reversal usually means two exports were concatenated);
    * a step more than ``max_step_ratio`` times the median, i.e. a dropout;
    * a declared unit that is not recognised, or one contradicted by the magnitudes present
      (a metre/kilometre confusion moves the geocentric radius by 1000x);
    * velocity columns that disagree with a central difference of the position columns by
      more than ``velocity_rel_tol``.

    Parameters
    ----------
    path
        File to read.
    max_step_ratio
        Gap threshold as a multiple of the median step. ``None`` disables.
    check_units
        Run the radius-magnitude plausibility check.
    velocity_rel_tol
        Threshold for the velocity/position cross-check. ``None`` disables.

    Returns
    -------
    Ephemeris
        Validated ephemeris in SI units.

    Raises
    ------
    EphemerisFormatError
        On any structural problem: header, columns, field counts, non-finite values.
    EphemerisTimeError
        If the time column is not strictly increasing.
    EphemerisGapError
        If the sample spacing jumps.
    EphemerisUnitError
        On an unknown unit, implausible magnitudes, or inconsistent velocity columns.
    OSError
        If the file cannot be read.

    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    origin = str(file_path)

    header: dict[str, str] = {}
    column_names: list[str] | None = None
    rows: list[tuple[int, list[str]]] = []

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if not body:
                continue
            if ":" not in body:
                raise EphemerisFormatError(
                    f"{origin}:{line_number}: header comment {body!r} is not 'key: value'"
                )
            key, _, value = body.partition(":")
            header[key.strip().lower()] = value.strip()
            continue
        if column_names is None:
            column_names = [name.strip().lower() for name in _split_fields(line)]
            continue
        rows.append((line_number, _split_fields(line)))

    missing_keys = [key for key in REQUIRED_HEADER_KEYS if key not in header]
    if missing_keys:
        raise EphemerisFormatError(
            f"{origin}: missing required header key(s) {missing_keys!r}; an ephemeris "
            "without a stated epoch, frame and time scale cannot be compared to anything"
        )
    if header["format"] != EPHEMERIS_FORMAT:
        raise EphemerisFormatError(
            f"{origin}: format is {header['format']!r}, expected {EPHEMERIS_FORMAT!r}; "
            "this reader does not sniff foreign ephemeris formats"
        )
    if column_names is None:
        raise EphemerisFormatError(f"{origin}: no column header line found after the comments")

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in column_names]
    if missing_columns:
        raise EphemerisFormatError(
            f"{origin}: missing required column(s) {missing_columns!r}; found {column_names!r}"
        )
    if not rows:
        raise EphemerisFormatError(f"{origin}: no data rows found after the column header")

    try:
        frame = ReferenceFrame(header["frame"].upper())
    except ValueError as exc:
        raise EphemerisFormatError(
            f"{origin}: unknown frame {header['frame']!r}; expected one of "
            f"{[member.value for member in ReferenceFrame]!r}"
        ) from exc
    try:
        scale = TimeScale(header["time_scale"].upper())
    except ValueError as exc:
        raise EphemerisFormatError(
            f"{origin}: unknown time_scale {header['time_scale']!r}; expected one of "
            f"{[member.value for member in TimeScale]!r}"
        ) from exc
    try:
        provenance = Provenance(header.get("provenance", Provenance.UNKNOWN.value).lower())
    except ValueError as exc:
        raise EphemerisFormatError(
            f"{origin}: unknown provenance {header['provenance']!r}; expected one of "
            f"{[member.value for member in Provenance]!r}"
        ) from exc

    position_unit = header["position_unit"].strip()
    velocity_unit = header["velocity_unit"].strip()
    if position_unit not in _POSITION_UNITS:
        raise EphemerisUnitError(
            f"{origin}: unknown position_unit {position_unit!r}; expected one of "
            f"{sorted(_POSITION_UNITS)!r}"
        )
    if velocity_unit not in _VELOCITY_UNITS:
        raise EphemerisUnitError(
            f"{origin}: unknown velocity_unit {velocity_unit!r}; expected one of "
            f"{sorted(_VELOCITY_UNITS)!r}"
        )

    try:
        epoch = Epoch.from_iso(header["epoch"], scale)
    except ValueError as exc:
        raise EphemerisFormatError(f"{origin}: bad epoch header: {exc}") from exc

    index_of = {name: column_names.index(name) for name in REQUIRED_COLUMNS}
    width = len(column_names)
    times = np.empty(len(rows), dtype=np.float64)
    states = np.empty((len(rows), 6), dtype=np.float64)

    for row_index, (line_number, fields) in enumerate(rows):
        if len(fields) != width:
            raise EphemerisFormatError(
                f"{origin}:{line_number}: row has {len(fields)} field(s), expected {width} "
                f"to match the column header {column_names!r}. A short or long row is never "
                "skipped: dropping it would silently shorten the compared arc"
            )
        try:
            times[row_index] = float(fields[index_of["t"]])
            for slot, name in enumerate(REQUIRED_COLUMNS[1:]):
                states[row_index, slot] = float(fields[index_of[name]])
        except ValueError as exc:
            raise EphemerisFormatError(
                f"{origin}:{line_number}: field is not a number ({exc})"
            ) from exc

    states[:, :3] *= _POSITION_UNITS[position_unit]
    states[:, 3:] *= _VELOCITY_UNITS[velocity_unit]

    _validate_arrays(times, states, max_step_ratio=max_step_ratio, origin=origin)
    if check_units:
        _check_unit_plausibility(states, position_unit, origin)
    if velocity_rel_tol is not None:
        _check_velocity_consistency(times, states, rel_tol=velocity_rel_tol, origin=origin)

    extra = {
        key: value
        for key, value in header.items()
        if key not in REQUIRED_HEADER_KEYS and key not in {"provenance", "source"}
    }
    return Ephemeris(
        epoch=epoch,
        times_s=times,
        states=states,
        frame=frame,
        provenance=provenance,
        source=header.get("source", ""),
        metadata=extra,
    )


def write_ephemeris(path: str | Path, ephemeris: Ephemeris) -> Path:
    """Write an ephemeris in the ``rpo-ephemeris/1.0`` contract.

    Always writes SI (``m``, ``m/s``): kilometres are an ingest convenience only, and
    re-emitting them would put a unit conversion back inside the pipeline. Values are
    written with ``repr``, the shortest decimal string that reads back as the identical
    double, so the file round-trips bitwise.

    Parameters
    ----------
    path
        Destination file. Parent directories are created.
    ephemeris
        Ephemeris to write.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    OSError
        If the file cannot be written.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# format: {EPHEMERIS_FORMAT}",
        f"# frame: {ephemeris.frame.value}",
        f"# time_scale: {ephemeris.epoch.scale.value}",
        f"# epoch: {ephemeris.epoch.to_iso()}",
        "# position_unit: m",
        "# velocity_unit: m/s",
        f"# provenance: {ephemeris.provenance.value}",
        f"# source: {ephemeris.source}",
    ]
    lines.extend(f"# {key}: {value}" for key, value in sorted(ephemeris.metadata.items()))
    lines.append(",".join(REQUIRED_COLUMNS))
    for index in range(ephemeris.size):
        values = (float(ephemeris.times_s[index]), *(float(v) for v in ephemeris.states[index]))
        lines.append(",".join(repr(value) for value in values))

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


# --------------------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------------------


def _lagrange_window(
    nodes_s: npt.NDArray[np.float64], values: npt.NDArray[np.float64], query_s: float
) -> npt.NDArray[np.float64]:
    """Evaluate the Lagrange interpolant through ``(nodes_s, values)`` by Neville's recurrence."""
    work = values.astype(np.float64, copy=True)
    count = nodes_s.size
    for step in range(1, count):
        for i in range(count - step):
            work[i] = (
                (query_s - nodes_s[i + step]) * work[i] + (nodes_s[i] - query_s) * work[i + 1]
            ) / (nodes_s[i] - nodes_s[i + step])
    return np.asarray(work[0], dtype=np.float64)


def interpolate_states(
    times_s: npt.NDArray[np.float64],
    states: npt.NDArray[np.float64],
    query_s: npt.ArrayLike,
    *,
    method: InterpolationMethod = InterpolationMethod.LAGRANGE,
    points: int = DEFAULT_LAGRANGE_POINTS,
) -> npt.NDArray[np.float64]:
    """Interpolate a tabulated state history onto arbitrary times inside its span.

    ``LAGRANGE`` fits a degree ``points - 1`` polynomial through the ``points`` samples
    surrounding each query, position and velocity treated independently. ``HERMITE`` fits a
    cubic per interval using the tabulated velocity as the slope, giving a position/velocity
    pair that is exactly self-consistent but only fourth-order accurate. See the module
    docstring for why Lagrange is the default.

    Extrapolation is refused. Beyond the tabulated span a Lagrange window diverges rapidly,
    and the resulting number would look like a physics disagreement.

    Parameters
    ----------
    times_s
        Strictly increasing sample times, shape ``(N,)``.
    states
        Sample states ``[r(3), v(3)]``, shape ``(N, 6)``.
    query_s
        Times to evaluate at. Must lie within ``[times_s[0], times_s[-1]]``.
    method
        Interpolation scheme.
    points
        Lagrange window size. Must be even and in ``[2, 16]``: even keeps the window centred
        on the query, and beyond about 16 the Runge phenomenon on an equally spaced grid
        makes a wider window worse, not better. Ignored by ``HERMITE``.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(query_s), 6)``.

    Raises
    ------
    InterpolationRangeError
        If any query time lies outside the tabulated span.
    ValueError
        If ``points`` is invalid, or the table is too short for the requested method.

    """
    queries = np.atleast_1d(np.asarray(query_s, dtype=np.float64))
    if queries.ndim != 1:
        raise ValueError(f"query_s must be 1-D, got shape {queries.shape}")
    if not np.all(np.isfinite(queries)):
        raise ValueError("query_s must be finite")

    lower = float(times_s[0])
    upper = float(times_s[-1])
    # A tolerance of one ULP of the span absorbs the round-trip of an endpoint through
    # epoch arithmetic without opening the door to genuine extrapolation.
    slack = np.spacing(max(abs(lower), abs(upper))) * 4.0
    outside = (queries < lower - slack) | (queries > upper + slack)
    if np.any(outside):
        bad = float(queries[np.flatnonzero(outside)[0]])
        raise InterpolationRangeError(
            f"query time {bad!r} s lies outside the tabulated span "
            f"[{lower!r}, {upper!r}] s; extrapolation is refused because a Lagrange window "
            "diverges outside its nodes and the result would be mistaken for a difference"
        )
    clipped = np.clip(queries, lower, upper)

    if method is InterpolationMethod.HERMITE:
        if times_s.size < 2:
            raise ValueError("Hermite interpolation needs at least 2 samples")
        left = np.clip(np.searchsorted(times_s, clipped, side="right") - 1, 0, times_s.size - 2)
        h = times_s[left + 1] - times_s[left]
        s = (clipped - times_s[left]) / h
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        d00 = 6.0 * s2 - 6.0 * s
        d10 = 3.0 * s2 - 4.0 * s + 1.0
        d01 = -6.0 * s2 + 6.0 * s
        d11 = 3.0 * s2 - 2.0 * s
        r0 = states[left, :3]
        r1 = states[left + 1, :3]
        v0 = states[left, 3:]
        v1 = states[left + 1, 3:]
        col = h[:, None]
        position = h00[:, None] * r0 + h10[:, None] * col * v0
        position += h01[:, None] * r1 + h11[:, None] * col * v1
        velocity = (d00[:, None] * r0 + d01[:, None] * r1) / col
        velocity += d10[:, None] * v0 + d11[:, None] * v1
        return np.ascontiguousarray(np.concatenate((position, velocity), axis=1))

    if points < 2 or points > 16 or points % 2 != 0:
        raise ValueError(f"points must be even and in [2, 16], got {points!r}")
    if times_s.size < points:
        raise ValueError(
            f"Lagrange interpolation with points={points} needs at least {points} samples, "
            f"got {times_s.size}"
        )
    result = np.empty((clipped.size, 6), dtype=np.float64)
    half = points // 2
    for index, t in enumerate(clipped):
        centre = int(np.searchsorted(times_s, t, side="right")) - 1
        start = min(max(centre - half + 1, 0), times_s.size - points)
        stop = start + points
        result[index] = _lagrange_window(times_s[start:stop], states[start:stop], float(t))
    return result


def estimate_interpolation_error_m(
    ephemeris: Ephemeris,
    *,
    method: InterpolationMethod = InterpolationMethod.LAGRANGE,
    points: int = DEFAULT_LAGRANGE_POINTS,
) -> float:
    """Measure the position interpolation error of an ephemeris at its own spacing.

    Decimates the table to every second sample (spacing ``2h``), interpolates back onto the
    samples that were removed, and takes the worst position error. That error is at spacing
    ``2h``; Richardson extrapolation brings it to the real spacing,

    ``eps_h ~ eps_2h / 2**p``

    with ``p = points`` for Lagrange and ``p = 4`` for cubic Hermite. Nothing here needs a
    truth model or knowledge of the dynamics, so it works unchanged on a real tool export.
    The estimate is deliberately conservative: the removed samples nearest the ends are
    interpolated from an off-centre window, so their error is worse than a mid-span query's.

    Parameters
    ----------
    ephemeris
        Ephemeris to measure.
    method
        Scheme whose error is wanted.
    points
        Lagrange window size.

    Returns
    -------
    float
        Estimated worst-case position interpolation error at the ephemeris's own spacing,
        metres.

    Raises
    ------
    ValueError
        If the ephemeris is too short to decimate for the requested method.

    """
    needed = 2 * points + 1 if method is InterpolationMethod.LAGRANGE else 5
    if ephemeris.size < needed:
        raise ValueError(
            f"interpolation error estimate needs at least {needed} samples for "
            f"{method.value} with points={points}, got {ephemeris.size}"
        )
    limit = ephemeris.size if ephemeris.size % 2 == 1 else ephemeris.size - 1
    coarse_times = ephemeris.times_s[:limit:2]
    coarse_states = ephemeris.states[:limit:2]
    check_times = ephemeris.times_s[1 : limit - 1 : 2]
    check_states = ephemeris.states[1 : limit - 1 : 2]

    predicted = interpolate_states(
        coarse_times, coarse_states, check_times, method=method, points=points
    )
    error_2h = float(np.max(np.linalg.norm(predicted[:, :3] - check_states[:, :3], axis=1)))
    order = points if method is InterpolationMethod.LAGRANGE else 4
    return error_2h / 2.0**order


# --------------------------------------------------------------------------------------
# Frame and time alignment
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Proof that two ephemerides describe the same frame and the same instants.

    Produced by :func:`check_alignment`, which raises rather than returning an object that
    says "not aligned": a caller who forgets to inspect a boolean gets a comparison, and a
    comparison across frames is worse than no comparison.

    Attributes
    ----------
    frame
        Common frame the comparison is performed in (the internal ephemeris's).
    frame_tie
        ``"identity"`` when the frames match exactly, ``"approximate_identity"`` when an
        approximate tie was explicitly permitted.
    frame_tie_error_m
        Estimated position difference attributable to the frame approximation over the
        compared span, metres. See :func:`estimate_frame_tie_error_m`.
    time_scale
        Common time scale.
    epoch_offset_s
        Seconds to add to the external ephemeris's times to express them on the internal
        ephemeris's epoch.
    time_scale_offset_s
        Offset applied to reconcile the two time scales, seconds. Zero when they matched.
    overlap_start_s, overlap_end_s
        Overlapping interval, in internal-epoch seconds.

    """

    frame: ReferenceFrame
    frame_tie: str
    frame_tie_error_m: float
    time_scale: TimeScale
    epoch_offset_s: float
    time_scale_offset_s: float
    overlap_start_s: float
    overlap_end_s: float


def check_alignment(
    internal: Ephemeris,
    external: Ephemeris,
    *,
    allow_approximate_frame_tie: bool = False,
    time_scale_offset_s: float | None = None,
) -> AlignmentReport:
    """Verify that two ephemerides can be compared, raising if they cannot.

    Three things must hold, and each failure raises rather than being papered over:

    1. **Frames.** Identical frames pass. A pair drawn from the inertial family
       (``GCRF_APPROX``, ``GCRF``, ``ICRF``, ``EME2000``) passes *only* with
       ``allow_approximate_frame_tie=True``, applying the identity rotation and recording
       the resulting error budget. Any date-dependent frame (``ITRF``, ``TEME``, ``MOD``,
       ``TOD``) is always refused, because rotating out of one needs precisely the
       Earth-orientation model ``docs/conventions.md`` declines to implement.
    2. **Time scales.** Identical scales pass. ``TAI <-> TT`` passes with the exact 32.184 s
       offset. Anything involving UTC, UT1 or TDB requires ``time_scale_offset_s`` from the
       caller.
    3. **Overlap.** After alignment the two arcs must share an interval. They usually do
       even when an epoch is wrong, which is exactly why check 2 is a refusal and not a
       fallback.

    Parameters
    ----------
    internal
        This project's ephemeris. Defines the frame and epoch the comparison is done in.
    external
        The other tool's (or synthetic reference's) ephemeris.
    allow_approximate_frame_tie
        Permit the identity rotation between two inertial-family frames, accepting the error
        budget recorded in the returned report.
    time_scale_offset_s
        Seconds to add to the external epoch's reading to obtain the internal scale's
        reading. Required when the scales differ and the offset is not exactly defined.

    Returns
    -------
    AlignmentReport
        Record of what was reconciled and at what cost.

    Raises
    ------
    FrameMismatchError
        If the frames differ and no permitted tie exists.
    EpochMismatchError
        If the time scales cannot be reconciled, or the arcs do not overlap.

    """
    tie = "identity"
    if internal.frame is not external.frame:
        both_inertial = (
            internal.frame in _APPROXIMATE_TIE_FRAMES and external.frame in _APPROXIMATE_TIE_FRAMES
        )
        if not both_inertial:
            raise FrameMismatchError(
                f"cannot compare {internal.frame.value} against {external.frame.value}: at "
                "least one is date-dependent, and rotating out of it requires the "
                "precession/nutation/polar-motion model this package explicitly neglects. "
                "Re-export the external ephemeris in an inertial frame"
            )
        if not allow_approximate_frame_tie:
            raise FrameMismatchError(
                f"frames differ ({internal.frame.value} vs {external.frame.value}). Both are "
                "inertial-family, so the identity rotation is available, but it is an "
                "approximation worth about "
                f"{estimate_frame_tie_error_m(internal.span_s, include_frame_bias=True):.3g} m "
                f"over this {internal.span_s:.6g} s arc. Pass "
                "allow_approximate_frame_tie=True to accept that, on the record"
            )
        tie = "approximate_identity"

    scale = internal.epoch.scale
    external_epoch = external.epoch
    scale_offset = 0.0
    if external_epoch.scale is not scale:
        exact = _EXACT_SCALE_OFFSETS_S.get((external_epoch.scale, scale))
        if exact is not None:
            scale_offset = exact
        elif time_scale_offset_s is not None:
            scale_offset = time_scale_offset_s
        else:
            raise EpochMismatchError(
                f"time scales differ ({internal.epoch.scale.value} vs "
                f"{external_epoch.scale.value}) and the offset is not exactly defined: UTC "
                "and UT1 need a leap-second/Earth-orientation table, TDB needs a "
                "relativistic series. Supply time_scale_offset_s. For scale, getting the "
                "37 s TAI-UTC difference wrong is "
                f"{along_track_error_from_time_offset_m(37.0):.4g} m of along-track "
                "difference in LEO"
            )
        external_epoch = external_epoch.in_scale(scale, offset_s=scale_offset)

    epoch_offset = external_epoch.offset_s(internal.epoch)
    external_start = float(external.times_s[0]) + epoch_offset
    external_end = float(external.times_s[-1]) + epoch_offset
    overlap_start = max(float(internal.times_s[0]), external_start)
    overlap_end = min(float(internal.times_s[-1]), external_end)
    if overlap_end <= overlap_start:
        raise EpochMismatchError(
            f"the two arcs do not overlap: internal covers "
            f"[{float(internal.times_s[0]):.6g}, {float(internal.times_s[-1]):.6g}] s and "
            f"external covers [{external_start:.6g}, {external_end:.6g}] s on the internal "
            f"epoch {internal.epoch.to_iso()} {scale.value}. The epoch offset applied was "
            f"{epoch_offset:.6g} s"
        )

    include_bias = internal.frame is not external.frame
    budget = estimate_frame_tie_error_m(
        overlap_end - overlap_start, include_frame_bias=include_bias
    )
    return AlignmentReport(
        frame=internal.frame,
        frame_tie=tie,
        frame_tie_error_m=budget,
        time_scale=scale,
        epoch_offset_s=epoch_offset,
        time_scale_offset_s=scale_offset,
        overlap_start_s=overlap_start,
        overlap_end_s=overlap_end,
    )


def align_ephemeris(
    ephemeris: Ephemeris,
    *,
    to_epoch: Epoch,
    to_frame: ReferenceFrame,
    allow_approximate_frame_tie: bool = False,
    time_scale_offset_s: float | None = None,
) -> Ephemeris:
    """Re-express an ephemeris on another epoch and frame.

    The epoch shift is exact arithmetic on the time column. The frame change is the identity
    rotation and is available only within the inertial family, and then only when explicitly
    permitted -- see :func:`check_alignment` for why nothing else is offered.

    Parameters
    ----------
    ephemeris
        Ephemeris to re-express.
    to_epoch
        Target epoch. Its time scale becomes the ephemeris's.
    to_frame
        Target frame.
    allow_approximate_frame_tie
        Permit the identity rotation between two distinct inertial-family frames.
    time_scale_offset_s
        Seconds to add to this ephemeris's epoch reading to obtain the target scale's.

    Returns
    -------
    Ephemeris
        Ephemeris with shifted times and the target frame. Provenance and source are
        carried through unchanged; alignment does not make synthetic data real.

    Raises
    ------
    FrameMismatchError
        If the frame change is not permitted.
    EpochMismatchError
        If the time scales cannot be reconciled.

    """
    if ephemeris.frame is not to_frame:
        both_inertial = (
            ephemeris.frame in _APPROXIMATE_TIE_FRAMES and to_frame in _APPROXIMATE_TIE_FRAMES
        )
        if not both_inertial or not allow_approximate_frame_tie:
            raise FrameMismatchError(
                f"no rotation from {ephemeris.frame.value} to {to_frame.value} is available "
                "without an Earth-orientation model"
                + ("" if not both_inertial else "; pass allow_approximate_frame_tie=True")
            )
    source_epoch = ephemeris.epoch.in_scale(to_epoch.scale, offset_s=time_scale_offset_s)
    shift = source_epoch.offset_s(to_epoch)
    return Ephemeris(
        epoch=to_epoch,
        times_s=ephemeris.times_s + shift,
        states=ephemeris.states,
        frame=to_frame,
        provenance=ephemeris.provenance,
        source=ephemeris.source,
        metadata=dict(ephemeris.metadata),
    )


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComponentStatistics:
    """Max-absolute, RMS and signed-mean summary of one error component.

    The signed mean is the point of this object. A component whose RMS is 12 m and whose
    mean is 12 m has a bias; one whose RMS is 12 m and whose mean is 0.1 m is noise. Only
    the first is a modelling difference.
    """

    max_abs: float
    rms: float
    mean: float

    def to_json_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping."""
        return {"max_abs": self.max_abs, "rms": self.rms, "mean": self.mean}


def _summarise(values: npt.NDArray[np.float64]) -> ComponentStatistics:
    """Summarise a 1-D error component."""
    return ComponentStatistics(
        max_abs=float(np.max(np.abs(values))),
        rms=float(np.sqrt(np.mean(values**2))),
        mean=float(np.mean(values)),
    )


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """The result of comparing this project's ephemeris against an external one.

    Every field a reader needs in order to decide whether to believe the headline number is
    present: what was compared, in which frame, over which arc, how the external side was
    produced, how much of the difference the frame approximation could account for, and how
    much of it the interpolation machinery could account for.

    ``external_provenance`` and ``is_external_tool_validated`` are load-bearing. A report
    generated against a synthetic reference is a useful self-consistency result and is *not*
    external-tool validation; a report that cannot distinguish the two is a liability, so
    the distinction is a required field rather than a convention.

    Attributes
    ----------
    label
        Free text naming the comparison, e.g. ``"vbar-baseline-24h"``.
    frame, time_scale, epoch_iso
        Common frame, time scale, and the internal epoch that ``times`` are measured from.
    internal_source, external_source
        Free text naming each producer.
    internal_provenance, external_provenance
        Where each side came from.
    is_external_tool_validated
        ``True`` only when the external side is :attr:`Provenance.TOOL_RUN`. This is the
        single flag that says whether the project may describe the run as external-tool
        validation.
    frame_tie, frame_tie_error_m
        How the frames were reconciled and the estimated position difference that choice can
        account for on its own.
    epoch_offset_s, time_scale_offset_s
        Epoch and time-scale offsets applied during alignment.
    num_points, start_s, end_s
        Comparison grid.
    interpolation_method, interpolation_points, interpolation_error_m
        Which side was interpolated how, and the measured error of doing so.
    interpolation_margin
        ``position_max_m / interpolation_error_m``. Large is good.
    interpolation_is_negligible
        ``True`` when ``interpolation_margin`` clears the required margin, i.e. the reported
        difference is not an artefact of the interpolation.
    difference_within_frame_budget
        ``True`` when the reported maximum position difference is smaller than
        ``frame_tie_error_m``, meaning the frame approximation alone could explain it and no
        dynamics conclusion may be drawn.
    position_max_m, position_rms_m, position_max_time_s
        Total position difference magnitude statistics.
    velocity_max_m_s, velocity_rms_m_s
        Total velocity difference magnitude statistics.
    radial_m, along_track_m, cross_track_m
        Hill-frame position breakdown.
    radial_m_s, along_track_m_s, cross_track_m_s
        Hill-frame velocity breakdown.
    rotating_frame_velocity
        Whether the transport theorem was applied to the velocity difference.

    """

    label: str
    frame: ReferenceFrame
    time_scale: TimeScale
    epoch_iso: str
    internal_source: str
    external_source: str
    internal_provenance: Provenance
    external_provenance: Provenance
    is_external_tool_validated: bool
    frame_tie: str
    frame_tie_error_m: float
    epoch_offset_s: float
    time_scale_offset_s: float
    num_points: int
    start_s: float
    end_s: float
    interpolation_method: InterpolationMethod
    interpolation_points: int
    interpolation_error_m: float
    interpolation_margin: float
    interpolation_is_negligible: bool
    difference_within_frame_budget: bool
    position_max_m: float
    position_rms_m: float
    position_max_time_s: float
    velocity_max_m_s: float
    velocity_rms_m_s: float
    radial_m: ComponentStatistics
    along_track_m: ComponentStatistics
    cross_track_m: ComponentStatistics
    radial_m_s: ComponentStatistics
    along_track_m_s: ComponentStatistics
    cross_track_m_s: ComponentStatistics
    rotating_frame_velocity: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole report.

        Returns
        -------
        dict
            Nested mapping; every value is a JSON primitive.

        """
        return {
            "schema": "rpo-comparison/1.0",
            "label": self.label,
            "alignment": {
                "frame": self.frame.value,
                "time_scale": self.time_scale.value,
                "epoch": self.epoch_iso,
                "frame_tie": self.frame_tie,
                "frame_tie_error_m": self.frame_tie_error_m,
                "epoch_offset_s": self.epoch_offset_s,
                "time_scale_offset_s": self.time_scale_offset_s,
            },
            "provenance": {
                "internal_source": self.internal_source,
                "internal_provenance": self.internal_provenance.value,
                "external_source": self.external_source,
                "external_provenance": self.external_provenance.value,
                "is_external_tool_validated": self.is_external_tool_validated,
            },
            "grid": {
                "num_points": self.num_points,
                "start_s": self.start_s,
                "end_s": self.end_s,
                "interpolation_method": self.interpolation_method.value,
                "interpolation_points": self.interpolation_points,
                "interpolation_error_m": self.interpolation_error_m,
                "interpolation_margin": self.interpolation_margin,
                "interpolation_is_negligible": self.interpolation_is_negligible,
            },
            "position_m": {
                "max": self.position_max_m,
                "rms": self.position_rms_m,
                "max_time_s": self.position_max_time_s,
                "radial": self.radial_m.to_json_dict(),
                "along_track": self.along_track_m.to_json_dict(),
                "cross_track": self.cross_track_m.to_json_dict(),
            },
            "velocity_m_s": {
                "max": self.velocity_max_m_s,
                "rms": self.velocity_rms_m_s,
                "radial": self.radial_m_s.to_json_dict(),
                "along_track": self.along_track_m_s.to_json_dict(),
                "cross_track": self.cross_track_m_s.to_json_dict(),
                "rotating_frame": self.rotating_frame_velocity,
            },
            "interpretation": {
                "difference_within_frame_budget": self.difference_within_frame_budget,
            },
        }

    def to_json(self) -> str:
        """Return the report as an indented JSON document.

        ``allow_nan=False``: a NaN in a validation record would be emitted as the
        non-standard token ``NaN`` and read back by a permissive parser as a number.
        """
        return json.dumps(self.to_json_dict(), indent=2, allow_nan=False) + "\n"


def write_comparison_report(path: str | Path, report: ComparisonReport) -> Path:
    """Write a :class:`ComparisonReport` as JSON.

    Parameters
    ----------
    path
        Destination file. Parent directories are created.
    report
        Report to write.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    OSError
        If the file cannot be written.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.to_json(), encoding="utf-8")
    return destination


def _ric_components(
    internal: npt.NDArray[np.float64],
    external: npt.NDArray[np.float64],
    *,
    rotating_frame_velocity: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate state differences into the Hill frame defined by the internal states."""
    count = internal.shape[0]
    position = np.empty((count, 3), dtype=np.float64)
    velocity = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        rotation, omega_eci = hill_basis(internal[index, :3], internal[index, 3:])
        dr = external[index, :3] - internal[index, :3]
        dv = external[index, 3:] - internal[index, 3:]
        position[index] = rotation @ dr
        if rotating_frame_velocity:
            dv = dv - np.cross(omega_eci, dr)
        velocity[index] = rotation @ dv
    return position, velocity


def compare_ephemerides(
    internal: Ephemeris,
    external: Ephemeris,
    *,
    label: str = "",
    times_s: npt.ArrayLike | None = None,
    method: InterpolationMethod = InterpolationMethod.LAGRANGE,
    points: int = DEFAULT_LAGRANGE_POINTS,
    allow_approximate_frame_tie: bool = False,
    time_scale_offset_s: float | None = None,
    required_interpolation_margin: float = DEFAULT_INTERPOLATION_MARGIN,
    rotating_frame_velocity: bool = False,
) -> ComparisonReport:
    """Compare two ephemerides and report the difference with its RIC breakdown.

    Alignment is checked first, every time, by :func:`check_alignment`; there is no path
    through this function that compares unaligned data. The external ephemeris is then
    interpolated onto the comparison grid and the difference ``external - internal`` is
    decomposed in the Hill frame of the internal state.

    The comparison grid defaults to the internal sample times that fall inside the aligned
    external span, so only the external side is interpolated. Supplying ``times_s``
    interpolates both, and the reported interpolation error is then the worse of the two.

    Parameters
    ----------
    internal
        This project's ephemeris. Defines the frame, the epoch, and the Hill triad.
    external
        The other side. Its :attr:`Ephemeris.provenance` propagates into the report.
    label
        Free text naming the comparison.
    times_s
        Explicit comparison grid in internal-epoch seconds. ``None`` uses the internal
        sample times inside the overlap.
    method, points
        Interpolation scheme and Lagrange window size.
    allow_approximate_frame_tie, time_scale_offset_s
        Passed to :func:`check_alignment`.
    required_interpolation_margin
        Ratio of reported maximum difference to measured interpolation error below which the
        report marks itself not interpolation-clean.
    rotating_frame_velocity
        Apply the transport theorem to the velocity difference. Off by default: see the
        module docstring for why a comparison residual is not a relative-motion state.

    Returns
    -------
    ComparisonReport
        The comparison, ready to serialise.

    Raises
    ------
    FrameMismatchError, EpochMismatchError
        From :func:`check_alignment`.
    InterpolationRangeError
        If an explicit ``times_s`` falls outside either aligned span.
    ValueError
        If the comparison grid is empty, or an ephemeris is too short to interpolate.

    """
    alignment = check_alignment(
        internal,
        external,
        allow_approximate_frame_tie=allow_approximate_frame_tie,
        time_scale_offset_s=time_scale_offset_s,
    )
    external_times = external.times_s + alignment.epoch_offset_s

    if times_s is None:
        mask = (internal.times_s >= alignment.overlap_start_s) & (
            internal.times_s <= alignment.overlap_end_s
        )
        grid = internal.times_s[mask]
        internal_states = internal.states[mask]
    else:
        grid = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
        internal_states = interpolate_states(
            internal.times_s, internal.states, grid, method=method, points=points
        )
    if grid.size == 0:
        raise ValueError(
            "the comparison grid is empty; the internal sample times and the aligned "
            f"external span [{alignment.overlap_start_s:.6g}, "
            f"{alignment.overlap_end_s:.6g}] s do not intersect at any sample"
        )

    external_states = interpolate_states(
        external_times, external.states, grid, method=method, points=points
    )

    interpolation_error_m = estimate_interpolation_error_m(external, method=method, points=points)
    if times_s is not None:
        interpolation_error_m = max(
            interpolation_error_m,
            estimate_interpolation_error_m(internal, method=method, points=points),
        )

    position_ric, velocity_ric = _ric_components(
        internal_states, external_states, rotating_frame_velocity=rotating_frame_velocity
    )
    position_norm = np.linalg.norm(position_ric, axis=1)
    velocity_norm = np.linalg.norm(velocity_ric, axis=1)
    worst = int(np.argmax(position_norm))
    position_max = float(position_norm[worst])

    margin = position_max / interpolation_error_m if interpolation_error_m > 0.0 else math.inf
    return ComparisonReport(
        label=label,
        frame=alignment.frame,
        time_scale=alignment.time_scale,
        epoch_iso=internal.epoch.to_iso(),
        internal_source=internal.source,
        external_source=external.source,
        internal_provenance=internal.provenance,
        external_provenance=external.provenance,
        is_external_tool_validated=external.provenance is Provenance.TOOL_RUN,
        frame_tie=alignment.frame_tie,
        frame_tie_error_m=alignment.frame_tie_error_m,
        epoch_offset_s=alignment.epoch_offset_s,
        time_scale_offset_s=alignment.time_scale_offset_s,
        num_points=int(grid.size),
        start_s=float(grid[0]),
        end_s=float(grid[-1]),
        interpolation_method=method,
        interpolation_points=points,
        interpolation_error_m=interpolation_error_m,
        interpolation_margin=margin,
        interpolation_is_negligible=margin >= required_interpolation_margin,
        difference_within_frame_budget=position_max <= alignment.frame_tie_error_m,
        position_max_m=position_max,
        position_rms_m=float(np.sqrt(np.mean(position_norm**2))),
        position_max_time_s=float(grid[worst]),
        velocity_max_m_s=float(np.max(velocity_norm)),
        velocity_rms_m_s=float(np.sqrt(np.mean(velocity_norm**2))),
        radial_m=_summarise(position_ric[:, 0]),
        along_track_m=_summarise(position_ric[:, 1]),
        cross_track_m=_summarise(position_ric[:, 2]),
        radial_m_s=_summarise(velocity_ric[:, 0]),
        along_track_m_s=_summarise(velocity_ric[:, 1]),
        cross_track_m_s=_summarise(velocity_ric[:, 2]),
        rotating_frame_velocity=rotating_frame_velocity,
    )
