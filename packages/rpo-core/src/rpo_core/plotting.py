r"""The mission plot suite, drawn entirely from :class:`~rpo_core.metrics.TrajectoryMetrics`.

The one rule
------------
**Every number a figure displays is read from a field on the metrics object. Nothing here
recomputes anything.** No ranges are re-derived from states, no delta-v is re-summed, no
violation is re-decided. That is what makes ``metrics.json`` and the figures incapable of
disagreeing: there is one arithmetic path, it runs in :mod:`rpo_core.metrics`, and this
module is a rendering of its output. A plot that quietly re-derives a quantity is a second
implementation of that quantity, and the day the two definitions drift the figure is the
one people believe.

The projections
---------------
The trajectory figure is the **in-plane** Hill projection, with the radial axis (R-bar,
Hill ``x``, positive away from Earth) drawn **up** and the along-track axis (V-bar, Hill
``y``) drawn **right**. Note that this is a transposition, not the usual "first coordinate
on the horizontal axis": Hill ``x`` is the vertical axis here. Drawing it the other way
round is the standard way to publish a rendezvous plot that reads backwards, because a
chaser trailing the target sits at negative ``y`` and belongs on the left. Cross-track
(Hill ``z``) is not shown; the baseline manoeuvre is planar and a third panel of a flat
line is chartjunk.

The keep-out sphere projects to a circle of radius ``keep_out_radius_m`` about the origin,
which is exact: a sphere's silhouette is its great circle in every projection. The approach
corridor projects to a circular wedge of radius ``corridor_activation_range_m`` spanning
``+/- corridor_half_angle_rad`` about the projected cone axis. That one is **not** exact --
a cone whose axis has a cross-track component projects to a hyperbolic sector, not a wedge
-- so the wedge is drawn only for an axis lying in the orbital plane, and an out-of-plane
axis raises rather than drawing a shape that is wrong by an amount nobody can eyeball.

Traceability
------------
Every figure carries the scenario name in its title and the configuration hash and seed in
a footer, so a figure printed and pinned to a wall names the run that produced it. The hash
is the content-addressed identity from :func:`rpo_core.config.config_hash`, the same string
that names the run directory.

Backend
-------
This module selects the ``Agg`` backend at import. Agg is a headless raster renderer: it
needs no display, no GUI toolkit, and renders identically in CI and on a workstation. The
selection is process-global, which is why this module is imported only by callers that
actually want figures -- :mod:`rpo_core.metrics` deliberately does not import it, so a
headless Monte Carlo run pulls no plotting stack at all.

Units are SI and are named on every axis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # headless: must render in CI with no display. Set before pyplot.

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Wedge

from .exceptions import RpoCoreError
from .metrics import TrajectoryMetrics

__all__ = [
    "FIGURE_FILENAMES",
    "PlottingError",
    "plot_all",
    "plot_delta_v_budget",
    "plot_hill_trajectory",
    "plot_range_and_rate",
]

#: Filenames :func:`plot_all` writes, keyed by the figure name it uses in its return value.
FIGURE_FILENAMES: dict[str, str] = {
    "hill_trajectory": "hill_trajectory.png",
    "range_and_rate": "range_and_rate.png",
    "delta_v_budget": "delta_v_budget.png",
}

#: Render resolution. 150 dpi keeps a 7 x 6 inch figure legible when a reviewer drops it
#: into a document at half size, which is the size these are actually read at.
FIGURE_DPI: int = 150

#: Out-of-plane component of the corridor axis above which the wedge projection is refused.
#:
#: The in-plane figure can only draw a cone whose axis lies in the orbital plane. 1e-9 is
#: far below any deliberate cross-track tilt and comfortably above the round-off left by
#: normalising an exactly in-plane axis.
AXIS_IN_PLANE_TOL: float = 1.0e-9

#: Grid weight. Light enough to read a curve through, dark enough to read a value off.
_GRID_ALPHA: float = 0.3
_GRID_LINEWIDTH: float = 0.6
_KEEP_OUT_COLOUR = "tab:red"
_CORRIDOR_COLOUR = "tab:green"
_BURN_COLOUR = "tab:orange"


class PlottingError(RpoCoreError, ValueError):
    """Raised when a figure cannot be drawn from the metrics it was given.

    A corridor axis with a cross-track component (which does not project to a wedge), or a
    directory that cannot be written. Drawing a shape that is wrong by an unquantified
    amount would be worse than refusing: nobody eyeballs a wedge and notices it should have
    been a hyperbola.
    """


def _style_grid(ax: Axes, *, axis: Literal["both", "x", "y"] = "both") -> None:
    """Apply the house grid weight to one axes."""
    ax.grid(True, "major", axis, alpha=_GRID_ALPHA, linewidth=_GRID_LINEWIDTH)


def _footer(fig: Figure, metrics: TrajectoryMetrics) -> None:
    """Stamp the run identity along the bottom of a figure."""
    fig.text(
        0.5,
        0.012,
        f"scenario {metrics.scenario_name} · config {metrics.config_hash} · seed {metrics.seed}",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.35",
    )


def _contiguous_spans(mask: tuple[bool, ...]) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs of each run of ``True``, ``stop`` exclusive."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def plot_hill_trajectory(metrics: TrajectoryMetrics) -> Figure:
    """Draw the in-plane Hill-frame trajectory with the safety geometry overlaid.

    Radial (R-bar, Hill x) is the vertical axis and along-track (V-bar, Hill y) the
    horizontal one, at equal aspect so the keep-out circle is a circle. Overlays: the
    keep-out sphere, the approach-corridor wedge, the scenario hold points, and a marker at
    every burn.

    Parameters
    ----------
    metrics
        The record to draw. Every plotted coordinate, radius and angle is a field on it.

    Returns
    -------
    matplotlib.figure.Figure
        The caller owns the figure and is responsible for closing it.

    Raises
    ------
    PlottingError
        If the corridor axis has a cross-track component; see the module docstring.

    """
    axis_x, axis_y, axis_z = metrics.corridor_axis_hill
    if abs(axis_z) > AXIS_IN_PLANE_TOL:
        raise PlottingError(
            f"corridor_axis_hill={metrics.corridor_axis_hill!r} has a cross-track component "
            f"|z|={abs(axis_z):.3e} above {AXIS_IN_PLANE_TOL:.1e}. An out-of-plane cone "
            "projects onto the in-plane view as a hyperbolic sector, not the circular wedge "
            "this figure draws, so the overlay would be wrong by an amount a reader cannot "
            "see. Plot an out-of-plane approach in a projection that can represent it."
        )

    fig, ax = plt.subplots(figsize=(7.0, 6.4))

    along_track = [position[1] for position in metrics.series.position_hill_m]
    radial = [position[0] for position in metrics.series.position_hill_m]

    keep_out = Circle(
        (0.0, 0.0),
        metrics.keep_out_radius_m,
        facecolor=_KEEP_OUT_COLOUR,
        alpha=0.15,
        edgecolor=_KEEP_OUT_COLOUR,
        linewidth=1.2,
        label=f"keep-out sphere, R = {metrics.keep_out_radius_m:,.0f} m",
        zorder=1,
    )
    ax.add_patch(keep_out)

    # Wedge angles are measured in the plot's own frame, where the horizontal axis is Hill y
    # and the vertical is Hill x -- hence atan2(x, y), not the usual atan2(y, x).
    axis_angle_deg = math.degrees(math.atan2(axis_x, axis_y))
    half_angle_deg = math.degrees(metrics.corridor_half_angle_rad)
    corridor = Wedge(
        (0.0, 0.0),
        metrics.corridor_activation_range_m,
        axis_angle_deg - half_angle_deg,
        axis_angle_deg + half_angle_deg,
        facecolor=_CORRIDOR_COLOUR,
        alpha=0.12,
        edgecolor=_CORRIDOR_COLOUR,
        linewidth=1.0,
        label=(
            f"approach corridor, ±{half_angle_deg:,.1f}° "
            f"within {metrics.corridor_activation_range_m:,.0f} m"
        ),
        zorder=1,
    )
    ax.add_patch(corridor)

    ax.plot(along_track, radial, color="tab:blue", linewidth=1.6, label="trajectory", zorder=3)
    ax.plot(0.0, 0.0, marker="*", markersize=13, color="black", linestyle="none", zorder=4)
    ax.annotate("target", (0.0, 0.0), textcoords="offset points", xytext=(8, 6), fontsize=8)

    for point in metrics.hold_points:
        ax.plot(
            point.position_hill_m[1],
            point.position_hill_m[0],
            marker="s",
            markersize=6,
            color="black",
            linestyle="none",
            zorder=5,
        )
        ax.annotate(
            point.name,
            (point.position_hill_m[1], point.position_hill_m[0]),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
        )

    if metrics.burns:
        ax.plot(
            [burn.position_hill_m[1] for burn in metrics.burns],
            [burn.position_hill_m[0] for burn in metrics.burns],
            marker="X",
            markersize=10,
            color=_BURN_COLOUR,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linestyle="none",
            zorder=6,
            label=f"burns ({len(metrics.burns)})",
        )
        for burn in metrics.burns:
            ax.annotate(
                f"{burn.label}\nΔv = {burn.magnitude_m_s:,.4f} m/s",
                (burn.position_hill_m[1], burn.position_hill_m[0]),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=7.5,
                color="0.2",
            )

    ax.set_xlabel("along-track  y  (V-bar) [m]")
    ax.set_ylabel("radial  x  (R-bar, away from Earth) [m]")
    ax.set_aspect("equal", adjustable="datalim")
    _style_grid(ax)
    ax.axhline(0.0, color="0.7", linewidth=0.6, zorder=0)
    ax.axvline(0.0, color="0.7", linewidth=0.6, zorder=0)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(
        f"{metrics.scenario_name} — Hill-frame trajectory (in-plane)\n"
        f"min range {metrics.min_koz_range_refined_m:,.1f} m (refined), "
        f"Δv {metrics.total_delta_v_m_s:,.4f} m/s",
        fontsize=10,
    )

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    _footer(fig, metrics)
    return fig


def plot_range_and_rate(metrics: TrajectoryMetrics) -> Figure:
    """Draw range and range rate against time, with the closing-velocity limit.

    Two stacked panels sharing the time axis. The upper panel is ``|rho|`` with the keep-out
    radius and the refined minimum marked; the lower is ``d|rho|/dt``, negative when
    closing, with the closing-velocity limit drawn at ``-max`` (the limit is on the closing
    velocity, which is the negation of the range rate) and every violating span shaded.

    The shaded spans are the per-sample mask carried on the metrics, not a comparison made
    here, so what is shaded and what is counted in ``constraint_violation_count`` are the
    same decision.

    Parameters
    ----------
    metrics
        The record to draw.

    Returns
    -------
    matplotlib.figure.Figure
        The caller owns the figure and is responsible for closing it.

    """
    fig, (ax_range, ax_rate) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True, height_ratios=(1.0, 1.0)
    )
    times = metrics.series.times_s

    ax_range.plot(times, metrics.series.range_m, color="tab:blue", linewidth=1.6, label="range")
    ax_range.axhline(
        metrics.keep_out_radius_m,
        color=_KEEP_OUT_COLOUR,
        linestyle="--",
        linewidth=1.2,
        label=f"keep-out radius {metrics.keep_out_radius_m:,.0f} m",
    )
    ax_range.plot(
        metrics.min_koz_time_refined_s,
        metrics.min_koz_range_refined_m,
        marker="v",
        markersize=8,
        color="tab:purple",
        linestyle="none",
        label=f"min range {metrics.min_koz_range_refined_m:,.1f} m (refined)",
    )
    ax_range.set_ylabel("range  |ρ| [m]")
    _style_grid(ax_range)
    ax_range.legend(fontsize=8, loc="best")

    ax_rate.plot(
        times, metrics.series.range_rate_m_s, color="tab:blue", linewidth=1.6, label="range rate"
    )
    ax_rate.axhline(0.0, color="0.7", linewidth=0.6)
    ax_rate.axhline(
        -metrics.closing_velocity_limit_m_s,
        color=_KEEP_OUT_COLOUR,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"closing-velocity limit {metrics.closing_velocity_limit_m_s:,.3g} m/s "
            f"(inside {metrics.closing_velocity_activation_range_m:,.0f} m)"
        ),
    )
    for first, last in _contiguous_spans(metrics.series.closing_velocity_violating):
        ax_rate.axvspan(
            times[first],
            times[last - 1],
            color=_KEEP_OUT_COLOUR,
            alpha=0.18,
            linewidth=0.0,
        )
    if metrics.max_closing_velocity_m_s is not None:
        ax_rate.plot(
            metrics.max_closing_velocity_time_s,
            -metrics.max_closing_velocity_m_s,
            marker="v",
            markersize=8,
            color="tab:purple",
            linestyle="none",
            label=f"max closing {metrics.max_closing_velocity_m_s:,.4g} m/s",
        )
    ax_rate.set_xlabel("time since epoch [s]")
    ax_rate.set_ylabel("range rate  d|ρ|/dt [m/s]\n(negative = closing)")
    _style_grid(ax_rate)
    ax_rate.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{metrics.scenario_name} — range and range rate  "
        f"(TOF {metrics.time_of_flight_s:,.1f} s = {metrics.time_of_flight_periods:.3f} orbits)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
    _footer(fig, metrics)
    return fig


def plot_delta_v_budget(metrics: TrajectoryMetrics) -> Figure:
    """Draw the per-burn delta-v budget as a labelled bar chart.

    One bar per burn, labelled with its magnitude in m/s, plus the total in the title. The
    magnitudes are the ``BurnMetrics.magnitude_m_s`` fields and the total is
    ``total_delta_v_m_s``; the bars are not re-summed here.

    Parameters
    ----------
    metrics
        The record to draw.

    Returns
    -------
    matplotlib.figure.Figure
        The caller owns the figure and is responsible for closing it. A plan with no burns
        produces an empty axes carrying that statement, rather than no figure at all.

    """
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    labels = [burn.label for burn in metrics.burns]
    magnitudes = [burn.magnitude_m_s for burn in metrics.burns]
    positions = list(range(len(metrics.burns)))

    if metrics.burns:
        bars = ax.bar(positions, magnitudes, color="tab:blue", width=0.55)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [
                f"{label}\nt = {burn.time_s:,.1f} s"
                for label, burn in zip(labels, metrics.burns, strict=True)
            ],
            fontsize=8,
        )
        # Headroom for the bar labels, which sit above the bar and are otherwise clipped.
        ax.set_ylim(0.0, 1.18 * max(magnitudes) if max(magnitudes) > 0.0 else 1.0)
    else:
        ax.text(
            0.5,
            0.5,
            "no burns in this plan\n(natural-motion trajectory)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="0.35",
        )
        ax.set_xticks([])

    ax.set_ylabel("impulse magnitude  |Δv| [m/s]")
    ax.set_xlabel("burn")
    _style_grid(ax, axis="y")
    ax.set_axisbelow(True)
    ax.set_title(
        f"{metrics.scenario_name} — Δv budget\n"
        f"total {metrics.total_delta_v_m_s:,.4f} m/s over {len(metrics.burns)} burn(s)",
        fontsize=10,
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    _footer(fig, metrics)
    return fig


def plot_all(metrics: TrajectoryMetrics, directory: str | Path) -> dict[str, Path]:
    """Render the whole suite into a run directory and close every figure.

    Parameters
    ----------
    metrics
        The record to draw.
    directory
        Run directory, typically the one holding ``metrics.json``. Created if missing.

    Returns
    -------
    dict
        Figure name to written path, keyed as in :data:`FIGURE_FILENAMES`.

    Raises
    ------
    PlottingError
        If a figure cannot be drawn, or the directory or a file cannot be written. Figures
        already opened are closed before the error propagates, so a failed run does not leak
        an open figure into the next one.

    """
    target = Path(directory)
    builders = {
        "hill_trajectory": plot_hill_trajectory,
        "range_and_rate": plot_range_and_rate,
        "delta_v_budget": plot_delta_v_budget,
    }
    written: dict[str, Path] = {}
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name, builder in builders.items():
            fig = builder(metrics)
            try:
                destination = target / FIGURE_FILENAMES[name]
                fig.savefig(destination, dpi=FIGURE_DPI)
            finally:
                plt.close(fig)
            written[name] = destination
    except OSError as exc:
        plt.close("all")
        raise PlottingError(f"cannot write figures into {target}: {exc}") from exc
    return written
