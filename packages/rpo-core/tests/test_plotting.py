"""The plot suite: structure, not pixels.

No test here compares images. A pixel comparison fails on a matplotlib point release and
passes on a figure that is wrong, so what is asserted instead is *structure*: how many axes
exist, that every axis label names its unit, that the keep-out circle has the configured
radius, that there are as many burn markers as burns, and that the files come out non-empty.

The load-bearing test in this file is
:func:`test_every_plotted_value_tracks_its_metrics_field`. It edits a field on the metrics
object and asserts the figure moves with it. A plot that recomputed the quantity from the
trajectory would ignore the edit and fail, which is precisely the drift between
``metrics.json`` and the figures that ``rpo_core.plotting`` exists to make impossible.

Every test is marked ``integration``: they run the whole config -> constraints -> metrics ->
figure chain, and they need the ``viz`` extra (``uv run --extra viz pytest``).
"""

import dataclasses
import json
import math
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Wedge
from rpo_core.config import ScenarioConfig
from rpo_core.constraints import (
    ApproachEllipsoid,
    ClosingVelocityLimit,
    KeepOutSphere,
    evaluate_constraints,
)
from rpo_core.exceptions import RpoCoreError
from rpo_core.metrics import Burn, TrajectoryMetrics, compute_metrics
from rpo_core.plotting import (
    FIGURE_FILENAMES,
    PlottingError,
    plot_all,
    plot_delta_v_budget,
    plot_hill_trajectory,
    plot_range_and_rate,
)

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------------------
# A scenario whose numbers are all small integers, so an assertion reads like the figure
# --------------------------------------------------------------------------------------

KEEP_OUT_RADIUS_M = 40.0
CORRIDOR_HALF_ANGLE_DEG = 10.0
CORRIDOR_ACTIVATION_M = 1000.0
CLOSING_LIMIT_M_S = 1.5
N_SAMPLES = 24
DURATION_S = 400.0

_SCENARIO: dict[str, Any] = {
    "name": "plot_fixture",
    "description": "straight-line pass, used to exercise the plot suite",
    "orbit": {"altitude_m": 420_000.0, "inclination_deg": 51.6},
    "start_hold_point": {"name": "start", "position_hill_m": [0.0, -400.0, 0.0]},
    "target_hold_point": {"name": "finish", "position_hill_m": [0.0, 400.0, 0.0]},
    "constraints": {
        "keep_out_sphere_radius_m": KEEP_OUT_RADIUS_M,
        "approach_ellipsoid_semi_axes_m": [2000.0, 4000.0, 2000.0],
        "approach_cone_half_angle_deg": CORRIDOR_HALF_ANGLE_DEG,
        "approach_cone_activation_range_m": CORRIDOR_ACTIVATION_M,
        # Below the 1.98 m/s peak of this pass, so the violating-span shading is exercised.
        "max_closing_velocity_m_s": CLOSING_LIMIT_M_S,
        "max_closing_velocity_activation_range_m": 5000.0,
    },
    "maneuver": {"tof_periods": 0.5},
    "seed": 7,
}


def scenario() -> ScenarioConfig:
    """Return the fixture scenario."""
    return ScenarioConfig.model_validate(json.loads(json.dumps(_SCENARIO)))


def build_metrics(burns: list[Burn] | None = None) -> TrajectoryMetrics:
    """Return metrics for the straight-line pass, with two burns by default."""
    config = scenario()
    times = np.linspace(0.0, DURATION_S, N_SAMPLES)
    states = np.zeros((N_SAMPLES, 6), dtype=np.float64)
    states[:, 0] = 30.0
    states[:, 1] = -400.0 + 2.0 * times
    states[:, 2] = 40.0
    states[:, 4] = 2.0

    limits = config.constraints
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(limits.keep_out_sphere_radius_m),
        ellipsoid=ApproachEllipsoid(limits.approach_ellipsoid_semi_axes_m),
        closing_velocity=ClosingVelocityLimit(
            limits.max_closing_velocity_m_s, limits.max_closing_velocity_activation_range_m
        ),
    )
    if burns is None:
        burns = [Burn("depart", 0.0, (3.0, 4.0, 0.0)), Burn("arrive", DURATION_S, (0.0, -6.0, 8.0))]
    return compute_metrics(
        config,
        times,
        states,
        burns,
        report,
        commanded_terminal_state_hill=(27.0, 396.0, 28.0, 0.0, -4.0, -8.0),
    )


@pytest.fixture
def metrics() -> TrajectoryMetrics:
    """Return the fixture metrics record."""
    return build_metrics()


@pytest.fixture(autouse=True)
def _no_leaked_figures() -> Iterator[None]:
    """Fail a test that leaves a figure open, and clean up so the next test starts empty."""
    assert plt.get_fignums() == [], "a previous test leaked an open figure"
    yield
    leaked = plt.get_fignums()
    plt.close("all")
    assert leaked == [], f"test left {len(leaked)} figure(s) open"


def circles(ax: Axes) -> list[Circle]:
    """Return the plain circles on an axes, excluding wedges."""
    return [patch for patch in ax.patches if type(patch) is Circle]


def wedges(ax: Axes) -> list[Wedge]:
    """Return the wedges on an axes."""
    return [patch for patch in ax.patches if isinstance(patch, Wedge)]


def line_with_marker(ax: Axes, marker: str) -> Line2D:
    """Return the single line drawn with ``marker``, asserting there is exactly one."""
    found = [line for line in ax.lines if line.get_marker() == marker]
    assert len(found) == 1, f"expected one line with marker {marker!r}, found {len(found)}"
    return found[0]


def line_with_label(ax: Axes, label: str) -> Line2D:
    """Return the single line whose legend label is exactly ``label``."""
    found = [line for line in ax.lines if str(line.get_label()) == label]
    assert len(found) == 1, f"expected one line labelled {label!r}, found {len(found)}"
    return found[0]


def line_label_starting(ax: Axes, prefix: str) -> Line2D:
    """Return the single line whose legend label starts with ``prefix``.

    Used for labels that carry a value, e.g. ``"keep-out radius 40 m"``.
    """
    found = [line for line in ax.lines if str(line.get_label()).startswith(prefix)]
    assert len(found) == 1, f"expected one line labelled {prefix!r}..., found {len(found)}"
    return found[0]


# --------------------------------------------------------------------------------------
# Hill-frame trajectory
# --------------------------------------------------------------------------------------


def test_hill_figure_has_one_axes_with_unit_labelled_axes(metrics: TrajectoryMetrics):
    fig = plot_hill_trajectory(metrics)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # The Hill convention has to be legible from the labels alone.
    assert "[m]" in ax.get_xlabel()
    assert "[m]" in ax.get_ylabel()
    assert "V-bar" in ax.get_xlabel() and "along-track" in ax.get_xlabel()
    assert "R-bar" in ax.get_ylabel() and "radial" in ax.get_ylabel()
    plt.close(fig)


def test_hill_figure_puts_radial_up_and_along_track_right(metrics: TrajectoryMetrics):
    """The transposition is the easy thing to get backwards, so it is asserted directly."""
    fig = plot_hill_trajectory(metrics)
    ax = fig.axes[0]
    trajectory = line_with_label(ax, "trajectory")
    horizontal = list(trajectory.get_xdata())
    vertical = list(trajectory.get_ydata())
    assert horizontal == [position[1] for position in metrics.series.position_hill_m]
    assert vertical == [position[0] for position in metrics.series.position_hill_m]
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_keep_out_circle_is_drawn_at_the_configured_radius(metrics: TrajectoryMetrics):
    fig = plot_hill_trajectory(metrics)
    ax = fig.axes[0]
    (keep_out,) = circles(ax)
    assert keep_out.radius == metrics.keep_out_radius_m == KEEP_OUT_RADIUS_M
    assert keep_out.center == (0.0, 0.0)
    plt.close(fig)


def test_corridor_wedge_spans_the_configured_half_angle(metrics: TrajectoryMetrics):
    fig = plot_hill_trajectory(metrics)
    ax = fig.axes[0]
    (corridor,) = wedges(ax)
    assert corridor.r == metrics.corridor_activation_range_m == CORRIDOR_ACTIVATION_M
    # The default V-bar axis (0, -1, 0) points along -y, which is 180 deg in the plot frame.
    assert corridor.theta1 == pytest.approx(180.0 - CORRIDOR_HALF_ANGLE_DEG, abs=1e-9)
    assert corridor.theta2 == pytest.approx(180.0 + CORRIDOR_HALF_ANGLE_DEG, abs=1e-9)
    assert corridor.theta2 - corridor.theta1 == pytest.approx(
        2.0 * math.degrees(metrics.corridor_half_angle_rad), rel=1e-12
    )
    plt.close(fig)


def test_corridor_wedge_follows_a_rotated_axis(metrics: TrajectoryMetrics):
    """Complement: the wedge is placed from the axis field, not hard-coded at 180 deg."""
    rotated = dataclasses.replace(metrics, corridor_axis_hill=(1.0, 0.0, 0.0))
    fig = plot_hill_trajectory(rotated)
    (corridor,) = wedges(fig.axes[0])
    # (1, 0, 0) is +R-bar, straight up in the plot frame: 90 deg.
    assert corridor.theta1 == pytest.approx(90.0 - CORRIDOR_HALF_ANGLE_DEG, abs=1e-9)
    assert corridor.theta2 == pytest.approx(90.0 + CORRIDOR_HALF_ANGLE_DEG, abs=1e-9)
    plt.close(fig)


@pytest.mark.parametrize("n_burns", [0, 1, 2, 3])
def test_burn_marker_count_equals_the_number_of_burns(n_burns: int):
    burns = [
        Burn(f"burn{index}", index * (DURATION_S / 3.0), (0.1 * (index + 1), 0.0, 0.0))
        for index in range(n_burns)
    ]
    record = build_metrics(burns)
    fig = plot_hill_trajectory(record)
    ax = fig.axes[0]
    markers = [line for line in ax.lines if line.get_marker() == "X"]
    if n_burns == 0:
        assert markers == []
    else:
        (burn_line,) = markers
        assert len(burn_line.get_xdata()) == n_burns == len(record.burns)
        assert list(burn_line.get_xdata()) == [burn.position_hill_m[1] for burn in record.burns]
        assert list(burn_line.get_ydata()) == [burn.position_hill_m[0] for burn in record.burns]
    plt.close(fig)


def test_hold_points_are_marked_and_named(metrics: TrajectoryMetrics):
    fig = plot_hill_trajectory(metrics)
    ax = fig.axes[0]
    squares = [line for line in ax.lines if line.get_marker() == "s"]
    assert len(squares) == len(metrics.hold_points) == 2
    annotations = {text.get_text() for text in ax.texts}
    for point in metrics.hold_points:
        assert point.name in annotations
    assert "target" in annotations
    plt.close(fig)


def test_an_out_of_plane_corridor_axis_is_refused_rather_than_drawn_wrong(
    metrics: TrajectoryMetrics,
):
    tilted = dataclasses.replace(metrics, corridor_axis_hill=(0.0, -0.8, 0.6))
    with pytest.raises(PlottingError, match="cross-track component"):
        plot_hill_trajectory(tilted)
    assert plt.get_fignums() == [], "the refused figure must not be created"


# --------------------------------------------------------------------------------------
# Range and range rate
# --------------------------------------------------------------------------------------


def test_range_figure_has_two_axes_labelled_with_units(metrics: TrajectoryMetrics):
    fig = plot_range_and_rate(metrics)
    assert len(fig.axes) == 2
    ax_range, ax_rate = fig.axes
    assert "[m]" in ax_range.get_ylabel()
    assert "[m/s]" in ax_rate.get_ylabel()
    assert "[s]" in ax_rate.get_xlabel()
    # The panels share a time axis, so only the lower one carries its label.
    assert ax_range.get_xlabel() == ""
    plt.close(fig)


def test_range_panel_draws_the_metric_series(metrics: TrajectoryMetrics):
    fig = plot_range_and_rate(metrics)
    ax_range, ax_rate = fig.axes
    assert list(line_with_label(ax_range, "range").get_ydata()) == list(metrics.series.range_m)
    assert list(line_with_label(ax_rate, "range rate").get_ydata()) == list(
        metrics.series.range_rate_m_s
    )
    assert list(line_with_label(ax_range, "range").get_xdata()) == list(metrics.series.times_s)
    plt.close(fig)


def test_closing_velocity_limit_is_drawn_at_minus_the_limit(metrics: TrajectoryMetrics):
    """The limit is on the closing velocity; the panel shows range rate, its negation."""
    fig = plot_range_and_rate(metrics)
    _, ax_rate = fig.axes
    limit_line = line_label_starting(ax_rate, "closing-velocity limit")
    assert set(limit_line.get_ydata()) == {-metrics.closing_velocity_limit_m_s}
    assert f"{CLOSING_LIMIT_M_S:g}" in str(limit_line.get_label())
    plt.close(fig)


def test_violating_spans_are_shaded_once_per_contiguous_run(metrics: TrajectoryMetrics):
    fig = plot_range_and_rate(metrics)
    _, ax_rate = fig.axes
    shaded = [patch for patch in ax_rate.patches if isinstance(patch, Rectangle)]
    mask = metrics.series.closing_velocity_violating
    expected_runs = sum(
        1 for index, flag in enumerate(mask) if flag and (index == 0 or not mask[index - 1])
    )
    assert expected_runs > 0, "the fixture is supposed to violate the closing-velocity limit"
    assert len(shaded) == expected_runs
    plt.close(fig)


def test_nothing_is_shaded_when_nothing_is_violated():
    """Complement: the shading is driven by the mask, not painted unconditionally."""
    raw = json.loads(json.dumps(_SCENARIO))
    raw["constraints"]["max_closing_velocity_m_s"] = 5.0
    config = ScenarioConfig.model_validate(raw)
    times = np.linspace(0.0, DURATION_S, N_SAMPLES)
    states = np.zeros((N_SAMPLES, 6), dtype=np.float64)
    states[:, 0] = 30.0
    states[:, 1] = -400.0 + 2.0 * times
    states[:, 2] = 40.0
    states[:, 4] = 2.0
    limits = config.constraints
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(limits.keep_out_sphere_radius_m),
        closing_velocity=ClosingVelocityLimit(
            limits.max_closing_velocity_m_s, limits.max_closing_velocity_activation_range_m
        ),
    )
    record = compute_metrics(
        config, times, states, [], report, commanded_terminal_state_hill=np.zeros(6)
    )
    assert not any(record.series.closing_velocity_violating)
    fig = plot_range_and_rate(record)
    _, ax_rate = fig.axes
    assert [patch for patch in ax_rate.patches if isinstance(patch, Rectangle)] == []
    plt.close(fig)


def test_the_max_closing_marker_is_omitted_when_the_limit_was_never_active():
    raw = json.loads(json.dumps(_SCENARIO))
    raw["constraints"]["max_closing_velocity_activation_range_m"] = 10.0
    config = ScenarioConfig.model_validate(raw)
    times = np.linspace(0.0, DURATION_S, N_SAMPLES)
    states = np.zeros((N_SAMPLES, 6), dtype=np.float64)
    states[:, 0] = 30.0
    states[:, 1] = -400.0 + 2.0 * times
    states[:, 2] = 40.0
    states[:, 4] = 2.0
    limits = config.constraints
    report = evaluate_constraints(
        times,
        states,
        keep_out=KeepOutSphere(limits.keep_out_sphere_radius_m),
        closing_velocity=ClosingVelocityLimit(
            limits.max_closing_velocity_m_s, limits.max_closing_velocity_activation_range_m
        ),
    )
    record = compute_metrics(
        config, times, states, [], report, commanded_terminal_state_hill=np.zeros(6)
    )
    assert record.max_closing_velocity_m_s is None
    fig = plot_range_and_rate(record)
    _, ax_rate = fig.axes
    assert [line for line in ax_rate.lines if line.get_marker() == "v"] == []
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Delta-v budget
# --------------------------------------------------------------------------------------


def test_delta_v_bars_match_the_burn_magnitudes(metrics: TrajectoryMetrics):
    fig = plot_delta_v_budget(metrics)
    ax = fig.axes[0]
    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert len(bars) == len(metrics.burns) == 2
    assert [bar.get_height() for bar in bars] == [burn.magnitude_m_s for burn in metrics.burns]
    assert "[m/s]" in ax.get_ylabel()
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    for burn, label in zip(metrics.burns, labels, strict=True):
        assert burn.label in label
    plt.close(fig)


def test_delta_v_title_reports_the_total_metric(metrics: TrajectoryMetrics):
    fig = plot_delta_v_budget(metrics)
    assert f"{metrics.total_delta_v_m_s:,.4f}" in fig.axes[0].get_title()
    plt.close(fig)


def test_a_plan_with_no_burns_still_produces_a_figure():
    record = build_metrics([])
    fig = plot_delta_v_budget(record)
    ax = fig.axes[0]
    assert [patch for patch in ax.patches if isinstance(patch, Rectangle)] == []
    assert any("no burns" in text.get_text() for text in ax.texts)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# The contract: plots read metrics, they never recompute
# --------------------------------------------------------------------------------------


def test_every_plotted_value_tracks_its_metrics_field(metrics: TrajectoryMetrics):
    """Edit a field, and the figure must move with it.

    Each edit below is *inconsistent* with the underlying series -- a 40 m keep-out sphere
    redrawn at 123.5 m, a minimum range moved to 7 m. Any plot that derived the quantity
    from the trajectory rather than reading the field would draw the old value and fail
    here, which is the only way to detect that drift before it reaches a report.
    """
    edited = dataclasses.replace(
        metrics,
        keep_out_radius_m=123.5,
        min_koz_range_refined_m=7.0,
        min_koz_time_refined_s=42.0,
        closing_velocity_limit_m_s=0.25,
        total_delta_v_m_s=99.0,
        corridor_activation_range_m=555.0,
    )

    hill = plot_hill_trajectory(edited)
    (keep_out,) = circles(hill.axes[0])
    assert keep_out.radius == 123.5
    (corridor,) = wedges(hill.axes[0])
    assert corridor.r == 555.0
    assert "7.0" in hill.axes[0].get_title()
    plt.close(hill)

    rates = plot_range_and_rate(edited)
    ax_range, ax_rate = rates.axes
    assert set(line_label_starting(ax_range, "keep-out radius").get_ydata()) == {123.5}
    minimum = line_with_marker(ax_range, "v")
    assert list(minimum.get_xdata()) == [42.0]
    assert list(minimum.get_ydata()) == [7.0]
    assert set(line_label_starting(ax_rate, "closing-velocity limit").get_ydata()) == {-0.25}
    plt.close(rates)

    budget = plot_delta_v_budget(edited)
    assert "99.0000" in budget.axes[0].get_title()
    plt.close(budget)


def test_marked_minimum_and_limit_come_from_the_unedited_metrics(metrics: TrajectoryMetrics):
    """The other half of the previous test: unedited, the marks sit on the real values."""
    fig = plot_range_and_rate(metrics)
    ax_range, ax_rate = fig.axes
    minimum = line_with_marker(ax_range, "v")
    assert list(minimum.get_ydata()) == [metrics.min_koz_range_refined_m]
    assert list(minimum.get_xdata()) == [metrics.min_koz_time_refined_s]
    peak = line_with_marker(ax_rate, "v")
    assert metrics.max_closing_velocity_m_s is not None
    assert list(peak.get_ydata()) == [-metrics.max_closing_velocity_m_s]
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Titles, footers, files, and figure hygiene
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder", [plot_hill_trajectory, plot_range_and_rate, plot_delta_v_budget]
)
def test_every_figure_names_the_scenario_and_stamps_the_run(
    metrics: TrajectoryMetrics, builder: Any
):
    fig = builder(metrics)
    titles = [fig.axes[0].get_title()] + [text.get_text() for text in fig.texts]
    assert any(metrics.scenario_name in title for title in titles)
    footers = [text.get_text() for text in fig.texts]
    assert any(metrics.config_hash in footer for footer in footers), "no config hash in footer"
    assert any(f"seed {metrics.seed}" in footer for footer in footers), "no seed in footer"
    plt.close(fig)


def test_plot_all_writes_three_non_empty_files(metrics: TrajectoryMetrics, tmp_path: Path):
    written = plot_all(metrics, tmp_path / "run")
    assert set(written) == set(FIGURE_FILENAMES)
    for name, path in written.items():
        assert path.name == FIGURE_FILENAMES[name]
        assert path.exists()
        # A truncated or blank PNG would still exist; 1 kB is far below the ~35 kB these
        # actually come out at and far above an empty or header-only file.
        assert path.stat().st_size > 1024


def test_plot_all_closes_every_figure(metrics: TrajectoryMetrics, tmp_path: Path):
    plot_all(metrics, tmp_path)
    assert plt.get_fignums() == []


def test_repeated_plotting_does_not_accumulate_open_figures(
    metrics: TrajectoryMetrics, tmp_path: Path
):
    """Matplotlib warns past 20 open figures; 22 rounds must produce no warning at all."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for round_index in range(22):
            plot_all(metrics, tmp_path / f"run{round_index}")
            assert plt.get_fignums() == []


def test_plot_all_reports_a_directory_it_cannot_create(metrics: TrajectoryMetrics, tmp_path: Path):
    blocked = tmp_path / "occupied"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PlottingError, match="cannot write figures"):
        plot_all(metrics, blocked)
    assert plt.get_fignums() == []


def test_plotting_error_is_an_rpo_core_error():
    assert issubclass(PlottingError, RpoCoreError)
    assert issubclass(PlottingError, ValueError)
