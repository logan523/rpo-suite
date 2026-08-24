"""Command line entry point for ``rpo-traj``.

Presentation only. Every number printed here is read off the
:class:`~rpo_traj.plan.RendezvousPlan` that :func:`~rpo_traj.plan.plan_rendezvous` returned,
and through it off the ``metrics.json`` written to disk; nothing is recomputed on the way to
the screen. That is the same rule :mod:`rpo_core.plotting` follows, for the same reason: a
second arithmetic path is a second definition, and the day the two drift, the one on the
screen is the one people quote.

The exit-code contract
----------------------
Three outcomes, and the middle one is the interesting one:

``0``
    The plan completed and every constraint was satisfied.
``1``
    The plan completed and at least one constraint was violated. **Every output is still
    written** -- run directory, ``provenance.json``, ``metrics.json``, figures. A violated
    constraint is a result about the trajectory, not a failure of the tool, and a tool that
    threw away its outputs on discovering one would be unusable for exactly the analysis it
    exists to support.
``2``
    A genuine error: an unreadable or malformed configuration, a singular transfer time, a
    differential correction that would not converge, a propagation that failed, or a
    directory that could not be written. No claim is made about the trajectory.

The shipped ``configs/vbar_baseline.yaml`` exits **1**, deliberately. A half-period
two-impulse V-bar hop bulges radially by exactly a quarter of the hop length -- 187.5 m for
the baseline's 750 m -- which reaches 20.56 deg against its own 10 deg approach corridor.
That is a geometric consequence of the transfer (``docs/project1/math-model.md``, model M4
corollary), not a mis-tuned scenario, and the corridor is not widened to hide it. Fixing it
takes a multi-burn approach, which is Phase B.

Argparse's own usage errors -- an unknown flag, a missing argument, ``--samples notanumber``
-- also exit ``2``, which is argparse's default and happens to be the right bucket: they are
errors, and no trajectory was planned.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
import traceback
from collections.abc import Sequence
from typing import TextIO

from rpo_core.config import DEFAULT_RUNS_DIR, load_scenario
from rpo_core.constraints import ConstraintResult, SafetyReport
from rpo_core.exceptions import RpoCoreError
from rpo_core.targeting import DEFAULT_TOLERANCE_M

from .plan import DEFAULT_SAMPLE_COUNT, RendezvousPlan, plan_rendezvous

__all__ = [
    "EXIT_CONSTRAINT_VIOLATED",
    "EXIT_ERROR",
    "EXIT_OK",
    "build_parser",
    "format_summary",
    "main",
]

#: Plan completed, every constraint satisfied.
EXIT_OK: int = 0

#: Plan completed and outputs were written, but at least one constraint was violated.
EXIT_CONSTRAINT_VIOLATED: int = 1

#: The plan could not be produced: bad config, or a propagation or targeting failure.
EXIT_ERROR: int = 2

# Output width. Matches the repository's own 100-column source limit, so a summary
# pasted into a review or a commit message lines up with the code around it.
_RULE_WIDTH = 100
_RULE = "=" * _RULE_WIDTH
_LABEL_WIDTH = 34

_EPILOG = f"""\
exit codes
  {EXIT_OK}  plan completed and every constraint was satisfied
  {EXIT_CONSTRAINT_VIOLATED}  plan completed and at least one constraint was VIOLATED.
     All outputs are still written: a violated constraint is a result about the
     trajectory, not a failure of the tool.
  {EXIT_ERROR}  error: unreadable or malformed config, singular transfer time, targeting
     or propagation failure, or an output that could not be written. No claim is
     made about the trajectory. Argparse usage errors also exit {EXIT_ERROR}.

note
  configs/vbar_baseline.yaml exits {EXIT_CONSTRAINT_VIOLATED} by design. A half-period
  two-impulse V-bar hop bulges radially by a quarter of the hop length, which for the
  baseline's 750 m reaches 20.56 deg against its own 10 deg approach corridor. That
  is geometry, not a tuning error; reporting it is the point, and the corridor is not
  widened to hide it. See docs/project1/math-model.md (model M4 corollary).

examples
  rpo-traj plan configs/vbar_baseline.yaml --seed 42
  rpo-traj plan configs/vbar_baseline.yaml --no-correct   # fly the raw CW impulses
  rpo-traj plan configs/vbar_baseline.yaml --no-plots --quiet --out /tmp/runs
"""


def build_parser() -> argparse.ArgumentParser:
    """Return the ``rpo-traj`` argument parser.

    Separated from :func:`main` so the surface can be inspected and tested without running
    a plan, and so ``--help`` text is checkable.

    Returns
    -------
    argparse.ArgumentParser
        Parser carrying the ``plan`` subcommand.

    """
    parser = argparse.ArgumentParser(
        prog="rpo-traj",
        description=(
            "Rendezvous and proximity-operations trajectory design. Plans a two-impulse "
            "transfer between two hold points, flies it under nonlinear relative dynamics, "
            "evaluates the approach safety constraints, and writes a reproducible run "
            "directory."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    plan_parser = subparsers.add_parser(
        "plan",
        help="plan a rendezvous from a scenario YAML file",
        description=(
            "Plan a two-impulse rendezvous from a scenario file and write metrics.json, "
            "provenance.json and the figure suite into a content-addressed run directory."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan_parser.add_argument(
        "config",
        metavar="CONFIG",
        help="path to a scenario YAML file, e.g. configs/vbar_baseline.yaml",
    )
    plan_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "run seed, overriding the scenario's own. Names the run directory and lands in "
            "provenance.json; it changes no number in a two-impulse plan, because nothing "
            "in one is stochastic. Default: the scenario's seed."
        ),
    )
    correction = plan_parser.add_mutually_exclusive_group()
    correction.add_argument(
        "--correct",
        dest="correct",
        action="store_true",
        default=True,
        help=(
            "correct the CW impulse onto nonlinear dynamics by differential correction "
            "(default). On the baseline this turns a 1.23 m terminal miss into 5.5e-07 m."
        ),
    )
    correction.add_argument(
        "--no-correct",
        dest="correct",
        action="store_false",
        help=(
            "fly the raw Clohessy-Wiltshire impulses. Not a degraded mode: the terminal "
            "position error then measures the linearisation error directly."
        ),
    )
    plan_parser.add_argument(
        "--no-plots",
        dest="make_plots",
        action="store_false",
        default=True,
        help="skip the figure suite. Does not import matplotlib at all.",
    )
    plan_parser.add_argument(
        "--out",
        metavar="DIR",
        default=str(DEFAULT_RUNS_DIR),
        help=(
            "parent directory for the run. The run lands in <DIR>/<config-hash>-<seed>/. "
            f"Default: {DEFAULT_RUNS_DIR}"
        ),
    )
    plan_parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        metavar="N",
        help=(
            "trajectory samples, uniform over the time of flight. The default is measured, "
            "not chosen: it resolves a 3.6 s keep-out minimum a coarser grid steps over, "
            "and holds the closing-velocity discretisation error to 0.5%% of the limit. "
            f"Default: {DEFAULT_SAMPLE_COUNT}"
        ),
    )
    plan_parser.add_argument(
        "--targeting-tol",
        type=float,
        default=DEFAULT_TOLERANCE_M,
        metavar="METRES",
        help=(
            "convergence tolerance on terminal position miss for the differential "
            "correction, metres. Below about 1e-08 m the nonlinear oracle's own noise "
            f"floor makes the request unsatisfiable and the run exits {EXIT_ERROR}. "
            f"Default: {DEFAULT_TOLERANCE_M:g}"
        ),
    )
    plan_parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the summary. Errors still print, and the exit code is unchanged.",
    )
    plan_parser.add_argument(
        "--traceback",
        action="store_true",
        help="print the full traceback on error instead of a one-line message.",
    )
    return parser


def _format_float(value: float, decimals: int = 6) -> str:
    """Return a fixed-point rendering, falling back to scientific for extreme magnitudes.

    Terminal errors in this suite legitimately span 1e-07 m to 1e+00 m in the same table --
    corrected versus raw -- and a fixed format wide enough for both is unreadable at one end
    and lies by rounding to zero at the other.
    """
    if value != 0.0 and (abs(value) < 1.0e-4 or abs(value) >= 1.0e6):
        return f"{value:.6e}"
    return f"{value:.{decimals}f}"


def _metric_line(label: str, value: str, unit: str = "", note: str = "") -> str:
    """Return one right-aligned metric row of the SRS section 4 block."""
    row = f"   {label:<{_LABEL_WIDTH}}{value:>14}"
    if unit:
        row += f"  {unit}"
    if note:
        row = f"{row:<70}{note}"
    return row.rstrip()


def _constraint_display(
    result: ConstraintResult, plan: RendezvousPlan
) -> tuple[str, str, str, str]:
    """Return ``(quantity, value, unit, limit)`` for one constraint, in its own units.

    Each constraint's ``worst_value`` is measured in something different -- metres of
    clearance, a dimensionless quadratic form, radians, metres per second -- and printing
    four numbers in one column without saying which is which is how a report gets misread.
    The limit is restated from the scenario rather than from the result, because a
    :class:`~rpo_core.constraints.ConstraintResult` does not carry the geometry it was
    judged against.
    """
    constraints = plan.config.constraints
    if result.name == "keep_out_sphere":
        return (
            "min clearance",
            _format_float(result.worst_value),
            "m",
            f"sphere r = {constraints.keep_out_sphere_radius_m:g} m",
        )
    if result.name == "approach_ellipsoid":
        semi_axes = "/".join(f"{axis:g}" for axis in constraints.approach_ellipsoid_semi_axes_m)
        return (
            "max quadratic form",
            _format_float(result.worst_value),
            "",
            f"<= 1 for {semi_axes} m",
        )
    if result.name == "approach_corridor":
        return (
            "max cone angle",
            _format_float(math.degrees(result.worst_value)),
            "deg",
            f"<= {constraints.approach_cone_half_angle_deg:g} deg inside"
            f" {constraints.approach_cone_activation_range_m:g} m",
        )
    return (
        "max closing velocity",
        _format_float(result.worst_value),
        "m/s",
        f"<= {constraints.max_closing_velocity_m_s:g} m/s inside"
        f" {constraints.max_closing_velocity_activation_range_m:g} m",
    )


def _constraint_rows(result: ConstraintResult, plan: RendezvousPlan) -> list[str]:
    """Return the display rows for one constraint: headline, then detail when violated."""
    status = "ok" if result.satisfied else "VIOLATED"
    prefix = f"   {result.name:<21}{status:<10}"
    if math.isnan(result.worst_value):
        # No sample fell inside the activation range, so the constraint was never actually
        # evaluated. It reports "satisfied" because nothing broke it, which is exactly the
        # vacuous pass configs/vbar_baseline.yaml's own comments warn about; saying so is
        # the only thing that stops it reading as a genuine clearance.
        return [f"{prefix}NOT EVALUATED: no sample fell inside its activation range"]

    quantity, value, unit, limit = _constraint_display(result, plan)
    rows = [f"{prefix}{quantity:<21}{value:>12} {unit:<4}  {limit}"]
    if not result.satisfied:
        first = result.first_violation_time_s
        when = (
            "detected only by sub-sample refinement, between samples"
            if first is None
            else f"first at t = {first:.3f} s"
        )
        rows.append(f"{'':<34}{result.n_violating_samples} of {plan.n_samples} samples, {when}")
    return rows


def _targeting_lines(plan: RendezvousPlan) -> list[str]:
    """Return the lines describing which impulses flew and what the correction bought."""
    if not plan.corrected:
        return [
            " targeting     raw Clohessy-Wiltshire impulses, UNCORRECTED",
            f"               terminal miss under nonlinear dynamics: "
            f"{plan.raw_cw_terminal_miss_m:.6e} m",
        ]
    achieved = plan.corrected_terminal_miss_m
    if achieved is None:  # pragma: no cover - set together with plan.corrected
        return [" targeting     differential correction requested but not recorded"]
    iterations = plan.targeting_iterations or 0
    plural = "" if iterations == 1 else "s"
    improvement = (
        f"{plan.raw_cw_terminal_miss_m / achieved:.3e}x better" if achieved > 0.0 else "exact"
    )
    cost = plan.metrics.total_delta_v_m_s - plan.cw_total_delta_v_m_s
    return [
        " targeting     differential correction onto nonlinear dynamics",
        f"               terminal miss {plan.raw_cw_terminal_miss_m:.6e} m"
        f" -> {achieved:.6e} m ({improvement})",
        f"               {iterations} Newton iteration{plural}; delta-v cost {cost:+.3e} m/s"
        f" on {plan.cw_total_delta_v_m_s:.6f} m/s of raw CW budget",
    ]


def format_summary(plan: RendezvousPlan) -> str:
    """Render the human-readable run summary.

    Reports every metric required by SRS section 4 that a single deterministic plan can
    produce -- total delta-v, time of flight, terminal position and velocity error, minimum
    keep-out distance, maximum closing velocity, constraint violation count -- followed by a
    per-constraint breakdown naming the limit each was judged against.

    Parameters
    ----------
    plan
        A completed plan, violated constraints or not.

    Returns
    -------
    str
        The summary, without a trailing newline.

    """
    metrics = plan.metrics
    config = plan.config
    lines: list[str] = [_RULE, f" rpo-traj plan : {config.name}", _RULE]

    if config.description:
        wrapped = textwrap.wrap(config.description.strip(), width=_RULE_WIDTH - 15)
        lines.append(f" scenario      {wrapped[0]}")
        lines.extend(f"{'':<15}{line}" for line in wrapped[1:])
    lines += [
        f" config hash   {metrics.config_hash}    seed {metrics.seed}",
        f" run directory {plan.run_dir}",
        f" dynamics      nonlinear two-body (differenced orbits), {plan.n_samples} samples"
        f" over {metrics.time_of_flight_s:.3f} s",
    ]
    lines += _targeting_lines(plan)
    lines.append("")

    lines.append(" MISSION METRICS  (SRS section 4)")
    lines.append(_metric_line("total delta-v", _format_float(metrics.total_delta_v_m_s), "m/s"))
    for burn in metrics.burns:
        lines.append(_metric_line(f"  {burn.label}", _format_float(burn.magnitude_m_s), "m/s"))
    lines.append(
        _metric_line(
            "time of flight",
            f"{metrics.time_of_flight_s:.4f}",
            "s",
            f"= {metrics.time_of_flight_periods:.6f} orbits",
        )
    )
    lines.append(
        _metric_line("terminal position error", f"{metrics.terminal_position_error_m:.6e}", "m")
    )
    lines.append(
        _metric_line("terminal velocity error", f"{metrics.terminal_velocity_error_m_s:.6e}", "m/s")
    )
    lines.append(
        _metric_line(
            "min keep-out distance (sampled)", _format_float(metrics.min_koz_range_sampled_m), "m"
        )
    )
    lines.append(
        _metric_line(
            "min keep-out distance (refined)",
            _format_float(metrics.min_koz_range_refined_m),
            "m",
            f"clearance {_format_float(metrics.min_koz_clearance_refined_m)} m",
        )
    )
    if not metrics.koz_refinement_applied:
        # The discrete minimum sat on an endpoint, so there was no bracketing triple to
        # refine and the refined column simply repeats the sampled one. Saying so stops the
        # two identical numbers reading as an independent confirmation (F-4.6).
        lines.append("   (no sub-sample refinement: the closest approach is an endpoint)")
    if metrics.max_closing_velocity_m_s is None:
        lines.append(
            _metric_line(
                "max closing velocity",
                "n/a",
                "",
                "no sample inside the activation range -- NOT a measured clearance",
            )
        )
    else:
        lines.append(
            _metric_line(
                "max closing velocity", _format_float(metrics.max_closing_velocity_m_s), "m/s"
            )
        )
    lines.append(
        _metric_line("constraint violations", str(metrics.constraint_violation_count), "samples")
    )
    lines.append("")

    lines.append(" CONSTRAINTS")
    for result in plan.report.results:
        lines.extend(_constraint_rows(result, plan))
    lines.append("")

    within = "within budget" if metrics.cw_within_budget else "OVER BUDGET"
    lines.append(
        f" CW LINEARISATION  conservative bound {metrics.cw_error_bound_m:.3f} m against a"
        f" {metrics.cw_error_budget_m:.3f} m budget ({within})"
    )
    if plan.figure_paths:
        lines.append(f" FIGURES           {len(plan.figure_paths)} written to the run directory")
    elif plan.plots_skipped_reason:
        lines.append(f" FIGURES           none: {plan.plots_skipped_reason}")
    lines.append("")

    lines.extend(_result_lines(plan))
    lines.append(_RULE)
    return "\n".join(lines)


def _result_lines(plan: RendezvousPlan) -> list[str]:
    """Return the closing verdict, naming the exit code the caller is about to get."""
    report: SafetyReport = plan.report
    evaluated = len(report.results)
    if plan.all_constraints_satisfied:
        return [
            f" RESULT  all {evaluated} constraints satisfied over {plan.n_samples} samples"
            f" -- exit {EXIT_OK}"
        ]
    failed = [result.name for result in report.results if not result.satisfied]
    return [
        f" RESULT  {len(failed)} of {evaluated} constraints VIOLATED"
        f" ({', '.join(failed)}) -- exit {EXIT_CONSTRAINT_VIOLATED}",
        "         over "
        f"{plan.violation_count} violating samples of {plan.n_samples}."
        " This is a reported result, not a failure:",
        f"         every output is written to {plan.run_dir}",
    ]


def _run_plan(args: argparse.Namespace, stdout: TextIO) -> int:
    """Load, plan, print, and return the exit code. Exceptions are the caller's problem."""
    config = load_scenario(args.config)
    plan = plan_rendezvous(
        config,
        seed=args.seed,
        correct=args.correct,
        n_samples=args.samples,
        base_dir=args.out,
        make_plots=args.make_plots,
        targeting_tolerance_m=args.targeting_tol,
    )
    if not args.quiet:
        print(format_summary(plan), file=stdout)
    return EXIT_OK if plan.all_constraints_satisfied else EXIT_CONSTRAINT_VIOLATED


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the ``rpo-traj`` command line and return its exit code.

    Parameters
    ----------
    argv
        Argument list excluding the program name. Defaults to ``sys.argv[1:]``.
    stdout, stderr
        Streams to write to. Default to the real ones. Injected so a test can capture output
        without touching global interpreter state.

    Returns
    -------
    int
        :data:`EXIT_OK`, :data:`EXIT_CONSTRAINT_VIOLATED`, or :data:`EXIT_ERROR`. See the
        module docstring for the contract.

    Raises
    ------
    SystemExit
        From argparse, on a usage error or ``--help``. Argparse's own exit status for a
        usage error is already :data:`EXIT_ERROR`.

    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)
    try:
        return _run_plan(args, out)
    except (RpoCoreError, OSError) as exc:
        if args.traceback:
            traceback.print_exc(file=err)
        else:
            # The message, not the traceback: every exception rpo_core raises carries the
            # numbers that motivated it -- condition number, residual history, offending
            # field and value -- and a stack of frames above that adds nothing a user can
            # act on. --traceback is there for when the frames are the point.
            print(f"rpo-traj: error: {type(exc).__name__}: {exc}", file=err)
            print("rpo-traj: rerun with --traceback for the full stack.", file=err)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
