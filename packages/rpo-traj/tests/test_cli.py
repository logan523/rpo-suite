"""Tests for :mod:`rpo_traj.cli`, and for the exit-code contract in particular.

The contract is the deliverable, so it is tested twice over. Every scenario is driven both
through ``main(argv)`` in process -- fast, and it reports coverage -- and through the real
``rpo-traj`` console script in a subprocess, which is the only thing that proves the entry
point in ``pyproject.toml`` is wired to the function this file imports. An in-process test
alone would still pass with the console script pointing at nothing.

The three exit codes each get a scenario that genuinely produces them: the shipped baseline
(violates its corridor by geometry), a deliberately shortened hop (satisfies everything),
and both a malformed configuration and an unreachable targeting tolerance (errors).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from rpo_traj.cli import (
    EXIT_CONSTRAINT_VIOLATED,
    EXIT_ERROR,
    EXIT_OK,
    build_parser,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "vbar_baseline.yaml"

#: Small enough to keep every CLI test near a second, large enough that the plan is a real
#: one: the constraints are still evaluated over a genuine arc, and the exit code is still
#: decided by physics. The convergence of the reported extrema is ``test_plan.py``'s job,
#: and repeating that work in every CLI test would buy nothing.
FAST_SAMPLES = "201"

#: A hop short enough that its ``dy/4`` radial bulge stays inside a 10 deg corridor, with
#: both activation ranges widened so the corridor and the closing-velocity limit are
#: genuinely enforced rather than vacuously satisfied. This is the exit-0 scenario.
SHORT_HOP_YAML = textwrap.dedent("""\
    name: vbar_short_hop
    description: Short V-bar hop whose radial bulge stays inside the corridor.
    orbit:
      altitude_m: 420000.0
      inclination_deg: 51.6
    start_hold_point:
      name: vbar_minus_1000
      position_hill_m: [0.0, -1000.0, 0.0]
    target_hold_point:
      name: vbar_minus_900
      position_hill_m: [0.0, -900.0, 0.0]
    constraints:
      keep_out_sphere_radius_m: 200.0
      approach_ellipsoid_semi_axes_m: [2000.0, 4000.0, 2000.0]
      approach_cone_half_angle_deg: 10.0
      approach_cone_activation_range_m: 1200.0
      max_closing_velocity_m_s: 0.1
      max_closing_velocity_activation_range_m: 1200.0
    maneuver:
      tof_periods: 0.5
    integrator:
      method: DOP853
      rtol: 1.0e-12
      atol: 1.0e-12
    seed: 42
""")

#: An altitude below the reentry floor and five missing required sections. Malformed in a
#: way the validator can describe field by field, which is what the error message has to
#: carry for the exit-2 path to be worth anything.
MALFORMED_YAML = "name: broken\norbit: {altitude_m: 1000.0, inclination_deg: 51.6}\n"


@pytest.fixture(scope="session")
def console_script() -> Path:
    """Return the installed ``rpo-traj`` console script, or skip if it is not there."""
    script = Path(sys.executable).parent / "rpo-traj"
    if not script.is_file():  # pragma: no cover - depends on how the venv was built
        pytest.skip(f"console script not installed at {script}")
    return script


@pytest.fixture
def short_hop_path(tmp_path: Path) -> Path:
    """Write the exit-0 scenario to a temporary YAML file and return its path."""
    path = tmp_path / "vbar_short_hop.yaml"
    path.write_text(SHORT_HOP_YAML, encoding="utf-8")
    return path


@pytest.fixture
def malformed_path(tmp_path: Path) -> Path:
    """Write the malformed scenario to a temporary YAML file and return its path."""
    path = tmp_path / "broken.yaml"
    path.write_text(MALFORMED_YAML, encoding="utf-8")
    return path


def run_in_process(*argv: str) -> tuple[int, str, str]:
    """Invoke :func:`rpo_traj.cli.main` directly, returning ``(code, stdout, stderr)``."""
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def run_console_script(script: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Invoke the installed console script in a subprocess."""
    return subprocess.run(
        [str(script), *argv], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )


def only_run_dir(base: Path) -> Path:
    """Return the single run directory under ``base``, asserting there is exactly one."""
    directories = [path for path in base.iterdir() if path.is_dir()]
    assert len(directories) == 1, directories
    return directories[0]


# ======================================================================================
# Exit 1 -- the shipped baseline violates its own corridor, by design
# ======================================================================================


@pytest.mark.integration
def test_baseline_exits_one_and_still_writes_every_output(tmp_path: Path) -> None:
    """The flagship scenario reports a violation, exits 1, and completes its outputs.

    This is the behaviour ``configs/vbar_baseline.yaml`` ships with deliberately: a
    half-period two-impulse V-bar hop bulges radially by a quarter of the hop length, which
    for 750 m is 187.5 m and reaches 20.56 deg against the scenario's own 10 deg corridor
    (``docs/project1/math-model.md``, model M4 corollary). Exiting 0 here, or exiting
    without writing the run, would both be regressions.
    """
    code, stdout, stderr = run_in_process(
        "plan",
        str(BASELINE_CONFIG_PATH),
        "--seed",
        "42",
        "--out",
        str(tmp_path),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
    )
    assert code == EXIT_CONSTRAINT_VIOLATED
    assert stderr == ""

    # The violation is named, quantified, and located -- not merely flagged.
    assert "VIOLATED" in stdout
    assert "approach_corridor" in stdout
    assert "closing_velocity" in stdout
    assert "max cone angle" in stdout
    assert "10 deg inside 1000 m" in stdout
    assert "first at t =" in stdout
    assert f"exit {EXIT_CONSTRAINT_VIOLATED}" in stdout
    assert "reported result, not a failure" in stdout

    # Every SRS section 4 metric a single deterministic plan can produce is on the screen.
    for required in (
        "total delta-v",
        "time of flight",
        "terminal position error",
        "terminal velocity error",
        "min keep-out distance",
        "max closing velocity",
        "constraint violations",
    ):
        assert required in stdout, required

    run_dir = only_run_dir(tmp_path)
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "provenance.json").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["all_constraints_satisfied"] is False
    assert metrics["constraint_violation_count"] > 0
    assert metrics["scenario_name"] == "vbar_baseline"
    assert metrics["seed"] == 42


@pytest.mark.integration
def test_baseline_exits_one_through_the_real_console_script(
    console_script: Path, tmp_path: Path
) -> None:
    """The same contract, through the entry point declared in ``pyproject.toml``."""
    result = run_console_script(
        console_script,
        "plan",
        str(BASELINE_CONFIG_PATH),
        "--seed",
        "42",
        "--out",
        str(tmp_path),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
    )
    assert result.returncode == EXIT_CONSTRAINT_VIOLATED, result.stderr
    assert "rpo-traj plan : vbar_baseline" in result.stdout
    assert "VIOLATED" in result.stdout
    assert (only_run_dir(tmp_path) / "metrics.json").is_file()


# ======================================================================================
# Exit 0 -- a scenario that genuinely satisfies every constraint
# ======================================================================================


@pytest.mark.integration
def test_a_satisfied_scenario_exits_zero(short_hop_path: Path, tmp_path: Path) -> None:
    """Exit 0 has to be reachable, or exit 1 is not a signal.

    The complement of the baseline test: same code path, same four constraints, shorter hop.
    Without it, the suite would still pass if the exit code were hard-wired to 1.
    """
    out_dir = tmp_path / "runs"
    code, stdout, stderr = run_in_process(
        "plan", str(short_hop_path), "--out", str(out_dir), "--no-plots", "--samples", FAST_SAMPLES
    )
    assert code == EXIT_OK, stdout + stderr
    assert stderr == ""
    assert "VIOLATED" not in stdout
    assert "all 4 constraints satisfied" in stdout
    assert f"exit {EXIT_OK}" in stdout

    metrics = json.loads((only_run_dir(out_dir) / "metrics.json").read_text())
    assert metrics["all_constraints_satisfied"] is True
    assert metrics["constraint_violation_count"] == 0
    # Not a vacuous pass: the closing-velocity limit had samples inside its activation range
    # and reported a real maximum. A None here would mean the constraint was never checked.
    assert metrics["max_closing_velocity_m_s"] is not None


@pytest.mark.integration
def test_a_satisfied_scenario_exits_zero_through_the_console_script(
    console_script: Path, short_hop_path: Path, tmp_path: Path
) -> None:
    result = run_console_script(
        console_script,
        "plan",
        str(short_hop_path),
        "--out",
        str(tmp_path / "runs"),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
    )
    assert result.returncode == EXIT_OK, result.stderr


# ======================================================================================
# Exit 2 -- genuine errors
# ======================================================================================


@pytest.mark.integration
def test_a_malformed_config_exits_two_with_a_readable_message(
    malformed_path: Path, tmp_path: Path
) -> None:
    """A bad scenario gets a message naming the fields, and no traceback."""
    code, stdout, stderr = run_in_process(
        "plan", str(malformed_path), "--out", str(tmp_path / "runs"), "--no-plots"
    )
    assert code == EXIT_ERROR
    assert stdout == ""

    assert "ScenarioConfigError" in stderr
    # The message carries the offending field and the value it received, which is the whole
    # point of preferring it to a stack.
    assert "altitude_m" in stderr
    assert "150,000 m floor" in stderr
    assert "start_hold_point" in stderr
    assert "--traceback" in stderr

    # No traceback unless asked for.
    assert "Traceback (most recent call last)" not in stderr
    assert 'File "' not in stderr

    # Nothing was written: no claim is made about a trajectory that was never planned.
    assert not (tmp_path / "runs").exists()


@pytest.mark.integration
def test_traceback_flag_prints_the_stack(malformed_path: Path, tmp_path: Path) -> None:
    """Complement: ``--traceback`` must actually change the output.

    Without this, the "no traceback" assertion above would still pass if the flag were
    ignored, or if nothing ever printed a traceback at all.
    """
    code, _, stderr = run_in_process(
        "plan", str(malformed_path), "--out", str(tmp_path / "runs"), "--no-plots", "--traceback"
    )
    assert code == EXIT_ERROR
    assert "Traceback (most recent call last)" in stderr
    assert "ScenarioConfigError" in stderr


@pytest.mark.integration
def test_a_missing_config_file_exits_two(tmp_path: Path) -> None:
    code, _, stderr = run_in_process(
        "plan", str(tmp_path / "nope.yaml"), "--out", str(tmp_path / "runs"), "--no-plots"
    )
    assert code == EXIT_ERROR
    assert "cannot read scenario file" in stderr
    assert "nope.yaml" in stderr


@pytest.mark.integration
def test_an_unreachable_targeting_tolerance_exits_two(tmp_path: Path) -> None:
    """A numerical failure is exit 2, distinctly from a configuration failure.

    The nonlinear oracle's noise floor is ~5e-09 m, so 1e-12 m cannot be delivered and the
    differential correction correctly refuses. The run must not fall back to a best-effort
    plan, and must not report a trajectory it could not certify.
    """
    out_dir = tmp_path / "runs"
    code, stdout, stderr = run_in_process(
        "plan",
        str(BASELINE_CONFIG_PATH),
        "--out",
        str(out_dir),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
        "--targeting-tol",
        "1e-12",
    )
    assert code == EXIT_ERROR
    assert stdout == ""
    assert "TargetingConvergenceError" in stderr
    # The message carries the residual history's endpoint, which is what tells a user their
    # tolerance was below the floor rather than their problem being divergent.
    assert "noise floor" in stderr
    assert not out_dir.exists()


@pytest.mark.integration
def test_the_same_scenario_succeeds_at_the_default_tolerance(tmp_path: Path) -> None:
    """Complement to the previous test: only the tolerance made it fail."""
    code, _, _ = run_in_process(
        "plan",
        str(BASELINE_CONFIG_PATH),
        "--out",
        str(tmp_path / "runs"),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
    )
    assert code == EXIT_CONSTRAINT_VIOLATED


@pytest.mark.unit
def test_argparse_usage_errors_exit_two() -> None:
    """Argparse's own failures land in the error bucket, which is where they belong."""
    for argv in (["plan"], ["plan", "x", "--samples", "notanumber"], ["nosuchcommand"], []):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == EXIT_ERROR, argv


@pytest.mark.integration
def test_console_script_exits_two_on_a_malformed_config(
    console_script: Path, malformed_path: Path, tmp_path: Path
) -> None:
    result = run_console_script(
        console_script, "plan", str(malformed_path), "--out", str(tmp_path / "runs"), "--no-plots"
    )
    assert result.returncode == EXIT_ERROR
    assert "Traceback (most recent call last)" not in result.stderr
    assert "altitude_m" in result.stderr


# ======================================================================================
# Options: --out, --seed, --no-correct, --no-plots, --quiet
# ======================================================================================


@pytest.mark.integration
def test_out_directory_is_honoured(short_hop_path: Path, tmp_path: Path) -> None:
    """``--out`` decides where the run lands, and nothing is written anywhere else."""
    requested = tmp_path / "somewhere" / "deep"
    default_location = REPO_ROOT / "results" / "runs"
    before = set(default_location.glob("*")) if default_location.exists() else set()

    code, _, _ = run_in_process(
        "plan",
        str(short_hop_path),
        "--out",
        str(requested),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
    )
    assert code == EXIT_OK

    run_dir = only_run_dir(requested)
    assert run_dir.parent == requested
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "provenance.json").is_file()
    # The run directory name is content-addressed: <config-hash>-<seed>.
    config_hash, _, seed = run_dir.name.rpartition("-")
    assert seed == "42"
    assert json.loads((run_dir / "metrics.json").read_text())["config_hash"] == config_hash

    after = set(default_location.glob("*")) if default_location.exists() else set()
    assert after == before, "a run leaked into the default results directory"


@pytest.mark.integration
def test_seed_option_renames_the_run_and_lands_in_the_record(
    short_hop_path: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "runs"
    assert (
        run_in_process(
            "plan",
            str(short_hop_path),
            "--out",
            str(out_dir),
            "--no-plots",
            "--samples",
            FAST_SAMPLES,
            "--seed",
            "7",
        )[0]
        == EXIT_OK
    )
    assert (
        run_in_process(
            "plan",
            str(short_hop_path),
            "--out",
            str(out_dir),
            "--no-plots",
            "--samples",
            FAST_SAMPLES,
            "--seed",
            "12345",
        )[0]
        == EXIT_OK
    )

    names = sorted(path.name for path in out_dir.iterdir() if path.is_dir())
    assert len(names) == 2, names
    assert names[0].endswith("-12345")
    assert names[1].endswith("-7")
    # Same scenario, so the same content hash: only the seed suffix differs.
    assert names[0].rpartition("-")[0] == names[1].rpartition("-")[0]
    for name, expected in ((names[0], 12345), (names[1], 7)):
        assert json.loads((out_dir / name / "metrics.json").read_text())["seed"] == expected
        assert json.loads((out_dir / name / "provenance.json").read_text())["seed"] == expected


@pytest.mark.integration
def test_no_correct_flies_the_raw_cw_impulses(tmp_path: Path) -> None:
    """``--no-correct`` must change the answer, and change it in the documented direction."""
    corrected_dir = tmp_path / "corrected"
    raw_dir = tmp_path / "raw"
    _, corrected_stdout, _ = run_in_process(
        "plan",
        str(BASELINE_CONFIG_PATH),
        "--out",
        str(corrected_dir),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
    )
    _, raw_stdout, _ = run_in_process(
        "plan",
        str(BASELINE_CONFIG_PATH),
        "--out",
        str(raw_dir),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
        "--no-correct",
    )

    assert "differential correction onto nonlinear dynamics" in corrected_stdout
    assert "UNCORRECTED" in raw_stdout
    assert "differential correction" not in raw_stdout

    corrected = json.loads((only_run_dir(corrected_dir) / "metrics.json").read_text())
    raw = json.loads((only_run_dir(raw_dir) / "metrics.json").read_text())
    # Measured: 5.52e-07 m corrected against 1.2253 m raw, an improvement of 2.2e+06.
    assert corrected["terminal_position_error_m"] < 1e-6
    assert 1.0 < raw["terminal_position_error_m"] < 1.5
    # Same scenario file, so the same config hash: correction is a run option, not part of
    # the scenario's identity. The two runs therefore share a directory name and would
    # collide were --out not distinct, which is the behaviour to be aware of, not a bug.
    assert corrected["config_hash"] == raw["config_hash"]


@pytest.mark.integration
def test_no_plots_writes_no_figures_and_never_imports_matplotlib(
    short_hop_path: Path, tmp_path: Path
) -> None:
    """``--no-plots`` must skip the whole plotting stack, not just the ``savefig`` calls.

    Checked in a fresh subprocess by inspecting ``sys.modules`` after the run, because
    ``rpo_core.plotting`` selects the ``Agg`` matplotlib backend at import time -- a
    process-global side effect that an in-process assertion could not distinguish from an
    import some earlier test performed.
    """
    out_dir = tmp_path / "runs"
    script = textwrap.dedent(f"""
        import sys
        from rpo_traj.cli import main
        code = main(["plan", {str(short_hop_path)!r}, "--out", {str(out_dir)!r},
                     "--no-plots", "--quiet", "--samples", {FAST_SAMPLES!r}])
        assert code == 0, code
        leaked = sorted(m for m in sys.modules if m.split(".")[0] == "matplotlib")
        print("LEAKED:" + ",".join(leaked))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    leaked_line = next(line for line in result.stdout.splitlines() if line.startswith("LEAKED:"))
    assert leaked_line == "LEAKED:", f"matplotlib was imported: {leaked_line}"

    run_dir = only_run_dir(out_dir)
    assert not list(run_dir.glob("*.png"))
    assert (run_dir / "metrics.json").is_file()


@pytest.mark.integration
def test_plots_are_written_by_default(short_hop_path: Path, tmp_path: Path) -> None:
    """Complement: without ``--no-plots`` the figures do appear, so the flag does work."""
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "runs"
    code, stdout, _ = run_in_process(
        "plan", str(short_hop_path), "--out", str(out_dir), "--samples", FAST_SAMPLES
    )
    assert code == EXIT_OK
    assert "3 written to the run directory" in stdout
    figures = sorted(path.name for path in only_run_dir(out_dir).glob("*.png"))
    assert figures == ["delta_v_budget.png", "hill_trajectory.png", "range_and_rate.png"]


@pytest.mark.integration
def test_quiet_suppresses_the_summary_but_not_the_exit_code(
    short_hop_path: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "runs"
    code, stdout, stderr = run_in_process(
        "plan",
        str(short_hop_path),
        "--out",
        str(out_dir),
        "--no-plots",
        "--samples",
        FAST_SAMPLES,
        "--quiet",
    )
    assert code == EXIT_OK
    assert stdout == ""
    assert stderr == ""
    # Quiet is about the summary, not about the work: the outputs are still complete.
    assert (only_run_dir(out_dir) / "metrics.json").is_file()


@pytest.mark.integration
def test_quiet_does_not_suppress_errors(malformed_path: Path, tmp_path: Path) -> None:
    """Complement: a quiet run that fails still says why."""
    code, stdout, stderr = run_in_process(
        "plan", str(malformed_path), "--out", str(tmp_path / "runs"), "--no-plots", "--quiet"
    )
    assert code == EXIT_ERROR
    assert stdout == ""
    assert "ScenarioConfigError" in stderr


# ======================================================================================
# Help and parser surface
# ======================================================================================


@pytest.mark.unit
def test_help_documents_the_exit_codes() -> None:
    """The exit-code contract is part of the interface, so it is part of ``--help``."""
    help_text = build_parser().format_help()
    assert "exit codes" in help_text
    for code in (EXIT_OK, EXIT_CONSTRAINT_VIOLATED, EXIT_ERROR):
        assert f"\n  {code}  " in help_text, code
    assert "VIOLATED" in help_text
    assert "All outputs are still written" in help_text
    assert "by design" in help_text


@pytest.mark.unit
@pytest.mark.parametrize("argv", [["--help"], ["plan", "--help"]])
def test_help_exits_zero(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0


@pytest.mark.unit
def test_correct_and_no_correct_are_mutually_exclusive() -> None:
    parser = build_parser()
    assert parser.parse_args(["plan", "c.yaml"]).correct is True
    assert parser.parse_args(["plan", "c.yaml", "--no-correct"]).correct is False
    assert parser.parse_args(["plan", "c.yaml", "--correct"]).correct is True
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["plan", "c.yaml", "--correct", "--no-correct"])
    assert excinfo.value.code == EXIT_ERROR
