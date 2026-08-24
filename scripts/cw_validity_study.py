"""Measure the Clohessy-Wiltshire validity envelope against nonlinear two-body motion.

Sweeps chaser-target separation and elapsed time, comparing the linear CW propagation
against a nonlinear reference built by differencing two independently propagated two-body
orbits. Writes a metrics JSON and a two-panel figure.

The target orbit is exactly circular throughout, which isolates linearisation error from
eccentricity error. Both are real error sources; conflating them produces an error budget
that cannot be attributed.

Run:
    uv run python scripts/cw_validity_study.py
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this must render identically in CI
import matplotlib.pyplot as plt
import numpy as np
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from rpo_core.relative.nonlinear import (
    CW_ERROR_COEFFICIENT,
    CW_ERROR_SAFETY_FACTOR,
    cw_position_error_m,
)

ALTITUDE_M = 420.0e3
INCLINATION_RAD = math.radians(51.6)
SEPARATIONS_M = (10.0, 100.0, 1_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0)
TIME_FRACTIONS = (0.25, 0.5, 1.0, 2.0)
SAMPLES_PER_SWEEP = 121

KEEP_OUT_ZONE_RADIUS_M = 200.0
#: Fraction of the keep-out radius allowed as linearisation error. 2.5 % against a 200 m
#: sphere is a large real margin; see docs/cw_validity.md for why 1 % was rejected.
ERROR_BUDGET_FRACTION = 0.025
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cw_validity"


def _git_sha() -> str:
    """Return the current commit SHA, or 'uncommitted' when there is no commit yet."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def main() -> int:
    """Run the sweep, write metrics and figure, and report the headline numbers."""
    radius_m = R_EARTH_EQUATORIAL_M + ALTITUDE_M
    v_circular = math.sqrt(MU_EARTH_M3_S2 / radius_m)
    n = mean_motion_rad_s(radius_m)
    period_s = orbital_period_s(radius_m)

    r_target = np.array([radius_m, 0.0, 0.0])
    v_target = v_circular * np.array([0.0, math.cos(INCLINATION_RAD), math.sin(INCLINATION_RAD)])

    errors: dict[float, dict[float, float]] = {}
    for separation in SEPARATIONS_M:
        state0 = np.array([0.0, -separation, 0.0, 0.0, 0.0, 0.0])
        errors[separation] = {}
        for fraction in TIME_FRACTIONS:
            times = np.linspace(0.0, fraction * period_s, SAMPLES_PER_SWEEP)
            errors[separation][fraction] = float(
                cw_position_error_m(r_target, v_target, state0, times, n).max()
            )

    # Largest separation whose one-orbit error stays under 1 % of the keep-out radius.
    budget_m = ERROR_BUDGET_FRACTION * KEEP_OUT_ZONE_RADIUS_M
    # Invert the CONSERVATIVE bound, not the central estimate: the usable envelope is
    # the separation you can defend, not the one you can hope for.
    usable_separation_m = math.sqrt(
        budget_m * radius_m / (CW_ERROR_SAFETY_FACTOR * CW_ERROR_COEFFICIENT)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "scenario": {
            "altitude_m": ALTITUDE_M,
            "orbit_radius_m": radius_m,
            "inclination_deg": math.degrees(INCLINATION_RAD),
            "mean_motion_rad_s": n,
            "orbital_period_s": period_s,
            "target_eccentricity": 0.0,
        },
        "error_law": {
            "form": "err_one_orbit_m = C * separation_m**2 / orbit_radius_m",
            "coefficient_C": CW_ERROR_COEFFICIENT,
            "coefficient_over_pi": CW_ERROR_COEFFICIENT / math.pi,
            "status": "measured, not derived",
        },
        "max_position_error_m": {
            f"{sep:.0f}": {f"{frac:g}_orbits": err for frac, err in by_time.items()}
            for sep, by_time in errors.items()
        },
        "usable_envelope": {
            "error_budget_m": budget_m,
            "budget_basis": (
                f"{ERROR_BUDGET_FRACTION:.1%} of a {KEEP_OUT_ZONE_RADIUS_M:.0f} m keep-out sphere"
            ),
            "safety_factor": CW_ERROR_SAFETY_FACTOR,
            "bound_basis": "conservative bound, not central estimate",
            "max_separation_m_at_one_orbit": usable_separation_m,
        },
        "provenance": {
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "integrator": "DOP853, rtol=atol=1e-12",
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))

    for fraction in TIME_FRACTIONS:
        ax_left.loglog(
            SEPARATIONS_M,
            [errors[s][fraction] for s in SEPARATIONS_M],
            marker="o",
            label=f"{fraction:g} orbit" + ("s" if fraction != 1.0 else ""),
        )
    reference = [CW_ERROR_COEFFICIENT * s**2 / radius_m for s in SEPARATIONS_M]
    ax_left.loglog(SEPARATIONS_M, reference, "k--", lw=1, label=r"$6\pi\rho^2/r$ (1 orbit)")
    ax_left.axhline(budget_m, color="tab:red", ls=":", lw=1)
    ax_left.text(
        SEPARATIONS_M[0],
        budget_m * 1.3,
        f"error budget {budget_m:.0f} m",
        fontsize=8,
        color="tab:red",
    )
    ax_left.set_xlabel("initial separation ρ [m]")
    ax_left.set_ylabel("max CW position error [m]")
    ax_left.set_title("CW linearisation error vs separation")
    ax_left.grid(True, which="both", alpha=0.3)
    ax_left.legend(fontsize=8)

    # Start the sweep just after the epoch: at t = 0 the error is identically zero, and
    # including it drags the log axis down twelve decades into integrator noise, visually
    # flattening the growth this panel exists to show.
    for separation in (100.0, 1_000.0, 10_000.0):
        times = np.linspace(0.0, 2.0 * period_s, SAMPLES_PER_SWEEP)
        state0 = np.array([0.0, -separation, 0.0, 0.0, 0.0, 0.0])
        error = cw_position_error_m(r_target, v_target, state0, times, n)
        ax_right.plot(times[1:] / period_s, error[1:], label=f"ρ = {separation:,.0f} m")
    ax_right.set_yscale("log")
    ax_right.set_ylim(1e-4, 1e4)
    ax_right.set_xlabel("elapsed time [orbits]")
    ax_right.set_ylabel("CW position error [m]")
    ax_right.set_title("CW linearisation error vs elapsed time")
    ax_right.grid(True, which="both", alpha=0.3)
    ax_right.legend(fontsize=8)

    fig.suptitle(
        f"Clohessy-Wiltshire validity envelope — {ALTITUDE_M / 1e3:.0f} km circular, "
        f"i = {math.degrees(INCLINATION_RAD):.1f}°  |  reference: nonlinear two-body",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cw_validity.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {OUTPUT_DIR / 'metrics.json'}")
    print(f"Wrote {OUTPUT_DIR / 'cw_validity.png'}")
    print()
    print(f"  Error law:  err(1 orbit) = {CW_ERROR_COEFFICIENT:.4f} * rho^2 / r   (= 6*pi)")
    print(f"  At   1 km separation, 1 orbit:  {errors[1_000.0][1.0]:8.3f} m")
    print(f"  At  10 km separation, 1 orbit:  {errors[10_000.0][1.0]:8.1f} m")
    print(
        f"  Error budget ({budget_m:.0f} m = {ERROR_BUDGET_FRACTION:.1%} of KOZ) holds out to "
        f"{usable_separation_m:,.0f} m separation over one orbit (conservative bound)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
