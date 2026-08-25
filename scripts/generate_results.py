"""Regenerate every headline number quoted in the README and technical write-up.

Run:
    uv run --extra viz python scripts/generate_results.py

Writes ``results/headline.md`` and ``results/headline.json``. Every figure in the project's
prose comes from here, so a claim and its evidence cannot drift apart.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from rpo_core.baselines import RendezvousProblem
from rpo_core.constants import (
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from rpo_core.frames import relative_state_hill_to_eci
from rpo_core.optimize import compare_baselines
from rpo_core.propagate import propagate_two_body, specific_energy_j_kg
from rpo_core.relative.nonlinear import (
    CW_ERROR_COEFFICIENT,
    conservative_cw_error_bound_m,
    cw_position_error_m,
)
from rpo_core.targeting import correct_two_impulse_transfer, raw_cw_terminal_miss_m

ALTITUDE_M = 420.0e3
INCLINATION_RAD = math.radians(51.6)


def _target_state() -> tuple[np.ndarray, np.ndarray, float, float, float]:
    a = R_EARTH_EQUATORIAL_M + ALTITUDE_M
    v_circ = math.sqrt(MU_EARTH_M3_S2 / a)
    n = mean_motion_rad_s(a)
    period = orbital_period_s(a)
    r = np.array([a, 0.0, 0.0])
    v = v_circ * np.array([0.0, math.cos(INCLINATION_RAD), math.sin(INCLINATION_RAD)])
    return r, v, n, period, a


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def main() -> int:
    """Compute every headline number and write the results artefacts."""
    r_t, v_t, n, period, a = _target_state()
    out: dict[str, object] = {"git_sha": _git_sha()}

    # --- 1. CW validity law, measured and derived ------------------------------------
    law = {}
    for rho in (1.0e3, 5.0e3, 1.0e4):
        measured = cw_position_error_m(
            r_t,
            v_t,
            np.array([0.0, -rho, 0.0, 0.0, 0.0, 0.0]),
            np.linspace(0.0, period, 121),
            n,
        ).max()
        predicted = CW_ERROR_COEFFICIENT * rho**2 / a
        r_c, v_c = relative_state_hill_to_eci(r_t, v_t, np.array([0.0, -rho, 0.0, 0.0, 0.0, 0.0]))
        a_c = -MU_EARTH_M3_S2 / (2.0 * specific_energy_j_kg(np.concatenate((r_c, v_c))))
        law[f"{rho:.0f}"] = {
            "measured_one_orbit_m": measured,
            "predicted_6pi_rho2_over_r_m": predicted,
            "delta_a_measured_m": a_c - a,
            "delta_a_predicted_2rho2_over_r_m": 2.0 * rho**2 / a,
            "three_pi_delta_a_m": 3.0 * math.pi * (a_c - a),
        }
    out["cw_error_law"] = law

    # --- 2. Targeting: corrected vs raw CW -------------------------------------------
    targeting = {}
    for rho in (100.0, 1000.0, 10000.0):
        r0 = np.array([0.0, -rho, 0.0])
        rf = np.array([0.0, -rho / 4.0, 0.0])
        raw = raw_cw_terminal_miss_m(r_t, v_t, r0, np.zeros(3), rf, np.zeros(3), 0.5 * period)
        res = correct_two_impulse_transfer(r_t, v_t, r0, np.zeros(3), rf, np.zeros(3), 0.5 * period)
        targeting[f"{rho:.0f}"] = {
            "raw_cw_miss_m": raw,
            "corrected_miss_m": res.final_residual_m,
            "improvement_x": raw / res.final_residual_m,
        }
    out["targeting"] = targeting

    # --- 3. Propagator vs an independent analytic Kepler reference --------------------
    times = np.linspace(0.0, 86400.0, 1441)
    numeric = propagate_two_body(np.concatenate((r_t, v_t)), times)
    analytic = np.empty_like(numeric)
    for k, t in enumerate(times):
        ecc_anom = n * float(t)  # circular: E == M == true anomaly
        f, g = math.cos(ecc_anom), math.sin(ecc_anom) / n
        fdot, gdot = -n * math.sin(ecc_anom), math.cos(ecc_anom)
        analytic[k, :3] = f * r_t + g * v_t
        analytic[k, 3:] = fdot * r_t + gdot * v_t
    dr = np.linalg.norm(numeric[:, :3] - analytic[:, :3], axis=1)
    out["propagator_vs_kepler"] = {
        "span_s": float(times[-1]),
        "span_orbits": float(times[-1] / period),
        "max_position_diff_m": float(dr.max()),
        "rms_position_diff_m": float(np.sqrt((dr**2).mean())),
    }

    # --- 4. Baseline comparison -------------------------------------------------------
    rho_far = 10000.0
    problem = RendezvousProblem(
        r_target0_eci_m=r_t,
        v_target0_eci_m_s=v_t,
        r0_hill_m=np.array([0.0, -rho_far, 0.0]),
        v0_hill_m_s=np.zeros(3),
        rf_hill_m=np.array([0.0, -rho_far / 4.0, 0.0]),
        vf_hill_m_s=np.zeros(3),
        tof_s=0.4 * period,
    )
    comparison = compare_baselines(problem)
    table = comparison.render_table()
    out["cw_bound_at_10km_0p4_orbits_m"] = conservative_cw_error_bound_m(rho_far, a, 0.4)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "headline.json").write_text(json.dumps(out, indent=2) + "\n")
    (results_dir / "baseline_comparison.txt").write_text(table + "\n")

    print(json.dumps(out, indent=2))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
