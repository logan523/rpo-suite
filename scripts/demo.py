"""End-to-end demo: plan, validate, compare. Roughly 60 seconds.

Run:
    uv run --extra viz python scripts/demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(title: str, argv: list[str]) -> int:
    """Run one demo step, echoing its command and output."""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print("$ " + " ".join(argv) + "\n", flush=True)
    return subprocess.run(argv, cwd=ROOT, check=False).returncode


def main() -> int:
    """Run the three demo steps and summarise."""
    print("rpo-suite demo — every number here is recomputed live, nothing is cached.")

    code = _run(
        "1/3  Plan the baseline rendezvous (expect exit 1: it violates its own corridor)",
        [
            "uv",
            "run",
            "--extra",
            "viz",
            "rpo-traj",
            "plan",
            "configs/vbar_baseline.yaml",
            "--seed",
            "42",
        ],
    )
    print(
        f"\n-> exit {code}: a reported constraint violation, not a crash. "
        "Outputs were still written."
    )

    _run(
        "2/3  Measure the Clohessy-Wiltshire validity envelope against nonlinear truth",
        ["uv", "run", "--extra", "viz", "python", "scripts/cw_validity_study.py"],
    )

    _run(
        "3/3  Compare rendezvous baselines, scored identically under nonlinear dynamics",
        ["uv", "run", "--extra", "viz", "python", "scripts/generate_results.py"],
    )

    print("\n" + "=" * 78)
    print("  Artefacts written to results/ — metrics.json, provenance.json, figures.")
    print("  Full technical write-up: docs/project1/write-up.md")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
