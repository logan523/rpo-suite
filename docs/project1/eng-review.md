# Engineering review — full codebase

Reviewed at 26 source modules / 18,295 lines, 22 test files / 17,085 lines, 1,657 fast tests.
Every finding below was **demonstrated numerically or reproduced**, not inferred. Findings I
could not demonstrate are excluded rather than listed at low confidence.

## Method note, and a failure worth recording

The review was first attempted as five parallel agents across independent dimensions
(correctness, test integrity, claims audit, architecture). **All five stalled and were killed
by a watchdog after 600 s without progress**, on a heavily loaded machine, returning nothing
usable. The review was then done directly and sequentially.

One stalled agent **left a mutation in the source tree** — `elements.py` was still carrying an
injected defect when it died. That was caught only because `git status --porcelain` was checked
before continuing. Any process that injects defects into source must verify tree cleanliness
after every run, and must not assume a `try/finally` survives an external kill.

That abandoned mutant turned out to be the review's most valuable finding.

---

## P0 — the first command a reviewer types failed

**`packages/rpo-core/tests/test_plotting.py`, `.github/workflows/ci.yml`, `README.md`**

`test_plotting.py` imported matplotlib at module scope. matplotlib lives in the optional `viz`
extra, deliberately, so a headless Monte Carlo run pulls no plotting stack. On a clean clone the
documented quickstart

```
uv sync && uv run pytest -q
```

raised `ModuleNotFoundError` **during collection**, which pytest treats as fatal: **zero tests
ran**. The first impression a hiring manager gets is a stack trace.

**CI carried the identical defect** (`uv sync --locked`, no extra) and would have failed on its
first run. It had never run, because nothing had been pushed — a gate that has never executed is
indistinguishable from a working one.

A first attempt to verify this appeared to pass, because a CLI invocation with `--extra viz` in
between had installed matplotlib and masked it. Only a genuinely fresh clone reproduced it.

**Fixed:** `pytest.importorskip` so a missing optional dependency skips that module instead of
aborting the suite; CI installs `--extra viz` so the plotting layer is exercised rather than
silently skipped; CI asserts nothing was skipped, because a skipped suite and a passing suite
look identical in the summary line; README corrected. Verified from a clean clone both ways:
1664 passed / 3 skipped without the extra, 1657 fast-tier passed with it.

## P1 — an uncovered guard in angle wrapping

**`packages/rpo-core/src/rpo_core/elements.py:273`**

```python
return 0.0 if wrapped >= _TWO_PI else wrapped
```

Deleting the upper guard leaves **all 383 element tests passing**. It is not dead code.

Demonstrated: for any angle of magnitude below about half an ulp of 2π — which `atan2` produces
routinely near zero, and which therefore reaches this function as a near-zero true anomaly, RAAN
or argument of periapsis — adding 2π to the tiny negative `fmod` result rounds to exactly 2π.

| input | guarded | unguarded |
|---|---|---|
| −1e-17 | `0.0` | `6.283185307179586` (== 2π) |
| −2.2e-16 | `0.0` | `6.283185307179586` |
| −4e-16 | `0.0` | `6.283185307179586` |

Every downstream comparison assuming the half-open interval `[0, 2π)` is then reading a value
outside the range it was promised.

**Fixed:** parametrised regression test plus an idempotence complement over 5,000 seeded angles
spanning ±40π. Confirmed the new test kills the mutant (5 failures where there were 0).

## P2 — validator duplication from parallel construction

**14 modules, 30 distinct private helpers, 26 shape-check call sites.**

Near-duplicates written independently: `_as_vec3` ×3, `_vec3` ×2, `_as_state6` ×2,
`_validate_times` ×2, `_validate_trajectory` ×2, `_validate_seed` ×2, `_validate_positive` ×2.

This is the predicted cost of building with parallel agents forbidden from touching shared
files — local quality stays high, global coherence drifts. It is not a defect today: each
implementation is correct and tested. It is a maintenance liability, and it will grow when
`rpo-rl` and `rpo-inspect` are added.

**Not fixed.** Consolidating into a `rpo_core._validate` module is a mechanical refactor
touching 14 files, and it belongs in its own commit with the full suite as the gate, not folded
into a review. Logged as backlog X7.

## P3 — `__all__` unsorted

**Fixed.** 71 names, now sorted; verified no missing and no undeclared entries.

---

## Verified clean

- **matplotlib isolation is real, not aspirational.** Checked by importing each of 7 entry
  points into a fresh module table and inspecting `sys.modules`: `rpo_core`, `propagate`,
  `montecarlo`, `targeting`, `metrics`, `rpo_traj.plan`, `rpo_traj.campaign` — matplotlib
  absent in every one.
- **`__all__` is accurate**: no declared-but-missing names, no public-but-undeclared names.
- **Fast tier is 1,657 tests in 40 s**, well inside the 5-minute CI budget, so the gate that
  runs on every push is the strong one rather than a token subset.
- **Tree cleanliness after mutation work** verified with `git status --porcelain`.

## Not covered by this review

The parallel agents that failed were to have covered: a systematic 15-mutant sweep across the
numerical core, a full claims-and-citations audit of every number in prose, and an
extensibility assessment against the planned `rpo-rl` and `rpo-inspect` packages. Those remain
**undone**, not passed. They should be re-run on an unloaded machine, sequentially.
