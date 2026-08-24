# Engineering standard for rpo-suite

Every module in this repository follows these rules. They are not style preferences; they
are what makes the numerical results defensible.

## Units and naming
- SI only in the core: metres, seconds, radians, kilograms. No attached unit objects.
- Units live in variable names: `r_eci_m`, `v_hill_m_s`, `n_rad_s`, `tof_s`, `mu_m3_s2`.
- Conversions happen at the I/O boundary, never inside numerics.

## Frames
Locked in `docs/conventions.md`. Hill/LVLH: `x` radial outward (R-bar), `y` along-track
(V-bar), `z` the **positive** orbit normal (`z = x × y = r̂ × v̂ = ĥ`). Chaser trailing the
target sits at negative `y`. Relative velocity is the rotating-frame derivative; converting
from inertial requires the transport theorem `R @ (dv - ω × dr)`.

## Docstrings
- Module docstring **states the equations before the code implements them**, in a
  `The equations` section, with a `Validity` section naming what the model assumes and what
  it neglects.
- Every public function has a numpy-style docstring with Parameters, Returns, and Raises.
- Explain *why*, not *what*. A comment that restates the code is deleted.

## Failure behaviour
- Numerical routines that cannot produce a valid answer **raise a typed exception** from
  `rpo_core.exceptions`. They never return a plausible-looking wrong value.
- Error messages carry the numbers that motivated them (condition number, residual,
  requested vs achievable), so a user can act on them without a debugger.
- Never swallow an integrator or optimiser failure into a partial result.

## Tests — the important part
Tests assert **closed-form solutions, limiting cases, and conservation laws**. They do not
assert golden numbers produced by an earlier run of the same code: a regression suite that
only compares against its own past output cannot detect an error that was present from the
start.

Required per module, where they apply:
- A known analytic solution the implementation must reproduce.
- A conservation law (energy, angular momentum, orthonormality, determinant).
- A limiting case that degenerates to something independently known.
- A **complement test**: prove the check is a knife edge, not a plateau. If you test that
  condition X gives result Y, also test that violating X gives a measurably different
  result. Otherwise the test would still pass if the term were dropped entirely.
- Every raise path, with `pytest.raises` matching on message content.
- Every input-validation branch (wrong shape, non-finite, non-positive).

Set numerical tolerances **from measurement**, not from feel. If you pick a bound, first
print the actual value and choose a bound with stated headroom, and say in a comment where
the number came from. Where possible assert convergence *behaviour* (monotone improvement
as tolerance tightens) rather than a single hand-picked threshold.

Mark tests: `@pytest.mark.unit` (fast, isolated), `@pytest.mark.integration` (multi-module),
`@pytest.mark.slow` (excluded from the fast CI job).

## Determinism
Every stochastic entry point takes an explicit `numpy.random.Generator`. No global RNG.

## Typing
`mypy --strict` must pass. Use `numpy.typing.NDArray[np.float64]`, accept `npt.ArrayLike`
at public boundaries and coerce internally.

## Gates — all four must pass before you report done
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```
