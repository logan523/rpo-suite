# Project 1 — issue backlog

Ready to create with `gh issue create`. Each item names its phase (see
`docs/build-phases.md`), the SRS requirements it closes (`docs/project1/srs.md`), its
dependencies, and an acceptance criterion that is a test or an artefact — never "looks right".

Labels: `phase:A`–`phase:D` · `type:feature|test|docs|validation` · `blocked-by:#N`

---

## Phase A — MVP: make it runnable

**A1 · Scenario configuration and reproducible run directories** — closes N-1, N-2, F-1.8
Pydantic v2 scenario models, YAML loading, stable config hash, run directory with
`provenance.json`. Cross-field validation rejects hold points inside the keep-out sphere and
times of flight at CW singularities.
*Accept:* the same config and seed produce a bitwise identical `metrics.json` twice; a typo'd
YAML key is rejected by name.

**A2 · Safety constraint evaluation** — closes F-4.1…F-4.6
Keep-out sphere, approach ellipsoid, approach cone with activation range, closing-velocity
limit, aggregate report with worst-case values and first-violation time. Sub-sample
refinement so discrete sampling does not understate a breach.
*Accept:* a trajectory tangent to the sphere reports clearance exactly 0; the refined minimum
is measurably closer to an analytically known between-samples minimum than the raw sample.

**A3 · Metrics and plotting** — closes §4 of the SRS
Δv, TOF, terminal errors, minimum KOZ distance, maximum closing velocity, violation count →
`metrics.json`. Matplotlib suite: Hill-frame trajectory with KOZ and corridor overlaid,
range and range-rate vs time, Δv budget.
*Accept:* every plotted number traces to a field in `metrics.json`; figures render headless.
*Blocked by:* A1, A2.

**A4 · `rpo-traj` package and CLI** — completes the MVP
`uv run rpo-traj plan configs/vbar_baseline.yaml --seed 42` → run directory with metrics,
trajectory, plots, provenance.
*Accept:* runs from a clean clone with no STK and no network; two identical invocations diff
clean. *Blocked by:* A1, A2, A3.

---

## Phase B — targeting depth

**B1 · Orbital elements** — Cartesian ⇄ classical, singular cases raise typed errors with
well-defined alternates (argument of latitude, true longitude, longitude of periapsis).
*Accept:* seeded round trip over thousands of orbits within a measured bound; elements
invariant under one period of validated two-body propagation while true anomaly sweeps 2π.

**B2 · Lambert solver** — closes F-2.3
*Accept:* propagating the returned `v₁` for the requested TOF arrives at `r₂` within a
measured bound, across a spread of transfer angles and eccentricities; matches the Hohmann
closed form in the 180° limit; non-convergence raises with iteration count and residual.

**B3 · Differential correction on nonlinear dynamics** — closes F-2.4
Shooting method with numerical Jacobian, targeting a terminal relative state under M5.
*Accept:* terminal miss driven below tolerance from a CW first guess; a deliberately
unreachable target raises rather than returning the last iterate. *Blocked by:* B2.

**B4 · Δv-vs-TOF optimisation and three-baseline comparison** — closes F-2.7, F-7.1…F-7.4
Hohmann/phasing baseline, Lambert baseline, optimised trajectory, one table, identical
scenarios. *Accept:* the table exists with all required metrics and no superiority claim
appears anywhere without it. *Blocked by:* B3.

---

## Phase C — fidelity and uncertainty

**C1 · J2 perturbation** — closes F-1.6
*Accept:* secular RAAN drift within 1 % of `−(3/2)·n·J₂·(Rₑ/p)²·cos i`. This is the highest
value single physics test in the project.

**C2 · Atmospheric drag** — closes F-1.7. Exponential or NRLMSISE-lite density with stated
provenance. *Accept:* decay rate within an order of magnitude of published ISS reboost cadence;
the density model's limitations stated in the module docstring.

**C3 · Finite burns** — closes F-2.6
*Accept:* converges to the impulsive solution as thrust → ∞ at fixed total impulse, and the
convergence *rate* is asserted, not just the limit.

**C4 · Monte Carlo utilities** — closes F-5.3
Dispersion spec → seeded runs → retained metrics → summary, with the four-part separation
(nominal config / dispersions / per-run execution / post-run analysis).
*Accept:* same seed gives bitwise identical metrics; runs are independent of execution order.

**C5 · Burn execution and navigation error** — closes F-5.1, F-5.2, F-5.4, F-5.5
*Accept:* Monte Carlo sample covariance converges to the linear-covariance prediction
`P⁺ = ΦPΦᵀ + Q` in the small-dispersion limit; success rate reported with a confidence
interval, never as a bare fraction. *Blocked by:* C4.

**C6 · Safety analysis under dispersion**
*Accept:* keep-out-zone breach probability reported with its confidence interval and the
number of samples behind it.

---

## Phase D — validation and polish

**D1 · GMAT cross-validation (licence-free path)** — closes F-6.5
*Accept:* max position difference vs GMAT reported in metres with a stated cause for the
residual; golden files vendored so CI needs no GMAT install.

**D2 · STK + Astrogator integration** — closes F-6.1…F-6.4
Programmatic scenario creation, MCS representation, manoeuvre export, data-provider readback.
*Accept:* Python-vs-STK difference reported in metres with a stated cause. **Every number in
the README must remain reproducible without an STK licence** — STK confirms, never produces.

**D3 · README, technical write-up, demo, results tables**
*Accept:* a reader with no STK licence reproduces every headline number from a clean clone.

---

## Cross-cutting

**X1 · Mutation-test the test suite.** Deliberately introduce a sign error in the Hill frame,
a dropped secular term in CW, and an off-by-one in the constraint sampler; confirm the suite
catches each. A gate that cannot fail is not a gate.

**X2 · Derive the `6π·ρ²/r` law analytically.** Currently measured, not derived. Likely route:
secular along-track drift of the semi-major-axis difference induced by the initial offset.
*Accept:* derivation reproduces the coefficient `6π`, or the discrepancy is explained.

**X3 · Eccentricity error study.** The CW validity study isolates linearisation error using a
circular target. Quantify the separate error from reference-orbit eccentricity.

**X4 · Public API surface consistency.** `rpo_core/__init__.py` re-exports constants and
exceptions but not frames or relative motion. Pick one convention.

**X5 · Scale-relative feasibility tolerance** in `two_impulse_transfer` (currently absolute).

**X6 · Verify the Curtis Example 5.2 citation.** `test_lambert.py` pins velocity values
recalled rather than read from the source. The physics is independently verified (propagating
`(r1, v1)` arrives at `r2` within 1.17e-5 m), but the *citation* is unconfirmed. Either check
a physical copy and remove the warning block, or drop the reference and keep the case as an
unattributed fixture. Publishing an incorrect textbook citation in a portfolio repository
costs more credibility than the test is worth.
