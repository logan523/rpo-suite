# Project 1 — Software Requirements Specification

**RPO Trajectory Design and Validation.** Requirements are testable statements; each carries
an acceptance criterion that maps to a test or a produced artefact. `SHALL` is binding,
`SHOULD` is a strong preference, `MAY` is optional.

Status legend: **[MET]** implemented and tested · **[PART]** partially met · **[OPEN]** not started.

## 1. Scope and intent

### 1.1 Purpose
Plan and validate a chaser spacecraft rendezvous with a target in LEO, from a phasing orbit
through terminal approach, under explicit safety constraints and quantified uncertainty.

### 1.2 Explicit non-goals
This is **not** flight software. It is not verified to DO-178C, NPR 7150.2, or ECSS. It
produces no operational product. See `DISCLAIMER.md`.

### 1.3 Reference scenario
ISS-like: 420 km circular, 51.6° inclination, `a = 6 798 137 m`, `n = 1.1264e-3 rad/s`,
`T = 5578 s`. Constraint values are publicly documented, order-of-magnitude representative
figures, not the requirements of any real programme.

## 2. Functional requirements

### 2.1 Dynamics and propagation
| ID | Requirement | Acceptance criterion | Status |
|----|-------------|---------------------|--------|
| F-1.1 | SHALL propagate absolute two-body inertial motion | Specific energy and angular momentum conserved to < 1e-10 relative over 10 orbits | **[MET]** |
| F-1.2 | SHALL propagate linear relative motion via the CW state transition matrix | `Φ(0)=I` exact; `Φ(t)Φ(−t)=I`; composition over split intervals | **[MET]** |
| F-1.3 | SHALL propagate nonlinear relative motion by differencing two absolute orbits | Chaser initialised on the target stays on it to < 1e-6 m over one orbit | **[MET]** |
| F-1.4 | SHALL quantify CW linearisation error against F-1.3 | Measured law `6π·ρ²/r`, reproducible to 6 s.f. across 3 altitudes | **[MET]** |
| F-1.5 | SHALL warn when a scenario exceeds the measured CW envelope | Warning fires above tolerance, silent for the baseline scenario | **[MET]** |
| F-1.6 | SHALL model J2 secular perturbation | RAAN drift within 1 % of `−(3/2)·n·J2·(Rₑ/p)²·cos i` | **[OPEN]** |
| F-1.7 | SHOULD model atmospheric drag | Decay rate within an order of magnitude of published ISS reboost cadence | **[OPEN]** |
| F-1.8 | SHALL expose integrator tolerances as configuration | Tolerance appears in `IntegratorConfig`, never as a module constant | **[MET]** |

### 2.2 Targeting and manoeuvre design
| ID | Requirement | Acceptance criterion | Status |
|----|-------------|---------------------|--------|
| F-2.1 | SHALL solve the two-impulse CW rendezvous problem | Solved Δv reproduces the commanded terminal state to < 1e-9 m | **[MET]** |
| F-2.2 | SHALL raise, not return a wrong answer, at singular transfer times | Typed error at `k·T` (in-plane) and `k·T/2` (cross-track) | **[MET]** |
| F-2.3 | SHALL solve Lambert's problem | Propagating the returned `v1` for the TOF arrives at `r2`; **measured worst 7.09e-4 m** across 20 cases, integrator-limited | **[MET]** |
| F-2.4 | SHALL perform differential correction on a nonlinear trajectory | Terminal miss reduced below tolerance; non-convergence raises | **[OPEN]** |
| F-2.5 | SHALL support multiple manoeuvre opportunities | A ≥3-burn sequence plans and executes end to end | **[OPEN]** |
| F-2.6 | SHALL model finite burns | Finite-burn result converges to the impulsive result as thrust → ∞ | **[OPEN]** |
| F-2.7 | SHALL optimise Δv against time of flight | A Pareto front is produced and archived | **[OPEN]** |

### 2.3 Mission phases
| ID | Requirement | Status |
|----|-------------|--------|
| F-3.1 | SHALL represent a phasing/drift orbit | **[OPEN]** |
| F-3.2 | SHALL represent far-range rendezvous (Lambert regime, outside CW validity) | **[OPEN]** |
| F-3.3 | SHALL represent relative-motion approach with hold points | **[PART]** — hold points in config |
| F-3.4 | SHALL support V-bar and R-bar approach geometries | **[OPEN]** |
| F-3.5 | SHALL generate retreat/abort trajectories from any hold point | **[OPEN]** |

### 2.4 Safety constraints
| ID | Requirement | Acceptance criterion | Status |
|----|-------------|---------------------|--------|
| F-4.1 | SHALL evaluate keep-out-sphere clearance along a trajectory | Analytic tangent case gives clearance exactly 0 | **[MET]** |
| F-4.2 | SHALL evaluate approach-cone containment inside an activation range | Boundary, inside, outside, and beyond-activation cases distinguished | **[MET]** |
| F-4.3 | SHALL evaluate closing velocity against a limit | Pure radial approach at known speed reports that speed | **[MET]** |
| F-4.4 | SHALL evaluate approach-ellipsoid containment | Quadratic form ≤ 1 inside | **[MET]** |
| F-4.5 | SHALL report violation counts, worst-case values, and first-violation time | Present in the aggregate report | **[MET]** |
| F-4.6 | SHALL NOT understate a violation due to discrete sampling | Sub-sample refinement is applied and its residual limitation documented | **[MET]** |

### 2.5 Uncertainty
| ID | Requirement | Status |
|----|-------------|--------|
| F-5.1 | SHALL model burn execution error (magnitude and pointing) | **[OPEN]** |
| F-5.2 | SHALL model navigation error (position, velocity, bias) | **[OPEN]** |
| F-5.3 | SHALL run seeded Monte Carlo campaigns | **[OPEN]** |
| F-5.4 | SHALL report success rate with a confidence interval | **[OPEN]** |
| F-5.5 | SHALL report sensitivity to burn and navigation error | **[OPEN]** |

### 2.6 Validation against external tools
| ID | Requirement | Status |
|----|-------------|--------|
| F-6.1 | SHALL generate an STK scenario programmatically | **[OPEN]** |
| F-6.2 | SHALL represent the plan as an Astrogator Mission Control Sequence | **[OPEN]** |
| F-6.3 | SHALL export Python-generated manoeuvre states to STK | **[OPEN]** |
| F-6.4 | SHALL report the Python-vs-STK position difference in metres with a stated cause | **[OPEN]** |
| F-6.5 | SHALL provide a licence-free validation path (GMAT) | **[OPEN]** |

### 2.7 Baselines and reporting
| ID | Requirement | Status |
|----|-------------|--------|
| F-7.1 | SHALL implement a Hohmann/phasing baseline | **[OPEN]** |
| F-7.2 | SHALL implement a Lambert-based rendezvous baseline | **[OPEN]** |
| F-7.3 | SHALL produce an optimised trajectory | **[OPEN]** |
| F-7.4 | SHALL compare all three in one table on identical scenarios | **[OPEN]** |
| F-7.5 | SHALL NOT claim superiority without that quantitative comparison | **[MET]** (policy) |

## 3. Non-functional requirements

| ID | Requirement | Acceptance criterion | Status |
|----|-------------|---------------------|--------|
| N-1 | Reproducibility: identical config + seed → bitwise identical metrics | Two runs diff clean | **[MET]** |
| N-2 | Provenance: every run records git SHA, config hash, seed, package versions | `provenance.json` present and complete | **[MET]** |
| N-3 | The core SHALL run with no STK and no network | Full suite passes from a clean clone | **[MET]** |
| N-4 | `mypy --strict` clean on `rpo-core` | CI gate | **[MET]** |
| N-5 | `ruff check` and `ruff format --check` clean | CI gate | **[MET]** |
| N-6 | Fast CI job under 5 minutes | Measured in CI | **[MET]** |
| N-7 | Numerical routines raise typed errors rather than returning wrong values | Every raise path tested | **[MET]** |
| N-8 | No global RNG state; every stochastic entry point takes a `Generator` | Code review + test | **[MET]** |
| N-9 | Tests assert closed-form/limiting-case results, not stored self-output | Code review | **[MET]** |
| N-10 | Every quoted numerical result survives a tolerance sweep | Convergence test per propagator | **[PART]** |

## 4. Required metrics (the results table)

Total Δv · time of flight · terminal position error · terminal velocity error · minimum
keep-out-zone distance · maximum closing velocity · constraint violation count · Monte Carlo
success rate · sensitivity to burn and navigation error · Python-vs-STK difference.

## 5. Traceability

`docs/build-phases.md` maps each phase to the requirement IDs it closes. Phase A closes the
`[PART]` entries in §2.4 and §3; Phase B closes §2.2; Phase C closes §2.1 (F-1.6/1.7) and
§2.5; Phase D closes §2.6 and §2.7.
