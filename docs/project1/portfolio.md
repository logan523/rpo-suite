# Portfolio material

Drafts for CV, applications and interviews. Every number here traces to
`scripts/generate_results.py` or a named test.

## 100-word description

> **rpo-suite** is a rendezvous and proximity operations toolkit: a validated astrodynamics core
> plus a mission planner that designs a chaser's approach to a target in LEO and reports how much
> to trust the answer. Every model is measured against an independent oracle — the propagator
> against an analytic Kepler solution to 8.4e-05 m, Lambert against the propagator, J2 against
> closed-form secular rates to 0.13 %. The Clohessy-Wiltshire linearisation error was measured,
> then derived in closed form as 6πρ²/r. 1,663 tests assert conservation laws and limiting cases
> rather than stored output. Python, NumPy, SciPy, mypy-strict, Apache-2.0.

## Three resume bullets

- Built a validated astrodynamics core (26 modules, 1,663 tests) for rendezvous and proximity
  operations, validating each model against an **independent oracle** rather than its own output —
  numerical propagator against an analytic Kepler solution to **8.4e-05 m over 15.5 orbits**, and
  J2 secular node regression against closed form to **0.13 %**, cross-checked by reproducing
  sun-synchronous inclination to 0.014 %.
- **Measured and then derived in closed form** the Clohessy-Wiltshire linearisation error,
  `6πρ²/r`, reproducing to six significant figures across three altitudes; used it to set a
  conservative validity bound and to show a differential-correction targeting scheme cutting
  terminal miss from **122.5 m to 7.5e-04 m** at 10 km separation.
- Quantified error-source dominance in dispersed Monte Carlo campaigns with Wilson confidence
  intervals, showing **navigation error exceeds delivery dispersion by five orders of magnitude**
  (1 mm/s of velocity knowledge error → 8.4 m of along-track miss), redirecting design margin from
  actuation to estimation.

## Five interview questions and model answers

**1. Your CW error law was empirical at first. How did you derive it, and why does the derivation
matter more than the measurement?**

A chaser at a pure along-track Hill offset, at rest in the rotating frame, is not co-orbital —
which is exactly what the linear model cannot see, since CW reports it as stationary. Its radius
exceeds the reference by `ρ²/2r`, and rigid co-rotation adds a radial velocity `nρ`. Each
contributes `n²ρ²/2` to specific energy, so `Δa = 2ρ²/r`. Substituting into the classical
`−3π·Δa` secular drift per revolution gives `−6πρ²/r`. The derivation matters because a fitted
coefficient is only valid where you fitted it; a structural one tells you the scaling holds
generally, and it explains *what* the error is — an unmodelled energy difference, not numerical
noise.

**2. You have a keep-out sphere and an approach corridor. Your baseline scenario violates the
corridor. Why ship it?**

Because it is geometry, not a tuning error. A half-period two-impulse V-bar hop bulges radially by
exactly `Δy/4` — independent of altitude and transfer time. For a 750 m hop that is 187.5 m,
giving 20.56° against a 10° corridor. Widening the cone to make the demo pass would hide a real
constraint on mission design: single large transfers cannot respect tight corridors, which is why
real proximity operations use staged short hops. The planner reports the violation and exits 1. A
baseline that fails honestly is more useful than one tuned to succeed.

**3. How do you know your tests actually test anything?**

I mutation-tested them: inject a deliberate defect, run the suite, confirm it fails, restore.
Across this build seven tests were found that passed while measuring something other than what
they claimed — a convergence sweep whose nested sample grids shared points, a `match=` string that
numpy's own exception also satisfied, a tolerance test written in terms of the constant it was
checking, Hill-frame tests using a state where the rotation is the identity matrix. The last one
let a transposed rotation survive the entire suite; a quarter orbit later it gives the opposite
V-bar direction, so a burn would brake instead of accelerate. Coverage tells you a line executed.
Mutation testing tells you an assertion would notice if it were wrong.

**4. What is the limiting error source in your project, and what would you do about it?**

The frame model, not the dynamics. My inertial frame is GCRF-approximated with precession,
nutation and polar motion neglected, which costs about 9.5 m per day in LEO. The propagator agrees
with an analytic reference to 8.4e-05 m, so the frame simplification is 1.1e+05 times larger. Any
disagreement with a rigorous tool beyond a few hours is measuring the frame, and tightening
integrator tolerances cannot help. I would implement a real IAU frame transformation before
attempting any absolute-ephemeris validation. For relative motion over a fraction of an orbit it
is common-mode between the vehicles and nearly cancels — which is why it has not blocked the
mission-level results.

**5. You had an STK licence available but the repository contains no STK code. Explain.**

The development machine is macOS with no STK, no GMAT, and no `pystk` bindings, so I could not
execute or verify a single API call. Writing code against a remembered API signature and marking
the requirement satisfied would be fabrication — and an experienced reviewer with STK installed
would find it in minutes. Instead I built the half that is honestly testable: the ephemeris
ingest, frame and epoch alignment, and the comparison harness, validated against an analytic
Kepler oracle. The requirements for STK integration are marked OPEN, not met, and every report
carries a flag that is False for anything not produced by a real tool run.

## Demo

```bash
uv run --extra viz python scripts/demo.py
```

Runs the baseline plan, the validity study, and the baseline comparison end to end, printing the
headline numbers and writing figures. Roughly 60 seconds.

## Follow-up improvements

1. Rigorous IAU frame transformation — the binding constraint on external validation.
2. Execute the GMAT comparison; implement STK/Astrogator on licensed hardware.
3. Multi-burn corridor-respecting approach, motivated by the baseline's deliberate failure.
4. Square-root covariance formulation — the current chain loses positive definiteness after 43
   half-period steps.
5. Consolidate 30 duplicated private validators (backlog X7).
6. Verify the Curtis Example 5.2 citation against a physical copy (backlog X6).
7. `rpo-rl`: Gymnasium environment, LQR/MPC baselines, safety-filtered RL.
8. `rpo-inspect`: sensor geometry, lighting, access windows, GA scheduler.
