# RPO Trajectory Design and Validation — technical write-up

## Problem

Plan a chaser spacecraft's rendezvous with a target in low Earth orbit, and know how much to
trust the answer. The second half is the harder and more interesting one: proximity-operations
guidance rests on a linearisation whose error is rarely quantified, and constraint margins are
often quoted from models nobody measured.

## Approach

Build in increasing fidelity, and give every model an oracle that did not produce it.

1. **Two-body propagation** — validated by conservation laws and, independently, by an analytic
   Kepler / Lagrange f–g solution sharing no code path. Agreement 8.4e-05 m over 15.5 orbits.
2. **Clohessy-Wiltshire relative motion** — validated by closed-form properties (drift-free
   condition, closed 2:1 ellipse, cross-track decoupling) and by differencing two independently
   propagated orbits.
3. **Targeting** — CW two-impulse solve, then differential correction onto nonlinear dynamics.
4. **Perturbations, finite burns, uncertainty** — J2 against closed-form secular rates, drag,
   mass-flow-coupled finite burns, Monte Carlo with navigation and execution error.

## Results

### 1. The linearisation error has a closed form

Measured, then derived: `err(one orbit) = 6π·ρ²/r`. Six significant figures across three
altitudes and two separations.

The derivation matters more than the measurement. A chaser placed at a pure along-track Hill
offset *at rest in the rotating frame* is not co-orbital — a fact the linear model cannot see,
because CW reports it as stationary. Its radius is larger by `ρ²/2r` and rigid co-rotation adds a
radial velocity `nρ`; each contributes `n²ρ²/2` to specific energy, giving `Δa = 2ρ²/r`. The
classical `−3π·Δa` secular drift per revolution then reproduces the coefficient exactly.

**Consequence.** With an error budget of 2.5 % of a 200 m keep-out sphere, CW is trustworthy to
≈1,100 m separation over one orbit on the conservative bound. Far-range work at 10 km incurs
277 m — larger than the keep-out sphere itself. That is why the project uses Lambert for phasing
rather than stretching CW.

### 2. The estimator was optimistic where it mattered most

The first validity guard scaled the error law linearly in elapsed time and was documented as
conservative. Measurement disagreed: linear scaling **under-predicts by up to 1.23×** between 0.4
and 1.0 orbits — precisely the regime a half-orbit V-bar hop occupies. A safety factor of 1.5,
set from the measured worst case rather than chosen by feel, converts the estimate into a bound
that a 54-case grid test confirms is never exceeded.

### 3. Navigation error dominates, by five orders of magnitude

Terminal error is `−Φ` applied to the *estimation* error. A delivery dispersion the filter can
observe therefore very nearly cancels: 5 m of it costs 6.4e-11 m at the terminal point. Knowledge
error does not cancel, and at τ = π the coefficient `Φ_rv[1,1] = −3π/n = −8367 s` converts
**1 mm/s of velocity uncertainty into 8.4 m of along-track miss** — 17 % of the margin between
the arrival hold point and the keep-out sphere.

Per-family breach counts over dispersed campaigns: navigation 14/60, burn execution 3/60,
delivery 0/60. The engineering conclusion is that safety here is bought with estimation, not
actuation.

### 4. Finite-burn loss is negligible; impulse placement is not

At 22 N — a real hydrazine monopropellant thruster — the extra Δv from finite burn duration is
1.4e-07 m/s, 7e-07 of the command. Reporting that plainly is more useful than dressing it up.

What does matter is bookkeeping. Placing the equivalent impulse at the burn's Δv centroid rather
than at ignition reduces the position offset from 3.998 m to 0.34 mm at 1 N — a factor of 1.2e+04
for no propellant. The centroid is not the midpoint: `t̄/t_b = ½ + x/12 + x²/24`.

### 5. The baseline scenario fails, on purpose

A half-period two-impulse V-bar hop bulges radially by exactly `Δy/4`, independent of altitude
and transfer time. For the 750 m baseline hop that is 187.5 m, reaching 20.56° against a 10°
approach corridor. The planner reports the violation and exits 1.

This is not a bug to be tuned away. It is a hard geometric limit that explains why real proximity
operations use staged short hops rather than one large transfer.

## What limits the result

The **frame model**, not the dynamics. The inertial frame is GCRF-approximated with precession,
nutation and polar motion neglected, costing ≈9.5 m/day in LEO. Against an integrator agreement
of 8.4e-05 m that is 1.1e+05 times larger. Any disagreement with a rigorous tool beyond a few
hours is measuring the frame simplification. Tightening tolerances cannot help.

For proximity operations over a fraction of an orbit the frame error is common-mode between two
vehicles a kilometre apart and very nearly cancels in the relative state — so it dominates
absolute ephemeris comparison and is irrelevant to the mission this suite targets.

## Verification method

Tests assert closed-form solutions, conservation laws and limiting cases — never golden numbers
from a previous run of the same code, which cannot detect an error present from the start.

Each module was **mutation-tested**: deliberate defects injected, the suite run, the defect
restored. Across the build, seven classes of *self-satisfying test* were found and fixed —
each one passing while measuring something other than what it claimed:

1. A convergence sweep using nested sample counts, agreeing exactly because the grids shared
   points.
2. A positive-definiteness test matching on a string numpy's own error also contains.
3. A tolerance-band test written in terms of the constant defining the band.
4. Hill-frame tests using a reference state where the rotation is the identity, so a transposed
   rotation survived.
5. A breach-rate denominator that was unobservable because no run ever failed.
6. A no-op test running at a tolerance where the solver exits immediately.
7. An angle-wrapping guard whose removal left all 383 element tests passing.

That last one was found by an abandoned mutation left behind when a review process was killed
mid-run — caught only by checking the working tree before continuing.

## Follow-up work

- Implement a rigorous frame transformation; it is the binding constraint on external validation.
- Execute the GMAT comparison and, on licensed hardware, the STK/Astrogator layer.
- Multi-burn corridor-respecting approach, which the baseline's failure motivates.
- Consolidate 30 duplicated private validators (backlog X7).
- Square-root covariance formulation: the current chain loses positive definiteness after 43
  half-period steps.
- Verify the Curtis Example 5.2 citation against a physical copy (backlog X6).
