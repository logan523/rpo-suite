# Inertial frame: what is approximated, and what it costs

This repository uses an Earth-centred **pseudo-inertial** frame, taken as GCRF-approximated.
Precession, nutation, polar motion and frame-tie corrections are **neglected**. That is a
modelling choice, and this page quantifies it rather than leaving it as a disclaimer.

## The error budget

| Term | Magnitude over 1 day in LEO | Note |
|---|---|---|
| General precession | 4.54 m | 50.2879 ″/yr |
| Nutation | 4.94 m | 0.15 ″/day drift, saturating at the 17.2 ″ / 18.6 yr amplitude |
| **Total, 1 day** | **≈ 9.5 m** | grows ≈9.5 m/day, then ≈4.54 m/day after ~115 days |
| EME2000 ↔ GCRF frame bias | 0.76 m, constant | 23 mas |
| Polar motion | excluded | a terrestrial-tie effect; does not enter an inertial-frame comparison |

Over 7 days: ≈66 m.

## Why this dominates everything else

The numerical propagator agrees with an **independent analytic Kepler / Lagrange f-and-g
solution** to **8.4e-05 m** over 86 400 s (15.5 orbits) at `rtol = atol = 1e-12`, with the
difference almost entirely along-track (radial 1.1e-05 m, along-track 8.4e-05 m, cross-track
1.5e-08 m).

So the frame approximation is roughly **1.1e+05 times larger than the integrator error.**

The consequence is concrete and worth stating plainly: for any comparison against a rigorous
tool (STK, GMAT, Orekit) beyond a few hours, a disagreement is **measuring this frame
simplification, not the dynamics**. Tightening integrator tolerances cannot help; only
implementing a real frame transformation can. `rpo_core.validation` carries this budget in
every comparison report and flags whether an observed difference falls inside it.

## When this matters, and when it does not

- **Proximity operations over a fraction of an orbit** — the regime this suite targets. The
  frame error is common-mode between two vehicles a kilometre apart and very nearly cancels in
  the relative state. Not a concern.
- **Absolute ephemeris comparison over days** — dominant. Report it, do not explain it away.
- **Anything requiring an Earth-fixed frame** (ground station access, geodetic coordinates) —
  polar motion and Earth rotation would have to be modelled. Out of scope here.
