# Clohessy-Wiltshire validity envelope

**Measured**, not assumed. Reproduce with:

```bash
uv run --extra viz python scripts/cw_validity_study.py
```

Outputs `results/cw_validity/metrics.json` and `results/cw_validity/cw_validity.png`.

## Method

CW is a linearisation of the difference between two two-body orbits about a circular
reference. The reference implementation here does that difference *without* linearising:
propagate target and chaser independently under full two-body dynamics (DOP853,
rtol = atol = 1e-12), then express the difference in the target's Hill frame at every
output epoch, recomputing the frame from the propagated target state each time.

The target orbit is **exactly circular** throughout. That is deliberate: CW makes two
separate approximations — linearisation in the separation, and a circular reference — and
driving the study with a circular target isolates the first. Eccentricity is a separate
error source and gets a separate study. Conflating them yields an error budget that cannot
be attributed to anything.

Initial condition: chaser at pure along-track offset `-ρ`, zero initial Hill-frame relative
velocity.

## Result

The one-orbit position error obeys

```
err_1_orbit  =  6π · ρ² / r
```

to six significant figures, verified across 400 / 800 / 1500 km altitudes and 1 km / 10 km
separations. The coefficient lands on exactly `6π = 18.8496`.

The law is **measured, not derived.** An analytic derivation from the secular along-track
drift of the induced semi-major-axis difference looks tractable and is worth doing; it is
not claimed here.

| Separation ρ | 0.25 orbit | 0.5 orbit | 1 orbit | 2 orbits |
|---:|---:|---:|---:|---:|
| 10 m | 0.036 mm | 0.15 mm | 0.28 mm | 0.55 mm |
| 100 m | 3.3 mm | 15 mm | 28 mm | 55 mm |
| 1 km | 0.33 m | 1.46 m | 2.77 m | 5.5 m |
| 10 km | 33 m | 146 m | 277 m | 555 m |
| 100 km | 3.3 km | 15 km | 28 km | 55 km |

Error grows quadratically in separation. Growth in **time is not linear**, and an earlier
version of this document got that wrong in the dangerous direction.

`estimated_cw_error_m` scales linearly in elapsed orbits. Measured against truth:

| Orbits | Measured (m) | Linear estimate (m) | Estimate / measured |
|---:|---:|---:|---:|
| 0.125 | 0.073 | 0.347 | 4.73 — conservative |
| 0.25 | 0.335 | 0.693 | 2.07 — conservative |
| 0.5 | 1.455 | 1.386 | **0.95 — optimistic** |
| 0.75 | 2.531 | 2.080 | **0.82 — optimistic** |
| 1.0 | 2.773 | 2.773 | 1.00 |

(ρ = 1 km, 420 km circular.) The linear law is optimistic between roughly 0.4 and 1.0
orbits — **precisely where a half-orbit V-bar hop operates**. Treating it as conservative,
as this document previously claimed, would have under-warned in the MVP's own regime.

Scanning measured/linear across 400/800/1500 km altitudes, 100 m/1 km/5 km separations and
0.1–3.0 orbits, the worst under-prediction is **1.2253×** (800 km, 100 m, 0.7 orbits).
`CW_ERROR_SAFETY_FACTOR = 1.5` clears that with 1.22× headroom.

**Use `conservative_cw_error_bound_m` for any guard or go/no-go decision**, never
`estimated_cw_error_m`. `check_cw_validity` guards on the bound. A validity check that
under-warns is worse than no check at all.

## What this means for the reference scenario

Budget: **2.5 % of the 200 m keep-out sphere = 5 m**. (An earlier 1 % / 2 m budget was
arbitrary and sat badly — the conservative bound for the baseline is 2.08 m, so the flagship
scenario warned about itself while its measured error was only 1.455 m. The guard exists to
catch CW being used at 10 km where error is 277 m, not to police 2.00 versus 2.08 m against
a 200 m sphere.)

- Against the 5 m budget, CW is trustworthy out to **≈1342 m separation over one orbit** on
  the central estimate and **≈1096 m** on the conservative bound. (Against the old 2 m
  budget those figures were 849 m and 694 m.)
- The MVP baseline — a V-bar hop from −1000 m to −250 m over **half** an orbit — measures
  **1.455 m** of linearisation error, with a conservative bound of **2.08 m** against the
  5 m budget (42 % of budget). Any claim
  about keep-out-zone clearance at the metre level in this scenario must be checked against
  the nonlinear reference, not asserted from CW alone.
- Far-range rendezvous (10 km and beyond) sits at 100–300 m error, comfortably larger than
  the keep-out sphere itself. CW is the wrong model there. This is why the plan uses
  Lambert targeting for far-range phasing rather than extending CW.

## Guard

`rpo_core.relative.check_cw_validity` warns when a scenario's estimated error exceeds a
caller-supplied tolerance. Call it **once per run at the scenario boundary**, never inside
a propagation or Monte Carlo loop — a warning issued 100 000 times trains the user to
ignore it.

```python
from rpo_core.relative import check_cw_validity

check_cw_validity(separation_m=1000.0, n_rad_s=n, n_orbits=0.5, tolerance_m=2.0)
```
