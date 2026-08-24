# GMAT validation runbook

Produce a licence-free external reference ephemeris and compare it against
`rpo_core.propagate.propagate_two_body`. Closes `docs/project1/srs.md` §2.6 F-6.4 (with a
stated cause for the difference) and F-6.5 (a licence-free validation path).

**This runbook has not been executed.** GMAT is not installed on the machine where it was
written; `vbar_baseline.script` is unverified. Everything below is what *should* happen, with
the failure modes called out. If a step does not behave as described, the script is the thing
to doubt first, not your GMAT.

---

## 0. Prerequisites

- GMAT R2020a or later. Official builds: Windows and Linux. On Apple silicon, build from
  source or use a Linux VM/container.
- This repository, with `uv sync` completed.
- No network access is needed at any point.

---

## 1. Generate the reference

```bash
# GUI
#   File > Open  ->  optional_stk/gmat/vbar_baseline.script  ->  Run

# Console (Linux/Windows)
GMAT --run --minimize optional_stk/gmat/vbar_baseline.script
```

Output: `vbar_baseline_gmat.txt` in GMAT's output directory (usually `GMAT/output/`).

**Before trusting a single row, check the run log for these three things.** Each is a silent
corruption if it goes the wrong way, and each produces a comparison that looks fine:

1. **Gravity degree and order actually used.** The script asks for 0/0. If your build falls
   back to a non-zero default, you have exported a J2 trajectory labelled two-body, and the
   comparison will read as hundreds of kilometres of "integrator error".
2. **`Earth.Mu`.** The script sets 398600.4418 to match this repository. If the assignment was
   rejected, GMAT used 398600.4415 and you owe the result an explanation of about 0.25 m/day of
   along-track drift.
3. **Row count.** 2881 rows plus the header (24 h at 30 s). Fewer means the loop terminated
   early; more means the report is firing at integration steps rather than at the loop's
   stopping condition, and the spacing will not be uniform — `read_ephemeris` will reject that
   as a gap, which is the intended behaviour but not the intended outcome.

---

## 2. Convert to the `rpo-ephemeris/1.0` contract

GMAT's `ReportFile` is fixed-width text with a one-line column header, kilometres and
kilometres per second, and a time column in **GMAT's** Modified Julian Date.

> **The MJD trap.** GMAT's `ModJulian` is `JD − 2 430 000.0` (reference epoch
> 05 Jan 1941 12:00:00), **not** the standard MJD `JD − 2 400 000.5`. The two differ by
> **29 999.5 days**. Convert with `MJD_standard = MJD_gmat + 29999.5`. Getting this wrong moves
> the epoch by 82 years, which the harness will catch as a non-overlapping arc — but only
> because the harness refuses to compare non-overlapping arcs. Do not rely on that.

The conversion below is deliberately explicit about units and epoch; those two lines are where
this whole exercise usually goes wrong.

```python
# convert_gmat_report.py  (run from the repository root)
import numpy as np
from rpo_core.validation import (
    Epoch,
    Provenance,
    ReferenceFrame,
    TimeScale,
    ephemeris_from_states,
    write_ephemeris,
)

RAW = "output/vbar_baseline_gmat.txt"  # GMAT ReportFile
OUT = "results/gmat/vbar_baseline_gmat.eph"  # what the harness reads

raw = np.loadtxt(RAW, skiprows=1)  # skip the GMAT column header
mjd_gmat, xyz_km, vxyz_km_s = raw[:, 0], raw[:, 1:4], raw[:, 4:7]

# GMAT ModJulian -> standard MJD. See the trap above.
mjd = mjd_gmat + 29999.5
epoch = Epoch.from_mjd(float(mjd[0]), TimeScale.TAI)  # TAIModJulian was reported
times_s = (mjd - mjd[0]) * 86400.0

# km -> m at the I/O boundary, and only here (docs/conventions.md).
states = np.hstack((xyz_km * 1.0e3, vxyz_km_s * 1.0e3))

ephemeris = ephemeris_from_states(
    times_s,
    states,
    epoch=epoch,
    frame=ReferenceFrame.EME2000,  # GMAT EarthMJ2000Eq
    provenance=Provenance.TOOL_RUN,  # a real tool actually ran
    source="GMAT R2022a ReportFile, vbar_baseline.script",  # put YOUR version here
)
print(write_ephemeris(OUT, ephemeris))
```

`provenance=Provenance.TOOL_RUN` is the only thing that lets a report claim external-tool
validation. Set it **only** if GMAT actually ran. If you are testing the pipeline with a file
you fabricated, use `SYNTHETIC_REFERENCE`.

`frame=ReferenceFrame.EME2000` is honest, not pedantic: GMAT's `EarthMJ2000Eq` is the mean
equator and equinox of J2000, which is not this repository's `GCRF_APPROX`. Declaring it
correctly is what makes the harness require an explicit decision in step 4.

Where the file goes: `results/gmat/` (git-ignored alongside the other run artefacts). Nothing
reads it automatically; you point step 4 at it.

---

## 3. Generate the matching internal ephemeris

Same initial state, same times, this repository's propagator:

```python
import numpy as np
from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from rpo_core.propagate import propagate_two_body
from rpo_core.validation import (
    Epoch,
    Provenance,
    ReferenceFrame,
    TimeScale,
    ephemeris_from_states,
)

a = R_EARTH_EQUATORIAL_M + 420.0e3
v = np.sqrt(MU_EARTH_M3_S2 / a)
inc = np.radians(51.6)
state0 = np.array([a, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])

times_s = np.arange(0.0, 86400.0 + 1.0, 60.0)
internal = ephemeris_from_states(
    times_s,
    propagate_two_body(state0, times_s),
    epoch=Epoch.from_iso("2026-03-01T00:00:00.000000", TimeScale.TAI),
    frame=ReferenceFrame.GCRF_APPROX,
    provenance=Provenance.SYNTHETIC_REFERENCE,
    source="rpo_core.propagate.propagate_two_body",
)
```

The epoch string must match `Target.Epoch` in the GMAT script, on the same time scale. If you
changed one, change the other.

---

## 4. Compare

```python
from rpo_core.validation import compare_ephemerides, read_ephemeris, write_comparison_report

external = read_ephemeris("results/gmat/vbar_baseline_gmat.eph")

report = compare_ephemerides(
    internal,
    external,
    label="vbar-baseline-24h-gmat",
    allow_approximate_frame_tie=True,  # GCRF_APPROX vs EME2000; see below
)
print(write_comparison_report("results/gmat/comparison.json", report))
print(f"max |dr| = {report.position_max_m:.3f} m at t = {report.position_max_time_s:.0f} s")
print(f"  radial       {report.radial_m.max_abs:8.3f} m  (mean {report.radial_m.mean:+.3f})")
print(
    f"  along-track  {report.along_track_m.max_abs:8.3f} m  (mean {report.along_track_m.mean:+.3f})"
)
print(
    f"  cross-track  {report.cross_track_m.max_abs:8.3f} m  (mean {report.cross_track_m.mean:+.3f})"
)
print(
    f"frame budget {report.frame_tie_error_m:.1f} m; "
    f"interp error {report.interpolation_error_m:.2e} m "
    f"(margin {report.interpolation_margin:.0f}x)"
)
```

`allow_approximate_frame_tie=True` is required and is a real decision, not a formality. The two
frames are genuinely different; you are asserting that the identity rotation between them is
acceptable, and the report records the resulting budget so a reader can see what you accepted.
Without the flag the comparison refuses to run.

---

## 5. What agreement to expect

Assuming the force models match, the epochs match, and the frame tie is the only approximation:

| Arc | Expected max position difference | Dominated by |
|---|---|---|
| 1 orbit (~5578 s) | ~0.5 – 1 m | frame approximation |
| 24 h | **~5 – 15 m** | frame approximation (~9.5 m budget) |
| 7 days | ~60 – 80 m (budget 66 m), growing ~9.5 m/day | precession + nutation |

The integrator contribution is five decades below all of these. This repository's DOP853 at
`rtol = atol = 1e-12` agrees with an independent analytic Kepler solution (Lagrange f and g,
no numerical integration) to a **measured 8.4×10⁻⁵ m maximum over 24 h**, 6.5×10⁻⁵ m RMS,
9.3×10⁻⁸ m/s in velocity — see `test_validation.py`, which prints the value on every run.
GMAT's PrinceDormand78 at `Accuracy = 1e-13` should be comparable. **The number you get is a frame
measurement, not an integrator measurement**, and the report says so: when
`difference_within_frame_budget` is `True`, the difference is smaller than what the neglected
precession and nutation could produce on their own, and no dynamics conclusion may be drawn
from it.

Expect the difference to sit mostly in **along-track and cross-track**, not radial. A rotation
about an axis near the celestial pole displaces a 51.6°-inclined orbit chiefly in-plane and
out-of-plane, barely at all radially. A large *radial* difference means something else is
wrong.

---

## 6. Diagnosing a disagreement — in order of likelihood

### 1. Frame mismatch (most likely, and the largest)

**Signature:** hundreds of metres to tens of kilometres. Cross-track is comparable to or larger
than along-track. The difference is roughly periodic at the orbit period rather than growing.

**Causes:** reporting in `EarthFixed` instead of `EarthMJ2000Eq` (that is Earth rotation:
hundreds of kilometres within minutes — the harness refuses `ITRF` outright, so this shows up as
a `FrameMismatchError` if you declared it honestly and as nonsense if you did not); reporting in
`EarthICRF` or a true-of-date frame; or declaring `GCRF_APPROX` on the GMAT file to dodge the
`allow_approximate_frame_tie` prompt. That last one is self-inflicted: the flag exists so the
approximation appears in the report instead of being hidden in a header.

**Check:** re-read the `EphemReport.Add` lines. Every element must carry the same
`EarthMJ2000Eq` prefix.

### 2. Epoch offset (next most likely, and easy to quantify)

**Signature:** almost pure along-track, with a constant signed mean — `report.along_track_m.mean`
close to `report.along_track_m.max_abs`. Radial and cross-track stay small.

**Arithmetic:** along-track difference ≈ |Δt| × 7657 m/s in LEO. So:

| Observed along-track | Implied epoch error | Usual cause |
|---|---|---|
| ~7.7 km | 1 s | rounding in the epoch string |
| ~2.8×10⁵ m | 37 s | TAI vs UTC |
| ~246 m | 32.184 ms | — |
| ~2.5×10⁹ m | 29 999.5 days | the GMAT MJD trap (§2) |

**Check:** `report.epoch_offset_s` and `report.time_scale_offset_s` in the JSON say exactly what
alignment applied. If `epoch_offset_s` is not what you expect, the two files disagree about
their own epochs and the states were never at fault.

### 3. Force-model difference (least likely once the first two are excluded)

**Signature:** along-track dominant and **growing secularly** — max at the end of the arc, mean
roughly half the max. Cross-track grows too if the difference is J2.

**Magnitudes, so you can identify which:**

| Difference | Effect over 24 h in LEO |
|---|---|
| J2 left enabled on one side | hundreds of km along-track — unmistakable |
| Earth `mu` 398600.4415 vs .4418 (7.5×10⁻¹⁰) | ~0.25 m along-track, linear in time |
| Drag or SRP left enabled | metres to tens of metres, altitude-dependent |
| Third bodies (Sun/Moon) left enabled | tens to hundreds of metres |
| Integrator accuracy 1e-13 vs 1e-12 | sub-millimetre; not your problem |

**Check:** GMAT's run log lists the force model it actually built. Compare it line by line
against the script rather than against your memory of the script.

### And before any of the above

If `report.interpolation_is_negligible` is `False`, stop: the difference you are looking at may
be an artefact of putting the two ephemerides on a common grid, not a difference between the
tools. The script already asks for 30 s, which measures 1.5×10⁻⁷ m of interpolation error;
halving it again to 15 s buys another factor of 2⁸ = 256, since 8-point Lagrange is O(h⁸).
Also check that the export is uniformly spaced — `read_ephemeris` will have raised
`EphemerisGapError` if it is not, which is itself the answer.

---

## 7. Recording the result

`results/gmat/comparison.json` is the artefact. It carries the provenance flag, the frame tie
and its budget, the epoch offsets applied, the measured interpolation error, and the full
radial/along-track/cross-track breakdown with signed means.

When quoting a number from it, quote the **cause** with it, as F-6.4 requires — "12 m over 24 h,
dominated by the neglected precession and nutation (budget 9.5 m)" is a result; "12 m" alone is
not. And quote `is_external_tool_validated` honestly: it is `True` only when the external file
declared `provenance: tool_run`.
