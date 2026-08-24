# optional_stk — external-tool validation, and what it is not

## Nothing in this directory has been executed

Read this first, because everything below depends on it.

The files in `optional_stk/` were written on **macOS (arm64)**, a machine with:

- no GMAT installation (no `/Applications/GMAT*`, no `gmat` on `PATH`);
- no STK installation, and no STK licence;
- none of `agi`, `pystk`, `comtypes`, or `win32com` importable in the project environment.

All three were checked directly rather than assumed. Consequently:

- **`gmat/vbar_baseline.script` has never been run.** It is a plain-text GMAT mission script
  written from the documented scripting language. It is *unverified*. Expect to fix syntax or
  resource names on first run against your GMAT version; the runbook says where to look.
- **No STK, Astrogator or PySTK code appears anywhere in this repository.** Writing it without
  the tool present would mean inventing an API, which `docs/CONTRIBUTING.md` forbids and which
  would be indistinguishable, to a reader, from code that had been tested.
- **No number in the top-level `README.md`, in `docs/`, or in any metrics artefact depends on
  anything in this directory.** Requirements F-6.1 through F-6.5 in `docs/project1/srs.md`
  §2.6 remain **[OPEN]**. If someone with the tools runs the runbook, they close F-6.4 and
  F-6.5; nothing here closes them in advance.

What *has* been built and tested is the other half: `rpo_core.validation`, the ingest,
frame/time alignment and comparison harness, exercised in
`packages/rpo-core/tests/test_validation.py` against an independent analytic Kepler solution
(Lagrange f and g functions, no numerical integration, no shared code path with
`rpo_core.propagate`). That is a genuine cross-check of the propagator. It is **not**
external-tool validation, and every report the harness produces records which of the two it
is, in a required `provenance` field. A comparison record that cannot tell you whether the
"external" side came from a real tool run or from a synthetic reference is a liability, so
the distinction is structural rather than a convention.

## What you need in order to actually run this

**GMAT path (licence-free, recommended).**

- GMAT R2020a or later. Official builds exist for Windows and Linux; on Apple silicon expect
  to build from source or run a Linux VM/container.
- No Python, no COM, no network. The script writes a text report; the harness reads it.

**STK path (not implemented here).**

- Windows, an STK installation with a valid licence, and Astrogator for the manoeuvre
  sequence. STK's Python interface additionally needs the vendor-shipped bindings.
- Deliberately left unwritten. A separate task covers it, with the tool present.

## The data contract

`rpo_core.validation.read_ephemeris` reads exactly one format, `rpo-ephemeris/1.0`. It does
not sniff, guess, or best-effort-parse anything else — a foreign header is an error. Converting
a tool's native export into this format is a documented step, not a hidden one, because that
conversion is where units and epochs get silently mangled.

```
# format: rpo-ephemeris/1.0
# frame: EME2000
# time_scale: TAI
# epoch: 2026-03-01T00:00:00.000000
# position_unit: km
# velocity_unit: km/s
# provenance: tool_run
# source: GMAT R2022a ReportFile, vbar_baseline.script
t,x,y,z,vx,vy,vz
0.0,6798.137,0.0,0.0,0.0,4.756292,6.000952
60.0,...
```

Rules, in full:

| Element | Requirement |
|---|---|
| Comment lines | Start with `#`, carry `key: value`. Anything else is an error. |
| `format` | Must be exactly `rpo-ephemeris/1.0`. |
| `frame` | One of `GCRF_APPROX`, `GCRF`, `ICRF`, `EME2000`, `TOD`, `MOD`, `TEME`, `ITRF`. |
| `time_scale` | One of `TAI`, `TT`, `TDB`, `UTC`, `UT1`. |
| `epoch` | `YYYY-MM-DDThh:mm:ss[.ffffff]`, a reading on `time_scale`. Required — there is no default. |
| `position_unit` | `m` or `km`. Converted to metres at ingest and nowhere else. |
| `velocity_unit` | `m/s` or `km/s`. |
| `provenance` | `tool_run`, `synthetic_reference`, or `unknown`. Optional; defaults to `unknown` and is never silently upgraded. |
| `source` | Optional free text naming the producer. Put the tool name and version here. |
| Column header | First non-comment line. Must contain `t,x,y,z,vx,vy,vz`. Matched **by name**, so extra columns and any ordering are fine. |
| Rows | Comma- or whitespace-separated. Field count must match the header exactly. |
| `t` | Seconds from `epoch`. Strictly increasing. |

Every one of the following is a refusal, not a warning, because the failure this harness exists
to prevent is a plausible-looking comparison built on a misread file:

- a missing header key, a missing column, or a row whose field count does not match;
- any non-finite value;
- times that are not strictly increasing (a duplicate breaks every interpolation window; a
  reversal usually means two exports were concatenated);
- a step more than 1.5× the median step — a dropout. Interpolating across a gap reports the
  accuracy of the nominal spacing over an arc that does not have it;
- a unit token that is not recognised, **or one contradicted by the data**: if the geocentric
  radius implied by the declared unit falls outside 1×10⁶ – 1×10⁹ m, the declaration is
  rejected. This is the metre/kilometre confusion, and it is worth a factor of 1000;
- velocity columns that disagree with a central difference of the position columns by more
  than 1 % — a gross-blunder detector for unit factors, swapped columns and sign flips. (A
  clean 60 s LEO ephemeris measures ~7.6×10⁻⁴ relative against its own central difference, so
  the 1 % threshold has a factor of ~13 of headroom.)

## What the harness will and will not do for you

- **It refuses to compare across frames.** Identical frames pass. Two *inertial-family* frames
  (`GCRF_APPROX`, `GCRF`, `ICRF`, `EME2000`) pass only when you pass
  `allow_approximate_frame_tie=True`, which applies the identity rotation and records the
  resulting error budget in the report. Any date-dependent frame (`ITRF`, `TEME`, `MOD`, `TOD`)
  is always refused: rotating out of one requires exactly the Earth-orientation model
  `docs/conventions.md` declines to implement.
- **It refuses to guess a time-scale offset.** `TAI ↔ TT` is applied automatically (32.184 s,
  exact by definition). UTC, UT1 and TDB require an offset you supply, and are therefore on
  your record rather than the library's. Getting the 37 s TAI−UTC difference wrong is
  2.8×10⁵ m of along-track difference in LEO.
- **It refuses to extrapolate.** Comparison happens only on the overlapping arc.
- **It measures its own interpolation error** by decimation plus Richardson extrapolation, and
  marks the report not-clean when that error is within two decades of the difference being
  reported.
- **It carries a frame-approximation budget.** This repository's inertial frame neglects
  precession, nutation, polar motion and the frame tie. Over one day in LEO that is worth
  **9.5 m** (4.54 m precession + 4.94 m nutation), and 66 m over a week — it grows at about
  9.5 m/day until the nutation term saturates at 115 days, 4.54 m/day after. Any report whose
  maximum position
  difference is smaller than that budget is flagged `difference_within_frame_budget`, meaning
  the frame approximation alone could account for it and no dynamics conclusion may be drawn.

See `gmat/runbook.md` for the step-by-step, and `gmat/vbar_baseline.script` for the (unverified)
mission script.
