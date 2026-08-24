# Simulation limitations and responsible-use statement

**This is educational and research simulation software. It is not flight software.**

- Not developed, verified, or validated to any flight software standard (DO-178C,
  NPR 7150.2, ECSS-E-ST-40C, or equivalent). There is no safety case, no requirements
  traceability to a flight programme, and no independent V&V.
- The dynamics models are deliberately simplified and their fidelity is stated per model.
  Linearised relative motion (Clohessy-Wiltshire) assumes a circular reference orbit and
  small separations; its error grows with separation and elapsed time.
- Constraint values used in the example scenarios (keep-out zone, approach corridor,
  closing-velocity limits) are drawn from **publicly documented, order-of-magnitude
  representative** figures. They are **not** the requirements of any real programme,
  vehicle, or visiting-vehicle agreement, and must not be treated as such.
- No result produced by this repository should be used for operational mission planning,
  collision avoidance, conjunction assessment, or any decision affecting real hardware or
  real people.
- Machine-learned policies in this repository are research artefacts. Good performance in
  simulation demonstrates nothing about flight readiness. Sim-to-real transfer for
  spacecraft GN&C is an open problem and is not addressed here.

## Export control and data provenance

All models, constants, and methods here derive from openly published textbooks, journal
literature, and public agency documentation. The repository contains:

- no controlled technical data,
- no real spacecraft design parameters, performance data, or telemetry,
- no vendor-proprietary or ITAR/EAR-controlled models.

Example scenarios use publicly available orbital parameters only.
