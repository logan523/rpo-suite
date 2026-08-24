# Project 1 — Mathematical model specification

Every model in the suite, with its governing equations, its assumptions, and the
independent check that proves the implementation. Models are listed in increasing fidelity;
each states what it neglects relative to the one below it.

Notation: `r` inertial position (m), `v` inertial velocity (m/s), `ρ = [x,y,z]` relative
position in the target Hill frame (m), `μ` gravitational parameter (m³/s²), `n` mean motion
(rad/s), `T = 2π/n` orbital period (s), `Rₑ` Earth equatorial radius (m).

---

## M1 — Restricted two-body motion  *(implemented)*

    r̈ = −μ r / |r|³

**Assumptions.** Point-mass central body, massless spacecraft, no third bodies, no
non-spherical gravity, no drag, no radiation pressure, Newtonian.

**Conserved.** Specific energy `ε = v²/2 − μ/r`; specific angular momentum `h = r × v`.

**Integration.** DOP853, `rtol = atol = 1e-12` (config, not constant).

**Independent checks.** Energy and angular momentum drift < 1e-10 relative over 10 orbits;
circular orbit period matches `2π√(a³/μ)`; apoapsis radius matches `a(1+e)` and apoapsis
speed matches vis-viva; motion stays planar; time-reversal round trip recovers the initial
state; deviation from a reference run shrinks monotonically as tolerance tightens.

---

## M2 — Hill / LVLH frame  *(implemented)*

    x̂ = r̂                  (radial, outward, R-bar)
    ẑ = ĥ = (r × v)/|r × v| (positive orbit normal)
    ŷ = ẑ × x̂              (along-track, V-bar; equals v̂ for a circular orbit)
    ω = h / |r|²

Relative state by the transport theorem:

    ρ    = R (r_c − r_t)
    ρ̇    = R (v_c − v_t − ω × (r_c − r_t))

**Why ŷ = ẑ × x̂ rather than v̂.** For an eccentric orbit `v` is not perpendicular to `r`, so
taking `ŷ = v̂` yields a non-orthonormal triad. The two coincide for a circular orbit.

**Failure mode this guards.** Dropping the `ω × dr` term costs ≈ 0.1 m/s per km of
separation in LEO — the same order as the manoeuvres being designed.

**Independent checks.** `R Rᵀ = I` and `det R = +1`; `ẑ` parallel to `h`; a chaser trailing
the target lands at negative `y`; relative speed for a co-moving chaser equals `n·ρ`;
degenerate geometry (`|r × v| → 0`) raises.

---

## M3 — Clohessy-Wiltshire linear relative motion  *(implemented)*

    ẍ − 3n²x − 2nẏ = 0
    ÿ + 2nẋ        = 0
    z̈ + n²z        = 0

**Assumptions relative to M1.** Linearised in `ρ/r`; circular reference orbit; both vehicles
under the same point-mass field.

**Closed-form STM.** With `τ = n·Δt`, `s = sin τ`, `c = cos τ`:

    Φ_rr = [[4−3c, 0, 0], [6(s−τ), 1, 0], [0, 0, c]]
    Φ_rv = [[s/n, 2(1−c)/n, 0], [−2(1−c)/n, (4s−3τ)/n, 0], [0, 0, s/n]]
    Φ_vr = [[3ns, 0, 0], [−6n(1−c), 0, 0], [0, 0, −ns]]
    Φ_vv = [[c, 2s, 0], [−2s, 4c−3, 0], [0, 0, c]]

**Analytic properties used as tests.**
- Drift-free condition `ẏ₀ = −2n·x₀` gives a closed relative orbit repeating each period.
- Along-track secular drift `Δy = −3t(2n·x₀ + ẏ₀)`, which over one period is `−12π·x₀`.
- A radial impulse from rest at the origin traces a closed 2:1 ellipse:
  `x = (Δv/n) sin nt`, `y = −(2Δv/n)(1 − cos nt)`.
- Cross-track is simple harmonic at frequency `n`, fully decoupled from in-plane motion.

---

## M4 — Two-impulse CW targeting  *(implemented)*

    v₀⁺ = Φ_rv⁻¹ (ρ_f − Φ_rr ρ₀)
    Δv₁ = v₀⁺ − v₀
    Δv₂ = ρ̇_f − (Φ_vr ρ₀ + Φ_vv v₀⁺)

**Singularities — different for the two subproblems, which is why they are solved separately.**

    in-plane   det Φ_rv(2×2) = (8 − 8c − 3τs)/n²  → 0 at τ = 2πk   (whole periods)
    cross-track           Φ_rv[2,2] = s/n          → 0 at τ = πk    (half periods)

A single 3×3 conditioning test would reject the half-period V-bar hop, which is a perfectly
well-posed planar transfer and the suite's baseline manoeuvre.

**Measured conditioning behaviour.** The in-plane condition number grows only as
`≈ 3/(1 − t/T)`; at 0.99999 periods it is 3.0e5 and the solved impulse is a sane 0.045 m/s.
The conditioning guard is therefore an exact-singularity backstop, not an accuracy envelope,
and is named accordingly.

**Closed-form case used as a test.** A coplanar V-bar hop of length `Δy` over exactly half a
period, starting and ending at rest, collapses to two equal purely-radial impulses:

    |Δv₁| = |Δv₂| = n·Δy/4       total Δv = n·Δy/2

---

## M5 — Nonlinear relative motion  *(implemented)*

No new equations: propagate target and chaser independently under M1, then difference them
through M2 at every epoch, recomputing the frame from the propagated target state.

**Role.** The external oracle for M3. It is the model M3 is a linearisation *of*.

**Measured result.** With an exactly circular target (isolating linearisation error from
eccentricity error), the one-orbit position error is

    err = 6π · ρ² / r

to six significant figures across 400/800/1500 km altitudes and 1 km/10 km separations.
Empirical; an analytic derivation is an open item. Consequences: with a 2 m budget (1 % of a
200 m keep-out sphere) CW holds to ≈850 m separation over one orbit; the MVP baseline hop
incurs ≈1.5 m; far-range work at 10 km incurs 277 m and requires Lambert instead.

---

## M6 — J2 secular perturbation  *(open)*

    a_J2 = −(3/2) J₂ (μ/r²) (Rₑ/r)² [ (1 − 5(z/r)²) x̂ᵢ + (1 − 5(z/r)²) ŷᵢ + (3 − 5(z/r)²) ẑᵢ ]

**Independent check.** Secular RAAN drift must match

    Ω̇ = −(3/2) n J₂ (Rₑ/p)² cos i

to within 1 %. This is the single most valuable physics-validation test in the project.

---

## M7 — Lambert's problem  *(in progress)*

Given `r₁`, `r₂`, and time of flight, find the connecting conic. Universal-variable or Izzo
formulation.

**Independent check.** Propagate the returned `v₁` under M1 for the requested TOF and confirm
arrival at `r₂`. M1 is already validated, so this is a genuine oracle rather than a
self-consistency check.

---

## M8 — Finite burns  *(open)*

    r̈ = −μ r/|r|³ + (F/m) û ,    ṁ = −F/(g₀ I_sp)

**Independent check.** As `F → ∞` at fixed total impulse, the finite-burn trajectory must
converge to the impulsive result. Convergence rate is itself the test.

---

## M9 — Uncertainty  *(open)*

Burn execution error: magnitude `1 + δ`, `δ ~ N(0, σ_mag²)`; pointing error as a small
rotation with per-axis `N(0, σ_point²)`. Navigation error: additive `N(0, P)` on the
estimated relative state plus a constant bias per run.

Linear covariance propagation: `P⁺ = Φ P Φᵀ + Q`.

**Independent check.** Monte Carlo sample covariance converges to the linear-covariance
prediction in the small-dispersion limit; NEES consistency for the estimator.
