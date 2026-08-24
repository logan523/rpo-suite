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

**Corollary — the radial bulge, and why it constrains mission design.** That departure
impulse is purely radial, and from the CW solution a radial impulse produces a radial
excursion peaking at `|Δv_x|/n`. Substituting:

    peak radial excursion = Δy/4

The transfer arc therefore bulges off the V-bar by a quarter of the hop length, *independent
of orbit altitude and of transfer time*. Measured for the baseline 750 m hop: 187.5 m, exactly
Δy/4, reaching **20.56° off the V-bar against a 10° approach corridor**.

This is a hard geometric limit, not a tuning problem. A single two-impulse half-period hop
cannot respect a tight approach corridor for any Δy where `Δy/4` is a significant fraction of
the standoff distance. Respecting a 10° corridor requires either a sequence of shorter hops,
a different transfer time, or a forced-motion (continuous-thrust) approach — which is why
real proximity operations use staged hops rather than one large transfer, and why Phase B's
multi-burn optimisation exists.

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

    k    = −(3/2) J₂ (μ/r²) (Rₑ/r)²
    a_x  = k (1 − 5(z/r)²) (x/r)
    a_y  = k (1 − 5(z/r)²) (y/r)
    a_z  = k (3 − 5(z/r)²) (z/r)

An earlier revision of this document omitted the `x/r`, `y/r`, `z/r` direction cosines,
which as written made the acceleration a constant vector independent of in-plane position.
Verified as exactly `−∇V_J2` by central differences to 9.8e-10.

**Independent check 1 — secular RAAN drift.**

    Ω̇ = −(3/2) n J₂ (Rₑ/p)² cos i

**Measured: 0.1285 % agreement** over 20 orbits at 700 km, e = 0.01, i = 51.6°. The residual
is physics, not numerics: unchanged to 8 significant figures across rtol 1e-9 → 1e-12, and it
scales with the omitted second-order term (0.1401/0.1285/0.1039/0.0735 % at 400/700/1500/3000
km, a constant ~1.46× the value of `J₂(Rₑ/p)²`).

**Independent check 2 — argument-of-perigee drift.**

    ω̇ = (3/4) n J₂ (Rₑ/p)² (5cos²i − 1)

Note `− 1`, not `− 3`. An earlier revision of this document and of the task brief had `− 3`,
which is wrong: the zero of this rate is the critical inclination, and only `− 1` yields the
published Molniya value of **63.4349°** (`−3` gives 39.23°, which corresponds to nothing).
At i = 45° the two forms differ in sign as well as magnitude.

**Independent check 3 — sun-synchronous inclination.** The inclination at which nodal
regression equals Earth's orbital rate (360°/365.25 d). **Measured: 98.6029° at 800 km**
(0.0138 % on the resulting drift rate) and **97.7875° at 600 km** (0.0131 %). Note these are
altitude-specific; the often-quoted "≈97.8°" is the 600 km value, not 800 km.

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

Burn execution error: magnitude scale `1 + δ`, `δ ~ N(0, σ_mag²)`, applied independently of
direction.

**Pointing error — normative definition.** `θ ~ N(0, σ_point²)` is the rotation angle about an
axis drawn uniformly **in the plane perpendicular to Δv**. The realised half-cone angle is then
exactly `|θ|`, half-normal distributed, so `σ_point` is the standard deviation of an
observable quantity.

An earlier revision specified instead "a rotation vector with per-axis `N(0, σ_point²)`". Both
are well-posed and both appear in the literature, but **they are not interchangeable**:

| Convention | Half-cone angle distribution | Mean for σ = 0.02 rad |
|---|---|---|
| Perpendicular-axis rotation (**normative here**) | half-normal, scale σ | 0.01597 rad = σ·√(2/π) |
| Per-axis rotation vector | Rayleigh, scale σ | 0.02505 rad = σ·√(π/2) |

Measured over 400 000 samples; both reproduce their closed forms to 4 significant figures. The
same σ therefore means something **1.568× larger** under the per-axis convention. A reader who
assumes the wrong one mis-sizes every dispersion study by that factor, so the convention is
stated here rather than left to the implementation.

Note also that a *uniform-on-the-sphere* axis would be wrong for either: rotating by θ about an
axis at angle φ from Δv turns the impulse by only `cos α = cos²φ + sin²φ·cos θ`, so the
component of the axis parallel to Δv does nothing and σ ceases to be the scale of anything
observable.

Navigation error: additive `N(0, P)` on the estimated relative state plus a constant bias per
run.

Linear covariance propagation: `P⁺ = Φ P Φᵀ + Q`.

**Independent check.** Monte Carlo sample covariance converges to the linear-covariance
prediction in the small-dispersion limit; NEES consistency for the estimator.
