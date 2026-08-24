# Conventions

Everything in this repository assumes the conventions on this page. They are locked; a
change here is a breaking change to every package.

## Units

SI throughout the core: **metres, seconds, radians, kilograms**. Units are carried in
variable *names* (`r_eci_m`, `v_hill_m_s`, `n_rad_s`, `tof_s`) and docstrings, never as
attached unit objects. Conversion happens only at the I/O boundary.

Rationale: attaching units to every array costs 10-100x in Monte Carlo loops, and the loops
are where this suite spends its time. The naming discipline buys most of the safety at none
of the cost.

## Inertial frame

An Earth-centred **pseudo-inertial** frame, taken as GCRF-approximated. Precession,
nutation, polar motion, and frame-tie corrections are **neglected**.

This is an explicit modelling choice. The induced error is to be quantified in
`docs/frames.md` against a rigorous frame implementation, not assumed negligible.

## Hill / LVLH frame

Origin at the **target** centre of mass, rotating with the target.

| Axis | Name | Definition |
|------|------|------------|
| `x` | radial, **R-bar** | `r_hat` — along the target radius, positive **away from Earth** |
| `z` | cross-track | `h_hat` — along `r × v`, the **positive** orbit normal |
| `y` | along-track, **V-bar** | `z_hat × x_hat` — completes the right-handed set; equals `v_hat` for a circular orbit |

Right-handed: `x_hat × y_hat = z_hat`.

Two points that are easy to get wrong and expensive to get wrong:

- **`z` is the *positive* orbit normal.** Since `x_hat = r_hat` and `y_hat ≈ v_hat`, the
  right-handed completion `x_hat × y_hat` is `r_hat × v_hat`, which is by definition
  `h_hat`. Not the negative normal.
- **`y` is built as `z_hat × x_hat`, not taken from `v_hat`.** For an eccentric orbit `v`
  is not perpendicular to `r`, so using `v_hat` directly would give a non-orthonormal
  triad. The two definitions coincide for a circular orbit.

A chaser trailing the target sits at **negative `y`**.

## State vectors

- Absolute: `[r_eci(3), v_eci(3)]` — metres, metres/second.
- Relative: `[x, y, z, xdot, ydot, zdot]` in the target's Hill frame — metres, metres/second.

Relative velocity is the **rotating-frame** derivative. Converting from inertial requires
the transport theorem, `R @ (dv_eci - omega × dr_eci)`. Dropping the `omega × dr` term
produces an error of roughly 0.1 m/s per km of separation in LEO — the same order as the
manoeuvres being designed, so it is not a small error.

## Numerics

- Reference integrator: `scipy.integrate.solve_ivp`, method `DOP853`, `rtol=1e-12`,
  `atol=1e-12` (absolute tolerance in metres and metres/second).
- Tolerances are **configuration fields**, not module constants. Any quoted numerical
  result must survive a tolerance sweep before it is reported.

## Randomness

Every stochastic entry point takes an explicit `numpy.random.Generator`. No global RNG
state, anywhere. Examples use fixed seeds.

## Failure behaviour

Numerical routines that cannot produce a valid answer **raise a typed exception**; they do
not return a plausible-looking wrong one. See `rpo_core.exceptions`.
