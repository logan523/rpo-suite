# Project 2 — Safe Reinforcement Learning for Autonomous Rendezvous

**Plan for review.** Nothing here is implemented yet.

## 1. Why this problem, and what it will not prove

Terminal rendezvous under uncertainty is a genuine sequential decision problem: the chaser
must trade propellant, time and safety margin under noisy state knowledge, and the optimal
policy is not obviously expressible in closed form. That is what makes it worth an RL
treatment rather than a lookup table.

**What a good result here does NOT prove.** Nothing about flight readiness. Sim-to-real
transfer for spacecraft GN&C is unsolved; a policy trained on this environment has seen one
dynamics model, one noise model, and one reward. The deliverable is a *quantified comparison*
between classical and learned control under identical randomised conditions, plus a safety
architecture that holds regardless of what the policy proposes. That framing goes in the
README, not in a footnote.

## 2. Formulation: impulsive, not low-thrust

**Recommendation: three-axis impulsive Δv at discrete decision points.**

- It matches the existing `two_impulse_transfer` and `correct_two_impulse_transfer` baselines
  exactly, so LQR, MPC and RL share one action space and the comparison is apples to apples.
- Episodes are ~10–40 decisions rather than thousands of integration steps, which is what
  makes training feasible on a CPU-only Apple Silicon laptop.
- It matches how LEO RPO vehicles actually manoeuvre.

Low-thrust becomes an extension once the impulsive case is measured, not the starting point.

## 3. Environment specification

`rpo_rl.envs.RendezvousEnv`, Gymnasium-compatible, passing `gymnasium.utils.env_checker`.

**Phase 1 — 2D Hill frame (in-plane only).** Reuses `rpo_core.relative.cw.cw_stm` for
propagation between decision points. Deterministic, fast, fully analytic.

**Phase 2 — 3D nonlinear, for EVALUATION not training.** Measured: the nonlinear propagator
costs **1937 µs per step against CW's 4.2 µs — 467× slower**, which is 33 min per million-step
run and roughly 20 h across the approved ablation matrix.

So: **train every agent on CW, then evaluate the trained policies on nonlinear dynamics as a
held-out fidelity test.** Evaluation is thousands of steps, not millions, so it costs almost
nothing. This is also the more interesting result — it measures directly how much a policy
trained on a linearisation degrades on the true dynamics, which is a sim-to-real question in
miniature and connects to the `6π·ρ²/r` error law already measured in Project 1.

**Observation** (normalised, and every element declared as truth or estimate):

| element | dim | source | units |
|---|---|---|---|
| relative position | 2 or 3 | **estimate** | m |
| relative velocity | 2 or 3 | **estimate** | m/s |
| fuel remaining | 1 | truth | fraction |
| time remaining | 1 | truth | fraction |
| navigation 1σ | 2 or 3 | truth | m |

Estimates come from `rpo_core.navigation.NavigationErrorModel` — a per-run constant bias plus
white noise, the distinction already measured to matter by a factor of 2.16.

**Action.** `Box(-1, 1, shape=(2 or 3,))`, scaled to a per-decision Δv cap.

**Episode.** Fixed decision cadence over a bounded horizon. Terminates on: terminal tolerance
met (success), keep-out sphere breached (collision), fuel exhausted, horizon reached, or an
**abort**.

**Abort is emergent, not an action.** The action space stays a pure `Box` of Δv — no discrete
abort channel, so Stable-Baselines3's continuous algorithms apply unmodified. An abort is
*recognised*, not commanded: the episode ends in `aborted` when range has increased monotonically
past a retreat threshold for N consecutive decisions, or when the safety filter's backup
controller has held command for N consecutive decisions. Both are observable consequences of
behaviour rather than a separate decision the agent makes.

Consequence to state in the results: the agent cannot "choose" to abort in one step, so abort
rate measures how often the situation became unrecoverable, not how often the policy judged it
so. That is a weaker claim than a commanded-abort design would support, and it is the price of
keeping a clean continuous action space.

**Randomisation.** Initial relative state, navigation covariance, burn execution error,
observation delay, and target hold point. Seeded via `rpo_core.montecarlo`'s substream scheme
so run *i* is reproducible independently of campaign size.

## 4. Reward

Sparse-plus-shaped, with every term justified:

```
r = -w_dv * |dv|                     propellant
    -w_t                             time (per decision)
    +w_term * terminal_bonus         on success, scaled by 1/(1+miss)
    -w_col * collision                terminal, large
    -w_kv * excess_closing_velocity   shaped
    +w_abort * safe_abort             a successful abort is NOT a failure
```

**The rule that governs this section:** the reward must not be the safety mechanism. Penalties
express preference; the safety filter enforces constraints. Any tuning that makes the agent
"learn to be safe" is a red flag, not a result.

## 5. Baselines — built before any RL

1. **LQR** on the CW system via `scipy.linalg.solve_discrete_are`, with the discrete-time
   plant from `cw_stm`. Cheap and exact for the linear case.
2. **Two-impulse CW targeting** — already implemented; the natural naive baseline.
3. **MPC** — finite-horizon convex QP over the CW dynamics with the half-space keep-out
   linearisation and closing-velocity constraints, via `scipy.optimize` or `cvxpy`.

**MPC and the safety filter are one component, not two.** Build MPC first; the filter is the
same QP with the policy's proposed action as the reference to stay close to. Two consequences
for the results section, both of which must be stated rather than buried: the filter inherits
MPC's feasibility behaviour, and "MPC vs PPO+filter" is partly a comparison of MPC against
itself.

## 6. Safety architecture

The policy proposes; the filter disposes. Three layers:

1. **Projection onto a rotating half-space.** `|ρ| ≥ R` is the complement of a ball and is
   therefore **non-convex** — it cannot be expressed in a QP directly, and an earlier draft of
   this plan said "a small QP" without noticing. At each decision point, linearise: replace the
   sphere with the half-space tangent to it at the *predicted closest approach point*. That is
   convex, so the projection stays a genuine QP, and it is **conservative** — it never permits
   something the true constraint forbids, which is the correct direction of error for a safety
   filter. Cost: it will occasionally refuse a manoeuvre that was actually safe near the
   boundary, so the reported intervention rate is an upper bound. Say so in the results.
2. **Backup controller.** If no admissible action exists, hand over to an LQR retreat to the
   last safe hold point.
3. **Intervention logging.** Every filter action is counted and reported. Intervention rate is
   a first-class metric, not a diagnostic.

Constraint set: keep-out sphere, approach corridor, closing-velocity limit, fuel remaining.

**Measured problem, and the fix.** `evaluate_constraints` costs **57.9 µs** against a **4.2 µs**
CW propagation step — safety checking would be 93 % of environment step time. It is built for a
sampled trajectory with sub-sample refinement, not a per-step predicate.

The fix is a fast scalar predicate added to **`rpo_core.constraints`**, called by BOTH the RL
environment and the Project 1 planner, plus a test asserting the fast predicate and the full
evaluator agree on the same trajectory. Writing the fast path inside `rpo-rl` was rejected: it
would create a second definition of "safe", and the moment the two disagree the comparison
between RL and the planner stops meaning anything.

## 7. Evaluation protocol

### The regime that makes this comparison worth running

**On an unconstrained linear system with a quadratic cost, LQR is optimal.** RL cannot beat it;
it can only approach it. Running the headline comparison there would produce "PPO approaches but
does not beat LQR", which reads as failure and teaches nothing.

So the evaluation is split deliberately:

**Primary regime — constrained, where LQR is not optimal.** Hard keep-out sphere, hard fuel
budget, observation delay, and a non-quadratic terminal cost. LQR has no principled way to
respect a hard state constraint; MPC does but pays for it in computation and feasibility
failures. This is where a learned policy can genuinely win, and it is what the results section
leads with.

**Secondary regime — unconstrained linear, used as a VERIFICATION not a headline.** Here LQR is
provably optimal, so PPO approaching it within a small margin is evidence that the RL
implementation is correct. A large gap means the agent, the reward, or the training budget is
broken. Reported as a sanity check on the method, not as a performance claim.

All methods evaluated on an **identical frozen set of randomised initial conditions**,
committed **as data, not as generating code**, with a hash asserted by a test. Regenerating the
set from code means any change to the sampling logic silently changes the test set and
invalidates every historical comparison. Methods: LQR · CW two-impulse · MPC · PPO · SAC ·
PPO+filter · SAC+filter.

**Metrics.** Success rate (Wilson interval) · collision rate · keep-out violation rate ·
terminal position and velocity error percentiles · total Δv · fuel remaining · time to
rendezvous · **out-of-distribution performance** (initial conditions and noise beyond training)
· safety-filter intervention count.

**Ablations.** No domain randomisation · no safety filter · no fuel in observation · no
observation delay · reward variants · linear vs nonlinear dynamics · training distribution
width.

Every RL number reported across **≥10 seeds with spread**, never a single run. Ten rather than
five: policy-gradient variance across seeds is wide enough that five leaves most between-method
differences statistically silent, and a Phase 1 training run costs about a minute, so the extra
seeds are nearly free.

## 8. Compute budget

Apple Silicon, CPU by default (`device` is a config field; MPS opt-in with a CPU-equivalence
check before any MPS number is quoted). Target: one PPO run ≤ 30 min on 8 CPU cores with
vectorised envs. If a configuration cannot train in that budget, the environment is too
expensive and gets simplified rather than the budget being raised.

## 9. Milestones

| # | Deliverable | Acceptance |
|---|---|---|
| M0 | Backlog X7: consolidate 30 duplicated validators into `rpo_core._validate` | full 1,664-test suite green |
| M1 | 2D env passing `env_checker`, seeded and reproducible | same seed → identical trajectory, **and the reward-hacking gate below passes** |
| M2 | LQR + CW baselines on the frozen test set | metrics table produced |
| M3 | MPC baseline | respects constraints explicitly; feasibility failures raise |
| M4 | Safety filter + backup, with intervention logging | zero collisions under a deliberately adversarial policy |
| M5 | PPO trained, ≥5 seeds | beats or matches LQR on success rate, or the failure is reported |
| M6 | SAC + curriculum | comparison table across all methods |
| M7 | Ablations + OOD evaluation | each ablation changes a metric measurably, or is reported as null |
| M8 | 3D nonlinear extension | same interface, ablation flag |
| M9 | README, write-up, portfolio material | reproducible from a clean clone |

## 9a. The reward-hacking gate (M1 acceptance, before any training)

Assert that hand-written degenerate policies score **worse** than one that completes the
rendezvous:

| degenerate policy | must score below a competent policy |
|---|---|
| never burn | ✓ |
| burn maximally every step | ✓ |
| drift past the target | ✓ |
| sit still just outside the keep-out sphere | ✓ |

A reward bug is the most common silent failure in RL, and every metric in §7 would still look
healthy while the agent optimised the wrong thing. Runs in seconds, needs no training.

## 9b. Testing gaps this plan must close

Identified in review, all currently unwritten: backup-controller engagement, intervention-count
accuracy, LQR closed-loop stability, frozen-test-set hash stability, and the fast-predicate /
full-evaluator agreement test above.

## 10. First minimal experiment

2D env, 3 fixed initial conditions, LQR baseline, PPO for 100k steps, single seed, CPU. Purpose
is plumbing, not results: confirm the env steps, the reward is not degenerate, and a policy
improves on random. Expected < 10 minutes.

## 11. Repository placement

New workspace member `packages/rpo-rl/`, depending on `rpo-core`. Optional extra `rl` for
`gymnasium`, `stable-baselines3`, `torch`. The numerics core must remain importable without any
of them.

**CI, decided.** The `rl` extra **is installed in CI** so env and baseline tests actually run and
mypy resolves the imports. Training runs are marked `slow` and excluded from the fast tier, with
a nightly job for a short smoke train. This is not optional: three commits ago the `viz` extra
was absent from CI, `test_plotting.py` aborted collection, and mypy failed on 3.12 only. The
`rl` extra is the same trap with a heavier dependency. Declare the extra on the package that
imports it, not on the workspace root.

---

## GSTACK REVIEW REPORT

Engineering review of the Project 2 plan, run against a complete, public, CI-green Project 1.

| Run | Status | Findings |
|-----|--------|----------|
| Step 0 scope challenge | complete | complexity gate triggered; user chose full 9-milestone scope |
| 1. Architecture | complete | 2 P1, 2 P2 |
| 2. Code quality | complete | 1 P1 (measured) |
| 3. Test coverage | complete | 6 gaps, 2 critical |
| 4. Performance | complete | 1 P1 (measured) |
| Outside voice | **degraded** | Codex CLI not installed; no genuine cross-model check available |

### Findings and resolutions

| # | Finding | Resolution |
|---|---|---|
| P1 | Keep-out `\|ρ\| ≥ R` is non-convex; the "small QP" cannot work as written | Rotating half-space linearisation at the predicted closest approach — convex, conservative |
| P1 | `evaluate_constraints` = 57.9 µs vs a 4.2 µs step: 93 % of env step time | Fast scalar predicate added to `rpo_core.constraints`, shared by env and planner, with an agreement test |
| P1 | Nonlinear propagation measured 467× slower (1937 µs); breaks the 30-min budget | Train on CW, evaluate on nonlinear as a held-out fidelity test |
| P2 | MPC and the safety filter are one component described as two | Build MPC first, derive the filter; state the MPC-vs-itself overlap in results |
| P2 | "Frozen" test set regenerated from code is not frozen | Commit as data with a hash asserted by a test |
| P2 | Abort appears in reward and termination but not in the action space | **Emergent retreat** — recognised from behaviour, action space stays continuous |
| CRIT | No test that the reward cannot be hacked | Degenerate-policy gate as an M1 acceptance criterion |
| CRIT | No test that env and planner agree on "safe" | Covered by the shared-predicate agreement test |

### Outside voice (degraded — same model, not independent)

Codex is not installed, so this is a self-challenge and should be weighted accordingly. Two
things the section review missed:

**The RL-vs-LQR comparison is rigged, in RL's disfavour, and the plan does not say so.** On a
linear system with a quadratic cost, LQR is *optimal*. RL cannot beat it; it can only approach
it. If the headline comparison is run on the unconstrained 2D linear environment, the honest
expected result is "PPO approaches but does not beat LQR" — which reads as a failure unless the
framing is set up front. The comparison only becomes genuinely interesting where LQR is **not**
optimal: hard keep-out constraints, fuel limits, observation delay, and non-quadratic terminal
costs. The evaluation protocol should lead with those regimes.

**Five seeds is thin for RL.** Policy-gradient variance across seeds is large, and 5 seeds gives
intervals wide enough to make most between-method differences statistically silent. Ten is the
current norm. Cheap here, since a Phase 1 run is ~1 minute.

Also unspecified: the observation delay is listed as an environment feature but never given a
magnitude.

### Sequencing

M0 (backlog X7, validator consolidation) runs before any `rpo-rl` code, by decision.

VERDICT: **APPROVE WITH CHANGES.** Scope confirmed at full 9 milestones by the user against a
recommendation to cut. Three P1 findings were measured, not asserted, and all are resolved in
the plan above. The plan is buildable as amended. The outside voice was degraded and a genuine
cross-model review has not been performed.

### Decisions taken after review

- **Abort = emergent retreat.** Action space stays a pure continuous `Box`; abort is recognised
  from sustained range increase or sustained backup-controller command. Keeps SB3 usable
  unmodified, at the cost of a weaker abort-rate claim, which the plan now states.
- **Headline comparison reframed onto constrained regimes** where LQR is not optimal. The
  unconstrained linear case is retained as a verification that the RL implementation is correct,
  not as a performance claim.
- **Seed count raised to 10.**

NO UNRESOLVED DECISIONS
