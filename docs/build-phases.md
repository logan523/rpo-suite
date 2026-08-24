# Build phase graph

Sequenced so that nothing is built on an unvalidated foundation. Within a phase, items on
the same row are independent and run in parallel; rows are barriers.

```
PHASE 0 — foundation                                                    [COMPLETE]
  frames · CW STM · two-impulse targeting · two-body propagator
  nonlinear relative oracle · measured CW validity envelope (6π·ρ²/r)

PHASE A — Project 1 MVP: make it runnable
  A1 config + scenario model + run dir     A2 constraints (KOZ/cone/closing velocity)
                          |                            |
                          +-------------+--------------+
                                        |
                          A3 metrics + plotting
                                        |
                          A4 rpo-traj package + CLI            <-- MVP COMPLETE

PHASE B — Project 1 Phase 2: targeting depth
  B1 orbital elements       B2 Lambert solver
              |                     |
              +----------+----------+
                         |
              B3 differential correction / nonlinear targeting
                         |
              B4 Δv-vs-TOF optimisation + three-baseline comparison

PHASE C — Project 1 Phase 3: fidelity and uncertainty
  C1 J2 + drag        C2 finite burns        C3 Monte Carlo utilities
                              |
                              +----> C4 navigation error + covariance propagation
                                            |
                                     C5 safety / collision analysis under dispersion

PHASE D — Project 1 Phase 4: validation and polish
  D1 GMAT oracle (licence-free path)     D2 STK + Astrogator MCS
                              |
                    D3 README · write-up · demo · results tables

PHASE E — Project 2: safe RL for autonomous rendezvous
  E1 Gymnasium 2D Hill env    E2 LQR baseline    E3 MPC baseline
                              |
              E4 safety filter (projection + backup controller)
                              |
              E5 PPO/SAC training harness + curriculum
                              |
              E6 evaluation protocol · ablations · results schema

PHASE F — Project 3: RPO payload inspection mission planner
  F1 payload + resource model    F2 observation-quality model    F3 access/lighting
                              |
              F4 trajectory planner    F5 GA scheduler (pymoo)
                              |
              F6 coupled planner + decoupled baseline (report the coupling gap)
                              |
              F7 STK sensors/access cross-check · outputs · report
```

## Parallelisation rule

Agents working concurrently are given **disjoint file sets** and are forbidden from touching
shared files (`pyproject.toml`, any `__init__.py`, `README.md`). Package wiring and export
surfaces are integrated by the orchestrator after each row completes. This avoids merge
conflicts without requiring a git worktree per agent.
