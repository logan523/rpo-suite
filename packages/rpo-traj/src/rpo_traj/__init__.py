"""rpo_traj: RPO trajectory design and validation (Flagship 1).

Builds on ``rpo_core`` for frames, propagation, relative motion, targeting, constraints,
and metrics. This package owns the mission-level workflow: turning a scenario configuration
into a planned trajectory, a constraint report, a metrics record, and figures.

Two entry points, and the split between them is deliberate. :func:`~rpo_traj.plan.
plan_rendezvous` is the library: it returns a :class:`~rpo_traj.plan.RendezvousPlan` and
prints nothing, so it composes into sweeps and Monte Carlo campaigns. :mod:`rpo_traj.cli`
is the presentation layer and the process exit code.

Nothing here imports matplotlib. The figure suite is reached through a local import inside
:func:`~rpo_traj.plan.plan_rendezvous`, so ``--no-plots`` costs nothing and a headless
install without the ``viz`` extra still runs the full numerics (N-3).
"""

from .plan import (
    DEFAULT_SAMPLE_COUNT,
    MIN_SAMPLE_COUNT,
    PlanningError,
    RendezvousPlan,
    plan_rendezvous,
    target_state_eci,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SAMPLE_COUNT",
    "MIN_SAMPLE_COUNT",
    "PlanningError",
    "RendezvousPlan",
    "__version__",
    "plan_rendezvous",
    "target_state_eci",
]
