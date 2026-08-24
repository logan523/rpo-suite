"""rpo_core: shared astrodynamics core for the RPO suite.

Units are SI throughout (metres, seconds, radians, kilograms) and are carried in variable
names rather than as attached unit objects. See ``docs/conventions.md`` for the frame,
state-vector, and tolerance conventions that every module in this package assumes, and
``docs/CONTRIBUTING.md`` for the engineering standard the modules are held to.

Only constants, exceptions, and the most frequently used entry points are re-exported here.
Everything else is reached by its module path (``rpo_core.elements``,
``rpo_core.constraints``, ``rpo_core.config``, ``rpo_core.relative.cw``, ...), which keeps
the top-level namespace readable and import cost low.
"""

from .constants import (
    J2_EARTH,
    MU_EARTH_M3_S2,
    R_EARTH_EQUATORIAL_M,
    mean_motion_rad_s,
    orbital_period_s,
)
from .elements import ClassicalElements, cartesian_to_classical, classical_to_cartesian
from .exceptions import (
    DegenerateGeometryError,
    InfeasibleTransferError,
    PropagationError,
    RpoCoreError,
    SingularTransferTimeError,
)
from .frames import hill_basis, relative_state_eci_to_hill, relative_state_hill_to_eci
from .lambert import LambertConvergenceError, solve_lambert
from .metrics import TrajectoryMetrics, compute_metrics, read_metrics, write_metrics
from .montecarlo import (
    CampaignResults,
    MagnitudePointingDispersion,
    NormalDispersion,
    UniformDispersion,
    VectorNormalDispersion,
    run_campaign,
    wilson_interval,
)
from .perturbations import (
    j2_acceleration_m_s2,
    propagate_perturbed,
    secular_raan_rate_rad_s,
    sun_synchronous_inclination_rad,
)
from .propagate import propagate_two_body
from .relative.cw import cw_stm, propagate_cw, two_impulse_transfer
from .relative.nonlinear import (
    check_cw_validity,
    conservative_cw_error_bound_m,
    propagate_relative_nonlinear,
)
from .targeting import (
    CorrectedTransfer,
    IllConditionedJacobianError,
    TargetingConvergenceError,
    correct_two_impulse_transfer,
    raw_cw_terminal_miss_m,
)

__version__ = "0.2.0"

__all__ = [
    "J2_EARTH",
    "MU_EARTH_M3_S2",
    "R_EARTH_EQUATORIAL_M",
    "CampaignResults",
    "ClassicalElements",
    "CorrectedTransfer",
    "DegenerateGeometryError",
    "IllConditionedJacobianError",
    "InfeasibleTransferError",
    "LambertConvergenceError",
    "MagnitudePointingDispersion",
    "NormalDispersion",
    "PropagationError",
    "RpoCoreError",
    "SingularTransferTimeError",
    "TargetingConvergenceError",
    "TrajectoryMetrics",
    "UniformDispersion",
    "VectorNormalDispersion",
    "__version__",
    "cartesian_to_classical",
    "check_cw_validity",
    "classical_to_cartesian",
    "compute_metrics",
    "conservative_cw_error_bound_m",
    "correct_two_impulse_transfer",
    "cw_stm",
    "hill_basis",
    "j2_acceleration_m_s2",
    "mean_motion_rad_s",
    "orbital_period_s",
    "propagate_cw",
    "propagate_perturbed",
    "propagate_relative_nonlinear",
    "propagate_two_body",
    "raw_cw_terminal_miss_m",
    "read_metrics",
    "relative_state_eci_to_hill",
    "relative_state_hill_to_eci",
    "run_campaign",
    "secular_raan_rate_rad_s",
    "solve_lambert",
    "sun_synchronous_inclination_rad",
    "two_impulse_transfer",
    "wilson_interval",
    "write_metrics",
]
