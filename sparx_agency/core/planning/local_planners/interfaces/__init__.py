"""
Local planning interfaces.

Contracts for short-horizon real-time replanners that:
- take current state + a local collision checker
- try to follow a reference (global/smoothed) while avoiding newly observed obstacles
- output a short time-parameterized trajectory (preferred) or a safe fallback
"""

from .local_planner import LocalPlanner
from .types import (
    LocalPlanInput,
    LocalPlanOutput,
    LocalPlanStatus,
    LocalFailureReason,
    LocalReference,
    CollisionChecker,
)
