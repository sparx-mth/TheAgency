"""
Exploration module.

Provides:
- Frontier extraction utilities (OccupancyGrid2D -> frontier Pose2D set)
- VisibilityCoverage: how much of a surveyed building the camera has looked at
- Simple exploration policies (e.g., random-walk goal selection)
- Re-export of canonical exploration interfaces
"""

from sparx_agency.core.planning.interfaces import ExplorationContext, ExplorationDecision, ExplorationPolicy
from .frontier import FrontierParams, extract_frontiers
from .visibility_coverage import SensorCone, VisibilityCoverage, cone_from_intrinsics
from .random_walk import RandomWalkParams, RandomWalkPolicy

__all__ = [
    "ExplorationContext",
    "ExplorationDecision",
    "ExplorationPolicy",
    "FrontierParams",
    "extract_frontiers",
    "SensorCone",
    "VisibilityCoverage",
    "cone_from_intrinsics",
    "RandomWalkParams",
    "RandomWalkPolicy",
]
