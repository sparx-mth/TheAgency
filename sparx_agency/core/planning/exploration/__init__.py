"""
Exploration module.

Provides:
- Frontier extraction utilities (OccupancyGrid2D -> frontier Pose2D set)
- Simple exploration policies (e.g., random-walk goal selection)
- Re-export of canonical exploration interfaces
"""

from sparx_agency.core.planning.interfaces import ExplorationContext, ExplorationDecision, ExplorationPolicy
from .frontier import FrontierParams, extract_frontiers
from .random_walk import RandomWalkParams, RandomWalkPolicy

__all__ = [
    "ExplorationContext",
    "ExplorationDecision",
    "ExplorationPolicy",
    "FrontierParams",
    "extract_frontiers",
    "RandomWalkParams",
    "RandomWalkPolicy",
]
