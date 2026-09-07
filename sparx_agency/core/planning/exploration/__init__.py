"""
Exploration module.

Provides:
- Frontier extraction utilities (OccupancyGrid2D -> frontier Pose2D set)
- VisibilityCoverage: how much of a surveyed building the camera has looked at
- RegionMap: the building as rooms, corridors and the openings between them
- ExplorationSupervisor: one bounded, concrete mission at a time, plus the
  briefing that says it in the policy's own grammar
- survey_state: carry a survey across flights, so one capsize costs a
  segment rather than the whole building
- Simple exploration policies (e.g., random-walk goal selection)
- Re-export of canonical exploration interfaces
"""

from sparx_agency.core.planning.interfaces import ExplorationContext, ExplorationDecision, ExplorationPolicy
from .frontier import FrontierParams, extract_frontiers
from .visibility_coverage import SensorCone, VisibilityCoverage, cone_from_intrinsics
from .region_map import NO_REGION, Portal, Region, RegionMap, load_region_map
from .region_coverage import RegionCoverage, RegionProgress
from .mission import (
    ExplorationSupervisor, Mission, SupervisorParams, SupervisorState,
    AT_DOORWAY, ENTER_ROOM, EXIT_ROOM, IN_CORRIDOR, INSIDE_ROOM, OFF_MAP,
    SCAN_AREA, SURVEY_COMPLETE, TRAVERSE,
)
from .briefing import BriefingStyle, brief
from .survey_state import load_survey, save_survey
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
    "NO_REGION", "Portal", "Region", "RegionMap", "load_region_map",
    "RegionCoverage", "RegionProgress",
    "ExplorationSupervisor", "Mission", "SupervisorParams", "SupervisorState",
    "AT_DOORWAY", "ENTER_ROOM", "EXIT_ROOM", "IN_CORRIDOR", "INSIDE_ROOM",
    "OFF_MAP", "SCAN_AREA", "SURVEY_COMPLETE", "TRAVERSE",
    "BriefingStyle", "brief",
    "load_survey", "save_survey",
    "RandomWalkParams",
    "RandomWalkPolicy",
]
