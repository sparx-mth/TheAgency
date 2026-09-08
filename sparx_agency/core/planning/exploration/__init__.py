"""
Exploration module.

Provides:
- Frontier extraction utilities (OccupancyGrid2D -> frontier Pose2D set)
- VisibilityCoverage: how much of a surveyed building the camera has looked at
- RegionMap: the building as rooms, corridors and the openings between them
- ExplorationSupervisor: one bounded, concrete mission at a time, plus the
  briefing that says it in the policy's own grammar
- RoomSearchPolicy: its sibling for a *ranked* search -- draw a room from a
  probability distribution over rooms, fly to it, dwell, repeat
- ObjectSearchSupervisor: find one named object fast -- select a room from a
  solver's order, fly to it, map it under a budget, repeat, stop on detection
- room_costs (by module path, not from here): arc weights between room
  centres and the HPP-PT instance a solver such as RPT* consumes
- rpt_room_solver (by module path, not from here): the solver itself --
  the room order that minimises the expected time to find the target
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
from .room_search_policy import (
    RoomSearchParams, RoomSearchPolicy, RoomSearchState, RoomCandidate,
    RoomOption, PublishGoal, Hold, ReSample, DWELL, IDLE, PURSUING,
)
from .object_search_supervisor import (
    ObjectSearchParams, ObjectSearchSupervisor, ObjectSearchState, RoomFacts,
    FlyTo, SearchRoom, Release, StandDown, weighted_order,
    SELECT, TRANSIT, SEARCH, FOUND,
    MAPPED, BUDGET_SPENT, STALLED, UNREACHABLE, TRANSIT_TIMEOUT, BLOCKED,
    PRODUCTIVE,
)
# ``room_costs`` is DELIBERATELY not re-exported. It needs numpy and scipy,
# and this facade is imported inside the Noetic FALCON container (whose
# mission_watchdog_node imports .progress_monitor through it) where scipy does
# not exist. Import it by module path -- ``from ...exploration.room_costs
# import build_instance`` -- from the ROS 2 side only.
#
# ``rpt_room_solver`` is not re-exported either, for a milder reason: it is
# standard-library-only and would import cleanly in the container, but it
# pulls the whole ``routing.rpt_star`` package in behind it and nothing in
# there has any use for a router. Same rule -- import it by module path,
# ``from ...exploration.rpt_room_solver import RptStarRoomSolver``.
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
    "RoomSearchParams", "RoomSearchPolicy", "RoomSearchState",
    "RoomCandidate", "RoomOption", "PublishGoal", "Hold", "ReSample",
    "DWELL", "IDLE", "PURSUING",
    "ObjectSearchParams", "ObjectSearchSupervisor", "ObjectSearchState",
    "RoomFacts", "FlyTo", "SearchRoom", "Release", "StandDown",
    "weighted_order", "SELECT", "TRANSIT", "SEARCH", "FOUND",
    "MAPPED", "BUDGET_SPENT", "STALLED", "UNREACHABLE", "TRANSIT_TIMEOUT",
    "BLOCKED", "PRODUCTIVE",
    "BriefingStyle", "brief",
    "load_survey", "save_survey",
    "RandomWalkParams",
    "RandomWalkPolicy",
]
