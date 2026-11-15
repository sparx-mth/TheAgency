"""The vocabulary of the ObjectNav layer: data only, no algorithms.

Every type here is a frozen dataclass or an enum, validated on construction.
The algorithms that consume them live in sibling packages
(``action_converter``, ``labels``, ``agent``) and import from here -- never the
other way round.

Python 3.8 syntax; numpy arrives through the observation and camera types.
"""
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS,
    LOOK_ACTIONS,
    NAVIGATION_ACTIONS,
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.command import (
    NavigationCommand,
    Waypoint,
)
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_DISTANCE_TO_GOAL,
    NATIVE_KEYS,
    NATIVE_SOFT_SPL,
    NATIVE_SPL,
    NATIVE_SUCCESS,
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    TERMINATIONS,
    EpisodeMeasurement,
)
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.objnav.types.target import LabelSpec, TargetLabels

__all__ = [
    # actions
    "DiscreteAction",
    "DiscreteActionSpec",
    "ALL_ACTIONS",
    "NAVIGATION_ACTIONS",
    "LOOK_ACTIONS",
    # what the agent sees
    "AgentPose",
    "CameraSpec",
    "ObjNavObservation",
    "ObjNavEpisode",
    # what the policy and the agent say
    "NavigationCommand",
    "Waypoint",
    "AgentDecision",
    # the target
    "LabelSpec",
    "TargetLabels",
    # what the environment measures
    "EpisodeMeasurement",
    "TERMINATION_STOP",
    "TERMINATION_STEP_LIMIT",
    "TERMINATIONS",
    "NATIVE_SUCCESS",
    "NATIVE_SPL",
    "NATIVE_SOFT_SPL",
    "NATIVE_DISTANCE_TO_GOAL",
    "NATIVE_KEYS",
]
