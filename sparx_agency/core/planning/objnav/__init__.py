"""The ObjectNav layer: our search method, detached from ROS, physics and perception noise.

Everything the ObjectNav benchmarks (MP3D, HM3D v1/v2, Gibson, RoboTHOR) share
and nothing any one of them owns: the contract between a simulator and an
agent, the translation from continuous waypoints to the benchmark's six
discrete actions, the mapping from a dataset's categories to our vision labels,
and the headless agent that composes them behind a one-step API. Every step
receives the simulator's ground-truth RGB-D and perfect pose, so a benchmark
run measures the search logic and nothing else.

Typical use::

    from sparx_agency.core.planning.objnav import (
        HeadlessObjNavAgent, NavigationCommand)

    class WalkToTheDoor:                     # any object with these is a SearchPolicy
        name = "door"

        def reset(self, episode, target):
            self.target = target             # TargetLabels: LLM query, prompts, accept set

        def plan(self, observation):         # world ENU metres in, one command out
            return NavigationCommand.follow([(2.0, 0.0), (2.0, 3.0)])

    agent = HeadlessObjNavAgent(WalkToTheDoor(), label_mapper)
    agent.reset(episode)
    decision = agent.act(observation)        # AgentDecision(action=DiscreteAction..., info=...)

Scoring, logging and the episode loop live in
``sparx_agency.tasks.planning.objnav_benchmark``.

Python 3.8 syntax, numpy only, no ROS.
"""
from sparx_agency.core.planning.objnav.action_converter import (
    ActionConverterParams,
    ConversionResult,
    DiscreteActionConverter,
    Rollout,
    apply_action,
)
from sparx_agency.core.planning.objnav.agent import (
    HeadlessAgentParams,
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.camera_geometry import (
    backproject_depth,
    camera_position_world,
    camera_rotation_world,
    project_to_image,
    world_T_camera_optical,
)
from sparx_agency.core.planning.objnav.camera_intrinsics import (
    intrinsics_from_hfov,
    intrinsics_from_vfov,
)
from sparx_agency.core.planning.objnav.errors import (
    AgentContractError,
    CommandError,
    EnvContractError,
    ObjNavError,
    ObjNavInternalError,
    ObservationError,
    UnknownCategoryError,
)
from sparx_agency.core.planning.objnav.interfaces import (
    LabelMapper,
    ObjNavAgent,
    ObjNavEnv,
    SearchPolicy,
)
from sparx_agency.core.planning.objnav.labels import (
    LabelMapperFactory,
    LabelMapperRegistry,
    TableLabelMapper,
    default_label_mapper_registry,
)
from sparx_agency.core.planning.objnav.types import (
    ALL_ACTIONS,
    LOOK_ACTIONS,
    NATIVE_DISTANCE_TO_GOAL,
    NATIVE_KEYS,
    NATIVE_SOFT_SPL,
    NATIVE_SPL,
    NATIVE_SUCCESS,
    NAVIGATION_ACTIONS,
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    TERMINATIONS,
    AgentDecision,
    AgentPose,
    CameraSpec,
    DiscreteAction,
    DiscreteActionSpec,
    EpisodeMeasurement,
    LabelSpec,
    NavigationCommand,
    ObjNavEpisode,
    ObjNavObservation,
    TargetLabels,
    Waypoint,
)

__all__ = [
    # the contracts
    "ObjNavEnv",
    "ObjNavAgent",
    "SearchPolicy",
    "LabelMapper",
    # the action space
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
    "TableLabelMapper",
    "LabelMapperFactory",
    "LabelMapperRegistry",
    "default_label_mapper_registry",
    # from waypoints to actions
    "DiscreteActionConverter",
    "ActionConverterParams",
    "ConversionResult",
    "Rollout",
    "apply_action",
    # the headless agent
    "HeadlessObjNavAgent",
    "HeadlessAgentParams",
    # camera geometry
    "intrinsics_from_hfov",
    "intrinsics_from_vfov",
    "camera_rotation_world",
    "camera_position_world",
    "world_T_camera_optical",
    "backproject_depth",
    "project_to_image",
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
    # failures
    "ObjNavError",
    "ObservationError",
    "UnknownCategoryError",
    "CommandError",
    "EnvContractError",
    "AgentContractError",
    "ObjNavInternalError",
]
