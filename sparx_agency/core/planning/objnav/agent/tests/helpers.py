"""The scripted policy, the two-category mapper and the episode and frame builders the headless-agent tests share.

Defined here rather than taken from the dataset tables, so a table change
cannot move these results. A test module, not a library: nothing outside these
tests imports it.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.errors import UnknownCategoryError
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.types.actions import (
    NAVIGATION_ACTIONS,
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.objnav.types.target import TargetLabels

#: Habitat's action geometry, camera tilt included.
HABITAT = DiscreteActionSpec()
#: A benchmark without camera tilt.
NO_TILT = DiscreteActionSpec(actions=NAVIGATION_ACTIONS)

STOP = DiscreteAction.STOP
FORWARD = DiscreteAction.MOVE_FORWARD
LEFT = DiscreteAction.TURN_LEFT
RIGHT = DiscreteAction.TURN_RIGHT

#: The origin, facing east.
ORIGIN = AgentPose(0.0, 0.0)
#: Three metres dead ahead of :data:`ORIGIN`.
AHEAD = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])


def camera(**overrides):
    """An 8x6 pinhole at a LoCoBot-like height, with any field replaced."""
    fields = dict(intrinsics=Intrinsics(width=8, height=6, fx=4.0, fy=4.0,
                                        cx=3.5, cy=2.5),
                  height_m=0.88, min_depth_m=0.1, max_depth_m=5.0)
    fields.update(overrides)
    return CameraSpec(**fields)


#: The camera of every episode and frame built here.
CAMERA = camera()


def episode(**overrides):
    """A Habitat-like episode looking for the TV, with any field replaced."""
    fields = dict(episode_id="ep0", scene_id="scene0", benchmark="fake",
                  split="test", target_category="tv_monitor", camera=CAMERA,
                  action_spec=HABITAT, max_steps=500)
    fields.update(overrides)
    return ObjNavEpisode(**fields)


def observation(step=0, pose=ORIGIN, **overrides):
    """A frame of :func:`episode` at ``step`` and ``pose``, with any field replaced."""
    fields = dict(rgb=np.zeros((6, 8, 3), np.uint8),
                  depth_m=np.ones((6, 8), np.float32), pose=pose,
                  camera=CAMERA, target_category="tv_monitor", step=step)
    fields.update(overrides)
    return ObjNavObservation(**fields)


class TinyMapper(LabelMapper):
    """Two HM3D categories, spelled as the dataset spells them."""

    name = "tiny"
    TARGETS = {
        "chair": TargetLabels(category="chair", query="chair",
                              detector_prompts=("chair",),
                              accept_labels=frozenset({"chair"})),
        "tv_monitor": TargetLabels(category="tv_monitor", query="tv",
                                   detector_prompts=("tv", "monitor"),
                                   accept_labels=frozenset(
                                       {"tv", "monitor", "television"})),
    }

    def categories(self):
        return tuple(self.TARGETS)

    def target_labels(self, category):
        if category not in self.TARGETS:
            raise UnknownCategoryError(category)
        return self.TARGETS[category]

    def vocabulary(self):
        return ("chair", "tv", "monitor")


class ScriptedPolicy:
    """Returns its commands in order, repeating the last; remembers what it was told."""

    name = "scripted"

    def __init__(self, *commands):
        self.commands = list(commands) or [NavigationCommand.hold()]
        self.resets = []
        self.planned = 0

    def reset(self, episode, target):
        self.resets.append((episode, target))
        self.planned = 0

    def plan(self, observation):
        command = self.commands[min(self.planned, len(self.commands) - 1)]
        self.planned += 1
        return command


def agent_with(*commands, **kwargs):
    """A headless agent around ``ScriptedPolicy(*commands)`` and :class:`TinyMapper`."""
    return HeadlessObjNavAgent(ScriptedPolicy(*commands), TinyMapper(), **kwargs)


def started(*commands, **kwargs):
    """:func:`agent_with`, already reset to the default :func:`episode`."""
    agent = agent_with(*commands, **kwargs)
    agent.reset(episode())
    return agent
