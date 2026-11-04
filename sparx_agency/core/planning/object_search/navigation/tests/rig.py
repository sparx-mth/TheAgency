"""The policy's own rig for the narrow-passage layer: a doorway world the planner, the follower and the guards share.

A test module, not a library (house rule: the suites keep their own fakes).
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.object_search.tests.fixtures import FakeDetector, FakeLLM, body_actions, make_policy
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose

RES = 0.1
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
#: The doorway's axis runs along +x through y = 6.0; the wall stands at x = 6.0.
DOOR_X, DOOR_Y = 6.0, 6.0


def doorway_world(width_m=0.8, size=120):
    """A ``size`` x ``size`` cell room split by a wall at ``DOOR_X`` with a doorway ``width_m`` wide on ``DOOR_Y``."""
    g = np.zeros((size, size), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[:, 59:61] = 100
    half = int(round(width_m / RES / 2))
    g[60 - half:60 + half, 59:61] = 0
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def doorway_policy(tilt=True, **overrides):
    """A policy with the LOOK pair (the look-down needs it), warmed up, on the 20 m synthetic map."""
    return make_policy(FakeDetector("bed"), FakeLLM(), actions=body_actions(tilt=tilt), **overrides)


def observe(ep, step, x, y, yaw, pitch=0.0):
    k = ep.camera.intrinsics
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8), np.full((k.height, k.width), 3.0, np.float32),
                             AgentPose(x, y, 0.0, yaw, pitch), ep.camera, "chair", step)


def planned_route(policy, world, start, goal):
    """The 2D route the policy's planner finds, adopted into its route memory as ``_navigate`` would."""
    result = policy.planner.plan(PlanRequest(Pose2D(*start), Pose2D(*goal), frame_id="world"), world)
    assert result.ok, result.message
    return result.path


def through_the_door(start=(3.0, 4.2, math.radians(10)), goal=(9.0, 7.5)):
    return start, goal
