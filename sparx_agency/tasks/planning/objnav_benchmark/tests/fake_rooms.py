"""Small rooms for the fake environment's tests, drawn so every expected number can be worked out by hand.

Shared by the environment, episode, rendering and oracle tests, so they all
read the same building: a room with a chair near its east wall, and the same
room split in two with no door.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.camera_geometry import (
    intrinsics_from_hfov,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import (
    FakeEpisodeSpec,
    FakeObjNavEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

RES = 0.25
#: One room, a chair near its east wall: cells 2..7 of the navigable rows are
#: more than 1 m from the chair, cells 8..10 of row 4 are in its goal region.
ROOM = (
    "################",
    "#..............#",
    "#..............#",
    "#..............#",
    "#...........c..#",
    "#..............#",
    "#..............#",
    "################",
)
#: Two rooms with no door: the chair is in the east one.
SEALED = (
    "################",
    "#.....#........#",
    "#.....#........#",
    "#.....#........#",
    "#.....#.....c..#",
    "#.....#........#",
    "#.....#........#",
    "################",
)
#: Stored ``(row, col)`` cells of ROOM: west of the goal region, and in it.
OUTSIDE, INSIDE = (4, 3), (4, 10)
#: Walking from OUTSIDE to the nearest goal cell, (4, 8): five cells east.
OUTSIDE_TO_GOAL_M = 5 * RES


def small_camera(max_depth_m=5.0):
    """9 x 5 pixels, 60 degrees across: odd, so the centre pixel is on the optical axis."""
    return CameraSpec(intrinsics_from_hfov(9, 5, 60.0), height_m=0.88,
                      min_depth_m=0.1, max_depth_m=max_depth_m)


def pose_at(world, cell, yaw=0.0, pitch=0.0):
    x, y = world.cell_center(*cell)
    return AgentPose(x, y, 0.0, yaw, pitch)


def room_env(starts=(("out", OUTSIDE, 0.0),), rows=ROOM, **options):
    """The fake on ``rows``; ``starts`` are ``(episode_id, cell, yaw)`` looking for the chair."""
    world = GridWorld.from_ascii(rows, {"c": "chair"})
    episodes = [FakeEpisodeSpec(episode_id, "room", "chair", pose_at(world, cell, yaw))
                for episode_id, cell, yaw in starts]
    options.setdefault("camera", small_camera())
    return FakeObjNavEnv(world, episodes, **options)


def play(env, episode_id, actions):
    """Reset ``episode_id`` and execute ``actions``; the last observation."""
    _, observation = env.reset(episode_id)
    for action in actions:
        observation = env.step(action)
    return observation
