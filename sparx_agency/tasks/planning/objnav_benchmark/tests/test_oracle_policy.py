"""The privileged oracle aims where it can finish, walks clear of the walls, and counts every blocked step it is told of.

It is the upper bound every pipeline test leans on, so the two subtleties
that decide whether it arrives -- how deep inside the goal region it aims,
and how far from the walls it walks -- are pinned here, with the hook through
which the headless agent tells it a step was refused.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.geodesics import (
    CLEARANCE_PENALTY,
    clearance_costs,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import (
    fake_label_mapper,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_aim import (
    erode_goal_region,
    erosion_cells,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_policy import (
    OracleSearchPolicy,
)
from sparx_agency.tasks.planning.objnav_benchmark.tests.fake_rooms import (
    INSIDE,
    OUTSIDE,
    RES,
    SEALED,
    pose_at,
    room_env,
)

A = DiscreteAction

#: A chair in open floor, four cells from every wall.
OPEN = (
    "#############",
    "#...........#",
    "#...........#",
    "#...........#",
    "#...........#",
    "#.....c.....#",
    "#...........#",
    "#...........#",
    "#...........#",
    "#...........#",
    "#############",
)


def oracle_for(env, episode_id):
    """An oracle reset for ``episode_id``, and that episode's first observation."""
    episode, observation = env.reset(episode_id)
    oracle = OracleSearchPolicy(env)
    oracle.reset(episode, fake_label_mapper().target_labels("chair"))
    return oracle, observation


# -- where it aims, and how it walks ------------------------------------------------

def test_the_aim_sits_two_cells_deep_for_the_default_tolerance_and_step():
    """(0.25 + 0.25) / 0.25 must read 2; a rounding 3 would empty small goal regions."""
    assert erosion_cells(0.25, 0.25, 0.25) == 2
    assert erosion_cells(0.25, 0.25, 0.2) == 3


def test_the_aim_is_eroded_from_the_open_side_only():
    """Eroding from the wall side too would empty goal regions against walls for nothing."""
    navigable = np.ones((9, 9), dtype=bool)
    navigable[:, 0] = False                          # a wall along column 0
    goal = np.zeros((9, 9), dtype=bool)
    goal[:, 1:6] = True                              # open side: column 6
    aim = erode_goal_region(goal, navigable, 2)
    expected = np.zeros((9, 9), dtype=bool)
    expected[:, 1:4] = True                          # columns 4 and 5 eroded
    assert (aim == expected).all()
    assert (erode_goal_region(goal, navigable, 0) == goal).all()


def test_clearance_costs_weigh_cells_next_to_anything_non_navigable():
    """The penalty is what keeps the oracle's route clear of the corners the converter cuts."""
    navigable = np.ones((5, 5), dtype=bool)
    navigable[2, 2] = False
    costs = clearance_costs(navigable)
    heavy = 1.0 + CLEARANCE_PENALTY
    assert costs[1, 1] == heavy and costs[2, 3] == heavy   # beside the blocked cell
    assert costs[0, 2] == heavy                             # on the grid's edge
    assert costs[2, 2] == 1.0                               # blocked: never walked


# -- what it plans ---------------------------------------------------------------------

def test_the_oracle_stops_in_the_goal_and_follows_every_second_descent_cell_elsewhere():
    """Its path must start where the agent stands and end deep inside the region it scores in."""
    env = room_env(starts=[("out", OUTSIDE, 0.0), ("in", INSIDE, 0.0)])
    oracle, observation = oracle_for(env, "in")
    assert oracle.plan(observation).stop
    oracle, observation = oracle_for(env, "out")
    command = oracle.plan(observation)
    world = env.privileged_world()
    points = command.waypoints
    assert points[0] == world.cell_center(*OUTSIDE)
    assert len(points) == command.info["path_cells"] // 2 + 1
    assert all(math.dist(a, b) <= 2 * RES * math.sqrt(2) + 1e-9
               for a, b in zip(points, points[1:]))
    last = world.cell_of(*points[-1])
    assert env.privileged_goal_mask("chair")[last]
    assert command.info["reason"] == "descend"
    assert oracle.episode_info() == {"erosion_cells": 2, "aim_is_full_region": False,
                                     "plans": 1, "stops": 0, "holds": 0, "follows": 1,
                                     "blocked": 0}


def test_the_oracle_holds_where_its_aim_cannot_be_reached():
    """No path is a hold -- the agent turns in place -- never a STOP short of the goal."""
    env = room_env(starts=[("east", (4, 8), 0.0)], rows=SEALED)
    oracle, observation = oracle_for(env, "east")
    world = env.privileged_world()
    west = ObjNavObservation(rgb=observation.rgb, depth_m=observation.depth_m,
                             pose=pose_at(world, (3, 3)), camera=observation.camera,
                             target_category="chair", step=1)
    command = oracle.plan(west)
    assert not command.stop and command.waypoints == ()
    assert command.info["reason"] == "no path"


def test_an_aim_eroded_to_nothing_falls_back_to_the_whole_region():
    """A thin goal region must still be aimed at, not reported unreachable."""
    # At 0.4 m the region is the one ring of cells just outside the chair's
    # clearance band, and open floor lies beside every one of them.
    env = room_env(starts=[("out", (2, 2), 0.0)], rows=OPEN, success_distance_m=0.4)
    goal = env.privileged_goal_mask("chair")
    assert goal.sum() == 12
    assert not erode_goal_region(goal, env.privileged_navigable_mask(), 2).any()
    oracle, observation = oracle_for(env, "out")
    assert oracle.episode_info()["aim_is_full_region"]
    last = env.privileged_world().cell_of(*oracle.plan(observation).waypoints[-1])
    assert goal[last]


def test_the_oracle_refuses_anything_but_the_fake_and_planning_before_reset():
    """It is privileged by construction: only the fake has the ground truth it reads."""
    with pytest.raises(TypeError, match="FakeObjNavEnv"):
        OracleSearchPolicy(object())
    env = room_env()
    observation = env.reset("out")[1]
    with pytest.raises(ObjNavError, match="before reset"):
        OracleSearchPolicy(env).plan(observation)


# -- when a step is blocked ------------------------------------------------------------

def test_the_oracle_counts_the_blocked_steps_it_is_told_of_and_keeps_its_route():
    """Its map is the ground truth: a block is the converter cutting a corner, to count and report, not to re-plan around."""
    env = room_env()
    oracle, observation = oracle_for(env, "out")
    route = oracle.plan(observation)
    assert oracle.notify_blocked(observation) is None
    oracle.notify_blocked(observation)
    assert oracle.plan(observation) == route
    assert oracle.episode_info()["blocked"] == 2
    oracle.reset(env.reset("out")[0], fake_label_mapper().target_labels("chair"))
    assert oracle.episode_info()["blocked"] == 0


def test_the_headless_agent_tells_the_oracle_when_its_forward_step_did_not_move():
    """The count is only worth reading if the agent really calls the hook on a refused step."""
    env = room_env()
    agent = HeadlessObjNavAgent(OracleSearchPolicy(env), fake_label_mapper())
    episode, observation = env.reset("out")
    agent.reset(episode)
    assert agent.act(observation).action == A.MOVE_FORWARD
    agent.act(dataclasses.replace(observation, step=1))   # refused: same pose, one step on
    info = agent.episode_info()
    assert info["blocked_notifications"] == info["policy"]["blocked"] == 1


def test_notify_blocked_refuses_a_call_before_reset_and_anything_but_an_observation():
    """A count taken outside an episode, or of something that is not a frame, would be reported against the wrong one."""
    env = room_env()
    observation = env.reset("out")[1]
    with pytest.raises(ObjNavError, match="before reset"):
        OracleSearchPolicy(env).notify_blocked(observation)
    oracle, _ = oracle_for(env, "out")
    with pytest.raises(TypeError, match="ObjNavObservation"):
        oracle.notify_blocked("frame")
