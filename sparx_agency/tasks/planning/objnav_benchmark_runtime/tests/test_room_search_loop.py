"""Regressions for the room-search loop: the seven steps, one action at a time.

The world is two rooms joined by a door, drawn by hand so every decision can be
checked against the drawing: room A on the west with an unexplored west end,
room B on the east with a small unknown patch near the door and an unexplored
east end. The scene graph is installed directly (the segmentation is not under
test), the room LLM is a counter, the solver is fixed so the order is known,
and A*, the frontier ranking and the supervisor are real.

Grid: 0.1 m cells, 12 m x 6 m, origin (0, 0); ``(x, y)`` world metres.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_registry import TrackedRoom
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    BUDGET_SPENT, EXHAUSTED, RECLASSIFIED, SEARCH, SELECT, TRANSIT, TRAVERSED, UNREACHABLE, ObjectSearchSupervisor,
    RoomFacts)
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods import frontier_sweep
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import ROOM_LLM
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_nodes import stair_node_id
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy

RES = 0.1
FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)
IN_A, IN_B, IN_DOOR = (3.0, 3.0), (9.0, 1.5), (6.0, 3.0)


def two_room_world(room_b_resolved=False):
    """Room A (x < 5.5) and room B (x > 6.6) joined by a 1.1 m door at y ~ 3."""
    h, w = 60, 120
    g = np.full((h, w), FREE, np.int8)
    g[0, :] = g[-1, :] = OCC
    g[:, 0] = g[:, -1] = OCC
    g[:, 55:66] = OCC                     # the wall between the rooms
    g[25:36, 55:66] = FREE                # the door through it
    g[1:59, 1:11] = UNK                   # room A: unexplored west end -> frontier at x ~ 1.15
    if not room_b_resolved:
        g[52:58, 68:73] = UNK             # room B: a patch near the door -> frontier ring around it
        g[1:59, 109:119] = UNK            # room B: unexplored east end -> frontier at x ~ 10.85
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    labels = np.zeros((h, w), np.int32)
    free = g == FREE
    cols = np.arange(w)[None, :]
    labels[free & (cols < 55)] = 1
    labels[free & (cols > 65)] = 2
    return world, labels


def install_rooms(policy, world, labels, probs=(0.6, 0.4)):
    """Put the two rooms into the policy's scene graph as the segmentation would."""
    rooms = {}
    for pid, label in ((0, 1), (1, 2)):
        mask = labels == label
        ys, xs = np.nonzero(mask)
        centroid = world.grid_to_world(int(round(xs.mean())), int(round(ys.mean())))
        rooms[pid] = TrackedRoom(id=pid, mask=mask, n_cells=int(mask.sum()),
                                 centroid=(float(centroid[0]), float(centroid[1])))
    g = policy.graph
    g.registry.rooms = rooms
    g.labels = labels
    g.probs = {0: probs[0], 1: probs[1]}
    g.options = [RoomOption(room_id=pid, prob=g.probs[pid], xy=room.centroid, label="R%d" % pid)
                 for pid, room in rooms.items()]
    g.facts = {pid: RoomFacts(pid, 1 if pid == 0 else 2, 0.0, room.n_cells) for pid, room in rooms.items()}
    return rooms


class RecordingSupervisor:
    """The real supervisor, recording the flags every call carried."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def update(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.inner.update(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def loop_policy(order=(1, 0), probs=(0.6, 0.4), room_b_resolved=False, llm=None, stair_prob=0.3, metadata=None,
                visit="sweep", **loop_overrides):
    """A policy on the two-room world, with a counting LLM and a fixed room order.

    The background process is stubbed: rooms and labels are installed by hand,
    so ``graph.update`` must not re-segment them, and ``graph.reason`` counts
    the steps it was called on instead of asking a model -- giving every room
    without a probability 0.05 and every stair node offered ``stair_prob``,
    and recording the nodes and context it was handed. ``llm`` replaces the
    neutral fake for the classifier (the oracle is stubbed either way);
    ``metadata`` is the episode's scene structure (the ground-truth stairs).
    ``visit`` is the loop's visit mode: these regressions were written for
    the bounded ``sweep`` and keep running it; the scan-mode tests ask for
    ``visit="scan"``.
    """
    policy, episode = setup_policy(llm=llm, metadata=metadata or {})
    policy.loop_settings = LoopSettings(visit=visit, **loop_overrides)
    policy.supervisor_params = policy.loop_settings.supervisor_params(policy.settings.seed)
    policy.loop.settings = policy.loop_settings
    world, labels = two_room_world(room_b_resolved)
    rooms = install_rooms(policy, world, labels, probs)
    reasoned, refreshed = [], []

    def reason(world_, target, step, extra_nodes=(), context=None, here_xy=None, action_time_s=1.0, exclude=()):
        reasoned.append(step)
        policy.reasoned_with.append({"stairs": [n.id for n in extra_nodes], "context": context, "here": here_xy,
                                     "exclude": tuple(exclude)})
        for pid in policy.graph.registry.rooms:             # like the real one: every room gets a probability
            policy.graph.probs.setdefault(pid, 0.05)
        for pid in exclude:
            policy.graph.probs[pid] = 0.0
        policy.graph.stair_probs = {n.id: stair_prob for n in extra_nodes}
        policy.graph.p_present = min(1.0, sum(policy.graph.probs.values()) + sum(policy.graph.stair_probs.values()))

    policy.reasoned_with = []
    policy.graph.reason = reason
    policy.graph.update = lambda *args, **kwargs: refreshed.append(kwargs.get("step"))
    policy.refreshed = refreshed

    def solve(candidates, instance=None):
        live = {c.room_id for c in candidates}
        return [r for r in order if r in live]

    policy.supervisor = RecordingSupervisor(ObjectSearchSupervisor(policy.supervisor_params, solver=solve))
    return policy, episode, world, rooms, reasoned


def obs_at(episode, step, xy, yaw=0.0):
    return replace(observation(episode, step), pose=AgentPose(xy[0], xy[1], 0.0, yaw))


def cell_in(world, room, xy):
    gx, gy = world.world_to_grid(*xy)
    return bool(room.mask[gy, gx])


# -- steps 2-6 on the first action --------------------------------------------
def test_the_first_action_reasons_once_and_transits_to_the_nearest_frontier_inside_the_chosen_room():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)

    assert reasoned == [0], "steps 2-3 ran once, before any room had a probability"
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    assert command.info["kind"] == "transit/1"
    goal = policy.route_memory.goal
    assert cell_in(world, rooms[1], goal), "the transit aims inside room B"
    assert goal != rooms[1].centroid, "... at a frontier, not the centroid"
    assert goal[0] < 8.0 and goal[1] > 4.5, "... the patch near the door, not the far east end"
    assert all(cell_in(world, rooms[1], (p.x, p.y)) or p.x <= 6.7 for p in policy.route_memory.path.points), (
        "the route runs from room A through the door into room B and nowhere else")
    assert policy.loop.events[-1]["event"] == "transit" and policy.loop.events[-1]["entry"] == "frontier"
    assert policy.loop.stats["entry_frontier"] == 1 and policy.loop.stats["llm_reasonings"] == 1


def test_the_estimate_record_carries_probability_frontiers_time_and_the_distance_the_order_was_charged():
    policy, episode, world, rooms, _ = loop_policy(order=(0, 1))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    estimates = policy.loop.estimates
    assert set(estimates) == {0, 1}
    assert estimates[0]["prob"] == 0.6 and estimates[1]["prob"] == 0.4
    assert estimates[0]["frontier_clusters"] == 1 and estimates[1]["frontier_clusters"] == 2
    assert estimates[0]["searched_s"] == 0.0
    assert estimates[0]["entry"] == estimates[1]["entry"] == "frontier"
    near, far = estimates[0]["distance_m"], estimates[1]["distance_m"]
    assert 1.5 < near < 2.5, "room A's frontier is under two metres west of the agent"
    assert 4.5 < far < 7.0, "room B's nearest frontier is through the door and up"
    assert near < far
    assert policy.loop.diagnostics()["estimates"]["0"]["label"] == "R0"


def test_without_entry_frontier_the_transit_aims_at_the_centroid_and_arrives_anywhere_in_the_room():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.settings = LoopSettings(visit="sweep", entry_frontier=False)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.route_memory.goal == rooms[1].centroid
    assert policy.loop.stats["entry_centroid"] == 1
    policy.loop.plan(obs_at(episode, 1, IN_B), world)         # inside room B, far from its centroid
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    assert policy.supervisor.calls[-1].get("arrived") is True


# -- step 7: arrival ----------------------------------------------------------
def test_arrival_is_the_agents_cell_inside_the_room_mask_at_its_frontier_and_resets_the_counter():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    goal = policy.route_memory.goal
    policy.loop.local_steps = 7                                # stale, from a previous room

    still_transit = policy.loop.plan(obs_at(episode, 1, IN_B), world)
    assert policy.supervisor.state == TRANSIT, "inside the room but not yet at its frontier"
    assert still_transit.info["kind"] == "transit/1"
    assert not any(call.get("arrived") for call in policy.supervisor.calls)

    arrived = policy.loop.plan(obs_at(episode, 2, goal), world)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    assert policy.loop.local_steps == 0, "step 7: the local counter restarts on arrival"
    assert policy.supervisor.stats["arrivals"] == 1
    assert arrived.info["kind"] == "frontier", "the same action starts the in-room sweep"
    assert cell_in(world, rooms[1], policy.route_memory.goal)
    assert reasoned == [0], "no LLM call on arrival"
    assert policy.loop.events[-1]["event"] == "arrive"


def test_arrival_by_the_mask_when_the_entry_frontier_resolves_as_the_agent_closes_in():
    """Inside the room, two metres from the point, but the boundary it aimed at is gone:
    the supervisor's distance test cannot fire, so the loop's own test must."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    goal = policy.route_memory.goal
    seen = two_room_world()[0]
    seen.grid[52:58, 68:73] = FREE                             # the near patch got observed
    inside_far = (goal[0] + 1.5, goal[1] - 1.5)
    assert cell_in(seen, rooms[1], inside_far) and math.dist(inside_far, goal) > 2.0
    command = policy.loop.plan(obs_at(episode, 1, inside_far), seen)
    assert policy.supervisor.calls[-1].get("arrived") is True
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    assert command.info["kind"] == "frontier"


# -- step 1: bounded, confined local exploration -------------------------------
def in_room(policy, episode, world, room_id=1, order=(1, 0)):
    """Drive the loop to SEARCH inside ``room_id`` and return the arrival step."""
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    goal = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, goal), world)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == room_id
    return goal


def test_in_room_routes_are_planned_with_every_other_room_blocked(monkeypatch):
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    here = in_room(policy, episode, world)
    planned = []

    def navigate(obs, planning_world, goal, kind, final_yaw=None):
        planned.append((planning_world, goal, kind))
        return frontier_sweep.NavigationCommand.follow([(here[0], here[1]), goal], info={"kind": kind})

    monkeypatch.setattr(policy, "_navigate", navigate)
    policy.loop.plan(obs_at(episode, 2, here), world)
    planning_world, goal, kind = planned[-1]
    assert kind == "frontier" and cell_in(world, rooms[1], goal)
    assert planning_world is not world
    assert (planning_world.grid[rooms[0].mask] == UNK).all(), "room A is blocked in the planning copy"
    assert (planning_world.grid[rooms[1].mask] == world.grid[rooms[1].mask]).all(), "room B is untouched"
    assert (world.grid[rooms[0].mask] == FREE).all(), "the observed map itself is not written"
    assert policy.loop.stats["confined_actions"] >= 1


def test_confinement_is_lifted_while_the_agent_stands_outside_the_room(monkeypatch):
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    in_room(policy, episode, world)
    planned = []
    monkeypatch.setattr(policy, "_navigate", lambda obs, w, goal, kind, final_yaw=None: planned.append(w) or
                        frontier_sweep.NavigationCommand.follow([IN_DOOR, goal], info={"kind": kind}))
    policy.loop.plan(obs_at(episode, 2, IN_DOOR), world)    # standing in the doorway, label 0
    assert planned[-1] is world
    assert policy.loop.stats["unconfined_actions"] == 1


def test_confine_routes_off_plans_on_the_observed_map(monkeypatch):
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.settings = LoopSettings(visit="sweep", confine_routes=False)
    here = in_room(policy, episode, world)
    planned = []
    monkeypatch.setattr(policy, "_navigate", lambda obs, w, goal, kind, final_yaw=None: planned.append(w) or
                        frontier_sweep.NavigationCommand.follow([here, goal], info={"kind": kind}))
    policy.loop.plan(obs_at(episode, 2, here), world)
    assert planned[-1] is world and policy.loop.stats["confined_actions"] == 0


def test_the_local_budget_ends_the_room_and_the_same_action_reasons_reorders_and_transits():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    in_room(policy, episode, world)
    n = policy.loop.settings.local_steps
    for _ in range(n - 1):
        policy.loop.charge()
    policy.loop.plan(obs_at(episode, 2, IN_B), world)
    assert policy.supervisor.state == SEARCH, "one action short of N is still local exploration"
    policy.loop.charge()
    assert policy.loop.local_steps == n

    command = policy.loop.plan(obs_at(episode, 3, IN_B), world)
    assert policy.supervisor.history[-1][:2] == (1, BUDGET_SPENT)
    assert policy.supervisor.calls[-2].get("budget_spent") is True
    assert policy.supervisor.calls[-1].get("budget_spent", False) is False, "the reselect carries no stale flag"
    assert reasoned == [0, 3], "steps 2-3 ran again at the loop point, and only there"
    assert policy.supervisor.state == TRANSIT, "steps 4-6 ran on the same action"
    assert command.info["kind"].startswith("transit/")
    assert policy.route_memory.path is not None and policy.route_memory.kind.startswith("transit/")
    assert policy.loop.stats["budget_releases"] == 1 and policy.loop.stats["loop_points"] == 1
    assert policy.loop.room_id is None
    assert [e["event"] for e in policy.loop.events[-2:]] == ["release", "transit"]
    assert policy.loop.events[-2]["verdict"] == BUDGET_SPENT and policy.loop.events[-2]["local_steps"] == n


def test_a_budget_spent_room_is_not_cooled_and_may_be_chosen_straight_back():
    """The fresh estimate still favours the room; a cooldown there would refuse the loop's own answer."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    in_room(policy, episode, world)
    for _ in range(policy.loop.settings.local_steps):
        policy.loop.charge()
    command = policy.loop.plan(obs_at(episode, 2, IN_B), world)
    assert policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1"
    assert policy.supervisor.stats["solver_calls"] == 2, "the order was re-solved, not walked"
    assert policy.route_memory.goal[0] > 10.0, "... to the room's frontier nearest the agent now, the east end"


def test_a_finished_room_the_agent_stands_in_is_left_on_the_same_action_not_re_entered():
    """Budget spent inside a room that has since been fully observed.

    The fresh solve still puts the room first, so it is chosen again -- at its
    centroid, since it has no frontier -- and the agent is already inside, so
    it arrives at once; the sweep finds nothing, the room ends EXHAUSTED and
    cools, and the reselect goes to room A. Three supervisor rounds, one
    action, no idle turn, and the room is not offered again while it cools.
    """
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    in_room(policy, episode, world)
    for _ in range(policy.loop.settings.local_steps):
        policy.loop.charge()
    swept = two_room_world(room_b_resolved=True)[0]
    command = policy.loop.plan(obs_at(episode, 2, IN_B), swept)
    assert [h[:2] for h in policy.supervisor.history] == [(1, BUDGET_SPENT), (1, EXHAUSTED)]
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0
    assert command.info["kind"] == "transit/0" and cell_in(swept, rooms[0], policy.route_memory.goal)
    assert policy.loop.estimates[1]["entry"] == "centroid"
    assert [e["event"] for e in policy.loop.events[-5:]] == ["release", "transit", "arrive", "release", "transit"]
    assert reasoned == [0, 2, 2], "one reasoning round per loop point, both on this action"
    assert policy.supervisor.stats["arrivals"] == 2


def test_the_round_guard_fits_the_longest_chain_and_an_overrun_is_visible_not_silent():
    """Budget release, arrival, exhausted release, transit: three rounds spent,
    none wasted on raising the budget flag. Cut the guard to two and the chain
    overruns: the pending verdict is counted and logged, and the fallback
    carries the action rather than an idle turn."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    in_room(policy, episode, world)
    for _ in range(policy.loop.settings.local_steps):
        policy.loop.charge()
    swept = two_room_world(room_b_resolved=True)[0]
    policy.loop.plan(obs_at(episode, 2, IN_B), swept)
    assert policy.loop.stats["supervisor_rounds"] == 2, "two rounds ended without a command; the third commanded"
    assert policy.loop.stats["rounds_exhausted"] == 0

    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.settings = LoopSettings(visit="sweep", supervisor_rounds=2)
    in_room(policy, episode, world)
    for _ in range(policy.loop.settings.local_steps):
        policy.loop.charge()
    command = policy.loop.plan(obs_at(episode, 2, IN_B), swept)
    assert policy.loop.stats["rounds_exhausted"] == 1
    assert policy.loop.events[-1] == {"step": 2, "floor_id": 0, "event": "rounds_exhausted",
                                      "pending": ["frontier_exhausted"]}
    assert command.info["kind"] == "frontier", "the floor-wide frontier, not an idle hold"
    assert policy.supervisor.state == SEARCH, "the verdict is re-derived on the next action"
    policy.loop.plan(obs_at(episode, 3, IN_B), swept)
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)


def test_a_lone_finished_room_is_not_re_entered_while_it_cools_the_loop_falls_back_to_the_floor():
    """The bug the first end-to-end trace showed: one room, exhausted, re-chosen
    by the every-room-cooling escape hatch on every action -- five releases in
    twelve actions and a room LLM call for each. A room the live map says is
    finished is left alone; the floor-wide frontier carries the search."""
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    only_b = {1: rooms[1]}
    policy.graph.registry.rooms = only_b
    policy.graph.labels = np.where(policy.graph.labels == 2, 2, 0).astype(np.int32)
    policy.graph.probs = {1: 0.9}
    policy.graph.options = [o for o in policy.graph.options if o.room_id == 1]
    policy.graph.facts = {1: policy.graph.facts[1]}
    in_room(policy, episode, world)
    swept = two_room_world(room_b_resolved=True)[0]
    first = policy.loop.plan(obs_at(episode, 2, IN_B), swept)
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.supervisor.state == SELECT, "nothing else to choose, and the finished room is not repeated"
    assert first.info["kind"] == "frontier", "the floor-wide frontier (room A's west end) carries the search"
    assert cell_in(swept, rooms[0], policy.route_memory.goal)
    calls = len(reasoned)
    for step in range(3, 8):
        command = policy.loop.plan(obs_at(episode, step, IN_B), swept)
        assert command.info["kind"] == "frontier"
    assert len(policy.supervisor.history) == 1, "no further release"
    assert len(reasoned) == calls, "no further reasoning round"
    assert policy.loop.stats["loop_points"] == 1


def test_a_swept_room_is_released_the_action_its_frontier_runs_out_and_the_next_room_is_entered():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    here = in_room(policy, episode, world)
    swept = two_room_world(room_b_resolved=True)[0]           # room B fully observed
    command = policy.loop.plan(obs_at(episode, 2, here), swept)
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.supervisor.calls[-2].get("frontier_exhausted") is True
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0, (
        "an exhausted room cools; the order falls through to room A")
    assert command.info["kind"] == "transit/0"
    assert cell_in(swept, rooms[0], policy.route_memory.goal)
    assert reasoned == [0, 2]
    assert policy.loop.stats["exhausted_releases"] == 1
    assert policy.sweep.stats["scan_turns"] == 0, "no look-around by default"


def test_charge_counts_only_while_a_room_is_in_force():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.charge()
    assert policy.loop.local_steps == 0, "nothing is in force before the first action"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    policy.loop.charge()
    assert policy.loop.local_steps == 0, "transit actions are not local steps"
    policy.loop.plan(obs_at(episode, 1, policy.route_memory.goal), world)
    policy.loop.charge()
    assert policy.loop.local_steps == 1


# -- step 6 while in transit -----------------------------------------------------
def test_a_room_resolved_from_outside_is_skipped_in_transit_and_the_order_is_re_solved():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.room_id == 1
    resolved, _ = two_room_world(room_b_resolved=True)        # the camera filled room B in from the door
    command = policy.loop.plan(obs_at(episode, 1, IN_A), resolved)
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.supervisor.stats["exhausted_in_transit"] == 1
    assert policy.supervisor.stats["arrivals"] == 0
    assert policy.loop.stats["skipped_in_transit"] == 1
    assert reasoned == [0, 1], "a release is a loop point"
    assert policy.supervisor.room_id == 0 and command.info["kind"] == "transit/0"
    assert policy.loop.estimates[1]["entry"] == "centroid", "a room with no frontier keeps a centroid entry"
    assert cell_in(resolved, rooms[0], policy.route_memory.goal)


def test_a_committed_entry_frontier_that_gets_resolved_is_re_aimed_at_the_rooms_next_one():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    first = policy.route_memory.goal
    partly = two_room_world()[0]
    partly.grid[52:58, 68:73] = FREE                           # only the near patch got seen
    command = policy.loop.plan(obs_at(episode, 1, IN_A), partly)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    assert command.info["kind"] == "transit/1"
    assert policy.route_memory.goal != first and policy.route_memory.goal[0] > 10.0, "re-aimed at the east end"
    assert policy.loop.stats["entry_reaimed"] == 1


def test_a_route_that_ends_on_the_rooms_boundary_counts_as_arrival():
    """The entry frontier sits on the boundary the eroded mask stops short of.

    Standing in the doorway, 0.6 m from room B's mask and 1.6 m from the
    entry point, with the route memory reporting the route finished: that
    is arrival. The first Ranchester run turned in place for 31 actions at
    exactly this kind of spot.
    """
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    policy.route_memory.arrived = lambda obs: True             # the converter says the route is done
    assert not policy.loop._inside(obs_at(episode, 1, IN_DOOR), world, rooms[1])
    command = policy.loop.plan(obs_at(episode, 1, IN_DOOR), world)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    assert policy.supervisor.calls[-1].get("arrived") is True
    assert command.info["kind"] == "frontier"
    assert policy.loop.stats["entry_lost"] == 0


def test_a_route_that_ends_away_from_the_room_re_aims_once_at_its_current_frontier():
    """The map re-segmented under the entry point: the route ended in room A.

    Not arrival, not idling. The old entry is retired, the room's nearest
    frontier now is aimed at, and the transit continues on the same action.
    """
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    first = policy.route_memory.goal
    policy.route_memory.arrived = lambda obs: True             # route finished -- 4 m from room B
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    assert command.info["kind"] == "transit/1"
    assert policy.loop.stats["entry_lost"] == 1 and policy.loop.stats["entry_reaimed"] == 1
    assert tuple(first) in policy._visited_frontiers, "the lost entry is retired, not offered again"
    assert policy.route_memory.goal != first and cell_in(world, rooms[1], policy.route_memory.goal)
    assert policy.loop.events[-1]["event"] == "entry_lost" and policy.loop.events[-1]["reaimed"] is True


def test_a_route_that_ends_away_from_a_room_with_no_frontier_left_releases_it_as_unreachable():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), room_b_resolved=True)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop.estimates[1]["entry"] == "centroid", "room B has no frontier: the centroid is the entry"
    policy.route_memory.arrived = lambda obs: True
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.history[-1][:2] == (1, UNREACHABLE)
    assert policy.loop.stats["entry_lost"] == 1 and policy.loop.stats["entry_reaimed"] == 0
    assert policy.supervisor.room_id == 0 and policy.supervisor.state == SEARCH, (
        "the reselect moves on to room A, which the agent already stands in, so it arrives at once")
    assert command.info["kind"] == "frontier" and cell_in(world, rooms[0], policy.route_memory.goal)
    assert reasoned == [0, 1], "a release is a loop point"


# -- the knobs --------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{"local_steps": 0}, {"supervisor_rounds": 0}, {"local_steps": 2.5},
                                    {"confine_routes": "yes"}, {"entry_frontier": 1}])
def test_loop_settings_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        LoopSettings(**kwargs)


def test_loop_settings_configure_the_supervisor_for_the_loop():
    params = LoopSettings().supervisor_params(seed=3)
    assert params.seed == 3 and params.resolve_on_release is True
    assert BUDGET_SPENT not in params.cooldown_verdicts and EXHAUSTED in params.cooldown_verdicts


def test_the_policy_reports_the_loop_and_its_cadence():
    policy, episode = setup_policy(label="bed")               # not the target, so the loop owns step 0
    configuration = policy.configuration()
    assert configuration["room_search_loop"]["local_steps"] == 10
    assert configuration["supervisor"]["resolve_on_release"] is True
    assert configuration["adaptation"]["graph_period_steps"] == 1
    assert "loop points" in configuration["reasoning_cadence"]
    policy.plan(observation(episode, 0, depth=3.0))
    info = policy.episode_info()
    assert info["room_search_loop"]["settings"]["local_steps"] == 10
    assert info["llm_queries"] == 1, "one reasoning round on the first action; none per action after"
    for step in range(1, 4):
        policy.plan(observation(episode, step, depth=3.0))
    assert policy.episode_info()["llm_queries"] == 1


def test_the_loop_travels_with_its_floor():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    ground = policy.loop
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    policy.floors.activate(1)
    assert policy.loop is not ground and policy.loop.room_id is None and policy.loop.local_steps == 0
    policy.floors.activate(0)
    assert policy.loop is ground


def test_a_loop_point_reasons_over_geometry_refreshed_on_that_action():
    """The background refresh ran on an earlier action: the loop point refreshes first.
    When it already ran on this action, nothing is repeated."""
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    policy._last_graph_step = -3                                # stale by three actions
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.refreshed == [0] and reasoned == [0]
    assert policy._last_graph_step == 0

    policy._last_graph_step = 2                                 # the policy refreshed on this action already
    in_room_goal = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, in_room_goal), world)
    policy._last_graph_step = 2
    for _ in range(policy.loop.settings.local_steps):
        policy.loop.charge()
    policy.loop.plan(obs_at(episode, 2, IN_B), world)
    assert reasoned == [0, 2] and policy.refreshed == [0], "fresh already; not refreshed twice"


def test_a_room_that_appears_while_nothing_is_in_force_is_estimated_at_once():
    """A room without a probability can never be chosen, and no release is
    coming to give it one -- so its appearance is the one event outside a
    loop point that asks the model."""
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    for pid in (0, 1):                                          # both rooms cooling under EXHAUSTED
        policy.supervisor.inner._cooling[pid] = 0.0
        policy.supervisor.inner._cooling_verdict[pid] = EXHAUSTED
    policy.graph.probs = {0: 0.6, 1: 0.4}
    policy.loop._needs_reason = False
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == SELECT and reasoned == [], "every room estimated and cooling: hold, no call"
    assert command.info["kind"] == "frontier", "the floor-wide frontier carries the search"

    new_room = TrackedRoom(id=2, mask=rooms[0].mask, n_cells=rooms[0].n_cells, centroid=rooms[0].centroid)
    policy.graph.registry.rooms[2] = new_room
    policy.graph.options.append(RoomOption(room_id=2, prob=0.0, xy=new_room.centroid, label="R2"))
    policy.graph.facts[2] = RoomFacts(2, 1, 0.0, new_room.n_cells)
    policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert reasoned == [1], "the new room triggered steps 2-3"
    policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert reasoned == [1], "... once; the graph's own probability record stops a repeat"


# -- the clue rule: a new kind of object names the room; the oracle values the new fact ----
class NamingLLM:
    """A classifier that names rooms by their objects. The oracle is the fixture's stub."""
    ROOMS = {"toilet": "bathroom", "sink": "bathroom", "bed": "bedroom", "coat rack": "hallway"}
    def __init__(self, confidence=0.9):
        self.confidence = confidence
        self.calls = 0
    def chat_json(self, system, user, **kwargs):
        if "Room observed objects" in user:
            self.calls += 1
            label = next((room for clue, room in self.ROOMS.items() if clue in user), "living_room")
            return {"label": label, "confidence": self.confidence, "reasoning": "the objects seen"}
        raise AssertionError("the oracle is stubbed in this fixture")
def see(policy, objects, step):
    """The background refresh's half of a clue: evidence recorded, no model call."""
    policy.graph._objects = {pid: list(names) for pid, names in objects.items()}
    policy.graph.label_tracker.update(objects, step, allow_query=False)
def test_a_room_re_identified_on_the_way_in_ends_its_turn_and_the_order_is_re_solved_before_it_is_entered():
    """A toilet seen through room B's doorway while walking to it: room B is a bathroom now, a fact the
    oracle has not valued. The turn ends on the action the clue lands, the estimate and the order are
    redone, and -- the stub oracle unmoved -- the same room is chosen straight back, uncooled."""
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), llm=llm, type_prior=False)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    see(policy, {1: ["toilet"]}, step=1)
    assert llm.calls == 0, "the background refresh records evidence without a call"
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert llm.calls == 1, "the room in transit is re-labelled the action a new kind of object lands in it"
    assert policy.graph.label_info(1)["label"] == "bathroom" and policy.graph.label_info(1)["strength"] == "weak"
    assert policy.supervisor.history[-1][:2] == (1, RECLASSIFIED)
    assert policy.supervisor.stats["reclassified_in_transit"] == 1 and policy.supervisor.stats["arrivals"] == 0
    assert policy.loop.stats["reclassified_releases"] == 1 and policy.loop.stats["reclassified_in_transit"] == 1
    assert reasoned == [0, 1], "a release is a loop point: every node re-valued over the new name"
    assert policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1"
    assert [e["event"] for e in policy.loop.events[-3:]] == ["relabel", "release", "transit"]
    assert policy.loop.events[-3]["where"] == "transit" and policy.loop.events[-3]["label"] == "bathroom"
    assert not policy.supervisor.is_cooling(1, policy._floor_time), "neutral: not cooled, not charged an attempt"
    assert policy.loop.estimates[1]["label"] == "bathroom" and policy.loop.estimates[1]["objects"] == ["toilet"]
def test_the_oracle_decides_what_the_new_name_is_worth():
    """The same clue, with an oracle that gives a bathroom nothing for a chair: the order goes elsewhere."""
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), llm=llm, type_prior=False)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    def oracle(world_, target, step, extra_nodes=(), context=None, here_xy=None, action_time_s=1.0, exclude=()):
        reasoned.append(step)
        info = policy.graph.label_info(1) or {}
        policy.graph.probs = {0: 0.6, 1: 0.0 if info.get("label") == "bathroom" else 0.4}
        policy.graph.p_present = sum(policy.graph.probs.values())
    policy.graph.reason = oracle
    see(policy, {1: ["toilet"]}, step=1)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.history[-1][:2] == (1, RECLASSIFIED)
    assert policy.supervisor.room_id == 0 and command.info["kind"] == "transit/0", "a chair is not in a bathroom"
    assert policy.loop.estimates[1]["prob"] == 0.0 and policy.loop.estimates[0]["prob"] == 0.6
    for step in range(2, 6):
        policy.loop.plan(obs_at(episode, step, IN_A), world)
    assert policy.supervisor.room_id == 0 and llm.calls == 1, "the bathroom is not chosen again, and not re-asked"
def test_a_room_in_force_re_identified_by_a_new_kind_of_object_ends_its_turn_on_that_action():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), llm=NamingLLM(), type_prior=False)
    in_room(policy, episode, world)                             # sweeping room B
    policy.loop.charge()
    see(policy, {1: ["sink"]}, step=2)
    command = policy.loop.plan(obs_at(episode, 2, IN_B), world)
    assert policy.supervisor.history[-1][:2] == (1, RECLASSIFIED)
    assert policy.supervisor.stats["reclassified"] == 1 and policy.supervisor.stats["reclassified_in_transit"] == 0
    assert reasoned == [0, 2]
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1"
    assert [e["event"] for e in policy.loop.events[-3:]] == ["relabel", "release", "transit"]
    assert policy.loop.events[-3]["where"] == "search"
    assert policy.loop.room_id is None and policy.loop.local_steps == 1, "the counter stops where the turn ended"
    assert RECLASSIFIED not in policy.supervisor_params.cooldown_verdicts
    assert TRAVERSED not in policy.supervisor_params.cooldown_verdicts and EXHAUSTED in policy.supervisor_params.cooldown_verdicts
def test_a_second_object_of_a_kind_already_seen_is_not_a_clue():
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), llm=llm, type_prior=False)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    see(policy, {1: ["toilet"]}, step=1)
    policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert llm.calls == 1 and len(policy.supervisor.history) == 1
    see(policy, {1: ["toilet", "toilet", "toilet"]}, step=2)
    policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert llm.calls == 1 and len(policy.supervisor.history) == 1, "three toilets are one kind of evidence"
    see(policy, {1: ["toilet", "sink"]}, step=3)
    policy.loop.plan(obs_at(episode, 3, IN_A), world)
    assert llm.calls == 2, "a new kind is asked about ..."
    assert len(policy.supervisor.history) == 1, "... and the name did not change, so the turn goes on"
    assert policy.graph.label_info(1)["strength"] == "strong"
def test_a_room_llm_in_back_off_leaves_the_name_standing_and_the_turn_running():
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), llm=llm, type_prior=False)
    obs = obs_at(episode, 0, IN_A)
    policy.loop.plan(obs, world)
    policy.fallback.note_service_failure(obs, ROOM_LLM, RuntimeError("away"))
    see(policy, {1: ["toilet"]}, step=1)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert llm.calls == 0 and len(policy.supervisor.history) == 0
    assert policy.supervisor.state == TRANSIT and command.info["kind"] == "transit/1"
# -- the stairs as nodes of the order ----------------------------------------------
FLIGHT = [[4.0, 1.0, 0.0], [4.0, 2.0, 0.9], [4.0, 3.0, 1.8], [4.0, 4.0, 2.7]]
STRUCTURE = {"stair_source": "navmesh",
             "floor_levels": [{"height_m": 0.0, "area_m2": 40.0}, {"height_m": 2.7, "area_m2": 40.0}],
             "stair_connectors": [{"id": 0, "bottom_xyz": FLIGHT[0], "top_xyz": FLIGHT[-1], "bottom_z": 0.0,
                                   "top_z": 2.7, "polyline_xyz": FLIGHT, "length_m": 4.0, "rise_m": 2.7}]}
STAIRS = stair_node_id(0)
FLIGHT_M = 3 * math.hypot(1.0, 0.9)
def stair_policy(order, stair_prob=0.3, probs=(0.6, 0.4), structure=STRUCTURE, **loop_overrides):
    """The two-room world with one ground-truth staircase whose foot stands in room A -- and has been SEEN.

    The coordinator knows only the stairs its perfect detector has had in
    frame; the fixture stands in for that sighting, so the connector becomes
    a portal of this floor when the building observes.
    """
    policy, episode, world, rooms, reasoned = loop_policy(order=order, probs=probs, stair_prob=stair_prob,
                                                          metadata=structure, **loop_overrides)
    obs = obs_at(episode, 0, IN_A)
    policy.mapping.atlas.update(obs.pose)                      # floor 0 at z=0, as the first mapping update does
    for connector in policy.building.ground_truth.connectors:
        policy.building.sightings.mark_seen(connector.id, 0, connector.polyline)
    policy.building.observe(obs)                               # the seen connector becomes a portal on this floor
    return policy, episode, world, rooms, reasoned
def test_a_staircase_is_offered_to_the_oracle_and_the_solver_as_a_node_charged_for_the_climb():
    policy, episode, world, rooms, reasoned = stair_policy(order=(1, 0, STAIRS))
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert len(policy.building.portals) == 1 and policy.building.portals[0]["cooldown_until"] == 0
    assert policy.reasoned_with[0]["stairs"] == [STAIRS], "the staircase went to the oracle beside the rooms"
    context = policy.reasoned_with[0]["context"]
    assert "2 storeys known" in context.storey and context.others == ()
    assert policy.reasoned_with[0]["here"] == IN_A
    assert policy.loop.stats["stairs_offered"] == 1 and sorted(policy.loop._stairs) == [STAIRS]
    estimate = policy.loop.estimates[STAIRS]
    assert estimate["kind"] == "stairs" and estimate["prob"] == 0.3 and estimate["direction"] == 1
    assert not estimate["destination_visited"] and not estimate["arrived_by"] and estimate["portal_id"] == 0
    assert estimate["leaf_m"] == pytest.approx(FLIGHT_M + policy.settings.multifloor.floor_change_cost_m, abs=0.01)
    assert 2.0 < estimate["approach_m"] < 3.0, "the foot of the flight is two-odd metres from the agent"
    assert estimate["distance_m"] == pytest.approx(estimate["approach_m"] + estimate["leaf_m"], abs=0.3), (
        "the leg the order was charged is the walk to the foot plus the flight plus the fixed cost of a storey change")
    assert estimate["entry"] == "stairs"
    assert policy.loop.order == (1, 0, STAIRS) and policy.loop.next_room == 1
    assert command.info["kind"] == "transit/1"
    assert policy.loop.estimate_events[0]["stairs"] == [STAIRS] and policy.loop.estimate_events[0]["nodes"] == 0
    assert policy.supervisor.stats["solver_calls"] == 1
def test_when_the_order_puts_the_stairs_first_the_building_climbs_and_the_node_turn_ends():
    policy, episode, world, rooms, reasoned = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.9)
    building = policy.building
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == STAIRS
    assert policy.loop.stats["stairs_chosen"] == 1 and policy.loop.next_room == STAIRS
    assert building.active is not None and building.active["selected_by"] == "rpt_star"
    assert building.phase == "APPROACH_STAIRS" and building.committed
    assert command.info["kind"] == "portal/0" and command.waypoints, "the approach to the foot of the flight"
    assert list(command.waypoints[-1]) == pytest.approx(FLIGHT[0][:2], abs=0.2)
    assert [e for e in building.events if e["event"] == "portal_selected"][0]["solver_source"] == "rpt_star"
    assert policy.loop.events[-1]["event"] == "transit" and policy.loop.events[-1]["entry"] == "stairs"
    assert not [e for e in building.events if e["event"] == "floor_decision"], "no explicit rule ran: the order decided"
    # From here the policy asks the building, not the loop (``_decide`` on ``building.committed``).
    climb = building.plan(obs_at(episode, 1, (4.2, 1.0)), world)
    assert building.transition is not None and building.phase == "TRAVERSE" and climb.info["kind"] == "TRAVERSE"
    assert climb.info["direction"] == "up" and climb.waypoints
    assert policy.supervisor.state == SELECT and policy.supervisor.history[-1][:2] == (STAIRS, TRAVERSED)
    assert policy.loop.stats["stairs_taken"] == 1 and policy.loop.next_room is None and policy.loop._needs_reason
    assert policy.loop.events[-1]["event"] == "stairs_taken" and policy.loop.events[-1]["portal"] == 0
    assert policy.loop.events[-1]["released"] is True
    started = [e for e in building.events if e["event"] == "traversal_started"][0]
    assert started["selected_by"] == "rpt_star" and started["direction"] == 1
    assert not policy.supervisor.is_cooling(STAIRS, policy._floor_time), "traversed is productive, not cooled"
    assert policy.route_memory.path is None
def test_the_stairs_the_agent_came_down_are_offered_again_with_the_arrival_noted_once_the_storey_was_looked_at():
    """The way back is a node again after ``arrival_grace_actions``, marked ``arrived_by``; not before.

    Offered on the arrival action, the Pomaria node oracle took it three
    times in a row on one tentative room's evidence. The storey is searched
    for the grace first; the decision to climb is given time to pay.
    """
    policy, episode, world, rooms, reasoned = stair_policy(order=(1, 0, STAIRS))
    grace = policy.settings.multifloor.arrival_grace_actions
    policy.building.arrived_by, policy.building.arrived_step = 0, 0
    policy.loop.plan(obs_at(episode, 5, IN_A), world)
    assert policy.reasoned_with[0]["stairs"] == [] and policy.loop._stairs == {}, "5 actions after arriving: not yet"
    assert policy.building.way_back_held(5) and not policy.building.way_back_held(grace)
    policy.loop._needs_reason = True
    policy.loop.plan(obs_at(episode, grace + 5, IN_A), world)
    assert STAIRS in policy.loop._stairs, "the grace is over: the way back is a node like any other"
    node = policy.loop._stairs[STAIRS].node
    assert node.arrived_by and node.arrived_ago_s == pytest.approx(grace + 5.0)
    assert policy.building.portals[0]["cooldown_until"] == 0, "no cooldown on the portal itself: the hold is the grace alone"
    assert policy.loop.estimates[STAIRS]["arrived_by"] is True
def test_a_staircase_the_observed_map_has_not_reached_is_not_a_node_this_action():
    far = [[14.0, 3.0, 0.0], [15.0, 3.0, 0.9], [16.0, 3.0, 1.8], [17.0, 3.0, 2.7]]     # beyond the 12 m world
    structure = dict(STRUCTURE, stair_connectors=[dict(STRUCTURE["stair_connectors"][0], bottom_xyz=far[0],
                                                       top_xyz=far[-1], polyline_xyz=far)])
    policy, episode, world, rooms, reasoned = stair_policy(order=(1, 0, STAIRS), structure=structure)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert len(policy.building.portals) == 1
    assert policy.reasoned_with[0]["stairs"] == [] and policy.loop._stairs == {}
    assert STAIRS not in policy.loop.estimates and policy.loop.order == (1, 0)
def test_a_cooling_staircase_is_not_offered_and_a_chosen_one_that_cooled_is_released_unreachable():
    policy, episode, world, rooms, reasoned = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.9)
    portal = policy.building.portals[0]
    portal["cooldown_until"] = 50                                # deferred after a failed approach
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.reasoned_with[0]["stairs"] == [] and command.info["kind"] == "transit/1"
    policy, episode, world, rooms, reasoned = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.9)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.room_id == STAIRS
    policy.building._abandon(obs_at(episode, 1, IN_A), "test")   # the building gave the portal up mid-approach
    command = policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert policy.supervisor.history[-1][:2] == (STAIRS, UNREACHABLE)
    assert policy.loop.stats["stairs_refused"] == 1 and command.info["kind"] == "transit/1"
def test_stairs_as_nodes_off_is_the_ablation_where_the_stairs_are_the_fallbacks_alone():
    policy, episode, world, rooms, reasoned = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.9)
    policy.loop_settings = policy.loop.settings = LoopSettings(visit="sweep", stairs_as_nodes=False)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.reasoned_with[0]["stairs"] == [] and policy.loop._stairs == {}
    assert policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1"
    assert policy.configuration()["floor_change_decision"] == "exploration fallback only"
def test_the_loop_reports_the_visit_order_and_the_next_node():
    policy, episode, world, rooms, _ = stair_policy(order=(1, STAIRS, 0))
    assert policy.loop.order == () and policy.loop.next_room is None
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop.order == (1, STAIRS, 0) and policy.loop.order_index == 0 and policy.loop.next_room == 1
    diagnostics = policy.loop.diagnostics()
    assert diagnostics["order"] == [1, STAIRS, 0] and diagnostics["next_room"] == 1
    assert diagnostics["stairs_offered"] == [STAIRS] and diagnostics["estimates"][str(STAIRS)]["kind"] == "stairs"
    configuration = policy.configuration()
    assert "RPT* over rooms and stair nodes" in configuration["floor_change_decision"]
    assert configuration["node_oracle"]["route"] == "LLM_REASONING_MODEL"
@pytest.mark.parametrize("kwargs", [{"stairs_as_nodes": 1}, {"entry_frontier": "no"}])
def test_loop_settings_reject_nonsense_flags(kwargs):
    with pytest.raises(ValueError):
        LoopSettings(**kwargs)
