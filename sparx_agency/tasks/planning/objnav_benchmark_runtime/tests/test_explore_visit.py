"""Regressions for the explore visit (2026-10-13): the frontier walk, the ``explored`` verdict, the folded openings.

Built on the two-room world of ``test_room_search_loop`` under the default
``visit="explore"``: room A (west) with an unexplored west end, room B (east)
with a small unknown patch by the door and an unexplored east end. The scene
graph is installed by hand, the frontier inventory is computed for real from
it, the oracle is the fixture's counter, the solver is fixed, and A* is real.
All seven complaints of the Darden recording ``ec8d17aeaaaa`` have a test
here or beside: the twelve-turn circle (``test_room_scans``, ``test_path_glances``,
``test_discovery``), the glance that finished nothing, the room walked to its
middle and turned in, the opening left standing after the camera looked
through it, the room and its own opening both in the order, and the camera
that never tilted (``test_gibson_protocol_tilt`` below).
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_registry import TrackedRoom
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    EXHAUSTED, SEARCH, TRANSIT, UNREACHABLE, RoomFacts)
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, GibsonProtocol
from sparx_agency.core.planning.object_search.rooms.opening_nodes import OpeningSettings
from sparx_agency.core.planning.object_search.rooms.room_scans import EXPLORED, SCAN_POINT_INSIDE
from sparx_agency.core.planning.object_search.rooms.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import openings_of, refresh, west_gap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, IN_B, NamingLLM, cell_in, loop_policy, obs_at, see, two_room_world)


def explore_policy(order=(1, 0), room_b_resolved=False, **overrides):
    """The default-visit fixture with a live frontier inventory (the sight ledger off, as in the opening tests)."""
    policy, episode, world, rooms, reasoned = loop_policy(order=order, visit="explore", room_b_resolved=room_b_resolved,
                                                          **overrides)
    policy.sight.settings = replace(policy.sight.settings, enabled=False)
    refresh(policy, world, IN_A)
    return policy, episode, world, rooms, reasoned


def walk(policy, episode, world, start_step, start_xy, max_actions=40):
    """Feed the loop its own frontier commands, teleporting the agent to each route's end and marking what the
    camera would have resolved there; return the commands until the room is released."""
    commands, xy, step = [], start_xy, start_step
    for _ in range(max_actions):
        obs = obs_at(episode, step, xy)
        policy.sight.observe(obs, world)                    # the policy records every pose before the loop runs
        command = policy.loop.plan(obs, world)
        commands.append(command)
        if policy.supervisor.state != SEARCH or policy.loop.room_id is None:
            return commands, step, xy
        policy.loop.charge()
        if command.waypoints:
            for point in command.waypoints:                 # the walk itself: every pose along the route is recorded
                policy.sight.observe(obs_at(episode, step, (float(point[0]), float(point[1]))), world)
            end = command.waypoints[-1]
            xy = (float(end[0]), float(end[1]))
            resolve_around(world, xy, 1.5)
            refresh(policy, world, xy)
        step += 1
    raise AssertionError("the visit never ended")


def resolve_around(world, xy, radius_m):
    """What a camera standing at ``xy`` would have mapped: the unknown within ``radius_m`` becomes free floor."""
    gx, gy = world.world_to_grid(*xy)
    r = int(radius_m / world.resolution)
    h, w = world.grid.shape
    patch = world.grid[max(0, gy - r):gy + r + 1, max(0, gx - r):gx + r + 1]
    patch[patch == world.values.unknown] = world.values.free


# -- the entry: the frontier that reveals the most, not the middle of the room ---------------------------
def test_the_default_visit_enters_a_room_at_its_most_revealing_frontier_not_its_vantage_point():
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    assert policy.loop.settings.visit == "explore" and policy.scans.explored
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    assert command.info["kind"] == "transit/1"
    goal = policy.route_memory.goal
    cost = policy.navigation_cost(world)
    vantage = policy.loop._vantage(world, cost, rooms[1])[0]
    assert goal != vantage and goal != rooms[1].centroid, "not the middle of the open floor"
    frontier = policy.loop._explore_goals(obs_at(episode, 0, IN_A), world, cost, rooms[1])
    assert frontier and tuple(frontier[0].xy) == tuple(goal), "the room's best frontier by view gain"
    assert frontier[0].utility >= max(g.utility for g in frontier)
    assert policy.loop.events[-1]["event"] == "transit" and policy.loop.events[-1]["entry"] == "frontier"
    assert policy.loop.stats["entry_frontier"] == 1 and policy.loop.estimates[1]["entry"] == "frontier"
    assert policy.loop.estimates[1]["frontier_clusters"] == 2


def test_a_room_with_no_frontier_left_is_entered_at_its_doorstep_not_its_vantage_point():
    """Since 2026-10-14: a room with nothing left to walk to is confirmed from its nearest passable cell -- the
    doorstep -- not from the middle of the room (the user's rule: no walk to the centre without a reason)."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), room_b_resolved=True)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.room_id == 1 and policy.loop.events[-1]["entry"] == "doorstep"
    cost = policy.navigation_cost(world)
    goal = policy.route_memory.goal
    vantage = policy.loop._vantage(world, cost, rooms[1])[0]
    assert goal == policy.loop._nearest_entry(obs_at(episode, 0, IN_A), world, cost, rooms[1])
    assert cell_in(world, rooms[1], goal), "inside room B's mask"
    assert math.dist(goal, IN_A) < math.dist(vantage, IN_A), "nearer the agent than the vantage point"
    assert policy.loop.estimates[1]["entry"] == "doorstep" and policy.loop.stats["entry_doorstep"] == 1


# -- the visit: frontier after frontier, then the room is explored ---------------------------------------
def test_the_visit_walks_the_rooms_frontiers_until_none_is_left_and_the_room_is_finished_explored():
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    entry = policy.route_memory.goal
    # Arrival: the agent's cell inside room B, at the entry frontier (which the arrival frame resolves).
    inside = (entry[0] - 0.3, entry[1]) if cell_in(world, rooms[1], (entry[0] - 0.3, entry[1])) else entry
    resolve_around(world, inside, 1.2)
    refresh(policy, world, inside)
    policy.sight.observe(obs_at(episode, 1, inside), world)
    first = policy.loop.plan(obs_at(episode, 1, inside), world)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    assert policy.loop._explore is not None and policy.loop._explore["room"] == 1
    assert first.info["kind"] == "frontier" and first.waypoints, "the visit walks to the next frontier, no circle"
    assert policy.loop.events[-2]["event"] == "explore_begin" and policy.loop.events[-1]["event"] == "explore_goal"
    policy.loop.charge()
    commands, last, where = walk(policy, episode, world, 2, inside)
    kinds = {c.info.get("kind") for c in commands[:-1]}
    assert kinds <= {"frontier"}, "every action of the visit is a walk toward a frontier: %s" % kinds
    assert not any(c.info.get("kind") == "room_scan" for c in commands), "no rotation at the vantage point"
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    complete = [e for e in policy.loop.events if e["event"] == "explore_complete"]
    assert len(complete) == 1 and complete[0]["frontier_left"] == 0 and complete[0]["goals"] >= 1
    assert policy.loop.stats["explore_visits"] == 1 and policy.loop.stats["explore_exhausted"] == 1
    # The loop point on the same action: room B is finished -- explored -- and not a node of the first pass.
    assert policy.loop._excluded == {1: "scanned:" + EXPLORED}
    assert policy.loop._finished[1][0] == EXPLORED
    assert policy.reasoned_with[-1]["scanned"][1][0].startswith("walked")
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0
    assert commands[-1].info["kind"] == "transit/0"
    assert policy.scans.records == [], "no rotation was turned, none is recorded"


def test_the_explored_verdict_needs_the_agent_inside_the_room_not_on_its_threshold():
    """The ledger's rule: no live frontier AND a pose well inside the room. A room only seen from its door is
    not explored, however completely its floor is mapped."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), room_b_resolved=True)
    ledger = policy.scans
    assert ledger.status(world, 1, rooms[1], frontier=0) is None, "nobody has stood in room B"
    policy.sight.observe(obs_at(episode, 1, (6.7, 3.0), yaw=math.pi), world)    # on the door's threshold, facing west
    assert ledger.status(world, 1, rooms[1], frontier=0) is None, "the doorway is not the room"
    policy.sight.observe(obs_at(episode, 2, IN_B), world)
    assert ledger.status(world, 1, rooms[1], frontier=1) is None, "a frontier left: not finished"
    assert ledger.status(world, 1, rooms[1], frontier=0) == EXPLORED
    assert ledger.status(world, 1, rooms[1]) == EXPLORED, "sticky"
    scan_mode = loop_policy(visit="scan")[0]
    assert not scan_mode.scans.explored, "under the scan visit the rotation is the finishing act"


def test_a_full_rotation_turned_on_the_way_finishes_the_room_and_ends_the_visit_at_once():
    """Darden ``ec8d17aeaaaa``, actions 61-106: a full glance inside R3 on the way to its vantage point, then
    the walk on and the same circle again. A rotation the ledger records inside the room in force ends it --
    under the deprecated exits-as-nodes mode, where the room's openings carry the search on as nodes of their
    own. Under room-level GAPs (the default) the rotation finishes the room in the ledger but the visit walks
    on to the GAPs left in it: see ``test_room_level_gaps``."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), openings=OpeningSettings(exits_as_nodes=True))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    entry = policy.route_memory.goal
    inside = (entry[0] - 0.3, entry[1]) if cell_in(world, rooms[1], (entry[0] - 0.3, entry[1])) else entry
    first = policy.loop.plan(obs_at(episode, 1, inside), world)
    assert policy.supervisor.state == SEARCH and first.info["kind"] == "frontier"
    policy.loop.charge()
    policy.scans.record(obs_at(episode, 3, IN_B), rooms[1], source="glance")      # a glance completed in room B
    command = policy.loop.plan(obs_at(episode, 4, IN_B), world)
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    finished = [e for e in policy.loop.events if e["event"] == "finished_by_rotation"]
    assert len(finished) == 1 and finished[0]["source"] == "glance" and finished[0]["rotation_step"] == 3
    assert policy.loop.stats["finished_by_rotation"] == 1
    assert policy.loop._excluded == {1: "scanned:" + SCAN_POINT_INSIDE}
    assert command.info["kind"] == "transit/0", "the loop point ran on the same action"
    # The same rule under the scan visit's approach phase.
    scan, episode_s, world_s, rooms_s, _ = loop_policy(order=(1, 0), visit="scan")
    scan.loop.plan(obs_at(episode_s, 0, IN_A), world_s)
    vantage = scan.route_memory.goal
    short = (vantage[0] - 1.5, vantage[1])
    scan.loop.plan(obs_at(episode_s, 1, short), world_s)
    assert scan.loop._scan["phase"] == "approach"
    scan.loop.charge()
    scan.scans.record(obs_at(episode_s, 2, short), rooms_s[1], source="glance")
    scan.loop.plan(obs_at(episode_s, 3, short), world_s)
    assert scan.supervisor.history[-1][:2] == (1, EXHAUSTED) and scan.loop.stats["finished_by_rotation"] == 1
    assert scan.loop.stats["scans_completed"] == 0, "the walk to the vantage point and its circle were never spent"


def test_a_visit_that_overruns_its_bound_is_released_budget_spent_and_not_finished():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), explore_visit_steps=3)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    entry = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, entry), world)
    assert policy.supervisor.state == SEARCH and policy.loop.visit_budget() == 3
    for _ in range(3):
        policy.loop.charge()
    policy.loop.plan(obs_at(episode, 2, entry), world)
    assert policy.supervisor.history[-1][:2] == (1, "budget_spent")
    assert 1 not in policy.loop._excluded, "a frontier is left: the room is not finished"


# -- openings: live every action, folded into the room whose walk covers them ---------------------------
# Under room-level GAPs (the default since 2026-10-21) EVERY exit is folded; the tests of the selective fold
# -- an exit off a room that is no node staying a node of its own -- run the deprecated ``exits_as_nodes``
# mode they describe. ``test_room_level_gaps`` covers the default.
LEGACY = dict(openings=OpeningSettings(exits_as_nodes=True))


def test_an_opening_off_a_room_that_is_a_node_is_folded_into_the_rooms_walk():
    """Darden ``ec8d17aeaaaa``, action 27: the order read R3 -> O3, a room and the gap at its own far end."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), **LEGACY)
    before = openings_of(policy, episode, world)
    assert len(before) == 3, "room A's west end; room B's patch by the door and its east end"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._folded_now == {o.node_id for o in before}, "both rooms are nodes: every opening is one of their frontiers"
    assert set(policy.loop._openings) == {o.node_id for o in before}, "... still on record, for the transit's lookups"
    assert policy.loop.stats["openings_folded"] == 3
    assert {e["room"] for e in policy.loop.events if e["event"] == "opening_folded"} == {0, 1}
    assert policy.reasoned_with[0]["stairs"] == [], "the oracle was shown the rooms, not the gaps"
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot
    shown = search_snapshot(policy)["openings"]
    assert {o["id"] for o in shown} == {o.node_id for o in before} and all(o["folded"] for o in shown), (
        "the HUD shows every live opening, the folded ones flagged -- never hidden (2026-10-13)")
    assert all(not o["targeted"] and o["live"] for o in shown)
    assert policy.loop.diagnostics()["openings_folded"] == sorted(policy.loop._folded_now)
    # A room the ledger has finished is no node, and its openings are nodes again.
    policy.scans.record(obs_at(episode, 1, IN_A), rooms[0], source="warmup")
    policy.graph.probs[0] = 0.0
    scan_mode, episode_s, world_s, rooms_s, _ = loop_policy(order=(1, 0), visit="scan")
    assert scan_mode.loop._fold_openings(obs_at(episode_s, 0, IN_A), before) == before, "the scan visit folds nothing"


def test_openings_of_a_room_the_oracle_valued_below_the_floor_stay_nodes():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), **LEGACY)
    gap = west_gap(openings_of(policy, episode, world))
    policy.graph.probs = {0: 0.0, 1: 0.6}                      # room A is not worth a visit; its gap leads on
    kept = policy.loop._fold_openings(obs_at(episode, 0, IN_A), [o for o in __import__(
        "sparx_agency.core.planning.object_search.rooms.opening_nodes", fromlist=["opening_options"]
    ).opening_options(policy, openings_of(policy, episode, world))])
    assert [o.node_id for o in kept] == [gap.node_id], "room B's openings fold into room B; room A's gap stays"


def test_an_opening_the_reply_itself_unfolded_is_a_node_at_the_provisional_value_not_zero():
    """Corozal/000142 (2026-10-14), action 446: the toilet's doorway, folded into its room at the call, unfolded
    by the reply that dropped the room to 1 %, read 0.0 for the 54 actions left."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), **LEGACY)
    gap = west_gap(openings_of(policy, episode, world))
    for pid in (0, 1):                                     # both rooms cooling: SELECT stays, as in the opening tests
        policy.supervisor.inner._cooling[pid] = 0.0
        policy.supervisor.inner._cooling_verdict[pid] = EXHAUSTED
    inner = policy.graph.reason

    def reason(*args, **kwargs):                           # the model's reply drops room A under the floor
        inner(*args, **kwargs)
        policy.graph.probs[0] = 0.0

    policy.graph.reason = reason
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert gap.node_id not in policy.reasoned_with[0]["stairs"], "folded at the call: room A was a node then"
    assert gap.node_id not in policy.loop._folded_now, "... and unfolded by the reply"
    assert gap.node_id not in policy.graph.stair_probs, "the model never saw it, so it has no number"
    estimate = policy.loop.estimates[gap.node_id]
    assert estimate["prob_source"] == "provisional" and estimate["prob"] == policy.loop.settings.openings.unvalued_prob == 0.10
    assert estimate["entry"] == "peek", "a node RPT* was handed, not one under the floor"
    assert estimate["why"] == "not yet shown to the model"
    assert policy.loop.stats["openings_provisional"] == 1
    [event] = [e for e in policy.loop.events if e["event"] == "opening_provisional"]
    assert event["node"] == gap.node_id and event["room"] == 0 and event["prob"] == 0.10
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot
    [shown] = [o for o in search_snapshot(policy)["openings"] if o["id"] == gap.node_id]
    assert shown["prob"] == 0.10 and shown["prob_source"] == "provisional" and not shown["folded"]
    # The node without a number still buys the next call, and the model's number then replaces the prior.
    policy.graph.reason = inner
    policy.loop.plan(obs_at(episode, 7, IN_A), world)
    assert len(policy.reasoned_with) == 2 and gap.node_id in policy.reasoned_with[1]["stairs"]
    assert policy.loop.estimates[gap.node_id]["prob_source"] == "oracle" and policy.loop.estimates[gap.node_id]["prob"] == 0.3
    assert policy.loop.stats["openings_provisional"] == 1, "logged once"
    # ... and is remembered across a later call that does not show it again.
    policy.graph.stair_probs = {}
    assert policy.loop._exit_prob(gap.node_id) == (0.3, "remembered")


def test_with_unvalued_prob_zero_an_unfolded_opening_reads_zero_as_before():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0),
                                                      openings=OpeningSettings(unvalued_prob=0.0, exits_as_nodes=True))
    gap = west_gap(openings_of(policy, episode, world))
    inner = policy.graph.reason

    def reason(*args, **kwargs):
        inner(*args, **kwargs)
        policy.graph.probs[0] = 0.0

    policy.graph.reason = reason
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop.estimates[gap.node_id]["prob"] == 0.0, "under the supervisor's floor: no node, as before"
    assert policy.loop.stats["openings_provisional"] == 0 and not any(
        e["event"] == "opening_provisional" for e in policy.loop.events)
    with pytest.raises(ValueError):
        OpeningSettings(unvalued_prob=1.0)


def test_the_openings_are_read_every_action_so_a_resolved_one_leaves_the_record_at_once():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    policy.graph.probs = {0: 0.0, 1: 0.6}
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    gap = west_gap(openings_of(policy, episode, world))
    assert gap.node_id in policy.loop._openings and policy.supervisor.state == TRANSIT
    # The west end gets mapped while the agent walks east: the opening is gone from the inventory ...
    resolve_around(world, gap.xy, 3.0)
    refresh(policy, world, IN_A)
    policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.state == TRANSIT, "mid-transit, no loop point"
    assert gap.node_id not in policy.loop._openings, "... and from the record on this very action"


# -- the camera tilts (item 7) ---------------------------------------------------------------------------
def test_gibson_protocol_tilt():
    """The published baselines' four actions plus Habitat's LOOK pair, bounded, as a declared protocol change."""
    spec = PROTOCOL.actions()
    assert spec.has_camera_tilt and DiscreteAction.LOOK_DOWN in spec.actions and DiscreteAction.LOOK_UP in spec.actions
    assert spec.tilt_angle_deg == 30.0 and spec.min_pitch_deg == -30.0 and spec.max_pitch_deg == 60.0
    assert PROTOCOL.protocol_id.endswith("/2") and PROTOCOL.camera_tilt
    body = GibsonProtocol(camera_tilt=False).actions()
    assert not body.has_camera_tilt and len(body.actions) == 4
    assert PROTOCOL.max_steps == 500 and PROTOCOL.turn_angle_deg == 30.0 and PROTOCOL.hfov_deg == 79.0, "nothing else moved"
    assert math.isclose(spec.turn_angle_rad, math.radians(30.0))


def test_a_rooms_service_time_is_a_peeks_worth_per_frontier_so_a_sliver_is_priced_like_the_peek_it_replaces():
    """The re-run folded the bathroom door into a one-frontier sliver room and then charged it a full room's
    visit (12), which put it behind every opening of equal probability; it is priced by its frontiers now."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    obs = obs_at(episode, 0, IN_A)
    settings = policy.loop.settings
    assert policy.loop._live_frontier(1, obs, world) == 2 and policy.loop._live_frontier(0, obs, world) == 1
    assert policy.loop._service_steps_for(1, obs, world) == settings.openings.service_steps + 2 * settings.explore_frontier_steps
    assert policy.loop._service_steps_for(0, obs, world) == settings.openings.service_steps + settings.explore_frontier_steps
    policy.loop._finished[0] = ("explored", 3)
    assert policy.loop._service_steps_for(0, obs, world) == settings.scan_approach_steps // 2 + policy.rotation_turns(), (
        "a second-pass revisit is a scan and is priced as one")
    del policy.loop._finished[0]
    policy.loop.plan(obs, world)
    charged = policy.loop._service_last
    assert charged[1] == pytest.approx(12.0) and charged[0] == pytest.approx(8.0), "what RPT* was handed per room"
    assert policy.loop.estimates[1]["distance_m"] == pytest.approx(
        policy.loop.estimates[1]["distance_m"]) and policy.loop.estimates[1]["distance_m"] > 0, "the leg is backed out of the charge"
    scan_mode = loop_policy(visit="scan")[0]
    assert scan_mode.loop._service_steps_for(1) == scan_mode.loop.settings.service_steps()


# -- a refused plan is re-aimed, not a room released -------------------------------------------------------
def test_a_plan_the_planner_refuses_re_aims_at_the_rooms_next_entry_instead_of_releasing_it_unreachable():
    """Darden/000034 (``4ae716bacfab``, 2026-10-13): the bathroom was released ``unreachable`` twice on one refused
    plan each, the second time 2.4 m from it after a 65-action walk; the toilet stood 1.3 m away at the step limit."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    obs = obs_at(episode, 0, IN_A)
    policy.loop.plan(obs, world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    committed = tuple(policy.route_memory.goal)
    navigate = policy._navigate

    def refusing(observation, planning_world, goal, kind, final_yaw=None):
        if tuple(goal) == committed:
            policy.route_memory.clear("test refusal")
            policy._route = policy._goal = None
            return None
        return navigate(observation, planning_world, goal, kind, final_yaw)

    policy._navigate = refusing
    policy.loop.charge()
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1, "the room is kept"
    assert command.waypoints and command.info["kind"] == "transit/1"
    assert tuple(policy.route_memory.goal) != committed, "a different entry of the same room"
    assert policy.loop.stats["route_reaimed"] == 1 and policy.loop.stats["plan_failures"] == 0
    event = [e for e in policy.loop.events if e["event"] == "route_reaimed"][-1]
    assert event["room"] == 1 and tuple(event["failed"]) == tuple(round(v, 2) for v in committed)
    assert policy.loop._entry[0] == tuple(policy.route_memory.goal)
    # Every entry refused: only then is the room released unreachable.
    policy._navigate = lambda *args, **kwargs: None
    policy.loop.charge()
    policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert policy.loop.stats["plan_failures"] >= 1, "the room, then whatever the re-solved order put next"
    assert [e for e in policy.loop.events if e["event"] == "release"][-1]["verdict"] == "unreachable"


# -- 2026-10-14: a room gone from the registry for one action does not end its transit -----------------------
def test_a_room_missing_from_the_registry_for_an_action_keeps_its_transit_walking():
    """Collierville/000023, action 170: the watershed re-segmented and room 4 was gone from the registry for exactly
    one action -- the transit was released ``unreachable`` and the order re-solved without it; the room was back the
    next action. The committed entry point is still a point on the floor: the walk goes on for ``room_missing_grace``
    actions, and the room is given up only if it stays missing."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    goal = policy.route_memory.goal
    missing = policy.graph.registry.rooms.pop(1)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1, "one action gone: the walk goes on"
    assert command.waypoints and policy.route_memory.goal == goal and command.info["kind"] == "transit/1"
    assert policy.loop.stats["room_missing_held"] == 1
    assert [e["event"] for e in policy.loop.events if e["event"] == "room_missing_held"] == ["room_missing_held"]
    policy.graph.registry.rooms[1] = missing                        # back the next action, as in the recording
    command = policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1" and policy.loop._room_missing == 0
    # Gone for longer than the grace: unreachable, as before.
    policy.graph.registry.rooms.pop(1)
    for step in range(3, 3 + policy.loop.settings.room_missing_grace):
        policy.loop.plan(obs_at(episode, step, IN_A), world)
        assert policy.supervisor.room_id == 1
    policy.loop.plan(obs_at(episode, 3 + policy.loop.settings.room_missing_grace, IN_A), world)
    assert policy.supervisor.history[-1][:2] == (1, UNREACHABLE)
    with pytest.raises(ValueError):
        LoopSettings(room_missing_grace=-1)


# -- 2026-10-22: the committed-target rule -- a target set is a target reached -------------------------------
def renumber(policy, world, old, new):
    """The watershed hands a room a new number: same floor, same mask, a fresh pid; the label image and the
    per-room facts follow, as the scene graph's background refresh would make them."""
    room = policy.graph.registry.rooms.pop(old)
    policy.graph.registry.rooms[new] = TrackedRoom(id=new, mask=room.mask, n_cells=room.n_cells, centroid=room.centroid)
    policy.graph.labels = np.where(policy.graph.labels == old + 1, new + 1, policy.graph.labels).astype(np.int32)
    fact = policy.graph.facts.pop(old)
    policy.graph.facts[new] = RoomFacts(new, fact.frontier_clusters, fact.time_in_room_s, fact.cells)
    policy.graph.options = [replace(o, room_id=new) if o.room_id == old else o for o in policy.graph.options]
    return policy.graph.registry.rooms[new]


def arrive(policy, episode, world, rooms, pid, step):
    """Stand the agent inside room ``pid`` at the committed entry (the frame there resolves it) and tick the loop."""
    entry = policy.route_memory.goal
    inside = (entry[0] - 0.3, entry[1]) if cell_in(world, rooms[pid], (entry[0] - 0.3, entry[1])) else entry
    resolve_around(world, inside, 1.2)
    refresh(policy, world, inside)
    policy.sight.observe(obs_at(episode, step, inside), world)
    return policy.loop.plan(obs_at(episode, step, inside), world), inside


def test_a_room_renumbered_in_transit_keeps_its_turn_under_the_new_number_without_a_loop_point():
    """Corozal/000003 (``c1af8954a378``, 2026-10-22): eleven of eighteen selections were ``unreachable`` releases of
    a room the watershed had merely renumbered on the way to it, each buying a room-LLM call (10 s) and an RPT*
    solve, and at action 137 the agent re-planned to another room while committed. The turn follows the place."""
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    goal = policy.route_memory.goal
    prob = policy.graph.probs[1]
    renumber(policy, world, 1, 7)
    refresh(policy, world, IN_A)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 7, "the same turn, under the new number"
    assert command.info["kind"] == "transit/7" and policy.route_memory.goal == goal, "the same walk to the same place"
    assert reasoned == [0] and policy.supervisor.history == [], "no release, no loop point, no LLM call"
    assert policy.loop.stats["rooms_rebound"] == 1 and policy.loop.stats["room_missing_held"] == 0
    [event] = [e for e in policy.loop.events if e["event"] == "room_rebound"]
    assert event["room"] == 1 and event["to"] == 7 and event["state"] == TRANSIT
    assert policy.loop.next_room == 7 and policy.loop.order == (7, 0) and policy.loop.estimates[7]["room"] == 7
    assert policy.graph.probs[7] == prob and policy.loop._entries[7] == (goal, True)
    # Arrival is arrival at room 7; the visit begins there with the counter reset.
    arrive(policy, episode, world, rooms | {7: policy.graph.registry.rooms[7]}, 7, 2)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 7 and policy.loop.local_steps == 0
    assert policy.supervisor.stats["arrivals"] == 1 and policy.supervisor.history == []


def test_a_room_renumbered_under_the_agents_feet_keeps_the_visit_running():
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    _, inside = arrive(policy, episode, world, rooms, 1, 1)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    policy.loop.charge()
    renumber(policy, world, 1, 9)
    refresh(policy, world, inside)
    policy.loop.plan(obs_at(episode, 2, inside), world)
    assert policy.supervisor.state == SEARCH and policy.supervisor.room_id == 9 and policy.loop.room_id == 9
    assert policy.loop.local_steps == 1 and policy.loop._explore["room"] == 9, "the visit and its counter go on"
    assert policy.supervisor.history == [] and reasoned == [0]


def test_the_rebinding_is_the_committed_rule_and_off_with_it():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), transit_commitment=False)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    renumber(policy, world, 1, 7)
    refresh(policy, world, IN_A)
    for step in range(1, 2 + policy.loop.settings.room_missing_grace):
        policy.loop.plan(obs_at(episode, step, IN_A), world)
    assert policy.supervisor.history[-1][:2] == (1, UNREACHABLE), "the historical release"
    assert policy.loop.stats["rooms_rebound"] == 0
    for kwargs in ({"transit_commitment": 1}, {"rebind_reach_m": -1.0}, {"entry_failure_limit": -1},
                   {"entry_failure_radius_m": 0.0}, {"entry_failure_backoff": 0.5}):
        with pytest.raises(ValueError):
            LoopSettings(**kwargs)


def test_a_new_kind_of_object_in_the_room_in_transit_is_asked_about_at_arrival_not_on_the_way():
    """The user's rule of 2026-10-22: a new name for the target room on the way is not a reason to re-plan, nor
    to spend a classifier call; the arrival asks, and the loop point after the visit values the answer."""
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0), llm=llm)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    see(policy, {1: ["sink"]}, step=1)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert llm.calls == 0 and policy.loop.stats["relabels_deferred"] == 1 and policy.loop.stats["relabels"] == 0
    assert policy.supervisor.state == TRANSIT and command.info["kind"] == "transit/1" and reasoned == [0]
    assert policy.supervisor.history == [], "no reclassified release on the way"
    arrive(policy, episode, world, rooms, 1, 2)
    assert policy.supervisor.state == SEARCH and llm.calls == 1, "the arrival asks"
    assert policy.graph.label_info(1)["label"] == "bathroom"
    assert [e for e in policy.loop.events if e["event"] == "relabel"][-1]["where"] == "search"


def test_a_room_finished_from_outside_on_the_way_is_still_walked_to_under_the_committed_rule():
    """The mirror of ``test_room_level_gaps``' release in transit: with the rule on, the walk is completed and the
    room's verdict is read at arrival by the visit's own rules."""
    policy, episode, world, rooms, _ = explore_policy(order=(0, 1))
    policy.scans.record(obs_at(episode, 0, IN_A), rooms[0], source="warmup")
    policy.loop.plan(obs_at(episode, 0, IN_B), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0 and policy.loop._reopened == {0}
    gap = west_gap(openings_of(policy, episode, world))
    resolve_around(world, gap.xy, 3.0)
    refresh(policy, world, IN_B)
    policy.loop.plan(obs_at(episode, 1, IN_B), world)
    assert policy.loop.stats["finished_in_transit"] == 0
    assert not any(e["event"] == "finished_in_transit" for e in policy.loop.events)
    # What DOES end this walk is the committed frontier itself resolving: the look through the door mapped the
    # west end, the room has no significant frontier left to re-aim at, and there is nothing left to walk to.
    assert policy.loop.stats["skipped_in_transit"] == 1 and policy.supervisor.history[-1][:2] == (0, EXHAUSTED)


# -- 2026-10-22: a place the agent keeps failing to reach is given up ----------------------------------------
def test_an_unreachable_release_is_remembered_by_place_and_a_repeat_withholds_the_room(monkeypatch):
    """A planner that refuses every route: room B is released ``unreachable`` at its frontier, then at its doorstep
    -- two failed turns at places of one room withhold the room (``unreachable:2``) and the order goes to room A.
    The memory is by position, not by room id, so the partition renumbering the room does not reset it."""
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    monkeypatch.setattr(policy.planner, "plan", lambda request, world_: type("R", (), {"ok": False, "path": None})())
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.history[0][:2] == (1, UNREACHABLE), "every way into room B refused: released"
    failures = [e for e in policy.loop.events if e["event"] == "entry_failure" and e["room"] == 1]
    assert [f["failures"] for f in failures] == [1, 1] and not any(f["withheld"] for f in failures)
    assert failures[0]["place"] != failures[1]["place"], "the frontier, then the doorstep: two places of one room"
    assert all(f["verdict"] == UNREACHABLE for f in failures)
    assert policy.loop._excluded.get(1) == "unreachable:2" and policy.loop.stats["excluded_unreachable"] == 1
    assert policy.supervisor.room_id == 0 or policy.supervisor.history[-1][0] == 0, "the order went elsewhere"
    assert policy.loop.estimates[1]["entry"] == "excluded" and policy.loop.estimates[1]["excluded"] == "unreachable:2"
    [event] = [e for e in policy.loop.events if e["event"] == "excluded" and "1" in e["rooms"]]
    assert event["rooms"]["1"] == "unreachable:2"
    assert policy.loop._room_unreachable(world, 1, 0) == "unreachable:2" and policy.loop._room_unreachable(world, 0, 0) is None
    # The back-off: ``failure_cooldown_steps * backoff ** (n - limit)`` actions after the last failure the room is open again.
    cooldown = policy.loop.settings.failure_cooldown_steps
    assert policy.loop._room_unreachable(world, 1, cooldown - 1) == "unreachable:2"
    assert policy.loop._room_unreachable(world, 1, cooldown) is None
    # The same by place: a spot failed twice is passed over for the room's next entry, and withheld when it is the last.
    place = tuple(failures[1]["place"])
    policy.loop._entry_failures = [(place, 2, 0)]
    assert policy.loop._entry_failed(place, cooldown - 1) == "unreachable:2" and policy.loop._entry_failed(place, cooldown) is None
    policy.loop._entry_failures = [(place, 3, 0)]
    assert policy.loop._entry_failed(place, 2 * cooldown - 1) == "unreachable:3", "a third failure doubles the wait"
    assert policy.loop._entry_failed(place, 2 * cooldown) is None
    assert policy.loop._entry_failed((place[0] + 5.0, place[1]), 1) is None, "another place is another place"
    assert policy.loop.diagnostics()["entry_failures"][0]["failures"] == 3
    # Off: nothing is remembered.
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), entry_failure_limit=0)
    monkeypatch.setattr(policy.planner, "plan", lambda request, world_: type("R", (), {"ok": False, "path": None})())
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._entry_failures == [] and not any(e["event"] == "entry_failure" for e in policy.loop.events)


# -- 2026-10-22: the scene graph is not rebuilt on every action of a committed transit ----------------------
def test_the_geometry_is_rebuilt_every_transit_period_while_a_committed_route_is_walked_and_at_arrival():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    assert policy.settings.transit_graph_period_steps == 5 and policy.graph_period() == 1, "idle: every action"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.refreshed == [0], "the loop point rebuilt it on its own action"
    assert policy.supervisor.state == TRANSIT and policy.route_memory.kind == "transit/1"
    assert policy.graph_period() == 5, "a committed transit route is being walked"
    accessibility = []
    policy.graph.refresh_accessibility = lambda *a, **k: accessibility.append(a[2])
    for step in range(1, 5):
        policy._refresh_graph(obs_at(episode, step, IN_A), world)
    assert policy.refreshed == [0] and accessibility == [], "four actions: neither the partition nor the inventory"
    policy._refresh_graph(obs_at(episode, 5, IN_A), world)
    assert policy.refreshed == [0, 5], "the fifth rebuilds"
    policy._refresh_graph(obs_at(episode, 6, IN_A), world, force=True)
    assert policy.refreshed == [0, 5, 6], "force rebuilds whatever the cadence"
    # Arrival forces the rebuild on its own action, before the visit's first decision.
    del policy.graph.refresh_accessibility
    arrive(policy, episode, world, rooms, 1, 7)
    assert policy.supervisor.state == SEARCH and policy.refreshed[-1] == 7
    assert policy.graph_period() == 1, "a visit reads the partition every action"
    configuration = policy.configuration()
    assert "every 5 while a committed transit route is walked" in configuration["reasoning_cadence"]
    assert configuration["adaptation"]["transit_graph_period_steps"] == 5


# -- 2026-10-22: the explore visit ranks a room's frontiers by where the agent has barely been ----------------
def test_the_explore_goals_discount_a_frontier_the_agent_has_spent_many_actions_beside():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    obs = obs_at(episode, 0, IN_A)
    cost = policy.navigation_cost(world)
    goals = policy.loop._explore_goals(obs, world, cost, rooms[1])
    assert len(goals) == 2, "room B: the patch by the door and the east end"
    first = goals[0]
    # Thirty actions spent standing beside the best frontier (a long look-around there): the other comes first now.
    for step in range(1, 31):
        policy.sight.observe(obs_at(episode, step, first.xy, yaw=step * 0.2), world)
    policy.loop._visits_cache = None
    visits = policy.loop._visits(obs_at(episode, 31, IN_A))
    assert visits is not None and visits(first.xy) == 30
    ranked = policy.loop._explore_goals(obs_at(episode, 31, IN_A), world, cost, rooms[1])
    assert ranked[0].xy != first.xy and ranked[-1].xy == first.xy, "the trodden frontier drops behind the untrodden one"
    assert policy.sweep.settings.ranking.visit_weight == 0.05 and policy.sweep.settings.ranking.visit_radius_m == 1.5
    policy.sweep.settings = replace(policy.sweep.settings, ranking=replace(policy.sweep.settings.ranking, visit_weight=0.0))
    policy.loop._visits_cache = None
    assert policy.loop._visits(obs_at(episode, 31, IN_A)) is None, "0 disables"
