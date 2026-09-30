"""Ground-truth floor transitions: the stairs the simulator knows, the decision to take them, the climb.

The recording that motivated this (``runs/room_search_loop_ranchester_20260928T093541Z_transitfix``)
started an observed "stair traversal" twice where there were no stairs -- at
a 12 cm height wobble on the real stair head, and on a 13 cm raised bathroom
floor -- and spent 120 and 12 actions turning in place "acquiring tread
support". With the connectors read from the navmesh, a transition begins only
ON a known staircase, the up/down choice is explicit and recorded, and the
climb follows the connector's own polyline.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_classifier import RoomLabel
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, MultiFloorParams, ObservedFloor
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_decision import (
    DecisionSettings, approach_points, decide_floor_change, floor_at, floor_worth_visiting, stair_cost_m)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_nodes import (
    STAIR_NODE_BASE, is_stair_node, portal_id_of, search_context, stair_node_id, stair_options, summarise_floor)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy

RES = 0.1
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)

#: One straight flight, ENU: bottom anchor at (5, 3, 0), top anchor at (8, 3, 2.7).
FLIGHT = [[5.0, 3.0, 0.0], [6.0, 3.0, 0.9], [7.0, 3.0, 1.8], [8.0, 3.0, 2.7]]
STRUCTURE = {"stair_source": "navmesh",
             "floor_levels": [{"height_m": 0.0, "area_m2": 40.0}, {"height_m": 2.7, "area_m2": 40.0}],
             "stair_connectors": [{"id": 0, "bottom_xyz": FLIGHT[0], "top_xyz": FLIGHT[-1], "bottom_z": 0.0,
                                   "top_z": 2.7, "polyline_xyz": FLIGHT, "length_m": 4.0, "rise_m": 2.7}]}


def hall(unknown_east_of=None):
    """A 12 m x 6 m free hall, optionally unknown beyond ``unknown_east_of`` metres."""
    g = np.zeros((60, 120), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    if unknown_east_of is not None:
        g[1:-1, int(unknown_east_of / RES):-1] = -1
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def gt_policy(**overrides):
    return setup_policy("bed", **dict({"map_size_m": 40.0, "metadata": STRUCTURE}, **overrides))


def close(points, expected):
    """Elementwise closeness of two point lists (``pytest.approx`` refuses nested sequences)."""
    return np.allclose(np.asarray([list(p) for p in points], dtype=float), np.asarray(expected, dtype=float), atol=1e-6)


def at(episode, step, x, y, z=0.0, yaw=0.0):
    return replace(observation(episode, step, depth=3), pose=AgentPose(x, y, z, yaw))


def cost_for(policy, world):
    return assemble_cost_grid(policy.planner.fields_for(world), policy.planner_params, policy.settings.body_radius_m)[0]


# -- the decision --------------------------------------------------------------
def portal(pid, entry, direction, destination_z, cooldown_until=0):
    return {"id": pid, "entry": list(entry), "direction": direction, "destination_z": destination_z,
            "cooldown_until": cooldown_until, "connector_id": pid, "floor_id": 0}


def test_floor_worth_visiting_reads_the_saved_search_not_the_visit_count():
    assert floor_worth_visiting(None), "never stood on: worth it"
    assert floor_worth_visiting({"_floor_time": 0.0, "graph": SimpleNamespace(facts={})})
    swept = SimpleNamespace(facts={0: RoomFacts(0, 0, 40.0, 500), 1: RoomFacts(1, 0, 20.0, 300)})
    assert not floor_worth_visiting({"_floor_time": 60.0, "graph": swept})
    left = SimpleNamespace(facts={0: RoomFacts(0, 0, 40.0, 500), 1: RoomFacts(1, 2, 20.0, 300)})
    assert floor_worth_visiting({"_floor_time": 60.0, "graph": left})


def test_the_decision_prefers_an_unvisited_storey_defers_a_swept_one_and_rechecks_an_unmapped_entry():
    policy, episode = gt_policy()
    world = hall(unknown_east_of=9.0)
    cost = cost_for(policy, world)
    atlas = FloorAtlas(MultiFloorParams())
    atlas.floors = {0: ObservedFloor(0, 0.0), 1: ObservedFloor(1, -2.7)}
    swept = {"_floor_time": 90.0, "graph": SimpleNamespace(facts={0: RoomFacts(0, 0, 90.0, 500)})}
    portals = [portal(0, (5.0, 3.0, 0.0), +1, 2.7),          # up, unvisited, on the map
               portal(1, (10.5, 3.0, 0.0), +1, 2.7),         # up, but its entry is still unknown
               portal(2, (2.0, 3.0, 0.0), -1, -2.7)]         # down to a storey already searched out
    obs = at(episode, 50, 3.0, 3.0)
    choice, record = decide_floor_change(portals, obs, world, cost, atlas, {1: swept}, 50, 0.3, 100)
    assert choice is not None and choice.portal is portals[0]
    assert choice.direction == 1 and not choice.destination_visited
    assert choice.approach_xy == pytest.approx((5.0, 3.0), abs=RES)
    assert 1.9 < choice.distance_m < 2.5
    verdicts = {row["portal_id"]: row["verdict"] for row in record["candidates"]}
    assert verdicts[0] == "candidate" and "re-check" in verdicts[1] and "searched out" in verdicts[2]
    assert portals[1]["cooldown_until"] == 50 + DecisionSettings().recheck_actions
    assert portals[2]["cooldown_until"] == 150
    assert record["chosen"] == 0 and record["direction"] == "up" and "unvisited" in record["reason"]


def test_a_visited_storey_with_frontier_left_is_taken_when_nothing_unvisited_remains():
    policy, episode = gt_policy()
    world, cost = hall(), None
    cost = cost_for(policy, world)
    atlas = FloorAtlas(MultiFloorParams())
    atlas.floors = {0: ObservedFloor(0, 0.0), 1: ObservedFloor(1, -2.7)}
    unfinished = {"_floor_time": 30.0, "graph": SimpleNamespace(facts={0: RoomFacts(0, 3, 30.0, 500)})}
    portals = [portal(2, (2.0, 3.0, 0.0), -1, -2.7)]
    choice, record = decide_floor_change(portals, at(episode, 10, 3.0, 3.0), world, cost, atlas, {1: unfinished}, 10, 0.3, 100)
    assert choice is not None and choice.direction == -1 and choice.destination_visited
    assert record["direction"] == "down" and "visited-with-frontier" in record["reason"]
    assert floor_at(atlas, -2.7, 0.3) == 1 and floor_at(atlas, 2.7, 0.3) is None


def test_no_eligible_connector_or_an_off_map_agent_is_a_recorded_no_not_a_crash():
    policy, episode = gt_policy()
    world = hall()
    cost = cost_for(policy, world)
    atlas = FloorAtlas(MultiFloorParams())
    atlas.floors = {0: ObservedFloor(0, 0.0)}
    cooling = [portal(0, (5.0, 3.0, 0.0), +1, 2.7, cooldown_until=999)]
    choice, record = decide_floor_change(cooling, at(episode, 10, 3.0, 3.0), world, cost, atlas, {}, 10, 0.3, 100)
    assert choice is None and record["reason"] == "no eligible connector on this floor"
    boxed = hall()
    boxed.grid[:, :] = 100
    choice, record = decide_floor_change([portal(0, (5.0, 3.0, 0.0), +1, 2.7)], at(episode, 10, 3.0, 3.0), boxed,
                                         cost_for(policy, boxed), atlas, {}, 10, 0.3, 100)
    assert choice is None and "off the observed passable map" in record["reason"]


def test_the_fallback_rule_charges_the_climb_and_prefers_the_unvisited_storey():
    """The explicit rule the exploration fallback asks: worth per metre, the flight and the storey change charged."""
    policy, episode = gt_policy()
    world = hall()
    cost = cost_for(policy, world)
    atlas = FloorAtlas(MultiFloorParams())
    atlas.floors = {0: ObservedFloor(0, 0.0), 1: ObservedFloor(1, -2.7)}
    unfinished = {"_floor_time": 30.0, "graph": SimpleNamespace(facts={0: RoomFacts(0, 3, 30.0, 500)})}
    up = dict(portal(0, (5.0, 3.0, 0.0), +1, 2.7), path=FLIGHT)
    down = dict(portal(1, (5.0, 2.0, 0.0), -1, -2.7), path=[[5.0, 2.0, 0.0], [6.0, 2.0, -0.9], [7.0, 2.0, -1.8], [8.0, 2.0, -2.7]])
    obs = at(episode, 10, 3.0, 3.0)
    choice, record = decide_floor_change([up, down], obs, world, cost, atlas, {1: unfinished}, 10, 0.3, 100,
                                         floor_change_cost_m=8.0)
    assert choice.portal is up and choice.direction == 1, "unvisited (worth 1.0) beats visited-with-frontier (0.5)"
    verdicts = {row["portal_id"]: row for row in record["candidates"]}
    assert verdicts[0]["cost_m"] == pytest.approx(verdicts[0]["distance_m"] + 3 * math.hypot(1.0, 0.9) + 8.0, abs=0.05), (
        "cost = approach + flight length + the fixed cost of a storey change")
    assert verdicts[0]["score"] == pytest.approx(1.0 / verdicts[0]["cost_m"], rel=1e-3)
    assert verdicts[1]["score"] == pytest.approx(0.5 / verdicts[1]["cost_m"], rel=1e-3)
    assert record["rule"] == "fallback" and record["score"] == verdicts[0]["score"]
    assert stair_cost_m(up, 8.0) == pytest.approx(3 * math.hypot(1.0, 0.9) + 8.0)
    assert stair_cost_m(portal(2, (1.0, 1.0, 0.0), +1, 2.7), 8.0) == 8.0, "no polyline: the fixed cost alone"
def test_a_dearer_connector_loses_at_equal_worth():
    policy, episode = gt_policy()
    world = hall()
    cost = cost_for(policy, world)
    atlas = FloorAtlas(MultiFloorParams())
    atlas.floors = {0: ObservedFloor(0, 0.0)}
    near = dict(portal(0, (4.0, 3.0, 0.0), +1, 2.7), path=FLIGHT)
    far = dict(portal(1, (10.0, 3.0, 0.0), +1, 2.7), path=[[10.0, 3.0, 0.0], [11.0, 3.0, 2.7]])
    choice, record = decide_floor_change([near, far], at(episode, 10, 3.0, 3.0), world, cost, atlas, {}, 10, 0.3, 100)
    assert choice.portal is near and record["chosen"] == 0
def test_approach_points_snap_every_reachable_entry_onto_the_observed_map_in_one_sweep():
    policy, episode = gt_policy()
    world = hall(unknown_east_of=9.0)
    cost = cost_for(policy, world)
    portals = [portal(0, (5.0, 3.0, 0.0), +1, 2.7), portal(1, (10.5, 3.0, 0.0), +1, 2.7)]
    approaches = approach_points(portals, at(episode, 0, 3.0, 3.0), world, cost)
    assert set(approaches) == {0}, "the entry in unseen space is absent, not given a made-up distance"
    assert approaches[0].xy == pytest.approx((5.0, 3.0), abs=RES) and 1.9 < approaches[0].distance_m < 2.5
    boxed = hall()
    boxed.grid[:, :] = 100
    assert approach_points(portals, at(episode, 0, 3.0, 3.0), boxed, cost_for(policy, boxed)) is None


def corridor_hall():
    """The hall with a stair-head corridor: a wall at x=8.0 from y=0.6 up, its mouth at y<0.6.

    The entry (9.0, 3.0) sits in the corridor east of the wall and is not on
    the observed map (unknown from y=1.5 up). The Euclidean-nearest passable
    cell to it is in the room WEST of the wall at (7.9, 3.0), 1.1 m away; the
    nearest cell in the corridor is its mouth at (9.0, 1.4), 1.6 m away.
    """
    world = hall()
    world.grid[6:, 80] = 100                         # the wall, x = 8.0 m, from y = 0.6 m to the top
    world.grid[15:-1, 81:-1] = -1                    # the corridor beyond y = 1.5 m is unseen
    return world


def test_the_approach_point_must_see_the_entry_not_merely_be_near_it():
    """The nearest passable cell to a stair head is as likely behind the corridor's wall as in the corridor.

    In the first recorded campaign it was, by one centimetre, and the climb
    was led straight into the wall. A candidate must have a clear line to the
    entry through what the map knows: unknown is allowed, occupied is not.
    """
    policy, episode = gt_policy()
    world = corridor_hall()
    cost = cost_for(policy, world)
    entry = portal(0, (9.0, 3.0, 0.0), -1, -2.7)
    approaches = approach_points([entry], at(episode, 0, 3.0, 3.0), world, cost, DecisionSettings(approach_snap_m=2.0))
    assert 0 in approaches
    x, y = approaches[0].xy
    assert x > 8.0 and 1.0 < y < 1.5, "in the corridor, at its mouth -- not in the room behind the wall: %r" % (approaches[0].xy,)
    walled = corridor_hall()
    walled.grid[1:-1, 81:-1] = 100                   # the corridor is all wall as far as the map knows
    assert approach_points([entry], at(episode, 0, 3.0, 3.0), walled, cost_for(policy, walled), DecisionSettings(approach_snap_m=2.0)) == {}, (
        "no cell sees the entry: the map has not reached it, rather than a made-up point behind a wall")


def test_an_approach_that_ends_short_of_the_entry_is_extended_as_the_map_grows(monkeypatch):
    """Arriving 1.6 m from the entry is not the foot of the flight: re-snap on the map seen from here and go on."""
    policy, episode = gt_policy()
    world = corridor_hall()
    policy.plan(at(episode, 0, 3.0, 3.0))
    building = policy.building
    building.active = building._portal_for(at(episode, 0, 3.0, 3.0), building.ground_truth.connectors[0], 0.0)
    building.active.update(entry=[9.0, 3.0, 0.0], approach_xy=[9.0, 1.4])
    routes = []

    def navigate(obs, w, goal, kind, final_yaw=None):
        routes.append(goal)
        policy.route_memory.path, policy.route_memory.goal, policy.route_memory.kind = [(obs.pose.x, obs.pose.y), goal], tuple(goal), kind
        return NavigationCommandStub(goal)

    monkeypatch.setattr(policy, "_navigate", navigate)
    monkeypatch.setattr(policy.route_memory, "arrived", lambda obs: math.dist((obs.pose.x, obs.pose.y), policy.route_memory.goal) < 0.3)
    building.plan(at(episode, 1, 3.0, 3.0), world)
    assert routes == [(9.0, 1.4)] and building.transition is None
    world.grid[15:45, 81:-1] = 0                     # standing at the corridor's mouth the agent sees up it
    command = building.plan(at(episode, 2, 9.0, 1.4), world)
    assert building.transition is None, "the map now reaches nearer the entry: the approach goes on"
    assert len(routes) == 2 and routes[-1][1] > 2.4 and abs(routes[-1][0] - 9.0) < 0.3
    assert [e["event"] for e in building.events if e["event"] == "approach_extended"] == ["approach_extended"]
    building.plan(at(episode, 3, routes[-1][0], routes[-1][1]), world)
    assert building.transition is not None and building.phase == "TRAVERSE", "within reach of the entry: the climb begins"


class NavigationCommandStub:
    def __init__(self, goal):
        self.goal, self.waypoints, self.stop, self.info = goal, ((goal[0], goal[1]),), False, {}
# -- the stairs as nodes of the room-search loop ------------------------------------
def test_stair_node_ids_live_far_above_every_room_pid():
    assert stair_node_id(0) == STAIR_NODE_BASE and stair_node_id(3) == STAIR_NODE_BASE + 3
    assert is_stair_node(stair_node_id(2)) and not is_stair_node(2) and not is_stair_node(None)
    assert portal_id_of(stair_node_id(7)) == 7
def test_summarise_floor_reads_a_saved_context_in_the_prompts_words():
    assert summarise_floor(None) == "unvisited"
    labels = {0: RoomLabel("kitchen", 0.9, ""), 1: RoomLabel("unknown", 0.0, ""), 2: RoomLabel("living_room", 0.8, "")}
    graph = SimpleNamespace(registry=SimpleNamespace(rooms={0: 1, 1: 1, 2: 1, 3: 1}),
                            label_tracker=SimpleNamespace(labels=labels),
                            facts={0: RoomFacts(0, 0, 40.0, 100), 1: RoomFacts(1, 2, 0.0, 300), 3: RoomFacts(3, 1, 0.0, 50)})
    line = summarise_floor({"graph": graph, "_floor_time": 125.0}, floor_id=1)
    assert line == "storey F1: rooms found: kitchen, living_room; 2 unknown; searched 2min; 2 rooms with frontier left"
    empty = SimpleNamespace(registry=SimpleNamespace(rooms={}), label_tracker=SimpleNamespace(labels={}), facts={})
    assert summarise_floor({"graph": empty, "_floor_time": 0.0}) == "rooms found: none yet; searched 0s; 0 rooms with frontier left"
def test_stair_options_offer_every_reachable_uncooled_connector_with_its_facts_and_leaf():
    policy, episode = gt_policy()
    world = hall()
    obs = at(episode, 12, 3.0, 3.0)
    policy.plan(obs)                                       # floor 0 settled, the connector a portal
    building = policy.building
    options = stair_options(building, obs, world, cost_for(policy, world), policy.floors.save(), action_time_s=1.0)
    [option] = options
    assert option.node_id == stair_node_id(0) and option.portal is building.portals[0]
    assert option.node.kind == "stairs" and option.node.label == "stairs up" and option.node.direction == 1
    assert not option.node.destination_visited and option.node.destination is None and not option.node.arrived_by
    assert option.leaf_m == pytest.approx(3 * math.hypot(1.0, 0.9) + building.params.floor_change_cost_m)
    assert option.approach.xy == pytest.approx((5.0, 3.0), abs=RES) and 1.9 < option.approach.distance_m < 2.5
    room_shaped = option.option(0.42)
    assert room_shaped.room_id == option.node_id and room_shaped.prob == 0.42 and room_shaped.xy == option.approach.xy
    building.arrived_by, building.arrived_step = 0, 4
    assert stair_options(building, obs, world, cost_for(policy, world), policy.floors.save()) == [], (
        "arrived by these stairs 8 actions ago: the way back is held for the arrival grace")
    building.arrived_step = 12 - building.params.arrival_grace_actions
    [came_by] = stair_options(building, obs, world, cost_for(policy, world), policy.floors.save(), action_time_s=1.0)
    assert came_by.node.arrived_by and came_by.node.arrived_ago_s == pytest.approx(float(building.params.arrival_grace_actions))
    building.arrived_by = building.arrived_step = None
    building.portals[0]["cooldown_until"] = 99
    assert stair_options(building, obs, world, cost_for(policy, world), policy.floors.save()) == []
    building.portals[0]["cooldown_until"] = 0
    policy.mapping.atlas.floors[1] = ObservedFloor(1, 2.7)
    policy.floors.contexts[1] = {"graph": SimpleNamespace(registry=SimpleNamespace(rooms={}),
                                                          label_tracker=SimpleNamespace(labels={}), facts={}),
                                 "_floor_time": 30.0}
    [visited] = stair_options(building, obs, world, cost_for(policy, world), policy.floors.save())
    assert visited.node.destination_visited and visited.node.destination.startswith("storey F1: rooms found: none yet")
    context = search_context(building, policy, policy.floors.save())
    assert "2 storeys known" in context.storey and context.others == (visited.node.destination,)
def test_the_coordinator_decides_nothing_by_a_clock_in_ground_truth_mode():
    """No allowance, no timer: without ``exhausted`` the building never chooses a portal on its own."""
    policy, episode = gt_policy()
    world = hall()
    policy.plan(at(episode, 0, 3.0, 3.0))
    building = policy.building
    for step in range(1, 400, 37):
        assert building.plan(at(episode, step, 3.0, 3.0), world) is None
    assert building.active is None and not [e for e in building.events if e["event"] in ("floor_decision", "portal_selected")]
    assert building.diagnostics()["decides_floor_changes"].startswith("room-search loop")
    assert building.portals[0]["cooldown_until"] == 0
def test_commit_makes_the_loops_choice_the_active_portal_and_the_climb_ends_the_nodes_turn():
    policy, episode = gt_policy()
    world = hall()
    obs = at(episode, 3, 3.0, 3.0)
    policy.plan(obs)
    building = policy.building
    taken = []
    policy.loop.stairs_taken = lambda o, portal: taken.append((o.step, portal["id"]))
    building.commit(obs, building.portals[0], (5.0, 3.0))
    assert building.active is building.portals[0] and building.active["selected_by"] == "rpt_star"
    assert building.committed and building.phase == "APPROACH_STAIRS"
    selected = [e for e in building.events if e["event"] == "portal_selected"]
    assert selected[-1]["solver_source"] == "rpt_star" and selected[-1]["portal_id"] == 0
    building.commit(obs, building.portals[0], (5.0, 3.0))
    assert len([e for e in building.events if e["event"] == "portal_selected"]) == 1, "idempotent for the same portal"
    approach = building.plan(at(episode, 4, 3.0, 3.0), world)
    assert approach is not None and approach.info["kind"] == "portal/0" and taken == []
    climb = building.plan(at(episode, 5, 4.6, 3.0), world)
    assert building.transition is not None and climb.info["kind"] == "TRAVERSE"
    assert taken == [(5, 0)], "the loop is told the action the climb begins"
    started = [e for e in building.events if e["event"] == "traversal_started"][-1]
    assert started["selected_by"] == "rpt_star"
def test_a_failed_approach_is_re_snapped_before_the_connector_is_deferred(monkeypatch):
    policy, episode = gt_policy()
    world = hall()
    policy.plan(at(episode, 0, 3.0, 3.0))
    building = policy.building
    building.plan(at(episode, 1, 3.0, 3.0), world, exhausted=True)
    assert building.active is not None
    calls = []
    monkeypatch.setattr(policy, "_navigate", lambda obs, w, goal, kind, final_yaw=None: calls.append(goal) or None)
    for step in range(2, 2 + building.params.approach_failures - 1):
        assert building.plan(at(episode, step, 3.0, 3.0), world) is None
        assert building.active is not None, "the approach is retried, not deferred on the first failure"
    assert building.plan(at(episode, 9, 3.0, 3.0), world) is None
    assert building.active is None
    deferred = [e for e in building.events if e["event"] == "portal_deferred"]
    assert deferred and deferred[-1]["reason"] == "portal approach unavailable"


# -- the coordinator ---------------------------------------------------------------
def test_ground_truth_is_the_default_and_the_connectors_become_portals_on_the_first_action():
    policy, episode = gt_policy()
    assert policy.settings.multifloor.stair_source == "ground_truth"
    assert policy.configuration()["ground_truth_stairs"] is True
    policy.plan(at(episode, 0, 3.0, 3.0))
    building = policy.building
    assert building.ground_truth.available and building.phase == "SEARCH"
    assert policy.mapping.atlas.plateau_observer is None, "the depth plateau gate belongs to the observed mode"
    [p] = building.portals
    assert p["connector_id"] == 0 and p["direction"] == 1 and p["stair_source"] == "ground_truth"
    assert p["entry"] == pytest.approx(FLIGHT[0]) and p["destination_z"] == pytest.approx(2.7)
    assert p["observations"] >= 2 and p["cooldown_until"] == 0


def test_observed_mode_remains_selectable_and_an_unknown_source_is_refused():
    policy, _ = setup_policy("bed", multifloor={"stair_source": "observed"})
    assert policy.building.ground_truth is None and policy.configuration()["ground_truth_stairs"] is False
    with pytest.raises(ValueError, match="stair_source"):
        RPTSettings(multifloor={"stair_source": "yolo"})


def test_a_raised_bathroom_floor_is_not_a_staircase():
    """13 cm up, nowhere near a connector: the observed mode began a traversal here."""
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 1.0, 1.0))
    building = policy.building
    for step in range(1, 6):
        command = policy.plan(at(episode, step, 1.0 + 0.1 * step, 1.0, z=0.13))
        assert building.transition is None and building.phase == "SEARCH"
        assert command.info["phase"] not in ("TRAVERSE", "APPROACH_STAIRS", "CONFIRM_DESTINATION", "RETREAT")
    assert not [e for e in building.events if e["event"] == "traversal_started"]


def test_a_height_departure_away_from_every_connector_is_ignored_once_and_logged_once():
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 1.0, 1.0))
    building = policy.building
    for step in range(1, 4):
        policy.plan(at(episode, step, 1.0 + 0.1 * step, 1.0, z=0.6))
    ignored = [e for e in building.events if e["event"] == "height_departure_ignored"]
    assert len(ignored) == 1 and ignored[0]["height_m"] == pytest.approx(0.6)
    assert building.transition is None and building.phase == "SEARCH"


def test_walking_onto_the_stairs_by_accident_completes_the_climb_along_the_connector():
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 4.0, 3.0))
    building = policy.building
    command = policy.plan(at(episode, 1, 6.0, 3.0, z=0.9))          # on the flight, half a storey up
    assert building.transition is not None and building.phase == "TRAVERSE"
    assert command.info["phase"] == "TRAVERSE" and command.waypoints, "a route along the connector, not a hold"
    assert command.info["direction"] == "up" and command.info["stair_source"] == "ground_truth"
    assert building.active["connector_id"] == 0 and building.active["direction"] == 1
    assert list(command.waypoints[-1]) == pytest.approx(FLIGHT[-1][:2])
    started = [e for e in building.events if e["event"] == "traversal_started"]
    assert len(started) == 1 and started[0]["stair_source"] == "ground_truth"


def test_arrival_at_the_top_anchor_settles_the_new_floor_and_offers_the_way_back_as_a_node():
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 4.0, 3.0))
    building = policy.building
    policy.plan(at(episode, 1, 6.0, 3.0, z=0.9))
    policy.plan(at(episode, 2, 7.0, 3.0, z=1.8))
    assert building.transition.phase == "TRAVERSE" and not building.transition.arrival_allowed
    policy.plan(at(episode, 3, 8.1, 3.0, z=2.7))
    assert building.transition.phase == "CONFIRM_DESTINATION" and building.transition.arrival_allowed
    step = 4
    for x in (8.5, 8.9, 9.3):
        policy.plan(at(episode, step, x, 3.0, z=2.7))
        step += 1
    assert policy.mapping.atlas.active_id == 1 and building.floor_id == 1 and building.transition is None
    arrival = [e for e in building.events if e["event"] == "floor_arrival"]
    assert arrival and arrival[0]["reason"] == "ground_truth_connector_end_reached"
    assert building.completed and building.completed[0]["stair_source"] == "ground_truth"
    upstairs = [p for p in building.portals if p["floor_id"] == 1]
    assert len(upstairs) == 1 and upstairs[0]["direction"] == -1
    assert upstairs[0]["cooldown_until"] == 0, "no clock keeps the way back off the list: the oracle values it"
    assert building.arrived_by == 0 and building.arrived_step == arrival[0]["action"]
    assert policy.mapping.floor_id == 1 and policy.floors.active == 1
    assert policy.loop is not None and policy.loop.room_id is None, "a fresh loop for the new floor"


def test_the_stairs_just_climbed_are_not_the_fallbacks_answer_on_the_new_floor():
    """Pomaria: confirmed upstairs at action 75, back down the same flight from action 78.

    The new floor's map was two frames old, so the fallback read the floor
    as exhausted and its rule chose the only staircase -- the one the agent
    stood at the head of. The way back is held out of the fallback (and out
    of the loop's node list) for ``arrival_grace_actions`` after an arrival.
    """
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 4.0, 3.0))
    building = policy.building
    for step, (x, z) in enumerate(((6.0, 0.9), (7.0, 1.8), (8.1, 2.7), (8.5, 2.7), (8.9, 2.7), (9.3, 2.7)), 1):
        policy.plan(at(episode, step, x, 3.0, z=z))
    assert building.floor_id == 1 and building.arrived_by == 0
    upstairs = hall()
    arrival = building.arrived_step
    grace = building.params.arrival_grace_actions
    assert building.way_back_held(arrival) and building.way_back_held(arrival + grace - 1)
    assert not building.way_back_held(arrival + grace)
    for step in range(arrival + 1, arrival + 4):
        assert building.plan(at(episode, step, 8.5, 3.0, z=2.7), upstairs, exhausted=True) is None
    assert building.active is None, "the way back was not chosen"
    held = [e for e in building.events if e["event"] == "way_back_held"]
    assert len(held) == 1 and held[0]["connector_id"] == 0 and held[0]["until"] == arrival + grace
    decisions = [e for e in building.events if e["event"] == "floor_decision" and e["action"] >= arrival]
    assert decisions and decisions[-1]["reason"] == "no eligible connector on this floor"
    assert len(decisions) == 1, "one record per distinct verdict: the held way back is not re-recorded every action"
    building.plan(at(episode, arrival + grace, 8.5, 3.0, z=2.7), upstairs, exhausted=True)
    assert building.active is not None and building.active["connector_id"] == 0, (
        "the grace is over: an exhausted storey may be left by the way it was entered")


def test_the_exhausted_floor_takes_the_stairs_by_the_fallback_rule_and_the_choice_is_recorded():
    policy, episode = gt_policy()
    world = hall()
    obs = at(episode, 0, 3.0, 3.0)
    policy.plan(obs)
    building = policy.building
    command = building.plan(at(episode, 1, 3.0, 3.0), world, exhausted=True)
    assert building.active is not None and building.phase == "APPROACH_STAIRS"
    assert building.active["selected_by"] == "floor_decision"
    assert command is not None and command.waypoints and command.info["kind"] == "portal/0"
    decision = [e for e in building.events if e["event"] == "floor_decision"]
    selected = [e for e in building.events if e["event"] == "portal_selected"]
    assert len(decision) == 1 and decision[0]["trigger"] == "floor_exhausted" and decision[0]["chosen"] == 0
    assert decision[0]["rule"] == "fallback"
    assert selected and selected[0]["direction"] == "up" and selected[0]["solver_source"] == "floor_decision"
    # Standing at the entry anchor commits the climb on the connector's own polyline.
    command = building.plan(at(episode, 2, 4.6, 3.0), world)
    assert building.transition is not None and command.info["kind"] == "TRAVERSE"
    assert list(command.waypoints[0]) == pytest.approx(FLIGHT[0][:2]) or list(command.waypoints[-1]) == pytest.approx(FLIGHT[-1][:2])
    started = [e for e in building.events if e["event"] == "traversal_started"][0]
    assert started["selected_by"] == "floor_decision", "a fallback climb ends no loop turn"


def test_no_look_down_inspections_and_no_repeated_decision_records_in_ground_truth_mode():
    policy, episode = gt_policy()
    world = hall(unknown_east_of=3.0)                      # the entry anchor at x=5 is 2 m into unknown space
    building = policy.building
    # The stairs have been seen (from afar); their foot is not on the observed map yet.
    building.sightings.mark_seen(0, 0)
    policy.plan(at(episode, 0, 1.5, 3.0))
    assert len(building.portals) == 1
    for step in range(200, 206):
        assert building.plan(at(episode, step, 1.5, 3.0), world) is None
    assert not [e for e in building.events if e["event"] == "floor_decision"], "no allowance, no unasked decision"
    for step in range(206, 212):                            # the fallback asks, every action
        assert building.plan(at(episode, step, 1.5, 3.0), world, exhausted=True) is None
    assert policy.camera_control.inspection is None
    decisions = [e for e in building.events if e["event"] == "floor_decision"]
    assert [d["trigger"] for d in decisions] == ["floor_exhausted"] * 2, "one record per distinct verdict, not per action"
    assert "re-check" in decisions[0]["candidates"][0]["verdict"]
    assert decisions[1]["reason"] == "no eligible connector on this floor"       # the re-check cooldown
    assert building.diagnostics()["stair_source"] == "ground_truth"
    assert building.diagnostics()["ground_truth"]["connectors"][0]["top_z"] == pytest.approx(2.7)


def test_a_staircase_never_seen_is_not_a_portal_and_not_a_candidate():
    """Ground truth about the stairs is a bounding box when they are in the frame -- nothing otherwise.

    The depth image here reads 3 m everywhere and the foot of the flight is
    3.5 m away: the perfect detector sees no stairs, so the connector is
    neither placed on the map nor offered to the fallback rule, however
    exhausted the floor is.
    """
    policy, episode = gt_policy()
    world = hall()
    policy.plan(at(episode, 0, 1.5, 3.0))
    building = policy.building
    assert building.portals == [] and not building.sightings.seen(0)
    for step in range(1, 6):
        assert building.plan(at(episode, step, 1.5, 3.0), world, exhausted=True) is None
    assert building.portals == [] and building.active is None
    decision = [e for e in building.events if e["event"] == "floor_decision"]
    assert decision and decision[0]["reason"] == "no eligible connector on this floor"
    assert not [e for e in building.events if e["event"] == "stairs_seen"]
    # Walk up to the foot: the flight comes within the depth image's 3 m and is seen.
    policy.plan(at(episode, 6, 3.0, 3.0))
    seen = [e for e in building.events if e["event"] == "stairs_seen"]
    assert len(seen) == 1 and seen[0]["connector_id"] == 0 and seen[0]["bbox"][2] > seen[0]["bbox"][0]
    placed = [e for e in building.events if e["event"] == "portal_placed"]
    assert len(placed) == 1 and placed[0]["seen_step"] == 6
    [portal] = building.portals
    assert portal["connector_id"] == 0 and portal["seen_step"] == 6 and len(portal["footprint"]) >= 6
    assert building.diagnostics()["stairs_seen"]["seen"] == [0]
    assert building.last_sightings and building.last_sightings[0]["label"] == "stairs"
    assert building.last_sightings[0]["source"] == "ground_truth" and building.last_sightings[0]["confidence"] == 1.0


def test_a_height_departure_on_stairs_never_seen_is_not_explained_by_them():
    """Off the floor by 0.6 m on the flight's own polyline, but the camera never had the stairs in frame."""
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 4.0, 3.0, yaw=math.pi))       # facing away from the flight
    building = policy.building
    building.detector = None                                 # the frame is synthetic; keep the stairs unseen
    building.sightings.records.clear()
    policy.plan(at(episode, 1, 6.0, 3.0, z=0.9, yaw=math.pi))
    assert building.transition is None and building.phase == "SEARCH"
    ignored = [e for e in building.events if e["event"] == "height_departure_ignored"]
    assert len(ignored) == 1 and "never been seen" in ignored[0]["reason"]


# -- the traversal -------------------------------------------------------------
def committed(policy, episode):
    obs = at(episode, 0, 4.6, 3.0)
    policy.plan(obs)
    building = policy.building
    building.active = building._portal_for(obs, building.ground_truth.connectors[0], 0.0)
    building._start(obs)
    return building, building.transition


def test_the_traversal_follows_the_polyline_and_never_asks_for_an_idle_hold():
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    assert t.direction == 1 and t.destination_height == pytest.approx(2.7) and not t.arrival_allowed
    poses = [(5.0, 0.0), (5.5, 0.4), (6.0, 0.9), (6.5, 1.35), (7.0, 1.8), (7.5, 2.25)]
    for step, (x, z) in enumerate(poses, 1):
        obs = at(episode, step, x, 3.0, z=z)
        t.observe(obs)
        command = t.plan(obs)
        assert command.waypoints and not command.stop, "every action is a route, never a turn in place"
        assert command.info["kind"] == "TRAVERSE"
    assert t.cursor >= 2 and t.phase == "TRAVERSE"
    obs = at(episode, 7, 8.1, 3.0, z=2.7)
    t.observe(obs)
    assert t.phase == "CONFIRM_DESTINATION" and t.arrival_allowed
    command = t.plan(obs)
    assert command.waypoints and command.waypoints[-1][0] > 8.1, "step clear of the stair head"


def test_a_stalled_traversal_retreats_along_the_way_it_came_and_a_spent_retreat_lets_the_atlas_settle():
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    obs = at(episode, 1, 6.0, 3.0, z=0.9)
    t.observe(obs)
    t.plan(obs)
    stuck = at(episode, 2, 6.0, 3.0, z=0.9)
    stall = building.params.commit_stall_actions
    for step in range(2, 2 + stall - 1):
        command = t.plan(replace(stuck, step=step))
        assert command.waypoints and not command.stop
        assert t.phase == "TRAVERSE", "the committed stall clock is the long one, not the observed mode's 18"
    command = t.plan(replace(stuck, step=2 + stall))
    assert command.waypoints and not command.stop
    assert t.phase == "RETREAT" and t.retreat_path is not None
    retreat = [e for e in building.events if e["event"] == "retreat_started"][0]
    assert retreat["reason"] == "stalled" and retreat["entered_flight"] is True
    assert close(t.retreat_path, [FLIGHT[1], FLIGHT[0], [4.6, 3.0, 0.0]]), (
        "back down the vertices walked to the entry anchor, then off the flight to where the approach ended")
    for step in range(2 + stall + 1, 2 + stall + 1 + building.params.retreat_actions + 1):
        command = t.plan(replace(stuck, step=step))
        assert command.waypoints and not command.stop, "no SAFE_HALT: the search goes on"
    assert t.arrival_allowed and t.completion_reason == "ground_truth_retreat_budget_exhausted"


def test_a_retreat_from_a_flight_never_entered_goes_back_to_where_the_traversal_began():
    """Stalled short of the bottom anchor: the way back is to the approach's end, not to the anchor it could not reach.

    In the first recorded campaign the approach ended 0.4-1.9 m short of the
    anchor (the anchor is rarely a passable cell of the observed map) and the
    retreat aimed at that same unreachable anchor -- one more route through the
    same wall. Here the agent is already where the retreat leads, so the return
    is confirmed on the spot and the coordinator can defer the connector.
    """
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    stuck = at(episode, 1, 4.6, 3.0)
    t.observe(stuck)
    for step in range(1, 2 + building.params.commit_stall_actions):
        t.plan(replace(stuck, step=step))
    assert t.phase == "RETREAT" and t.cursor == 0
    retreat = [e for e in building.events if e["event"] == "retreat_started"][0]
    assert retreat["entered_flight"] is False
    assert close(t.retreat_path, [[4.6, 3.0, 0.0], [4.6, 3.0, 0.0]])
    t.observe(replace(stuck, step=2 + building.params.commit_stall_actions))
    assert t.arrival_allowed and t.completion_reason == "ground_truth_source_returned"
    assert not policy.mapping.atlas.in_transition, "the return is settled on the spot, not left to a plateau test"
    assert policy.mapping.atlas.active_id == 0


def test_a_blocked_step_tightens_the_following_and_then_goes_back_to_the_last_vertex_never_skipping():
    """The first block aims straight at the next vertex; when that is blocked twice the agent goes BACK, not on.

    Skipping a vertex aimed the route at the vertex after it through the
    landing wall (Ranchester, twice). The last vertex reached is a point of
    the centreline from which the next leg is walkable, so the agent returns
    there and comes at the vertex again; the route itself never loses a point.
    """
    policy, episode = gt_policy()
    building, t = committed(policy, episode)                 # the traversal began at (4.6, 3.0)
    obs = at(episode, 1, 5.0, 3.0, z=0.0)
    t.observe(obs)
    before = t.cursor
    building.blocked(obs)
    assert t.failures == 1 and t.cursor == before and t.tight and t.recoveries == 0
    command = t.plan(replace(obs, step=2))
    assert command.info["tight"] is True and command.info["recovering"] is False
    assert len(command.waypoints) == 2 and list(command.waypoints[0]) == pytest.approx([5.0, 3.0])
    assert list(command.waypoints[1]) == pytest.approx(FLIGHT[before + 1][:2]), "a straight line to the next vertex"
    building.blocked(obs)                                     # the tight aim was blocked too
    assert t.cursor == before and t.tight and t.recoveries == 0, "one tight attempt is not enough to give up the aim"
    building.blocked(obs)
    assert t.cursor == before and t.recoveries == 1 and not t.tight, "twice: go back to the last vertex reached"
    assert len(t.route) == len(FLIGHT), "no vertex is ever skipped"
    assert t._recovery == pytest.approx((4.6, 3.0, 0.0)), "standing on the foot itself, the way back is to where the traversal began"
    command = t.plan(replace(obs, step=3))
    assert command.info["recovering"] is True and len(command.waypoints) == 2
    assert list(command.waypoints[1]) == pytest.approx([4.6, 3.0])
    recovery = [e for e in building.events if e["event"] == "traversal_recovery"]
    assert len(recovery) == 1 and recovery[0]["vertex"] == before + 1
    building.blocked(obs)                                     # a block on the way back changes nothing but the count
    assert t.failures == 4 and t.recoveries == 1 and t._recovery is not None
    back = at(episode, 4, 4.62, 3.02, z=0.0)
    t.observe(back)
    assert t._recovery is None and not t.tight, "back on the way: the route resumes"
    onward = t.plan(back)
    assert onward.info["recovering"] is False and list(onward.waypoints[0]) == pytest.approx([4.62, 3.02])
    assert list(onward.waypoints[1]) == pytest.approx(FLIGHT[0][:2]), "back through the foot, which it left 0.4 m behind"
    assert list(onward.waypoints[2]) == pytest.approx(FLIGHT[before + 1][:2]), "then toward the same vertex, along the route"
    reached = at(episode, 5, *FLIGHT[before + 1][:2], z=FLIGHT[before + 1][2])
    t.observe(reached)
    assert t.cursor == before + 1 and t.phase != "RETREAT"
    assert t.failures == 4, "four blocks would have turned the observed mode round"


def test_a_recovery_from_mid_flight_goes_back_to_the_previous_vertex():
    """Blocked twice on the way to vertex 2 while standing past vertex 1: the way back is vertex 1, not the foot."""
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    t.observe(at(episode, 1, 5.0, 3.0, z=0.0))
    t.observe(at(episode, 2, 6.0, 3.0, z=0.9))                # vertex 1 reached
    assert t.cursor == 1
    stuck = at(episode, 3, 6.5, 3.1, z=1.3)                   # half way to vertex 2, against something
    t.observe(stuck)
    for _ in range(3):
        building.blocked(stuck)
    assert t.recoveries == 1 and t._recovery == pytest.approx(tuple(FLIGHT[1]))
    assert t.cursor == 1 and len(t.route) == len(FLIGHT)


def test_only_many_blocked_steps_turn_the_climb_back():
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    obs = at(episode, 1, 5.0, 3.0, z=0.0)
    t.observe(obs)
    for _ in range(building.params.commit_failures - 1):
        building.blocked(obs)
    assert t.plan(replace(obs, step=2)).waypoints and t.phase != "RETREAT"
    building.blocked(obs)
    command = t.plan(replace(obs, step=3))
    assert t.phase == "RETREAT" and command.waypoints
    started = [e for e in building.events if e["event"] == "retreat_started"][0]
    assert started["reason"] == "blocked" and started["failures"] == building.params.commit_failures


def test_the_transition_budget_no_longer_turns_the_climb_round():
    """Out of budget mid-flight: keep following, and let the atlas settle if the agent stands on a storey."""
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    late = building.params.transition_actions + 5
    for step in range(1, late):
        # Creeping up the second leg at 4 mm an action: real progress -- the route ahead
        # shortens -- but never the next vertex, never the destination height.
        f = 0.004 * step
        obs = at(episode, step, 6.0 + f, 3.0, z=0.9 + 0.9 * f)
        t.observe(obs)
        command = t.plan(obs)
        assert command.waypoints and not command.stop
    assert t.phase == "TRAVERSE" and t.retreat_path is None
    assert t.arrival_allowed, "the clock only lets the atlas settle; it does not send the agent back down"
    assert not [e for e in building.events if e["event"] == "retreat_started"]


def test_a_shuffle_on_one_tread_is_a_stall_however_much_it_moves():
    """Moving is not progress: the route ahead has to get shorter.

    The Ranchester campaign's first episode skidded 4-9 cm along a stair-well
    wall on every one of 463 actions and never stalled, because the stall
    clock was reset by any XY displacement. Five centimetres back and forth on
    one tread is the same motion in miniature.
    """
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    stall = building.params.commit_stall_actions
    for step in range(1, stall + 1):
        obs = at(episode, step, 6.0 + 0.05 * (step % 2), 3.0, z=0.9)
        t.observe(obs)
        t.plan(obs)
        assert t.phase == "TRAVERSE", "one action short of the stall clock"
    obs = at(episode, stall + 1, 6.05, 3.0, z=0.9)
    t.observe(obs)
    t.plan(obs)
    assert t.phase == "RETREAT"
    assert [e for e in building.events if e["event"] == "retreat_started"][0]["reason"] == "stalled"


#: A switchback: two flights 1 m apart in XY around a landing, the top anchor a metre from the bottom one.
SWITCHBACK = [[5.0, 3.0, 0.0], [6.0, 3.0, 0.9], [6.6, 3.5, 1.35], [6.0, 4.0, 1.8], [5.0, 4.0, 2.7]]
SWITCHBACK_STRUCTURE = {"stair_source": "navmesh",
                        "floor_levels": [{"height_m": 0.0, "area_m2": 40.0}, {"height_m": 2.7, "area_m2": 40.0}],
                        "stair_connectors": [{"id": 0, "bottom_xyz": SWITCHBACK[0], "top_xyz": SWITCHBACK[-1],
                                              "bottom_z": 0.0, "top_z": 2.7, "polyline_xyz": SWITCHBACK,
                                              "length_m": 4.6, "rise_m": 2.7}]}


def test_the_route_is_led_in_from_where_the_agent_stands_so_a_folded_flight_is_walked_from_its_foot():
    """The converter must never project onto the far leg of a flight that folds back over its own foot.

    Handed the bare polyline of a U-shaped staircase from 0.6 m short of the
    bottom anchor -- nearer the TOP anchor in XY -- the converter aimed across
    the stair well at the top and ground on the wall for the rest of the
    episode. The command now leads in from the agent's own position, and that
    lead-in stays put while the next vertex does, so the converter keeps its
    forward-only progress from one action to the next.
    """
    from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
    policy, episode = gt_policy(metadata=SWITCHBACK_STRUCTURE)
    start = at(episode, 0, 4.6, 3.6, yaw=math.pi / 2)         # 0.72 m from the foot, 0.57 m from the top anchor
    policy.plan(start)
    building = policy.building
    building.active = building._portal_for(start, building.ground_truth.connectors[0], 0.0)
    building._start(start)
    t = building.transition
    command = t.plan(start)
    assert close(command.waypoints, [[4.6, 3.6], [5.0, 3.0]]), (
        "the lead-in from the agent to the foot of the flight, and nothing past it: the route bends there "
        "(56 degrees), and a corner is walked TO, never cut by the converter's lookahead")
    converter = DiscreteActionConverter(policy.episode.action_spec, policy.converter_params)
    result = converter.step(start.pose, command)
    assert math.dist(result.target_xy, SWITCHBACK[0][:2]) < math.dist(result.target_xy, SWITCHBACK[-1][:2]), (
        "the first aim is toward the foot of the flight, not across the well at its top")
    nearer = at(episode, 1, 4.75, 3.4, yaw=-0.9)
    t.observe(nearer)
    again = t.plan(nearer)
    assert again.waypoints == command.waypoints, "the same path while the next vertex is the same: progress is kept"
    reached = at(episode, 2, 5.05, 3.05, z=0.05)
    t.observe(reached)
    onward = t.plan(reached)
    assert t.cursor == 0 and close(onward.waypoints, [[5.05, 3.05], [6.0, 3.0]]), (
        "the foot reached (on it, so no lead back through it): a new lead-in toward the next vertex, ending "
        "where the flight turns onto the landing")
    beside = at(episode, 3, 5.9, 3.2, z=0.85)                # the first tread vertex reached from 0.22 m beside it
    t.observe(beside)
    assert t.cursor == 1
    through = t.plan(beside)
    assert close(through.waypoints[:2], [[5.9, 3.2], [6.0, 3.0]]), (
        "reached from off to one side: the route leads back THROUGH the vertex, so the aim lies on the "
        "centreline past it and never on a chord from beside the route to the vertex after")
    assert close(through.waypoints[2:3], [[6.6, 3.5]])


def test_a_vertex_is_reached_in_three_dimensions_not_in_plan_view():
    """On a switchback the flight above passes within a hand's breadth of the flight below in XY."""
    policy, episode = gt_policy(metadata=SWITCHBACK_STRUCTURE)
    start = at(episode, 0, 4.6, 3.0)
    policy.plan(start)
    building = policy.building
    building.active = building._portal_for(start, building.ground_truth.connectors[0], 0.0)
    building._start(start)
    t = building.transition
    t.observe(at(episode, 1, 5.0, 3.0, z=0.0))
    assert t.cursor == 0
    t.observe(at(episode, 2, 6.0, 3.9, z=0.9))               # under the upper flight's vertex (6, 4, 1.8)
    assert t.cursor == 0, "0.1 m away in plan view and 0.9 m below: not that vertex"
    t.observe(at(episode, 3, 6.0, 3.1, z=0.9))
    assert t.cursor == 1


def test_the_destination_is_confirmed_outright_when_the_atlas_cannot_settle_at_the_stair_head():
    """At the top, at the destination height, but boxed in so no translated plateau forms."""
    policy, episode = gt_policy()
    policy.plan(at(episode, 0, 4.0, 3.0))
    building = policy.building
    policy.plan(at(episode, 1, 6.0, 3.0, z=0.9))
    policy.plan(at(episode, 2, 7.0, 3.0, z=1.8))
    policy.plan(at(episode, 3, 8.1, 3.0, z=2.7))
    assert building.transition.phase == "CONFIRM_DESTINATION"
    step = 4
    for _ in range(building.params.confirm_actions + 1):      # standing still at the stair head
        policy.plan(at(episode, step, 8.1, 3.0, z=2.7))
        step += 1
    assert policy.mapping.atlas.active_id == 1 and building.floor_id == 1 and building.transition is None
    forced = [e for e in building.events if e["event"] == "destination_forced"]
    assert len(forced) == 1 and forced[0]["waited_actions"] >= building.params.confirm_actions
    arrival = [e for e in building.events if e["event"] == "floor_arrival"]
    assert arrival and arrival[0]["reason"] == "ground_truth_destination_forced"
    assert policy.mapping.atlas.floors[1].elevation_m == pytest.approx(2.7)


def test_the_atlas_refuses_to_settle_a_landing_as_a_storey():
    atlas = FloorAtlas(MultiFloorParams())
    atlas.update(AgentPose(0, 0, 0, 0))
    assert not atlas.settle(AgentPose(1, 0, 0.9, 0), 0.9), "0.9 m above the only floor: a landing"
    assert atlas.active_id == 0 and len(atlas.floors) == 1
    assert atlas.settle(AgentPose(2, 0, 2.7, 0), 2.7)
    assert atlas.active_id == 1 and atlas.floors[1].elevation_m == pytest.approx(2.7)
    assert atlas.completion_reason == "destination_platform_confirmed" and atlas.connections


def test_returning_to_the_source_after_a_retreat_defers_the_connector_instead_of_halting():
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    obs = at(episode, 1, 6.0, 3.0, z=0.9)
    t.observe(obs)
    t._begin_retreat(replace(obs, step=2), "test")
    for step, x in enumerate((5.6, 5.2, 4.8, 4.4), 3):
        policy.plan(at(episode, step, x, 3.0, z=0.0))
    assert building.transition is None and building.active is None and building.phase == "SEARCH"
    deferred = [e for e in building.events if e["event"] == "portal_deferred"]
    assert deferred and deferred[0]["reason"] == "ground_truth_source_returned"
    assert building.portals[0]["cooldown_until"] > 0
    assert policy.mapping.atlas.active_id == 0 and not policy.mapping.atlas.in_transition




