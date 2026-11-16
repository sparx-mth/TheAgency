"""Staircases as nodes of the room-search loop: ids, facts, the approach point, the climb's cost -- and who may supply them.

The loop's side of "a staircase is a node like a room" (``rooms/stair_nodes.py``,
``multifloor/stair_approach.py``, ``MultiFloorSearch.commit``) is source-neutral: it
reads the building coordinator's portals. A portal becomes a node only when it says
which storey its flight leads to, and the one source that ever said so -- stair
connectors and storey levels read from the simulator's navmesh, handed to the agent in
the episode metadata -- was privileged scene geometry and is removed (2026-11-15). The
observed discovery measures a flight's treads, not where it ends, so today no run
offers a stair node; these tests place such a portal by hand (``place_stairs``) to keep
the loop's side honest for an observed source that does measure it, and pin that the
simulator's stairs cannot be asked for.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.localization.floor_atlas import MultiFloorParams, ObservedFloor
from sparx_agency.core.mapping.topology.room_classifier import RoomLabel
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.object_search.multifloor.stair_approach import (
    ApproachSettings, approach_points, floor_at, stair_cost_m)
from sparx_agency.core.planning.object_search.rooms.stair_nodes import (
    STAIR_NODE_BASE, is_stair_node, portal_id_of, search_context, stair_node_id, stair_options, summarise_floor)
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import place_stairs

RES = 0.1
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)

#: One straight flight, ENU: its foot at (5, 3, 0), its head at (8, 3, 2.7).
FLIGHT = [[5.0, 3.0, 0.0], [6.0, 3.0, 0.9], [7.0, 3.0, 1.8], [8.0, 3.0, 2.7]]


def hall(unknown_east_of=None):
    """A 12 m x 6 m free hall, optionally unknown beyond ``unknown_east_of`` metres."""
    g = np.zeros((60, 120), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    if unknown_east_of is not None:
        g[1:-1, int(unknown_east_of / RES):-1] = -1
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def stairs_allowed(**overrides):
    # These are the multi-storey regressions: the stairs must be takeable, so the
    # benchmark default (``allow_stair_traversal=False``) is lifted explicitly.
    return setup_policy("bed", **dict({"map_size_m": 40.0, "allow_stair_traversal": True}, **overrides))


def at(episode, step, x, y, z=0.0, yaw=0.0):
    return replace(observation(episode, step, depth=3), pose=AgentPose(x, y, z, yaw))


def cost_for(policy, world):
    return assemble_cost_grid(policy.planner.fields_for(world), policy.planner_params, policy.settings.body_radius_m)[0]


def portal(pid, entry, direction, destination_z, cooldown_until=0):
    return {"id": pid, "entry": list(entry), "direction": direction, "destination_z": destination_z,
            "cooldown_until": cooldown_until, "stairs_id": pid, "floor_id": 0}


# -- ids and the prompt's words ------------------------------------------------------
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
    # A weak label is marked as the node lines mark it: a cabinet's "kitchen?" is not a kitchen found.
    weak = SimpleNamespace(registry=SimpleNamespace(rooms={0: 1, 2: 1}),
                           label_tracker=SimpleNamespace(labels=labels, metadata={0: {"strength": "weak"}, 2: {"strength": "strong"}}),
                           facts={})
    assert summarise_floor({"graph": weak, "_floor_time": 0.0}).startswith("rooms found: kitchen?, living_room;")


# -- where a staircase is entered from, and what it costs -----------------------------
def test_approach_points_snap_every_reachable_entry_onto_the_observed_map_in_one_sweep():
    policy, episode = stairs_allowed()
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
    entry through what the map knows: unknown is allowed, occupied is not --
    the entry's own cell excepted, and nothing more of the obstacle it may
    stand in (the standing floor's snap exempts the whole obstacle; this one
    does not, and a stair head walled in on the map has no approach point).
    """
    policy, episode = stairs_allowed()
    world = corridor_hall()
    cost = cost_for(policy, world)
    entry = portal(0, (9.0, 3.0, 0.0), -1, -2.7)
    approaches = approach_points([entry], at(episode, 0, 3.0, 3.0), world, cost, ApproachSettings(approach_snap_m=2.0))
    assert 0 in approaches
    x, y = approaches[0].xy
    assert x > 8.0 and 1.0 < y < 1.5, "in the corridor, at its mouth -- not in the room behind the wall: %r" % (approaches[0].xy,)
    walled = corridor_hall()
    walled.grid[1:-1, 81:-1] = 100                   # the corridor is all wall as far as the map knows
    assert approach_points([entry], at(episode, 0, 3.0, 3.0), walled, cost_for(policy, walled), ApproachSettings(approach_snap_m=2.0)) == {}, (
        "no cell sees the entry: the map has not reached it, rather than a made-up point behind a wall")


def test_a_storey_change_costs_the_flight_plus_the_fixed_change_and_a_height_names_a_visited_storey_or_none():
    flight = dict(portal(0, FLIGHT[0], +1, 2.7), path=FLIGHT)
    assert stair_cost_m(flight, 8.0) == pytest.approx(3 * math.hypot(1.0, 0.9) + 8.0)
    assert stair_cost_m(portal(1, FLIGHT[0], +1, 2.7), 8.0) == pytest.approx(8.0), "no polyline: the fixed cost alone"
    assert stair_cost_m(flight, -3.0) == pytest.approx(3 * math.hypot(1.0, 0.9)), "a negative change cost is no discount"
    atlas = SimpleNamespace(floors={0: ObservedFloor(0, 0.0), 1: ObservedFloor(1, 2.75)})
    assert floor_at(atlas, 2.7, 0.3) == 1 and floor_at(atlas, 0.1, 0.3) == 0
    assert floor_at(atlas, 5.4, 0.3) is None, "a storey never stood on is unvisited, whatever the building has"


# -- the node list ---------------------------------------------------------------------
def test_stair_options_offer_every_reachable_uncooled_staircase_with_its_facts_and_leaf():
    policy, episode = stairs_allowed()
    world = hall()
    obs = at(episode, 12, 3.0, 3.0)
    policy.plan(obs)                                       # floor 0 settled
    building = policy.building
    placed = place_stairs(policy, obs, FLIGHT)
    options = stair_options(building, obs, world, cost_for(policy, world), policy.floors.save(), action_time_s=1.0)
    [option] = options
    assert option.node_id == stair_node_id(placed["id"]) and option.portal is placed
    assert option.node.kind == "stairs" and option.node.label == "stairs up" and option.node.direction == 1
    assert not option.node.destination_visited and option.node.destination is None and not option.node.arrived_by
    assert option.leaf_m == pytest.approx(3 * math.hypot(1.0, 0.9) + building.params.floor_change_cost_m)
    assert option.approach.xy == pytest.approx((5.0, 3.0), abs=RES) and 1.9 < option.approach.distance_m < 2.5
    room_shaped = option.option(0.42)
    assert room_shaped.room_id == option.node_id and room_shaped.prob == 0.42 and room_shaped.xy == option.approach.xy
    building.arrived_by, building.arrived_step = placed["stairs_id"], 4
    assert stair_options(building, obs, world, cost_for(policy, world), policy.floors.save()) == [], (
        "arrived by these stairs 8 actions ago: the way back is held for the arrival grace")
    building.arrived_step = 12 - building.params.arrival_grace_actions
    [came_by] = stair_options(building, obs, world, cost_for(policy, world), policy.floors.save(), action_time_s=1.0)
    assert came_by.node.arrived_by and came_by.node.arrived_ago_s == pytest.approx(float(building.params.arrival_grace_actions))
    building.arrived_by = building.arrived_step = None
    placed["cooldown_until"] = 99
    assert stair_options(building, obs, world, cost_for(policy, world), policy.floors.save()) == []
    placed["cooldown_until"] = 0
    policy.mapping.atlas.floors[1] = ObservedFloor(1, 2.7)
    policy.floors.contexts[1] = {"graph": SimpleNamespace(registry=SimpleNamespace(rooms={}),
                                                          label_tracker=SimpleNamespace(labels={}), facts={}),
                                 "_floor_time": 30.0}
    [visited] = stair_options(building, obs, world, cost_for(policy, world), policy.floors.save())
    assert visited.node.destination_visited and visited.node.destination.startswith("storey F1: rooms found: none yet")
    context = search_context(policy, policy.floors.save())
    assert context.others == (visited.node.destination,), "the storey the robot has stood on, in its own words"


def test_a_flight_the_robot_only_measured_is_not_a_node_and_the_prompt_names_no_storey_it_has_not_stood_on():
    """What the observed discovery places: a flight, its direction and its treads -- not the storey it ends on.

    Such a portal is masked on the planning grid and may be walked by the
    coordinator's own rule; it is not offered to the oracle, because every
    fact a stair node carries is about a destination nobody has measured.
    And the prompt's storey lines hold what the robot found on the storeys
    it stood on -- never how many the building has or where this one ranks,
    which until 2026-11-15 came from the simulator's navmesh.
    """
    policy, episode = stairs_allowed()
    world = hall()
    obs = at(episode, 12, 3.0, 3.0)
    policy.plan(obs)
    building = policy.building
    measured = place_stairs(policy, obs, FLIGHT, leads_to=False)
    assert "destination_z" not in measured and "stairs_id" not in measured
    assert stair_options(building, obs, world, cost_for(policy, world), policy.floors.save()) == []
    building.arrived_by, building.arrived_step = None, 4
    assert stair_options(building, obs, world, cost_for(policy, world), policy.floors.save()) == [], (
        "a portal with no identity is never the stairs the robot arrived by, and still not a node")
    context = search_context(policy, policy.floors.save())
    assert context.others == () and not context.multi_storey
    for phrase in ("storeys known", "known to the building", "lowest", "highest"):
        assert phrase not in context.storey
    assert not hasattr(building, "ground_truth") and not hasattr(building, "sightings")
    assert "ground_truth" not in building.diagnostics() and "stairs_seen" not in building.diagnostics()
    assert building.diagnostics()["decides_floor_changes"].startswith("allowance")


def test_commit_makes_the_loops_choice_the_active_portal_and_the_climb_ends_the_nodes_turn():
    policy, episode = stairs_allowed()
    world = hall()
    obs = at(episode, 3, 3.0, 3.0)
    policy.plan(obs)
    building = policy.building
    placed = place_stairs(policy, obs, FLIGHT)
    taken = []
    policy.loop.stairs_taken = lambda o, portal: taken.append((o.step, portal["id"]))
    assert building.commit(obs, placed, (5.0, 3.0))
    assert building.active is placed and building.active["selected_by"] == "rpt_star"
    assert building.committed and building.phase == "APPROACH_STAIRS"
    selected = [e for e in building.events if e["event"] == "portal_selected"]
    assert selected[-1]["solver_source"] == "rpt_star" and selected[-1]["portal_id"] == placed["id"]
    building.commit(obs, placed, (5.0, 3.0))
    assert len([e for e in building.events if e["event"] == "portal_selected"]) == 1, "idempotent for the same portal"
    approach = building.plan(at(episode, 4, 3.0, 3.0), world)
    assert approach is not None and approach.info["kind"] == "portal/%d" % placed["id"] and taken == []
    foot = at(episode, 5, 4.6, 3.0)
    building.prepare_observation(foot)
    climb = building.plan(foot, world)
    assert building.transition is not None and climb.info["kind"] == "TRAVERSE"
    assert type(building.transition).__name__ == "StairTraversal", "walked on measured tread support: the only traversal there is"
    assert taken == [(5, placed["id"])], "the loop is told the action the climb begins"
    started = [e for e in building.events if e["event"] == "traversal_started"][-1]
    assert started["selected_by"] == "rpt_star" and started["stair_source"] == "observed"


# -- who may say where the stairs are ----------------------------------------------------
def test_observed_is_the_only_stair_source_and_the_simulators_is_refused_not_fallen_back_from():
    assert MultiFloorParams().stair_source == "observed"
    policy, _ = setup_policy("bed", multifloor={"stair_source": "observed"})
    configuration = policy.configuration()
    assert configuration["stair_source"] == "observed" and configuration["ground_truth_stairs"] is False
    assert configuration["floor_change_decision"] == "exploration fallback only"
    with pytest.raises(ValueError, match="removed"):
        RPTSettings(multifloor={"stair_source": "ground_truth"})
    with pytest.raises(ValueError, match="stair_source"):
        RPTSettings(multifloor={"stair_source": "yolo"})
    # The knobs that only the navmesh connectors' traversal read went with it: naming one is an error, not a no-op.
    for knob in ("near_connector_m", "commit_failures", "commit_stall_actions", "confirm_actions"):
        with pytest.raises(TypeError):
            MultiFloorParams(**{knob: 1})
