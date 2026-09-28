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
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, MultiFloorParams, ObservedFloor
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_decision import (
    DecisionSettings, decide_floor_change, floor_at, floor_worth_visiting)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
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
    return setup_policy("bed", metadata=STRUCTURE, **dict({"map_size_m": 40.0}, **overrides))


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


def test_arrival_at_the_top_anchor_settles_the_new_floor_and_cools_the_way_back():
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
    assert upstairs[0]["cooldown_until"] >= step - 1 + building.params.floor_search_actions - 1, "searched before descending"
    assert policy.mapping.floor_id == 1 and policy.floors.active == 1


def test_the_exhausted_floor_takes_the_stairs_and_the_choice_is_recorded():
    policy, episode = gt_policy()
    world = hall()
    obs = at(episode, 0, 3.0, 3.0)
    policy.plan(obs)
    building = policy.building
    command = building.plan(at(episode, 1, 3.0, 3.0), world, exhausted=True)
    assert building.active is not None and building.phase == "APPROACH_STAIRS"
    assert command is not None and command.waypoints and command.info["kind"] == "portal/0"
    decision = [e for e in building.events if e["event"] == "floor_decision"]
    selected = [e for e in building.events if e["event"] == "portal_selected"]
    assert len(decision) == 1 and decision[0]["trigger"] == "floor_exhausted" and decision[0]["chosen"] == 0
    assert selected and selected[0]["direction"] == "up" and selected[0]["solver_source"] == "floor_decision"
    # Standing at the entry anchor commits the climb on the connector's own polyline.
    command = building.plan(at(episode, 2, 4.6, 3.0), world)
    assert building.transition is not None and command.info["kind"] == "TRAVERSE"
    assert list(command.waypoints[0]) == pytest.approx(FLIGHT[0][:2]) or list(command.waypoints[-1]) == pytest.approx(FLIGHT[-1][:2])


def test_no_look_down_inspections_and_no_repeated_decision_records_in_ground_truth_mode():
    policy, episode = gt_policy()
    world = hall(unknown_east_of=3.0)                      # the entry anchor at x=5 is 2 m into unknown space
    policy.plan(at(episode, 0, 1.5, 3.0))
    building = policy.building
    for step in range(200, 206):                            # allowance spent
        assert building.plan(at(episode, step, 1.5, 3.0), world) is None
    assert policy.camera_control.inspection is None
    decisions = [e for e in building.events if e["event"] == "floor_decision"]
    assert [d["trigger"] for d in decisions] == ["allowance_spent"] * 2, "one record per distinct verdict, not per action"
    assert "re-check" in decisions[0]["candidates"][0]["verdict"]
    assert decisions[1]["reason"] == "no eligible connector on this floor"       # the re-check cooldown
    assert building.diagnostics()["stair_source"] == "ground_truth"
    assert building.diagnostics()["ground_truth"]["connectors"][0]["top_z"] == pytest.approx(2.7)


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
    for step in range(2, 2 + 18):
        command = t.plan(replace(stuck, step=step))
        assert command.waypoints and not command.stop
    assert t.phase == "RETREAT" and t.retreat_path is not None
    assert [e for e in building.events if e["event"] == "retreat_started"][0]["reason"] == "stalled"
    assert list(t.retreat_path[-1]) == pytest.approx(FLIGHT[0]), "back to the entry anchor"
    for step in range(20, 20 + building.params.retreat_actions + 1):
        command = t.plan(replace(stuck, step=step))
        assert command.waypoints and not command.stop, "no SAFE_HALT: the search goes on"
    assert t.arrival_allowed and t.completion_reason == "ground_truth_retreat_budget_exhausted"


def test_blocked_steps_skip_a_corner_then_turn_the_climb_back():
    policy, episode = gt_policy()
    building, t = committed(policy, episode)
    obs = at(episode, 1, 5.0, 3.0, z=0.0)
    t.observe(obs)
    before = t.cursor
    building.blocked(obs)
    assert t.failures == 1 and t.cursor == before + 1
    for _ in range(building.params.max_failures - 1):
        building.blocked(obs)
    command = t.plan(replace(obs, step=2))
    assert t.phase == "RETREAT" and command.waypoints
    assert [e for e in building.events if e["event"] == "retreat_started"][0]["reason"] == "blocked"


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




