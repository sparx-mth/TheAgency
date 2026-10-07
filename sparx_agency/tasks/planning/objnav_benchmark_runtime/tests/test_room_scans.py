"""Regressions for the scan visit (2026-10-04): vantage point, full rotation, finished rooms, type priors.

Built on the two-room world of ``test_room_search_loop`` with ``visit="scan"``:
room A on the west, room B on the east, a 1.1 m door between them. The
scene graph is installed by hand, the oracle is the fixture's counter, the
solver is fixed, and A* is real.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_registry import TrackedRoom
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.exploration.object_search_supervisor import EXHAUSTED, SEARCH, SELECT, TRANSIT
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import PeekSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_priors import (
    home_object, implausible_room, ruled_out, target_key)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_scans import (
    SCAN_POINT_INSIDE, SEEN_FROM_SCAN, RoomScanLedger, visible_fraction)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_vantage import vantage_point
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM, observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    FREE, IN_A, IN_B, OCC, RES, UNK, VALUES, NamingLLM, cell_in, loop_policy, obs_at, see, two_room_world)


def scan_policy(order=(1, 0), **overrides):
    return loop_policy(order=order, visit="scan", **overrides)


def turn_through(policy, episode, world, xy, start_step, start_yaw=0.0, max_turns=20):
    """Feed the loop its own turn commands until it stops asking for them; return (commands, last step)."""
    turn = episode.action_spec.turn_angle_rad
    commands, yaw, step = [], start_yaw, start_step
    for _ in range(max_turns):
        command = policy.loop.plan(obs_at(episode, step, xy, yaw), world)
        commands.append(command)
        policy.loop.charge()
        if command.info.get("kind") != "room_scan":
            return commands, step
        yaw = float(command.final_yaw)
        step += 1
    raise AssertionError("the rotation never ended")


# -- the vantage point ---------------------------------------------------------------
def test_the_vantage_point_is_the_interior_cell_of_greatest_clearance():
    world, labels = two_room_world()
    mask = labels == 2
    (x, y), clearance = vantage_point(world, mask)
    gx, gy = world.world_to_grid(x, y)
    assert mask[gy, gx]
    assert 2.0 < clearance <= 3.0, "room B is 5.9 m tall: its middle is nearly three metres from any wall"
    assert 7.0 < x < 11.0 and 2.0 < y < 4.0, "the middle of the open floor, not the doorway or the far frontier"
    assert vantage_point(world, mask, reachable=np.zeros_like(mask)) is None
    assert vantage_point(world, mask, min_clearance_m=10.0) is None
    assert vantage_point(world, np.zeros_like(mask)) is None


# -- the ledger ---------------------------------------------------------------------
def open_hall(width_cells=120):
    """One open hall, no walls inside: labels split it into a west room (1) and a narrow east room (2)."""
    h, w = 60, width_cells
    g = np.full((h, w), FREE, np.int8)
    g[0, :] = g[-1, :] = OCC
    g[:, 0] = g[:, -1] = OCC
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    labels = np.zeros((h, w), np.int32)
    cols = np.arange(w)[None, :]
    free = g == FREE
    labels[free & (cols < 55)] = 1
    labels[free & (cols >= 55) & (cols < 75)] = 2           # x in [5.5, 7.5): two metres wide
    return world, labels


def room_of(labels, label, world):
    mask = labels == label
    ys, xs = np.nonzero(mask)
    return TrackedRoom(id=label - 1, mask=mask, n_cells=int(mask.sum()),
                       centroid=tuple(float(v) for v in world.grid_to_world(int(xs.mean()), int(ys.mean()))))


def test_visible_fraction_runs_through_observed_free_space_only():
    world, labels = open_hall()
    narrow = labels == 2
    assert visible_fraction(world, (4.0, 3.0), narrow, 5.0) > 0.9, "an open hall: the narrow room is in full view"
    assert visible_fraction(world, (4.0, 3.0), narrow, 1.0) == 0.0, "... but not beyond the camera's range"
    walled = OccupancyGrid2D(world.grid.copy(), world.params, values=world.values)
    walled.grid[:, 50:52] = OCC
    assert visible_fraction(walled, (4.0, 3.0), narrow, 5.0) == 0.0, "a wall ends the ray"
    unseen = OccupancyGrid2D(world.grid.copy(), world.params, values=world.values)
    unseen.grid[:, 50:52] = UNK
    assert visible_fraction(unseen, (4.0, 3.0), narrow, 5.0) == 0.0, "so does unknown space: unresolved is unseen"


def test_a_scan_finishes_the_room_it_stood_in_and_the_rooms_it_saw_most_of():
    policy, episode = setup_policy()
    world, labels = open_hall()
    west, narrow = room_of(labels, 1, world), room_of(labels, 2, world)
    ledger = RoomScanLedger(policy, seen_fraction=0.5)
    assert ledger.status(world, 0, west) is None and ledger.status(world, 1, narrow) is None
    ledger.record(obs_at(episode, 5, (4.0, 3.0)), west)
    assert ledger.status(world, 0, west) == SCAN_POINT_INSIDE
    assert ledger.status(world, 1, narrow) == SEEN_FROM_SCAN, "two metres wide, four metres away, nothing in between"
    assert ledger.finished(world, policy.graph) == {}, "the policy's own graph holds neither room"
    assert ledger.records[0]["room"] == 0 and ledger.records[0]["source"] == "room_scan"


def test_the_two_room_world_is_not_finished_through_its_door():
    policy, episode = setup_policy()
    world, labels = two_room_world()
    a, b = room_of(labels, 1, world), room_of(labels, 2, world)
    ledger = RoomScanLedger(policy)
    ledger.record(obs_at(episode, 1, IN_A), a)
    assert ledger.status(world, 0, a) == SCAN_POINT_INSIDE
    assert ledger.status(world, 1, b) is None, "a 1.1 m door shows a sliver of room B, not half of it"


def test_a_finished_verdict_is_sticky_and_floor_local():
    policy, episode = setup_policy()
    world, labels = two_room_world()
    a = room_of(labels, 1, world)
    ledger = RoomScanLedger(policy)
    ledger.record(obs_at(episode, 1, IN_A), a)
    assert ledger.status(world, 0, a) == SCAN_POINT_INSIDE
    shifted = TrackedRoom(id=0, mask=labels == 2, n_cells=int((labels == 2).sum()), centroid=a.centroid)
    assert ledger.status(world, 0, shifted) == SCAN_POINT_INSIDE, "re-segmented away from the point: still finished"
    policy.mapping.floor_id = 1
    assert ledger.status(world, 0, a) is None, "another storey, same pid: a different room"
    assert ledger.on_floor(0) and not ledger.on_floor(1)
    diagnostics = ledger.diagnostics()
    assert diagnostics["finished"] == ["f0/r0:scan_point_inside"] and diagnostics["finished_here"] == []


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_the_ledger_rejects_a_nonsense_seen_fraction(fraction):
    policy, _ = setup_policy()
    with pytest.raises(ValueError):
        RoomScanLedger(policy, seen_fraction=fraction)


def test_a_verdict_is_re_judged_when_the_room_grows_well_past_what_it_was_reached_on():
    """The registry keeps a room's number when the new mask contains the old one, so the sliver of a
    room seen through its door -- finished as a fragment -- can grow into the whole room under the
    same pid. A verdict reached on 1 m2 does not finish 4 m2 unseen; a scan point still inside does."""
    policy, episode = setup_policy()
    world, _ = open_hall()
    ledger = RoomScanLedger(policy)

    def room(x0, x1, y0, y1):
        mask = np.zeros(world.grid.shape, bool)
        mask[y0:y1, x0:x1] = True
        return TrackedRoom(id=7, mask=mask, n_cells=int(mask.sum()), centroid=world.grid_to_world((x0 + x1) // 2, (y0 + y1) // 2))

    sliver = room(20, 30, 20, 30)                                   # 100 cells: 1 m2, no frontier, no door
    assert ledger.status(world, 7, sliver, frontier=0, doored=False) == "fragment"
    grown = room(20, 40, 20, 40)                                    # 400 cells: four times the size
    assert ledger.status(world, 7, grown, frontier=1, doored=True) is None, "the walked-into room is re-judged"
    assert ledger.diagnostics()["regrown"] == [{"floor": 0, "room": 7, "reason": "fragment", "cells_then": 100, "cells_now": 400}]
    assert ledger.status(world, 7, room(20, 30, 20, 30), frontier=0, doored=False) == "fragment", "and judged afresh"
    scanned = RoomScanLedger(policy)
    scanned.record(obs_at(episode, 5, world.grid_to_world(25, 25)), sliver)
    assert scanned.status(world, 7, sliver) == SCAN_POINT_INSIDE
    assert scanned.status(world, 7, grown) == SCAN_POINT_INSIDE, "the scan point lies in the grown room: finished again"
    assert scanned.status(world, 7, room(20, 32, 20, 32)) == SCAN_POINT_INSIDE, "edge wobble is read sticky"
    sticky = RoomScanLedger(policy, regrow_factor=0.0)
    assert sticky.status(world, 7, sliver, frontier=0, doored=False) == "fragment"
    assert sticky.status(world, 7, grown, frontier=1, doored=True) == "fragment", "0 keeps every verdict sticky"
    with pytest.raises(ValueError):
        RoomScanLedger(policy, regrow_factor=0.5)


# -- the visit ----------------------------------------------------------------------
def test_the_transit_aims_at_the_vantage_point_not_a_frontier():
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0))
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    assert command.info["kind"] == "transit/1"
    goal = policy.route_memory.goal
    cost = policy.navigation_cost(world)
    assert goal == policy.loop._vantage(world, cost, rooms[1])[0]
    assert cell_in(world, rooms[1], goal) and goal != rooms[1].centroid
    assert 7.0 < goal[0] < 11.0 and 2.0 < goal[1] < 4.0, "the middle of room B's open floor"
    assert policy.loop.events[-1]["entry"] == "vantage" and policy.loop.stats["entry_vantage"] == 1
    assert policy.loop.estimates[1]["entry"] == "vantage"


def test_a_scan_visit_turns_a_full_circle_at_the_vantage_point_and_finishes_the_room():
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    turn = episode.action_spec.turn_angle_rad
    first = policy.loop.plan(obs_at(episode, 1, vantage), world)        # arrived: inside the room, at the point
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1
    assert first.info["kind"] == "room_scan" and first.info["scan_turn"] == 1
    assert first.waypoints == () and math.isclose(first.final_yaw, turn)
    assert policy.loop._scan["phase"] == "rotate" and not policy.loop._scan["in_place"]
    policy.loop.charge()

    commands, last = turn_through(policy, episode, world, vantage, start_step=2, start_yaw=turn)
    turns = [c for c in commands if c.info.get("kind") == "room_scan"]
    assert len(turns) == 11, "twelve turns of 30 degrees make the circle; the first was emitted on arrival"
    assert [c.info["scan_turn"] for c in turns] == list(range(2, 13))
    assert len(policy.scans.records) == 1 and policy.scans.records[0]["room"] == 1
    assert policy.scans.records[0]["xy"] == tuple(vantage) and policy.scans.records[0]["step"] == last
    assert policy.loop.stats["scans_completed"] == 1 and policy.loop.stats["scans_in_place"] == 0
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED), "finished -- productive, cooled, not repeatable"
    assert policy.loop.events[-4]["event"] == "scan_complete" and policy.loop.events[-4]["turns"] == 12
    assert policy.loop.events[-4]["swept_degrees"] == pytest.approx(360.0, abs=1.0)
    # The loop point on the same action: room B is finished, so it is not a node any more.
    assert reasoned == [0, last]
    assert policy.reasoned_with[-1]["exclude"] == (1,)
    assert policy.loop._excluded == {1: "scanned:" + SCAN_POINT_INSIDE}
    assert policy.loop.estimates[1]["entry"] == "excluded" and policy.loop.estimates[1]["excluded"].startswith("scanned")
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0
    assert commands[-1].info["kind"] == "transit/0"
    assert policy.loop.stats["excluded_scanned"] == 1


def test_a_visit_arriving_short_of_the_vantage_point_walks_to_it_first():
    policy, episode, world, rooms, _ = scan_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    short = (vantage[0] - 1.5, vantage[1])                              # inside room B, 1.5 m west of the point
    assert cell_in(world, rooms[1], short)
    command = policy.loop.plan(obs_at(episode, 1, short), world)
    assert policy.supervisor.state == SEARCH and policy.loop._scan["phase"] == "approach"
    assert command.info["kind"] == "vantage" and command.waypoints, "the approach is a walk to the point"
    assert math.dist(command.waypoints[-1], vantage) < 0.4
    policy.loop.charge()
    assert policy.loop._scan["approach_actions"] == 1 and policy.loop.stats["scan_approach_actions"] == 1
    turning = policy.loop.plan(obs_at(episode, 2, vantage), world)
    assert turning.info["kind"] == "room_scan" and policy.loop._scan["phase"] == "rotate"
    assert policy.loop.events[-1]["event"] == "scan_rotate" and policy.loop.events[-1]["why"] == "vantage reached"


def test_an_approach_that_runs_out_of_budget_inside_the_room_scans_in_place():
    policy, episode, world, rooms, _ = scan_policy(order=(1, 0), scan_approach_steps=2, scan_visit_steps=20)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    short = (vantage[0] - 1.5, vantage[1])
    policy.loop.plan(obs_at(episode, 1, short), world)
    policy.loop.charge()
    policy.loop.plan(obs_at(episode, 2, short), world)                  # did not move
    policy.loop.charge()
    command = policy.loop.plan(obs_at(episode, 3, short), world)
    assert command.info["kind"] == "room_scan" and policy.loop._scan["in_place"]
    assert policy.loop.stats["scans_in_place"] == 1
    assert "budget" in policy.loop.events[-1]["why"]


def test_a_visit_that_overruns_its_bound_is_released_budget_spent():
    policy, episode, world, rooms, _ = scan_policy(order=(1, 0), scan_approach_steps=2, scan_visit_steps=3)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, vantage), world)
    for _ in range(3):
        policy.loop.charge()
    assert policy.loop.visit_budget() == 3
    policy.loop.plan(obs_at(episode, 2, vantage, yaw=0.5), world)
    assert policy.supervisor.history[-1][0] == 1 and policy.supervisor.history[-1][1] == "budget_spent"
    assert policy.loop.stats["budget_releases"] == 1
    assert policy.scans.records == [], "an unfinished rotation is not a scan"
    assert 1 not in policy.loop._excluded, "... and the room is not finished by it"


def test_a_clue_during_a_visit_is_kept_for_the_loop_point_not_acted_on_mid_visit():
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, vantage), world)
    assert policy.supervisor.state == SEARCH
    policy.loop.reconsider(obs_at(episode, 2, vantage), world)
    assert reasoned == [0], "no reasoning round mid-visit"
    assert policy.supervisor.state == SEARCH and policy.supervisor.room_id == 1, "the visit goes on"
    assert policy.loop.stats["reconsiders_deferred"] == 1 and policy.loop._needs_reason
    assert policy.loop.events[-1]["event"] == "reconsider_deferred"


def test_the_warmup_scan_finishes_the_spawn_room_before_the_first_room_is_chosen():
    policy, episode, world, rooms, reasoned = scan_policy(order=(0, 1))     # the solver would prefer room A
    policy.scans.record(obs_at(episode, 12, IN_A), rooms[0], source="warmup")
    command = policy.loop.plan(obs_at(episode, 12, IN_A), world)
    assert policy.loop._excluded == {0: "scanned:" + SCAN_POINT_INSIDE}
    assert policy.reasoned_with[-1]["exclude"] == (0,)
    assert policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1", "room A is not a node; room B is next"
    assert policy.loop.estimates[0]["entry"] == "excluded"


def test_a_room_whose_type_cannot_hold_the_target_is_not_a_node_unless_the_target_was_seen_in_it():
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=llm)
    assert policy.target.query == "chair" and target_key(policy.target) == "chair"
    see(policy, {1: ["toilet", "sink"]}, step=0)                     # two kinds of object: a STRONG bathroom
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["label"] == "bathroom" and policy.graph.label_info(1)["strength"] == "strong"
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {1: "type:bathroom"} and policy.loop.stats["excluded_type"] == 1
    assert policy.supervisor.room_id == 0, "a chair is not searched for in a bathroom"
    assert command.info["kind"] in ("vantage", "room_scan"), "... and the agent already stands in room A, so its visit begins"
    assert policy.loop.estimates[1]["excluded"] == "type:bathroom"
    # A chair confirmed in the bathroom outranks the prior.
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    see(policy, {1: ["toilet", "sink", "chair"]}, step=0)
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["label"] == "bathroom"
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {} and policy.supervisor.room_id == 1
    # A weak label -- one kind of object that names no room on its own -- is the oracle's to value,
    # never an exclusion (2026-10-05): a sink alone may be a kitchen's.
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    see(policy, {1: ["sink"]}, step=0)
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["strength"] == "weak"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {} and policy.loop.stats["weak_type_kept"] == 1
    # One SIGNATURE object names its room on its own (2026-10-05): a toilet is a bathroom, and the
    # bathroom is ruled out for a chair with no second kind of object needed.
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    see(policy, {1: ["toilet"]}, step=0)
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["strength"] == "strong" and policy.graph.label_info(1)["signature"]
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {1: "type:bathroom"} and policy.supervisor.room_id == 0
    assert ruled_out(policy.target, "bathroom", "weak", ["sink"]) is None
    assert ruled_out(policy.target, "bathroom", "strong", ["toilet", "sink"]) == "type:bathroom"
    assert ruled_out(policy.target, "bathroom", "strong", ["toilet", "sink", "chair"]) is None, "the target itself"
    assert ruled_out(policy.target, "bathroom", "strong", ["toilet", "sink", "desk"]) is None, "a desk is where chairs live"
    assert home_object("toilet", "sink") and home_object("toilet", "shower") and not home_object("toilet", "sofa")


def test_the_type_prior_table_is_exclusions_not_permissions():
    assert implausible_room("sofa", "bedroom") and implausible_room("couch", "kitchen")
    assert not implausible_room("sofa", "hallway"), "a hallway leads on"
    assert not implausible_room("sofa", "unknown") and not implausible_room("sofa", None)
    assert implausible_room("toilet", "bedroom") and not implausible_room("toilet", "bathroom")
    assert not implausible_room("frying pan", "bathroom"), "a target the table does not know is never excluded"
    assert target_key("tv") == "television" and target_key("plant") == "potted plant"


def test_a_room_holding_a_home_object_of_the_target_is_not_finished_by_sight_from_outside():
    """Allensville toilet run, step 26: the warm-up spin at the spawn point saw more than half of the
    bathroom's floor through its door (``seen_from_scan``), finished it with the bathtub inside and the
    toilet behind the jamb, and the strong bathroom read 0 for the toilet while twenty openings were
    peeked. A scan the agent STOOD in still finishes it."""
    from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_scans import SEEN_THROUGH
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    policy.target = gibson_label_mapper().target_labels("toilet")
    see(policy, {1: ["bathtub"]}, step=0)
    policy.scans.mark(1, SEEN_FROM_SCAN)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert 1 not in policy.loop._excluded, "a bathtub is where a toilet lives: the room stays a node"
    kept = [e for e in policy.loop.events if e["event"] == "home_object_kept" and "scanned" in e]
    assert policy.loop.stats["home_object_kept"] >= 1 and len(kept) == 1
    assert kept[0]["scanned"] == SEEN_FROM_SCAN and kept[0]["home_objects"] == ["bathtub"]
    assert policy.loop.estimates[1]["entry"] != "excluded", "... and it is offered to the solver"
    # Looked into with no frontier left: the same guard.
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    policy.target = gibson_label_mapper().target_labels("toilet")
    see(policy, {1: ["sink"]}, step=0)
    policy.scans.mark(1, SEEN_THROUGH)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert 1 not in policy.loop._excluded
    # Stood in and turned a full circle: finished, bathtub or not.
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    policy.target = gibson_label_mapper().target_labels("toilet")
    see(policy, {1: ["bathtub"]}, step=0)
    policy.scans.mark(1, SCAN_POINT_INSIDE)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {1: "scanned:" + SCAN_POINT_INSIDE} and policy.loop.stats["home_object_kept"] == 0
    # A room with no home object seen from outside is finished, as before.
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    policy.target = gibson_label_mapper().target_labels("toilet")
    see(policy, {1: ["coat rack"]}, step=0)
    policy.scans.mark(1, SEEN_FROM_SCAN)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {1: "scanned:" + SEEN_FROM_SCAN}
    assert LoopSettings().home_floor == pytest.approx(0.60)
    with pytest.raises(ValueError):
        LoopSettings(home_floor=1.0)


def test_a_room_whose_evidence_rules_it_out_on_this_action_is_excluded_on_this_action():
    """Hanson 2026-10-05, action 150: the bedroom's label became strong at the loop point's own
    re-classification, AFTER the nodes had been chosen; the oracle was shown it and valued it at 0.10
    ("bedroom, fully seen, no toilet"), and it stayed in the order until the next loop point. The
    labels are read from the evidence before the nodes are chosen now."""
    llm = NamingLLM()
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=llm)
    see(policy, {1: ["toilet"]}, step=0)             # the background refresh: evidence recorded, no label yet
    assert policy.graph.label_info(1) is None
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert llm.calls == 1, "one classifier call for the one room with a new kind of object"
    assert policy.graph.label_info(1)["label"] == "bathroom" and policy.graph.label_info(1)["strength"] == "strong"
    assert policy.loop._excluded == {1: "type:bathroom"}
    assert policy.reasoned_with[-1]["exclude"] == (1,), "withheld from the oracle on the same action"
    assert policy.supervisor.room_id == 0 and command.info["kind"] in ("vantage", "room_scan")
    assert policy.loop.estimates[1]["excluded"] == "type:bathroom"


def test_the_room_in_force_is_never_excluded_mid_visit():
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, vantage), world)
    policy.scans.record(obs_at(episode, 1, vantage), rooms[1])       # as if a scan had stood here already
    policy.loop._exclusions(obs_at(episode, 2, vantage), world)
    assert 1 not in policy.loop._excluded, "its turn ends by the visit's own rule"


def test_a_relabel_mid_scan_keeps_the_visit_unless_the_new_type_rules_the_room_out():
    """Upstairs Ranchester: a potted plant renamed the room 'living room' at 180 degrees of its
    rotation and the turn ended -- half a scan wasted. A toilet, on the other hand, means leave now."""
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    vantage = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, vantage), world)
    assert policy.supervisor.state == SEARCH and policy.loop._scan["phase"] == "rotate"
    see(policy, {1: ["coat rack"]}, step=2)                          # NamingLLM: a hallway -- fine for a chair
    command = policy.loop.plan(obs_at(episode, 2, vantage, yaw=0.5), world)
    assert policy.graph.label_info(1)["label"] == "hallway"
    assert policy.supervisor.state == SEARCH and policy.supervisor.room_id == 1, "the rotation goes on"
    assert command.info["kind"] == "room_scan" and policy.loop._needs_reason
    assert policy.loop.stats["relabels_kept_visit"] == 1 and policy.loop.stats["reclassified_releases"] == 0
    assert policy.loop.events[-1]["event"] == "relabel_kept_visit"
    see(policy, {1: ["coat rack", "toilet"]}, step=3)                # ... but a bathroom is no place for a chair
    command = policy.loop.plan(obs_at(episode, 3, vantage, yaw=1.0), world)
    assert policy.graph.label_info(1)["label"] == "bathroom"
    assert policy.supervisor.history[-1][:2] == (1, "reclassified")
    assert policy.loop._excluded.get(1) == "type:bathroom" and policy.supervisor.room_id == 0
    assert policy.loop.stats["reclassified_releases"] == 1


def test_a_relabel_in_transit_under_scan_keeps_the_transit_unless_ruled_out():
    policy, episode, world, rooms, reasoned = scan_policy(order=(1, 0), llm=NamingLLM())
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1
    see(policy, {1: ["bed"]}, step=1)                                # a bedroom may hold a chair
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.graph.label_info(1)["label"] == "bedroom"
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1"
    assert reasoned == [0], "no loop point mid-transit for a name that does not rule the room out"
    assert policy.loop.stats["relabels_kept_visit"] == 1


# -- the knobs -----------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{"visit": "wander"}, {"scan_approach_steps": 40}, {"vantage_arrival_m": 0.0},
                                    {"scan_seen_fraction": 0.0}, {"scan_seen_fraction": 1.5}, {"type_prior": 1},
                                    {"min_prob": 1.0}, {"vantage_min_clearance_m": -1.0}])
def test_scan_settings_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        LoopSettings(**kwargs)


def test_scan_settings_stand_the_supervisors_own_done_tests_down():
    params = LoopSettings().supervisor_params(seed=1)
    assert params.min_frontier_clusters == -1, "a frontier count of zero never finishes a room; the rotation does"
    assert params.frontier_stall_s >= 3 * LoopSettings().scan_visit_steps
    assert params.search_timeout_s >= 3 * LoopSettings().scan_visit_steps
    assert params.min_prob == 0.05
    sweep = LoopSettings(visit="sweep").supervisor_params(seed=1)
    assert sweep.min_frontier_clusters == 0 and sweep.frontier_stall_s == 30.0
    assert LoopSettings().service_steps() == 22 and LoopSettings(visit="sweep").service_steps() == 10


def test_the_policy_reports_the_scan_visit_and_the_open_floor_exit():
    policy, _ = setup_policy()
    configuration = policy.configuration()
    assert configuration["room_search_loop"]["visit"] == "scan"
    assert "full rotation" in configuration["room_visit"]
    assert configuration["floor_exit_gate"].startswith("none")
    assert "stairs are never a room" in configuration["room_partition"]
    assert policy.settings.doorway_peek.gate_floor_departure is False
    assert policy.settings.doorway_peek.enabled is False, "peeks are off under the scan visit"
    assert policy.settings.warmup_steps == 12
    info = policy.episode_info()
    assert "room_scans" in info
    assert "room_search_loop" in info["building"]["floor_contexts"][0], "each storey keeps its loop's record"


# -- stairs are not a room ------------------------------------------------------------
def test_excluded_cells_belong_to_no_room():
    world, _ = open_hall()
    graph = ObservedSceneGraph(FakeLLM())
    graph.update(world, [], None, step=0, reason=False)
    assert len(graph.registry.rooms) >= 1
    assert (graph.labels[4:-4, 4:-4] > 0).all(), "every interior free cell is somebody's room"
    exclude = np.zeros(world.grid.shape, bool)
    exclude[:, 60:66] = True                                           # a 0.6 m band, as a stair footprint would be
    graph = ObservedSceneGraph(FakeLLM())
    graph.update(world, [], None, step=0, reason=False, exclude=exclude)
    assert (graph.labels[exclude] == 0).all(), "a staircase is a connector, not a room"
    assert (graph.labels[4:-4, 4:56] > 0).all() and (graph.labels[4:-4, 70:-4] > 0).all()
    assert all(not room.mask[exclude].any() for room in graph.registry.rooms.values())


def test_the_policy_excludes_seen_stair_footprints_from_the_partition(monkeypatch):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import stair_policy, STAIRS
    policy, episode, world, rooms, _ = stair_policy(order=(STAIRS, 1, 0))
    exclusion = policy.room_exclusion(world)
    assert exclusion is not None and exclusion.any(), "the seen flight's foot is on this floor"
    gx, gy = world.world_to_grid(4.0, 1.0)
    assert exclusion[gy, gx]
    seen = {}
    monkeypatch.setattr(policy.graph, "update", lambda *a, **k: seen.update(k))
    policy._last_graph_step = -5
    policy._refresh_graph(obs_at(episode, 3, IN_A), world)
    assert seen["exclude"] is not None and seen["exclude"].any()
    policy_without_stairs, _ = setup_policy()
    assert policy_without_stairs.room_exclusion(world) is None


def test_the_floor_departure_gate_is_off_by_default_and_opt_in():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import stair_policy, STAIRS
    policy, episode, world, rooms, _ = stair_policy(order=(STAIRS, 1, 0))
    policy.peek.floor_ready = lambda: False                            # rooms still "pending" by the peek ledger
    assert policy.building.can_leave_floor(obs_at(episode, 0, IN_A)) is True
    assert not any(e.get("event") == "floor_change_held" for e in policy.building.events)
    from dataclasses import replace
    policy.settings = replace(policy.settings, doorway_peek=PeekSettings(gate_floor_departure=True))
    assert policy.building.can_leave_floor(obs_at(episode, 0, IN_A)) is False
    assert policy.building.events[-1]["event"] == "floor_change_held"
    with pytest.raises(ValueError):
        PeekSettings(gate_floor_departure="yes")


def test_the_way_back_is_withheld_while_this_storey_still_has_rooms_to_scan():
    """Ranchester attempt 3: the 3B sent the agent back up the stairs it had just come down, twice,
    with unscanned rooms on the list each time. The storey is looked at before the way back is a choice."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import stair_policy, STAIRS
    policy, episode, world, rooms, reasoned = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.9, visit="scan")
    building = policy.building
    building.arrived_by, building.arrived_step = 0, -1000             # came down these stairs long ago: the grace is spent
    obs = obs_at(episode, 0, IN_A)
    cost = policy.navigation_cost(world)
    policy.loop._exclusions(obs, world)
    assert policy.loop._excluded == {}, "both rooms unscanned"
    assert policy.loop._stair_options(obs, world, cost) == [], "the way back is not a node yet"
    assert policy.loop.stats["way_back_held_for_rooms"] == 1 and policy.loop.events[-1]["event"] == "way_back_held"
    assert policy.loop.events[-1]["rooms_left"] == [0, 1]
    policy.scans.record(obs, rooms[0], source="room_scan")
    policy.scans.record(obs_at(episode, 1, IN_B), rooms[1], source="room_scan")
    policy.loop._exclusions(obs_at(episode, 2, IN_A), world)
    assert set(policy.loop._excluded) == {0, 1}
    [option] = policy.loop._stair_options(obs_at(episode, 2, IN_A), world, cost)
    assert option.node_id == STAIRS and option.node.arrived_by, "every room finished: the way back is offered, marked"


def test_a_warmup_moved_mid_rotation_starts_its_circle_again():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.discovery import discover
    p, episode = setup_policy("bed", discovery=True)
    world = p.mapping.update(observation(episode, 0, depth=3))
    cost = p.navigation_cost(world)
    for step in range(4):
        obs = observation(episode, step, depth=3)
        assert discover(p, obs, world, cost).info["kind"] == "warmup"
        p.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert p.warmup_actions == 4 and p._warmup_xy == (0.0, 0.0)
    moved = replace(observation(episode, 4, depth=3), pose=AgentPose(2.0, 0.5, 0.0, 0.0))
    command = discover(p, moved, world, cost)
    assert command.info["kind"] == "warmup" and command.info["warmup_step"] == 1, "four turns elsewhere do not count here"
    assert p.warmup_actions == 0 and p._warmup_xy == (2.0, 0.5)
    nearby = replace(observation(episode, 5, depth=3), pose=AgentPose(2.2, 0.6, 0.0, 0.5))
    p.notify_action(moved, DiscreteAction.TURN_LEFT)
    assert discover(p, nearby, world, cost).info["warmup_step"] == 2, "a 0.2 m drift is the same spot"


