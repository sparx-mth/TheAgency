"""One-time unknown-room peek eligibility gates floor choices without demanding retries."""
from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import PeekSettings, doorway_candidates
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_peek import DoorwayPeek
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_nodes import stair_options
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_discovery import peek_rig, refresh
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_doorway_actions import moved
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import IN_A, STAIRS, obs_at, stair_policy


def coverage_rig():
    p, ep, world, rooms, _ = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.99)
    p.settings = replace(p.settings, doorway_peek=PeekSettings())
    p.peek = DoorwayPeek(p)  # the real, initially empty ledger; no fixture readiness shortcut
    p.graph.refresh_accessibility(world, p.navigation_cost(world), IN_A)
    return p, ep, world, rooms


def scanned(p, world, pid, xy):
    """Install explicit scan evidence for focused guard/identity tests."""
    gx, gy = world.world_to_grid(*xy)
    p.peek.records.append(dict(floor=p.mapping.floor_id, room=pid, mask=p.graph.registry.rooms[pid].mask.copy(),
                               done=True, attempts=1, retry=0, scan_cell=[gy, gx], scan_xy=list(xy),
                               reason="scanned", swept_degrees=180.0))


def test_high_stair_probability_and_no_frontiers_do_not_waive_room_visits():
    p, ep, world, rooms = coverage_rig()
    obs = obs_at(ep, 0, IN_A)
    cost = p.navigation_cost(world)
    p.graph.probs = {pid: 0.0 for pid in rooms}
    p.graph.facts = {pid: replace(fact, frontier_clusters=0) for pid, fact in p.graph.facts.items()}
    p.peek._remember_start(obs, world)
    assert p.peek.pending_rooms() == [0, 1]
    assert not p.building.commit(obs, p.building.portals[0], (4.0, 1.0))
    assert stair_options(p.building, obs, world, cost, p.floors.save()) == []
    assert p.building.plan(obs, world, exhausted=True) is None
    scanned(p, world, 1, (9.0, 3.0))
    assert not p.building.can_leave_floor(obs), "spawn room also needs a scan"
    p.peek.records = [r for r in p.peek.records if r["room"] != 0]
    scanned(p, world, 0, IN_A)
    assert p.peek.floor_ready()
    assert stair_options(p.building, obs, world, cost, p.floors.save())
    assert p.building.commit(obs, p.building.portals[0], (4.0, 1.0))


def test_floor_without_geometry_is_incomplete_but_attempted_rooms_are_exempt():
    p, ep, world, rooms = coverage_rig()
    p.graph.registry.rooms = {}
    assert not p.peek.floor_ready()
    p.graph.registry.rooms = rooms
    scanned(p, world, 0, IN_A)
    p.peek.records.append(dict(floor=0, room=1, mask=rooms[1].mask.copy(), done=False,
                               attempts=1, room_peeked=True, retry=999, reason="entry_unreachable"))
    assert p.peek.pending_rooms() == []
    assert p.building.can_leave_floor(obs_at(ep, 500, IN_A))
    assert not p.peek.records[-1]["done"], "exempt does not mean falsely certified scanned"


def test_a_new_room_revokes_an_already_selected_stair_approach():
    p, ep, world, rooms = coverage_rig()
    for pid, xy in ((0, IN_A), (1, (9.0, 3.0))):
        scanned(p, world, pid, xy)
    obs = obs_at(ep, 1, IN_A)
    assert p.building.commit(obs, p.building.portals[0], (4.0, 1.0))
    mask = np.zeros_like(rooms[1].mask)
    mask[2:20, 1:10] = True  # new region, not a resegmented child of a consumed room
    p.graph.registry.rooms[7] = replace(rooms[1], id=7, mask=mask, n_cells=int(mask.sum()))
    assert p.building.plan(obs_at(ep, 2, (4.0, 1.0)), world) is None
    assert p.building.active is None and p.building.transition is None
    assert p.peek.pending_rooms() == [7]


def test_visit_evidence_does_not_leak_across_floors_or_reused_room_ids():
    p, ep, world, rooms = coverage_rig()
    scanned(p, world, 0, IN_A)
    assert p.peek.pending_rooms() == [1]
    p.mapping.floor_id = 1
    assert p.peek.pending_rooms() == [0, 1]
    p.mapping.floor_id = 0
    assert p.peek.pending_rooms() == [1]
    p.graph.registry.rooms[0] = replace(rooms[0], mask=rooms[1].mask.copy())
    assert p.peek.pending_rooms() == [0, 1], "same ID, different region is not a visit"


def test_repartition_does_not_grant_another_peek_to_a_consumed_room():
    p, ep, world, rooms = coverage_rig()
    scanned(p, world, 1, (9.0, 3.0))
    # Most of the parent's mask survives, but it does not contain the scan pose.
    mask = rooms[1].mask.copy()
    gx, gy = world.world_to_grid(9.0, 3.0)
    mask[gy - 3:gy + 4, gx - 3:gx + 4] = False
    p.graph.registry.rooms[1] = replace(rooms[1], mask=mask)
    assert 1 not in p.peek.pending_rooms()


def test_real_converter_goes_deeper_and_finishes_a_semicircle_before_floor_ready():
    p, ep, world, _, _ = peek_rig()
    p.graph.doors = [{"xy": [6.6, 3.0], "rooms": [0, 1]}]
    converter = DiscreteActionConverter(ep.action_spec, p.converter_params)
    pose = obs_at(ep, 1, (5.0, 3.0)).pose
    scanned_poses = []
    for step in range(1, 100):
        obs, _ = refresh(p, ep, world, step, (pose.x, pose.y), pose.yaw)
        command = p.peek.plan(obs, world)
        if p.peek.active is None:
            break
        assert not p.peek.floor_ready()
        if command.info["kind"] == "doorway_peek_scan":
            scanned_poses.append(pose)
        decision = converter.step(pose, command)
        action = DiscreteAction.TURN_LEFT if decision.idle else decision.action
        p.peek.charge()
        pose = moved(obs, action, ep.action_spec)
    else:
        pytest.fail("interior scan did not complete")
    assert scanned_poses and min(q.x for q in scanned_poses) > 7.6, "not at the x=6.6 doorway"
    assert p.peek.floor_ready()
    assert p.peek.events[-1]["swept_degrees"] >= 180 - 1e-6


def test_the_spawn_room_is_scanned_not_auto_credited():
    p, ep, world, _ = coverage_rig()
    obs, _ = refresh(p, ep, world, 0, IN_A)
    command = p.peek.plan(obs, world)
    assert command is not None and p.peek.active["room"] == 0
    assert not p.peek.floor_ready() and not p.peek.records[0]["done"]


def test_small_room_uses_its_deepest_observed_safe_viewpoint():
    p, ep, world, rooms, _ = peek_rig()
    mask = rooms[1].mask.copy()
    mask[:, 80:] = False  # only 1.4 m of observed room depth, not enough for the full inset
    p.graph.registry.rooms[1] = replace(rooms[1], mask=mask, n_cells=int(mask.sum()))
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    candidates = doorway_candidates(p, obs, world, p.peek.settings, room_ids={1})
    assert len(candidates) == 1
    goal = candidates[0][2]
    assert 7.2 < goal[0] < 8.0
    gx, gy = world.world_to_grid(*goal)
    assert mask[gy, gx] and np.isfinite(p.graph.frontier_inventory.distance_m[gy, gx])


def test_fallback_finishes_distant_peeks_even_when_opportunistic_peeks_are_disabled(monkeypatch):
    p, ep, world, _, _ = peek_rig()
    p.peek.settings = replace(p.peek.settings, enabled=False)
    obs, cost = refresh(p, ep, world, 1, IN_A)
    assert p.peek.plan(obs, world) is None
    monkeypatch.setattr(p.sweep, "admissible", lambda *args: [])
    command = p.fallback.plan(obs, world, cost)
    assert command.info["fallback_stage"] == "room_peek"
    assert p._action_owner == "doorway_peek" and p.peek.active["room"] == 1


def test_frontier_forward_step_cannot_bypass_gate_but_motion_away_is_allowed():
    p, ep, world, _ = coverage_rig()
    # The flight departs +Y from (4, 1); floor-wide exploration is not committed.
    obs = obs_at(ep, 1, (4.0, 0.9), math.pi / 2)
    assert p.filter_action(obs, DiscreteAction.MOVE_FORWARD) == DiscreteAction.TURN_LEFT
    assert p.building.departure.vetoes == 1
    away = obs_at(ep, 2, (4.0, 0.9), -math.pi / 2)
    assert p.filter_action(away, DiscreteAction.MOVE_FORWARD) == DiscreteAction.MOVE_FORWARD


def test_accidental_tread_entry_retreats_to_source_before_visiting_other_floor():
    p, ep, world, _ = coverage_rig()
    p.building.source_trail = [(4.0, 0.5, 0.0), (4.0, 0.9, 0.0)]
    obs = replace(obs_at(ep, 1, (4.0, 2.0), math.pi / 2), pose=replace(obs_at(ep, 1, (4.0, 2.0)).pose, z=0.9))
    p.building._start_unplanned(obs, 0.0)
    transition = p.building.transition
    assert transition is not None and transition.phase == "RETREAT"
    assert transition.retreat_path[-1] == (4.0, 0.5, 0.0), "clear of the first tread's arrival band"
    assert max(point[2] for point in transition.retreat_path) <= 0.9
    assert not transition.arrival_allowed
    assert p.building.events[-1]["room_coverage_complete"] is False
