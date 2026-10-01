"""One-time room peeks cannot steal stair control or require forbidden retries."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_classifier import RoomLabel
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import PeekSettings, doorway_candidates
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import peek_planning_world, stair_peek_mask
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_discovery import peek_rig, refresh
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_coverage import coverage_rig
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import IN_A, obs_at


def stair_at_door(p):
    p.mapping.atlas.update(SimpleNamespace(x=0, y=0, z=0))
    p.building.portals = [{"id": 0, "floor_id": 0, "entry": [6.0, 3.0, 0.0],
                           "path": [(6.0, 2.5, 0.0), (6.0, 3.5, -0.4), (6.0, 5.0, -2.0)]}]


def test_weak_initial_label_exempts_room_without_scan_or_attempt():
    p, ep, world, _, _ = peek_rig()
    p.graph.label_tracker.labels[1] = RoomLabel("bedroom", 0.15, "one bed through doorway")
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    assert p.peek.plan(obs, world) is None
    assert p.peek.floor_ready()
    record = p.peek.records[-1]
    assert record["classified"] and not record["done"] and record["attempts"] == 0
    p.graph.label_tracker.labels[1] = RoomLabel("unknown", 0, "partition changed")
    assert p.peek.plan(replace(obs, step=20), world, force=True) is None
    assert p.peek.floor_ready(), "an initial classification exemption is not refunded"


def test_one_confirmed_object_is_classified_before_starting_a_peek():
    p, ep, world, _, _ = peek_rig()
    p.graph.label_tracker.update({0: [], 1: ["bed"]}, 1, allow_query=False)
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    assert p.peek.plan(obs, world) is None
    assert p.graph.label_info(1)["label"] != "unknown"
    assert p.graph.label_info(1)["strength"] == "weak"
    assert not [e for e in p.peek.events if e["event"] == "peek_started"]


@pytest.mark.parametrize("reason", ["scan_budget", "entry_unreachable", "room_disappeared", "target_priority"])
def test_cancelled_attempt_is_never_repeated_even_when_forced(reason):
    p, ep, world, _, _ = peek_rig()
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    p.peek.plan(obs, world)
    assert p.peek.active is not None
    p.peek.cancel(obs, reason)
    assert p.peek.records[-1]["room_peeked"] and not p.peek.records[-1]["done"]
    for step in (2, 100, 1000):
        assert p.peek.plan(replace(obs, step=step), world, force=True) is None
    assert p.peek.floor_ready()
    assert len([e for e in p.peek.events if e["event"] == "peek_started"]) == 1


def test_initial_classification_interrupts_active_peek_without_refunding_it():
    p, ep, world, _, _ = peek_rig()
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    p.peek.plan(obs, world)
    p.graph.label_tracker.labels[1] = RoomLabel("bathroom", .2, "one sink")
    assert p.peek.plan(replace(obs, step=2), world) is None
    assert p.peek.active is None and p.peek.events[-1]["reason"] == "classified"
    assert p.peek.records[-1]["room_peeked"] and p.peek.floor_ready()


def test_split_and_renumbered_regions_cannot_buy_a_second_attempt():
    p, ep, world, rooms, _ = peek_rig()
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    p.peek.plan(obs, world)
    p.peek.cancel(obs, "room_disappeared")
    original = p.graph.registry.rooms.pop(1)
    for pid, lo, hi in ((7, 66, 90), (8, 90, 120)):
        mask = original.mask.copy()
        mask[:, :lo] = False
        mask[:, hi:] = False
        p.graph.registry.rooms[pid] = replace(original, id=pid, mask=mask)
    assert not p.peek.pending_rooms()
    assert p.peek.plan(replace(obs, step=80), world, force=True) is None
    p.mapping.floor_id = 1
    assert set(p.peek.pending_rooms()) == {0, 7, 8}, "no cross-floor exemption"


def test_expansion_and_id_change_do_not_refund_consumed_peek():
    p, ep, world, rooms, _ = peek_rig()
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    p.peek.plan(obs, world)
    p.peek.cancel(obs, "scan_budget")
    original = p.graph.registry.rooms.pop(1)
    mask = original.mask.copy()
    mask[:, 55:] = True
    p.graph.registry.rooms[17] = replace(original, id=17, mask=mask)
    assert 17 not in p.peek.pending_rooms()
    separate = np.zeros_like(mask)
    separate[1:20, 1:10] = True
    p.graph.registry.rooms[18] = replace(original, id=18, mask=separate)
    assert 18 in p.peek.pending_rooms()


@pytest.mark.parametrize("mode", ["approach", "traverse", "atlas_transition"])
def test_peek_cannot_preempt_any_stair_transaction(mode):
    p, ep, world, _, _ = peek_rig()
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    p.peek.plan(obs, world)
    if mode == "approach":
        p.building.active = {"id": 0}
    elif mode == "traverse":
        p.building.transition = SimpleNamespace(phase="TRAVERSE")
    else:
        p.mapping.atlas.in_transition = True
    p._route = [(4, 0), (4, 1)]
    assert p.peek.plan(replace(obs, step=2), world, force=True) is None
    assert p.peek.active is None and p._route == [(4, 0), (4, 1)]
    assert p.peek.events[-1]["reason"] == "exclusive_navigation"


def test_peek_path_cannot_cross_seen_stairs_even_to_a_nonstair_goal():
    p, ep, world, _, _ = peek_rig()
    stair_at_door(p)
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    candidates = doorway_candidates(p, obs, world, p.peek.settings, room_ids={1})
    assert candidates, "room itself is off the stairs"
    assert p.peek.plan(obs, world) is None, "only path to room crosses excluded connector"
    assert p.peek.events[-1]["reason"] == "entry_unreachable"
    assert p.peek.records[-1]["room_peeked"]
    assert p.peek.plan(replace(obs, step=80), world) is None


def test_stair_mask_is_height_local_and_never_changes_navigation_map():
    p, ep, world, _, _ = peek_rig()
    stair_at_door(p)
    before = world.grid.copy()
    mask = stair_peek_mask(p, world)
    assert mask.any()
    guarded = peek_planning_world(p, world, mask)
    assert np.all(guarded.grid[mask] == world.values.occupied)
    np.testing.assert_array_equal(world.grid, before)
    p.building.portals[0]["path"] = [(6, 2.5, 2.5), (6, 3.5, 2.9)]
    assert not stair_peek_mask(p, world).any(), "overhead flight must not exclude a ground-floor room"


def test_stair_only_region_is_not_a_pending_room():
    p, ep, world, rooms, _ = peek_rig()
    stair_at_door(p)
    mask = np.zeros_like(rooms[1].mask)
    mask[25:36, 56:65] = True
    p.graph.registry.rooms = {7: replace(rooms[1], id=7, mask=mask)}
    p.last_world = world
    assert p.peek.pending_rooms() == [] and p.peek.floor_ready()


def test_vetoed_room_transit_is_released_instead_of_replanned_forever():
    p, ep, world, _ = coverage_rig()
    p.loop.plan(obs_at(ep, 0, IN_A), world)
    obs = obs_at(ep, 1, (4.0, .9), math.pi / 2)
    assert p.supervisor.state == "transit"
    assert p.filter_action(obs, DiscreteAction.MOVE_FORWARD) == DiscreteAction.TURN_LEFT
    assert p.supervisor.state == "select" and p.loop._needs_reason


def test_peek_attempt_limit_cannot_be_relaxed_by_configuration():
    assert PeekSettings().max_attempts == 1
    with pytest.raises(ValueError, match="at most once"):
        PeekSettings(max_attempts=2)

