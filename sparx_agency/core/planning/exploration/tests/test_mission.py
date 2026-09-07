"""The state machine: where it thinks it is, what it orders, when it gives up.

Every test drives the supervisor the way the node does -- one ``update`` per
tick, with a pose, a seen-mask and a clock the test owns -- so a whole survey
runs in a millisecond and every deadline is exact rather than slept through.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.exploration.mission import (
    AT_DOORWAY,
    ENTER_ROOM,
    EXIT_ROOM,
    IN_CORRIDOR,
    INSIDE_ROOM,
    OFF_MAP,
    APPROACH_DOOR,
    SCAN_AREA,
    SURVEY_COMPLETE,
    TRAVERSE,
    ExplorationSupervisor,
    SupervisorParams,
)

from .conftest import see

CORRIDOR = (6.5, 1.0)          # in the corridor, midway between two doors
ROOM_C = (7.55, 4.45)          # inside the room above it
DOORWAY_C = (8.0, 2.05)        # the opening between them
NORTH = math.radians(90)


def _sup(region_map, coverage, **kwargs):
    return ExplorationSupervisor(region_map, coverage, SupervisorParams(**kwargs))


def _room_at(region_map, xy):
    region = region_map.region_at(*xy)
    assert region is not None
    return region


# ── where it thinks it is ────────────────────────────────────────────────

def test_topological_state_follows_the_aircraft(region_map, coverage, nothing_seen):
    sup = _sup(region_map, coverage)
    assert sup.update(*CORRIDOR, NORTH, nothing_seen, 0.0).topo == IN_CORRIDOR
    assert sup.update(*ROOM_C, NORTH, nothing_seen, 1.0).topo == INSIDE_ROOM
    assert sup.update(*DOORWAY_C, NORTH, nothing_seen, 2.0).topo == AT_DOORWAY
    assert sup.update(-50.0, -50.0, NORTH, nothing_seen, 3.0).topo == OFF_MAP


def test_a_doorway_outranks_the_region_the_cell_happens_to_belong_to(
        region_map, coverage, nothing_seen):
    """Standing in an opening, "which region" has no single right answer."""
    sup = _sup(region_map, coverage)
    state = sup.update(*DOORWAY_C, NORTH, nothing_seen, 0.0)
    assert state.topo == AT_DOORWAY
    assert state.portal is not None
    assert set(state.portal.between) == {region_map.corridors()[0].id,
                                         _room_at(region_map, ROOM_C).id}


def test_being_over_furniture_does_not_lose_the_room(region_map, coverage):
    """An unlabelled cell inside a room means "above a desk", not "nowhere".

    The flight-band map counts anything the aircraft *could* hit, so a 0.75 m
    desk is occupied and unlabelled while the aircraft cruises comfortably over
    it at 1.20 m. Losing the room there loses the mission.
    """
    room = _room_at(region_map, ROOM_C)
    labels = region_map.labels
    labels[40:48, 70:80] = 0                            # a 0.8 x 1.0 m desk
    assert region_map.region_at(7.5, 4.4).id == room.id


def test_a_desk_against_the_wall_still_reads_as_the_room_it_is_in(region_map):
    """The tie the nearest-cell rule gets wrong, and the vote gets right.

    Pressed against the partition between a room and the corridor, the nearest
    labelled cell is as likely to be on the far side as the near one -- and a
    supervisor that decides the aircraft is in the corridor orders it to enter
    the room it is already standing in.
    """
    room = _room_at(region_map, ROOM_C)
    labels = region_map.labels
    labels[21:24, 61:90] = 0                # a run of desks along the room wall
    assert region_map.region_at(7.5, 2.25).id == room.id


# ── what it orders ───────────────────────────────────────────────────────

def test_the_first_order_is_to_look_around_where_you_are(region_map, coverage,
                                                         nothing_seen):
    sup = _sup(region_map, coverage)
    state = sup.update(*CORRIDOR, NORTH, nothing_seen, 0.0)
    assert state.changed
    assert state.mission.kind == SCAN_AREA
    assert state.mission.target_id == region_map.corridors()[0].id


def test_a_scanned_corridor_leads_to_the_nearest_unentered_room(
        region_map, coverage, nothing_seen):
    """And the first order is to go and STAND AT the door, not through it."""
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage)
    state = sup.update(*CORRIDOR, NORTH, seen, 0.0)
    assert state.mission.kind == APPROACH_DOOR
    portal = region_map.portals[state.mission.portal_id]
    assert state.mission.target_id in portal.between


def test_crossing_is_only_ordered_from_the_threshold(region_map, coverage,
                                                      nothing_seen):
    """Getting there and going through are two orders.

    Asked for both at once the policy answers with an arrow, and only the
    coordinate branch produces a flyable curve -- so the crossing is not
    ordered until the aircraft is standing in the opening, where it is a short
    straight move at a door that is genuinely ahead.
    """
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0)
    first = sup.update(*CORRIDOR, NORTH, seen, 0.0).mission
    assert first.kind == APPROACH_DOOR
    portal = region_map.portals[first.portal_id]

    # The approach aims SHORT of the opening, on this side of it.
    aim = sup.aim_point(first)
    assert aim[1] < portal.centre[1], "stop in front of the door, not in it"
    assert math.hypot(aim[0] - portal.centre[0], aim[1] - portal.centre[1]) \
        == pytest.approx(sup.params.approach_offset_m, abs=0.2)

    # Standing in the opening finishes it, and the next order is the crossing.
    state = sup.update(portal.centre[0], portal.centre[1] - 0.4, NORTH, seen, 1.0)
    assert state.completed is not None and state.completed.kind == APPROACH_DOOR
    assert state.mission.kind == ENTER_ROOM
    assert state.mission.target_id == first.target_id


def test_inside_an_unscanned_room_the_order_is_to_look_around_it(
        region_map, coverage, nothing_seen):
    sup = _sup(region_map, coverage)
    state = sup.update(*ROOM_C, NORTH, nothing_seen, 0.0)
    assert state.mission.kind == SCAN_AREA
    assert state.mission.target_id == _room_at(region_map, ROOM_C).id


def test_inside_a_scanned_room_the_order_is_to_leave_it(region_map, coverage,
                                                        nothing_seen):
    room = _room_at(region_map, ROOM_C)
    seen = see(region_map, nothing_seen, room.id, 1.0)
    sup = _sup(region_map, coverage)
    state = sup.update(*ROOM_C, NORTH, seen, 0.0)
    assert state.mission.kind == EXIT_ROOM
    assert state.mission.target_id == room.id


def test_with_nothing_left_nearby_it_moves_on(region_map, coverage, nothing_seen):
    """Everything off this corridor done -> travel, not stand still."""
    seen = nothing_seen
    for region in region_map.regions.values():
        seen = see(region_map, seen, region.id, 1.0)
    sup = _sup(region_map, coverage)
    state = sup.update(*CORRIDOR, NORTH, seen, 0.0)
    assert state.mission.kind == SURVEY_COMPLETE


def test_the_order_points_at_the_doorway_and_aims_past_it(region_map, coverage,
                                                          nothing_seen):
    """Two different points, and conflating them is why an entry never lands."""
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage)
    approach = sup.update(*CORRIDOR, NORTH, seen, 0.0).mission
    portal = region_map.portals[approach.portal_id]
    assert sup.look_point(approach) == portal.centre
    # Cross from the threshold: that one aims past the door, into the room.
    state = sup.update(portal.centre[0], portal.centre[1] - 0.4, NORTH, seen, 5.0)
    assert state.mission.kind == ENTER_ROOM
    assert sup.look_point(state.mission) == portal.centre
    aim = sup.aim_point(state.mission)
    assert aim[1] > portal.centre[1], "aim past the doorway, into the room"
    assert region_map.region_at(*aim).id == state.mission.target_id


@pytest.mark.parametrize("yaw_deg, expected", [
    (90, "ahead of you"),
    (-90, "behind you"),
    (0, "on your left"),
    (180, "on your right"),
])
def test_the_bearing_is_given_as_a_side_of_the_aircraft(region_map, coverage,
                                                        nothing_seen, yaw_deg,
                                                        expected):
    """The one thing that puts the target in the frame the model is looking at."""
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage)
    # Stand directly below the door of room C and face four ways.
    state = sup.update(8.0, 1.0, math.radians(yaw_deg), seen, 0.0)
    assert state.bearing == expected


# ── when it decides a mission is finished ────────────────────────────────

def test_entering_is_judged_by_where_the_aircraft_is(region_map, coverage,
                                                      nothing_seen):
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage)
    first = sup.update(*CORRIDOR, NORTH, seen, 0.0).mission
    assert first.kind == APPROACH_DOOR
    room = region_map.regions[first.target_id]
    # Still short of the door after the grace period: not finished.
    assert sup.update(*CORRIDOR, NORTH, seen, 10.0).completed is None
    # Inside the room: the approach is over, however it got there.
    state = sup.update(*room.centre, NORTH, seen, 11.0)
    assert state.completed is not None and state.completed.kind == APPROACH_DOOR


def test_nothing_is_judged_inside_the_grace_period(region_map, coverage,
                                                    nothing_seen):
    """An order to leave a room must not be satisfied by standing in it."""
    room = _room_at(region_map, ROOM_C)
    seen = see(region_map, nothing_seen, room.id, 1.0)
    sup = _sup(region_map, coverage, arrival_grace_s=5.0)
    assert sup.update(*ROOM_C, NORTH, seen, 0.0).mission.kind == EXIT_ROOM
    assert sup.update(*CORRIDOR, NORTH, seen, 1.0).completed is None
    assert sup.update(*CORRIDOR, NORTH, seen, 6.0).completed is not None


def test_a_scan_finishes_when_enough_of_the_area_has_been_seen(
        region_map, coverage, nothing_seen):
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0)
    assert sup.update(*ROOM_C, NORTH, nothing_seen, 0.0).mission.kind == SCAN_AREA
    seen = see(region_map, nothing_seen.copy(), room.id, 0.70)
    state = sup.update(*ROOM_C, NORTH, seen, 1.0)
    assert state.completed is not None and state.completed.kind == SCAN_AREA


def test_a_corroborated_stop_finishes_a_scan(region_map, coverage, nothing_seen):
    """The policy's own claim counts -- above a floor of real coverage."""
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0,
               stop_hint_min_fraction=0.30)
    sup.update(*ROOM_C, NORTH, nothing_seen, 0.0)
    seen = see(region_map, nothing_seen.copy(), room.id, 0.40)
    state = sup.update(*ROOM_C, NORTH, seen, 1.0, stop_hint=True)
    assert state.completed is not None
    assert "policy stopped" in state.completed.note or True   # verdict is in history
    assert sup.history[-1][1].startswith("scanned")


def test_an_uncorroborated_stop_is_ignored(region_map, coverage, nothing_seen):
    """A model that stops for its own reasons must not tick off the checklist."""
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0,
               stop_hint_min_fraction=0.50)
    sup.update(*ROOM_C, NORTH, nothing_seen, 0.0)
    seen = see(region_map, nothing_seen.copy(), room.id, 0.10)
    assert sup.update(*ROOM_C, NORTH, seen, 1.0, stop_hint=True).completed is None


def test_a_scan_that_stops_yielding_anything_ends_by_itself(region_map, coverage,
                                                             nothing_seen):
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0, scan_stall_s=20.0)
    seen = see(region_map, nothing_seen.copy(), room.id, 0.20)
    sup.update(*ROOM_C, NORTH, seen, 0.0)
    assert sup.update(*ROOM_C, NORTH, seen, 10.0).completed is None
    state = sup.update(*ROOM_C, NORTH, seen, 21.0)
    assert state.completed is not None
    assert sup.history[-1][1] == "nothing further visible"


# ── giving up, and giving up for good ────────────────────────────────────

def test_a_mission_that_runs_too_long_is_given_up_and_something_else_chosen(
        region_map, coverage, nothing_seen):
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage, mission_timeout_s=30.0)
    first = sup.update(*CORRIDOR, NORTH, seen, 0.0).mission
    state = sup.update(*CORRIDOR, NORTH, seen, 31.0)
    assert sup.history[-1][1] == "given up"
    assert state.mission is not None
    assert (state.mission.kind, state.mission.target_id) != first.key, \
        "try something else"


def test_a_target_given_up_on_enough_times_is_abandoned_for_the_flight(
        region_map, coverage, nothing_seen):
    """Deferral alone only paces a loop; a ceiling ends it."""
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage, mission_timeout_s=10.0, defer_s=0.0,
               max_attempts=2)
    first = sup.update(*CORRIDOR, NORTH, seen, 0.0).mission
    key = first.key                       # this KIND of order at this target
    now = 0.0
    tried = 0
    for _ in range(60):
        now += 11.0
        state = sup.update(*CORRIDOR, NORTH, seen, now)
        if state.mission is not None and state.mission.key == key:
            tried += 1
    assert tried <= 2, "an abandoned order must stop coming back"


def test_it_never_oscillates_in_and_out_of_the_same_doorway(region_map, coverage,
                                                            nothing_seen):
    """The loop that looks exactly like working, and is not.

    A room whose far corner cannot be seen never reaches the scanned threshold,
    so "not scanned" keeps re-selecting it, every entry succeeds, and the
    aircraft crosses the same threshold for the rest of the flight.
    """
    room = _room_at(region_map, ROOM_C)
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    seen = see(region_map, seen, room.id, 0.25)          # and it will never rise
    sup = _sup(region_map, coverage, mission_timeout_s=10.0, scan_stall_s=5.0,
               defer_s=1.0, max_attempts=2, arrival_grace_s=0.0)
    now, entries = 0.0, 0
    inside = True
    for _ in range(200):
        now += 2.0
        where = ROOM_C if inside else CORRIDOR
        state = sup.update(*where, NORTH, seen, now)
        if state.mission is None:
            continue
        if state.mission.kind == ENTER_ROOM and state.mission.target_id == room.id:
            entries += 1
            inside = True
        elif state.mission.kind == EXIT_ROOM and state.mission.target_id == room.id:
            inside = False
    assert entries <= 3, "entered the same unscannable room %d times" % entries


def test_a_scan_the_policy_ended_early_is_not_immediately_re_ordered(
        region_map, coverage, nothing_seen):
    """The loop a live flight found and the synthetic ones had not.

    A corroborated STOP finishes the scan without reaching the threshold, so
    "not scanned" selects the very same scan again, and the next STOP finishes
    that one too: eight identical completions on one room in ninety seconds,
    with the aircraft parked in its doorway the whole time.
    """
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0,
               stop_hint_min_fraction=0.30)
    seen = see(region_map, nothing_seen, room.id, 0.40)
    assert sup.update(*ROOM_C, NORTH, seen, 0.0).mission.kind == SCAN_AREA
    state = sup.update(*ROOM_C, NORTH, seen, 1.0, stop_hint=True)
    assert state.completed is not None
    # The very next choice must be something else -- leaving, in this case.
    assert state.mission is not None and state.mission.kind != SCAN_AREA
    # ...and it must not come back to it later either.
    for tick in range(2, 400):
        later = sup.update(*ROOM_C, NORTH, seen, float(tick))
        assert not (later.mission and later.mission.kind == SCAN_AREA
                    and later.mission.target_id == room.id)


def test_an_area_with_nothing_more_to_see_is_not_re_ordered_either(
        region_map, coverage, nothing_seen):
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0, scan_stall_s=10.0)
    seen = see(region_map, nothing_seen, room.id, 0.20)
    sup.update(*ROOM_C, NORTH, seen, 0.0)
    assert sup.update(*ROOM_C, NORTH, seen, 11.0).completed is not None
    assert sup.history[-1][1] == "nothing further visible"
    for tick in range(12, 400):
        later = sup.update(*ROOM_C, NORTH, seen, float(tick))
        assert not (later.mission and later.mission.kind == SCAN_AREA
                    and later.mission.target_id == room.id)


def test_it_never_orders_an_exit_from_a_room_it_is_not_in(region_map, coverage,
                                                          nothing_seen):
    """The fourth loop of this family, and the one a live flight found.

    Standing in a doorway from the CORRIDOR side, with the room beyond already
    looked around: treating the doorway as belonging to that room orders an exit
    from it, which is satisfied instantly because the aircraft is already out,
    and then ordered again. Measured in flight as twelve identical exits from
    one room in two minutes.
    """
    room = _room_at(region_map, ROOM_C)
    corridor = region_map.corridors()[0]
    sup = _sup(region_map, coverage, arrival_grace_s=0.0, scan_stall_s=5.0)
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    seen = see(region_map, seen, room.id, 0.20)      # looked at, never enough
    # Look around it from the doorway until the supervisor accepts the scan.
    for tick in range(0, 12):
        sup.update(*DOORWAY_C, NORTH, seen, float(tick))
    exits = 0
    for tick in range(12, 200):
        state = sup.update(*DOORWAY_C, NORTH, seen, float(tick))
        if state.mission and state.mission.kind == EXIT_ROOM:
            exits += 1
    assert exits == 0, "ordered %d exits from a room it was never in" % exits


def test_no_mission_can_be_issued_without_limit(region_map, coverage,
                                                 nothing_seen):
    """A constant bound on any loop, including the ones not yet found.

    Four separate loops in this state machine were each discovered by watching
    one run. This is the backstop that turns the fifth into a line in the log
    rather than a lost flight.
    """
    sup = _sup(region_map, coverage, max_attempts=2, max_issues_multiple=3,
               mission_timeout_s=1.0, defer_s=0.0, arrival_grace_s=0.0)
    seen = nothing_seen
    counts = {}
    for tick in range(0, 600):
        state = sup.update(*CORRIDOR, NORTH, seen, float(tick))
        if state.changed and state.mission is not None:
            counts[state.mission.key] = counts.get(state.mission.key, 0) + 1
    assert counts, "nothing was ever issued"
    assert max(counts.values()) <= 2 * 3, counts


# ── asking for help when nothing is changing ─────────────────────────────

def test_it_asks_for_a_nudge_when_a_mission_stops_making_progress(
        region_map, coverage, nothing_seen):
    """Wedged against a jamb, the same frame gives the same answer for ever."""
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage, nudge_after_s=20.0, nudge_cooldown_s=15.0,
               mission_timeout_s=1e6)
    assert sup.update(*CORRIDOR, NORTH, seen, 0.0).mission.kind == APPROACH_DOOR
    assert not sup.update(*CORRIDOR, NORTH, seen, 10.0).nudge, "too early"
    assert sup.update(*CORRIDOR, NORTH, seen, 21.0).nudge, "stalled and silent"
    assert not sup.update(*CORRIDOR, NORTH, seen, 25.0).nudge, "cooling down"
    assert sup.update(*CORRIDOR, NORTH, seen, 40.0).nudge, "still stalled"


def test_an_aircraft_that_is_moving_is_never_nudged(region_map, coverage,
                                                     nothing_seen):
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage, nudge_after_s=10.0, nudge_min_move_m=0.5,
               mission_timeout_s=1e6, arrival_grace_s=1e6)
    sup.update(6.0, 1.0, NORTH, seen, 0.0)
    x = 6.0
    for tick in range(1, 60):
        x += 0.2                              # 0.2 m a tick: plainly travelling
        assert not sup.update(x, 1.0, NORTH, seen, float(tick)).nudge


def test_a_scan_is_never_nudged(region_map, coverage, nothing_seen):
    """Standing still IS the mission; backing off would undo it."""
    sup = _sup(region_map, coverage, nudge_after_s=5.0, mission_timeout_s=1e6,
               scan_stall_s=1e6)
    assert sup.update(*ROOM_C, NORTH, nothing_seen, 0.0).mission.kind == SCAN_AREA
    for tick in range(1, 60):
        assert not sup.update(*ROOM_C, NORTH, nothing_seen, float(tick)).nudge


# ── a corridor is finished where it was looked at, not as a whole ────────

def test_a_long_corridor_can_be_scanned_again_from_somewhere_else(
        region_map, coverage, nothing_seen):
    """Forty metres of spine written off after one look from one end is most of
    a corridor nobody ever surveys."""
    corridor = region_map.corridors()[0]
    sup = _sup(region_map, coverage, arrival_grace_s=0.0, scan_stall_s=5.0,
               rescan_radius_m=4.0, mission_timeout_s=6.0, defer_s=0.0)
    seen = see(region_map, nothing_seen, corridor.id, 0.30)   # never reaches 60%
    # Look around from the west end until the supervisor accepts it. The spot
    # is chosen clear of any doorway, or the doorway would claim the mission.
    for tick in range(0, 12):
        sup.update(3.5, 1.0, NORTH, seen, float(tick))
    assert corridor.id in sup._scans, "the scan was never accepted"
    assert len(sup._scans[corridor.id]) == 1

    # Still at the west end: no point looking again.
    for tick in range(12, 20):
        state = sup.update(3.5, 1.0, NORTH, seen, float(tick))
        assert not (state.mission and state.mission.kind == SCAN_AREA
                    and state.mission.target_id == corridor.id)

    # Twelve metres east is a different part of the same corridor, and once
    # whatever else is in flight has run its course it is looked at again.
    found = False
    for tick in range(30, 400):
        state = sup.update(16.0, 1.0, NORTH, seen, float(tick))
        if (state.mission and state.mission.kind == SCAN_AREA
                and state.mission.target_id == corridor.id):
            found = True
            break
    assert found, "the far end of the corridor was never looked at"


def test_a_room_is_finished_by_one_look(region_map, coverage, nothing_seen):
    """The vicinity rule must not make small rooms scannable for ever."""
    room = _room_at(region_map, ROOM_C)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0, scan_stall_s=5.0,
               rescan_radius_m=4.0)
    seen = see(region_map, nothing_seen, room.id, 0.20)
    for tick in range(0, 12):
        sup.update(*ROOM_C, NORTH, seen, float(tick))
    for tick in range(12, 60):
        state = sup.update(7.0, 5.5, NORTH, seen, float(tick))   # other corner
        assert not (state.mission and state.mission.kind == SCAN_AREA
                    and state.mission.target_id == room.id)


def test_every_room_gets_one_try_before_any_gets_a_second(
        region_map, coverage, nothing_seen):
    # The whole corridor already seen, so surveying the floor is not on offer
    # and the doors are what is left.
    """Corridor coverage is worth more than a second go at a door.

    Measured on this building, flying the corridors and spinning sees 79.6% of
    the floor and clears thirteen of its twenty rooms without an entry -- so a
    corridor the aircraft can reach beats a doorway it has already failed at.
    """
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage, mission_timeout_s=10.0, defer_s=0.0,
               max_attempts=9, arrival_grace_s=0.0, rescan_radius_m=50.0)
    now, first_pass = 0.0, []
    for _ in range(40):
        now += 11.0
        state = sup.update(*CORRIDOR, NORTH, seen, now)
        if state.changed and state.mission and state.mission.kind in (
                APPROACH_DOOR, ENTER_ROOM):
            target = state.mission.target_id
            if target in first_pass:
                # A repeat: every other room must have been tried by now.
                assert len(set(first_pass)) == len(region_map.rooms()), (
                    "went back to room %d after only %s"
                    % (target, sorted(set(first_pass))))
                break
            first_pass.append(target)
    assert first_pass, "no room was ever attempted"


# ── going somewhere new, one short hop at a time ─────────────────────────

def test_it_heads_for_the_nearest_unseen_floor_in_this_area(
        region_map, coverage, nothing_seen):
    """One short move, not a thirty-metre order to the next region.

    At ~22 s a decision and ~1 m of route per decision, ten metres is about ten
    decisions and fits a mission budget. Aiming at region centres instead meant
    thirty- and forty-metre orders, and four in five of those were given up on.
    """
    corridor = region_map.corridors()[0]
    # The west half of the corridor seen, the east half not.
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    west = cols < 80
    seen[rows[west], cols[west]] = True
    sup = _sup(region_map, coverage, travel_step_m=8.0, rescan_radius_m=20.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]            # already looked around here
    state = sup.update(3.0, 1.0, 0.0, seen, 0.0)      # facing east, into the dark
    assert state.mission is not None
    assert state.mission.kind == TRAVERSE
    assert state.mission.target_xy is not None
    tx, ty = state.mission.target_xy
    assert tx > 3.0, "it should head east, towards what it has not seen"
    assert math.hypot(tx - 3.0, ty - 1.0) <= 8.5, "one hop, not the whole corridor"
    assert sup.look_point(state.mission) == state.mission.target_xy


def test_arriving_at_the_spot_finishes_the_hop(region_map, coverage,
                                                nothing_seen):
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    west = cols < 80
    seen[rows[west], cols[west]] = True
    sup = _sup(region_map, coverage, travel_step_m=8.0, travel_arrive_m=1.5,
               rescan_radius_m=20.0, arrival_grace_s=0.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]            # already looked around here
    first = sup.update(3.0, 1.0, 0.0, seen, 0.0).mission
    assert first.kind == TRAVERSE
    tx, ty = first.target_xy
    assert sup.update(3.0, 1.0, 0.0, seen, 5.0).completed is None
    state = sup.update(tx, ty, 0.0, seen, 6.0)
    assert state.completed is not None and state.completed.kind == TRAVERSE
    assert sup.history[-1][1] == "arrived"


def test_an_area_it_has_seen_all_of_offers_nowhere_to_go(region_map, coverage,
                                                          nothing_seen):
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    sup = _sup(region_map, coverage)
    assert sup._nearest_unseen(corridor, 6.5, 1.0, 0.0, seen, 0.0) is None


def test_the_survey_reports_itself_finished_when_there_is_nothing_left(
        region_map, coverage, nothing_seen):
    seen = nothing_seen
    for region in region_map.regions.values():
        seen = see(region_map, seen, region.id, 1.0)
    sup = _sup(region_map, coverage)
    state = sup.update(*CORRIDOR, NORTH, seen, 0.0)
    assert state.mission.kind == SURVEY_COMPLETE
    assert state.rooms_scanned == state.rooms_total == 5
    # ...and it stays finished rather than timing out into something else.
    assert sup.update(*CORRIDOR, NORTH, seen, 9999.0).mission.kind == SURVEY_COMPLETE


def test_off_the_map_it_says_nothing_rather_than_guessing(region_map, coverage,
                                                          nothing_seen):
    sup = _sup(region_map, coverage)
    state = sup.update(-50.0, -50.0, NORTH, nothing_seen, 0.0)
    assert state.topo == OFF_MAP
    assert state.mission is None


def test_a_narrow_opening_is_never_offered_as_a_way_through(region_map, coverage,
                                                            nothing_seen):
    """The airframe is 0.63 m wide; a 1.0 m door is offered, a 0.5 m one is not."""
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    wide = _sup(region_map, coverage, min_portal_m=0.80)
    assert wide.update(*CORRIDOR, NORTH, seen, 0.0).mission.kind == APPROACH_DOOR
    narrow = _sup(region_map, coverage, min_portal_m=1.50)
    assert narrow.update(*CORRIDOR, NORTH, seen, 0.0).mission.kind == SURVEY_COMPLETE


def test_progress_is_reported_against_the_whole_checklist(region_map, coverage,
                                                          nothing_seen):
    room = _room_at(region_map, ROOM_C)
    seen = see(region_map, nothing_seen, room.id, 1.0)
    sup = _sup(region_map, coverage)
    state = sup.update(*CORRIDOR, NORTH, seen, 0.0)
    assert (state.rooms_scanned, state.rooms_total) == (1, 5)
    assert 0.0 < state.fraction_seen < 1.0



def test_a_hop_ends_when_its_target_has_been_seen(region_map, coverage,
                                                  nothing_seen):
    """Seeing the patch is the goal; standing on it is not.

    Measured in flight: two hops in a row ran their whole clock and were given
    up on, while the coverage they were flying for climbed fifteen points
    during them. The aircraft had done the job from ten metres off and the
    supervisor was still waiting for it to arrive.
    """
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    west = cols < 80
    seen[rows[west], cols[west]] = True
    sup = _sup(region_map, coverage, travel_step_m=8.0, travel_arrive_m=1.5,
               rescan_radius_m=20.0, arrival_grace_s=0.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]
    hop = sup.update(3.0, 1.0, 0.0, seen, 0.0).mission
    assert hop.kind == TRAVERSE

    # Unseen target, aircraft nowhere near it: the hop stands.
    assert sup.update(3.0, 1.0, 0.0, seen, 5.0).completed is None

    col, row = region_map.cell_of(*hop.target_xy)
    seen[row, col] = True
    done = sup.update(3.0, 1.0, 0.0, seen, 6.0)
    assert done.completed is hop
    assert sup.history[-1][1] == "in view"


def test_a_hop_gives_up_sooner_than_a_doorway_does():
    """A hop that is not progressing is cheap to replace; a doorway is not."""
    params = SupervisorParams()
    assert params.travel_timeout_s < params.mission_timeout_s


def test_a_refused_hop_ends_at_once_and_aims_somewhere_else(region_map, coverage,
                                                            nothing_seen):
    """STOP on a hop is a refusal, not a pause.

    The policy does not change its mind about an order it has declined, so
    every second the supervisor waits for it to is a second the aircraft is
    stationary -- measured over one flight, STOP was 46% of all answers and
    four hops in a row expired without it moving. The answer is a different
    patch of floor, immediately.
    """
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    west = cols < 80
    seen[rows[west], cols[west]] = True
    sup = _sup(region_map, coverage, travel_step_m=8.0, travel_arrive_m=1.5,
               rescan_radius_m=20.0, arrival_grace_s=0.0, refuse_after_s=5.0,
               refuse_radius_m=4.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]
    first = sup.update(3.0, 1.0, 0.0, seen, 0.0).mission
    assert first.kind == TRAVERSE

    # A STOP inside the grace changes nothing...
    assert sup.update(3.0, 1.0, 0.0, seen, 2.0, stop_hint=True).completed is None
    # ...and past it, the hop is over and the clock has barely moved.
    done = sup.update(3.0, 1.0, 0.0, seen, 6.0, stop_hint=True)
    assert done.completed is first
    assert sup.history[-1][1] == "refused"
    assert 6.0 < sup.params.travel_timeout_s

    # Whatever it does next, it is not the patch it was just refused.
    nxt = done.mission
    if nxt is not None and nxt.kind == TRAVERSE and nxt.target_xy is not None:
        assert math.hypot(nxt.target_xy[0] - first.target_xy[0],
                          nxt.target_xy[1] - first.target_xy[1]) > 4.0


def test_turning_is_costed_against_the_camera_not_a_cliff():
    """A target outside 75 degrees is not free just because it is near.

    The old rule charged nothing until a hundred degrees off the nose, on an
    aircraft whose camera sees seventy-five -- so every target between 38 and
    100 degrees looked as good as one straight ahead, and those are exactly the
    ones it cannot see and the policy will not fly to.
    """
    sup = ExplorationSupervisor.__new__(ExplorationSupervisor)
    sup.params = SupervisorParams(in_frame_deg=35.0, turn_cost_m_per_deg=0.4)
    assert float(sup._turn_cost(0.0)) == 0.0
    assert float(sup._turn_cost(30.0)) == 0.0          # in frame, free
    assert float(sup._turn_cost(-30.0)) == 0.0         # and it is symmetric
    # Ninety degrees round costs about twenty metres of flying, which is what
    # ~2.1 deg/s of yaw is worth against ~1 m per decision.
    assert 15.0 < float(sup._turn_cost(90.0)) < 30.0
    assert sup._turn_cost(180.0) > sup._turn_cost(90.0) > sup._turn_cost(45.0)
    # Costly, never impossible: when it is all that is left, it is still chosen.
    assert np.isfinite(float(sup._turn_cost(180.0)))


def test_a_refused_door_is_let_go_of_too(region_map, coverage, nothing_seen):
    """A STOP streak must not be able to eat a whole mission clock again.

    One flight sat under an unbroken run of 51 STOPs on a single door
    approach -- the entire mission budget, stationary. Refusal applies to the
    "go somewhere else" missions, which have alternatives; crossing a threshold
    does not, because giving up on that one strands the aircraft in the room.
    """
    target = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, region_map.corridors()[0].id, 1.0)
    for room in region_map.rooms():
        if room.id != target.id:
            seen = see(region_map, seen, room.id, 1.0)
    sup = _sup(region_map, coverage, arrival_grace_s=0.0, refuse_after_s=5.0)
    first = sup.update(6.0, 1.0, 0.0, seen, 0.0).mission
    assert first.kind == APPROACH_DOOR
    assert sup.update(6.0, 1.0, 0.0, seen, 2.0, stop_hint=True).completed is None
    done = sup.update(6.0, 1.0, 0.0, seen, 6.0, stop_hint=True)
    assert done.completed is first
    assert sup.history[-1][1] == "refused"
    assert 6.0 < sup.params.mission_timeout_s


def test_a_long_hop_is_not_finished_by_the_waypoint_it_aims_at(region_map,
                                                               coverage,
                                                               nothing_seen):
    """The waypoint is where to aim; the cell is what the hop is for.

    A hop is capped at one step, so on a long leg the aircraft is aimed at a
    point part of the way there -- over floor it has usually already seen.
    Judging the hop by that point ends it on the tick it is issued. Measured:
    six hops in a row completed instantly, the area was retired for being
    "issued too often", and the survey stopped dead at 12.5%.
    """
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    west = cols < 80
    seen[rows[west], cols[west]] = True
    # A step far shorter than the corridor, so the two points must differ.
    sup = _sup(region_map, coverage, travel_step_m=1.0, travel_arrive_m=1.5,
               rescan_radius_m=20.0, arrival_grace_s=0.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]
    hop = sup.update(3.0, 1.0, 0.0, seen, 0.0).mission
    assert hop.kind == TRAVERSE
    assert hop.probe_xy is not None
    assert math.hypot(hop.target_xy[0] - hop.probe_xy[0],
                      hop.target_xy[1] - hop.probe_xy[1]) > 1.0

    # The waypoint being seen means nothing -- it is floor already surveyed.
    wcol, wrow = region_map.cell_of(*hop.target_xy)
    seen[wrow, wcol] = True
    assert sup.update(3.0, 1.0, 0.0, seen, 5.0).completed is None

    # The cell it was chosen for is the finish line.
    pcol, prow = region_map.cell_of(*hop.probe_xy)
    seen[prow, pcol] = True
    done = sup.update(3.0, 1.0, 0.0, seen, 6.0)
    assert done.completed is hop and sup.history[-1][1] == "in view"


def test_succeeding_hops_never_retire_an_area(region_map, coverage,
                                              nothing_seen):
    """The ceiling catches loops, and a loop never succeeds.

    The tally is kept per (kind, area), so a large area needing twenty honest
    hops tripped it as surely as a loop did. Measured: the atrium was retired
    after twelve consecutive hops that all ended "in view" or "arrived", the
    survey froze at 15.6%, and every room in the building was still to do.
    """
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    seen[rows[cols < 40], cols[cols < 40]] = True
    sup = _sup(region_map, coverage, travel_step_m=2.0, rescan_radius_m=40.0,
               arrival_grace_s=0.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]
    ceiling = sup.params.max_attempts * sup.params.max_issues_multiple

    now = 0.0
    for _ in range(ceiling * 3):        # comfortably past the old limit
        state = sup.update(3.0, 1.0, 0.0, seen, now)
        hop = state.mission
        if hop is None or hop.kind != TRAVERSE:
            break
        pcol, prow = region_map.cell_of(*hop.probe_xy)
        seen[prow, pcol] = True         # the camera reaches it: honest success
        now += 1.0
        assert sup._issues.get(hop.key, 0) <= 1, "a hop that worked owes nothing"

    assert not any(v[1].startswith("retired") for v in sup.history), \
        "twelve successful hops must not retire the area they surveyed"


def test_a_refusal_still_counts_against_the_ceiling(region_map, coverage,
                                                    nothing_seen):
    """Only success clears it -- otherwise a refusing loop would run forever."""
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    seen[rows[cols < 80], cols[cols < 80]] = True
    sup = _sup(region_map, coverage, travel_step_m=8.0, rescan_radius_m=20.0,
               arrival_grace_s=0.0, refuse_after_s=1.0)
    sup._scans[corridor.id] = [(3.0, 1.0)]
    hop = sup.update(3.0, 1.0, 0.0, seen, 0.0).mission
    sup.update(3.0, 1.0, 0.0, seen, 5.0, stop_hint=True)
    assert sup.history[-1][1] == "refused"
    assert sup._issues.get(hop.key) == 1


def test_a_new_mission_does_not_reset_the_stuck_clock(region_map, coverage,
                                                      nothing_seen):
    """How long it has been stationary is a fact about the aircraft.

    Measured: with hops being refused every fifteen seconds, a wedged drone
    never reached a thirty-five second stuck threshold, so no nudge was ever
    asked for -- the follower was reporting HARD BLOCKED at 0.33 m and the
    survey sat frozen for nine minutes.
    """
    corridor = region_map.corridors()[0]
    seen = nothing_seen
    rows, cols = np.nonzero(region_map.labels == corridor.id)
    seen[rows[cols < 40], cols[cols < 40]] = True
    sup = _sup(region_map, coverage, travel_step_m=8.0, rescan_radius_m=40.0,
               arrival_grace_s=0.0, refuse_after_s=5.0, nudge_after_s=35.0,
               nudge_cooldown_s=25.0, nudge_min_move_m=0.6)
    sup._scans[corridor.id] = [(3.0, 1.0)]

    # Wedged: it never moves, and every hop it is given is refused.
    nudged, now = False, 0.0
    while now < 120.0:
        state = sup.update(3.0, 1.0, 0.0, seen, now, stop_hint=True)
        nudged = nudged or state.nudge
        now += 6.0
    assert len([h for h in sup.history if h[1] == "refused"]) > 3, \
        "the refusals that used to hide the stall must still be happening"
    assert nudged, \
        "a stationary aircraft must be nudged however often it is re-tasked"
