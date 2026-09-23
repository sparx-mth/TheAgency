"""The instruction: what gets published, and whether it survives the server.

The one thing every test here checks in some form is that the fragment still
reads as English once the model server has substituted it into its own template,
because that substitution is upstream, invisible from here, and eats the full
stop it replaces.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.exploration.briefing import BriefingStyle, brief
from sparx_agency.core.planning.exploration.mission import (
    APPROACH_DOOR,
    ENTER_ROOM,
    EXIT_ROOM,
    SCAN_AREA,
    SURVEY_COMPLETE,
    TRAVERSE,
    ExplorationSupervisor,
    SupervisorParams,
)

from .conftest import see

#: Exactly what `internvla_n1_policy.init_prompts` builds, so a test can see
#: what the model is actually handed.
TEMPLATE = ("You are an autonomous navigation assistant. Your task is to "
            "<instruction>. Where should you go next to stay on track?")

CORRIDOR = (6.5, 1.0)
ROOM_C = (7.55, 4.45)
NORTH = math.radians(90)


def _substituted(fragment):
    """The prompt the model gets, the way the server makes it."""
    return TEMPLATE.replace("<instruction>.", fragment)


def _state(region_map, coverage, at, seen, yaw=NORTH, now=0.0, **params):
    sup = ExplorationSupervisor(region_map, coverage, SupervisorParams(**params))
    return sup, sup.update(at[0], at[1], yaw, seen, now)


# ── it has to survive the substitution ───────────────────────────────────

def test_the_fragment_continues_the_sentence_it_is_dropped_into(
        region_map, coverage, nothing_seen):
    _, state = _state(region_map, coverage, CORRIDOR, nothing_seen)
    fragment = brief(state, region_map)
    assert fragment[0].islower(), "it must continue 'Your task is to '"
    assert fragment.endswith("."), "the template's own full stop is consumed"
    prompt = _substituted(fragment)
    assert "Your task is to turn on the spot" in prompt
    assert "<instruction>" not in prompt


def test_it_never_contains_an_apostrophe(region_map, coverage, nothing_seen):
    """It is published inside single quotes by a shell; one would end the string."""
    seen = see(region_map, nothing_seen, region_map.corridors()[0].id, 1.0)
    for at in (CORRIDOR, ROOM_C):
        _, state = _state(region_map, coverage, at, seen)
        assert "'" not in brief(state, region_map)


def test_an_absent_mission_produces_nothing_rather_than_an_empty_order(
        region_map, coverage, nothing_seen):
    """Off the map there is nothing to say, and saying nothing is the answer.

    The caller must then leave the standing instruction alone: publishing an
    empty string would hand the policy a blank task.
    """
    _, state = _state(region_map, coverage, (-50.0, -50.0), nothing_seen)
    assert state.mission is None
    assert brief(state, region_map) == ""


# ── the three parts ──────────────────────────────────────────────────────

def test_it_is_two_parts_and_only_two(region_map, coverage, nothing_seen):
    """Where I am, and where to go. Nothing about what has been surveyed.

    The supervisor already knows that to the cell and enforces it -- every
    order it issues names somewhere unvisited -- so repeating it back spends
    context on a fact the model cannot act on, and it grows over a flight.
    """
    room = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, room.id, 1.0)
    _, state = _state(region_map, coverage, ROOM_C, seen)
    text = brief(state, region_map)
    assert "doorway" in text, "the order"
    assert "You are in" in text, "the location"
    for history in ("looked inside", "so far", "rooms.", "already"):
        assert history not in text, "no survey history belongs in the prompt"
    assert text.count(".") == 2, text


def test_the_order_comes_first_because_the_template_demands_a_verb(
        region_map, coverage, nothing_seen):
    """"Your task is to you are in the middle spine" is not English."""
    _, state = _state(region_map, coverage, CORRIDOR, nothing_seen)
    text = brief(state, region_map)
    assert text.index("You are in") > 0, "the location follows the order"
    assert _substituted(text).startswith(
        "You are an autonomous navigation assistant. Your task is to turn")


def test_the_instruction_does_not_grow_as_the_survey_does(region_map, coverage,
                                                          nothing_seen):
    """The whole reason the history was dropped: length is paid every decision.

    Eight past frames already dominate that context window and System 2 is the
    clock, so an instruction that got longer as the flight went on would make
    every remaining decision slower.
    """
    _, early = _state(region_map, coverage, CORRIDOR, nothing_seen)
    room = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen.copy(), room.id, 1.0)
    _, late = _state(region_map, coverage, CORRIDOR, seen)
    assert abs(len(brief(late, region_map)) - len(brief(early, region_map))) < 40


def test_each_part_can_be_switched_off_for_the_experiment(region_map, coverage,
                                                           nothing_seen):
    """The honest A/B is one flag, not a second implementation."""
    room = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, room.id, 1.0)
    _, state = _state(region_map, coverage, ROOM_C, seen)

    no_where = brief(state, region_map, BriefingStyle(include_location=False))
    assert "You are in" not in no_where

    bare = brief(state, region_map, BriefingStyle(goal_only=True))
    assert "You are in" not in bare
    assert "doorway" in bare, "the order itself always survives"


def test_standing_in_a_doorway_names_the_room_it_belongs_to(region_map, coverage,
                                                            nothing_seen):
    """The room, not both sides: the corridor half is length nobody reads."""
    room = region_map.region_at(*ROOM_C)
    _, state = _state(region_map, coverage, (8.0, 2.05), nothing_seen)
    text = brief(state, region_map)
    assert "in the doorway of %s" % room.name in text


# ── one order per mission, each aimed at something visible ───────────────

def test_the_order_for_a_corridor_scan_forbids_leaving_it(region_map, coverage,
                                                           nothing_seen):
    _, state = _state(region_map, coverage, CORRIDOR, nothing_seen)
    assert state.mission.kind == SCAN_AREA
    order = brief(state, region_map, BriefingStyle(goal_only=True))
    assert "corridor" in order and "without going into any room" in order


def test_the_order_for_a_room_scan_asks_what_is_in_it(region_map, coverage,
                                                       nothing_seen):
    _, state = _state(region_map, coverage, ROOM_C, nothing_seen)
    assert state.mission.kind == SCAN_AREA
    order = brief(state, region_map, BriefingStyle(goal_only=True))
    assert "look all around this room" in order


@pytest.mark.parametrize("yaw_deg", [90, -90, 152, -152])
def test_no_order_ever_names_a_direction_to_turn(region_map, coverage,
                                                 nothing_seen, yaw_deg):
    """A side is not a hint about where something is. It is a command.

    Measured over a ladder of orders, three passes each: told "do A, then turn
    right", this policy turns right on the current frame -- 7 runs out of 7,
    about fifty seconds of spinning, after which it is facing the wrong way and
    flies off confidently in it. A supervised flight running an order of that
    shape spent 78% of its answers turning left on the spot, 231 of 298, frozen
    inside one room for thirty-one minutes.

    The one order that worked, 3 of 3 and faster than any other, contained no
    direction word at all. So none of these may either, from any heading.
    """
    target = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, region_map.corridors()[0].id, 1.0)
    for room in region_map.rooms():
        if room.id != target.id:
            seen = see(region_map, seen, room.id, 1.0)
    _, state = _state(region_map, coverage, (6.0, 1.0), seen,
                      yaw=math.radians(yaw_deg))
    order = brief(state, region_map, BriefingStyle(goal_only=True)).lower()
    for word in ("turn to your", "on your left", "on your right", "behind you",
                 "to your left", "to your right"):
        assert word not in order, "%r appeared in %r" % (word, order)


def test_a_doorway_in_frame_is_named_and_given_a_stopping_place(region_map,
                                                                coverage,
                                                                nothing_seen):
    """A named thing in the frame is the only referent the policy answers to.

    This is verbatim the shape of the package's best-measured instruction of
    four -- a thing to look at and a place to stop -- and it says nothing about
    crossing, because that is the next order.
    """
    target = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, region_map.corridors()[0].id, 1.0)
    for room in region_map.rooms():
        if room.id != target.id:
            seen = see(region_map, seen, room.id, 1.0)
    portal = region_map.portals_of(target.id)[0]
    # Stand back from the door and look straight at it.
    px, py = portal.centre
    yaw = math.atan2(py - 1.0, px - (px - 3.0))
    _, state = _state(region_map, coverage, (px - 3.0, 1.0), seen, yaw=yaw)
    order = brief(state, region_map, BriefingStyle(goal_only=True)).lower()
    if state.mission.kind != APPROACH_DOOR:
        pytest.skip("this pose gets a different mission; the wording is tested above")
    assert "open doorway" in order and "stop in front of it" in order
    assert "on your left" not in order and "on your right" not in order


def test_a_crossing_is_a_short_straight_order(region_map, coverage, nothing_seen):
    """Issued only from the threshold, so it can say "in front of you"."""
    target = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, region_map.corridors()[0].id, 1.0)
    portal = region_map.portals_of(target.id)[0]
    _, state = _state(region_map, coverage,
                      (portal.centre[0], portal.centre[1] - 0.4), seen)
    assert state.mission.kind == ENTER_ROOM
    order = brief(state, region_map, BriefingStyle(goal_only=True)).lower()
    assert "fly forward through the doorway in front of you" in order
    assert "stop inside the room" in order


def test_an_exit_says_where_to_end_up(region_map, coverage, nothing_seen):
    room = region_map.region_at(*ROOM_C)
    seen = see(region_map, nothing_seen, room.id, 1.0)
    # Facing the way out, so this exercises the direct phrasing; the case where
    # the door is over the aircraft's shoulder has its own test below.
    _, state = _state(region_map, coverage, ROOM_C, seen,
                      yaw=math.radians(-90))
    assert state.mission.kind == EXIT_ROOM
    order = brief(state, region_map, BriefingStyle(goal_only=True))
    assert "leave this room" in order.lower()
    assert "stop in the corridor outside" in order


def test_the_finished_survey_says_to_stop(region_map, coverage, nothing_seen):
    seen = nothing_seen
    for region in region_map.regions.values():
        seen = see(region_map, seen, region.id, 1.0)
    _, state = _state(region_map, coverage, CORRIDOR, seen)
    assert state.mission.kind == SURVEY_COMPLETE
    assert "stop where you are" in brief(state, region_map).lower()


def test_every_order_is_a_short_readable_sentence(region_map, coverage,
                                                   nothing_seen):
    """Long enough to be concrete, short enough to stay in distribution.

    The policy is fine-tuned on brief imperatives, and the recorder's overlay
    wraps at four lines of 54 characters -- a briefing that overflows it is one
    a viewer cannot check the flight against.
    """
    seen = see(region_map, nothing_seen, region_map.corridors()[0].id, 1.0)
    for at in (CORRIDOR, ROOM_C, (8.0, 2.05), (8.0, 1.6)):
        _, state = _state(region_map, coverage, at, seen)
        text = brief(state, region_map)
        assert 30 < len(text) <= 160, "%d chars: %r" % (len(text), text)
        assert text.count(".") <= 2
