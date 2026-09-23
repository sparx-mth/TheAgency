"""Say the mission in the grammar the policy is actually given.

**Two parts, and only two: where the aircraft is, and where it must go next.**

There is deliberately no third part saying what has been surveyed so far. The
supervisor already knows that, to the cell, and acts on it -- every order it
issues names somewhere unvisited, and a room it has finished is never named
again. Repeating the history back to the model spends context on a fact the
model cannot use and the supervisor is already enforcing, and it grows over a
flight, which makes every remaining decision slower on a policy that is already
the clock.

**It writes a fragment, not a sentence, and the reason is upstream.** The model
server builds its prompt as::

    "You are an autonomous navigation assistant. Your task is to <instruction>.
     Where should you go next to stay on track? ..."

and substitutes with ``value.replace('<instruction>.', instruction)`` -- the
trailing full stop is consumed, so whatever is produced here has to continue
"Your task is to ..." and supply its own punctuation.

That grammar is also why the **order comes first and the location second**,
which is the reverse of how a person would say it. "Your task is to you are in
the middle spine" is not English; "Your task is to go through the open doorway
on your right and stop inside the room beyond it. You are in the middle spine."
is, and it puts the imperative where the template expects it -- immediately
before the model is asked what to do.

No apostrophes anywhere: the instruction is published by a shell as
``ros2 topic pub ... "{data: '<instruction>'}"`` and one would end the string.

ROS-free, no numpy, Python 3.8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sparx_agency.core.planning.exploration.mission import (
    APPROACH_DOOR,
    AT_DOORWAY,
    ENTER_ROOM,
    EXIT_ROOM,
    SCAN_AREA,
    SURVEY_COMPLETE,
    TRAVERSE,
    SupervisorState,
)
from sparx_agency.core.planning.exploration.region_map import RegionMap


@dataclass(frozen=True)
class BriefingStyle:
    """What goes into the instruction.

    Attributes:
        include_location: Part one -- which area the aircraft is in. Names come
            from the region map and are handed over verbatim, so they are worth
            editing there.
        goal_only: Drop the location too, leaving the bare order. The floor of
            the experiment: everything the supervisor is worth, with none of the
            context.
    """

    include_location: bool = True
    goal_only: bool = False


def brief(state, region_map, style=BriefingStyle()):
    # type: (SupervisorState, RegionMap, BriefingStyle) -> str
    """The instruction to publish for the state the supervisor is in.

    Args:
        state: This tick's supervisor state.
        region_map: Used to name the doorway the aircraft is standing in.
        style: Which parts to include.

    Returns:
        A fragment continuing "Your task is to ", ending in a full stop. Empty
        when there is no mission to give -- the caller should then leave the
        current instruction alone rather than publishing nothing.
    """
    order = _order(state, region_map)
    if not order:
        return ""
    parts = [_sentence(order)]
    if style.include_location and not style.goal_only:
        where = _where(state, region_map)
        if where:
            parts.append(where)
    return " ".join(parts)


def _sentence(text):
    # type: (str) -> str
    """Capitalise, and end it.

    The full stop is not decoration. The server substitutes into
    "Your task is to <instruction>." and the replace consumes that stop, so a
    fragment without one of its own runs straight into "Where should you go
    next to stay on track?" and the model is handed one long malformed
    sentence.
    """
    if not text:
        return text
    ended = text if text.endswith((".", "!", "?")) else text + "."
    # NOT capitalised: it continues "Your task is to ...".
    return ended


def _where(state, region_map):
    # type: (SupervisorState, RegionMap) -> str
    """Part two: which area the aircraft is in, or which doorway it is in."""
    if state.topo == AT_DOORWAY and state.portal is not None:
        sides = [region_map.regions.get(rid) for rid in state.portal.between]
        sides = [side for side in sides if side is not None]
        # Name the room, not both sides: "the doorway between the middle spine
        # and the south-east room" is the same fact twice as long, and length
        # here is paid for in every decision of the flight.
        rooms = [side for side in sides if side.is_room]
        named = rooms[0] if rooms else (sides[0] if sides else None)
        if named is not None:
            return "You are in the doorway of %s." % named.name
    if state.region is None:
        return ""
    return "You are in %s." % state.region.name


def _order(state, region_map):
    # type: (SupervisorState, RegionMap) -> str
    """Part one: the mission, aimed at something the camera can find.

    Every branch names a physical feature and a stopping condition, because
    those are the two things a measured A/B found the policy answers with
    coordinates to -- and coordinates are the only branch that runs System 1.
    """
    mission = state.mission
    if mission is None:
        return ""
    target = region_map.regions.get(mission.target_id)
    where = _at(state.bearing)

    # OUT OF FRAME, THERE IS NO ORDER AT ALL -- and that is the conclusion of
    # every measurement this package has. Three phrasings were tried against a
    # target the camera could not see:
    #
    #   "turn to your X until you can see open floor
    #    in front of you, then fly towards it"        16 turns,  6 STOP
    #   "turn to your X until you can see an open
    #    doorway, then fly over to it"                 5 turns, 42 STOP
    #   "turn to your X and look around you"           0 turns, 59 STOP
    #
    # The first looked like the winner and it is the one that shipped. It was
    # wrong. A later ladder of orders, three passes each, found what the STOP
    # counts had hidden: THIS POLICY CANNOT DEFER A STEP. Given "do A, then
    # turn right" it turns right on the current frame -- 7 runs out of 7, about
    # fifty seconds of spinning, after which it is facing the wrong way and
    # flies off confidently in it. "Turn until you can see floor, THEN fly
    # towards it" is that same sentence: the turn is executed and the second
    # half never is. A supervised flight running this order spent 78% of its
    # answers on TURN_LEFT -- 231 of 298 -- frozen inside one room for
    # thirty-one minutes with the coverage number unchanged.
    #
    # The order that worked, 3 passes out of 3 and faster than any other, had
    # no direction word in it anywhere and named a place to end up:
    # "go through the one with a refrigerator behind it and stop inside it."
    #
    # So the answer to "the target is out of frame" is not a better sentence.
    # It is to NOT ISSUE THAT MISSION -- the chooser prefers what the camera can
    # already see, and when nothing at all is in frame it issues a scan, which
    # is a turn the policy does take and which ends on its own. Nothing below
    # ever names a direction to turn in.
    if mission.kind == SCAN_AREA:
        if target is not None and target.is_room:
            return "turn on the spot and look all around this room"
        return ("turn on the spot and look along this corridor, without going "
                "into any room")

    if mission.kind == APPROACH_DOOR:
        # Verbatim the shape of this package's best-measured instruction: a
        # thing in the frame, and a place to stop. Nothing about what comes
        # after it, because that is the next order.
        return "go to the open doorway%s and stop in front of it" % where

    if mission.kind == ENTER_ROOM:
        # Issued only from the threshold, so the door really is ahead and the
        # move really is short and straight.
        return ("fly forward through the doorway in front of you and stop "
                "inside the room")

    if mission.kind == EXIT_ROOM:
        return ("leave this room through the open doorway%s and stop in the "
                "corridor outside" % where)

    if mission.kind == TRAVERSE:
        # The target is a patch of floor about ten metres off, which may be
        # further down a corridor or the far side of the room the aircraft is
        # standing in. Name whichever it is: "along the corridor" said inside a
        # room points at nothing the camera can see.
        if target is not None and target.is_room:
            return "fly to the far side of this room and stop there"
        return ("keep flying straight ahead along this corridor to the open "
                "space in front of you")

    if mission.kind == SURVEY_COMPLETE:
        return "stop where you are, every room here has been looked into"
    return ""


def _at(bearing):
    # type: (Optional[str]) -> str
    """" ahead of you", or nothing. NEVER a side.

    The only direction this phrase book is allowed to utter is the one that
    means "straight in front of the camera", because that is the only one the
    policy acts on correctly. A side -- "on your left", "on your right" --
    is read as an instruction to turn, immediately, whatever the rest of the
    sentence says; see the note above `_order`. Naming a side is therefore not
    a hint about where something is, it is a command, and it is one this layer
    never means to give.
    """
    if bearing == "ahead of you":
        return " ahead of you"
    return ""
