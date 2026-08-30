"""Turn "map the hospital" into one small, concrete, reachable order at a time.

The deficit this closes was measured, not assumed. Five recorded flights under
*"Explore the entire hospital. Enter every room you pass..."* ended with 9-16 %
of the building seen, and four of the five stopped themselves part-way through:
System 2 emitted a real ``STOP``, coverage froze at that instant to the cell, and
the remaining minutes added nothing. The policy is not built to hold a goal that
has no state at which it is satisfied. It is built to fly to a thing it can see.

So this layer holds the goal instead, and hands down only what the policy is good
at: **one bounded mission at a time, aimed at a physical feature.** Four of them,
which is all a floor plan needs:

* :data:`SCAN_AREA` -- turn and look around where you are, without leaving it.
* :data:`APPROACH_DOOR` -- go and stand in front of that doorway.
* :data:`ENTER_ROOM` -- now go through it.
* :data:`EXIT_ROOM` -- come back out of this one.
* :data:`TRAVERSE` -- carry on along the corridor to the next stretch.

``STOP`` then changes meaning entirely, and that is the point of the design.
Under one unsatisfiable order it ended the flight. Under a sequence of small ones
it is the policy saying *this* one is finished, which is a claim it is far better
placed to make -- and the supervisor issues the next mission, so the aircraft
flies on. It is still only a **hint** here: geometry decides. The aircraft is
inside the room or it is not, the room's floor has been seen or it has not, and
those are things the map knows and the model is guessing at.

Nothing in here touches the flight loop, and nothing in here is ROS. It consumes
a pose, a seen-mask and an optional stop hint; it produces a mission. What
carries a mission to the aircraft is one string on one topic.

Python 3.8 syntax, numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from sparx_agency.core.planning.exploration.region_coverage import (
    RegionCoverage,
    RegionProgress,
)
from sparx_agency.core.planning.exploration.region_map import Portal, Region, RegionMap

# ── what the aircraft is doing, topologically ────────────────────────────
IN_CORRIDOR = "in_corridor"
INSIDE_ROOM = "inside_room"
AT_DOORWAY = "at_doorway"
OFF_MAP = "off_map"

# ── the four missions, and the end ───────────────────────────────────────
SCAN_AREA = "scan_area"
#: Getting to the threshold and crossing it are two orders, not one. Measured
#: over 211 System-2 decisions: asked to "go through the doorway and stop inside
#: the room beyond it", the policy answers with an arrow rather than
#: coordinates, and only the coordinate branch produces a flyable curve. The
#: package's own counterbalanced A/B found its best-performing instruction of
#: four was "Go to the doorway ahead of you and stop in front of it" -- the
#: approach half, alone. It is also the split ``EnterPortalBehavior`` has always
#: used, which is why portals carry a normal.
APPROACH_DOOR = "approach_door"
ENTER_ROOM = "enter_room"
EXIT_ROOM = "exit_room"
TRAVERSE = "traverse"
SURVEY_COMPLETE = "survey_complete"


@dataclass(frozen=True)
class Mission:
    """One bounded order, and everything needed to judge it finished.

    Attributes:
        kind: One of the five constants above.
        target_id: The region the mission is about -- the room to enter, the
            area to scan, the corridor stretch to reach.
        portal_id: The opening it goes through, where one is involved.
        issued_s: When it was handed down.
        seen_at_issue: The target's coverage fraction at that moment, so
            progress can be measured against it rather than against zero.
        probe_xy: The unseen cell this hop was chosen for. Distinct from
            ``target_xy``, and the distinction is the whole point: a hop is
            capped at one step, so on a long leg the aircraft is aimed at a
            waypoint part of the way there -- over floor it has usually
            already seen. Judging the hop by whether *that* has been seen ends
            it on the tick it is issued, which is exactly what happened: six
            hops in a row completed instantly, the area was retired for being
            "issued too often", and the survey stopped at 12.5%.
        target_xy: An explicit world point, when the mission is "go over
            there" rather than "go to that region". Used by :data:`TRAVERSE`,
            which aims at the nearest patch of floor the survey has not seen
            rather than at a region centre that may be forty metres away.
        note: Short human reason, for the log and the operator.
    """

    kind: str
    target_id: int
    portal_id: Optional[int] = None
    target_xy: Optional[Tuple[float, float]] = None
    probe_xy: Optional[Tuple[float, float]] = None
    issued_s: float = 0.0
    seen_at_issue: float = 0.0
    note: str = ""

    @property
    def key(self) -> Tuple[str, int]:
        """What a deferral is keyed on: this kind of mission, this target."""
        return (self.kind, self.target_id)


@dataclass(frozen=True)
class SupervisorParams:
    """The knobs, all of them.

    Attributes:
        scanned_fraction: Share of a region's floor that counts as scanned.
            Well under 1.0 on purpose -- the corner behind a bed is occluded
            from every pose the aircraft can reach.
        doorway_radius_m: How close to a portal centre counts as being in it,
            and therefore when an approach is finished and a crossing may start.
        approach_offset_m: How far back from a doorway the approach aims, along
            the portal normal on the aircraft's own side. Far enough that the
            opening is comfortably in frame, close enough that crossing it is
            then a short straight move.
        min_portal_m: An opening narrower than this is not offered as a way
            through. The airframe is 0.63 m wide and this world's doors are
            0.90 m, so the margin is real and small.
        mission_timeout_s: How long one mission may run before it is given up
            on. A policy that cannot get through a door will not start being
            able to.
        defer_s: How long a given-up mission is left alone before it may be
            chosen again -- long enough that the supervisor tries everything
            else first, short enough that a door blocked by a passing obstacle
            is retried within a flight.
        max_issues_multiple: A backstop against loops nobody has found yet.
            No mission key may be ISSUED more than this many times its attempt
            ceiling in one flight, whatever the reason -- four separate loops in
            this state machine were each discovered only by watching one run
            them, and a constant bound turns the next one from a lost flight
            into a line in the log.
        max_attempts: How many times one mission may be given up on before it
            is abandoned for the rest of the flight. Without a ceiling the
            deferral only paces a loop instead of ending it, and a survey whose
            last unreachable room keeps coming back round can never report
            itself finished.
        scan_stall_s: While scanning, how long coverage may fail to rise before
            the area counts as done for want of anything more to see.
        scan_stall_gain: How much it has to rise inside that window to count as
            still making progress, as a fraction of the region.
        stop_hint_min_fraction: A ``STOP`` during a scan is accepted as
            completion only above this coverage -- below it, the model is
            stopping for its own reasons and the area genuinely is not scanned.
        travel_step_m: How far a "go and look over there" order reaches. Sized
            to the rescan radius, so arriving somewhere new is immediately worth
            a fresh look around, and to the decision rate: at ~22 s a decision
            and ~1 m of route per decision, ten metres is about ten decisions,
            which fits inside one mission budget. Aiming at the next region
            instead meant orders of thirty and forty metres, and four of five
            of those were given up on.
        travel_arrive_m: How close counts as having got there. Rarely the
            thing that ends a hop -- seeing the target normally comes first.
        travel_timeout_s: A hop's own clock, shorter than the general one. A
            hop ends the moment its target patch is seen, so one still running
            after this is one the aircraft is not making progress on, and the
            cheapest answer is a different patch rather than more waiting.
            Crossing a doorway is the opposite case and keeps the long clock.
        refuse_after_s: How long a hop may sit under a STOP before the target
            is written off. STOP is the policy saying it will not fly this
            order, and it does not change its mind: measured over one flight
            it was 46% of every answer given, and every second spent waiting
            for it to lift was a second the aircraft stood still. Long enough
            only to outlast the one STOP that follows a genuine arrival.
        refuse_radius_m: How wide a patch a refusal writes off, so the next
            target is somewhere else rather than the cell next door.
        in_frame_deg: Half the camera's field of view. Inside it a target is
            something the policy can be shown; outside it the target has to be
            turned to first, and every out-of-frame order is one the policy is
            measurably worse at.
        turn_cost_m_per_deg: What a degree of turning is worth in metres of
            flying, when choosing between targets. The aircraft yaws at about
            2.1 deg/s and covers about a metre per decision, so turning ninety
            degrees costs roughly what flying twenty metres does -- and unlike
            flying, it surveys almost nothing on the way. Graded, not a cliff:
            a target just outside the frame is barely penalised, one behind is
            heavily so, and when nothing else is left it is still reachable.
        rescan_radius_m: How far the aircraft must move before an area it has
            already looked around may be looked around again. A scan clears the
            VICINITY, not the region: the middle spine is forty metres long, and
            "60% of it seen from one end" written off as finished leaves its far
            third unsurveyed for the rest of the campaign. Set larger than the
            biggest room, so one look still finishes a room.
        first_try_rooms_first: Try every room once before re-attacking one that
            has already been given up on. Measured on this building, flying the
            corridors and spinning would see 79.6% of the floor and 13 of its 20
            rooms clear the scanned threshold without ever being entered -- so
            corridor coverage is worth more than a second attempt at a door, and
            the instruction still gets its one honest try at every room.
        bearing_hold_s: How long the direction word in an order stays fixed
            before it may be recomputed. It has to be held: the phrase is
            derived from live yaw, the order is republished whenever its text
            changes, and the policy turns 15 degrees at a time -- so recomputing
            it every tick renames the target faster than a 0.2-0.4 Hz model can
            act on it, and the two chase each other.
        nudge_after_s: How long a mission may run without the aircraft moving
            before the supervisor asks for a short reverse. A drone wedged
            against a jamb gets the same frame and the same answer for ever, so
            the cheapest thing that changes anything is to break contact.
        nudge_min_move_m: How far it has to have travelled since the mission was
            issued to count as making progress rather than being stuck.
        nudge_cooldown_s: Minimum gap between two requests, so a genuinely
            immovable aircraft does not reverse continuously.
        arrival_grace_s: A mission is not judged for this long after it is
            issued, so a room the aircraft is still standing in does not
            instantly satisfy an order to leave it.
    """

    scanned_fraction: float = 0.60
    doorway_radius_m: float = 1.2
    approach_offset_m: float = 1.6
    min_portal_m: float = 0.80
    mission_timeout_s: float = 75.0
    defer_s: float = 180.0
    max_attempts: int = 3
    max_issues_multiple: int = 4
    travel_step_m: float = 10.0
    travel_arrive_m: float = 2.0
    travel_timeout_s: float = 60.0
    refuse_after_s: float = 12.0
    refuse_radius_m: float = 4.0
    in_frame_deg: float = 35.0
    turn_cost_m_per_deg: float = 0.4
    rescan_radius_m: float = 9.0
    first_try_rooms_first: bool = True
    bearing_hold_s: float = 20.0
    nudge_after_s: float = 35.0
    nudge_min_move_m: float = 0.6
    nudge_cooldown_s: float = 25.0
    scan_stall_s: float = 25.0
    scan_stall_gain: float = 0.02
    stop_hint_min_fraction: float = 0.35
    arrival_grace_s: float = 4.0


@dataclass(frozen=True)
class SupervisorState:
    """Everything the supervisor knows this tick, ready to be phrased or logged.

    Attributes:
        topo: One of :data:`IN_CORRIDOR`, :data:`INSIDE_ROOM`,
            :data:`AT_DOORWAY`, :data:`OFF_MAP`.
        region: Where the aircraft is, or None off the map.
        portal: The opening it is standing in, when ``topo`` is at a doorway.
        mission: The order in force, or None before the first one.
        bearing: Where the mission's target lies relative to the nose --
            ``"ahead"``, ``"on your right"``, ``"on your left"``,
            ``"behind you"`` -- or None when there is no target to point at.
        range_m: How far that target is.
        rooms_scanned: How many rooms are ticked off.
        rooms_total: How many there are.
        fraction_seen: Share of the whole building's floor seen.
        changed: True on the tick a new mission was issued.
        completed: The mission that just finished, on the tick it finished.
        nudge: True on the tick the supervisor wants the aircraft backed off a
            little -- it has been on this mission a while and has not moved.
    """

    topo: str
    region: Optional[Region]
    portal: Optional[Portal]
    mission: Optional[Mission]
    bearing: Optional[str]
    range_m: Optional[float]
    rooms_scanned: int
    rooms_total: int
    fraction_seen: float
    changed: bool = False
    completed: Optional[Mission] = None
    nudge: bool = False


class ExplorationSupervisor:
    """Holds the survey, and issues one mission at a time.

    Args:
        region_map: The building.
        coverage: Per-region progress over the shared seen mask.
        params: Tuning.

    The contract is one call per tick: :meth:`update` takes where the aircraft
    is and what has been seen, and returns the state, including the mission in
    force. It never blocks, never talks to a network, and holds no timers of its
    own -- every deadline is measured against the ``now`` it is given, so a test
    can run a whole survey in a millisecond.
    """

    def __init__(self, region_map, coverage, params=SupervisorParams()):
        # type: (RegionMap, RegionCoverage, SupervisorParams) -> None
        self.region_map = region_map
        self.coverage = coverage
        self.params = params
        self._mission = None       # type: Optional[Mission]
        self._deferred = {}        # type: Dict[Tuple[str, int], float]
        self._attempts = {}        # type: Dict[Tuple[str, int], int]
        self._exhausted = set()    # type: Set[Tuple[str, int]]
        #: Rooms whose scan is over, however it ended. NOT the same as
        #: "scanned to the threshold": a policy that stops early, and an area
        #: with nothing more visible in it, both finish the mission without
        #: reaching it. Without this the room stays eligible, the next choice is
        #: the same scan, and the next STOP completes it again -- measured in
        #: flight as eight identical "scanned (policy stopped)" verdicts on one
        #: room in ninety seconds while the aircraft sat in its doorway.
        self._accepted = set()     # type: Set[int]
        #: WHERE each accepted scan was flown from. A corridor is not finished
        #: because it was looked at once: it is finished where it was looked at.
        self._scans = {}           # type: Dict[int, List[Tuple[float, float]]]
        self._issues = {}          # type: Dict[Tuple[str, int], int]
        self._issued_at = None     # type: Optional[Tuple[float, float]]
        self._refused = []         # type: List[Tuple[Tuple[float, float], float]]
        self._bearing = None       # type: Optional[str]
        self._bearing_s = -1e9
        self._moved_at = 0.0
        self._last_nudge_s = -1e9
        self._best_seen = 0.0      # coverage of the current target, high-water
        self._best_seen_s = 0.0    # ...and when it was last beaten
        self.history = []          # type: List[Tuple[Mission, str, float]]

    # -- the tick ---------------------------------------------------------

    def update(self, x, y, yaw, seen_mask, now, stop_hint=False, busy=False):
        # type: (float, float, float, np.ndarray, float, bool, bool) -> SupervisorState
        """Advance the survey by one observation.

        Args:
            x: World x of the aircraft, metres.
            y: World y, metres.
            yaw: Heading, radians CCW from +x, used only to say which side of
                the aircraft a target is on.
            seen_mask: The coverage mask, indexed like the region grid.
            now: Monotonic seconds. All deadlines are relative to this.
            stop_hint: Whether the policy has just claimed the task is done.
            busy: Whether the aircraft is deliberately stationary right now --
                thinking, turning, settling or dipping for a look-down. It is
                not stuck, and reversing it would undo work it is in the middle
                of. Measured in flight: three nudges in one run at an aircraft
                that was rotating freely with nothing in front of it.

        Returns:
            The state, with the mission that should now be in force.
        """
        progress = self.coverage.progress(seen_mask)
        region = self.coverage.note_pose(x, y)
        portal = self._portal_here(x, y)
        topo = self._topo(region, portal)

        # A doorway cell belongs to one side or the other and the aircraft
        # standing in it is genuinely in both. Choose the side that still has
        # work on it, or the supervisor spends the whole approach re-deciding
        # which room it is about to be in.
        working = self._working_region(region, portal, progress, now)

        completed = None
        if self._mission is not None:
            verdict = self._judge(self._mission, region, progress, now,
                                  stop_hint, x, y, seen_mask)
            if verdict is not None:
                completed = self._mission
                self.history.append((self._mission, verdict, now))
                productive = verdict not in _UNPRODUCTIVE_VERDICTS
                if productive:
                    # A MISSION THAT ACHIEVED SOMETHING CLEARS ITS OWN TALLY.
                    # The issue ceiling exists to catch loops, and a loop by
                    # definition never succeeds -- but the tally is kept per
                    # (kind, area), so a large area needing twenty honest hops
                    # tripped it as surely as a loop did. Measured: the atrium
                    # was retired after twelve consecutive hops that all ended
                    # "in view" or "arrived", with the survey at 15.6% and
                    # every room in the building still to do.
                    self._issues.pop(self._mission.key, None)
                if self._mission.kind == SCAN_AREA and productive:
                    self._accepted.add(self._mission.target_id)
                    if np.isfinite(x) and np.isfinite(y):
                        self._scans.setdefault(
                            self._mission.target_id, []).append((float(x), float(y)))
                if verdict in _UNFINISHED_VERDICTS:
                    # Not only "given up". An area that stopped yielding
                    # anything new is equally unfinished, and re-choosing it the
                    # instant it ends is a loop that issues a mission every
                    # stall window for the rest of the flight while the
                    # aircraft goes nowhere -- measured in simulation as the
                    # last 1900 seconds of a 2000 second survey.
                    self._defer(self._mission.key, now)
                self._mission = None

        changed = False
        if self._mission is None:
            self._mission = self._choose(working, progress, now, x, y,
                                         standing_in=region, yaw=yaw,
                                         portal=portal, seen=seen_mask)
            if self._mission is not None and self._count_issue(self._mission):
                # Issued too many times to be making progress. Retire it, record
                # why, and take whatever the next tick chooses instead.
                self.history.append((self._mission, "retired: issued too often", now))
                self._exhausted.add(self._mission.key)
                self._accepted.add(self._mission.target_id)
                self._mission = None
            changed = self._mission is not None
            if changed:
                self._best_seen = self._mission.seen_at_issue
                self._best_seen_s = now
                # Re-anchor WHERE it started from, but NOT WHEN it last
                # moved: how long the aircraft has been stationary is a fact
                # about the aircraft, not about the mission in force. Resetting
                # it here meant a wedged drone whose hops were being refused
                # every fifteen seconds never reached the stuck threshold at
                # all -- measured, with the follower reporting HARD BLOCKED at
                # 0.33 m and the survey frozen for nine minutes, no nudge was
                # ever asked for. The per-tick branch in `_wants_nudge` clears
                # it the moment the aircraft actually travels.
                self._issued_at = (x, y)

        # The bearing points at the DOORWAY -- that is the thing in the frame
        # the instruction names, and the only kind of referent the policy
        # answers to with coordinates. Where the aircraft has to end up is a
        # different point, past it; see `aim_point`.
        #
        # HELD, not recomputed every tick. The order is republished whenever its
        # text changes and this phrase is derived from live yaw, so a model that
        # answers a turn every seven seconds gets its target renamed before it
        # can act on the name -- the two chase each other and nothing moves.
        live, range_m = _relative_to(x, y, yaw, self.look_point(self._mission))
        if changed or self._bearing is None \
                or (now - self._bearing_s) >= self.params.bearing_hold_s:
            self._bearing, self._bearing_s = live, now
        bearing = self._bearing
        done, total, fraction = self.coverage.summary(progress)
        return SupervisorState(
            nudge=self._wants_nudge(x, y, now, busy),
            topo=topo, region=region, portal=portal, mission=self._mission,
            bearing=bearing, range_m=range_m, rooms_scanned=done,
            rooms_total=total, fraction_seen=fraction,
            changed=changed, completed=completed)

    # -- where are we -----------------------------------------------------

    def _working_region(self, region, portal, progress, now):
        # type: (Optional[Region], Optional[Portal], Dict[int, RegionProgress], float) -> Optional[Region]
        """Which side of a doorway the aircraft should be treated as being on.

        Not where it is -- where the work is. Crossing a threshold, the label
        under the aircraft flips between the room and the corridor from one
        frame to the next, and a mission chosen from the raw label flips with
        it: entered, left, entered, left, at a metre a second, for ever.
        """
        if region is None or portal is None:
            return region
        sides = [self.region_map.regions.get(rid) for rid in portal.between]
        sides = [s for s in sides if s is not None]
        # "Worth visiting", not "unscanned". A room that has been looked around
        # as far as it can be still reads unscanned for ever, so treating the
        # doorway as belonging to it orders an exit from a room the aircraft is
        # not in -- which succeeds instantly, because it is not in it -- and
        # then orders it again. Measured in flight: twelve identical exits from
        # one room in two minutes.
        pending = [s for s in sides if self._worth_visiting(s.id, progress, now)]
        if len(pending) == 1:
            return pending[0]
        # Both done, or neither: prefer the corridor, which is where the next
        # mission is chosen from anyway.
        for side in sides:
            if not side.is_room:
                return side
        return region

    def _wants_nudge(self, x, y, now, busy=False):
        # type: (float, float, float, bool) -> bool
        """Has this mission stalled with the aircraft in one place?

        Not the depth reflex's job and not a duplicate of it: that one fires
        when the corridor ahead is blocked *right now*, and this one fires when
        nothing has changed for half a minute, whatever the depth says. The
        recovery is the same manoeuvre either way -- the follower's, which
        already knows how to back off without hitting anything it has not just
        been sitting in.
        """
        if self._mission is None or self._issued_at is None:
            return False
        if self._mission.kind in (SCAN_AREA, SURVEY_COMPLETE):
            return False        # standing still is the mission
        if busy:
            # Thinking, turning, settling, dipping: all legitimately motionless,
            # and at ~22 s a decision the aircraft spends a lot of a flight in
            # them. Reversing out of a rotation is worse than doing nothing.
            self._moved_at = now
            return False
        if (now - self._last_nudge_s) < self.params.nudge_cooldown_s:
            return False
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        moved = math.hypot(x - self._issued_at[0], y - self._issued_at[1])
        if moved >= self.params.nudge_min_move_m:
            # It is going somewhere. Re-anchor, and reset the clock with it.
            self._issued_at = (x, y)
            self._moved_at = now
            return False
        # The test is "nothing has changed for a while", NOT "it has not moved
        # far since the last anchor" -- an aircraft travelling steadily only
        # re-anchors every `nudge_min_move_m`, so between anchors it looks
        # motionless and would be nudged mid-flight.
        if (now - self._moved_at) < self.params.nudge_after_s:
            return False
        self._last_nudge_s = now
        return True

    def _tried(self, region_id):
        # type: (int) -> bool
        """Has any attempt at getting into this room been made and lost?"""
        rid = int(region_id)
        return any(self._attempts.get((kind, rid), 0) > 0
                   for kind in (APPROACH_DOOR, ENTER_ROOM))

    def _at(self, portal, x, y):
        # type: (Optional[Portal], Optional[float], Optional[float]) -> bool
        """Is the aircraft standing in this opening?"""
        if portal is None or x is None or y is None:
            return False
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return math.hypot(portal.centre[0] - x,
                          portal.centre[1] - y) <= self.params.doorway_radius_m

    def _portal_here(self, x, y):
        # type: (float, float) -> Optional[Portal]
        """The opening the aircraft is standing in, if any -- nearest wins."""
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        best, best_d = None, self.params.doorway_radius_m
        for portal in self.region_map.portals.values():
            if portal.width_m < self.params.min_portal_m:
                continue
            d = math.hypot(portal.centre[0] - x, portal.centre[1] - y)
            if d <= best_d:
                best, best_d = portal, d
        return best

    def _topo(self, region, portal):
        # type: (Optional[Region], Optional[Portal]) -> str
        if region is None:
            return OFF_MAP
        if portal is not None:
            # A doorway outranks the region it is counted in: standing in one is
            # the moment an enter or an exit is about to succeed or fail, and it
            # is the only state where "which region am I in" is genuinely
            # ambiguous rather than merely uncertain.
            return AT_DOORWAY
        return INSIDE_ROOM if region.is_room else IN_CORRIDOR

    # -- is the mission finished ------------------------------------------

    def _judge(self, mission, region, progress, now, stop_hint, x=None,
               y=None, seen=None):
        # type: (Mission, Optional[Region], Dict[int, RegionProgress], float, bool, Optional[float], Optional[float], Optional[np.ndarray]) -> Optional[str]
        """Why this mission is over, or None if it is not.

        Geometry decides. ``stop_hint`` is the policy's own claim and is
        accepted only where it is corroborated -- during a scan, above a floor
        of real coverage -- because a model that says STOP for its own reasons
        would otherwise walk the supervisor through the whole checklist without
        the aircraft moving.
        """
        age = now - mission.issued_s
        if age < self.params.arrival_grace_s:
            return None

        here = region.id if region is not None else None
        entry = progress.get(mission.target_id)
        fraction = entry.fraction if entry is not None else 0.0

        if fraction > self._best_seen + self.params.scan_stall_gain:
            self._best_seen, self._best_seen_s = fraction, now

        if mission.kind == APPROACH_DOOR:
            portal = (self.region_map.portals.get(mission.portal_id)
                      if mission.portal_id is not None else None)
            if here == mission.target_id:
                return "arrived early"     # went straight in; take it
            if self._at(portal, x, y):
                return "at the door"
            if stop_hint and age >= self.params.refuse_after_s:
                # Same as a hop: STOP is the policy declining this door, and it
                # does not change its mind. One flight sat under a single
                # unbroken run of 51 STOPs on one door approach -- the whole
                # mission clock, stationary. Take the attempt and go elsewhere;
                # `max_attempts` still brings it back later, from a different
                # place and a different heading.
                return "refused"
        elif mission.kind == ENTER_ROOM:
            if here == mission.target_id:
                return "entered"
        elif mission.kind == EXIT_ROOM:
            if here is not None and here != mission.target_id:
                return "left"
        elif mission.kind == TRAVERSE:
            if mission.target_xy is not None:
                # SEEING THE PATCH IS THE POINT; STANDING ON IT IS NOT. The
                # camera reaches ten metres, so a patch of unseen floor is
                # normally surveyed from well short of it and often from off to
                # one side. Waiting for the aircraft to come within two metres
                # of a cell it has already looked at means the hop can only end
                # by timing out: measured, two hops in a row ran the full 400 s
                # and were given up on while coverage climbed fifteen points
                # during them. The flying was working; the finish line was in
                # the wrong place.
                #
                # BOTH TESTS ARE AGAINST THE CELL, NEVER THE WAYPOINT. The
                # waypoint is one step along the way and is routinely both
                # already-seen and within arm's reach, so judging either
                # condition by it ends the hop on the tick it is issued --
                # measured too: six hops completing instantly, the area retired
                # for being "issued too often", the survey stopped at 12.5%.
                probe = mission.probe_xy or mission.target_xy
                if seen is not None and self._seen_at(seen, probe):
                    return "in view"
                if (x is not None and y is not None
                        and np.isfinite(x) and np.isfinite(y)
                        and math.hypot(probe[0] - x, probe[1] - y)
                        <= self.params.travel_arrive_m):
                    return "arrived"
                if stop_hint and age >= self.params.refuse_after_s:
                    # NOT a completion. The policy has looked at this order and
                    # declined it, and the supervisor's answer is a different
                    # patch of floor -- immediately, not when the clock runs
                    # out. `_refused` keeps this one off the list for a while
                    # so the replacement is genuinely somewhere else.
                    self._refused.append((mission.target_xy, now))
                    return "refused"
            elif here == mission.target_id:
                return "arrived"
        elif mission.kind == SCAN_AREA:
            if fraction >= self.coverage.scanned_fraction:
                return "scanned"
            if stop_hint and fraction >= self.params.stop_hint_min_fraction:
                return "scanned (policy stopped)"
            if (now - self._best_seen_s) >= self.params.scan_stall_s:
                # Nothing new for a while. The rest of this area is behind
                # something, and waiting longer buys nothing.
                return "nothing further visible"
        elif mission.kind == SURVEY_COMPLETE:
            return None

        limit = (self.params.travel_timeout_s if mission.kind == TRAVERSE
                 else self.params.mission_timeout_s)
        if age >= limit:
            return "given up"
        return None

    def _turn_cost(self, off_deg):
        # type: (Any) -> Any
        """What it costs, in metres of flying, to bring a target into frame.

        Zero inside the field of view and rising steadily outside it. The old
        rule was a 200 m cliff at a hundred degrees, which on a camera that
        sees seventy-five made everything from 38 to 100 degrees look free --
        and those are exactly the targets the aircraft cannot see and the
        policy will not fly to.

        Works on a scalar or an array; numpy handles both.
        """
        beyond = np.maximum(0.0, np.abs(off_deg) - self.params.in_frame_deg)
        return beyond * self.params.turn_cost_m_per_deg

    def _not_refused(self, xs, ys, x, y, now):
        # type: (np.ndarray, np.ndarray, float, float, float) -> np.ndarray
        """Which of these candidates the policy has not just declined.

        Compared as the *orders they would become*, not as cells. A hop is
        capped at one step, so every unseen cell along the same line out of
        here collapses to the same waypoint and the same sentence -- refusing
        one has to refuse all of them, or the supervisor reissues the order the
        policy just turned down with a different cell behind it.

        Refusals expire: the aircraft moves, the view changes, and a patch that
        could not be flown to from one end of a corridor is often ordinary from
        the other. Holding them forever would shrink the building with every
        STOP.
        """
        keep = np.ones(xs.shape, dtype=bool)
        cutoff = now - self.params.defer_s
        self._refused = [(pt, t) for pt, t in self._refused if t >= cutoff]
        if not self._refused:
            return keep
        step = self.params.travel_step_m
        dx, dy = xs - x, ys - y
        d = np.hypot(dx, dy)
        scale = np.where(d > step, step / np.maximum(d, 1e-9), 1.0)
        wx, wy = x + dx * scale, y + dy * scale
        r2 = self.params.refuse_radius_m ** 2
        for (rx, ry), _ in self._refused:
            keep &= ((wx - rx) ** 2 + (wy - ry) ** 2) > r2
        return keep

    def _seen_at(self, seen, point):
        # type: (np.ndarray, Tuple[float, float]) -> bool
        """Has the camera reached this world point yet?

        Off the grid counts as seen: a target that is no longer on the map is
        not one the aircraft can be sent to, and calling it unseen would hold
        the mission open until it timed out.
        """
        col, row = self.region_map.cell_of(point[0], point[1])
        grid = np.asarray(seen, dtype=bool)
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return True
        return bool(grid[row, col])

    # -- what next --------------------------------------------------------

    def _choose(self, region, progress, now, x=None, y=None, standing_in=None,
                yaw=None, portal=None, seen=None):
        # type: (Optional[Region], Dict[int, RegionProgress], float, Optional[float], Optional[float], Optional[Region], Optional[float], Optional[Portal], Optional[np.ndarray]) -> Optional[Mission]
        """The next mission, in priority order, from where the aircraft is.

        The order is the one a person would use, and it matters: look around
        where you are before deciding where to go, because the doorways you can
        act on are the ones you have seen.
        """
        if region is None:
            return None            # off the map; say nothing until it is back

        def issue(kind, target, portal=None, note=""):
            entry = progress.get(target)
            return Mission(kind=kind, target_id=target,
                           portal_id=portal.id if portal is not None else None,
                           issued_s=now,
                           seen_at_issue=entry.fraction if entry else 0.0,
                           note=note)

        # 0. STANDING IN THE DOORWAY OF A ROOM STILL TO DO: go in.
        #    This outranks looking around, and the ranking is the whole point of
        #    the two-step entry. Scanning from the threshold is what the earlier
        #    campaigns did: it clears a room on the coverage scoreboard without
        #    the aircraft ever crossing, which is not what the instruction asks
        #    for and leaves the far half of the room permanently occluded.
        if portal is not None and standing_in is not None:
            far = self.region_map.regions.get(portal.other(standing_in.id)) \
                if standing_in.id in portal.between else None
            if (far is not None and far.is_room
                    and self._worth_visiting(far.id, progress, now)
                    and self._eligible(ENTER_ROOM, far.id, now)):
                entry = progress.get(far.id)
                return Mission(kind=ENTER_ROOM, target_id=far.id,
                               portal_id=portal.id, issued_s=now,
                               seen_at_issue=entry.fraction if entry else 0.0,
                               note="cross into %s" % far.name)

        # 1. Look around here first, room or corridor alike -- unless this
        #    particular spot has already been looked around from. A region is
        #    finished WHERE it was scanned, not as a whole: forty metres of
        #    spine written off after one look from one end is most of a corridor
        #    nobody ever surveys.
        if self._scannable(region, progress, now, x, y):
            return issue(SCAN_AREA, region.id, note="look around %s" % region.name)

        # 2. In a room that is done: come back out. Deferral applies here too --
        #    without it, a room whose exit has just been given up on is left
        #    ordering the same exit for ever, which is the one loop a supervisor
        #    can fall into that looks from outside exactly like working.
        # Only from inside it. `region` here may be the doorway's other side,
        # and an exit ordered from the corridor is satisfied the moment it is
        # judged -- the aircraft is already out.
        inside = standing_in is None or standing_in.id == region.id
        if region.is_room and inside:
            portal = self._way_out(region)
            if portal is not None and self._eligible(EXIT_ROOM, region.id, now):
                return issue(EXIT_ROOM, region.id, portal,
                             note="leave %s" % region.name)
            if portal is None:
                return issue(SCAN_AREA, region.id,
                             note="no way out of %s is wide enough" % region.name)
        elif region.is_room and standing_in is not None:
            # At a room's doorway from the corridor side, with nothing left to
            # do in that room. Carry on from where the aircraft actually is.
            region = standing_in

        # 3. In a corridor: the nearest unscanned room off it, in two orders.
        #    Stand in front of the door first, cross it second. One order that
        #    asks for both is the one the policy answers with an arrow.
        def room_order(room, portal, tried):
            if self._at(portal, x, y):
                return issue(ENTER_ROOM, room.id, portal,
                             note="cross into %s" % room.name)
            if self._eligible(APPROACH_DOOR, room.id, now):
                return issue(APPROACH_DOOR, room.id, portal,
                             note="%s the door of %s"
                                  % ("back to" if tried else "go to", room.name))
            return issue(ENTER_ROOM, room.id, portal, note="enter %s" % room.name)

        # 3a. Somewhere in THIS area the camera has not reached. One short move
        #     at a time: ten metres is about ten decisions, which fits a mission
        #     budget, and arriving is immediately worth a fresh look around
        #     because it is a rescan radius away. Aiming at the next region
        #     instead meant thirty- and forty-metre orders, and four in five of
        #     those were given up on.
        found = self._nearest_unseen(region, x, y, yaw, seen, now)
        if found is not None:
            spot, probe = found
            entry = progress.get(region.id)
            return Mission(kind=TRAVERSE, target_id=region.id, target_xy=spot,
                           probe_xy=probe, issued_s=now,
                           seen_at_issue=entry.fraction if entry else 0.0,
                           note="on to an unseen part of %s" % region.name)

        # 3b. A room nobody has tried yet, once this area is surveyed. The
        #     instruction asks for every room and every room gets one honest
        #     attempt -- but AFTER the floor the aircraft is standing on, not
        #     before it. Tried first, five untried doors off one corridor is
        #     five 400 s attempts, which is most of a ninety-minute flight spent
        #     on the one thing this policy has never once managed. A door it is
        #     already standing at is still taken immediately, by rule 0 above.
        room, portal = self._nearest_room_off(region, progress, now, x, y, yaw,
                                              untried_only=True)
        if room is not None:
            return room_order(room, portal, tried=False)

        # 4. Then the corridors, ahead of a second go at a door that has already
        #    been given up on. Measured on this building: flying the corridors
        #    and spinning sees 79.6% of the floor, and thirteen of its twenty
        #    rooms clear the scanned threshold without ever being entered. A
        #    corridor the aircraft can reach is worth more than a doorway it has
        #    already failed to cross once.
        hop = self._next_hop(region, progress, now)
        if hop is not None:
            neighbour, portal = hop
            return issue(TRAVERSE, neighbour.id, portal,
                         note="on towards %s" % neighbour.name)

        # 5. Only now, another go at a room that did not work the first time.
        room, portal = self._nearest_room_off(region, progress, now, x, y, yaw)
        if room is not None:
            return room_order(room, portal, tried=True)

        return Mission(kind=SURVEY_COMPLETE, target_id=region.id, issued_s=now,
                       note="every room reachable from here has been scanned")

    def _count_issue(self, mission):
        # type: (Mission) -> bool
        """Record that this mission was issued; True if it has been too often."""
        count = self._issues.get(mission.key, 0) + 1
        self._issues[mission.key] = count
        ceiling = self.params.max_attempts * self.params.max_issues_multiple
        return count > ceiling

    def _defer(self, key, now):
        # type: (Tuple[str, int], float) -> None
        """Put a mission aside, and give up on it entirely after enough tries."""
        attempts = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempts
        if attempts >= self.params.max_attempts:
            self._exhausted.add(key)
        else:
            self._deferred[key] = now + self.params.defer_s

    def _scannable(self, region, progress, now, x, y):
        # type: (Region, Dict[int, RegionProgress], float, Optional[float], Optional[float]) -> bool
        """Is looking around from here worth an order?"""
        if not self._eligible(SCAN_AREA, region.id, now):
            return False
        if self.coverage.is_scanned(progress, region.id):
            return False
        if region.is_room:
            # A room is smaller than the rescan radius by construction, so one
            # look finishes it and the vicinity test would never fire.
            return region.id not in self._accepted
        return not self._scanned_near(region.id, x, y)

    def _scanned_near(self, region_id, x, y):
        # type: (int, Optional[float], Optional[float]) -> bool
        """Has an accepted scan of this region already been flown from here?"""
        vantages = self._scans.get(int(region_id))
        if not vantages:
            return False
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            return True         # nowhere to compare: treat as covered
        return any(math.hypot(x - vx, y - vy) <= self.params.rescan_radius_m
                   for vx, vy in vantages)

    def _eligible(self, kind, target_id, now):
        # type: (str, int, float) -> bool
        key = (kind, int(target_id))
        if key in self._exhausted:
            return False
        until = self._deferred.get(key)
        return until is None or now >= until

    def _way_out(self, room):
        # type: (Region) -> Optional[Portal]
        """The widest opening from a room into a corridor."""
        for portal in self.region_map.portals_of(room.id):
            if portal.width_m < self.params.min_portal_m:
                continue
            other = self.region_map.regions.get(portal.other(room.id))
            if other is not None and not other.is_room:
                return portal
        return None

    def _nearest_room_off(self, corridor, progress, now, x=None, y=None, yaw=None,
                          untried_only=False):
        # type: (Region, Dict[int, RegionProgress], float, Optional[float], Optional[float], Optional[float], bool) -> Tuple[Optional[Region], Optional[Portal]]
        """The best unscanned room off this corridor: near, and already in view.

        Near to the AIRCRAFT, not to the corridor's centre -- measuring from the
        centre sends it back past doors it has just flown by, which on a long
        spine is most of them.

        And **in front of it**, which matters more than distance. Measured over
        five flights the aircraft turns at about 2.1 deg/s, so bringing
        something behind it into frame costs more than a whole mission's budget:
        a nearer door over its shoulder is a worse target than a further one it
        can already see. Doors behind are still eligible -- when they are all
        that is left, one has to be chosen -- they just lose every tie.
        """
        from_x = corridor.centre[0] if x is None or not np.isfinite(x) else x
        from_y = corridor.centre[1] if y is None or not np.isfinite(y) else y
        best = (None, None, float("inf"))
        for neighbour, portal in self.region_map.neighbours(corridor.id):
            if not neighbour.is_room or portal.width_m < self.params.min_portal_m:
                continue
            if not self._worth_visiting(neighbour.id, progress, now):
                continue
            if untried_only and self._tried(neighbour.id):
                continue
            cost = math.hypot(portal.centre[0] - from_x, portal.centre[1] - from_y)
            if yaw is not None and np.isfinite(yaw):
                off = abs(math.degrees(_relative_angle(
                    from_x, from_y, yaw, portal.centre)))
                cost += float(self._turn_cost(off))
            if cost < best[2]:
                best = (neighbour, portal, cost)
        return (best[0], best[1])

    def _nearest_unseen(self, region, x, y, yaw, seen, now):
        # type: (Region, Optional[float], Optional[float], Optional[float], Optional[np.ndarray], float) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
        """``(waypoint, unseen cell)`` for this area, or None if it is done.

        Two points, not one. The waypoint is where to aim -- capped at a step,
        so the order stays a short one -- and the cell is what the hop is
        actually for, and the only honest test of whether it succeeded.

        Restricted to the region the aircraft is already in, deliberately: a
        straight line to the nearest unseen cell anywhere in the building runs
        through walls, and this order is handed to a policy that is looking for
        somewhere to fly, not solving a maze. Crossing into another area is the
        portal graph's job, one hop below.

        Ties are broken towards what the camera can already see, for the same
        reason every other target is: at ~2.1 deg/s, turning to face something
        costs more than most of a mission.
        """
        if seen is None or x is None or y is None \
                or not (np.isfinite(x) and np.isfinite(y)):
            return None
        unseen = (self.region_map.mask_of(region.id)
                  & self.coverage.countable & ~np.asarray(seen, dtype=bool))
        rows, cols = np.nonzero(unseen)
        if not rows.size:
            return None
        res = self.region_map.resolution
        xs = self.region_map.origin_x + (cols + 0.5) * res
        ys = self.region_map.origin_y + (rows + 0.5) * res
        keep = self._not_refused(xs, ys, x, y, now)
        if not keep.all():
            if not keep.any():
                return None       # everything here has been declined; move on
            xs, ys = xs[keep], ys[keep]
        d = np.hypot(xs - x, ys - y)
        # Far enough to be a journey, near enough to be one mission.
        far_enough = d >= self.params.travel_arrive_m * 1.5
        if not far_enough.any():
            return None
        cost = np.where(far_enough, d, np.inf)
        if yaw is not None and np.isfinite(yaw):
            bearing = np.abs((np.arctan2(ys - y, xs - x) - yaw + np.pi)
                             % (2.0 * np.pi) - np.pi)
            cost = cost + self._turn_cost(np.degrees(bearing))
        pick = int(np.argmin(cost))
        if not np.isfinite(cost[pick]):
            return None
        # Step towards it rather than all the way, when it is far off -- and
        # hand back the cell as well, because that is what finishing means.
        probe = (float(xs[pick]), float(ys[pick]))
        return (_towards((x, y), probe, self.params.travel_step_m), probe)

    def _next_hop(self, region, progress, now):
        # type: (Region, Dict[int, RegionProgress], float) -> Optional[Tuple[Region, Portal]]
        """One step along the portal graph towards the nearest unfinished work.

        A breadth-first search rather than a nearest-in-metres guess, because
        the aircraft travels through doorways and the shortest straight line to
        an unscanned room routinely points at the wall behind it.
        """
        wanted = self._unfinished(progress, now)
        if not wanted:
            return None
        frontier = [(region.id, None)]   # type: List[Tuple[int, Optional[Tuple[Region, Portal]]]]
        seen = {region.id}               # type: Set[int]
        while frontier:
            nxt = []
            for rid, first in frontier:
                for neighbour, portal in self.region_map.neighbours(rid):
                    if neighbour.id in seen:
                        continue
                    if portal.width_m < self.params.min_portal_m:
                        continue
                    seen.add(neighbour.id)
                    step = first if first is not None else (neighbour, portal)
                    if neighbour.id in wanted:
                        return step
                    if not neighbour.is_room:
                        nxt.append((neighbour.id, step))
            frontier = nxt
        return None

    def _unfinished(self, progress, now):
        # type: (Dict[int, RegionProgress], float) -> Set[int]
        """Regions still worth travelling to."""
        return set(rid for rid in progress
                   if self._worth_visiting(rid, progress, now))

    def _worth_visiting(self, region_id, progress, now):
        # type: (int, Dict[int, RegionProgress], float) -> bool
        """Is there anything left to gain by going to this region?

        Three conditions, and the third is the one that stops the supervisor
        pacing. A room the aircraft has already been into, and looked around in
        as far as it could, stays below the scanned threshold for ever if its
        far corner is occluded -- so "not scanned" alone keeps re-selecting it,
        the entry succeeds every time, and the survey oscillates in and out of
        the same doorway at a mission every four seconds. Once its scan has
        been given up on for good, the room is finished as far as any flight
        can finish it.
        """
        rid = int(region_id)
        entry = progress.get(rid)
        if entry is None or entry.fraction >= self.coverage.scanned_fraction:
            return False
        if (SCAN_AREA, rid) in self._exhausted:
            return False
        if not entry.region.is_room:
            # A corridor stays worth going to while any of it is unseen: it is
            # finished at the vantages it was scanned from, not as a whole.
            return self._eligible(SCAN_AREA, rid, now)
        if rid in self._accepted:
            return False
        # A room is finished when BOTH ways at it are spent: the approach and
        # the crossing fail differently -- one is "could not get to the door",
        # the other "could not get through it" -- and giving up on the first
        # should not retire a room a direct attempt might still enter.
        if (APPROACH_DOOR, rid) in self._exhausted \
                and (ENTER_ROOM, rid) in self._exhausted:
            return False
        return (self._eligible(APPROACH_DOOR, rid, now)
                or self._eligible(ENTER_ROOM, rid, now))

    def look_point(self, mission):
        # type: (Optional[Mission]) -> Optional[Tuple[float, float]]
        """What the instruction points at: the doorway, the spot, else the area."""
        if mission is None or mission.kind == SURVEY_COMPLETE:
            return None
        if mission.target_xy is not None:
            return mission.target_xy
        if mission.portal_id is not None:
            portal = self.region_map.portals.get(mission.portal_id)
            if portal is not None:
                return portal.centre
        region = self.region_map.regions.get(mission.target_id)
        return region.centre if region is not None else None

    def aim_point(self, mission):
        # type: (Optional[Mission]) -> Optional[Tuple[float, float]]
        """Where the aircraft has to end up for the mission to be satisfied.

        Past the doorway, not in it. A target on the threshold is a target the
        aircraft can reach without having gone anywhere, and the region label
        there belongs to whichever side won a tie -- so an "enter" aimed at the
        opening can be flown perfectly and still never complete.
        """
        if mission is None or mission.kind == SURVEY_COMPLETE:
            return None
        if mission.target_xy is not None:
            return mission.target_xy
        region = self.region_map.regions.get(mission.target_id)
        if mission.kind == SCAN_AREA:
            return region.centre if region is not None else None
        portal = (self.region_map.portals.get(mission.portal_id)
                  if mission.portal_id is not None else None)
        if mission.kind == APPROACH_DOOR and portal is not None:
            # Short of the opening, on the side the aircraft is on -- which is
            # the side the target room is NOT. Backing off along the normal is
            # exactly EnterPortalBehavior's approach waypoint.
            near = self.region_map.regions.get(portal.other(mission.target_id))
            if near is not None:
                return _towards(portal.centre, near.centre,
                                self.params.approach_offset_m)
            return portal.centre
        if mission.kind == EXIT_ROOM and portal is not None:
            far = self.region_map.regions.get(portal.other(mission.target_id))
            return _towards(portal.centre, far.centre) if far is not None else None
        if region is None:
            return None
        if portal is None:
            return region.centre
        return _towards(portal.centre, region.centre)


#: How far past a doorway "beyond it" is, metres. Far enough to be
#: unambiguously on the far side of a 0.30 m threshold and inside the smallest
#: room this building has, which is 8 m2.
THROUGH_M = 1.5


def _towards(origin, destination, metres=None):
    # type: (Tuple[float, float], Tuple[float, float], Optional[float]) -> Tuple[float, float]
    """A point ``metres`` from ``origin`` towards ``destination``."""
    reach = THROUGH_M if metres is None else float(metres)
    dx, dy = destination[0] - origin[0], destination[1] - origin[1]
    distance = math.hypot(dx, dy)
    if distance <= reach or distance < 1e-6:
        return destination
    scale = reach / distance
    return (origin[0] + dx * scale, origin[1] + dy * scale)


#: Verdicts that end a mission without achieving it. Each one costs the target
#: an attempt, and enough of them retire it for the rest of the flight.
#:
#: "nothing further visible" is deliberately NOT one of them any more. It used
#: to be, to stop an area being re-scanned the instant its scan ended -- but
#: that job now belongs to the vicinity rule, which remembers WHERE each scan
#: was flown from. Deferring the whole region as well blocks looking at a
#: different part of it, which on a forty-metre corridor is the part that most
#: needs looking at.
_UNFINISHED_VERDICTS = ("given up",)

#: Verdicts that mean the mission achieved nothing, and so leave the "issued
#: too often" tally standing. Everything else clears it: the tally is a loop
#: detector, and a loop never succeeds.
_UNPRODUCTIVE_VERDICTS = ("given up", "refused")


def _relative_angle(x, y, yaw, target):
    # type: (float, float, float, Tuple[float, float]) -> float
    """Signed angle from the nose to a world point, radians in (-pi, pi]."""
    relative = math.atan2(target[1] - y, target[0] - x) - yaw
    return (relative + math.pi) % (2.0 * math.pi) - math.pi


def _relative_to(x, y, yaw, target):
    # type: (float, float, float, Optional[Tuple[float, float]]) -> Tuple[Optional[str], Optional[float]]
    """Which way to look for the target, in words the instruction can use.

    The one thing that turns a coordinate the supervisor knows into something
    the policy can act on: it answers with a pixel when it is told about a thing
    in the frame in front of it, and "on your right" is what puts it there.
    """
    if target is None or not all(np.isfinite(v) for v in (x, y, yaw)):
        return (None, None)
    dx, dy = target[0] - x, target[1] - y
    range_m = float(math.hypot(dx, dy))
    if range_m < 1e-3:
        return ("right here", range_m)
    relative = math.atan2(dy, dx) - yaw
    relative = (relative + math.pi) % (2.0 * math.pi) - math.pi
    degrees = math.degrees(relative)
    if abs(degrees) <= 35.0:
        return ("ahead of you", range_m)
    if abs(degrees) >= 145.0:
        return ("behind you", range_m)
    return ("on your left" if degrees > 0 else "on your right", range_m)
