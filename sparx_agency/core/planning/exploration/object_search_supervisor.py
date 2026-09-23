"""Find one named object fast: fly to a room, map it under a budget, repeat.

The sibling of :mod:`sparx_agency.core.planning.exploration.room_search_policy`,
and the state machine that closes the loop its ``DWELL`` docstring left open --
"the searching is not done by this layer; it is done by whatever explores the
room once the aircraft is inside it". Here the searching IS a state, with its
own budget and its own three ways to end, and the room order comes from a
solver rather than from a single weighted draw.

Four states, and the reason there are exactly four:

* :data:`SELECT` -- nothing is in force. Filter the ranking down to rooms
  worth flying to, hand them and the arc weights to the solver, and commit to
  the order it returns.
* :data:`TRANSIT` -- flying to the head of that order under our own planner.
  It ends four ways: arrival, a planner that never answered, a clock, or a
  follower that reported itself blocked.
* :data:`SEARCH` -- inside the room, mapping it under a per-room budget. It
  ends three ways, and all three are needed (see :attr:`ObjectSearchParams
  .search_timeout_s`).
* :data:`FOUND` -- the detector saw the target. Terminal, deliberately: the
  ``/target_seen`` latch never un-latches, so a resume path would be dead code
  that nobody could ever exercise.

**The solver is injected, exactly as the random generator is.** It is called
``solver(candidates, instance) -> Sequence[int]`` and returns room ids in visit
order; ``instance`` is passed through opaquely and never inspected here, which
is what keeps this module standard-library-only while the arc weights it is
built from need numpy and scipy. The default stub draws ONE room weighted by
probability -- exactly what ``RoomSearchPolicy`` does today -- so the machine
flies before the real solver exists, and the real solver drops in without a
line changing here.

**Why the order is re-asked rather than re-solved every tick.** The oracle
republishes continuously and its ranking is noisy. A machine that re-solved on
a timer would flip the head of its order between two near-equal rooms and fly
the aircraft back and forth between them without ever searching either -- the
same failure the smart-replan and commit-horizon work fixed for paths. So the
order is committed, and a new one is asked for only when it is spent, when its
head stops being eligible, or when the room set itself changes.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from sparx_agency.core.planning.exploration.room_search_policy import (
    Hold, RoomCandidate, RoomOption)

# -- what the search is doing ---------------------------------------------
SELECT = "select"
TRANSIT = "transit"
SEARCH = "search"
FOUND = "found"

# -- how a room's turn ended ----------------------------------------------
MAPPED = "mapped"
"""Every frontier inside the room was consumed. The room is finished."""
BUDGET_SPENT = "budget_spent"
"""The per-room clock ran out with frontier still showing."""
STALLED = "stalled"
"""The frontier count stopped falling: what is left is not reachable from here."""
UNREACHABLE = "unreachable"
"""No route was ever produced to the room."""
TRANSIT_TIMEOUT = "transit_timeout"
"""The aircraft did not arrive in time."""
BLOCKED = "blocked"
"""The follower reported itself blocked for too long on the way in."""

PRODUCTIVE = (MAPPED, BUDGET_SPENT, STALLED)
"""Verdicts that mean the room was actually visited and searched."""


@dataclass(frozen=True)
class RoomFacts:
    """What the scene graph says about one room, this tick.

    Attributes:
        room_id: The persistent room id (a ``RoomRegistry`` pid on the wire).
        frontier_clusters: Unscanned regions still inside the room. The
            search's primary done-test, and the reason it is not the ONLY
            done-test is written on :attr:`ObjectSearchParams.frontier_stall_s`.
        time_in_room_s: Cumulative dwell the mapper has credited to this room
            (tau_r). Carried for the operator payload and the oracle; the
            machine's own budget is measured from arrival, not from this,
            because this one resets when a room's pid changes.
        cells: The room's size in grid cells, so a heartbeat can say whether
            a 90 s budget was generous or mean.
    """

    room_id: int
    frontier_clusters: int = 0
    time_in_room_s: float = 0.0
    cells: int = 0


# -- what the caller should do about it -----------------------------------
@dataclass(frozen=True)
class FlyTo:
    """Plan to ``xy`` and command the aircraft there. We fly, not FALCON."""

    room_id: int
    xy: Tuple[float, float]
    label: str = "?"
    prob: float = 0.0
    order_index: int = 0
    note: str = ""


@dataclass(frozen=True)
class SearchRoom:
    """The aircraft is in the room. Map it, and do not leave it.

    Attributes:
        room_id: The room to sweep.
        xy: Where the aircraft arrived, so a caller with nothing better can
            hold station there.
        label: The room's name, for the log.
        deadline_s: When the budget expires, on the caller's own clock.
        note: Human sentence, logged and published verbatim.
    """

    room_id: int
    xy: Tuple[float, float]
    label: str = "?"
    deadline_s: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class Release:
    """The room in force is finished. Stop commanding it and say why.

    An action rather than a silent transition because the caller has real work
    to do on it: drop the route it was flying, and record the verdict. A search
    that quietly re-aims is a search nobody can debug from a recording.
    """

    room_id: Optional[int] = None
    verdict: str = ""
    note: str = ""


@dataclass(frozen=True)
class StandDown:
    """The target was seen. Stop everything and hand the aircraft over."""

    note: str = ""


Action = Union[FlyTo, SearchRoom, Hold, Release, StandDown]


@dataclass(frozen=True)
class ObjectSearchParams:
    """The knobs, all of them.

    Attributes:
        min_prob: A room ranked below this is not worth flying to and is
            dropped before the solver ever sees it.
        seed: Seed for the internal generator when none is injected. Negative
            means OS entropy, i.e. a different search every flight.
        visit_cooldown: Whether a room whose turn just ended is skipped for a
            while. Inherited from the sibling policy for the same reason, and
            with the same escape hatch: when EVERY survivor is cooling the
            cooldown is dropped for that selection rather than the aircraft
            standing still.
        visit_cooldown_s: How long that skip lasts.
        max_attempts: How many unproductive turns a room may take before it is
            deferred for :attr:`defer_s`. Three unreachable attempts is a room
            the map cannot currently route to, not a room worth a fourth try.
        defer_s: How long a room is set aside after :attr:`max_attempts`.
        arrival_tol_m: How close to the room's centre counts as being in it.
            Generous on purpose: the centre is a snapped mask mean, and an
            arrival test the aircraft cannot satisfy turns every transit into
            a timeout.
        plan_grace_s: How long a chosen room may go without the caller
            producing a single route before it is written off as unreachable.
            Judged once per goal, not per tick.
        max_transit_s: The whole clock on one transit. 120 s, not 60: the
            follower cruises at 0.30 m/s, so 60 s buys only 18 m of PATH in a
            26x55 m building, and every room across the hospital timed out
            before the aircraft could reach it. Measured on a hospital flight:
            3 transit timeouts in 9 selections at 60 s, and the rooms lost
            were exactly the distant ones a search most needs to be able to
            reach.
        blocked_abandon_s: How long the follower may report itself blocked
            before the room is abandoned. The follower has no reversing
            escape enabled, so a wedge is permanent until something else
            re-aims it, and this is that something.
        search_grace_s: How long after arrival before the done-tests are
            allowed to fire. Mandatory: on the arrival tick the room's
            frontier count is still its PRE-arrival value, and a naive test
            retires the room unmapped.
        search_timeout_s: The per-room mapping budget -- the T of the method's
            bounded local exploration.
        min_frontier_clusters: The count at or below which a room counts as
            mapped.
        frontier_clear_ticks: How many CONSECUTIVE ticks that must hold. One
            tick is a dropout; three is a fact.
        frontier_stall_s: How long the frontier count may fail to fall before
            the room is called stalled. A room whose far corner is
            permanently occluded keeps one cluster for ever -- frontier
            clusters are assigned to exactly one room by majority vote, so a
            cluster straddling a doorway is credited to the neighbour and
            never clears here. Without this exit such a room would always
            burn its whole budget.
        tick_hz: The rate the caller is expected to call :meth:`update` at.
            The machine holds no timer and does not use it; it is here so the
            node and the machine read their cadence off one dataclass.
    """

    min_prob: float = 0.01
    seed: int = -1
    visit_cooldown: bool = True
    visit_cooldown_s: float = 120.0
    max_attempts: int = 3
    defer_s: float = 180.0
    arrival_tol_m: float = 0.6
    plan_grace_s: float = 5.0
    max_transit_s: float = 120.0
    blocked_abandon_s: float = 6.0
    search_grace_s: float = 8.0
    search_timeout_s: float = 90.0
    min_frontier_clusters: int = 0
    frontier_clear_ticks: int = 3
    frontier_stall_s: float = 30.0
    tick_hz: float = 1.0


@dataclass(frozen=True)
class ObjectSearchState:
    """Everything the machine knows this tick, ready to publish or log.

    Attributes:
        state: One of :data:`SELECT`, :data:`TRANSIT`, :data:`SEARCH`,
            :data:`FOUND`.
        action: What the caller should do about it.
        room_id: The room in force, or None.
        label: That room's name.
        prob: The probability it was chosen on.
        goal_xy: Where the caller flies to, or where it arrived.
        order: The room ids the solver returned, in visit order.
        order_index: How far into that order the machine is.
        candidates: The survivors of the most recent selection, with
            renormalised probabilities -- so an operator payload can show
            what the choice was made from.
        elapsed_s: Seconds since the room in force was chosen.
        search_left_s: Seconds of mapping budget remaining, 0.0 outside
            :data:`SEARCH`.
        frontier_clusters: The room's latest frontier count, or None.
        rooms_done: How many rooms have been searched to a productive verdict.
        changed: True only on the tick a new room was committed to.
        completed: ``(room_id, verdict)`` on the tick a room's turn ends.
    """

    state: str
    action: Action
    room_id: Optional[int] = None
    label: Optional[str] = None
    prob: Optional[float] = None
    goal_xy: Optional[Tuple[float, float]] = None
    order: Tuple[int, ...] = ()
    order_index: int = 0
    candidates: Tuple[RoomCandidate, ...] = ()
    elapsed_s: float = 0.0
    search_left_s: float = 0.0
    frontier_clusters: Optional[int] = None
    rooms_done: int = 0
    changed: bool = False
    completed: Optional[Tuple[int, str]] = None


def weighted_order(candidates, instance=None):
    # type: (Sequence[RoomCandidate], Any) -> List[int]
    """The default solver: one room, drawn weighted by probability.

    Stands in for RPT* until it exists, and reproduces today's flown
    ``RoomSearchPolicy`` behaviour exactly -- the highest-ranked room is drawn
    most often, not always, because a ranking is a belief and an argmax loop
    that believes the wrong room re-flies to it for the rest of the flight.

    Returns a ONE-element order on purpose. A stub that invented a full tour
    would be asserting an ordering it has no cost information to justify, and
    the difference between one room and a tour is exactly what RPT* is for.

    Args:
        candidates: The eligible rooms, carrying ``prob_renorm``.
        instance: The arc weights. Ignored here; a real solver needs them.

    Returns:
        A single-element list holding one room id.
    """
    if not candidates:
        return []
    threshold = random.random()
    cumulative = 0.0
    for candidate in candidates:
        cumulative += float(candidate.prob_renorm)
        if threshold <= cumulative:
            return [int(candidate.room_id)]
    return [int(candidates[-1].room_id)]


class ObjectSearchSupervisor:
    """Runs the select / transit / search loop over a ranked scene graph.

    Args:
        params: Tuning.
        solver: ``solver(candidates, instance) -> Sequence[int]`` returning
            room ids in visit order. Defaults to a weighted single draw.
        rng: The generator the default draw uses. Injected so a test is
            deterministic without touching global random state.

    The contract is one call per tick: :meth:`update` takes the ranking, the
    scene graph's per-room facts, where the aircraft is and what time it is,
    and returns the state including the action to take. It reads no clock,
    plans nothing and publishes nothing, so a whole mission replays in a test
    in microseconds.
    """

    def __init__(self, params=ObjectSearchParams(), solver=None, rng=None):
        # type: (ObjectSearchParams, Optional[Callable], Optional[random.Random]) -> None
        self.params = params
        self.rng = rng if rng is not None else random.Random(
            params.seed if params.seed >= 0 else None)
        self._solver = solver
        self._state = SELECT
        self._room_id = None            # type: Optional[int]
        self._label = None              # type: Optional[str]
        self._prob = None               # type: Optional[float]
        self._goal_xy = None            # type: Optional[Tuple[float, float]]
        self._goal_s = None             # type: Optional[float]
        self._plan_s_at_goal = None     # type: Optional[float]
        self._search_end_s = None       # type: Optional[float]
        self._clear_ticks = 0
        #: Lowest frontier count seen in the room in force, and when it fell
        #: there. The stall test is "has this improved lately", which needs
        #: both numbers and cannot be recovered from the current count alone.
        self._frontier_low = None       # type: Optional[int]
        self._frontier_low_s = None     # type: Optional[float]
        self._frontier_now = None       # type: Optional[int]
        self._cooling = {}              # type: Dict[int, float]
        self._attempts = {}             # type: Dict[int, int]
        self._deferred = {}             # type: Dict[int, float]
        self._order = ()                # type: Tuple[int, ...]
        self._order_index = 0
        self._order_rooms = ()          # type: Tuple[int, ...]
        self._candidates = ()           # type: Tuple[RoomCandidate, ...]
        self._rooms_done = 0
        self.history = []               # type: List[Tuple[int, str, float]]
        self.stats = dict(selections=0, transits=0, arrivals=0, mapped=0,
                          budget_spent=0, stalls=0, plan_fails=0,
                          transit_timeouts=0, blocked=0, solver_calls=0)

    @property
    def state(self):
        # type: () -> str
        """Which of the four states is in force."""
        return self._state

    @property
    def room_id(self):
        # type: () -> Optional[int]
        """The room in force, or None."""
        return self._room_id

    def forget_rooms(self):
        # type: () -> None
        """Forget every per-room memory, because the room ids restarted.

        Cooldowns, attempt counts and deferrals are all keyed by room id, and
        the segmentation that produces those ids renumbers every room whenever
        the map's geometry changes. A memory left standing across a
        renumbering does not skip the room that was searched -- it skips
        whichever room happens to inherit the id, which is the one mistake a
        memory of visits must not make. The caller that can see the
        renumbering calls this; the machine cannot, because it is only ever
        shown ids.
        """
        self._cooling = {}
        self._attempts = {}
        self._deferred = {}
        self._order = ()
        self._order_index = 0
        self._order_rooms = ()

    # -- the tick ---------------------------------------------------------
    def update(self, rooms, facts=None, xy=None, now=0.0, last_plan_s=None,
               target_seen=False, instance=None, airborne=True,
               blocked_since=None):
        # type: (Sequence[RoomOption], Optional[Mapping[int, RoomFacts]], Optional[Tuple[float, float]], float, Optional[float], bool, Any, bool, Optional[float]) -> ObjectSearchState
        """Advance the search by one observation.

        Args:
            rooms: The ranking, one entry per room. Empty or entirely
                filtered is a safe no-op: the machine holds and waits.
            facts: ``{room_id: RoomFacts}`` from the scene graph. The search
                state's done-tests read the room in force out of this; without
                it the budget clock is the only exit.
            xy: Where the aircraft is, world metres. None -- no pose yet --
                holds, because every transition is a distance or a deadline
                measured from it.
            now: Monotonic seconds. Every deadline is relative to this.
            last_plan_s: When the caller last produced a route, same clock.
            target_seen: The detector's latch. True ends the mission.
            instance: The arc weights, passed to the solver untouched.
            airborne: Whether the aircraft is flying. False holds, and this
                matters more than it looks: taking control during a climb
                strands the aircraft on its skids.
            blocked_since: When the follower first reported itself blocked,
                or None if it is not.

        Returns:
            The state, with the action to take on it.
        """
        facts = facts or {}
        if self._room_id is not None:
            room_fact = facts.get(int(self._room_id))
            self._frontier_now = (None if room_fact is None
                                  else int(room_fact.frontier_clusters))

        if self._state == FOUND:
            # Terminal, and sticky rather than re-tested: ``/target_seen`` is a
            # latch, but a dropped subscription or a re-publish with an empty
            # payload must not restart a mission that has already succeeded.
            return self._snapshot(StandDown("target seen"), now)
        if target_seen:
            self._state = FOUND
            return self._snapshot(
                StandDown("target seen -- standing down in R%s"
                          % (self._room_id,)), now, changed=True)
        if xy is None:
            return self._snapshot(Hold("no pose yet"), now)
        if not airborne:
            return self._snapshot(Hold("waiting for the aircraft to be airborne"),
                                  now)
        if self._state == SELECT:
            return self._select(rooms, now, last_plan_s, instance)
        if self._state == TRANSIT:
            return self._transit(xy, now, last_plan_s, blocked_since)
        return self._search(now)

    # -- the three working states -----------------------------------------
    def _select(self, rooms, now, last_plan_s, instance):
        # type: (Sequence[RoomOption], float, Optional[float], Any) -> ObjectSearchState
        """Commit to the head of an order, asking for a new one if needed."""
        self._candidates = self._eligible(rooms, now)
        if not self._candidates:
            return self._snapshot(Hold("no room worth flying to"), now)
        by_id = dict((c.room_id, c) for c in self._candidates)
        live = tuple(sorted(by_id))

        head = self._next_in_order(by_id)
        if head is None or live != self._order_rooms:
            solver = self._solver if self._solver is not None else self._draw
            order = solver(self._candidates, instance) or []
            self._order = tuple(int(r) for r in order)
            self._order_index = 0
            self._order_rooms = live
            self.stats["solver_calls"] += 1
            head = self._next_in_order(by_id)
        if head is None:
            return self._snapshot(Hold("the order held no eligible room"), now)

        pick = by_id[head]
        self._state = TRANSIT
        self._room_id, self._label, self._prob = pick.room_id, pick.label, pick.prob
        self._goal_xy = pick.xy
        self._goal_s = now
        self._plan_s_at_goal = last_plan_s
        self.stats["selections"] += 1
        self.stats["transits"] += 1
        note = ("R%d (%s) chosen at p=%.2f -- %d of %d in the order, %d candidates"
                % (pick.room_id, pick.label, pick.prob, self._order_index + 1,
                   max(1, len(self._order)), len(self._candidates)))
        return self._snapshot(
            FlyTo(room_id=pick.room_id, xy=pick.xy, label=pick.label,
                  prob=pick.prob, order_index=self._order_index, note=note),
            now, changed=True)

    def _transit(self, xy, now, last_plan_s, blocked_since):
        # type: (Tuple[float, float], float, Optional[float], Optional[float]) -> ObjectSearchState
        """Hold the chosen room until arrival, a dead planner, a clock or a wedge."""
        params = self.params
        distance = math.hypot(self._goal_xy[0] - xy[0], self._goal_xy[1] - xy[1])
        elapsed = now - self._goal_s
        if distance < params.arrival_tol_m:
            self._state = SEARCH
            self._search_end_s = now + params.search_timeout_s
            self._clear_ticks = 0
            self._frontier_low = self._frontier_now
            self._frontier_low_s = now
            self.stats["arrivals"] += 1
            note = ("arrived at R%d (%.2f m) after %.1f s -- mapping it for "
                    "up to %.0f s" % (self._room_id, distance, elapsed,
                                      params.search_timeout_s))
            return self._snapshot(
                SearchRoom(room_id=self._room_id, xy=(float(xy[0]), float(xy[1])),
                           label=self._label or "?",
                           deadline_s=self._search_end_s, note=note),
                now, changed=True)
        if blocked_since is not None and (now - blocked_since) > params.blocked_abandon_s:
            self.stats["blocked"] += 1
            return self._end_room(
                BLOCKED, "R%d: follower blocked for %.0f s -- abandoning"
                % (self._room_id, now - blocked_since), now)
        planned = (last_plan_s is not None
                   and (self._plan_s_at_goal is None
                        or last_plan_s > self._plan_s_at_goal))
        if elapsed > params.plan_grace_s and not planned:
            self.stats["plan_fails"] += 1
            return self._end_room(
                UNREACHABLE, "no route to R%d in %.0f s -- centre unreachable"
                % (self._room_id, params.plan_grace_s), now)
        if elapsed > params.max_transit_s:
            self.stats["transit_timeouts"] += 1
            return self._end_room(
                TRANSIT_TIMEOUT, "R%d not reached in %.0f s -- giving up"
                % (self._room_id, params.max_transit_s), now)
        return self._snapshot(
            Hold("flying to R%d, %.2f m to run" % (self._room_id, distance)), now)

    def _search(self, now):
        # type: (float) -> ObjectSearchState
        """Map the room under its budget, and decide when its turn is over."""
        params = self.params
        elapsed = now - self._goal_s if self._goal_s is not None else 0.0
        left = max(0.0, (self._search_end_s or now) - now)
        since_arrival = (0.0 if self._search_end_s is None
                         else params.search_timeout_s - left)

        frontier = self._frontier_now
        if frontier is not None:
            if self._frontier_low is None or frontier < self._frontier_low:
                self._frontier_low = frontier
                self._frontier_low_s = now
            if frontier <= params.min_frontier_clusters:
                self._clear_ticks += 1
            else:
                self._clear_ticks = 0

        # Order matters. MAPPED is the truest verdict whenever it holds, so it
        # is tested first. BUDGET_SPENT then beats STALLED: once the clock is
        # gone the room's turn is over on the budget, and reporting a stall
        # instead would blame the geometry for a decision the budget made.
        if (since_arrival >= params.search_grace_s and frontier is not None
                and self._clear_ticks >= params.frontier_clear_ticks):
            self.stats["mapped"] += 1
            return self._end_room(
                MAPPED, "R%d mapped -- no frontier left after %.0f s"
                % (self._room_id, since_arrival), now)
        if left <= 0.0:
            self.stats["budget_spent"] += 1
            return self._end_room(
                BUDGET_SPENT, "R%d budget of %.0f s spent with %s clusters left"
                % (self._room_id, params.search_timeout_s, frontier), now)
        if (since_arrival >= params.search_grace_s
                and self._frontier_low_s is not None
                and (now - self._frontier_low_s) > params.frontier_stall_s):
            self.stats["stalls"] += 1
            return self._end_room(
                STALLED,
                "R%d stalled at %s clusters for %.0f s -- nothing further "
                "visible from in here"
                % (self._room_id, self._frontier_low,
                   now - self._frontier_low_s), now)
        return self._snapshot(
            Hold("mapping R%d, %.0f s left, %s clusters"
                 % (self._room_id, left, frontier), ), now)

    # -- the selection ----------------------------------------------------
    def _eligible(self, rooms, now):
        # type: (Sequence[RoomOption], float) -> Tuple[RoomCandidate, ...]
        """Filter the ranking down to what may be chosen, and renormalise it."""
        survivors = [room for room in rooms
                     if room.xy is not None
                     and float(room.prob) >= self.params.min_prob
                     and not self._is_deferred(room.room_id, now)]
        if self.params.visit_cooldown:
            fresh = [room for room in survivors
                     if not self._is_cooling(room.room_id, now)]
            # EVERY survivor cooling is not a reason to stand still. The
            # cooldown exists to spread the search out, and a search that
            # stops entirely is strictly worse than one that repeats a room.
            if fresh:
                survivors = fresh
        total = sum(float(room.prob) for room in survivors)
        if total <= 0.0:
            return ()
        return tuple(
            RoomCandidate(room_id=int(room.room_id), label=str(room.label),
                          prob=float(room.prob),
                          prob_renorm=float(room.prob) / total,
                          xy=(float(room.xy[0]), float(room.xy[1])))
            for room in survivors)

    def _draw(self, candidates, instance=None):
        # type: (Sequence[RoomCandidate], Any) -> List[int]
        """The built-in stub solver, on the machine's own injected generator."""
        if not candidates:
            return []
        threshold = self.rng.random()
        cumulative = 0.0
        for candidate in candidates:
            cumulative += float(candidate.prob_renorm)
            if threshold <= cumulative:
                return [int(candidate.room_id)]
        return [int(candidates[-1].room_id)]

    def _next_in_order(self, by_id):
        # type: (Mapping[int, RoomCandidate]) -> Optional[int]
        """The first still-eligible room at or after the current order index."""
        while self._order_index < len(self._order):
            room_id = self._order[self._order_index]
            if room_id in by_id:
                return int(room_id)
            self._order_index += 1
        return None

    # -- bookkeeping ------------------------------------------------------
    def _end_room(self, verdict, note, now):
        # type: (str, str, float) -> ObjectSearchState
        """Finish the room in force with a verdict, and go back to SELECT."""
        room_id = self._room_id
        if room_id is not None:
            self._cooling[int(room_id)] = float(now)
            self.history.append((int(room_id), str(verdict), float(now)))
            if verdict in PRODUCTIVE:
                self._rooms_done += 1
                self._attempts.pop(int(room_id), None)
            else:
                tries = self._attempts.get(int(room_id), 0) + 1
                self._attempts[int(room_id)] = tries
                if tries >= self.params.max_attempts:
                    self._deferred[int(room_id)] = float(now)
        self._order_index += 1
        self._release()
        return self._snapshot(
            Release(room_id=room_id, verdict=verdict, note=note), now,
            completed=(int(room_id), str(verdict)) if room_id is not None else None)

    def _is_cooling(self, room_id, now):
        # type: (int, float) -> bool
        """Whether this room's turn ended too recently to choose it again."""
        ended = self._cooling.get(int(room_id))
        return ended is not None and (now - ended) < self.params.visit_cooldown_s

    def _is_deferred(self, room_id, now):
        # type: (int, float) -> bool
        """Whether this room failed too often to be worth another attempt yet."""
        since = self._deferred.get(int(room_id))
        if since is None:
            return False
        if (now - since) >= self.params.defer_s:
            self._deferred.pop(int(room_id), None)
            self._attempts.pop(int(room_id), None)
            return False
        return True

    def _release(self):
        # type: () -> None
        """Drop the room in force. The next tick selects a new one."""
        self._state = SELECT
        self._room_id = self._label = self._prob = None
        self._goal_xy = self._goal_s = None
        self._plan_s_at_goal = self._search_end_s = None
        self._clear_ticks = 0
        self._frontier_low = self._frontier_low_s = self._frontier_now = None

    def _snapshot(self, action, now, changed=False, completed=None):
        # type: (Action, float, bool, Optional[Tuple[int, str]]) -> ObjectSearchState
        """Freeze what is known this tick around ``action``."""
        elapsed = 0.0 if self._goal_s is None else now - self._goal_s
        left = 0.0
        if self._state == SEARCH and self._search_end_s is not None:
            left = max(0.0, self._search_end_s - now)
        return ObjectSearchState(
            state=self._state, action=action, room_id=self._room_id,
            label=self._label, prob=self._prob, goal_xy=self._goal_xy,
            order=self._order, order_index=self._order_index,
            candidates=self._candidates, elapsed_s=elapsed,
            search_left_s=left, frontier_clusters=self._frontier_now,
            rooms_done=self._rooms_done, changed=changed, completed=completed)
