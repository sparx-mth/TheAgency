"""Let a room ranking choose where the aircraft flies next, one room at a time.

The sibling of :mod:`sparx_agency.core.planning.exploration.mission`, and the
same shape: a pure state machine over injected facts, one call per tick, no
clock of its own and no I/O. Where ``ExplorationSupervisor`` turns "map the
building" into one reachable order, this turns a *probability distribution over
rooms* -- the thing a language model is actually good at producing -- into one
reachable order. It is the half of the flown search loop that closes it: the
oracle's ranking stops being a picture on a dashboard and becomes the goal.

Three states, and the reason there are exactly three:

* :data:`IDLE` -- nothing is being pursued. Draw one room from the ranking,
  weighted by it, and go.
* :data:`PURSUING` -- flying to that room's centroid. It ends three ways:
  arrival, a planner that never answered, or a clock.
* :data:`DWELL` -- standing in the room that was chosen, deliberately silent.
  The searching is not done by this layer; it is done by whatever explores the
  room once the aircraft is inside it. Issuing the next goal immediately would
  drag the aircraft straight back out of the room it just asked to see.

**Sampling, not argmax.** The highest-ranked room is drawn most often, not
always. A ranking is a belief and a belief is wrong: an argmax loop that
believes the wrong room re-flies to it every cycle for the rest of the flight,
whereas sampling spends most of its time on the model's best guess and still
visits the rest. The draw is over SURVIVORS ONLY -- rooms under
``min_prob`` and rooms with no centroid to fly to are dropped first and the
remaining mass renormalised -- so a filter never quietly biases the draw toward
whichever rooms happened to be dropped.

ONE DELIBERATE DEPARTURE FROM THE FLOWN VERSION, and it is
:attr:`RoomSearchParams.visit_cooldown`. The flown sampler had no memory at
all: the room it had just spent fifteen seconds inside was as likely to be
drawn next as any other, and the only thing discouraging it was the oracle's
own ``tau`` term -- a *soft* penalty computed by a language model that only
sees a room's dwell time if the mapper's accounting is running. So the cooldown
here is geometry's own answer to the same problem: a room whose pursuit has
just ENDED, however it ended, is ineligible for
:attr:`RoomSearchParams.visit_cooldown_s`. "However it ended" is deliberate --
an unreachable centroid is exactly as worth skipping as a searched room, and
without it the sampler retries the blocked room every ``plan_grace_s`` for
ever. It cannot deadlock: when every survivor is cooling, the cooldown is
dropped for that draw rather than the aircraft standing still. Set it False to
reproduce the flown behaviour exactly.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

# -- what the search is doing ---------------------------------------------
IDLE = "idle"
PURSUING = "pursuing"
DWELL = "dwell"


@dataclass(frozen=True)
class RoomOption:
    """One room the ranking scored, and somewhere in it to fly to.

    Attributes:
        room_id: The room's persistent id (a ``RoomRegistry`` pid on the wire).
        prob: Its share of the ranking, as published. Not assumed normalised --
            the policy renormalises over whatever survives its own filter.
        xy: World centroid, metres. None when the scene graph has no centroid
            for this room yet, which makes it unflyable and drops it.
        label: Human name, carried through only so the log and the operator
            payload can say "ward" rather than "R7".
    """

    room_id: int
    prob: float
    xy: Optional[Tuple[float, float]] = None
    label: str = "?"


@dataclass(frozen=True)
class RoomCandidate:
    """A survivor of the filter, with its share of the surviving mass.

    Attributes:
        room_id: The room's id.
        label: Its name.
        prob: The probability the ranking gave it.
        prob_renorm: That probability over the survivors only -- what the draw
            actually used, and the number worth showing an operator who is
            wondering why an unlikely-looking room was chosen.
        xy: The centroid this candidate would be flown to.
    """

    room_id: int
    label: str
    prob: float
    prob_renorm: float
    xy: Tuple[float, float]


@dataclass(frozen=True)
class PublishGoal:
    """A room was just drawn: plan to ``xy`` and command the aircraft there."""

    room_id: int
    xy: Tuple[float, float]
    label: str = "?"
    prob: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class Hold:
    """Nothing to issue this tick. Whatever is in force stays in force."""

    note: str = ""


@dataclass(frozen=True)
class ReSample:
    """The pursuit in force has ended. The next tick draws a new room.

    Emitted as an action rather than handled silently because the caller has
    work to do on it: stop commanding the route it was flying, and say in the
    log *why* the room was abandoned. A search that quietly re-aims is a search
    nobody can debug from a recording.
    """

    note: str = ""


Action = Union[PublishGoal, Hold, ReSample]


@dataclass(frozen=True)
class RoomSearchParams:
    """The knobs, all of them. Defaults are the flown values.

    Attributes:
        min_prob: A room ranked below this is not worth flying to and is
            dropped before the draw.
        arrival_tol_m: How close to the centroid counts as being in the room.
            Generous on purpose: a centroid is the mean of a room's floor
            cells, which on an L-shaped room can sit inside its own wall, and
            an arrival test the aircraft cannot satisfy turns every pursuit
            into a timeout.
        plan_grace_s: How long a chosen room may go without the caller
            producing a single route to it before it is written off as
            unreachable. Judged ONCE per goal -- "has any plan been made since
            this room was chosen" -- not per tick, so a route that is planned
            and then goes stale is caught by ``max_pursue_s``, not by this.
        max_pursue_s: The whole clock on one room. An aircraft that has not
            arrived in this long is not about to.
        dwell_after_arrival_s: How long to stay silent after arriving, so
            whatever explores a room gets the aircraft to itself.
        tick_hz: The rate the caller is expected to call :meth:`update` at.
            The policy holds no timer and does not use this -- it is here so
            the node and the policy read their cadence off one dataclass
            instead of two.
        seed: Seed for the internal :class:`random.Random` when no generator is
            injected. Negative means OS entropy, i.e. a different search every
            flight; a non-negative value makes a whole run reproducible.
        visit_cooldown: Whether a room whose pursuit just ended is skipped for
            a while. See the module docstring -- this is the one deliberate
            departure from the flown sampler. False reproduces it exactly.
        visit_cooldown_s: How long that skip lasts. Longer than
            ``dwell_after_arrival_s`` by a wide margin, or it expires before
            the aircraft has left the room it is meant to stop it returning to.
    """

    min_prob: float = 0.01
    arrival_tol_m: float = 0.6
    plan_grace_s: float = 5.0
    max_pursue_s: float = 60.0
    dwell_after_arrival_s: float = 15.0
    tick_hz: float = 1.0
    seed: int = -1
    visit_cooldown: bool = True
    visit_cooldown_s: float = 120.0


@dataclass(frozen=True)
class RoomSearchState:
    """Everything the policy knows this tick, ready to be published or logged.

    Attributes:
        state: One of :data:`IDLE`, :data:`PURSUING`, :data:`DWELL`.
        action: What the caller should do about it.
        room_id: The room being pursued or dwelt in, or None when idle.
        label: That room's name.
        prob: The probability it was drawn on, as the ranking gave it.
        goal_xy: Its centroid -- where the caller plans to.
        candidates: The survivors of the most recent draw, with renormalised
            probabilities. Kept after the draw so an operator payload can still
            show what the choice was made from.
        elapsed_s: Seconds since the room in force was chosen.
        dwell_left_s: Seconds of dwell remaining, 0.0 outside :data:`DWELL`.
        changed: True only on the tick a new room was drawn.
    """

    state: str
    action: Action
    room_id: Optional[int] = None
    label: Optional[str] = None
    prob: Optional[float] = None
    goal_xy: Optional[Tuple[float, float]] = None
    candidates: Tuple[RoomCandidate, ...] = ()
    elapsed_s: float = 0.0
    dwell_left_s: float = 0.0
    changed: bool = False


class RoomSearchPolicy:
    """Draws a room from a ranking and holds the pursuit until it ends.

    Args:
        params: Tuning.
        rng: The generator the draw uses. Injected so a test is deterministic
            without touching global random state; defaults to a
            :class:`random.Random` seeded from ``params.seed``.

    The contract is one call per tick: :meth:`update` takes the ranking, where
    the aircraft is, the time, and when the caller last produced a route, and
    returns the state including the action to take. It reads no clock, plans
    nothing and publishes nothing, so a whole search can be replayed in a test
    in microseconds.
    """

    def __init__(self, params=RoomSearchParams(), rng=None):
        # type: (RoomSearchParams, Optional[random.Random]) -> None
        self.params = params
        self.rng = rng if rng is not None else random.Random(
            params.seed if params.seed >= 0 else None)
        self._state = IDLE
        self._room_id = None        # type: Optional[int]
        self._label = None          # type: Optional[str]
        self._prob = None           # type: Optional[float]
        self._goal_xy = None        # type: Optional[Tuple[float, float]]
        self._goal_s = None         # type: Optional[float]
        #: The caller's last-plan stamp AT THE MOMENT the room was chosen. The
        #: grace test is "is there a newer one than this", which is why the
        #: snapshot has to be taken and not re-read.
        self._plan_s_at_goal = None  # type: Optional[float]
        self._dwell_end_s = None    # type: Optional[float]
        self._cooling = {}          # type: Dict[int, float]
        self._candidates = ()       # type: Tuple[RoomCandidate, ...]
        self.stats = dict(samples=0, arrivals=0, plan_fails=0, timeouts=0,
                          dwell_completes=0)

    @property
    def state(self):
        # type: () -> str
        """Which of the three states is in force."""
        return self._state

    def forget_visits(self):
        # type: () -> None
        """Forget which rooms were visited, because the room ids restarted.

        A cooldown is keyed by ``room_id``, and the segmentation that produces
        those ids renumbers every room whenever the map's geometry changes. A
        cooldown left standing across a renumbering does not skip the room that
        was searched -- it skips whichever room happens to inherit a cooling
        id, which is the one mistake a memory of visits must not make. The
        caller that can see the renumbering calls this; the policy cannot,
        because it is only ever shown ids.
        """
        self._cooling = {}

    # -- the tick ---------------------------------------------------------

    def update(self, rooms, xy, now, last_plan_s=None):
        # type: (Sequence[RoomOption], Optional[Tuple[float, float]], float, Optional[float]) -> RoomSearchState
        """Advance the search by one observation.

        Args:
            rooms: The ranking, one entry per room. An empty or all-filtered
                sequence is a safe no-op: the policy holds and waits.
            xy: Where the aircraft is, world metres. None -- no pose yet --
                holds, because every transition out of a pursuit is a distance
                or a deadline measured from it.
            now: Monotonic seconds. Every deadline is relative to this.
            last_plan_s: When the caller last produced a route, on the same
                clock, or None if it never has.

        Returns:
            The state, with the action to take on it.
        """
        if xy is None:
            return self._snapshot(Hold("no pose yet"), now)
        if self._state == IDLE:
            return self._draw(rooms, now, last_plan_s)
        if self._state == PURSUING:
            return self._pursue(xy, now, last_plan_s)
        return self._dwell(now)

    # -- the three states -------------------------------------------------

    def _draw(self, rooms, now, last_plan_s):
        # type: (Sequence[RoomOption], float, Optional[float]) -> RoomSearchState
        """Pick one room from the ranking and commit to it."""
        self._candidates = self._eligible(rooms, now)
        if not self._candidates:
            return self._snapshot(Hold("no room worth flying to"), now)
        pick = self._candidates[
            self._weighted_index([c.prob_renorm for c in self._candidates])]
        self._state = PURSUING
        self._room_id, self._label, self._prob = pick.room_id, pick.label, pick.prob
        self._goal_xy = pick.xy
        self._goal_s = now
        self._plan_s_at_goal = last_plan_s
        self.stats["samples"] += 1
        note = ("R%d (%s) drawn at p=%.2f (%.2f of %d candidates)"
                % (pick.room_id, pick.label, pick.prob, pick.prob_renorm,
                   len(self._candidates)))
        return self._snapshot(
            PublishGoal(room_id=pick.room_id, xy=pick.xy, label=pick.label,
                        prob=pick.prob, note=note),
            now, changed=True)

    def _pursue(self, xy, now, last_plan_s):
        # type: (Tuple[float, float], float, Optional[float]) -> RoomSearchState
        """Hold the room in force until arrival, a dead planner or a clock."""
        params = self.params
        distance = math.hypot(self._goal_xy[0] - xy[0], self._goal_xy[1] - xy[1])
        elapsed = now - self._goal_s
        if distance < params.arrival_tol_m:
            self._state = DWELL
            self._dwell_end_s = now + params.dwell_after_arrival_s
            self._note_visited(now)
            self.stats["arrivals"] += 1
            return self._snapshot(
                Hold("arrived at R%d (%.2f m) after %.1f s -- dwelling %.0f s"
                     % (self._room_id, distance, elapsed,
                        params.dwell_after_arrival_s)), now)
        planned = (last_plan_s is not None
                   and (self._plan_s_at_goal is None
                        or last_plan_s > self._plan_s_at_goal))
        if elapsed > params.plan_grace_s and not planned:
            self.stats["plan_fails"] += 1
            return self._abandon(
                "no route to R%d in %.0f s -- centroid unreachable"
                % (self._room_id, params.plan_grace_s), now)
        if elapsed > params.max_pursue_s:
            self.stats["timeouts"] += 1
            return self._abandon(
                "R%d not reached in %.0f s -- giving up"
                % (self._room_id, params.max_pursue_s), now)
        return self._snapshot(
            Hold("pursuing R%d, %.2f m to run" % (self._room_id, distance)), now)

    def _dwell(self, now):
        # type: (float) -> RoomSearchState
        """Stay silent in the room that was reached, then release it."""
        if self._dwell_end_s is None or now >= self._dwell_end_s:
            self.stats["dwell_completes"] += 1
            room_id = self._room_id
            self._release()
            return self._snapshot(
                ReSample("dwell in R%s complete" % (room_id,)), now)
        return self._snapshot(
            Hold("dwelling in R%d, %.1f s left"
                 % (self._room_id, self._dwell_end_s - now)), now)

    # -- the draw ---------------------------------------------------------

    def _eligible(self, rooms, now):
        # type: (Sequence[RoomOption], float) -> Tuple[RoomCandidate, ...]
        """Filter the ranking down to what may be drawn, and renormalise it."""
        survivors = [room for room in rooms
                     if room.xy is not None
                     and float(room.prob) >= self.params.min_prob]
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

    def _weighted_index(self, weights):
        # type: (List[float]) -> int
        """Cumulative-sum draw over normalised weights, as flown."""
        threshold = self.rng.random()
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                return index
        return len(weights) - 1

    # -- bookkeeping ------------------------------------------------------

    def _abandon(self, note, now):
        # type: (str, float) -> RoomSearchState
        """Give up on the room in force and go idle, cooling it on the way out."""
        self._note_visited(now)
        self._release()
        return self._snapshot(ReSample(note), now)

    def _note_visited(self, now):
        # type: (float) -> None
        """Start the cooldown on the room whose pursuit has just ended."""
        if self._room_id is not None:
            self._cooling[int(self._room_id)] = float(now)

    def _is_cooling(self, room_id, now):
        # type: (int, float) -> bool
        """Whether this room's pursuit ended too recently to draw it again."""
        ended = self._cooling.get(int(room_id))
        return ended is not None and (now - ended) < self.params.visit_cooldown_s

    def _release(self):
        # type: () -> None
        """Drop the room in force. The next tick draws a new one."""
        self._state = IDLE
        self._room_id = self._label = self._prob = None
        self._goal_xy = self._goal_s = None
        self._plan_s_at_goal = self._dwell_end_s = None

    def _snapshot(self, action, now, changed=False):
        # type: (Action, float, bool) -> RoomSearchState
        """Freeze what is known this tick around ``action``."""
        elapsed = 0.0 if self._goal_s is None else now - self._goal_s
        dwell_left = 0.0
        if self._state == DWELL and self._dwell_end_s is not None:
            dwell_left = max(0.0, self._dwell_end_s - now)
        return RoomSearchState(
            state=self._state, action=action, room_id=self._room_id,
            label=self._label, prob=self._prob, goal_xy=self._goal_xy,
            candidates=self._candidates, elapsed_s=elapsed,
            dwell_left_s=dwell_left, changed=changed)
