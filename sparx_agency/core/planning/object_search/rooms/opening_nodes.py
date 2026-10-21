"""Openings: a doorway or gap at the edge of the mapped floor -- a room's exit -- and the quick look into it.

The room partition is a watershed over OBSERVED free space, so a room the
agent has only glimpsed through its door is not a room yet: its few seen
cells hang off the hallway's region, and whatever object stood in the glimpse
names the hallway. In the Ranchester recording of 2026-10-04 (attempt 7, step
75) the agent stood in a hallway facing three doors; the toilet seen through
one of them made the whole west half of the storey a "bathroom", every door
off the hallway was demoted with it, and nothing was left to value but the
stairs it happened to turn towards. The decision to descend was right; the
reasoning that produced it was not, and in a search for something on THIS
storey it would have been fatal.

An **opening** is the search's name for such a place: a genuine exit of the
mapped floor -- a frontier that is not an object's shadow, not the floor
under the agent's feet, wide enough to pass -- taken together with its
neighbours within :attr:`OpeningSettings.merge_m`, described by the room it
opens from, whether a door frame was seen there, and the objects glimpsed
through it.

**Room-level since 2026-10-21.** An exit is an ATTRIBUTE of the room it
opens from, not a node: the room's line to the oracle carries its exits
(:func:`room_exits` -> ``SearchNode.exits`` / ``doorways`` / ``glimpsed``,
rendered ``exits=2 (1 doorway)  glimpsed through its exits: toilet``), the
model values the room by its type AND what its exits may lead to (a
bathroom's unlooked door raises it for a toilet; a living room's gaps do
not), and the room's visit -- the explore visit's frontier walk -- walks the
exits itself. The global decision is over rooms; the look through a doorway
is a step of the room's local exploration. Standalone opening NODES --
valued by the oracle beside the rooms, offered to RPT* at the threshold and
visited by a **peek** (walk to the threshold, look in -- straight, one turn
left, one turn right -- and move on, priced at a few actions) -- are the
2026-10-04 design, kept as the deprecated ``exits_as_nodes`` mode: a
standalone number per gap let the order chase open passes for their own
sake (every gap of a living room, in a search for a toilet) instead of the
room-level prior. The ``scan`` and ``sweep`` visits have no local walk to
fold an exit into and keep the exits as peek nodes whatever the knob. A
peeked opening is retired for the storey (:class:`OpeningRegistry`), as is
one the planner cannot reach.

A **target landmark** is the second kind of opening (``kind="landmark"``,
since attempt 9 of 2026-10-04), and stays a node in every mode: a confirmed
landmark whose class the target accepts, not yet looked at from close and
not in the takeover's rejection memory. The downstairs warm-up of that
attempt saw the couch three times at 0.41-0.45 -- under the takeover's 0.50,
the far frames without depth -- put a ``sofa`` landmark on the map 3.8 m
away, and the search then walked west for forty actions chasing frontiers
while the one object it was looking for stood confirmed on its own map. Such
a landmark is offered to RPT* at a fixed probability (``landmark_prob``; the
detector's word is the evidence, no oracle call), at a standoff point
``landmark_standoff_m`` from it on the agent's side, and its visit is the
same peek: walk there, face it, one look to each side -- a few close frames
for the detector, and the takeover does the rest or the landmark is marked
inspected for the storey.

Under ``exits_as_nodes`` an exit node is **valued by what is behind it**
(since 2026-10-13): the oracle says how likely the room behind an opening
holds the target; the rays say how much unseen floor the look from the
threshold would show (``Opening.gain_m2``), and :func:`opening_value_factor`
scales the model's number down for a gap onto a strip -- never for a
``doorway``, a room's door whatever the rays count.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.mapping.topology.search_node_oracle import OPENING, SearchNode
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.exploration.view_gain import UnknownView, cone_from_camera
from sparx_agency.core.planning.object_search.perception.doors import DOOR_LABELS
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import (
    goal_rooms, shadow_exemption, split_exits)

#: Opening node ids start here, above the stair band (100 000 - 199 999).
OPENING_NODE_BASE = 200_000
#: Within the opening band, target landmarks take indices from here: ``OPENING_NODE_BASE + 50 000 + landmark id``.
LANDMARK_INDEX_BASE = 50_000
EXIT, LANDMARK = "exit", "landmark"


def opening_node_id(index: int) -> int:
    """The loop's node id for the opening with registry index ``index``."""
    return OPENING_NODE_BASE + int(index)


def is_opening_node(node_id) -> bool:
    """Whether a supervisor room id names an opening rather than a room or a staircase."""
    return node_id is not None and int(node_id) >= OPENING_NODE_BASE


def opening_index_of(node_id: int) -> int:
    """The registry index behind an opening node id."""
    return int(node_id) - OPENING_NODE_BASE


@dataclass(frozen=True)
class OpeningSettings:
    """What counts as an opening, and what a peek is. Distances in metres, counts in actions or cells.

    Attributes:
        enabled: Read the floor's openings at all (the default). Off, the
            openings are the exploration fallback's business alone: no
            room carries an ``exits=`` count and no exit is a node.
        exits_as_nodes: **Deprecated (2026-10-21).** Offer the floor's
            exits to the oracle and to RPT* as nodes of their own beside
            the rooms, visited by a peek -- the 2026-10-04 design. A
            standalone probability per gap let the order chase open passes
            for their own sake: every gap of a living room, in a search for
            a toilet, while the room-level prior said bathroom. Off (the
            default): under the ``explore`` visit every exit is folded into
            the room it opens from (:meth:`RoomSearchLoop._fold_openings`),
            the room's node line carries its exits as attributes
            (``exits=2 (1 doorway)``, the objects glimpsed through them),
            the model raises a room for its exits only where its type fits
            the target, and the room's own visit walks the exits; a room
            the ledger has finished but with GAPs still left to walk is
            offered again for what lies behind them
            (:attr:`RoomSearchLoop._reopened`). The ``scan`` and ``sweep``
            visits have no local walk to fold an exit into and keep the
            exits as peek nodes whatever this says. Target landmarks are
            nodes in every mode (``landmarks_enabled``).
        min_cells: Fewest frontier cells an opening must hold -- a gap the
            body cannot pass is not a way anywhere. 5 cells (0.5 m at the
            map's resolution, since 2026-10-13): the body is 0.36 m across
            and a bathroom door 0.6-0.7 m, and at the former 8 (0.8 m) the
            Darden bathroom's door -- 7 or 8 frontier cells as the map
            re-snapped -- flickered in and out of the openings from one
            loop point to the next and was never offered when it mattered
            (``ec8d17aeaaaa`` and its re-runs; the toilet was behind it).
        merge_m: Exit frontiers within this distance of an opening's seed
            are the same opening.
        match_m: An opening within this distance of one already known keeps
            that one's id (the frontier re-snaps as the map grows).
        done_m: An opening within this distance of a completed peek is
            retired for the storey.
        door_m: A confirmed door landmark within this distance makes the
            opening a ``doorway``; otherwise it is a ``gap``.
        glimpse_m: Confirmed landmarks up to this far beyond the threshold
            along its heading were glimpsed through it.
        glimpse_cone_deg: ... and within this many degrees of the heading.
            Ranchester attempt 8: a half-plane test credited a bed standing
            beside the threshold IN the hallway to the opening, which then
            read as a bedroom's door for sixty actions; it was the passage
            to the stairs.
        glimpse_min_m: ... and at least this far beyond the threshold, so an
            object at the threshold itself is not "through" it.
        approach_steps: Actions the walk to the threshold may take at least;
            a farther threshold gets ``approach_factor`` actions per forward
            step of distance plus six for the turns (attempt 8's first peek
            spent 20 actions on a 4 m walk and gave up 1.8 m short).
        approach_factor: See ``approach_steps``.
        look_turns: Turns to each side of the heading the look takes (1 =
            straight, left, right).
        step_through: After the look from the standoff, walk ON TO the
            threshold and look again, wider (since 2026-10-14). A frontier
            cell borders unknown, unknown inflates, so the planner's route
            ends up to 0.6 m behind the threshold and the first look sees
            the room beyond through a wedge the jambs leave -- Collierville/
            000023 looked from 0.58 m behind its doorway at three headings
            and saw nothing of the room. Once the first look has mapped the
            floor just beyond, the threshold cell is passable: the peek
            walks to it (``through_steps`` actions at most) and takes a
            second look of ``through_look_turns`` to each side of the way
            it came through -- left and right, standing in the door. Only
            when the first look stood more than the arrival tolerance short
            of the threshold; off, the peek is the one look as before.
        through_look_turns: Turns to each side of the second look (2 = five
            headings, +-60 degrees, +-100 with the camera's half field).
        through_steps: Actions the walk from the standoff to the threshold
            may take before the peek is retired with the first look alone.
        visit_steps: Hard bound on the look, from arrival.
        service_steps: What RPT* charges a peek for.
        revalue_actions: While nothing is in force, a NEW opening buys the
            oracle a call at most this often (the room oracle is asked at
            once for a new room or staircase; openings appear and re-snap
            with every step of the fallback, and attempt 8 asked thirty
            times in sixty actions).
        unvalued_prob: What an exit the oracle has NOT valued is handed to
            RPT* with, until it is asked (since 2026-10-14). The oracle's
            numbers are replaced wholesale at every call, so an exit that
            was folded into its room's walk at the call (the room offered,
            :meth:`RoomSearchLoop._fold_openings`) and unfolded by the
            reply (the room dropped under ``min_prob``) comes out of the
            call with no number at all -- it was not shown -- and read 0.0:
            no node, under the floor, until the next call, which comes at
            the next SELECT, after whatever the solver chose instead is
            done. Corozal/000142 (2026-10-14, action 446): the toilet's own
            doorway, 12 m2 unseen behind it and 6 m away, was unfolded by
            the reply that rated the room it opened from "office, entered"
            at 1 %, read 0.0, and the search walked 12 m the other way to a
            gap the model gave 10 %; no SELECT came in the 54 actions left.
            An exit the model has not been shown is worth what the model
            gives an unseen doorway it has (10 % that action). The oracle's
            own last number for the exit is used first when it has one from
            an earlier call; the provisional value is only for an exit it
            has never seen. The next call values it (a node without a
            number still buys the call). 0 disables: unvalued exits read
            0.0 as before. (0.05 -- the floor -- was tried on 2026-10-14 and
            cost two episodes that 0.10 had won: the user's verdict was the
            run with 0.10, ``benchmark_toilet_10episodes_20261014T143259Z``.)
        landmarks_enabled: Offer confirmed landmarks of the target's class
            as nodes (the default).
        landmark_prob: The fixed probability such a node is handed to RPT*
            with -- no oracle call: the detector saw the target's class
            there more than once.
        landmark_standoff_m: Where the look stands: this far from the
            landmark, on the agent's side, on a reachable cell.
        landmark_sighting_confidence: A landmark of the target's class seen
            ONCE is a node too when the detector read it at this confidence
            or better in that one frame (since 2026-10-14; the landmark map
            needs two observations to confirm an object, which a far
            glimpse through a doorway does not get: Wiconisco/000014 saw its
            toilet at 0.54 and 3.2 m, the verification lost it after one
            step and released it, and with no node on the map the search
            went elsewhere for 65 actions). The closing's own start
            threshold by default -- a sighting good enough to start a
            takeover is good enough to walk back to. 1.0 disables: confirmed
            landmarks only.
        gain_full_m2: The unknown floor the camera would reach through an
            opening (``Opening.gain_m2``: the cone from the threshold along
            the heading, rays through the unknown, blocked by walls and by
            the settled unknown) at which the opening counts in full. Below
            it the oracle's probability is scaled down in proportion
            (:func:`opening_value_factor`), to ``gain_min_factor`` at the
            least: a one-metre gap beside a room already walked shows a
            strip, a doorway onto an unseen wing shows a wing, and the
            model values both "an opening with nothing glimpsed" the same
            (Corozal/000076, 2026-10-13, action 126: a 10-cell gap 4.6 m
            away, 1 m2 behind it, was walked to before a 148-cell opening
            onto the unseen half of the storey). A ``doorway`` -- a confirmed
            door at the threshold -- is a room's door whatever the rays
            count and is never scaled. 0 disables the scaling.
        gain_min_factor: The least an exit's probability is scaled to.
        stair_exclusion_m: An exit this close to a cell the spawn-floor
            guard has masked (a seen staircase's footprint, an observed
            drop) is not an opening of this storey: the unknown beyond it
            is the flight or the level below (Corozal/000076, actions
            168-255: an 87-action loop round the capsule of a staircase to
            peek at the gap at its foot). Checked against the portal points
            as well, as before, and again every action of a peek's walk.
            Directional since 2026-10-14 (evening): only masked cells and
            portal points on the exit's UNKNOWN side count
            (:func:`near_stair_mask` with the heading, :func:`beyond`) --
            Darden/000034's bathroom door, 1.3-1.5 m beside the flight with
            its room opening the other way, was struck off the openings for
            the whole episode by the omnidirectional test.
        vanish_actions: A peek's opening absent from the live inventory for
            this many actions running, while its threshold is still more
            than ``merge_m`` away, was resolved by the camera on the way:
            the peek ends there (``peek_resolved``), nothing is walked to.
    """

    enabled: bool = True
    exits_as_nodes: bool = False
    min_cells: int = 5
    merge_m: float = 1.5
    match_m: float = 1.0
    done_m: float = 1.0
    door_m: float = 1.0
    glimpse_m: float = 3.0
    glimpse_cone_deg: float = 35.0
    glimpse_min_m: float = 0.3
    approach_steps: int = 20
    approach_factor: float = 1.6
    look_turns: int = 1
    step_through: bool = True
    through_look_turns: int = 2
    through_steps: int = 8
    visit_steps: int = 30
    service_steps: int = 4
    revalue_actions: int = 6
    unvalued_prob: float = 0.10
    landmarks_enabled: bool = True
    landmark_prob: float = 0.85
    landmark_standoff_m: float = 1.5
    landmark_sighting_confidence: float = 0.50
    gain_full_m2: float = 4.0
    gain_min_factor: float = 0.25
    stair_exclusion_m: float = 1.5
    vanish_actions: int = 3

    def __post_init__(self):
        for name in ("enabled", "exits_as_nodes", "landmarks_enabled", "step_through"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("openings.%s must be a bool" % name)
        for name in ("through_look_turns", "through_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("openings.%s must be a non-negative integer" % name)
        for name in ("gain_full_m2", "stair_exclusion_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be non-negative and finite" % name)
        if (isinstance(self.gain_min_factor, bool) or not math.isfinite(self.gain_min_factor)
                or not 0.0 < self.gain_min_factor <= 1.0):
            raise ValueError("gain_min_factor must lie in (0, 1]")
        for name in ("min_cells", "approach_steps", "visit_steps", "service_steps", "revalue_actions", "vanish_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if type(self.look_turns) is not int or self.look_turns < 0:
            raise ValueError("look_turns must be a non-negative integer")
        for name in ("merge_m", "match_m", "done_m", "door_m", "glimpse_m", "glimpse_cone_deg", "glimpse_min_m",
                     "approach_factor", "landmark_standoff_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if isinstance(self.landmark_prob, bool) or not math.isfinite(self.landmark_prob) or not 0 < self.landmark_prob < 1:
            raise ValueError("landmark_prob must lie in (0, 1)")
        if isinstance(self.unvalued_prob, bool) or not math.isfinite(self.unvalued_prob) or not 0 <= self.unvalued_prob < 1:
            raise ValueError("unvalued_prob must lie in [0, 1)")
        if (isinstance(self.landmark_sighting_confidence, bool) or not math.isfinite(self.landmark_sighting_confidence)
                or not 0 < self.landmark_sighting_confidence <= 1):
            raise ValueError("landmark_sighting_confidence must lie in (0, 1]")
        if self.glimpse_min_m >= self.glimpse_m:
            raise ValueError("glimpse_min_m must lie inside glimpse_m")
        if self.glimpse_cone_deg > 90.0:
            raise ValueError("glimpse_cone_deg must be at most 90 (a half-plane is not a glimpse)")
        if 2 * self.look_turns + 1 >= self.visit_steps:
            raise ValueError("visit_steps must leave room for the look's headings and the turns to face them")

    def approach_bound(self, geodesic_m: float, forward_step_m: float) -> int:
        """Actions the walk to a threshold ``geodesic_m`` away may take."""
        steps = math.ceil(max(0.0, float(geodesic_m)) / max(1e-6, float(forward_step_m)))
        return max(int(self.approach_steps), int(math.ceil(self.approach_factor * steps)) + 6)


@dataclass(frozen=True)
class Opening:
    """One exit of the mapped floor -- or one target landmark -- as the loop values and visits it.

    Attributes:
        index: The registry's sticky id for this opening on its storey; for
            a landmark, :data:`LANDMARK_INDEX_BASE` plus the landmark's id.
        xy: Where the peek stands: the exit's nearest passable cell, or the
            landmark's standoff point.
        heading: Which way the unknown -- or the landmark -- lies from there, radians.
        size_cells: Frontier cells across the opening and its merged neighbours; 0 for a landmark.
        geodesic_m: Walking distance from the agent to ``xy``.
        room_pid: The mapped room the opening leads out of (or the landmark stands in), or None.
        door_id: The confirmed door landmark at the threshold, or None.
        glimpsed: Object classes already seen beyond the threshold; for a landmark, its class.
        kind: :data:`EXIT` or :data:`LANDMARK`.
        landmark_id: The landmark's id on this storey's map, for a landmark.
        landmark_xy: Where the landmark stands, for a landmark.
        gain_m2: The unknown floor the camera's cone reaches from ``xy``
            along ``heading`` (what a look from the threshold would show);
            0 for a landmark. Read by :func:`opening_value_factor`.
    """

    index: int
    xy: Tuple[float, float]
    heading: float
    size_cells: int
    geodesic_m: float
    room_pid: Optional[int]
    door_id: Optional[int]
    glimpsed: Tuple[str, ...]
    kind: str = EXIT
    landmark_id: Optional[int] = None
    landmark_xy: Optional[Tuple[float, float]] = None
    gain_m2: float = 0.0

    @property
    def node_id(self) -> int:
        return opening_node_id(self.index)

    @property
    def label(self) -> str:
        if self.kind == LANDMARK:
            return "%s landmark" % (self.glimpsed[0] if self.glimpsed else "target")
        return "doorway" if self.door_id is not None else "gap"


@dataclass(frozen=True)
class OpeningOption:
    """One opening as the loop offers it to the supervisor and the solver (mirrors ``StairOption``)."""

    node_id: int
    opening: Opening
    node: SearchNode

    def option(self, prob: float) -> RoomOption:
        return RoomOption(room_id=self.node_id, prob=float(prob), xy=self.opening.xy, label=self.node.label)


@dataclass(frozen=True)
class RoomExits:
    """One room's open exits, as its node line shows them to the oracle (room-level GAPs, since 2026-10-21).

    Attributes:
        count: Exits (doorways and gaps to space not seen yet) credited to the room.
        doorways: How many of them stand at a confirmed door frame.
        glimpsed: Object classes seen through them, beyond their thresholds, sorted.
    """

    count: int = 0
    doorways: int = 0
    glimpsed: Tuple[str, ...] = ()


def room_exits(openings: Sequence) -> Dict[int, RoomExits]:
    """``{pid: RoomExits}`` over the EXITS among ``openings`` (:class:`Opening` or :class:`OpeningOption`), by the room each opens from.

    Target landmarks are not exits and an exit off no room (``room_pid``
    None) is nobody's attribute: both are left out.
    """
    counts: Dict[int, List] = {}
    for item in openings:
        opening = getattr(item, "opening", item)
        pid = opening.room_pid
        if opening.kind == LANDMARK or pid is None:
            continue
        entry = counts.setdefault(int(pid), [0, 0, set()])
        entry[0] += 1
        entry[1] += 1 if opening.door_id is not None else 0
        entry[2].update(str(c) for c in opening.glimpsed if str(c).strip())
    return {pid: RoomExits(count=n, doorways=doors, glimpsed=tuple(sorted(seen)))
            for pid, (n, doors, seen) in counts.items()}


class OpeningRegistry:
    """Sticky opening ids per storey, and the openings already looked into.

    Lives on the policy for the episode: ids are unique across the building,
    like room pids, so ``O3`` names one opening in the recording.
    """

    def __init__(self, match_m: float = 1.0, done_m: float = 1.0):
        self.match_m, self.done_m = float(match_m), float(done_m)
        self._known: Dict[int, Dict[int, Tuple[float, float]]] = {}
        self._peeked: Dict[int, List[Dict]] = {}
        self._next = 0

    def identify(self, floor: int, xy: Tuple[float, float]) -> int:
        """The id of the known opening within ``match_m`` of ``xy`` (moved to it), else a new id."""
        known = self._known.setdefault(int(floor), {})
        best = None
        for index, where in known.items():
            distance = math.dist(where, xy)
            if distance <= self.match_m and (best is None or distance < best[0]):
                best = (distance, index)
        if best is not None:
            known[best[1]] = (float(xy[0]), float(xy[1]))
            return best[1]
        index = self._next
        self._next += 1
        known[index] = (float(xy[0]), float(xy[1]))
        return index

    def peeked(self, floor: int, xy: Tuple[float, float]) -> bool:
        """Whether a completed peek already covers ``xy``: within ``done_m`` of its threshold, or inside what it looked at.

        A look from a threshold reveals one to three metres of floor beyond
        it, and the frontier then sits that far past the old threshold --
        through the same door. Before 2026-10-07 that re-snapped seed was a
        new opening (``match_m`` away from the old one) and bought another
        walk and look at the same doorway; the Allensville toilet run peeked
        twenty openings that way. A seed inside the cone a completed look
        swept (``heading`` +/- ``cone_rad``, out to ``looked_m``) is covered
        by that look.
        """
        for record in self._peeked.get(int(floor), ()):
            dx, dy = float(xy[0]) - record["xy"][0], float(xy[1]) - record["xy"][1]
            distance = math.hypot(dx, dy)
            if distance <= self.done_m:
                return True
            heading = record.get("heading")
            if heading is None or distance > float(record.get("looked_m", 0.0)):
                continue
            if abs(normalize_angle(math.atan2(dy, dx) - heading)) <= float(record.get("cone_rad", 0.0)):
                return True
        return False

    def mark_peeked(self, floor: int, xy: Tuple[float, float], step: int, why: str = "looked",
                    heading: Optional[float] = None, cone_rad: float = 0.0, looked_m: float = 0.0) -> None:
        """Remember a finished peek at ``xy``; with ``heading``, the cone it looked into counts as peeked too."""
        self._peeked.setdefault(int(floor), []).append(
            {"xy": (float(xy[0]), float(xy[1])), "step": int(step), "why": str(why),
             "heading": None if heading is None else float(heading), "cone_rad": float(cone_rad),
             "looked_m": float(looked_m)})

    def diagnostics(self) -> Dict:
        return {"known": {str(f): {str(i): [round(v, 2) for v in xy] for i, xy in known.items()}
                          for f, known in self._known.items()},
                "peeked": {str(f): [dict(r, xy=[round(v, 2) for v in r["xy"]],
                                         heading=None if r.get("heading") is None else round(r["heading"], 3))
                                    for r in records]
                           for f, records in self._peeked.items()}}


def unknown_heading(world, cell: Tuple[int, int], radius_cells: int = 6) -> Optional[float]:
    """Which way the unknown lies from a frontier cell: from the known floor around it toward the unknown.

    The direction from the centroid of the known FREE cells in the window to
    the centroid of its UNKNOWN cells. The mean direction to the unknown
    alone is not it: a frontier cell has unknown on several sides -- the
    strip behind a bed beside it, the gap it stands in -- and Ranchester
    attempt 8 read the stair passage's heading as pointing back into the
    hallway, three times in sixty actions. The known floor is the one side
    the agent came from, so away from it is through the opening. None when
    the window holds no unknown or no free cell.
    """
    gx, gy = int(cell[0]), int(cell[1])
    h, w = world.grid.shape
    x0, x1 = max(0, gx - radius_cells), min(w, gx + radius_cells + 1)
    y0, y1 = max(0, gy - radius_cells), min(h, gy + radius_cells + 1)
    window = world.grid[y0:y1, x0:x1]
    unknown_ys, unknown_xs = np.nonzero(window == world.values.unknown)
    free_ys, free_xs = np.nonzero(window == world.values.free)
    if not len(unknown_xs) or not len(free_xs):
        return None
    dx = float(np.mean(unknown_xs) - np.mean(free_xs))
    dy = float(np.mean(unknown_ys) - np.mean(free_ys))
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def glimpsed_through(threshold: Tuple[float, float], heading: float, landmarks, settings: OpeningSettings) -> Tuple[str, ...]:
    """The classes of confirmed landmarks seen THROUGH an opening: in its cone, beyond its threshold, within range."""
    ahead = (math.cos(heading), math.sin(heading))
    cone = math.radians(settings.glimpse_cone_deg)
    names = set()
    for landmark in landmarks:
        if landmark.class_name in DOOR_LABELS:
            continue
        dx, dy = landmark.xy[0] - threshold[0], landmark.xy[1] - threshold[1]
        distance = math.hypot(dx, dy)
        if not settings.glimpse_min_m <= distance <= settings.glimpse_m:
            continue
        along = dx * ahead[0] + dy * ahead[1]
        if along <= 0.0 or math.acos(max(-1.0, min(1.0, along / distance))) > cone:
            continue
        names.add(landmark.class_name)
    return tuple(sorted(names))


def stair_points(policy) -> List[Tuple[float, float]]:
    """Where the storey's seen staircases meet its floor: every portal's entry and path on this floor."""
    building = getattr(policy, "building", None)
    if building is None:
        return []
    points = []
    for portal in getattr(building, "portals", ()):
        if portal.get("floor_id") != building.floor_id:
            continue
        for point in [portal.get("entry")] + list(portal.get("path") or ()):
            if point is not None and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    return points


def opening_value_factor(opening: Opening, settings: OpeningSettings) -> float:
    """How much of the oracle's probability an opening keeps: 1 for a doorway or a landmark, else by the unknown behind it.

    ``clamp(gain_m2 / gain_full_m2, gain_min_factor, 1)``: an exit the
    camera would see ``gain_full_m2`` or more of unknown floor through is
    worth the model's full number; a gap onto a strip is worth a fraction,
    never under ``gain_min_factor``. Distance is RPT*'s business, not this
    factor's.
    """
    if opening.kind == LANDMARK or opening.door_id is not None or settings.gain_full_m2 <= 0.0:
        return 1.0
    return float(min(1.0, max(settings.gain_min_factor, float(opening.gain_m2) / settings.gain_full_m2)))


def beyond(xy, heading: Optional[float], point) -> bool:
    """Whether ``point`` lies on the unknown side of a frontier at ``xy`` facing ``heading`` (any side when heading is None)."""
    if heading is None:
        return True
    return (float(point[0]) - float(xy[0])) * math.cos(heading) + (float(point[1]) - float(xy[1])) * math.sin(heading) > 0.0


def near_stair_mask(policy, world, xy, radius_m: float, heading: Optional[float] = None) -> bool:
    """Whether a cell the spawn-floor guard has masked (stairs, drops) lies within ``radius_m`` of ``xy`` -- beyond it, when ``heading`` is given.

    The exclusion says "the unknown beyond this frontier is the flight or
    the level below", so with the frontier's unknown heading known only the
    masked cells on THAT side count (since 2026-10-14 evening): a doorway
    beside a staircase whose room opens the other way is a room's door.
    Darden/000034's bathroom door stands 1.3-1.5 m from the flight, and
    the omnidirectional test struck both frontiers onto it off the
    openings for the whole episode. Without a heading every masked cell in
    the window counts, as before.
    """
    guard = getattr(policy, "floor_guard", None)
    mask = getattr(guard, "mask", None)
    if mask is None or radius_m <= 0.0:
        return False
    gx, gy = world.world_to_grid(float(xy[0]), float(xy[1]))
    cells = int(math.ceil(radius_m / world.resolution))
    h, w = mask.shape
    y0, x0 = max(0, gy - cells), max(0, gx - cells)
    window = mask[y0:min(h, gy + cells + 1), x0:min(w, gx + cells + 1)]
    if heading is None:
        return bool(window.any())
    ys, xs = np.nonzero(window)
    for cy, cx in zip(ys, xs):
        if beyond(xy, heading, world.grid_to_world(int(cx + x0), int(cy + y0))):
            return True
    return False


def unknown_view_for(policy, world):
    """The ray caster over ``world`` for the openings' gains: the camera's cone, the settled unknown blocking."""
    camera = policy.episode.camera
    k = camera.intrinsics
    blind = getattr(getattr(policy, "glance_settings", None), "blind_m", 0.0)
    sight = getattr(policy, "sight", None)
    return UnknownView(world, cone_from_camera(k.width, k.fx, float(camera.max_depth_m), float(blind or 0.0)),
                       resolved=sight.resolved(world) if sight is not None else None)


def detect_openings(policy, obs, world, settings: OpeningSettings, registry: OpeningRegistry) -> List[Opening]:
    """The storey's openings this action, best first, from the frontier inventory.

    Exits by the exploration fallback's own test (:func:`split_exits`
    without its type rule -- a hallway named after the toilet glimpsed from
    it must still offer its doors), merged within ``merge_m``, at least
    ``min_cells`` wide, not yet peeked, and not at the foot or head of a
    seen staircase -- the stairs are a node of their own, and the unknown
    beyond a flight is the other storey, not a room of this one.
    """
    p = policy
    inventory = getattr(p.graph, "frontier_inventory", None)
    if inventory is None or not inventory.goals:
        return []
    goals = p.sweep.admissible(obs, world, list(inventory.goals))
    landmarks = list(p.landmarks.confirmed()) if getattr(p, "landmarks", None) is not None else []
    view = unknown_view_for(p, world) if settings.gain_full_m2 > 0.0 else None
    # A frontier with a room's floor right behind it is nobody's shadow (``shadow_exemption``, since
    # 2026-10-14): the strip behind a bed has the wall behind it, and the 2.9 m frontier onto
    # Collierville/000023's toilet room was dropped as the refrigerator's.
    exits, _ = split_exits(goals, landmarks, world.resolution, p.fallback.settings, type_rule=False,
                           exempt=shadow_exemption(view, p.fallback.settings))
    stairs = stair_points(p)
    # The stairs' test is directional (since 2026-10-14 evening): a frontier beside a flight
    # whose unknown opens the other way is a room's door, not the flight's foot.
    headings = {g: unknown_heading(world, g.cell) for g in exits}
    exits = [g for g in exits
             if not any(math.dist(g.xy, point) <= settings.merge_m and beyond(g.xy, headings[g], point) for point in stairs)
             and not near_stair_mask(p, world, g.xy, settings.stair_exclusion_m, heading=headings[g])]
    if not exits:
        return []
    room_of = goal_rooms(inventory)
    clusters: List[List] = []
    for goal in exits:                                  # utility order: the seed is the best exit of its cluster
        for cluster in clusters:
            if math.dist(cluster[0].xy, goal.xy) <= settings.merge_m:
                cluster.append(goal)
                break
        else:
            clusters.append([goal])
    floor = p.mapping.floor_id
    doors = list(p.doors.confirmed()) if getattr(p, "doors", None) is not None else []
    rooms = p.graph.registry.rooms
    openings = []
    for cluster in clusters:
        seed = cluster[0]
        size = int(sum(int(g.size_cells) for g in cluster))
        if size < settings.min_cells or registry.peeked(floor, seed.xy):
            continue
        pid = room_of.get(seed)
        heading = headings.get(seed)
        if heading is None:
            heading = unknown_heading(world, seed.cell)
        if heading is None:
            origin = rooms[pid].centroid if pid in rooms else (obs.pose.x, obs.pose.y)
            heading = math.atan2(seed.xy[1] - origin[1], seed.xy[0] - origin[0])
        door = None
        for landmark in doors:
            distance = math.dist(landmark.xy, seed.xy)
            if distance <= settings.door_m and (door is None or distance < door[0]):
                door = (distance, landmark.id)
        glimpsed = glimpsed_through(seed.xy, heading, landmarks, settings)
        gain = 0.0
        if view is not None:
            gain = float(view.unknown_ahead(seed.xy, heading).sum()) * view.cell_area_m2
        openings.append(Opening(index=registry.identify(floor, seed.xy), xy=(float(seed.xy[0]), float(seed.xy[1])),
                                heading=float(normalize_angle(heading)), size_cells=size,
                                geodesic_m=float(seed.geodesic_m), room_pid=pid,
                                door_id=None if door is None else int(door[1]), glimpsed=tuple(glimpsed),
                                gain_m2=gain))
    return openings


def landmark_openings(policy, obs, world, cost, settings: OpeningSettings, inspected) -> List[Opening]:
    """Confirmed landmarks of the target's class as nodes: a standoff point on the agent's side, facing them.

    The landmarks are :func:`target_landmarks`: confirmed by the map, or
    seen once at ``landmark_sighting_confidence`` (since 2026-10-14).
    Skipped: landmarks in ``inspected`` (looked at from close on this
    storey already), landmarks a release of the takeover DISPROVED from
    close (``TargetClosing.landmark_refused``; a far verification that
    failed does not hide the landmark), and landmarks no reachable standoff
    cell can be found for (a ring of ``landmark_standoff_m``, then one and
    a half times it; the cell nearest the agent along the floor).
    """
    p = policy
    if not settings.landmarks_enabled or getattr(p, "landmarks", None) is None:
        return []
    target = p.target
    closing = getattr(p, "closing", None)
    floor = p.mapping.floor_id
    inventory = getattr(p.graph, "frontier_inventory", None)
    distance = (inventory.distance_m if inventory is not None and getattr(inventory, "distance_m", None) is not None
                and inventory.distance_m.shape == world.grid.shape else None)
    here = (obs.pose.x, obs.pose.y)
    out = []
    for landmark in target_landmarks(p, settings):
        if landmark.class_name in DOOR_LABELS:
            continue
        if landmark.id in inspected:
            continue
        if closing is not None and closing.landmark_refused(landmark.xy, floor):
            continue
        standoff = landmark_standoff(world, cost, distance, landmark.xy, here, settings.landmark_standoff_m)
        if standoff is None:
            continue
        xy, geodesic = standoff
        heading = math.atan2(landmark.xy[1] - xy[1], landmark.xy[0] - xy[0])
        out.append(Opening(index=LANDMARK_INDEX_BASE + int(landmark.id), xy=xy, heading=float(normalize_angle(heading)),
                           size_cells=0, geodesic_m=float(geodesic), room_pid=p.graph.object_room(world, landmark),
                           door_id=None, glimpsed=(str(landmark.class_name),), kind=LANDMARK,
                           landmark_id=int(landmark.id), landmark_xy=(float(landmark.xy[0]), float(landmark.xy[1]))))
    return out


def target_landmarks(policy, settings: OpeningSettings):
    """The landmarks of the target's class worth a node: confirmed, or seen once at ``landmark_sighting_confidence``.

    The landmark map confirms an object on its second observation; the
    best detector confidence a landmark was ever observed at is kept by
    the perception cycle (``policy._landmark_best_conf``), and a single
    sighting at the threshold counts (the closing's start threshold by
    default). Id order, as :meth:`ObjectLandmarkMap.confirmed`.
    """
    p = policy
    target = p.target
    best = getattr(p, "_landmark_best_conf", None) or {}
    threshold = float(settings.landmark_sighting_confidence)
    out = []
    for landmark in p.landmarks.all_landmarks():
        if not target.accepts(landmark.class_name):
            continue
        if p.landmarks.is_confirmed(landmark) or (threshold < 1.0 and best.get(landmark.id, 0.0) >= threshold):
            out.append(landmark)
    return out


def landmark_standoff(world, cost, distance, landmark_xy, here, standoff_m, angles=16):
    """``((x, y), geodesic_m)`` of the reachable cell on a ring around the landmark nearest the agent, or None."""
    for radius in (standoff_m, 1.5 * standoff_m):
        best = None
        for k in range(angles):
            angle = 2 * math.pi * k / angles
            x, y = landmark_xy[0] + radius * math.cos(angle), landmark_xy[1] + radius * math.sin(angle)
            gx, gy = world.world_to_grid(x, y)
            if not world.in_bounds(gx, gy) or not np.isfinite(cost[gy, gx]):
                continue
            if distance is not None:
                geodesic = float(distance[gy, gx])
                if not math.isfinite(geodesic):
                    continue
            else:
                geodesic = math.dist((x, y), here)
            if best is None or geodesic < best[1]:
                best = ((float(x), float(y)), geodesic)
        if best is not None:
            return best
    return None


def opening_options(policy, openings: Sequence[Opening]) -> List[OpeningOption]:
    """The openings as nodes for the oracle, with the room each opens from named as the prompt names rooms.

    A target landmark's node carries its class and room too, but the loop
    does not show it to the oracle: its probability is fixed.
    """
    graph = policy.graph
    out = []
    for opening in openings:
        via = None
        if opening.room_pid is not None:
            info = graph.label_info(opening.room_pid) or {}
            label = info.get("label") or "unknown"
            if info.get("strength") == "weak" and label != "unknown":
                label += "?"
            via = "room %d (type=%s)" % (opening.room_pid, label)
        node = SearchNode(id=opening.node_id, kind=OPENING, label=opening.label, objects=tuple(opening.glimpsed), via=via)
        out.append(OpeningOption(node_id=opening.node_id, opening=opening, node=node))
    return out

