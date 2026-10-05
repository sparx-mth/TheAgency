"""Openings as search nodes: a doorway or gap at the edge of the mapped floor, and the quick look into it.

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
neighbours within :attr:`OpeningSettings.merge_m`. It is offered to the
oracle and to RPT* as a node beside the rooms and the stairs (ids from
:data:`OPENING_NODE_BASE`), described by the room it opens from, whether a
door frame was seen there, and the objects glimpsed through it. Its visit is
a **peek**: walk to the threshold, look in -- straight, one turn left, one
turn right -- and move on; a peek is priced to RPT* at a few actions, not a
room's scan. What the peek sees becomes floor in the partition: the room
behind separates, is named by its objects, and is excluded by type or
scanned like any other. A peeked opening is retired for the storey
(:class:`OpeningRegistry`), as is one the planner cannot reach.

A **target landmark** is the second kind of opening (``kind="landmark"``,
since attempt 9 of 2026-10-04): a confirmed landmark whose class the target
accepts, not yet looked at from close and not in the takeover's rejection
memory. The downstairs warm-up of that attempt saw the couch three times at
0.41-0.45 -- under the takeover's 0.50, the far frames without depth -- put
a ``sofa`` landmark on the map 3.8 m away, and the search then walked west
for forty actions chasing frontiers while the one object it was looking for
stood confirmed on its own map. Such a landmark is offered to RPT* at a
fixed probability (``landmark_prob``; the detector's word is the evidence,
no oracle call), at a standoff point ``landmark_standoff_m`` from it on the
agent's side, and its visit is the same peek: walk there, face it, one look
to each side -- a few close frames for the detector, and the takeover does
the rest or the landmark is marked inspected for the storey.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.mapping.topology.search_node_oracle import OPENING, SearchNode
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import goal_rooms, split_exits

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
        enabled: Offer the floor's openings as nodes (the default). Off, the
            openings are the exploration fallback's business alone.
        min_cells: Fewest frontier cells an opening must hold -- a gap the
            body cannot pass is not a way anywhere (8 cells = 0.8 m at the
            map's resolution).
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
        visit_steps: Hard bound on the look, from arrival.
        service_steps: What RPT* charges a peek for.
        revalue_actions: While nothing is in force, a NEW opening buys the
            oracle a call at most this often (the room oracle is asked at
            once for a new room or staircase; openings appear and re-snap
            with every step of the fallback, and attempt 8 asked thirty
            times in sixty actions).
        landmarks_enabled: Offer confirmed landmarks of the target's class
            as nodes (the default).
        landmark_prob: The fixed probability such a node is handed to RPT*
            with -- no oracle call: the detector saw the target's class
            there more than once.
        landmark_standoff_m: Where the look stands: this far from the
            landmark, on the agent's side, on a reachable cell.
    """

    enabled: bool = True
    min_cells: int = 8
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
    visit_steps: int = 30
    service_steps: int = 4
    revalue_actions: int = 6
    landmarks_enabled: bool = True
    landmark_prob: float = 0.85
    landmark_standoff_m: float = 1.5

    def __post_init__(self):
        for name in ("enabled", "landmarks_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("openings.%s must be a bool" % name)
        for name in ("min_cells", "approach_steps", "visit_steps", "service_steps", "revalue_actions"):
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
        return any(math.dist(record["xy"], xy) <= self.done_m for record in self._peeked.get(int(floor), ()))

    def mark_peeked(self, floor: int, xy: Tuple[float, float], step: int, why: str = "looked") -> None:
        self._peeked.setdefault(int(floor), []).append(
            {"xy": (float(xy[0]), float(xy[1])), "step": int(step), "why": str(why)})

    def diagnostics(self) -> Dict:
        return {"known": {str(f): {str(i): [round(v, 2) for v in xy] for i, xy in known.items()}
                          for f, known in self._known.items()},
                "peeked": {str(f): [dict(r, xy=[round(v, 2) for v in r["xy"]]) for r in records]
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
    exits, _ = split_exits(goals, landmarks, world.resolution, p.fallback.settings, type_rule=False)
    stairs = stair_points(p)
    exits = [g for g in exits if not any(math.dist(g.xy, point) <= settings.merge_m for point in stairs)]
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
        openings.append(Opening(index=registry.identify(floor, seed.xy), xy=(float(seed.xy[0]), float(seed.xy[1])),
                                heading=float(normalize_angle(heading)), size_cells=size,
                                geodesic_m=float(seed.geodesic_m), room_pid=pid,
                                door_id=None if door is None else int(door[1]), glimpsed=tuple(glimpsed)))
    return openings


def landmark_openings(policy, obs, world, cost, settings: OpeningSettings, inspected) -> List[Opening]:
    """Confirmed landmarks of the target's class as nodes: a standoff point on the agent's side, facing them.

    Skipped: landmarks in ``inspected`` (looked at from close on this
    storey already), landmarks the takeover's rejection memory holds a
    released candidate at, and landmarks no reachable standoff cell can be
    found for (a ring of ``landmark_standoff_m``, then one and a half times
    it; the cell nearest the agent along the floor).
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
    for landmark in p.landmarks.confirmed():
        if landmark.class_name in DOOR_LABELS or not target.accepts(landmark.class_name):
            continue
        if landmark.id in inspected:
            continue
        if closing is not None and closing.rejected_near(landmark.xy, floor):
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

