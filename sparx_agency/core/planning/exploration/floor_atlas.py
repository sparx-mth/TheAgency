"""Observed building topology with persistent floors and measured stair edges.

Floor IDs are discovery IDs, not surveyed storey numbers. Intermediate stair
heights and stationary turns cannot create floors. This module consumes only
poses, never scene bounds, semantic ground truth or a simulator pathfinder.
Python 3.8, standard library only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math
from typing import Dict, List, Optional, Tuple


#: Where the building coordinator learns where the stairs are.
STAIR_SOURCE_GROUND_TRUTH = "ground_truth"
STAIR_SOURCE_OBSERVED = "observed"
STAIR_SOURCES = (STAIR_SOURCE_GROUND_TRUTH, STAIR_SOURCE_OBSERVED)


@dataclass(frozen=True)
class MultiFloorParams:
    """Floor hysteresis for the atlas plus the building coordinator's knobs.

    Attributes:
        stair_source: ``ground_truth`` -- stair connectors and floor levels
            come from the simulator's navmesh through the episode metadata,
            and a floor transition is started only at one of them;
            ``observed`` -- the former RGB-D support-surface discovery, which
            mistook a raised bathroom floor for a staircase. The atlas itself
            reads poses only in either mode; this is the coordinator's choice.
        near_connector_m: How close (XY, metres) to a ground-truth connector's
            polyline an unplanned height departure must be to count as being
            on those stairs rather than on a step or a threshold.
        floor_search_actions: OBSERVED mode only -- actions to search a floor
            before its portals are considered. In ground-truth mode nothing
            decides a floor change by a clock: a staircase is a node of the
            room-search loop's RPT* instance, valued by the LLM and charged
            for the climb, and taken when the order says so.
        floor_change_cost_m: Ground truth -- the fixed cost of a storey
            change in metres of walking, added to the flight's own length on
            every arc into or out of a staircase node: the climb's turns, the
            settle at the top and the exit stub, whatever the flight's
            length. Eight metres is about thirty actions at the benchmark's
            step -- what the Ranchester recordings spent per completed
            flight. A cost, not a budget: it makes RPT* prefer a room here
            to a storey there at equal probability; it never forbids the
            stairs, and nothing counts actions against it.
        commit_failures: Ground truth only. Blocked forward steps a committed
            traversal absorbs -- tightening its following, then skipping a
            grazed vertex -- before it turns back. Generous on purpose: the
            connector is walkable by construction, and a retreat spends the
            climb twice and the search once more on the same decision.
        commit_stall_actions: Ground truth only. Actions with neither XY
            progress nor a polyline vertex reached before a traversal turns
            back -- a stuck-agent bound, not a budget for the climb.
        confirm_actions: Ground truth only. Actions at the destination
            height near the far anchor after which the storey is confirmed
            outright when the atlas's translated-plateau test has not fired
            -- a landing walled on three sides, an exit stub blocked at every
            heading. The traversal has done its part; the transition ends.
        approach_failures: Consecutive actions the approach to a chosen
            connector may fail to plan (the entry re-snapped each time)
            before the connector is deferred.
        arrival_grace_actions: Ground truth only. Actions after arriving on a
            storey before the staircase just climbed is offered again as a
            node or a fallback candidate. A storey entered two frames ago has
            one tentative room and no frontier on its map, and the node
            oracle -- told the agent had just arrived -- sent it straight
            back down three times in one Pomaria episode (78, 270, 493): half
            the episode on a flawless staircase. The decision to change
            floors, once executed, is given at least this long to pay for
            itself; walking onto the stairs by accident still completes the
            climb, and any OTHER staircase is offered at once.
    """

    enabled: bool = True
    stair_source: str = STAIR_SOURCE_GROUND_TRUTH
    near_connector_m: float = 0.75
    departure_m: float = 0.45
    floor_match_m: float = 0.30
    min_floor_separation_m: float = 1.5
    min_floor_area_m2: float = 3.0
    stable_height_m: float = 0.12
    stable_distance_m: float = 0.35
    stable_samples: int = 3
    floor_search_actions: int = 140
    transition_actions: int = 120
    retreat_actions: int = 80
    exit_distance_m: float = 0.75
    portal_cooldown_actions: int = 100
    max_step_m: float = 0.24
    terrain_radius_m: float = 6.0
    stair_min_rise_m: float = 0.40
    max_failures: int = 4
    floor_change_cost_m: float = 8.0
    commit_failures: int = 12
    commit_stall_actions: int = 30
    confirm_actions: int = 24
    approach_failures: int = 3
    arrival_grace_actions: int = 40

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("multifloor.enabled must be boolean")
        if self.stair_source not in STAIR_SOURCES:
            raise ValueError("multifloor.stair_source must be one of %s; no silent fallback" % (STAIR_SOURCES,))
        for name in ("departure_m", "floor_match_m", "min_floor_separation_m",
                     "min_floor_area_m2", "stable_height_m", "stable_distance_m", "max_step_m",
                     "terrain_radius_m", "stair_min_rise_m", "exit_distance_m", "near_connector_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if (isinstance(self.floor_change_cost_m, bool) or not isinstance(self.floor_change_cost_m, (int, float))
                or not math.isfinite(self.floor_change_cost_m) or self.floor_change_cost_m < 0):
            raise ValueError("floor_change_cost_m must be finite and non-negative")
        for name in ("stable_samples", "floor_search_actions", "transition_actions",
                     "retreat_actions", "portal_cooldown_actions", "max_failures",
                     "commit_failures", "commit_stall_actions", "confirm_actions", "approach_failures",
                     "arrival_grace_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if not self.stable_height_m < self.floor_match_m < self.departure_m < self.min_floor_separation_m:
            raise ValueError("Floor hysteresis thresholds must be strictly ordered")
        if self.stable_samples < 3:
            raise ValueError("At least three translated samples must confirm a floor")


@dataclass
class ObservedFloor:
    id: int
    elevation_m: float
    visits: int = 1


@dataclass
class FloorConnection:
    id: int
    source: int
    destination: int
    path_xyz: List[Tuple[float, float, float]]
    traversals: int = 1

    @property
    def length_m(self):
        return sum(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
                   for a, b in zip(self.path_xyz, self.path_xyz[1:]))

    def oriented(self, floor_id):
        if floor_id == self.source:
            return self.destination, list(self.path_xyz)
        if floor_id == self.destination:
            return self.source, list(reversed(self.path_xyz))
        raise ValueError("Connection does not touch floor")


class FloorAtlas:
    """Height hysteresis, revisits, and bidirectional measured connectivity.

    A floor is confirmed on a translated plateau, not by each 0.6 m of climb.
    Links require an actually executed continuous path between settled floors;
    merely seeing another level never manufactures a connection.
    """

    def __init__(self, params=None):
        self.params = params or MultiFloorParams()
        self.plateau_observer = None
        self.floors: Dict[int, ObservedFloor] = {}
        self.connections: List[FloorConnection] = []
        self.active_id = 0
        self.revision = 0
        self.in_transition = False
        self.last_connection: Optional[int] = None
        self._samples = deque(maxlen=self.params.stable_samples)
        self._trail: List[Tuple[float, float, float]] = []
        self._previous = None
        self._approach = deque(maxlen=20)
        self.destination_height_m = None
        self.completion_reason = None

    @property
    def elevation_m(self):
        return self.floors[self.active_id].elevation_m

    def begin_transition(self, pose):
        """Commit before the first tread, independently of departure hysteresis."""
        if not self.floors:
            self.update(pose)
        if not self.in_transition:
            self.in_transition = True
            self._trail = list(self._approach) or [(pose.x, pose.y, pose.z)]
            self._samples.clear()
            self.completion_reason = None

    def update(self, pose, plateau_area_m2=None, *, arrival_allowed=True):
        if plateau_area_m2 is None and self.plateau_observer is not None:
            plateau_area_m2 = self.plateau_observer(pose)
        xyz = (float(pose.x), float(pose.y), float(pose.z))
        if not self.floors:
            self.floors[0] = ObservedFloor(0, xyz[2])
        if self._previous is None:
            self._previous = xyz
        moved = math.hypot(xyz[0] - self._previous[0], xyz[1] - self._previous[1])
        if not self.in_transition:
            if abs(xyz[2] - self.elevation_m) < self.params.stable_height_m:
                self._approach.clear()
            if not self._approach or moved > 0.04:
                self._approach.append(xyz)
        if abs(xyz[2] - self.elevation_m) > self.params.departure_m and not self.in_transition:
            self.in_transition = True
            self._trail = list(self._approach) or [self._previous]
            self._samples.clear()
        if self.in_transition:
            if not self._trail or moved > 0.04:
                self._trail.append(xyz)
            if moved > 0.04:
                self._samples.append(xyz)
            if self._settled():
                height = sum(p[2] for p in self._samples) / len(self._samples)
                self.destination_height_m = height
                matches = [f for f in self.floors.values()
                           if abs(f.elevation_m - height) <= self.params.floor_match_m]
                supported = plateau_area_m2 is None or plateau_area_m2 >= self.params.min_floor_area_m2
                if not arrival_allowed or not supported:
                    destination = None
                elif matches:
                    destination = min(matches, key=lambda f: abs(f.elevation_m - height)).id
                elif min(abs(f.elevation_m - height) for f in self.floors.values()) >= self.params.min_floor_separation_m:
                    destination = len(self.floors)
                    self.floors[destination] = ObservedFloor(destination, height, visits=0)
                else:
                    destination = None  # a stair landing, not another storey
                if destination is not None:
                    self._arrive(destination)
            else:
                self.destination_height_m = None
        self._previous = xyz
        return self.active_id

    def _settled(self):
        if len(self._samples) < self.params.stable_samples:
            return False
        heights = [p[2] for p in self._samples]
        # Net translation, not a back-and-forth shuffle on one tread.
        distance = math.hypot(self._samples[-1][0] - self._samples[0][0],
                              self._samples[-1][1] - self._samples[0][1])
        return max(heights) - min(heights) <= self.params.stable_height_m and distance >= self.params.stable_distance_m

    def settle(self, pose, height):
        """Confirm arrival at the storey of ``height`` now, without the plateau test.

        For a caller that KNOWS the agent stands on a storey -- the
        ground-truth traversal at a connector's far anchor, at the
        destination height the navmesh reported -- and has waited long
        enough for the translated-plateau test to be evidently not going to
        fire: a landing walled on three sides, an exit stub blocked at every
        heading. Poses only, still: the height is the caller's, but the
        storey it becomes is matched against the floors this atlas has
        measured, or created at the separation rule every other floor obeys.

        Args:
            pose: The agent's pose, whose XY ends the measured stair edge.
            height: The storey height to confirm.

        Returns:
            True when a floor was matched or created and arrival recorded;
            False when ``height`` is neither an existing floor nor far enough
            from every floor to be a new one -- a landing, refused.
        """
        if not self.floors:
            self.update(pose)
        xyz = (float(pose.x), float(pose.y), float(pose.z))
        if not self.in_transition:
            self.in_transition = True
            self._trail = list(self._approach) or [xyz]
        if not self._trail or self._trail[-1] != xyz:
            self._trail.append(xyz)
        matches = [f for f in self.floors.values()
                   if abs(f.elevation_m - float(height)) <= self.params.floor_match_m]
        if matches:
            destination = min(matches, key=lambda f: abs(f.elevation_m - float(height))).id
        elif min(abs(f.elevation_m - float(height)) for f in self.floors.values()) >= self.params.min_floor_separation_m:
            destination = len(self.floors)
            self.floors[destination] = ObservedFloor(destination, float(height), visits=0)
        else:
            return False
        self.destination_height_m = float(height)
        self._arrive(destination)
        return True

    def _arrive(self, destination):
        self.completion_reason = "destination_platform_confirmed" if destination != self.active_id else "source_platform_returned"
        if destination != self.active_id:
            match = None
            for edge in self.connections:
                if self.active_id not in (edge.source, edge.destination):
                    continue
                other, path = edge.oriented(self.active_id)
                if other == destination and math.hypot(path[0][0] - self._trail[0][0], path[0][1] - self._trail[0][1]) < 1.5:
                    match = edge
                    break
            if match is None:
                match = FloorConnection(len(self.connections), self.active_id, destination, list(self._trail))
                self.connections.append(match)
            else:
                match.traversals += 1
            self.last_connection = match.id
            self.active_id = destination
            self.floors[destination].visits += 1
            self.revision += 1
        self.in_transition = False
        self.destination_height_m = None
        self._samples.clear()
        self._approach.clear()
        if self._trail:
            self._approach.append(self._trail[-1])
        self._trail = []

    def first_hop(self, destination):
        """Dijkstra on measured 3D transition lengths, not XY-overlaid floors."""
        queue = [(0.0, self.active_id, -1)]
        seen = set()
        while queue:
            cost, floor_id, first = heapq.heappop(queue)
            if floor_id in seen:
                continue
            seen.add(floor_id)
            if floor_id == destination:
                return None if first < 0 else self.connections[first]
            for edge in self.connections:
                if floor_id in (edge.source, edge.destination):
                    other, _ = edge.oriented(floor_id)
                    heapq.heappush(queue, (cost + edge.length_m, other, edge.id if first < 0 else first))
        return None

    def diagnostics(self):
        return {"active_floor": self.active_id, "in_transition": self.in_transition,
                "destination_height_m": self.destination_height_m, "completion_reason": self.completion_reason,
                "revision": self.revision, "floors": [asdict(f) for f in self.floors.values()],
                "connections": [dict(asdict(e), length_m=e.length_m) for e in self.connections]}

