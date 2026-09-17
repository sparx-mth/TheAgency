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


@dataclass(frozen=True)
class MultiFloorParams:
    enabled: bool = True
    departure_m: float = 0.45
    floor_match_m: float = 0.30
    min_floor_separation_m: float = 1.5
    min_floor_area_m2: float = 3.0
    stable_height_m: float = 0.12
    stable_distance_m: float = 0.35
    stable_samples: int = 3
    floor_search_actions: int = 140
    transition_actions: int = 120
    portal_cooldown_actions: int = 100
    max_step_m: float = 0.24
    terrain_radius_m: float = 6.0
    stair_min_rise_m: float = 0.40
    max_failures: int = 4

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("multifloor.enabled must be boolean")
        for name in ("departure_m", "floor_match_m", "min_floor_separation_m",
                     "min_floor_area_m2", "stable_height_m", "stable_distance_m", "max_step_m",
                     "terrain_radius_m", "stair_min_rise_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        for name in ("stable_samples", "floor_search_actions", "transition_actions",
                     "portal_cooldown_actions", "max_failures"):
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

    @property
    def elevation_m(self):
        return self.floors[self.active_id].elevation_m

    def update(self, pose, plateau_area_m2=None):
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
                matches = [f for f in self.floors.values()
                           if abs(f.elevation_m - height) <= self.params.floor_match_m]
                if matches:
                    destination = min(matches, key=lambda f: abs(f.elevation_m - height)).id
                elif (min(abs(f.elevation_m - height) for f in self.floors.values()) >= self.params.min_floor_separation_m
                      and (plateau_area_m2 is None or plateau_area_m2 >= self.params.min_floor_area_m2)):
                    destination = len(self.floors)
                    self.floors[destination] = ObservedFloor(destination, height, visits=0)
                else:
                    destination = None  # a stair landing, not another storey
                if destination is not None:
                    self._arrive(destination)
        self._previous = xyz
        return self.active_id

    def _settled(self):
        if len(self._samples) < self.params.stable_samples:
            return False
        heights = [p[2] for p in self._samples]
        distance = sum(math.hypot(a[0] - b[0], a[1] - b[1])
                       for a, b in zip(self._samples, list(self._samples)[1:]))
        return max(heights) - min(heights) <= self.params.stable_height_m and distance >= self.params.stable_distance_m

    def _arrive(self, destination):
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
                "revision": self.revision, "floors": [asdict(f) for f in self.floors.values()],
                "connections": [dict(asdict(e), length_m=e.length_m) for e in self.connections]}

