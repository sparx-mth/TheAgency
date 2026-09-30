"""What the policy knows about the building's stairs when the simulator tells it.

The evaluator attaches the navmesh's storeys and stair connectors to the
episode metadata (``gibson/stair_connectors.py``). This module is the policy's
reading of that block: which connectors touch the floor the agent stands on,
which way each one goes, and whether an unplanned change of height happened
ON a staircase or on a step. It holds geometry only -- no decision, no
motion; those are ``floor_decision.py`` and ``ground_truth_traversal.py``.

**Knowing the geometry is not knowing the stairs.** The coordinator learns
that a staircase exists only when the perfect detector
(:mod:`stair_sightings`) has seen it in a frame; until then a connector here
is the detector's instrument, not a portal, not a node, not an explanation.
:meth:`GroundTruthStairs.touching` and :meth:`GroundTruthStairs.nearest`
take an ``among`` filter so callers ask only about the stairs they have seen.

Floor ids stay the atlas's discovery ids: a connector names storey HEIGHTS,
and the atlas floor at a height is looked up when it is needed, so a storey
the agent has never stood on has no id yet -- which is exactly what
"unvisited" means downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Xyz = Tuple[float, float, float]


@dataclass(frozen=True)
class Connector:
    """One staircase, ENU, bottom to top.

    Attributes:
        id: Index in the scene's connector list.
        bottom: Floor anchor at the lower storey.
        top: Floor anchor at the upper storey.
        bottom_z: Lower storey height.
        top_z: Upper storey height.
        polyline: Walkable centreline from ``bottom`` to ``top``.
        length_m: Polyline length.
        surface: A sample of the stair surface, for the detector's visibility
            test; empty when the metadata carried none (the polyline stands in).
        bottom_exit: Flat floor off the foot of the flight, walkable from
            ``bottom``, where the atlas can confirm the lower storey; None
            when the extraction found none.
        top_exit: The same off the head of the flight.
        traversable: False when the extraction found the flight split
            across navmesh islands -- a staircase to look at, never to
            climb: the simulator lets no agent across such a gap.
    """

    id: int
    bottom: Xyz
    top: Xyz
    bottom_z: float
    top_z: float
    polyline: Tuple[Xyz, ...]
    length_m: float
    surface: Tuple[Xyz, ...] = ()
    bottom_exit: Optional[Xyz] = None
    top_exit: Optional[Xyz] = None
    traversable: bool = True

    def touches(self, height: float, tolerance_m: float) -> bool:
        """Does either end of this staircase stand on a storey at ``height``?"""
        return abs(self.bottom_z - height) <= tolerance_m or abs(self.top_z - height) <= tolerance_m

    def oriented(self, height: float, tolerance_m: float) -> Tuple[int, Xyz, Xyz, float, Tuple[Xyz, ...]]:
        """The staircase as seen from the storey at ``height``.

        Returns:
            ``(direction, entry, exit, destination_z, polyline)`` -- ``+1``
            climbing / ``-1`` descending, the anchor on this storey, the
            anchor on the other one, the other storey's height, and the
            polyline from this storey's anchor to the other.

        Raises:
            ValueError: When neither end is on a storey at ``height``.
        """
        if abs(self.bottom_z - height) <= tolerance_m:
            return 1, self.bottom, self.top, self.top_z, self.polyline
        if abs(self.top_z - height) <= tolerance_m:
            return -1, self.top, self.bottom, self.bottom_z, tuple(reversed(self.polyline))
        raise ValueError("Connector %d does not touch a storey at %.2f m" % (self.id, height))

    def far_exit(self, height: float, tolerance_m: float) -> Optional[Xyz]:
        """The verified exit point on the OTHER storey, as seen from the storey at ``height``; None when unknown."""
        if abs(self.bottom_z - height) <= tolerance_m:
            return self.top_exit
        if abs(self.top_z - height) <= tolerance_m:
            return self.bottom_exit
        raise ValueError("Connector %d does not touch a storey at %.2f m" % (self.id, height))

    def distance_xy(self, x: float, y: float) -> float:
        """Shortest XY distance from ``(x, y)`` to the polyline."""
        best = math.inf
        for (ax, ay, _), (bx, by, _) in zip(self.polyline, self.polyline[1:]):
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 <= 1e-12 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
            best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
        if len(self.polyline) == 1:
            best = math.hypot(x - self.polyline[0][0], y - self.polyline[0][1])
        return best


class GroundTruthStairs:
    """The scene's connectors and storeys, read once per episode.

    Attributes:
        connectors: Every staircase in the scene.
        levels: Storey heights, ascending.
        available: False when the episode carried no scene structure -- a
            simulator that cannot provide it, or a single-storey scene with
            nothing to connect. Either way there is nothing to climb.
    """

    def __init__(self, connectors: Sequence[Connector], levels: Sequence[float], available: bool):
        self.connectors = list(connectors)
        self.levels = sorted(float(level) for level in levels)
        self.available = available

    @classmethod
    def from_metadata(cls, metadata: Optional[Dict]) -> "GroundTruthStairs":
        """Read the ``stair_connectors`` / ``floor_levels`` block the evaluator attached."""
        metadata = metadata or {}
        rows = metadata.get("stair_connectors")
        if rows is None:
            return cls([], [], available=False)
        connectors = []
        for row in rows:
            polyline = tuple(tuple(float(v) for v in point[:3]) for point in row["polyline_xyz"])
            if len(polyline) < 2:
                raise ValueError("Stair connector %r needs a polyline of at least two points" % (row.get("id"),))
            surface = tuple(tuple(float(v) for v in point[:3]) for point in row.get("surface_xyz", ()) or ())
            exits = [None if row.get(key) is None else tuple(float(v) for v in row[key][:3])
                     for key in ("bottom_exit_xyz", "top_exit_xyz")]
            connectors.append(Connector(
                id=int(row["id"]), bottom=tuple(float(v) for v in row["bottom_xyz"][:3]),
                top=tuple(float(v) for v in row["top_xyz"][:3]), bottom_z=float(row["bottom_z"]),
                top_z=float(row["top_z"]), polyline=polyline, length_m=float(row.get("length_m", 0.0)),
                surface=surface, bottom_exit=exits[0], top_exit=exits[1],
                traversable=bool(row.get("traversable", True))))
        levels = [float(level["height_m"]) for level in metadata.get("floor_levels", [])]
        return cls(connectors, levels, available=True)

    def by_id(self, connector_id: int) -> Optional[Connector]:
        return next((c for c in self.connectors if c.id == int(connector_id)), None)

    def touching(self, height: float, tolerance_m: float,
                 among: Optional[Callable[[int], bool]] = None) -> List[Connector]:
        """Traversable connectors with an end on the storey at ``height`` -- only those ``among`` admits, when given."""
        return [c for c in self.connectors
                if c.traversable and c.touches(height, tolerance_m) and (among is None or among(c.id))]

    def nearest(self, x: float, y: float, height: float, tolerance_m: float,
                within_m: float, among: Optional[Callable[[int], bool]] = None) -> Optional[Connector]:
        """The connector touching this storey whose polyline passes within ``within_m`` of ``(x, y)``.

        ``among`` restricts the candidates -- the coordinator passes "seen",
        so a change of height on stairs the agent has never looked at is not
        explained by them.
        """
        candidates = [(c.distance_xy(x, y), c) for c in self.touching(height, tolerance_m, among)]
        candidates = [(d, c) for d, c in candidates if d <= within_m]
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def diagnostics(self) -> Dict:
        return {"available": self.available, "levels_m": list(self.levels),
                "connectors": [{"id": c.id, "bottom_z": c.bottom_z, "top_z": c.top_z,
                                "bottom_xy": list(c.bottom[:2]), "top_xy": list(c.top[:2]),
                                "length_m": c.length_m, "surface_points": len(c.surface),
                                "traversable": c.traversable} for c in self.connectors]}

