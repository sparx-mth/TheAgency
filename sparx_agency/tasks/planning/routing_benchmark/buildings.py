"""Buildings to search, laid out the way buildings actually are.

The temptation when benchmarking a routing algorithm is to scatter points in a
square and call the straight-line distances between them a problem. That is a
bad test, and specifically it is a bad test *of this algorithm*, because in a
square every place is roughly as reachable as every other and the ordering
barely matters. In a building it does: two rooms either side of a corridor wall
are three metres apart and forty metres of walking, and a wing you have to
double back out of is expensive in a way no point cloud reproduces.

So a building here is a **corridor graph**. Corridor nodes carry the walkable
spine; rooms hang off them through doors. Travel cost between two rooms is the
shortest path through that graph -- out of the first room, along the corridors,
into the second. Three things follow, and all three matter:

* the costs are genuinely building-shaped: same-corridor rooms are cheap, rooms
  in different wings are dear, and a dead-end wing costs double to visit;
* they are **shortest paths, so they satisfy the triangle inequality exactly**,
  which is what RPT*'s pruning requires. A generator that produced costs
  violating it would be testing the validator, not the search;
* everything has a 2D position, so a scenario can be drawn.

Four topologies, chosen because they make the ordering problem hard in
different ways -- see :data:`TOPOLOGIES`.

Standard library only, so this runs anywhere the solver does.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: The topologies a building can have. Each stresses the ordering differently:
#:
#: ``corridor``
#:     One straight spine with rooms alternating left and right. The easy case:
#:     a sensible route is roughly "walk down, looking in doors".
#: ``ring``
#:     A loop of corridor with rooms on the outside. There are two ways round,
#:     so committing to the wrong direction is expensive -- this is the shape
#:     of the hospital floor in the paper's own Fig. 12.
#: ``cross``
#:     Four wings from a central hub. Every wing is a dead end, so visiting one
#:     room deep in a wing costs the walk back out. Punishes greedy ordering
#:     hardest.
#: ``suite``
#:     Rooms chained through one another, apartment-style, with no corridor.
#:     Reaching the far room means passing through every room before it, so
#:     the ordering interacts with the geometry rather than sitting on top of
#:     it.
TOPOLOGIES = ("corridor", "ring", "cross", "suite")


@dataclass(frozen=True)
class Room:
    """One searchable place, and where it is.

    Attributes:
        id: Stable identifier, unique within a building.
        name: Human label, e.g. ``"room 7"``.
        xy: Centroid in metres, for drawing and for nothing else -- costs come
            from the corridor graph, never from this.
        corridor_node: The corridor node its door opens onto.
        door_m: How far inside the room its centroid is from that door.
    """

    id: str
    name: str
    xy: Tuple[float, float]
    corridor_node: int
    door_m: float


@dataclass(frozen=True)
class Building:
    """A floor plan: rooms, a corridor spine, and the walking distances.

    Attributes:
        topology: Which of :data:`TOPOLOGIES` it is.
        rooms: The searchable places.
        entrance: The corridor node the robot starts at.
        entrance_xy: Where that is, in metres.
        node_xy: Position of every corridor node, for drawing.
        corridor_edges: The corridor spine, as node pairs, for drawing.
        distance: ``distance[a][b]`` in metres between room indices, and the
            last row/column is the entrance. Shortest paths, so metric.
    """

    topology: str
    rooms: Tuple[Room, ...]
    entrance: int
    entrance_xy: Tuple[float, float]
    node_xy: Tuple[Tuple[float, float], ...]
    corridor_edges: Tuple[Tuple[int, int], ...]
    distance: Tuple[Tuple[float, ...], ...]

    @property
    def n_rooms(self):
        # type: () -> int
        """How many searchable places there are."""
        return len(self.rooms)


def generate_building(topology, n_rooms, rng, segment_m=6.0, door_m=3.0):
    # type: (str, int, random.Random, float, float) -> Building
    """Lay out a floor plan of the requested shape and size.

    Args:
        topology: One of :data:`TOPOLOGIES`.
        n_rooms: How many searchable rooms to place.
        rng: Randomness, so a scenario is reproducible from its seed.
        segment_m: Nominal corridor segment length in metres; jittered per
            segment so no two buildings are identical.
        door_m: Nominal distance from a door to the room's centroid.

    Returns:
        The building, with its distance matrix already computed.

    Raises:
        ValueError: On an unknown topology or fewer than two rooms.
    """
    if n_rooms < 2:
        raise ValueError("a building needs at least 2 rooms, got %d" % n_rooms)
    if topology not in TOPOLOGIES:
        raise ValueError("unknown topology %r, expected one of %r"
                         % (topology, TOPOLOGIES))

    builder = {
        "corridor": _corridor,
        "ring": _ring,
        "cross": _cross,
        "suite": _suite,
    }[topology]
    node_xy, edges, attachments, entrance = builder(n_rooms, rng, segment_m)

    rooms = []                                  # type: List[Room]
    for index, node in enumerate(attachments):
        depth = door_m * rng.uniform(0.7, 1.3)
        rooms.append(Room(
            id="room_%02d" % index,
            name="room %d" % index,
            xy=_offset(node_xy, edges, node, depth, index),
            corridor_node=node,
            door_m=depth,
        ))
    node_distance = _all_pairs(len(node_xy), edges, node_xy)
    distance = _room_distances(rooms, entrance, node_distance)
    return Building(
        topology=topology,
        rooms=tuple(rooms),
        entrance=entrance,
        entrance_xy=node_xy[entrance],
        node_xy=tuple(node_xy),
        corridor_edges=tuple(edges),
        distance=distance,
    )


# -- the four shapes ------------------------------------------------------

def _corridor(n_rooms, rng, segment_m):
    """A straight spine, rooms alternating either side."""
    spans = (n_rooms + 1) // 2
    xy = [(i * segment_m * rng.uniform(0.85, 1.15), 0.0)
          for i in range(spans + 1)]
    edges = [(i, i + 1) for i in range(spans)]
    attachments = [1 + (i // 2) for i in range(n_rooms)]
    return xy, edges, attachments, 0


def _ring(n_rooms, rng, segment_m):
    """A loop of corridor with the rooms on the outside."""
    spans = max(4, (n_rooms + 1) // 2)
    radius = spans * segment_m / (2.0 * math.pi)
    xy = [(radius * math.cos(2 * math.pi * i / spans),
           radius * math.sin(2 * math.pi * i / spans))
          for i in range(spans)]
    edges = [(i, (i + 1) % spans) for i in range(spans)]
    attachments = [i % spans for i in range(n_rooms)]
    return xy, edges, attachments, 0


def _cross(n_rooms, rng, segment_m):
    """Four dead-end wings off a central hub."""
    xy = [(0.0, 0.0)]
    edges = []
    attachments = []
    per_wing = max(1, (n_rooms + 3) // 4)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for wing, (dx, dy) in enumerate(directions):
        previous = 0
        for step in range(1, per_wing + 1):
            reach = step * segment_m * rng.uniform(0.85, 1.15)
            xy.append((dx * reach, dy * reach))
            edges.append((previous, len(xy) - 1))
            previous = len(xy) - 1
        # rooms on this wing hang off its nodes, deepest last
        base = len(xy) - per_wing
        for step in range(per_wing):
            attachments.append(base + step)
    return xy, edges, attachments[:n_rooms], 0


def _suite(n_rooms, rng, segment_m):
    """Rooms chained through one another, with no corridor at all."""
    xy = [(i * segment_m * rng.uniform(0.85, 1.15), 0.0)
          for i in range(n_rooms + 1)]
    edges = [(i, i + 1) for i in range(n_rooms)]
    attachments = [i + 1 for i in range(n_rooms)]
    return xy, edges, attachments, 0


# -- geometry and distances ----------------------------------------------

def _offset(node_xy, edges, node, depth, index):
    """Put a room to one side of its corridor node, alternating sides."""
    neighbours = [b for a, b in edges if a == node] + \
                 [a for a, b in edges if b == node]
    origin = node_xy[node]
    if neighbours:
        other = node_xy[neighbours[0]]
        dx, dy = other[0] - origin[0], other[1] - origin[1]
        length = math.hypot(dx, dy) or 1.0
        normal = (-dy / length, dx / length)
    else:
        normal = (0.0, 1.0)
    side = 1.0 if index % 2 == 0 else -1.0
    return (origin[0] + normal[0] * depth * side,
            origin[1] + normal[1] * depth * side)


def _all_pairs(count, edges, node_xy):
    """Shortest walking distance between every pair of corridor nodes."""
    big = float("inf")
    best = [[0.0 if i == j else big for j in range(count)]
            for i in range(count)]
    for a, b in edges:
        length = math.hypot(node_xy[a][0] - node_xy[b][0],
                            node_xy[a][1] - node_xy[b][1])
        best[a][b] = min(best[a][b], length)
        best[b][a] = min(best[b][a], length)
    for via in range(count):
        row_via = best[via]
        for a in range(count):
            through = best[a][via]
            if through == big:
                continue
            row_a = best[a]
            for b in range(count):
                candidate = through + row_via[b]
                if candidate < row_a[b]:
                    row_a[b] = candidate
    return best


def _room_distances(rooms, entrance, node_distance):
    """Room-to-room walking distance, entrance last.

    Out of one room to its door, along the corridors, in through the other
    door. Because every leg is a shortest path and the door offsets are added
    symmetrically, the result satisfies the triangle inequality -- which is
    what the solver requires and what a naive generator gets wrong.
    """
    places = [(room.corridor_node, room.door_m) for room in rooms]
    places.append((entrance, 0.0))
    size = len(places)
    matrix = []                                 # type: List[Tuple[float, ...]]
    for a in range(size):
        node_a, depth_a = places[a]
        row = []                                # type: List[float]
        for b in range(size):
            if a == b:
                row.append(0.0)
                continue
            row.append(depth_a + node_distance[node_a][places[b][0]]
                       + places[b][1])
        matrix.append(tuple(row))
    return tuple(matrix)
