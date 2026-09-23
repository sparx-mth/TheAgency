"""Payload assembly for the scene-graph latched topics.

Pure functions, no ROS imports — the mapper nodes call these to build the
``/perception/objects``, ``/scene_graph`` and
``/scene_graph/room_labels_grid`` messages, and the unit tests round-trip them
through ``json`` without an rclpy context. Every value is coerced to a plain
Python ``int``/``float``/``list`` here, so a numpy scalar smuggled in by a
caller cannot make ``json.dumps`` raise at publish time.

The dict shapes are the fixed scene-graph topic contract; renaming a key here
breaks every downstream consumer (room classifier, oracle, target watcher,
viz).

Room grid values (the ``room_labels_grid`` indirection)
-------------------------------------------------------
``nav_msgs/OccupancyGrid`` data is ``int8`` while room pids come from
``RoomRegistry`` and grow without bound — pid 130 written straight into a cell
would wrap negative. So the grid carries a small **grid value** per room
(``NO_ROOM_VALUE`` where there is no room, else ``MIN_ROOM_VALUE`` ..
``MAX_ROOM_VALUE``) and the ``/scene_graph`` payload carries the
``grid_pid_map`` that resolves a grid value back to its pid.

:func:`assign_room_grid_values`, :func:`room_value_grid` and
:func:`scene_graph_payload` are meant to be fed the *same* ``grid_values``
mapping in one tick, which is what makes the pair coherent: every non-zero
cell in the published grid then has an entry in the published map, so the
viz's fallback (tint by the raw cell value, i.e. a foreign room's colour)
never fires.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.mapping.objects.landmarks import ObjectLandmark

NO_ROOM_VALUE = 0
MIN_ROOM_VALUE = 1
MAX_ROOM_VALUE = 100


def object_entry(landmark: ObjectLandmark) -> Dict:
    """One confirmed landmark as its wire dict: ``{id, class, xy, count}``."""
    return {
        "id": int(landmark.id),
        "class": str(landmark.class_name),
        "xy": [float(landmark.xy[0]), float(landmark.xy[1])],
        "count": int(landmark.count),
    }


def objects_payload(stamp: float,
                    landmarks: Iterable[ObjectLandmark]) -> Dict:
    """The full ``/perception/objects`` payload for a confirmed-landmark set."""
    return {
        "stamp": float(stamp),
        "objects": [object_entry(lm) for lm in landmarks],
    }


def room_entry(room_id: int,
               centroid_xy: Tuple[float, float],
               n_cells: int,
               time_in_room_s: float,
               frontier_clusters: int,
               color_rgb: Tuple[float, float, float],
               objects: Sequence[Dict],
               door_indices: Sequence[int]) -> Dict:
    """One room of the ``/scene_graph`` payload.

    Args:
        room_id: Persistent room id (pid).
        centroid_xy: Room centroid in world ENU metres.
        n_cells: Cell count of the room mask.
        time_in_room_s: Accumulated drone dwell time in this room.
        frontier_clusters: Frontier cluster count assigned to this room.
        color_rgb: Room display color, floats in [0, 1].
        objects: Object entries already shaped by :func:`object_entry`.
        door_indices: Indices (into the payload's ``doors`` list) of the
            doors linked to this room.
    """
    return {
        "id": int(room_id),
        "centroid": [float(centroid_xy[0]), float(centroid_xy[1])],
        "cells": int(n_cells),
        "time_in_room_s": float(time_in_room_s),
        "frontier_clusters": int(frontier_clusters),
        "color": [float(c) for c in color_rgb],
        "objects": list(objects),
        "doors": [int(i) for i in door_indices],
    }


def door_entry(index: int,
               xy: Tuple[float, float],
               discovered: bool,
               room_ids: Sequence[int],
               room_pairs: Sequence[Sequence[int]] = ()) -> Dict:
    """One door of the ``/scene_graph`` payload.

    Args:
        index: Index into the payload's ``doors`` list, and into the
            door YAML the mapper loaded.
        xy: Door position in world ENU metres.
        discovered: Whether the map has seen this door at all.
        room_ids: Pids of the rooms this door connects — the union of
            ``room_pairs``, so a door that connects nothing lists no
            room.
        room_pairs: The room-to-room EDGES this door carries, as
            ``(pid, pid)`` pairs. A separate field rather than every
            pair of ``rooms`` because the pairs are NOT a clique: a
            door can sit where rooms A-B and B-C touch while A and C
            have a wall between them, and pairing up the room list
            would invent the A-C edge. Measured on a captured hospital
            BEV, 23 of its 35 doors see three or more rooms and 4 of
            the 61 pairs their proximity proposes cross a wall.
            Consumers draw these pairs and nothing else.
    """
    return {
        "index": int(index),
        "xy": [float(xy[0]), float(xy[1])],
        "discovered": bool(discovered),
        "rooms": [int(r) for r in room_ids],
        "room_pairs": [[int(a), int(b)] for a, b in room_pairs],
    }


def assign_room_grid_values(pids: Iterable[int],
                            previous: Optional[Mapping[int, int]] = None
                            ) -> Dict[int, int]:
    """Stable ``pid -> grid value`` assignment inside ``1..100``.

    A room keeps the value it was given for as long as it keeps its pid, so
    the viz never re-tints a persisting room; a value is recycled only once
    its room is gone from ``pids``. New pids take the lowest free value, in
    the order they are given, so the assignment is deterministic.

    Args:
        pids: Pids of the rooms alive this tick (duplicates ignored).
        previous: The mapping returned by the previous call, or None on the
            first tick / after a grid reshape (which restarts pids, and so
            must restart this mapping too).

    Returns:
        ``{pid: grid value}`` for the rooms alive this tick. A value carried
        in through ``previous`` that is out of band or already claimed is
        dropped and its pid reassigned — no two live rooms may share a value.
        With more live rooms than the band has values, the surplus rooms get
        no entry at all (they are simply absent from the grid and the map,
        never mis-coloured); callers should compare the returned length
        against their room count and say so.
    """
    previous = dict(previous or {})
    ordered = []  # type: List[int]
    seen = set()
    for pid in pids:
        pid = int(pid)
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)

    values = {}  # type: Dict[int, int]
    taken = set()
    for pid in ordered:
        value = previous.get(pid)
        if value is None:
            continue
        value = int(value)
        if MIN_ROOM_VALUE <= value <= MAX_ROOM_VALUE and value not in taken:
            values[pid] = value
            taken.add(value)

    free = (v for v in range(MIN_ROOM_VALUE, MAX_ROOM_VALUE + 1)
            if v not in taken)
    for pid in ordered:
        if pid in values:
            continue
        value = next(free, None)
        if value is None:
            break
        values[pid] = value
    return values


def grid_pid_map(grid_values: Mapping[int, int]) -> Dict[str, int]:
    """The ``grid value -> pid`` indirection, in its JSON-key form.

    Inverts :func:`assign_room_grid_values`: the viz reads a cell value out of
    ``/scene_graph/room_labels_grid`` and looks its pid up here by the decimal
    string of that value (JSON has no integer keys), then colours the cell by
    pid. Keys are strings in the returned dict too, so it is identical before
    and after ``json.dumps``.
    """
    return {str(int(value)): int(pid) for pid, value in grid_values.items()}


def room_value_grid(shape: Tuple[int, int],
                    masks_by_pid: Mapping[int, np.ndarray],
                    grid_values: Mapping[int, int]) -> np.ndarray:
    """The ``/scene_graph/room_labels_grid`` data as an ``int8`` image.

    Args:
        shape: ``(height, width)`` of the BEV grid this mirrors.
        masks_by_pid: ``{pid: (H, W) bool mask}`` of the rooms alive this tick.
        grid_values: ``{pid: grid value}`` from
            :func:`assign_room_grid_values`.

    Returns:
        ``(H, W)`` int8, ``NO_ROOM_VALUE`` outside every room and the room's
        grid value inside it. Row 0 is minimum y, as ``nav_msgs/OccupancyGrid``
        requires.

    Raises:
        KeyError: If a pid holds a grid value but no mask — the grid and the
            ``grid_pid_map`` would then disagree about what is on the map.
    """
    out = np.full((int(shape[0]), int(shape[1])), NO_ROOM_VALUE, dtype=np.int8)
    for pid, value in grid_values.items():
        if pid not in masks_by_pid:
            raise KeyError("room pid %d has grid value %d but no mask"
                           % (int(pid), int(value)))
        out[np.asarray(masks_by_pid[pid], dtype=bool)] = np.int8(int(value))
    return out


def scene_graph_payload(stamp: float,
                        resolution: float,
                        origin_xy: Tuple[float, float],
                        rooms: Sequence[Dict],
                        doors: Sequence[Dict],
                        drone_xy: Optional[Tuple[float, float]],
                        drone_room_id: Optional[int],
                        grid_values: Optional[Mapping[int, int]] = None
                        ) -> Dict:
    """The full ``/scene_graph`` payload.

    Args:
        stamp: Publish time, float seconds.
        resolution: BEV grid resolution, metres per cell.
        origin_xy: BEV grid origin (lower-left corner) in world ENU metres.
        rooms: Room entries from :func:`room_entry`.
        doors: Door entries from :func:`door_entry`.
        drone_xy: Latest drone position in world ENU metres, or None when no
            odometry has arrived yet.
        drone_room_id: Pid of the room the drone is in, or None.
        grid_values: ``{pid: grid value}`` from
            :func:`assign_room_grid_values` — the same mapping used to build
            the ``/scene_graph/room_labels_grid`` published this tick. It is
            emitted inverted, as ``grid_pid_map``.
    """
    return {
        "stamp": float(stamp),
        "resolution": float(resolution),
        "origin": [float(origin_xy[0]), float(origin_xy[1])],
        "rooms": list(rooms),
        "doors": list(doors),
        "drone": {
            "xy": ([float(drone_xy[0]), float(drone_xy[1])]
                   if drone_xy is not None else None),
            "room_id": (int(drone_room_id)
                        if drone_room_id is not None else None),
        },
        "grid_pid_map": grid_pid_map(grid_values or {}),
    }
