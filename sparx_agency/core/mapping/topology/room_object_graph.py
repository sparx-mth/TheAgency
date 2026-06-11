# core/mapping/topology/room_object_graph.py
"""
Hierarchical room–object graph construction.

Builds a directed graph following the MORE (Werby et al., 2025) convention:

    root ──► room_0 ──► object_a
                    ──► object_b
         ──► room_1 ──► object_c
         ...

Each room corresponds to a connected component of the door-separated
Voronoi graph.  Objects are assigned to the room whose component
contains the nearest Voronoi node.

Dependencies: numpy, scipy, networkx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix as scipy_dist_matrix


# ── Node-type enum (mirrors MORE's NODETYPE) ────────────────────────────────

class NodeType(IntEnum):
    """Categorical labels stored in the ``node_type`` attribute."""
    ROOT = 0
    ROOM = 1
    OBJECT = 2

    @staticmethod
    def roomname(room_id: int) -> str:
        """Canonical room-node key, e.g. ``'room_0'``."""
        return f"room_{room_id}"


# ── Object descriptor ───────────────────────────────────────────────────────

@dataclass
class ObjectInfo:
    """
    A tangible object living inside the occupancy grid.

    Attributes:
        name:            Unique human-readable label (e.g. ``'tv'``).
        position:        (row, col) centre in **pixel** coordinates.
        bbox:            (r1, r2, c1, c2) axis-aligned bounding box in the grid.
        semantic_class:  Free-form category string (``'furniture'``,
                         ``'appliance'``, …).
        room_id:         Assigned after :func:`assign_objects_to_rooms`.
        closest_vor_node: Nearest Voronoi-graph node (row, col) tuple.
    """
    name: str
    position: np.ndarray                       # (2,)
    bbox: Tuple[int, int, int, int]            # (r1, r2, c1, c2)
    semantic_class: str = "furniture"
    room_id: Optional[int] = None
    closest_vor_node: Optional[Tuple[float, ...]] = None


# ── Object-to-room assignment ───────────────────────────────────────────────

def assign_objects_to_rooms(
    objects: List[ObjectInfo],
    separated_graph: nx.Graph,
) -> List[ObjectInfo]:
    """
    Assign each object to the room whose Voronoi component has the
    closest node (Euclidean in pixel space).

    The *separated_graph* must already be split into per-room connected
    components (output of :func:`separate_rooms`).

    Mutates ``room_id`` and ``closest_vor_node`` on each object **in-place**
    and returns the same list for convenience.
    """
    if len(objects) == 0 or len(separated_graph) == 0:
        return objects

    components = list(nx.connected_components(separated_graph))
    # Pre-stack node positions per component  [(N_i, 2), …]
    comp_arrays = [np.array(list(comp)) for comp in components]

    obj_positions = np.array([o.position for o in objects])  # (M, 2)

    for obj, pos in zip(objects, obj_positions):
        best_dist = np.inf
        best_node: Optional[tuple] = None
        best_room: int = -1

        for room_id, nodes_arr in enumerate(comp_arrays):
            dists = np.linalg.norm(nodes_arr - pos, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_node = tuple(nodes_arr[idx])
                best_room = room_id

        obj.room_id = best_room
        obj.closest_vor_node = best_node

    return objects


# ── Hierarchical graph builder ──────────────────────────────────────────────

def build_room_object_graph(
    separated_graph: nx.Graph,
    objects: List[ObjectInfo],
) -> nx.DiGraph:
    """
    Build a MORE-compatible hierarchical ``DiGraph``:

        root  →  room_i  →  object_name

    Room centres are computed as the graph-theoretic centre of each
    component subgraph (same heuristic as MORE).

    Node attributes
    ---------------
    **root**:   ``node_type = NodeType.ROOT``
    **room_i**: ``node_type, room_id, pos_map (row,col),
                  frontier_points (set), closed_doors (set),
                  open_doors (set)``
    **object**: ``node_type, name, pos_map, bbox,
                  semantic_class, room_id, closest_vor_node``

    Args:
        separated_graph: Door-separated Voronoi graph (undirected).
        objects:         List of :class:`ObjectInfo` **already assigned**
                         (call :func:`assign_objects_to_rooms` first).

    Returns:
        ``nx.DiGraph`` with the structure above.
    """
    G = nx.DiGraph()
    G.add_node("root", node_type=NodeType.ROOT, pos_map=(0, 0))

    components = list(nx.connected_components(separated_graph))

    # ── Room nodes ──────────────────────────────────────────────────────
    for room_id, comp in enumerate(components):
        sub = separated_graph.subgraph(comp).copy()
        # Graph-theoretic centre (minimises eccentricity)
        centre_node = nx.center(sub)[0] if len(sub) > 0 else list(comp)[0]

        rname = NodeType.roomname(room_id)
        G.add_node(
            rname,
            node_type=NodeType.ROOM,
            room_id=room_id,
            pos_map=centre_node,
            frontier_points=set(),
            closed_doors=set(),
            open_doors=set(),
        )
        G.add_edge("root", rname)

    # ── Object nodes ────────────────────────────────────────────────────
    for obj in objects:
        if obj.room_id is None or obj.room_id < 0:
            continue
        rname = NodeType.roomname(obj.room_id)
        if rname not in G:
            continue

        G.add_node(
            obj.name,
            node_type=NodeType.OBJECT,
            name=obj.name,
            pos_map=tuple(obj.position),
            bbox=obj.bbox,
            semantic_class=obj.semantic_class,
            room_id=obj.room_id,
            closest_vor_node=obj.closest_vor_node,
        )
        G.add_edge(rname, obj.name)

    return G


# ── Query helpers ───────────────────────────────────────────────────────────

def get_room_nodes(G: nx.DiGraph) -> List[str]:
    """Return all room-node keys."""
    return [n for n, d in G.nodes(data=True) if d.get("node_type") == NodeType.ROOM]


def get_objects_in_room(G: nx.DiGraph, room_id: int) -> List[str]:
    """Return object-node keys belonging to a given room."""
    rname = NodeType.roomname(room_id)
    if rname not in G:
        return []
    return [
        n for n in G.successors(rname)
        if G.nodes[n].get("node_type") == NodeType.OBJECT
    ]


def get_object_node(G: nx.DiGraph, name: str) -> Optional[dict]:
    """Look up an object by name, return its attribute dict or None."""
    if name in G and G.nodes[name].get("node_type") == NodeType.OBJECT:
        return dict(G.nodes[name])
    return None