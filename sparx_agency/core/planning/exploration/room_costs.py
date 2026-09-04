"""Arc weights between room centres, and the HPP-PT instance RPT* consumes.

Two failures this exists to prevent, both measured on the committed hospital
BEV fixture (413x200 @ 0.15 m, 29 watershed rooms):

* **A room centre that is not a place.** A room's centroid is the arithmetic
  mean of its mask cells, so on an L-shaped room it lands inside the room's own
  wall, and inside the planner's 0.4 m inflation skirt it is unreachable even
  when the floor beneath it is free. FIVE of the fixture's 29 centroids read as
  blocked. A search that plans to them silently never visits those rooms and
  reports nothing at all. :func:`room_nodes` snaps every centre onto a cell the
  planner will actually accept, and DROPS a room that has none rather than
  handing the solver a fabricated arc weight.
* **N^2 A\\* is too slow to run in a tick.** 29 rooms is 406 unordered pairs,
  and pairwise :class:`WeightedAStarPlanner2D` costs 6.6 s -- six ticks of a
  1 Hz replan loop, on the single-threaded ROS executor. One multi-source
  Dijkstra sweep over the same passable set answers all 841 entries in 111 ms
  (4 ms cost field + 6 ms graph build + 102 ms for 29 sources), leaving 89% of
  the tick free.

**The weight is BINARY, never the planner's shaped cost, and this is a
correctness requirement rather than a preference.** RPT*'s dominance pruning
consumes the triangle inequality directly, so a non-metric cost matrix makes
the solver discard optimal orders with no error anywhere. Sweeping the shaped
clearance cost is 1.6x inflated, is not in metres, and is not metric. Sweeping
unit step weights over the same passable cells gives a matrix whose worst
triangle-inequality violation over all 24389 triples of the fixture is exactly
0.000000000 m, and which is symmetric by construction because the graph is
built with forward neighbours only and traversed undirected.

The cost field is passed IN rather than rebuilt here, so the matrix and the
transit leg the aircraft actually flies share one passable set. They still
disagree by a few percent on any individual leg -- the flown A* centres itself
in corridors and chamfers the grid staircase, this sweep does neither -- so
treat an arc weight as a good estimate of the leg, not a promise about it.

numpy and scipy, no ROS. Deliberately NOT re-exported from this package's
``__init__.py``: the facade is imported inside the Noetic FALCON container,
which has neither.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple)

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.environment import OccupancyGrid2D

SQRT2 = float(np.sqrt(2.0))
"""Diagonal step, in cells. The graph is 8-connected."""

P_CLAMP_DEFAULT = 1.0 - 1e-6
"""RPT*'s heuristic divides by ``1 - p(v)``, so p must stay strictly below 1."""


@dataclass(frozen=True)
class RoomNode:
    """One room reduced to a single place the aircraft can be sent.

    Attributes:
        pid: The scene-graph persistent room id this node stands for.
        cell: ``(gx, gy)`` of a cell the planner accepts -- finite cost, so
            outside every wall and every inflation skirt.
        xy: That cell's CENTRE in world ENU metres. Not the raw centroid:
            when ``snapped`` is set they are different places, and this is
            the one that can be flown to.
        snapped: True when the room's raw centroid was not passable and this
            node is the nearest cell that is. Worth surfacing to an operator
            wondering why a goal sits off-centre in its room.
    """

    pid: int
    cell: Tuple[int, int]
    xy: Tuple[float, float]
    snapped: bool


@dataclass(frozen=True)
class HppPtInstance:
    """The Hamiltonian-Path-with-Probabilistic-Terminals instance for RPT*.

    Every array is over the SAME index space, ``0 .. N-1``, and
    :attr:`index_to_pid` is the only bridge back to scene-graph pids. A
    solver may treat this as opaque numeric data.

    Attributes:
        C: ``float64 [N, N]`` arc weights, ``C[i, j]`` = cost of going from
            i to j. Symmetric, metric, and finite everywhere -- rooms that
            could not reach the depot are removed before this is built, not
            left as infinities, because RPT* requires a complete graph.
        p: ``float64 [N]`` probability the target is at each vertex, each
            strictly inside ``[0, 1)``. NOT normalised to sum to 1 by
            default: normalising over only the rooms mapped so far asserts
            the target is certainly in one of them, which structurally
            suppresses flying toward unmapped space.
        depot: Index of the vertex the tour starts from -- the room the
            aircraft is in or nearest to, or an appended vertex at its
            actual pose. Free to be any index; nothing assumes 0.
        index_to_pid: ``index -> pid``. The depot's entry is its room's pid,
            or ``-1`` when the depot is the aircraft's own pose.
        nodes: The :class:`RoomNode` behind each index, same order.
        units: ``'metres'`` or ``'seconds'``. Seconds once a cruise speed
            and a per-room search budget have been folded in.
        build_ms: Wall time this instance took to assemble, so a node can
            watch the cost of the sweep grow as the map fills.
    """

    C: np.ndarray
    p: np.ndarray
    depot: int
    index_to_pid: Tuple[int, ...]
    nodes: Tuple[RoomNode, ...]
    units: str
    build_ms: float

    @property
    def n(self) -> int:
        """Number of vertices."""
        return int(self.C.shape[0])


# -- the passable graph ---------------------------------------------------
def passable_graph(cost: np.ndarray) -> Tuple[np.ndarray, Any]:
    """Compact the planner's passable cells into a unit-weight 8-graph.

    Args:
        cost: ``(H, W)`` float cost array from
            :meth:`WeightedAStarPlanner2D.cost_for`, ``inf`` where blocked.

    Returns:
        ``(ids, graph)`` where ``ids`` is ``int64 (H, W)`` carrying each
        passable cell's node index and ``-1`` off-graph, and ``graph`` is a
        CSR matrix over ``ids.max() + 1`` nodes holding only FORWARD edges.
        Traverse it with ``directed=False``: that supplies the reverse arcs
        and is what makes the resulting matrix symmetric by construction.

    Only the passable cells become nodes (58053 of the fixture's 82600),
    which is what keeps the build at ~6 ms rather than ~9 ms.
    """
    passable = np.isfinite(cost)
    h, w = cost.shape
    ids = np.full((h, w), -1, dtype=np.int64)
    n = int(passable.sum())
    if n == 0:
        return ids, coo_matrix((0, 0), dtype=np.float64).tocsr()
    ids[passable] = np.arange(n, dtype=np.int64)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    wts: List[np.ndarray] = []
    # Forward neighbours only. (0,-1) and (-1,*) are the same edges reversed.
    for dy, dx, step in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
        ay0, ay1 = max(0, -dy), h - max(0, dy)
        ax0, ax1 = max(0, -dx), w - max(0, dx)
        by0, by1 = max(0, dy), h + min(0, dy) if dy else h
        bx0, bx1 = max(0, dx), w + min(0, dx) if dx else w
        a_ids = ids[ay0:ay1, ax0:ax1]
        b_ids = ids[by0:by1, bx0:bx1]
        both = (a_ids >= 0) & (b_ids >= 0)
        if not both.any():
            continue
        rows.append(a_ids[both])
        cols.append(b_ids[both])
        wts.append(np.full(int(both.sum()), step, dtype=np.float64))

    if not rows:
        return ids, coo_matrix((n, n), dtype=np.float64).tocsr()
    graph = coo_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n), dtype=np.float64).tocsr()
    return ids, graph


def snap_cell(ids: np.ndarray, gx: int, gy: int,
              max_radius: int) -> Optional[Tuple[int, int]]:
    """The nearest ON-GRAPH cell to ``(gx, gy)``, or None within the radius.

    Returns the cell itself when it is already on the graph. Otherwise
    searches expanding square windows and returns the EUCLIDEAN-nearest
    passable cell in the first window that contains one -- nearest by true
    distance rather than by ring order, so a diagonal neighbour never beats
    an orthogonal one that is genuinely closer.

    Args:
        ids: The ``(H, W)`` index image from :func:`passable_graph`.
        gx: Cell column.
        gy: Cell row.
        max_radius: Largest window half-width to try, in cells.

    Returns:
        ``(gx, gy)`` of a passable cell, or None.
    """
    h, w = ids.shape
    if 0 <= gx < w and 0 <= gy < h and ids[gy, gx] >= 0:
        return int(gx), int(gy)
    for r in range(1, int(max_radius) + 1):
        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        window = ids[y0:y1, x0:x1]
        ys, xs = np.nonzero(window >= 0)
        if ys.size == 0:
            continue
        cy = ys + y0
        cx = xs + x0
        d2 = (cx - gx) ** 2 + (cy - gy) ** 2
        k = int(np.argmin(d2))
        return int(cx[k]), int(cy[k])
    return None


def room_nodes(world: OccupancyGrid2D,
               cost: np.ndarray,
               centroids: Mapping[int, Tuple[float, float]],
               snap_radius_m: float = 2.0,
               ids: Optional[np.ndarray] = None
               ) -> Tuple[List[RoomNode], List[int]]:
    """Turn ``{pid: centroid_xy}`` into flyable nodes, dropping the unflyable.

    Args:
        world: The grid the centroids are expressed against.
        cost: The planner's cost array for ``world``.
        centroids: ``{pid: (x, y)}`` in world ENU metres.
        snap_radius_m: How far a blocked centroid may be moved.
        ids: The index image from :func:`passable_graph`, when the caller
            already has one. Recomputed if omitted.

    Returns:
        ``(nodes, dropped_pids)``, nodes in ascending pid order so the index
        space is deterministic across ticks for an unchanged room set.
    """
    if ids is None:
        ids, _ = passable_graph(cost)
    radius = max(1, int(round(float(snap_radius_m) / world.resolution)))
    nodes: List[RoomNode] = []
    dropped: List[int] = []
    for pid in sorted(int(p) for p in centroids):
        wx, wy = centroids[pid]
        gx, gy = world.world_to_grid(float(wx), float(wy))
        cell = snap_cell(ids, gx, gy, radius)
        if cell is None:
            dropped.append(int(pid))
            continue
        snapped = (cell[0], cell[1]) != (gx, gy)
        x, y = world.grid_to_world(cell[0], cell[1])
        nodes.append(RoomNode(pid=int(pid), cell=(int(cell[0]), int(cell[1])),
                              xy=(float(x), float(y)), snapped=bool(snapped)))
    return nodes, dropped


def cost_matrix(world: OccupancyGrid2D,
                graph: Any,
                ids: np.ndarray,
                cells: Sequence[Tuple[int, int]]) -> np.ndarray:
    """All-pairs shortest path length in METRES between the given cells.

    One multi-source Dijkstra sweep, then the columns at the sources. Entries
    for a pair with no path through free space are ``inf``.

    Args:
        world: The grid, for its resolution.
        graph: The CSR graph from :func:`passable_graph`.
        ids: The matching index image.
        cells: ``(gx, gy)`` per vertex, every one already on the graph.

    Returns:
        ``float64 [N, N]``, symmetric, zero on the diagonal.
    """
    n = len(cells)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    src = np.array([ids[gy, gx] for gx, gy in cells], dtype=np.int64)
    dist = dijkstra(graph, directed=False, indices=src)
    out = np.asarray(dist[:, src], dtype=np.float64) * float(world.resolution)
    # The sweep is exact and undirected, so any asymmetry here is float noise
    # from the two orderings; averaging keeps the matrix exactly symmetric,
    # which RPT* checks for.
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


def build_instance(world: OccupancyGrid2D,
                   cost: np.ndarray,
                   centroids: Mapping[int, Tuple[float, float]],
                   probs: Mapping[int, float],
                   depot_xy: Optional[Tuple[float, float]] = None,
                   snap_radius_m: float = 2.0,
                   cruise_speed_mps: float = 0.30,
                   search_time_s: float = 0.0,
                   frontier_weight: float = 0.0,
                   frontier_counts: Optional[Mapping[int, int]] = None,
                   p_clamp: float = P_CLAMP_DEFAULT
                   ) -> Tuple[HppPtInstance, List[int]]:
    """Assemble the complete, finite, metric instance RPT* takes.

    Args:
        world: The BEV grid, wrapped with the BEV's own occupancy values.
        cost: ``WeightedAStarPlanner2D.cost_for(world)[0]`` -- passed in so
            the matrix and the flown transit share one passable set.
        centroids: ``{pid: (x, y)}`` room centres from the scene graph.
        probs: ``{pid: probability}`` from the LLM oracle. A pid absent here
            scores 0.0 rather than being dropped -- an unranked room is a
            room worth visiting last, not one that does not exist.
        depot_xy: The aircraft's pose. When given it becomes its OWN vertex,
            appended last with ``p = 0`` and pid ``-1``, so the first leg is
            costed from where the aircraft actually is rather than from the
            centre of whichever room it happens to be standing in. When
            omitted the depot is the room nearest the first centroid.
        snap_radius_m: How far a blocked centre may be moved.
        cruise_speed_mps: Divides metres into seconds. Ignored when it is
            not positive, which leaves ``units='metres'``.
        search_time_s: The per-room mapping budget, folded into every arc
            ENTERING a room. HPP-PT has no per-vertex service time, but our
            loop spends this at every vertex and in a hospital it dominates
            inter-room transit -- so optimising the paper's objective
            verbatim would optimise the wrong quantity. Folding it into
            entering arcs provably preserves the triangle inequality
            (``c(u,w) + T_w <= c(u,v) + T_v + c(v,w) + T_w`` for ``T_v >=
            0``) and makes the solver charge it under the correct
            not-found-yet discount.
        frontier_weight: Blend weight in ``[0, 1]`` mixing an unexplored-space
            term into p. 0 uses the oracle alone.
        frontier_counts: ``{pid: frontier_clusters}`` for that blend.
        p_clamp: Ceiling applied to every probability. Mandatory: RPT*'s
            heuristic divides by ``1 - p(v)``, and a scene graph with one
            room hands a sum-to-1 oracle vector straight to a division by
            zero.

    Returns:
        ``(instance, dropped_pids)``. Dropped covers both rooms with no
        passable cell within ``snap_radius_m`` and rooms with no path from
        the depot -- RPT* requires a complete finite graph, so an
        unreachable room is withheld this tick rather than given a
        fabricated weight. It comes back as soon as the map connects it.

    Raises:
        ValueError: If ``centroids`` is empty. There is no instance over no
            rooms, and returning a degenerate one would hide the real
            failure (the mapper has not produced a room yet).
    """
    if not centroids:
        raise ValueError("build_instance needs at least one room centroid")
    t0 = time.perf_counter()

    ids, graph = passable_graph(cost)
    nodes, dropped = room_nodes(world, cost, centroids, snap_radius_m, ids=ids)

    depot_node: Optional[RoomNode] = None
    if depot_xy is not None:
        gx, gy = world.world_to_grid(float(depot_xy[0]), float(depot_xy[1]))
        cell = snap_cell(ids, gx, gy, max(1, int(round(snap_radius_m / world.resolution))))
        if cell is not None:
            x, y = world.grid_to_world(cell[0], cell[1])
            depot_node = RoomNode(pid=-1, cell=(int(cell[0]), int(cell[1])),
                                  xy=(float(x), float(y)),
                                  snapped=(cell[0], cell[1]) != (gx, gy))

    all_nodes = list(nodes) + ([depot_node] if depot_node is not None else [])
    if not all_nodes:
        raise ValueError("no room centroid was reachable on this grid")
    depot = len(all_nodes) - 1 if depot_node is not None else 0

    full = cost_matrix(world, graph, ids, [nd.cell for nd in all_nodes])

    # RPT* needs a complete finite graph. Keep the depot and everything it
    # can reach; withhold the rest and say which.
    reach = np.isfinite(full[depot])
    reach[depot] = True
    keep = [i for i in range(len(all_nodes)) if reach[i]]
    # A pair that is finite from the depot but infinite to each other cannot
    # happen on an undirected graph (reachability is an equivalence class),
    # but assert it rather than trusting it: a NaN here is a silent wrong tour.
    sub = full[np.ix_(keep, keep)]
    if not np.isfinite(sub).all():
        finite_rows = np.isfinite(sub).all(axis=1)
        keep = [keep[i] for i in range(len(keep)) if finite_rows[i]]
        sub = full[np.ix_(keep, keep)]

    dropped = dropped + [all_nodes[i].pid for i in range(len(all_nodes))
                         if i not in keep and all_nodes[i].pid >= 0]
    kept_nodes = [all_nodes[i] for i in keep]
    depot_idx = keep.index(depot)

    units = "metres"
    C = np.array(sub, dtype=np.float64, copy=True)
    if cruise_speed_mps and cruise_speed_mps > 0.0:
        C = C / float(cruise_speed_mps)
        units = "seconds"
        if search_time_s and search_time_s > 0.0:
            budget = np.full(C.shape[1], float(search_time_s))
            budget[depot_idx] = 0.0
            C = C + budget[None, :]
            np.fill_diagonal(C, 0.0)

    p = np.zeros(len(kept_nodes), dtype=np.float64)
    for i, nd in enumerate(kept_nodes):
        if nd.pid < 0:
            continue
        base = float(probs.get(nd.pid, 0.0))
        if frontier_weight > 0.0 and frontier_counts is not None:
            fr = min(1.0, float(frontier_counts.get(nd.pid, 0)) / 4.0)
            base = (1.0 - frontier_weight) * base + frontier_weight * fr
        p[i] = base
    np.clip(p, 0.0, float(p_clamp), out=p)

    inst = HppPtInstance(
        C=C,
        p=p,
        depot=int(depot_idx),
        index_to_pid=tuple(int(nd.pid) for nd in kept_nodes),
        nodes=tuple(kept_nodes),
        units=units,
        build_ms=float((time.perf_counter() - t0) * 1e3),
    )
    return inst, sorted(set(dropped))


# -- in-room sweep goals --------------------------------------------------
def in_room_frontier_goals(world: OccupancyGrid2D,
                           cost: np.ndarray,
                           room_mask: np.ndarray,
                           min_cluster_cells: int = 4,
                           ids: Optional[np.ndarray] = None
                           ) -> List[Tuple[float, float]]:
    """Where to look next INSIDE one room, largest unscanned region first.

    A frontier cell is a free cell 4-adjacent to an unknown one -- the same
    definition ``room_stats.count_frontier_clusters`` uses, so the goals this
    returns and the ``frontier_clusters`` the scene graph reports are the
    same population, and a room whose count reaches zero has genuinely had
    every goal here consumed.

    Computed with four array shifts rather than
    ``core/planning/exploration/frontier.py``, whose pure-Python double loop
    over 82600 cells is far too slow to run in a control tick.

    Args:
        world: The BEV grid.
        cost: The planner's cost array, used only to place each goal on a
            cell the planner accepts.
        room_mask: ``(H, W)`` bool, True inside the room. The mask is the
            only thing keeping the aircraft in the room: every goal comes
            from inside it, so the sweep cannot walk out through a door.
        min_cluster_cells: Clusters smaller than this are noise and dropped.
        ids: The index image from :func:`passable_graph`, if already built.

    Returns:
        World ``(x, y)`` goals, largest cluster first, each on a passable
        cell. Empty when the room has no unscanned boundary left.
    """
    if ids is None:
        ids, _ = passable_graph(cost)
    grid = world.grid
    free = grid == world.values.free
    unknown = grid == world.values.unknown
    adj = np.zeros_like(unknown)
    adj[:-1, :] |= unknown[1:, :]
    adj[1:, :] |= unknown[:-1, :]
    adj[:, :-1] |= unknown[:, 1:]
    adj[:, 1:] |= unknown[:, :-1]
    frontier = free & adj & np.asarray(room_mask, dtype=bool)
    if not frontier.any():
        return []

    lbl, n = cc_label(frontier, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return []
    out: List[Tuple[int, Tuple[float, float]]] = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lbl == k)
        if ys.size < int(min_cluster_cells):
            continue
        gx = int(round(float(xs.mean())))
        gy = int(round(float(ys.mean())))
        cell = snap_cell(ids, gx, gy, 8)
        if cell is None:
            continue
        x, y = world.grid_to_world(cell[0], cell[1])
        out.append((int(ys.size), (float(x), float(y))))
    out.sort(key=lambda t: -t[0])
    return [xy for _, xy in out]
