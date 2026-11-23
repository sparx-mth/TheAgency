"""Frontier nodes: the places a picture is asked about -- a stretch of the map's boundary, its cells and a sticky id.

The search this package prepares for values **frontier clusters** from
pictures of them instead of rooms from the objects found in them. Three
things are needed of the unit a picture is asked about, and no existing type
has all three:

* **the cells of the boundary** -- which past frame looked at it is a
  question about cells. :class:`~sparx_agency.core.planning.exploration.room_costs.FrontierCluster`
  has them; a :class:`~sparx_agency.core.planning.exploration.frontier_ranking.FrontierGoal`
  and an :class:`~sparx_agency.core.planning.object_search.rooms.opening_nodes.Opening`
  are a point each.
* **an id that stays** -- a node is asked about once and its answer kept, so
  the node of this action must be recognisably the node of the last. The
  openings have that (:class:`~sparx_agency.core.planning.object_search.rooms.opening_nodes.OpeningRegistry`:
  a place within a metre of a known one keeps its id); clusters are
  extracted afresh every action and have none.
* **a point a planner accepts** -- RPT* needs ``id -> (x, y), probability``
  and nothing else (``exploration/room_costs.build_instance``). A cluster's
  goal cell is that point.

A frontier node is therefore built from the clusters of the frontier
inventory, the opening's way, without the opening's dependencies:

1. the clusters, longest first, are merged greedily within ``merge_m`` of a
   seed (:func:`~sparx_agency.core.planning.object_search.rooms.opening_nodes.merge_near_seed`,
   the rule the openings merge exits by): two boundaries a doorway apart
   open onto one unseen space, and two numbers for one space would count it
   twice in an order that treats nodes as independent;
2. a group under ``min_cells`` is no way anywhere and is dropped;
3. the seed's goal point asks the registry for the node's id.

**Why not the opening itself.** ``detect_openings`` reads the legacy loop's
state -- the multi-class landmarks whose shadows it discards, room numbers,
doors, peeks, the sweep's memory -- and runs only while that loop does: of
the six MP3D episodes this was first replayed on, three never listed an
opening, while their maps held six to thirteen reachable frontier clusters
an action on average. The pipeline this prepares for has neither rooms nor a
multi-class detector. What the opening got right -- merge, width floor,
sticky id -- is kept by calling the same code.

**Longest first, not best first.** The openings seed a group with the exit
of the highest utility, which depends on where the agent stands; a node's
seed should not move because the agent did. The longest cluster is the
seed, ties broken by cell.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.room_costs import FrontierCluster
from sparx_agency.core.planning.exploration.spatial_ids import SpatialIds, merge_near_seed


@dataclass(frozen=True)
class NodeSettings:
    """What a frontier node is. Distances in metres.

    Attributes:
        min_cells: Fewest frontier cells a node must hold, merged
            neighbours included: half a metre of boundary at the map's
            resolution, the openings' own floor -- a gap the body cannot
            pass is no way anywhere.
        merge_m: Clusters whose goal points lie within this distance of a
            seed are one node.
        match_m: A node whose seed lies within this distance of a known
            node's keeps that node's id. Must not exceed ``merge_m``: two
            seeds are more than ``merge_m`` apart, so no two nodes of one
            action can then claim one id.

    Raises:
        ValueError: On a value outside its range.
    """

    min_cells: int = 5
    merge_m: float = 1.5
    match_m: float = 1.0

    def __post_init__(self) -> None:
        if type(self.min_cells) is not int or self.min_cells < 1:
            raise ValueError("min_cells must be a positive integer, got %r" % (self.min_cells,))
        for name in ("merge_m", "match_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be positive and finite, got %r" % (name, value))
        if self.match_m > self.merge_m:
            raise ValueError("match_m (%r) cannot exceed merge_m (%r): two nodes of one action could claim one id"
                             % (self.match_m, self.merge_m))

    def registry(self) -> SpatialIds:
        """A fresh id registry for one episode, matching at ``match_m``."""
        return SpatialIds(match_m=self.match_m)


@dataclass(frozen=True, eq=False)
class FrontierNode:
    """One stretch of the map's boundary a picture is asked about.

    Compared by identity: two nodes of two actions with one id are the same
    place, not the same value.

    Attributes:
        node_id: Its sticky id, unique across the storeys of an episode.
        floor_id: The storey it lies on.
        xy: World metres of the seed cluster's goal cell: where a planner
            would send the agent to look into it.
        cell: That cell, ``(gx, gy)``.
        rows: Row index of every frontier cell of the node, seed cluster first.
        cols: Column index of every cell, same order.
        parts: How many clusters were merged into it.
        goal_cells: The constituent clusters' goal cells, so the geometric fallback can also withhold a blocked node.
    """

    node_id: int
    floor_id: int
    xy: Tuple[float, float]
    cell: Tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    parts: int
    goal_cells: Tuple[Tuple[int, int], ...] = ()

    @property
    def size_cells(self) -> int:
        return int(self.rows.size)


def frontier_nodes(world: OccupancyGrid2D, clusters: Sequence[FrontierCluster], registry: SpatialIds,
                   floor_id: int, settings: Optional[NodeSettings] = None) -> List[FrontierNode]:
    """The frontier nodes of one action, longest first, from the clusters of its frontier inventory.

    Args:
        world: The grid the clusters were extracted from (for the cells' world coordinates).
        clusters: The inventory's clusters (``FrontierInventory.clusters``):
            reachable, each with the goal cell the planner accepts.
        registry: The episode's id registry (:meth:`NodeSettings.registry`);
            it is told where each node now is.
        floor_id: The storey ``world`` is the map of.
        settings: What a node is.

    Returns:
        One :class:`FrontierNode` per merged group of at least ``min_cells`` cells.
    """
    settings = settings or NodeSettings()

    def point(cluster):
        x, y = world.grid_to_world(int(cluster.cell[0]), int(cluster.cell[1]))
        return float(x), float(y)

    ordered = sorted(clusters, key=lambda cluster: (-int(cluster.size), int(cluster.cell[1]), int(cluster.cell[0])))
    nodes = []
    for group in merge_near_seed(ordered, settings.merge_m, xy=point):
        rows = np.concatenate([np.asarray(cluster.rows, dtype=np.int64) for cluster in group])
        cols = np.concatenate([np.asarray(cluster.cols, dtype=np.int64) for cluster in group])
        if rows.size < settings.min_cells:
            continue
        seed = group[0]
        xy = point(seed)
        nodes.append(FrontierNode(node_id=int(registry.identify(int(floor_id), xy)), floor_id=int(floor_id), xy=xy,
                                  cell=(int(seed.cell[0]), int(seed.cell[1])), rows=rows, cols=cols, parts=len(group),
                                  goal_cells=tuple(cluster.cell for cluster in group)))
    return nodes
