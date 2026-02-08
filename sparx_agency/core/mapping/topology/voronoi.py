# core/mapping/topology/voronoi.py
"""
Voronoi-based topology extraction from 2D occupancy grids.

Implements the pipeline from Werby et al. (MORE, 2025):
  occupancy → boundary cost field → Voronoi skeleton → sparse navigation graph

Dependencies: numpy, scipy, networkx (all already in use).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial import Voronoi

from sparx_agency.core.mapping.costmap.sdf import boundary_cost_field
from sparx_agency.core.mapping.topology.graph_utils import sparsify_graph


@dataclass(frozen=True, slots=True)
class TopologyParams:
    """
    Parameters for Voronoi topology extraction.

    Attributes:
        sdf_scale_m:      Obstacle influence radius for boundary cost (meters).
        inner_inflate_m:  Small extra inflation to avoid edges crossing thin walls.
        sparsify_dist_m:  Max chain length (meters) to collapse degree-2 nodes.
                          Set to 0 to skip sparsification.
        min_component:    Drop connected components smaller than this.
    """
    sdf_scale_m: float = 3.0
    inner_inflate_m: float = 0.33
    sparsify_dist_m: float = 0.6
    min_component: int = 4


def extract_voronoi_graph(
    occupancy: np.ndarray,
    resolution: float,
    params: TopologyParams = TopologyParams(),
) -> nx.Graph:
    """
    Full pipeline: occupancy → sparse Voronoi navigation graph.

    Args:
        occupancy: (H, W) binary grid. Nonzero = occupied.
        resolution: Meters per cell.
        params: Tuning knobs.

    Returns:
        nx.Graph with:
          - nodes keyed by (row, col) float tuples
          - edges carry ``dist`` attribute (Euclidean distance in cells)
    """
    H, W = occupancy.shape

    # 1. Boundary cost field
    cost = boundary_cost_field(occupancy, resolution, params.sdf_scale_m)
    cost[occupancy == 0] = 0.0
    if params.inner_inflate_m > 0:
        np.maximum(cost, boundary_cost_field(occupancy, resolution, params.inner_inflate_m), out=cost)

    # 2. Voronoi on obstacle / boundary points
    obs_pts = np.argwhere(cost > 0).astype(np.float32)
    if len(obs_pts) < 4:
        return nx.Graph()

    vor = Voronoi(obs_pts)

    # 3. Filter vertices to free space (all 4 surrounding cells must be free)
    verts = vor.vertices
    in_bounds = (
        (verts[:, 0] >= 0) & (verts[:, 0] < H - 1) &
        (verts[:, 1] >= 0) & (verts[:, 1] < W - 1)
    )
    r_lo = np.floor(verts[:, 0]).astype(int)
    r_hi = np.ceil(verts[:, 0]).astype(int)
    c_lo = np.floor(verts[:, 1]).astype(int)
    c_hi = np.ceil(verts[:, 1]).astype(int)

    blocked = cost.astype(bool)
    free_count = np.zeros(len(verts), dtype=int)
    for r, c in [(r_lo, c_lo), (r_lo, c_hi), (r_hi, c_lo), (r_hi, c_hi)]:
        free_count[in_bounds] += ~blocked[
            np.clip(r[in_bounds], 0, H - 1),
            np.clip(c[in_bounds], 0, W - 1),
        ]
    valid_mask = in_bounds & (free_count == 4)

    # Remap vertex indices
    vert_remap = np.full(len(verts), -1, dtype=int)
    valid_indices = np.nonzero(valid_mask)[0]
    vert_remap[valid_indices] = np.arange(len(valid_indices))
    valid_verts = np.round(verts[valid_indices], 3)

    # 4. Build graph (vectorized edge creation)
    ridges = np.asarray(vor.ridge_vertices)
    finite = np.all(ridges >= 0, axis=1)
    r0 = vert_remap[ridges[finite, 0]]
    r1 = vert_remap[ridges[finite, 1]]
    both_valid = (r0 >= 0) & (r1 >= 0)
    edges_idx = np.stack([r0[both_valid], r1[both_valid]], axis=1)

    diffs = valid_verts[edges_idx[:, 0]] - valid_verts[edges_idx[:, 1]]
    dists = np.linalg.norm(diffs, axis=1)

    # Midpoint obstacle check
    mids = (valid_verts[edges_idx[:, 0]] + valid_verts[edges_idx[:, 1]]) * 0.5
    edge_free = ~blocked[
        np.clip(mids[:, 0].astype(int), 0, H - 1),
        np.clip(mids[:, 1].astype(int), 0, W - 1),
    ]

    G = nx.Graph()
    for v in valid_verts:
        G.add_node(tuple(v))

    node_list = list(G.nodes)
    for idx in np.nonzero(edge_free)[0]:
        i, j = edges_idx[idx]
        G.add_edge(node_list[i], node_list[j], dist=float(dists[idx]))

    # Final safety: remove nodes on walls
    G.remove_nodes_from([n for n in list(G.nodes) if occupancy[int(n[0]), int(n[1])] != 0])

    # 5. Drop tiny components
    if params.min_component > 1:
        for comp in [c for c in nx.connected_components(G) if len(c) < params.min_component]:
            G.remove_nodes_from(comp)

    # 6. Sparsify
    if params.sparsify_dist_m > 0 and len(G) > 0:
        sparsify_graph(G, resolution, params.sparsify_dist_m)

    return G