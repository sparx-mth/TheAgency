# core/mapping/topology/graph_utils.py
"""
Graph sparsification and query utilities for Voronoi topology graphs.
"""

from __future__ import annotations

from typing import List

import networkx as nx


def sparsify_graph(G: nx.Graph, resolution: float, max_chain_m: float) -> None:
    """
    Remove degree-2 pass-through nodes in-place when the total chain
    length (sum of both edges) is below *max_chain_m*.

    Args:
        G: Graph to sparsify (modified in-place).
        resolution: Meters per cell (used to convert threshold).
        max_chain_m: Maximum chain length in meters to collapse.
    """
    threshold_cells = max_chain_m / resolution
    deg2 = [n for n in G.nodes if G.degree(n) == 2]

    for node in deg2:
        if node not in G or G.degree(node) != 2:
            continue
        a, b = list(G.neighbors(node))
        w = G[node][a]["dist"] + G[node][b]["dist"]
        if w <= threshold_cells:
            G.add_edge(a, b, dist=w)
            G.remove_node(node)


def get_junctions(G: nx.Graph) -> List[tuple]:
    """Return nodes with degree >= 3 (topological waypoints / intersections)."""
    return [n for n in G.nodes if G.degree(n) >= 3]


def get_dead_ends(G: nx.Graph) -> List[tuple]:
    """Return nodes with degree == 1 (corridor endpoints / frontiers)."""
    return [n for n in G.nodes if G.degree(n) == 1]