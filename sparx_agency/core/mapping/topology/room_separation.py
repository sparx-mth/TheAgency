# core/mapping/topology/room_separation.py
"""
Room separation via door-probability edge cutting.

Implements the room-detection pipeline from Werby et al. (MORE, 2025):
  door positions + orientations → Gaussian probability field
  → boundary integral along graph edges → cut high-scoring edges
  → connected components ≈ rooms

Dependencies: numpy, scipy, networkx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal


@dataclass(frozen=True)
class DoorInfo:
    """
    Single door descriptor in **map (pixel)** coordinates.

    Attributes:
        position:   (row, col) center of the door in the occupancy grid.
        size:       (extent_row, extent_col) bounding-box half-extents in meters.
        rotation:   2×2 rotation matrix (top-left block of the 3D orientation).
                    Identity if the door is axis-aligned.
    """
    position: np.ndarray          # shape (2,)
    size: np.ndarray              # shape (2,)  – meters
    rotation: np.ndarray = field(  # shape (2, 2)
        default_factory=lambda: np.eye(2)
    )


@dataclass(frozen=True)
class RoomSeparationParams:
    """
    Parameters for door-based room separation.

    Attributes:
        edge_score_threshold: Edges whose boundary integral exceeds this
                              value are cut.  Tune ↑ to merge more rooms,
                              ↓ to split more aggressively.
        normalize_integral:   Divide integral by path length (makes the
                              threshold length-independent).
        min_component:        Drop components smaller than this after cutting.
    """
    edge_score_threshold: float = 0.1
    normalize_integral: bool = False
    min_component: int = 4


# ── Gaussian door-probability field ─────────────────────────────────────────

def compute_door_probability_field(
    grid_shape: Tuple[int, int],
    doors: List[DoorInfo],
    resolution: float,
) -> np.ndarray:
    """
    Build a 2-D probability density over the grid, peaked at each door.

    Each door contributes a Gaussian whose covariance is derived from
    the door's bounding-box size and orientation (following MORE).

    Args:
        grid_shape: (H, W) of the occupancy grid.
        doors: List of DoorInfo descriptors (pixel coords).
        resolution: Meters per cell (used to convert door sizes to pixels).

    Returns:
        field: (H, W) float64 – unnormalized density (sum of per-door PDFs).
    """
    H, W = grid_shape
    if len(doors) == 0:
        return np.zeros((H, W), dtype=np.float64)

    # Evaluate all doors on a shared meshgrid
    rr, cc = np.mgrid[0:H, 0:W]
    points = np.column_stack([rr.ravel(), cc.ravel()])  # (H*W, 2)

    density = np.zeros(H * W, dtype=np.float64)
    for door in doors:
        cov_local = np.diag(door.size / resolution)          # pixels
        cov = door.rotation @ cov_local @ door.rotation.T    # rotated
        # Regularize: ensure positive-definite
        cov += np.eye(2) * 1e-6
        density += multivariate_normal.pdf(points, mean=door.position, cov=cov)

    return density.reshape(H, W)


# ── Boundary integral along an edge ────────────────────────────────────────

def _boundary_integral(
    field: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    normalize: bool,
) -> float:
    """
    Integrate *field* along the straight segment p1→p2 (Bresenham-style).

    Args:
        field: (H, W) scalar field.
        p1, p2: (row, col) endpoints.
        normalize: If True, divide by segment length.

    Returns:
        Scalar integral value.
    """
    n_samples = max(int(np.hypot(p1[0] - p2[0], p1[1] - p2[1])), 1)
    ts = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    rows = (p1[0] + ts * (p2[0] - p1[0])).astype(int)
    cols = (p1[1] + ts * (p2[1] - p1[1])).astype(int)

    np.clip(rows, 0, field.shape[0] - 1, out=rows)
    np.clip(cols, 0, field.shape[1] - 1, out=cols)

    total = float(field[rows, cols].sum())
    if normalize and n_samples > 0:
        total /= n_samples
    return total


# ── Main entry point ────────────────────────────────────────────────────────

def separate_rooms(
    G: nx.Graph,
    grid_shape: Tuple[int, int],
    doors: List[DoorInfo],
    resolution: float,
    params: RoomSeparationParams = RoomSeparationParams(),
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Separate a Voronoi topology graph into per-room components by cutting
    edges that pass through detected doors.

    Args:
        G: Voronoi navigation graph (nodes keyed by (row, col)).
        grid_shape: (H, W) of the underlying occupancy grid.
        doors: Door descriptors in pixel coordinates.
        resolution: Meters per cell.
        params: Tuning knobs.

    Returns:
        separated: A **copy** of *G* with door-crossing edges removed
                   and tiny components pruned.
        door_field: (H, W) door probability field (useful for visualization).
    """
    door_field = compute_door_probability_field(grid_shape, doors, resolution)

    separated = G.copy()

    # Score every edge and collect those above threshold
    edges_to_cut: list = []
    for u, v in separated.edges():
        score = _boundary_integral(door_field, u, v, params.normalize_integral)
        if score > params.edge_score_threshold:
            edges_to_cut.append((u, v))

    separated.remove_edges_from(edges_to_cut)
    separated.remove_nodes_from(list(nx.isolates(separated)))

    # Drop tiny components
    if params.min_component > 1:
        for comp in list(nx.connected_components(separated)):
            if len(comp) < params.min_component:
                separated.remove_nodes_from(comp)

    return separated, door_field