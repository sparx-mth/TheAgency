"""Ground-truth floor levels and stair connectors from a scene's navmesh.

The simulator's navmesh is the walkable surface of the building. Its
triangles at a storey's height are floor; the triangles in between are the
stairs that join two storeys. This module reads that geometry ONCE per scene
and hands the policy a short list of connectors -- where each staircase
starts, where it ends, which heights it joins and the walkable polyline
along it -- so that a floor transition is started only at a real staircase,
never at a raised bathroom floor that a depth sensor read as a first tread.

Everything is expressed in the public world frame the policy already
navigates in: right-handed Z-up ENU, ``(x, y, z) = (-h_z, -h_x, h_y)`` from
Habitat's +Y-up ``(h_x, h_y, h_z)`` -- the same conversion as
:func:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator.habitat_pose`.

This is scene structure, not task information: no goal, no distance to it
and no floor id of the target leave here. It is still ground truth, and a
method that consumes it says so in its configuration
(``ground_truth_stairs``); a run with it is not an observed-only run.

Host side only (numpy, scipy); ``habitat_sim`` is imported only by
:func:`scene_structure_from_pathfinder`'s caller.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import label

#: Vertical bin of the area histogram that finds the storeys, metres.
LEVEL_BIN_M = 0.10
#: The same generation thresholds as ``multifloor_generation.sampled_levels``.
MIN_LEVEL_AREA_M2 = 8.0
MIN_LEVEL_SEPARATION_M = 1.5
#: A sample farther than this from every storey height is on a connector.
OFF_LEVEL_M = 0.35
#: A connector must climb at least this much to join two storeys.
MIN_CONNECTOR_RISE_M = 1.0
#: XY cell used to cluster off-level samples into one staircase, metres.
CLUSTER_CELL_M = 0.5
#: Surface sample spacing on the triangles, metres.
SAMPLE_SPACING_M = 0.15

Vec3 = Sequence[float]


def habitat_to_enu(point: Vec3) -> List[float]:
    """Habitat ``(x, y_up, z)`` -> ENU ``(x, y, z_up)``."""
    return [-float(point[2]), -float(point[0]), float(point[1])]


def enu_to_habitat(point: Vec3) -> List[float]:
    """ENU ``(x, y, z_up)`` -> Habitat ``(x, y_up, z)``; inverse of :func:`habitat_to_enu`."""
    return [-float(point[1]), float(point[2]), -float(point[0])]


def triangles(vertices) -> np.ndarray:
    """``(N, 3, 3)`` triangles from the flat vertex list ``build_navmesh_vertices`` returns."""
    array = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if array.size == 0 or len(array) % 3:
        raise ValueError("navmesh vertices must come in triangles, got %d vertices" % len(array))
    return array.reshape(-1, 3, 3)


def surface_samples(tris: np.ndarray, spacing_m: float = SAMPLE_SPACING_M) -> np.ndarray:
    """Points spread over every triangle at roughly ``spacing_m``, Habitat frame."""
    out = []
    for a, b, c in tris:
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        n = max(2, int(math.ceil(math.sqrt(max(area, 1e-9)) / spacing_m * 2)))
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u, v = i[keep] / n, j[keep] / n
        out.append(a + np.outer(u, b - a) + np.outer(v, c - a))
    return np.concatenate(out) if out else np.zeros((0, 3))


def floor_levels(tris: np.ndarray, min_area_m2: float = MIN_LEVEL_AREA_M2,
                 min_separation_m: float = MIN_LEVEL_SEPARATION_M) -> List[Dict[str, float]]:
    """Storey heights, ascending: the area-supported modes of the navmesh height.

    Area-weighted histogram of triangle-centroid heights in :data:`LEVEL_BIN_M`
    bins; the largest bins win, at least ``min_separation_m`` apart and with
    at least ``min_area_m2`` of navmesh within +-0.30 m. The same rule the
    multi-story episode generator uses to pick its storeys, so the policy's
    levels and the evaluator's agree.
    """
    centroids = tris.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    heights = centroids[:, 1]
    bins = np.round(heights / LEVEL_BIN_M)
    unique, inverse = np.unique(bins, return_inverse=True)
    weights = np.bincount(inverse, weights=area)
    levels: List[Dict[str, float]] = []
    for index in np.argsort(-weights, kind="stable"):
        height = float(unique[index] * LEVEL_BIN_M)
        if any(abs(height - level["height_m"]) < min_separation_m for level in levels):
            continue
        near = np.abs(heights - height) <= 0.30
        supported = float(area[near].sum())
        if supported >= min_area_m2:
            median = float(np.average(heights[near], weights=area[near]))
            levels.append({"height_m": median, "area_m2": supported})
    return sorted(levels, key=lambda level: level["height_m"])


def _cluster_xy(points: np.ndarray, cell_m: float) -> np.ndarray:
    """8-connected cluster label per point over an XZ grid of ``cell_m`` cells."""
    ij = np.floor(points[:, [0, 2]] / cell_m).astype(int)
    ij -= ij.min(axis=0)
    grid = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, _ = label(grid, structure=np.ones((3, 3)))
    return labels[ij[:, 0], ij[:, 1]]


def _anchor(end: np.ndarray, level_height: float, floor: np.ndarray) -> np.ndarray:
    """The floor point just off one end of a staircase, at the storey's height.

    The mean of the on-level floor samples within 1 m (then 1.5 m) of the
    end's XZ centroid; the end centroid itself, lifted to the level, when
    the staircase meets the floor without any level surface around it.
    """
    centre = end[:, [0, 2]].mean(axis=0)
    on_level = floor[np.abs(floor[:, 1] - level_height) <= 0.15]
    for radius in (1.0, 1.5):
        if len(on_level):
            near = on_level[np.linalg.norm(on_level[:, [0, 2]] - centre, axis=1) <= radius]
            if len(near) >= 3:
                return near.mean(axis=0)
    return np.array([centre[0], level_height, centre[1]])


def _nearest_level(levels: Sequence[Dict[str, float]], height: float,
                   below: bool) -> Optional[Dict[str, float]]:
    """The storey nearest ``height`` on the requested side, or None."""
    if below:
        side = [level for level in levels if level["height_m"] <= height + 0.15]
    else:
        side = [level for level in levels if level["height_m"] >= height - 0.15]
    if not side:
        return None
    return min(side, key=lambda level: abs(level["height_m"] - height))


def stair_connectors(tris: np.ndarray, levels: Sequence[Dict[str, float]],
                     path_between: Optional[Callable[[Vec3, Vec3], Optional[Sequence[Vec3]]]] = None
                     ) -> List[Dict[str, object]]:
    """The staircases joining the storeys, in ENU.

    Args:
        tris: Navmesh triangles, Habitat frame.
        levels: :func:`floor_levels` of the same mesh.
        path_between: ``(start, end) -> points`` in the Habitat frame -- the
            navmesh shortest path, when a pathfinder is available -- or None
            for a height-ordered polyline through the cluster's own samples.

    Returns:
        One dict per connector: ``id``, ``bottom_xyz``, ``top_xyz`` (the
        floor anchors, ENU), ``bottom_z``/``top_z`` (the storey heights they
        join), ``polyline_xyz`` from bottom to top, ``length_m`` and
        ``rise_m``. Empty for a single-storey scene.
    """
    if len(levels) < 2:
        return []
    points = surface_samples(tris)
    heights = np.asarray([level["height_m"] for level in levels])
    off_level = np.min(np.abs(points[:, 1][:, None] - heights[None, :]), axis=1) > OFF_LEVEL_M
    stairs, floor = points[off_level], points[~off_level]
    if not len(stairs):
        return []
    clusters = _cluster_xy(stairs, CLUSTER_CELL_M)
    connectors: List[Dict[str, object]] = []
    for cluster in np.unique(clusters):
        member = stairs[clusters == cluster]
        low, high = float(member[:, 1].min()), float(member[:, 1].max())
        bottom_level = _nearest_level(levels, low, below=True)
        top_level = _nearest_level(levels, high, below=False)
        if bottom_level is None or top_level is None or bottom_level is top_level:
            continue
        if top_level["height_m"] - bottom_level["height_m"] < MIN_CONNECTOR_RISE_M:
            continue
        bottom = _anchor(member[member[:, 1] <= low + 0.25], bottom_level["height_m"], floor)
        top = _anchor(member[member[:, 1] >= high - 0.25], top_level["height_m"], floor)
        polyline = _polyline(bottom, top, member, path_between)
        enu = [habitat_to_enu(p) for p in polyline]
        length = sum(math.dist(a, b) for a, b in zip(enu, enu[1:]))
        connectors.append({
            "id": len(connectors),
            "bottom_xyz": habitat_to_enu(bottom), "top_xyz": habitat_to_enu(top),
            "bottom_z": float(bottom_level["height_m"]), "top_z": float(top_level["height_m"]),
            "polyline_xyz": enu, "length_m": float(length),
            "rise_m": float(top_level["height_m"] - bottom_level["height_m"])})
    connectors.sort(key=lambda c: (c["bottom_z"], c["bottom_xyz"][0], c["bottom_xyz"][1]))
    for index, connector in enumerate(connectors):
        connector["id"] = index
    return connectors


def _polyline(bottom: np.ndarray, top: np.ndarray, member: np.ndarray,
              path_between) -> List[List[float]]:
    """Bottom anchor -> top anchor along the stairs, Habitat frame."""
    if path_between is not None:
        found = path_between(bottom.tolist(), top.tolist())
        if found is not None and len(found) >= 2:
            return [[float(v) for v in p] for p in found]
    bins = np.round(member[:, 1] / 0.20)
    inner = [member[bins == b].mean(axis=0).tolist() for b in np.unique(bins)]
    return [bottom.tolist()] + inner + [top.tolist()]


def scene_structure(vertices, path_between=None) -> Dict[str, object]:
    """The metadata block for an episode: storeys and connectors, ENU."""
    tris = triangles(vertices)
    levels = floor_levels(tris)
    return {"stair_source": "navmesh",
            "floor_levels": levels,
            "stair_connectors": stair_connectors(tris, levels, path_between)}


def scene_structure_from_pathfinder(pathfinder) -> Dict[str, object]:
    """:func:`scene_structure` over a loaded ``habitat_sim.PathFinder``."""
    import habitat_sim

    def path_between(start, end):
        query = habitat_sim.ShortestPath()
        query.requested_start = np.asarray(start, dtype=np.float32)
        query.requested_end = np.asarray(end, dtype=np.float32)
        if not pathfinder.find_path(query) or not math.isfinite(query.geodesic_distance):
            return None
        return [np.asarray(p, dtype=float).tolist() for p in query.points]

    return scene_structure(pathfinder.build_navmesh_vertices(), path_between)


