"""Ground-truth floor levels and stair connectors from a scene's navmesh.

The simulator's navmesh is the walkable surface of the building. Its
triangles at a storey's height are floor; the triangles in between are the
stairs that join two storeys. This module reads that geometry ONCE per scene
and hands the evaluator a short list of connectors -- where each staircase
starts, where it ends, which two ADJACENT storeys it joins, the centreline
to walk it by and a sample of its surface -- so that the policy's perfect
stair detector has something to look for, and a committed climb has a route
that keeps to the middle of the flight.

Four rules, each bought by a recording:

* **A connector joins adjacent storeys.** The XY clustering puts a whole
  stairwell into one cluster; a three-storey house (Pomaria) then read as
  one 5.4 m connector from the basement straight to the top, passing
  THROUGH the middle storey, which no connector touched. A cluster is split
  at every storey height it crosses.
* **The polyline is the centreline, not the shortest path.** A navmesh
  shortest path is string-pulled: it touches every corner it rounds and runs
  along the inner wall of every flight, and a discrete agent following it
  grinds on that wall at the landing (Ranchester, twice). The route is one
  vertex per height band -- the centroid of the flight's cross-section, so
  the middle of a flight and the middle of a landing -- and the storey
  anchors lie on the flight's own line, set out onto the floor.
* **Every leg is walked before it is handed out.** When a pathfinder is
  available each leg is stepped with the simulator's own collision model
  (``try_step``) at the agent's stride; a leg the agent could not walk is
  repaired through the pathfinder's path between its ends.
* **The stairs are a surface, not a line.** A sample of the stair surface
  (ENU) rides with each connector so the policy's detector can test, frame
  by frame, whether THIS staircase is in view -- knowledge of the stairs
  comes from seeing them, never from this file.

Everything is expressed in the public world frame the policy already
navigates in: right-handed Z-up ENU, ``(x, y, z) = (-h_z, -h_x, h_y)`` from
Habitat's +Y-up ``(h_x, h_y, h_z)`` -- the same conversion as
:func:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator.habitat_pose`.

This is scene structure, not task information: no goal, no distance to it
and no floor id of the target leave here. It is still ground truth, and a
method that consumes it says so in its configuration
(``ground_truth_stairs``); a run with it is not an observed-only run.

Host side only (numpy, scipy); ``habitat_sim`` is imported only by
:func:`pathfinder_hooks`.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
#: Height band of the centreline: one polyline vertex per band, metres.
BAND_M = 0.20
#: XZ cell that joins a band's samples into one component; a gap wider than this is another run.
BAND_COMPONENT_CELL_M = 0.35
#: Components in adjacent bands whose samples come within this of each other touch: one flight.
BAND_GAP_M = 0.5
#: The least of a storey interval's climbable height a run of samples must span to be its staircase.
MIN_SPAN_FRACTION = 0.6
#: The fewest surface samples a run needs to be a staircase at all.
MIN_RUN_SAMPLES = 8
#: How far past the foot (head) of the flight the storey anchor is set out onto the floor, metres.
ANCHOR_OUT_M = 0.35
#: Band vertices at a flight's end whose principal axis gives the direction the anchor is set out along.
ANCHOR_BANDS = 4
#: The most on-level floor samples tried as an anchor after the flight's own line.
ANCHOR_CANDIDATES = 40
#: How far off the flight the exit point is looked for, farthest first, metres.
EXIT_REACH_M = (1.2, 1.0, 0.8)
#: The swings either side of the flight's line the exit point may take, degrees; never behind.
EXIT_SWING_DEG = (0, 30, -30, 60, -60, 90, -90)
#: An exit point stands on the storey within this much: flat floor, not the eased last tread.
EXIT_FLAT_M = 0.08
#: Horizontal spacing the polyline is densified to for the walkability check -- one agent stride.
STEP_CHECK_M = 0.25
#: A checked stride that ends farther than this from where it aimed did not get there.
STEP_MISS_M = 0.05
#: Legs shorter than this are merged into their neighbour when the merged leg is walkable.
MIN_LEG_M = 0.30
#: A vertex the navmesh snap moves farther than this was off the walkable surface.
SNAP_MOVE_M = 0.10
#: Spacing of the surface sample kept per connector for the visibility test, metres.
SURFACE_SPACING_M = 0.25
MAX_SURFACE_POINTS = 400

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


def _horizontal(a, b) -> float:
    return math.hypot(float(b[0] - a[0]), float(b[2] - a[2]))


def _interval_runs(member: np.ndarray, levels: Sequence[Dict[str, float]]):
    """The runs of one cluster's off-level samples between ADJACENT storeys: ``(lower, upper, samples)``.

    A stairwell is one XY cluster however many storeys it serves; each pair
    of adjacent storeys it joins is one connector. A run must climb most of
    the interval between its two storeys (:data:`MIN_SPAN_FRACTION` of the
    height that lies outside the storeys' own bands) -- a raised platform
    half a metre up is not a staircase to the floor above.
    """
    for lower, upper in zip(levels, levels[1:]):
        lo, hi = float(lower["height_m"]), float(upper["height_m"])
        if hi - lo < MIN_CONNECTOR_RISE_M:
            continue
        part = member[(member[:, 1] > lo) & (member[:, 1] < hi)]
        if len(part) < MIN_RUN_SAMPLES:
            continue
        climbable = (hi - lo) - 2.0 * OFF_LEVEL_M
        span = float(part[:, 1].max() - part[:, 1].min())
        if span < MIN_SPAN_FRACTION * climbable:
            continue
        yield lower, upper, part


def _band_components(part: np.ndarray, lo: float) -> List[List[np.ndarray]]:
    """Per height band, ascending, the XZ-connected components of that band's samples."""
    bands = np.floor((part[:, 1] - lo) / BAND_M).astype(int)
    out: List[List[np.ndarray]] = []
    for band in np.unique(bands):
        samples = part[bands == band]
        labels = _cluster_xy(samples, BAND_COMPONENT_CELL_M) if len(samples) > 1 else np.zeros(len(samples), int)
        out.append([samples[labels == label_] for label_ in np.unique(labels)])
    return out


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """The least XZ distance between two sample sets."""
    d = a[:, None, [0, 2]] - b[None, :, [0, 2]]
    return float(np.sqrt((d ** 2).sum(axis=2)).min())


def _centreline(part: np.ndarray, lo: float) -> List[np.ndarray]:
    """One vertex per height band, ascending: the centroid of the flight's cross-section there.

    On a straight flight that is the middle of the flight; in the band a
    landing falls into, the middle of the landing -- which is where the
    agent must turn. A band whose samples fall into several XZ components
    (two flights side by side in one stairwell, a landing and the platform
    beside it) contributes the component that lies on a chain of TOUCHING
    components from the bottom band to the top one -- adjacent bands of one
    flight share an edge, so their samples are a sample spacing apart; a
    surface that merely happens to be at that height is not. The nearest
    centroid was the first rule and chose, by a centimetre, a platform off
    Pomaria's landing over the flight above it, truncating the connector a
    storey short. Nothing is ever truncated here: a flight the chain cannot
    join (a hole in the navmesh) falls back to the largest component per
    band. Habitat frame.
    """
    bands = _band_components(part, lo)
    if not bands:
        return []
    gap = BAND_GAP_M
    reachable: List[List[bool]] = []
    for gap in (BAND_GAP_M, 3.0 * BAND_GAP_M):
        # reachable[b][i]: a chain of touching components leads from component i of band b to the top band.
        reachable = [[True] * len(bands[-1])]
        for b in range(len(bands) - 2, -1, -1):
            above, ok_above = bands[b + 1], reachable[0]
            reachable.insert(0, [any(ok and _gap(c, d) <= gap for d, ok in zip(above, ok_above)) for c in bands[b]])
        if any(reachable[0]):
            break
    else:
        return [max(components, key=len).mean(axis=0) for components in bands]
    out: List[np.ndarray] = []
    chosen: Optional[np.ndarray] = None
    for b, components in enumerate(bands):
        candidates = [c for c, ok in zip(components, reachable[b]) if ok] or list(components)
        if chosen is None:
            candidates.sort(key=len, reverse=True)
        else:
            touching = [c for c in candidates if _gap(c, chosen) <= gap]
            candidates = touching or candidates
            candidates.sort(key=lambda c: (_gap(c, chosen), -len(c)))
        chosen = candidates[0]
        out.append(chosen.mean(axis=0))
    return out


def _anchor_floor_mean(end: np.ndarray, level_height: float, floor: np.ndarray) -> np.ndarray:
    """The floor near one end of a staircase, at the storey's height.

    The mean of the on-level floor samples within 1 m (then 1.5 m) of the
    end's XZ position; the end itself, lifted to the level, when the
    staircase meets the floor without any level surface around it.
    """
    centre = np.asarray([end[0], end[2]], dtype=float)
    on_level = floor[np.abs(floor[:, 1] - level_height) <= 0.15]
    for radius in (1.0, 1.5):
        if len(on_level):
            near = on_level[np.linalg.norm(on_level[:, [0, 2]] - centre, axis=1) <= radius]
            if len(near) >= 3:
                return near.mean(axis=0)
    return np.array([centre[0], level_height, centre[1]])


def _flight_line(bands: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
    """The horizontal direction of a flight's end and its run per metre of rise, from its last few bands.

    ``bands`` run from the inside of the flight to its end. Two adjacent
    band centroids are a noisy pair -- the foot of a flight flares into the
    floor and skews the lowest band sideways -- so the direction is the
    principal axis of the last :data:`ANCHOR_BANDS` centroids, pointed from
    the inside toward the end. ``(None, 0)`` when they do not spread.
    """
    used = np.asarray(bands[-ANCHOR_BANDS:], dtype=float)
    if len(used) < 2:
        return None, 0.0
    xz = used[:, [0, 2]]
    run = float(np.linalg.norm(xz[-1] - xz[0]))
    if run < 1e-6:
        return None, 0.0
    _, _, axes = np.linalg.svd(xz - xz.mean(axis=0), full_matrices=False)
    axis = axes[0]
    if float(np.dot(xz[-1] - xz[0], axis)) < 0.0:
        axis = -axis
    rise = abs(float(used[-1, 1] - used[0, 1]))
    return np.asarray([axis[0], 0.0, axis[1]], dtype=float), (run / rise if rise > 1e-3 else 1.5)


def _anchor(bands: Sequence[np.ndarray], level_height: float, floor: np.ndarray,
            snap: Optional[Callable[[Vec3], Optional[Vec3]]],
            step: Optional[Callable[[Vec3, Vec3], Optional[Vec3]]],
            path_between: Optional[Callable[[Vec3, Vec3], Optional[Sequence[Vec3]]]] = None,
            toward_flight: bool = False) -> np.ndarray:
    """The storey anchor at one end of a flight: on the storey, on the flight's line, and WALKABLE to the flight.

    ``bands`` are the flight's band vertices ordered toward the end in
    question. Candidates, in order of preference: the flight's own line
    (:func:`_flight_line`) continued down (up) to the storey height and set
    out :data:`ANCHOR_OUT_M` beyond -- then shorter and longer set-outs --
    and then the on-level floor samples near the end, nearest to that line
    first. Every candidate is snapped onto the navmesh, must land on the
    storey, and, when the simulator's stride check is available, must be
    walkable between itself and the flight's end in the direction the
    climb walks it (``toward_flight`` for the foot of a flight, which the
    approach reaches and the climb leaves from). The mean of the on-level
    floor near the end -- the first version's only rule -- put the foot of
    Hanson's basement flight in the room BEHIND the stairwell wall, one
    metre away and unreachable, and every climb of it failed; it is the
    last resort here.
    """
    end = np.asarray(bands[-1], dtype=float)
    fallback = _anchor_floor_mean(end, level_height, floor)
    direction, run_per_rise = _flight_line(bands)
    candidates: List[np.ndarray] = []
    base = np.asarray([end[0], level_height, end[2]], dtype=float)
    if direction is not None:
        remaining = abs(float(end[1]) - level_height)
        base = base + direction * (remaining * run_per_rise)
        for out in (ANCHOR_OUT_M, 0.15, 0.6, 0.9):
            candidates.append(base + direction * out)
    on_level = floor[np.abs(floor[:, 1] - level_height) <= 0.15]
    if len(on_level):
        near = on_level[np.linalg.norm(on_level[:, [0, 2]] - np.asarray([end[0], end[2]]), axis=1) <= 1.5]
        if len(near):
            order = np.argsort(np.linalg.norm(near[:, [0, 2]] - np.asarray([base[0], base[2]]), axis=1), kind="stable")
            candidates.extend(near[order[:ANCHOR_CANDIDATES]])
    if snap is None:
        return candidates[0] if candidates else fallback
    for candidate in candidates:
        snapped = snap(np.asarray(candidate, dtype=float).tolist())
        if snapped is None:
            continue
        snapped = np.asarray(snapped, dtype=float)
        if abs(float(snapped[1]) - level_height) > 0.15 or _horizontal(snapped, candidate) > 0.5:
            continue
        if step is None:
            return snapped
        start, goal = (snapped, end) if toward_flight else (end, snapped)
        if _walkable(start, goal, step, {"checked": 0}):
            return snapped
        # Not in a straight line: accept the candidate when the navmesh joins it to the flight by a
        # short, walkable path -- validate_polyline will put that path's vertices into the route.
        if path_between is not None:
            joined = path_between(start.tolist(), goal.tolist())
            if joined is not None and len(joined) >= 2:
                chain = [np.asarray(p, dtype=float) for p in joined]
                length = sum(_horizontal(a, b) for a, b in zip(chain, chain[1:]))
                if length <= 2.5 and all(_walkable(a, b, step, {"checked": 0}) for a, b in zip(chain, chain[1:])):
                    return snapped
    return fallback


def _exit_point(anchor: np.ndarray, bands: Sequence[np.ndarray], level_height: float,
                snap: Optional[Callable[[Vec3], Optional[Vec3]]],
                step: Optional[Callable[[Vec3, Vec3], Optional[Vec3]]]) -> Optional[np.ndarray]:
    """Where the agent walks to, off the flight, for the atlas to confirm the storey: flat floor, walkable from the anchor.

    The atlas confirms a storey on a translated plateau -- three poses at
    one height, :attr:`MultiFloorParams.stable_distance_m` apart -- and the
    first metre off a flight is rarely that: the navmesh eases the last
    treads into a ramp, and an exit stub pointed by rule at 0.75 m, turned
    a quarter turn per blocked step, walked Coffeen's agent BACK UP the
    flight and out of the destination height. So the exit is found here,
    once, with the geometry: :data:`EXIT_REACH_M` out along the flight's
    line, then nearer, then swung to either side up to a right angle, the
    first point that snaps onto the storey within :data:`EXIT_FLAT_M` and
    is walkable from the anchor by the stride check. None when nothing
    qualifies (the traversal then falls back to its stub, swung but never
    turned round). Habitat frame.
    """
    direction, _ = _flight_line(bands)
    if direction is None:
        return None
    outward = np.asarray([direction[0], direction[2]], dtype=float)
    for reach in EXIT_REACH_M:
        for angle in EXIT_SWING_DEG:
            rad = math.radians(angle)
            swung = np.asarray([outward[0] * math.cos(rad) - outward[1] * math.sin(rad),
                                outward[0] * math.sin(rad) + outward[1] * math.cos(rad)])
            candidate = np.asarray([anchor[0] + swung[0] * reach, level_height, anchor[2] + swung[1] * reach], dtype=float)
            point = candidate
            if snap is not None:
                snapped = snap(candidate.tolist())
                if snapped is None:
                    continue
                point = np.asarray(snapped, dtype=float)
                if _horizontal(point, candidate) > 0.3:
                    continue
            if abs(float(point[1]) - level_height) > EXIT_FLAT_M:
                continue
            if step is not None and not _walkable(anchor, point, step, {"checked": 0}):
                continue
            return point
    return None


def densify(points: Sequence[Vec3], spacing_m: float) -> List[np.ndarray]:
    """``points`` with extra samples so that no horizontal gap exceeds ``spacing_m``; endpoints kept."""
    out: List[np.ndarray] = []
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[2] - a[2]) / spacing_m)))
        for index in range(count):
            out.append(a + (b - a) * (index / count))
    out.append(np.asarray(points[-1], float))
    return out


def _dedupe(points: Sequence[np.ndarray], spacing_m: float = 0.05) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if not out or _horizontal(out[-1], p) >= spacing_m or abs(float(p[1] - out[-1][1])) >= spacing_m:
            out.append(p)
    return out


def validate_polyline(points: Sequence[Vec3],
                      snap: Optional[Callable[[Vec3], Optional[Vec3]]] = None,
                      step: Optional[Callable[[Vec3, Vec3], Optional[Vec3]]] = None,
                      path_between: Optional[Callable[[Vec3, Vec3], Optional[Sequence[Vec3]]]] = None
                      ) -> Tuple[List[List[float]], Dict[str, int]]:
    """Walk the polyline with the simulator's collision model and repair the legs it cannot walk.

    Every vertex is snapped onto the navmesh (a band centroid can fall a
    hand's breadth off a curved landing); each leg is densified to one
    stride and each stride tried with ``step`` -- ``(start, end) -> where
    the agent ends up``; a leg with a stride that ends :data:`STEP_MISS_M`
    or more short of its aim is replaced by the pathfinder's own path
    between the leg's ends, whose strides are tried in turn. Habitat frame.

    Returns:
        ``(polyline, report)`` -- the polyline, and ``{"checked",
        "repaired", "unwalkable", "disconnected", "snapped"}`` counts;
        ``checked`` is 0 without ``step``. ``disconnected`` legs have NO
        navmesh path between their ends: the storeys lie on different
        navmesh islands (Hanson, Leonardo, Marstons, Shelbyville), which
        no agent crosses in this simulator -- the connector is then not
        traversable, however real the staircase looks.
    """
    report = {"checked": 0, "repaired": 0, "unwalkable": 0, "disconnected": 0, "snapped": 0}
    vertices: List[np.ndarray] = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if snap is not None:
            snapped = snap(p.tolist())
            if snapped is None:
                continue
            snapped = np.asarray(snapped, dtype=float)
            if _horizontal(snapped, p) > SNAP_MOVE_M:
                report["snapped"] += 1
            p = snapped
        vertices.append(p)
    vertices = _dedupe(vertices)
    if step is None or len(vertices) < 2:
        return [[float(v) for v in p] for p in vertices], report
    out: List[np.ndarray] = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        legs = [(a, b)]
        if not _walkable(a, b, step, report):
            repaired = path_between(a.tolist(), b.tolist()) if path_between is not None else None
            if repaired is not None and len(repaired) >= 2:
                chain = [a] + [np.asarray(p, dtype=float) for p in repaired[1:-1]] + [b]
                legs = list(zip(chain, chain[1:]))
                report["repaired"] += 1
                report["unwalkable"] += sum(0 if _walkable(c, d, step, report) else 1 for c, d in legs)
            else:
                report["unwalkable"] += 1
                report["disconnected"] += 1
        out.extend(d for _, d in legs)
    return [[float(v) for v in p] for p in _simplify(_dedupe(out), step)], report


def _simplify(points: List[np.ndarray], step) -> List[np.ndarray]:
    """Drop a vertex whose legs are shorter than :data:`MIN_LEG_M` when the leg that replaces them is walkable.

    A pathfinder repair round a stair-head corner is a hook of 0.15 m
    legs; a discrete agent reaching its vertices from a quarter-metre away
    passes three of them in one step and aims at the fourth from beside
    the route (Hanson, Coffeen). One walkable leg in their place is walked
    as one leg. The height between the kept vertices stays within a band.
    """
    if step is None or len(points) < 3:
        return points
    out = [points[0]]
    index = 1
    while index < len(points) - 1:
        current, following = points[index], points[index + 1]
        if (_horizontal(out[-1], current) < MIN_LEG_M and abs(float(following[1] - out[-1][1])) <= BAND_M
                and _walkable(out[-1], following, step, {"checked": 0})):
            index += 1                                          # current is dropped; out[-1] -> following
            continue
        out.append(current)
        index += 1
    out.append(points[-1])
    return out


def _walkable(a: np.ndarray, b: np.ndarray, step, report: Dict[str, int]) -> bool:
    """Whether every stride from ``a`` to ``b`` ends where it aimed."""
    strides = densify([a, b], STEP_CHECK_M)
    ok = True
    for c, d in zip(strides, strides[1:]):
        report["checked"] += 1
        reached = step(c.tolist(), d.tolist())
        if reached is None or _horizontal(np.asarray(reached, dtype=float), d) > STEP_MISS_M:
            ok = False
    return ok


def _subsample(points: np.ndarray, spacing_m: float, limit: int) -> np.ndarray:
    """At most one point per ``spacing_m`` voxel, at most ``limit`` of them; deterministic."""
    if not len(points):
        return points
    keys = np.floor(points / spacing_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    kept = points[np.sort(first)]
    if len(kept) > limit:
        kept = kept[np.linspace(0, len(kept) - 1, limit).astype(int)]
    return kept


def stair_connectors(tris: np.ndarray, levels: Sequence[Dict[str, float]],
                     path_between: Optional[Callable[[Vec3, Vec3], Optional[Sequence[Vec3]]]] = None,
                     snap: Optional[Callable[[Vec3], Optional[Vec3]]] = None,
                     step: Optional[Callable[[Vec3, Vec3], Optional[Vec3]]] = None
                     ) -> List[Dict[str, object]]:
    """The staircases joining ADJACENT storeys, in ENU.

    Args:
        tris: Navmesh triangles, Habitat frame.
        levels: :func:`floor_levels` of the same mesh.
        path_between: ``(start, end) -> points`` in the Habitat frame -- the
            navmesh shortest path, when a pathfinder is available -- used to
            REPAIR a leg the collision model cannot walk. Never the route
            itself: it hugs the walls.
        snap: ``point -> nearest navmesh point``, when available.
        step: ``(start, end) -> where the agent ends up`` -- the simulator's
            own collision model (``PathFinder.try_step``), when available.

    Returns:
        One dict per connector: ``id``, ``bottom_xyz``, ``top_xyz`` (the
        polyline's two ends, ENU: the floor anchors on the flight's line),
        ``bottom_z``/``top_z`` (the storey heights they join),
        ``polyline_xyz`` from bottom to top along the middle of the flight,
        ``surface_xyz`` (a sample of the stair surface, for the detector),
        ``length_m``, ``rise_m`` and ``walkability`` (the stride check's
        counts). Empty for a single-storey scene.
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
        for lower, upper, part in _interval_runs(member, levels):
            lo, hi = float(lower["height_m"]), float(upper["height_m"])
            centre = _centreline(part, lo)
            if len(centre) < 2:
                continue
            bottom = _anchor(list(reversed(centre)), lo, floor, snap, step, path_between, toward_flight=True)
            top = _anchor(centre, hi, floor, snap, step, path_between, toward_flight=False)
            polyline, report = validate_polyline([bottom] + centre + [top], snap, step, path_between)
            if len(polyline) < 2:
                continue
            bottom_exit = _exit_point(np.asarray(polyline[0]), list(reversed(centre)), lo, snap, step)
            top_exit = _exit_point(np.asarray(polyline[-1]), centre, hi, snap, step)
            enu = [habitat_to_enu(p) for p in polyline]
            length = sum(math.dist(a, b) for a, b in zip(enu, enu[1:]))
            surface = [habitat_to_enu(p) for p in _subsample(part, SURFACE_SPACING_M, MAX_SURFACE_POINTS)]
            connectors.append({
                "id": len(connectors),
                "bottom_xyz": list(enu[0]), "top_xyz": list(enu[-1]),
                "bottom_exit_xyz": None if bottom_exit is None else habitat_to_enu(bottom_exit),
                "top_exit_xyz": None if top_exit is None else habitat_to_enu(top_exit),
                "bottom_z": lo, "top_z": hi,
                "polyline_xyz": enu, "surface_xyz": surface, "length_m": float(length),
                "rise_m": float(hi - lo), "walkability": dict(report),
                # A leg with no navmesh path between its ends joins two islands: nothing walks it here.
                "traversable": report["disconnected"] == 0})
    connectors.sort(key=lambda c: (c["bottom_z"], c["bottom_xyz"][0], c["bottom_xyz"][1]))
    for index, connector in enumerate(connectors):
        connector["id"] = index
    return connectors


def scene_structure(vertices, path_between=None, snap=None, step=None) -> Dict[str, object]:
    """The metadata block for an episode: storeys and connectors, ENU."""
    tris = triangles(vertices)
    levels = floor_levels(tris)
    return {"stair_source": "navmesh",
            "floor_levels": levels,
            "stair_connectors": stair_connectors(tris, levels, path_between, snap, step)}


def pathfinder_hooks(pathfinder):
    """``(path_between, snap, step)`` over a loaded ``habitat_sim.PathFinder``, Habitat frame."""
    import habitat_sim

    def path_between(start, end):
        query = habitat_sim.ShortestPath()
        query.requested_start = np.asarray(start, dtype=np.float32)
        query.requested_end = np.asarray(end, dtype=np.float32)
        if not pathfinder.find_path(query) or not math.isfinite(query.geodesic_distance):
            return None
        return [np.asarray(p, dtype=float).tolist() for p in query.points]

    def snap(point):
        snapped = np.asarray(pathfinder.snap_point(np.asarray(point, dtype=np.float32)), dtype=float)
        return snapped.tolist() if np.all(np.isfinite(snapped)) else None

    def step(start, end):
        # The simulator moves its agent exactly so: the aimed position filtered by the navmesh, sliding on.
        reached = np.asarray(pathfinder.try_step(np.asarray(start, dtype=np.float32),
                                                 np.asarray(end, dtype=np.float32)), dtype=float)
        return reached.tolist() if np.all(np.isfinite(reached)) else None

    return path_between, snap, step


def scene_structure_from_pathfinder(pathfinder) -> Dict[str, object]:
    """:func:`scene_structure` over a loaded ``habitat_sim.PathFinder``, every leg stride-checked."""
    path_between, snap, step = pathfinder_hooks(pathfinder)
    return scene_structure(pathfinder.build_navmesh_vertices(), path_between, snap, step)

