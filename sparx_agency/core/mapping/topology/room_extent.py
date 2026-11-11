"""Where a room's walls are, how big it is, and how much of it the camera has seen.

The scene graph's room node used to carry one size number -- the area of
the free cells the watershed labelled -- beside the time spent inside and the
frontier count. The step count alone cannot tell two situations apart: a room
almost fully mapped and seen without the target, which can be given up, and a
large room most of which is still beyond the camera's reach, which can
reasonably take many more actions. This module reads, from the depth-derived
occupancy grid and the room partition alone (no ground truth, no simulator
geometry), for every room of the current partition:

* **where each wall is**: the room's boundary faces -- every edge between a
  room cell and a cell outside the room -- classified by what lies beyond
  them: a WALL (an OCCUPIED cell: a wall or a piece of furniture the depth
  put in the height band), another ROOM (a door cut or a threshold), SETTLED
  unknown (looked through without a return, or an enclosed pocket: nothing
  behind it), or UNSEEN unknown (the depth camera has not reached it -- its
  5 m range, an occlusion, a doorway nobody looked through). Faces are
  grouped into the room's FOUR SIDES in the room's own frame (the dominant
  wall direction read off the map, see :func:`wall_orientation`), by the
  outward direction of the face, so an L-shaped room's two north-facing walls
  are one side and a ragged frontier arc is credited to the side it faces;
* **how far each wall is**: the room's extent along each of its two axes, as
  the line each side stands on (world coordinates, for the record and the
  display) and the agent's distance to each when it stands in the room;
* **its dimensions and area**: length and width along the two axes and the
  floor area of the room's footprint -- the labelled free cells plus the
  islands enclosed by them (the floor under a bed or a table, which the
  occupancy reads OCCUPIED and the watershed never labels);
* **whether that size is FINAL or a LOWER BOUND**: a side is *seen* when the
  share of its faces that open onto UNSEEN unknown, in runs wider than a
  doorway, or onto known floor of no room, is at most ``open_tolerance``. A
  narrow run -- at most ``doorway_max_m`` wide, a jamb at each end, in a
  side that is mostly located -- is an exit in a located wall, not a missing
  wall; a wide one, or the whole far side of a sliver, is the camera's reach
  ending inside the room. A room with all four sides seen has a final size;
  otherwise its area and the dimension across each unseen side are lower
  bounds, and nothing here presents them as more
  (:attr:`RoomExtent.bounded`, :attr:`RoomExtent.open_sides`);
* **how much of it has been covered**: the floor the camera has seen from
  within detection range with a clear line of sight, as recorded per cell by
  the sight ledger (``observed``) -- not the cells the agent walked over --
  plus the furniture islands inside the room in proportion to the observed
  floor around them (a bed is seen from the floor beside it), plus the
  settled pockets nothing can stand in. In square metres
  (:attr:`RoomExtent.covered_m2`), as a share of the WHOLE room
  (:attr:`RoomExtent.covered_fraction`: exact when the size is final, an
  UPPER bound when it is not -- the true room is at least the mapped
  footprint), and as what is left (:attr:`RoomExtent.left_m2`: exact, or a
  lower bound). The observed share of the mapped free floor alone is kept
  beside it (``observed_fraction``) for the record.

Everything is recomputed from the live map and the live partition on every
update, and the only accumulated state it reads -- the per-cell observation
mask -- is keyed by map cell on the storey, never by room id. So a room that
merges, splits or is renumbered by the watershed keeps correct walls, sizes
and coverage by construction: the numbers belong to the floor, not the id.

Python 3.8 syntax; numpy and scipy only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import (binary_dilation, binary_fill_holes, find_objects, gaussian_filter,
                           label as connected_components, maximum_filter, sobel, uniform_filter)

#: The four sides of a room, in the room's frame: along its first axis (``+u`` / ``-u``) and its second (``+v`` / ``-v``).
SIDES = ("+u", "-u", "+v", "-v")
#: What a boundary face opens onto.
WALL, ROOM, SETTLED, UNSEEN, OTHER = "wall", "room", "settled", "unseen", "other"
FACE_CLASSES = (WALL, ROOM, SETTLED, UNSEEN, OTHER)
#: ``size_status`` values: every side seen, or at least one side still open to unseen space.
FINAL, LOWER_BOUND = "final", "lower_bound"


@dataclass(frozen=True)
class ExtentParams:
    """The thresholds of the reading. Distances in metres.

    Attributes:
        open_tolerance: A side is SEEN when at most this share of its faces
            open onto unseen unknown in runs wider than a doorway. 0.2: a
            4 m wall with a 0.8 m stretch the depth never resolved is still
            a located wall; a wall half of which is beyond the camera is not.
        doorway_max_m: An unseen run along a side at most this wide is an
            exit through a located wall (a doorway, a gap between a wardrobe
            and the wall), not a missing wall. 1.2 m: wider than any door of
            a home, narrower than the camera's reach ending inside a room.
        min_side_faces: A side with fewer boundary faces than this is read
            as unseen whatever they are: three cells say nothing about a wall.
        orientation_min_faces: The wall faces a room needs before its own
            wall direction overrides the storey's.
        orientation_override_deg: ... and by how many degrees it must differ
            from the storey's for the override to apply: a room turned
            against the house, not a noisy estimate of the same axis.
        orientation_bin_deg: Width of the orientation histogram's bins.
        seen_hysteresis: A side read as seen on the previous update stays
            seen while its open share is at most ``open_tolerance`` plus
            this (``previous`` of :func:`room_extents`), so a wall at the
            threshold does not flip the room's size line -- and the oracle's
            prompt -- on every frame. 0 disables it.
    """

    open_tolerance: float = 0.2
    doorway_max_m: float = 1.2
    min_side_faces: int = 3
    orientation_min_faces: int = 40
    orientation_override_deg: float = 15.0
    orientation_bin_deg: float = 2.0
    seen_hysteresis: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.open_tolerance) < 1.0):
            raise ValueError("open_tolerance must lie in [0, 1), got %r" % (self.open_tolerance,))
        if isinstance(self.seen_hysteresis, bool) or not (0.0 <= float(self.seen_hysteresis) < 1.0):
            raise ValueError("seen_hysteresis must lie in [0, 1), got %r" % (self.seen_hysteresis,))
        for name in ("doorway_max_m", "orientation_override_deg", "orientation_bin_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("%s must be positive and finite, got %r" % (name, value))
        for name in ("min_side_faces", "orientation_min_faces"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer, got %r" % (name, value))


@dataclass(frozen=True)
class SideExtent:
    """One side of a room: what its boundary faces open onto, where it stands, how far the agent is from it.

    Attributes:
        name: One of :data:`SIDES`.
        faces: Boundary faces (cell edges) facing this way; each is one cell long.
        wall: Faces onto an OCCUPIED cell: the wall, or furniture against it.
        room: Faces onto another room (a door cut or a threshold).
        settled: Faces onto unknown the search has settled (nothing behind it).
        unseen: Faces onto unknown the camera has not reached, in runs WIDER
            than a doorway -- the camera's reach ending inside the room.
        gaps: Faces onto unseen unknown in runs no wider than a doorway --
            exits through a located wall.
        other: Faces onto known floor that belongs to no room, or the map's
            edge. No wall is located there either: open for the verdict.
        seen: Whether the wall on this side is located (see the module docstring).
        open_m: Metres of this side open to unseen space (``unseen`` faces).
        line: The side's line in world coordinates, ``((x0, y0), (x1, y1))``:
            the room's extent on this side -- the wall when seen, where the
            map ends when not.
        distance_m: The agent's distance to that line along the side's
            outward axis, when the agent stands in the room; None otherwise.
    """

    name: str
    faces: int
    wall: int
    room: int
    settled: int
    unseen: int
    gaps: int
    other: int
    seen: bool
    open_m: float
    line: Tuple[Tuple[float, float], Tuple[float, float]]
    distance_m: Optional[float] = None

    @property
    def open_fraction(self) -> float:
        """The share of this side's faces that open onto unseen space in wide runs."""
        return float(self.unseen) / float(self.faces) if self.faces > 0 else 1.0


@dataclass(frozen=True)
class RoomExtent:
    """A room's walls, size and coverage, read from the live map and partition.

    Attributes:
        pid: The room's persistent id.
        area_m2: Floor area of the room's footprint: its labelled free cells
            plus the islands they enclose (furniture, enclosed unknown).
        free_m2: Area of the labelled free cells alone (the old ``size``).
        length_m: Extent along the room's first axis (``u``), in metres.
        width_m: Extent along its second axis (``v``); never larger than
            ``length_m``.
        orientation_rad: Direction of the ``u`` axis in the world frame.
        sides: The four :class:`SideExtent` records, in :data:`SIDES` order.
        bounded: Every side seen: the size is FINAL.
        open_sides: Names of the sides not seen (the size is a lower bound
            across them); empty when bounded.
        length_final: Both sides across the ``u`` axis are seen.
        width_final: Both sides across the ``v`` axis are seen.
        perimeter_m: Length of the room's outer boundary (all faces).
        open_perimeter_m: Metres of it open to unseen space in wide runs.
        closed_fraction: ``1 - open_perimeter_m / perimeter_m``.
        observed_fraction: Share of the room's mapped free cells the camera
            has seen from within detection range with a clear line of sight
            (the sight ledger's per-cell record); 0 when no record was given.
        observed_m2: Those cells' area.
        covered_m2: The floor COVERED: the observed free cells, the furniture
            islands the room encloses in proportion to the observed floor
            around them, and the settled pockets inside it.
        covered_fraction: ``covered_m2`` over the footprint ``area_m2`` --
            the share of the WHOLE room covered when the size is final, and
            an UPPER bound on it when the size is a lower bound (the true
            room is at least the mapped footprint). In ``[0, 1]``.
        left_m2: ``area_m2 - covered_m2``: the floor still to be looked at
            -- exact when bounded, a LOWER bound when not.
        centre: World ``(x, y)`` of the footprint's centroid.
        here: The agent stands in this room (its distances are filled).
        size_status: :data:`FINAL` or :data:`LOWER_BOUND`.
    """

    pid: int
    area_m2: float
    free_m2: float
    length_m: float
    width_m: float
    orientation_rad: float
    sides: Tuple[SideExtent, ...]
    bounded: bool
    open_sides: Tuple[str, ...]
    length_final: bool
    width_final: bool
    perimeter_m: float
    open_perimeter_m: float
    closed_fraction: float
    observed_fraction: float
    observed_m2: float
    covered_m2: float
    covered_fraction: float
    left_m2: float
    centre: Tuple[float, float]
    here: bool = False

    @property
    def size_status(self) -> str:
        return FINAL if self.bounded else LOWER_BOUND

    @property
    def walls_seen(self) -> int:
        """How many of the four sides are seen."""
        return sum(1 for side in self.sides if side.seen)

    @property
    def unobserved_m2(self) -> float:
        """The mapped free floor the camera has NOT seen from detection range -- a lower bound on what is left."""
        return max(0.0, self.free_m2 - self.observed_m2)

    @property
    def covered_is_bound(self) -> bool:
        """Whether ``covered_fraction`` is an upper bound (the size is a lower bound) rather than the room's share."""
        return not self.bounded

    def side(self, name: str) -> SideExtent:
        for side in self.sides:
            if side.name == name:
                return side
        raise KeyError(name)

    def distances(self) -> Dict[str, Optional[float]]:
        """``{side name: the agent's distance to it}`` (None when the agent is not in the room)."""
        return {side.name: side.distance_m for side in self.sides}

    def to_record(self) -> Dict[str, object]:
        """A JSON-able summary for the recording and the display."""
        out = asdict(self)
        out["sides"] = [dict(side) for side in out["sides"]]
        out["open_sides"] = list(self.open_sides)
        out["size_status"] = self.size_status
        out["walls_seen"] = self.walls_seen
        out["unobserved_m2"] = round(self.unobserved_m2, 2)
        out["covered_is_bound"] = self.covered_is_bound
        for key in ("area_m2", "free_m2", "length_m", "width_m", "perimeter_m", "open_perimeter_m", "observed_m2",
                    "covered_m2", "left_m2"):
            out[key] = round(float(out[key]), 2)
        for key in ("closed_fraction", "observed_fraction", "covered_fraction"):
            out[key] = round(float(out[key]), 3)
        out["orientation_deg"] = round(math.degrees(self.orientation_rad), 1)
        out["centre"] = [round(float(v), 2) for v in self.centre]
        for side in out["sides"]:
            side["open_m"] = round(float(side["open_m"]), 2)
            side["line"] = [[round(float(v), 2) for v in point] for point in side["line"]]
            if side["distance_m"] is not None:
                side["distance_m"] = round(float(side["distance_m"]), 2)
        return out


# -- orientation ----------------------------------------------------------------
#: Side of the window (cells) the local wall direction is read over: 1.1 m at 0.1 m cells, enough to average
#: out the staircase a diagonal wall rasterises to, less than the width of a corridor.
ORIENTATION_WINDOW_CELLS = 11


def _edge_angles(occupied: np.ndarray, free: np.ndarray, where: Optional[np.ndarray] = None,
                 window: int = ORIENTATION_WINDOW_CELLS) -> Tuple[np.ndarray, np.ndarray]:
    """The local wall directions (mod 90 degrees) and their weights at the edges between ``free`` and ``occupied``.

    At each edge cell the direction is the principal axis of the edge cells
    in a ``window`` x ``window`` neighbourhood (their second moments: the
    structure tensor of the point set), weighted by how line-like that
    neighbourhood is -- 1 for a straight run of wall, 0 for a blob or a
    corner. A pixel gradient would read the staircase a diagonal wall
    rasterises to; the moments over a metre of wall do not. ``where``
    restricts the edges read (a room's own wall faces).
    """
    edges = occupied & binary_dilation(free, structure=np.ones((3, 3), dtype=bool))
    if where is not None:
        edges &= where
    if not edges.any():
        return np.zeros(0), np.zeros(0)
    e = edges.astype(np.float32)
    h, w = e.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    size = max(3, int(window))
    mean = lambda a: uniform_filter(a, size=size, mode="constant")
    n = mean(e)
    good = n > 1e-9
    mx = np.where(good, mean(e * xs) / np.maximum(n, 1e-9), 0.0)
    my = np.where(good, mean(e * ys) / np.maximum(n, 1e-9), 0.0)
    cxx = np.where(good, mean(e * xs * xs) / np.maximum(n, 1e-9) - mx * mx, 0.0)
    cyy = np.where(good, mean(e * ys * ys) / np.maximum(n, 1e-9) - my * my, 0.0)
    cxy = np.where(good, mean(e * xs * ys) / np.maximum(n, 1e-9) - mx * my, 0.0)
    ey, ex = np.nonzero(edges)
    a, b, c = cxx[ey, ex], cyy[ey, ex], cxy[ey, ex]
    trace = a + b
    root = np.sqrt(np.maximum(0.0, (a - b) ** 2 + 4.0 * c * c))
    anisotropy = np.where(trace > 1e-9, root / np.maximum(trace, 1e-9), 0.0)
    angle = np.mod(0.5 * np.arctan2(2.0 * c, a - b), math.pi / 2.0)
    keep = anisotropy > 0.3
    return angle[keep], anisotropy[keep]


def _histogram_peak(angles: np.ndarray, weights: np.ndarray, bin_deg: float) -> Tuple[Optional[float], float]:
    """The peak of a circular histogram over [0, 90) degrees: ``(angle_rad, share of the weight within +-6 degrees)``."""
    if not len(angles):
        return None, 0.0
    bins = max(4, int(round(90.0 / float(bin_deg))))
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, math.pi / 2.0), weights=weights)
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    peak = int(np.argmax(smoothed))
    window = max(1, int(round(6.0 / float(bin_deg))))
    around = sum(hist[(peak + k) % bins] for k in range(-window, window + 1))
    total = float(hist.sum())
    share = float(around) / total if total > 0 else 0.0
    centre = (peak + 0.5) * (math.pi / 2.0) / bins
    # Refine within the peak's neighbourhood by the circular mean (period 90 degrees) of the angles there.
    period = math.pi / 2.0
    offset = np.mod(angles - centre + period / 2.0, period) - period / 2.0
    near = np.abs(offset) <= math.radians(6.0)
    if near.any():
        phase = np.angle(np.sum(weights[near] * np.exp(1j * offset[near] * 4.0))) / 4.0
        centre = float(np.mod(centre + phase, period))
    return centre, share


def wall_orientation(occupied: np.ndarray, free: np.ndarray, bin_deg: float = 2.0,
                     where: Optional[np.ndarray] = None, min_faces: int = 1) -> Tuple[Optional[float], float]:
    """The dominant wall direction of ``occupied`` against ``free``, in radians mod 90 degrees, and how dominant it is.

    Homes are mostly rectilinear: one axis, read off every wall the map
    holds, orients every room's frame, and a room turned against the house
    gets its own when its walls say so clearly (:func:`room_extents`).
    Returns ``(None, 0.0)`` when fewer than ``min_faces`` edge cells exist.
    """
    angles, weights = _edge_angles(np.asarray(occupied, dtype=bool), np.asarray(free, dtype=bool), where)
    if len(angles) < max(1, int(min_faces)):
        return None, 0.0
    return _histogram_peak(angles, weights, bin_deg)


def _angle_gap_deg(a: float, b: float) -> float:
    """The difference between two axis directions, mod 90 degrees, in [0, 45]."""
    diff = math.degrees(abs(a - b)) % 90.0
    return min(diff, 90.0 - diff)


# -- the reading -----------------------------------------------------------------
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))     # (dx, dy) in grid cells: +x, -x, +y, -y


def _shifted(source: np.ndarray, direction: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """``source`` read at each cell's neighbour in ``direction``, and whether that neighbour lies in the array."""
    dx, dy = direction
    hh, ww = source.shape
    out = np.zeros(source.shape, dtype=bool)
    inside = np.zeros(source.shape, dtype=bool)
    src_y, src_x = slice(max(0, -dy), hh - max(0, dy)), slice(max(0, -dx), ww - max(0, dx))
    dst_y, dst_x = slice(max(0, dy), hh - max(0, -dy)), slice(max(0, dx), ww - max(0, -dx))
    out[src_y, src_x] = source[dst_y, dst_x]
    inside[src_y, src_x] = True
    return out, inside


def boundary_normals(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The outward unit normal at every cell of ``mask`` -- ``(nx, ny, strong)``, ``strong`` False where it vanishes.

    The normal is the negated gradient of the smoothed mask, so a wall at
    45 degrees to the grid -- whose faces alternate between two grid
    directions -- has one normal along its whole length. It vanishes on a
    one-cell sliver, whose two edges cancel; such a face is placed by its
    own direction alone (:func:`face_sides`).
    """
    smooth = gaussian_filter(mask.astype(np.float32), sigma=1.5)
    nx, ny = -sobel(smooth, axis=1), -sobel(smooth, axis=0)
    norm = np.hypot(nx, ny)
    strong = norm > 1e-4
    safe = np.where(strong, norm, 1.0)
    return nx / safe, ny / safe, strong


def room_axes(theta: float) -> np.ndarray:
    """The outward unit vectors of the four :data:`SIDES` for a room frame turned ``theta``: a ``(4, 2)`` array."""
    return np.array([[math.cos(theta), math.sin(theta)], [-math.cos(theta), -math.sin(theta)],
                     [-math.sin(theta), math.cos(theta)], [math.sin(theta), -math.cos(theta)]])


def face_sides(ys: np.ndarray, xs: np.ndarray, direction: Tuple[int, int], normals, axes: np.ndarray) -> np.ndarray:
    """Which side each face at cells ``(ys, xs)`` with outward grid ``direction`` belongs to.

    A face is scored on each side's axis by its own direction plus its
    cell's outward normal: a corner cell's two faces go to two sides (the
    face direction decides), a diagonal wall's alternating faces go to one
    (the normal decides), and a sliver's faces, whose normal vanishes, go
    by direction alone.
    """
    nx, ny, strong = normals
    d = np.array(direction, dtype=float)
    scores = np.broadcast_to(axes @ d, (len(ys), len(axes))).copy()
    n = np.stack([nx[ys, xs], ny[ys, xs]], axis=1)
    scores += np.where(strong[ys, xs][:, None], n @ axes.T, 0.0)
    return np.argmax(scores, axis=1)


def side_index(mask: np.ndarray, theta: float) -> np.ndarray:
    """Which of :data:`SIDES` each boundary cell of ``mask`` leans to by its outward normal: ``(H, W)`` int, -1 off the boundary.

    A corner cell is placed on one side here; :func:`face_sides` places
    each of its faces. Kept for the display and the tests.
    """
    nx, ny, strong = boundary_normals(mask)
    axes = room_axes(theta)
    boundary = np.zeros(mask.shape, dtype=bool)
    fallback = np.full(mask.shape, -1, dtype=np.int64)
    for direction in _DIRECTIONS:
        neighbour, _ = _shifted(mask, direction)
        face = mask & ~neighbour
        boundary |= face
        first = face & (fallback < 0)
        fallback[first] = int(np.argmax(axes @ np.array(direction, dtype=float)))
    out = np.full(mask.shape, -1, dtype=np.int64)
    ys, xs = np.nonzero(boundary)
    dots = np.stack([nx[ys, xs], ny[ys, xs]], axis=1) @ axes.T
    out[ys, xs] = np.where(strong[ys, xs], np.argmax(dots, axis=1), fallback[ys, xs])
    return out


def room_extents(grid: np.ndarray, labels: np.ndarray, resolution: float, origin: Tuple[float, float],
                 free_value: int = 0, occupied_value: int = 100, unknown_value: int = -1,
                 observed: Optional[np.ndarray] = None, resolved: Optional[np.ndarray] = None,
                 here_xy: Optional[Tuple[float, float]] = None, params: Optional[ExtentParams] = None,
                 orientation: Optional[float] = None,
                 previous: Optional[Dict[int, "RoomExtent"]] = None) -> Dict[int, RoomExtent]:
    """Read :class:`RoomExtent` for every room of ``labels`` from the map ``grid``.

    Args:
        grid: ``(H, W)`` occupancy values (``free_value`` / ``occupied_value`` / ``unknown_value``).
        labels: ``(H, W)`` int room label image -- ``pid + 1`` inside a room, 0 elsewhere.
        resolution: Metres per cell.
        origin: World ``(x, y)`` of the grid's corner (cell ``(0, 0)`` spans ``origin`` to ``origin + resolution``).
        observed: Optional ``(H, W)`` bool -- the cells the camera has seen from detection range.
        resolved: Optional ``(H, W)`` bool -- unknown cells the search has settled (looked through, pockets):
            closed for the reading, as the frontier logic treats them.
        here_xy: The agent's world position, for the distances of the room it stands in.
        params: :class:`ExtentParams`.
        orientation: The storey's wall direction in radians, when the caller has it; else read here.
        previous: The previous update's reading, for ``seen_hysteresis``: a side seen then stays seen a
            little past the tolerance. Keyed by pid only for this; a pid handed to another room costs at
            most one borderline verdict on one update.

    Returns:
        ``{pid: RoomExtent}`` for every pid in ``labels``.
    """
    params = params or ExtentParams()
    grid = np.asarray(grid)
    labels = np.asarray(labels)
    if grid.shape != labels.shape:
        raise ValueError("grid %r and labels %r differ in shape" % (grid.shape, labels.shape))
    res = float(resolution)
    h, w = grid.shape
    free = grid == free_value
    occupied = grid == occupied_value
    unknown = grid == unknown_value
    settled = np.zeros_like(unknown)
    if resolved is not None:
        settled = np.asarray(resolved, dtype=bool) & unknown
        unknown = unknown & ~settled
    if observed is not None:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != grid.shape:
            raise ValueError("observed %r and grid %r differ in shape" % (observed.shape, grid.shape))
    storey_theta = orientation
    if storey_theta is None:
        # Over the known part of the map only: the grid is 80 m square, the home a fraction of it.
        known_rows = np.flatnonzero((occupied | free).any(axis=1))
        known_cols = np.flatnonzero((occupied | free).any(axis=0))
        if len(known_rows) and len(known_cols):
            box = (slice(known_rows[0], known_rows[-1] + 1), slice(known_cols[0], known_cols[-1] + 1))
            storey_theta, _ = wall_orientation(occupied[box], free[box], params.orientation_bin_deg,
                                               min_faces=params.orientation_min_faces)
    if storey_theta is None:
        storey_theta = 0.0
    here_cell = None
    if here_xy is not None:
        gx = int(math.floor((float(here_xy[0]) - float(origin[0])) / res))
        gy = int(math.floor((float(here_xy[1]) - float(origin[1])) / res))
        if 0 <= gx < w and 0 <= gy < h:
            here_cell = (gx, gy)
    out = {}  # type: Dict[int, RoomExtent]
    slices = find_objects(labels)
    for index, box in enumerate(slices):
        if box is None:
            continue
        value = index + 1
        pid = value - 1
        y0, y1 = max(0, box[0].start - 1), min(h, box[0].stop + 1)
        x0, x1 = max(0, box[1].start - 1), min(w, box[1].stop + 1)
        window = (slice(y0, y1), slice(x0, x1))
        mask = labels[window] == value
        if not mask.any():
            continue
        before = previous.get(pid) if previous else None
        seen_before = None if before is None else {side.name: side.seen for side in before.sides}
        extent = _room_extent(pid, mask, grid[window], labels[window], free[window], occupied[window],
                              unknown[window], settled[window],
                              None if observed is None else observed[window],
                              (x0, y0), res, origin, storey_theta, params, here_cell, here_xy, seen_before)
        out[pid] = extent
    return out


def _room_extent(pid, mask, grid, labels, free, occupied, unknown, settled, observed, offset, res, origin,
                 storey_theta, params, here_cell, here_xy=None, seen_before=None):
    """One room's reading over its bounding window (``offset`` is the window's ``(x, y)`` cell origin)."""
    hh, ww = mask.shape
    ox, oy = offset
    # -- the room's frame ------------------------------------------------------------------------------------------
    wall_edges = occupied & binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    theta = storey_theta
    own, share = wall_orientation(occupied, mask, params.orientation_bin_deg, where=wall_edges,
                                  min_faces=params.orientation_min_faces)
    if own is not None and share >= 0.5 and _angle_gap_deg(own, storey_theta) > params.orientation_override_deg:
        theta = own
    u_axis = (math.cos(theta), math.sin(theta))
    v_axis = (-math.sin(theta), math.cos(theta))
    # -- footprint and extents -------------------------------------------------------------------------------------
    filled = binary_fill_holes(mask)
    holes = filled & ~mask & (labels == 0)          # islands the room encloses that belong to no other room
    footprint = mask | holes
    ys, xs = np.nonzero(mask)
    cx = (xs + ox + 0.5) * res + float(origin[0])
    cy = (ys + oy + 0.5) * res + float(origin[1])
    us = cx * u_axis[0] + cy * u_axis[1]
    vs = cx * v_axis[0] + cy * v_axis[1]
    u_min, u_max = float(us.min()) - 0.5 * res, float(us.max()) + 0.5 * res
    v_min, v_max = float(vs.min()) - 0.5 * res, float(vs.max()) + 0.5 * res
    fy, fx = np.nonzero(footprint)
    centre = (float(((fx + ox + 0.5) * res + origin[0]).mean()), float(((fy + oy + 0.5) * res + origin[1]).mean()))
    # -- the boundary faces, by side (the cell's outward normal) and by what they open onto ------------------------
    normals = boundary_normals(mask)
    axes = room_axes(theta)
    counts = np.zeros((len(SIDES), len(FACE_CLASSES)), dtype=np.int64)
    unseen_cells = np.zeros(mask.shape, dtype=bool)        # boundary cells with a face onto unseen unknown
    unseen_side = np.full(mask.shape, -1, dtype=np.int64)  # ... and the side that face was placed on
    closed_cells = np.zeros(mask.shape, dtype=bool)        # boundary cells with a wall, room or settled face
    others = labels > 0
    for direction in _DIRECTIONS:
        neighbour_room, inside = _shifted(mask, direction)
        face = mask & ~neighbour_room
        if not face.any():
            continue
        rim = face & ~inside                                    # the map's edge: nothing is known beyond it
        face &= inside
        interior, _ = _shifted(holes, direction)
        face &= ~interior                                       # a face onto an enclosed island is not a wall of the room
        side_of = np.full(mask.shape, -1, dtype=np.int64)
        fy, fx = np.nonzero(face | rim)
        if len(fy):
            side_of[fy, fx] = face_sides(fy, fx, direction, normals, axes)
        remaining = face
        for column, source in ((0, occupied), (1, others), (2, settled), (3, unknown)):
            beyond, _ = _shifted(source, direction)
            hit = remaining & beyond
            if hit.any():
                counts[:, column] += np.bincount(side_of[hit], minlength=len(SIDES))[:len(SIDES)]
                if column == 3:
                    unseen_cells |= hit
                    unseen_side[hit] = side_of[hit]
                else:
                    closed_cells |= hit
                remaining = remaining & ~hit
        leftover = remaining | rim
        if leftover.any():
            counts[:, 4] += np.bincount(side_of[leftover], minlength=len(SIDES))[:len(SIDES)]
    # -- unseen runs: a doorway-wide run is an exit, a wider one is the camera's reach ending ---------------------
    sides = []
    bounded = True
    open_sides = []
    perimeter_faces = 0
    open_faces_total = 0
    seen_by_name = {}
    comp, n_comp = connected_components(unseen_cells, structure=np.ones((3, 3), dtype=bool))
    is_gap = np.zeros(n_comp + 1, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    for k in range(1, n_comp + 1):
        run = comp == k
        ky, kx = np.nonzero(run)
        px = (kx + ox + 0.5) * res + float(origin[0])
        py = (ky + oy + 0.5) * res + float(origin[1])
        along_u = px * u_axis[0] + py * u_axis[1]
        along_v = px * v_axis[0] + py * v_axis[1]
        span_u, span_v = float(along_u.max() - along_u.min()), float(along_v.max() - along_v.min())
        # The run's width is its extent along the boundary: the larger of its spans on the two axes.
        if max(span_u, span_v) + res > params.doorway_max_m:
            continue
        # A doorway has a jamb on each side: closed boundary within a cell of the run, beyond both its ends
        # along the axis it runs on. A narrow piece of frontier with nothing located beside it is not a door.
        tangent = u_axis if span_u >= span_v else v_axis
        along = along_u if span_u >= span_v else along_v
        # Closed boundary within a cell of the run, the run's own corner cells included (a cell with an unseen
        # face one way and a wall face the other is where the wall meets the opening).
        flank = binary_dilation(run, structure=structure) & closed_cells
        if not flank.any():
            continue
        fy, fx = np.nonzero(flank)
        ft = ((fx + ox + 0.5) * res + float(origin[0])) * tangent[0] + ((fy + oy + 0.5) * res + float(origin[1])) * tangent[1]
        is_gap[k] = bool(ft.min() <= along.min() + 1e-9 and ft.max() >= along.max() - 1e-9)
    wide_cells = unseen_cells & ~is_gap[comp]
    for s, name in enumerate(SIDES):
        total = int(counts[s].sum())
        unseen_faces = int(counts[s, 3])
        other_faces = int(counts[s, 4])
        closed_faces = total - unseen_faces - other_faces
        # Faces of this side whose cell lies in a doorway-wide run are gaps, the rest of the unseen are open --
        # provided the side is mostly LOCATED: a doorway is an opening in a wall that is there. On a sliver seen
        # through a door the whole far side is narrower than a doorway and flanked by the jambs, and it is not a
        # door, it is the camera's reach ending (00880/000009, action 62: a 0.7 m2 "room, all walls seen" that was
        # 7.9 m2 three actions later).
        narrow_faces = 0
        if unseen_faces and closed_faces * 2 >= total:
            cells_here = (unseen_side == s) & unseen_cells
            n_wide = int((cells_here & wide_cells).sum())
            n_cells = int(cells_here.sum())
            # Faces per cell are not tracked per run; apportion the side's unseen faces by its cells' share.
            narrow_faces = int(round(unseen_faces * (1.0 - (n_wide / float(n_cells) if n_cells else 0.0))))
        wide_faces = unseen_faces - narrow_faces
        # Known floor that belongs to no room (``other``) is no wall either: the room's floor may go on into it
        # (the stub beyond a door before the watershed seeds the room behind it). Open for the verdict, kept
        # apart in the record.
        open_share = (wide_faces + other_faces) / float(total) if total else 1.0
        tolerance = params.open_tolerance
        if seen_before is not None and seen_before.get(name):
            tolerance += params.seen_hysteresis
        seen = total >= params.min_side_faces and open_share <= tolerance
        seen_by_name[name] = seen
        if not seen:
            bounded = False
            open_sides.append(name)
        perimeter_faces += total
        open_faces_total += wide_faces
        if name == "+u":
            a, b = (u_max, v_min), (u_max, v_max)
        elif name == "-u":
            a, b = (u_min, v_min), (u_min, v_max)
        elif name == "+v":
            a, b = (u_min, v_max), (u_max, v_max)
        else:
            a, b = (u_min, v_min), (u_max, v_min)
        line = tuple((p[0] * u_axis[0] + p[1] * v_axis[0], p[0] * u_axis[1] + p[1] * v_axis[1]) for p in (a, b))
        sides.append(dict(name=name, faces=total, wall=int(counts[s, 0]), room=int(counts[s, 1]),
                          settled=int(counts[s, 2]), unseen=int(wide_faces), gaps=int(narrow_faces),
                          other=int(counts[s, 4]), seen=bool(seen), open_m=float(wide_faces) * res,
                          line=line, distance_m=None))
    # -- the agent's distances, when it stands in the room ---------------------------------------------------------
    here = False
    if here_cell is not None:
        gx, gy = here_cell[0] - ox, here_cell[1] - oy
        if 0 <= gy < hh and 0 <= gx < ww and mask[gy, gx]:
            here = True
            hx, hy = float(here_xy[0]), float(here_xy[1])
            hu = hx * u_axis[0] + hy * u_axis[1]
            hv = hx * v_axis[0] + hy * v_axis[1]
            for side in sides:
                if side["name"] == "+u":
                    side["distance_m"] = max(0.0, u_max - hu)
                elif side["name"] == "-u":
                    side["distance_m"] = max(0.0, hu - u_min)
                elif side["name"] == "+v":
                    side["distance_m"] = max(0.0, v_max - hv)
                else:
                    side["distance_m"] = max(0.0, hv - v_min)
    length_u = u_max - u_min
    length_v = v_max - v_min
    length_final = seen_by_name["+u"] and seen_by_name["-u"]
    width_final = seen_by_name["+v"] and seen_by_name["-v"]
    if length_v > length_u:
        length_u, length_v = length_v, length_u
        length_final, width_final = width_final, length_final
    n_free = int(mask.sum())
    observed_cells = int((observed & mask).sum()) if observed is not None else 0
    covered_cells = float(observed_cells) + _covered_islands(mask, holes, occupied, settled,
                                                             None if observed is None else (observed & mask))
    n_footprint = int(footprint.sum())
    perimeter_m = perimeter_faces * res
    open_perimeter_m = open_faces_total * res
    return RoomExtent(
        pid=int(pid), area_m2=float(n_footprint) * res * res, free_m2=float(n_free) * res * res,
        length_m=float(length_u), width_m=float(length_v), orientation_rad=float(theta),
        sides=tuple(SideExtent(**side) for side in sides), bounded=bool(bounded), open_sides=tuple(open_sides),
        length_final=bool(length_final), width_final=bool(width_final),
        perimeter_m=float(perimeter_m), open_perimeter_m=float(open_perimeter_m),
        closed_fraction=float(1.0 - open_perimeter_m / perimeter_m) if perimeter_m > 0 else 0.0,
        observed_fraction=float(observed_cells) / n_free if n_free else 0.0,
        observed_m2=float(observed_cells) * res * res,
        covered_m2=float(covered_cells) * res * res,
        covered_fraction=min(1.0, float(covered_cells) / n_footprint) if n_footprint else 0.0,
        left_m2=max(0.0, float(n_footprint - covered_cells)) * res * res,
        centre=centre, here=here)


def _covered_islands(mask, holes, occupied, settled, observed):
    """How many cells of the islands ``mask`` encloses count as covered.

    A furniture island (OCCUPIED cells the room surrounds) is seen from
    the floor around it: its cells are credited in proportion to the
    observed share of the free cells adjacent to it -- a bed with every
    cell of floor beside it observed is covered whole, one seen from one
    side half. A settled pocket (unknown nothing can stand in) is covered
    by definition. Unknown islands the camera never resolved are not.
    """
    if not holes.any():
        return 0.0
    credited = float((holes & settled).sum())
    furniture = holes & occupied & ~settled
    if not furniture.any() or observed is None:
        return credited
    comp, n = connected_components(furniture, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return credited
    # Each free cell of the room beside an island carries that island's label (the largest, where two touch).
    ring = maximum_filter(comp, size=3) * mask
    total = np.bincount(ring.ravel(), minlength=n + 1).astype(float)
    seen = np.bincount((ring * observed).ravel(), minlength=n + 1).astype(float)
    sizes = np.bincount(comp.ravel(), minlength=n + 1).astype(float)
    share = np.where(total[1:] > 0, seen[1:] / np.maximum(total[1:], 1.0), 0.0)
    return credited + float((sizes[1:] * share).sum())


# -- prompt words ------------------------------------------------------------------
def coarse_fraction(value: float, step: float = 0.1) -> int:
    """A fraction as a percentage rounded DOWN to ``step`` (0.1: tens), so an unchanged map gives an unchanged prompt."""
    value = max(0.0, min(1.0, float(value)))
    return int(math.floor(value / step + 1e-9) * step * 100.0 + 0.5)


def coarse_metres(value: float, step: float = 0.5) -> str:
    """Metres in ``step`` steps (``3.5``), never ``3.50``; whole metres without the decimal."""
    rounded = round(float(value) / step) * step
    text = "%.1f" % rounded
    return text[:-2] if text.endswith(".0") else text


def size_words(extent: Optional[RoomExtent], area_m2: float = 0.0) -> str:
    """The ``size=`` value of a room's prompt line.

    ``12m2 (3.5x3.5m, all walls seen)`` when the size is final;
    ``at least 12m2 (5+ x 3.5m, 1 wall not seen yet)`` when a side is
    still open to unseen space -- the ``+`` marks the dimension that is a
    lower bound -- and ``under 1m2 mapped so far (...)`` for a sliver;
    ``under 1m2`` / ``unknown size`` as before when nothing better is known.
    The oracle renders its node from the same rule (``format_size``).
    """
    if extent is None:
        area = float(area_m2)
        return "unknown size" if area <= 0.0 else "under 1m2" if area < 1.0 else "%dm2" % int(round(area))
    area = max(float(extent.area_m2), float(extent.free_m2))
    area_text = "under 1m2" if area < 1.0 else "%dm2" % int(round(area))
    length = coarse_metres(extent.length_m) + ("" if extent.length_final else "+")
    width = coarse_metres(extent.width_m) + ("" if extent.width_final else "+")
    if extent.bounded:
        return "%s (%sx%sm, all walls seen)" % (area_text, length, width)
    missing = len(extent.open_sides)
    walls = "1 wall not seen yet" if missing == 1 else "%d of 4 walls not seen yet" % missing
    if area < 1.0:
        return "under 1m2 mapped so far (%s x %sm, %s)" % (length, width, walls)
    return "at least %s (%s x %sm, %s)" % (area_text, length, width, walls)


def area_words(value: float) -> str:
    """Square metres for the prompt: ``0m2``, ``under 1m2``, else the nearest whole ``12m2``."""
    value = float(value)
    if value < 0.05:
        return "0m2"
    if value < 1.0:
        return "under 1m2"
    return "%dm2" % int(round(value))


def coverage_words(covered_m2: float, fraction: float, left_m2: float, bounded: bool) -> str:
    """The ``covered=`` value of a room's prompt line: the floor the camera has looked at from detection range, as final results.

    Bounded room: ``10m2 (80% of the room, 2m2 left)``, the share rounded
    DOWN to tens, or ``12m2 (all of it)``. Size a lower bound: the share is
    an UPPER bound rounded UP, the remainder a LOWER bound -- ``3m2 (at most
    40% of the room, at least 4m2 left)`` -- or, with everything mapped
    covered, ``7m2 (all of the mapped part; the unmapped rest is unknown)``.
    The model is handed the numbers it would otherwise derive.
    """
    covered = area_words(covered_m2)
    fraction = max(0.0, min(1.0, float(fraction)))
    if bounded:
        if fraction >= 0.995:
            return "%s (all of it)" % covered
        percent = int(math.floor(fraction * 10.0 + 1e-9)) * 10
        left = area_words(left_m2)
        return "%s (%d%% of the room, %s left)" % (covered, percent, left)
    percent = min(100, int(math.ceil(fraction * 10.0 - 1e-9)) * 10)
    if percent >= 100:
        # "at most 100%" says nothing: what is known is that everything mapped was looked at.
        return "%s (all of the mapped part; the unmapped rest is unknown)" % covered
    if float(left_m2) < 1.0:
        return "%s (at most %d%% of the room)" % (covered, percent)      # "at least under 1m2 left" says nothing
    return "%s (at most %d%% of the room, at least %s left)" % (covered, percent, area_words(left_m2))


def seen_words(extent: Optional[RoomExtent]) -> Optional[str]:
    """The ``covered=`` value for ``extent`` (:func:`coverage_words`), or None without one."""
    if extent is None:
        return None
    return coverage_words(extent.covered_m2, extent.covered_fraction, extent.left_m2, extent.bounded)
