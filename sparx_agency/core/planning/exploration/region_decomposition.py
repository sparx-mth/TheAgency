"""Cut a surveyed building into rooms, corridors and doorways, from two height bands.

The trick, and the reason this is short: **ask the building where its doors are
by looking above them.** An occupancy map sliced through the band a drone flies
in has every doorway standing open, so its free space is one connected blob and
no amount of eroding separates the rooms -- measured on the SJTU hospital,
shrinking the floor by 0.70 m still left a single 532 m² component, because its
wards open onto the spine through metre-wide gaps rather than through doors.

Slice the *same world* through the band just under the ceiling instead and every
opening is closed by its own lintel, so the rooms fall apart into connected
components on their own. The rooms are what that band separates; the doorways are
exactly where the two bands disagree.

Nothing here is hand-drawn, and that is the point: the decomposition is the
world's own architecture, so it is right for the same reason the occupancy map is
right, and it is regenerated rather than re-drawn when the world changes.

Two things it does not know and must be told: which component is the circulation
space (give it a point inside, or accept the largest), and what to call anything.

Offline. Pure numpy, no scipy, no OpenCV -- it runs once and its output is
committed, so it does not have to be fast, only legible.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types.geometry import Pose2D
from sparx_agency.core.planning.environment.grid_regions import (
    connected_regions,
    largest_enclosed_region,
)
from sparx_agency.core.planning.exploration.region_map import (
    NO_REGION,
    Portal,
    Region,
    RegionMap,
)


def decompose_regions(flight_occupied, wall_occupied, resolution,
                      origin_x, origin_y, min_room_m2=3.0, min_portal_m=0.30,
                      corridor_seed=None, names=None):
    # type: (np.ndarray, np.ndarray, float, float, float, float, float, Optional[Tuple[float, float]], Optional[Dict[int, str]]) -> RegionMap
    """Build a :class:`RegionMap` from a flight-band and a wall-band occupancy.

    Args:
        flight_occupied: ``(H, W)`` boolean, True where geometry blocks the
            band the aircraft flies in. Its largest enclosed free component is
            the building's floor and becomes the extent of the decomposition.
        wall_occupied: ``(H, W)`` boolean over the same grid, sliced through a
            band where doorways are closed by their lintels -- for the SJTU
            hospital, 2.10-2.40 m. Only its connectivity is used.
        resolution: Metres per cell.
        origin_x: World x of column 0's left edge.
        origin_y: World y of row 0's bottom edge.
        min_room_m2: Components smaller than this are not rooms; their cells are
            absorbed into whichever region grows into them.
        min_portal_m: Openings narrower than this are label-adjacency noise
            rather than doors, and are dropped. Note the airframe is wider than
            this -- filtering by what can be *flown* is the mission generator's
            job, not the map's.
        corridor_seed: ``(x, y)`` inside the circulation space. Whichever region
            contains it is the corridor and every other is a room. Defaults to
            the largest region, which is the same answer in any building whose
            corridors connect.
        names: Optional ``{region id: name}``. Ids are assigned largest-first
            and are stable for a given pair of bands, so a name table can be
            written once and kept.

    Returns:
        The decomposition, with every cell of the floor assigned.

    Raises:
        ValueError: The two bands are different shapes, or the flight band has
            no enclosed free region to decompose.
    """
    flight_occupied = np.asarray(flight_occupied, dtype=bool)
    wall_occupied = np.asarray(wall_occupied, dtype=bool)
    if flight_occupied.shape != wall_occupied.shape:
        raise ValueError("the two bands must share a grid: %r vs %r"
                         % (flight_occupied.shape, wall_occupied.shape))

    floor = largest_enclosed_region(~flight_occupied, connectivity=4)
    if floor is None:
        raise ValueError("the flight band has no enclosed free region -- there "
                         "is no building here to decompose")

    cell_m2 = resolution * resolution
    # The wall band's free space, clipped to the floor: one component per room,
    # plus one for everything the corridors join together.
    seeds = (~wall_occupied) & floor
    components = [c for c in connected_regions(seeds, connectivity=4)
                  if c.sum() * cell_m2 >= min_room_m2]

    labels = np.zeros(floor.shape, dtype=np.int32)
    for index, component in enumerate(components, start=1):
        labels[component] = index
    # Everything the wall band closed -- the door thresholds themselves, and any
    # component too small to be a room -- goes to whichever region reaches it
    # first. A threshold is about six cells thick, so the split line through a
    # doorway lands in the doorway, which is the only place it could sensibly go.
    labels = _grow_into(labels, floor)
    labels[~floor] = NO_REGION

    present = [int(v) for v in np.unique(labels) if int(v) != NO_REGION]
    corridor_id = _corridor_id(labels, present, resolution, origin_x, origin_y,
                               corridor_seed)

    portals = _portals(labels, resolution, origin_x, origin_y, min_portal_m)
    by_region = {}  # type: Dict[int, List[str]]
    for portal in portals:
        for rid in portal.between:
            by_region.setdefault(rid, []).append(portal.id)

    names = dict(names or {})
    regions = []
    for rid in present:
        mask = labels == rid
        kind = "corridor" if rid == corridor_id else "room"
        default = "%s %d" % (kind, rid)
        regions.append(Region(
            id=rid, name=names.get(rid, default), kind=kind,
            area_m2=float(mask.sum() * cell_m2),
            centre=_inside_centre(mask, resolution, origin_x, origin_y),
            portals=tuple(sorted(by_region.get(rid, ()), key=int))))
    return RegionMap(labels, resolution, origin_x, origin_y, regions, portals)


def split_region_by_bands(region_map, region_id, bands):
    # type: (RegionMap, int, Sequence[Tuple[str, float, float]]) -> RegionMap
    """Cut one region into named horizontal slices, and rebuild the portals.

    For the circulation space, which no height band can separate -- corridors
    have no doors between them, so they are one component by construction and
    stay one however the building is sliced. Where to cut is an architectural
    judgement rather than a measurement, so it is a caller's argument and
    belongs in a file a person can read.

    Args:
        region_map: The decomposition to refine.
        region_id: Which region to split. Its kind is inherited by every slice.
        bands: ``(name, y_min, y_max)`` in world metres, half-open in y_max.
            Cells of the region outside every band keep the original id, so a
            band list that does not cover the region leaves a remainder rather
            than losing it.

    Returns:
        A new :class:`RegionMap`. Ids of other regions are unchanged; the
        slices take fresh ids above the current maximum.

    Raises:
        ValueError: ``region_id`` is not in the map, or two bands overlap.
    """
    if int(region_id) not in region_map.regions:
        raise ValueError("no region %r to split" % (region_id,))
    ordered = sorted(bands, key=lambda b: b[1])
    for (_, _, upper), (name, lower, _) in zip(ordered, ordered[1:]):
        if lower < upper:
            raise ValueError("bands overlap at %r" % (name,))

    labels = region_map.labels.copy()
    target = labels == int(region_id)
    rows = np.arange(region_map.height)
    y_lo = region_map.origin_y + rows * region_map.resolution
    y_hi = y_lo + region_map.resolution
    y_mid = 0.5 * (y_lo + y_hi)

    next_id = max(region_map.regions) + 1
    names = {}  # type: Dict[int, str]
    kind = region_map.regions[int(region_id)].kind
    for name, low, high in ordered:
        in_band = (y_mid >= low) & (y_mid < high)
        slice_mask = target & in_band[:, None]
        if not slice_mask.any():
            continue
        labels[slice_mask] = next_id
        names[next_id] = name
        next_id += 1

    kinds = {r.id: r.kind for r in region_map.regions.values()}
    kinds.update({rid: kind for rid in names})
    keep_names = {r.id: r.name for r in region_map.regions.values()}
    keep_names.update(names)
    return _rebuild(labels, region_map, keep_names, kinds)


def _rebuild(labels, template, names, kinds):
    # type: (np.ndarray, RegionMap, Dict[int, str], Dict[int, str]) -> RegionMap
    """A fresh RegionMap over edited labels, with portals recomputed."""
    res, ox, oy = template.resolution, template.origin_x, template.origin_y
    cell_m2 = res * res
    portals = _portals(labels, res, ox, oy, 0.0)
    by_region = {}  # type: Dict[int, List[str]]
    for portal in portals:
        for rid in portal.between:
            by_region.setdefault(rid, []).append(portal.id)
    regions = []
    for value in np.unique(labels):
        rid = int(value)
        if rid == NO_REGION:
            continue
        mask = labels == rid
        regions.append(Region(
            id=rid, name=names.get(rid, "region %d" % rid),
            kind=kinds.get(rid, "room"),
            area_m2=float(mask.sum() * cell_m2),
            centre=_inside_centre(mask, res, ox, oy),
            portals=tuple(sorted(by_region.get(rid, ()), key=int))))
    return RegionMap(labels, res, ox, oy, regions, portals)


# ── the pieces ───────────────────────────────────────────────────────────

def _grow_into(labels, allowed, max_steps=None):
    # type: (np.ndarray, np.ndarray, Optional[int]) -> np.ndarray
    """Flood every labelled region outward, one cell a step, inside ``allowed``.

    A simultaneous multi-source dilation, so cells go to the nearest region by
    4-connected hop count. Neighbours are consulted up, down, left, right and
    the first non-zero wins, which makes ties deterministic rather than fair --
    acceptable, because the only cells in contention are the few inside a
    doorway.
    """
    out = np.asarray(labels).copy()
    if max_steps is None:
        max_steps = int(out.shape[0] + out.shape[1])
    for _ in range(max_steps):
        todo = allowed & (out == NO_REGION)
        if not todo.any():
            break
        pick = _shift_up(out)
        for shifted in (_shift_down(out), _shift_right(out), _shift_left(out)):
            pick = np.where(pick == NO_REGION, shifted, pick)
        take = todo & (pick != NO_REGION)
        if not take.any():
            break            # an island with no labelled neighbour anywhere
        out[take] = pick[take]
    return out


def _shift_up(a):
    """At each cell, the value of the cell BELOW it in array order (row - 1)."""
    out = np.zeros_like(a)
    out[1:, :] = a[:-1, :]
    return out


def _shift_down(a):
    out = np.zeros_like(a)
    out[:-1, :] = a[1:, :]
    return out


def _shift_right(a):
    out = np.zeros_like(a)
    out[:, 1:] = a[:, :-1]
    return out


def _shift_left(a):
    out = np.zeros_like(a)
    out[:, :-1] = a[:, 1:]
    return out


def _corridor_id(labels, present, resolution, origin_x, origin_y, seed):
    # type: (np.ndarray, List[int], float, float, float, Optional[Tuple[float, float]]) -> int
    """Which region is the circulation space.

    The point given, if one is; otherwise the largest, which is the same answer
    wherever the corridors connect to each other -- and where they do not, the
    caller has a building this function cannot read for them and should say so
    with a seed.
    """
    if not present:
        raise ValueError("nothing was labelled, so there is no corridor to find")
    if seed is not None:
        gx = int(np.floor((seed[0] - origin_x) / resolution))
        gy = int(np.floor((seed[1] - origin_y) / resolution))
        height, width = labels.shape
        if not (0 <= gx < width and 0 <= gy < height):
            raise ValueError("corridor_seed %r is off the map" % (seed,))
        value = int(labels[gy, gx])
        if value == NO_REGION:
            raise ValueError("corridor_seed %r is not inside the building" % (seed,))
        return value
    return max(present, key=lambda rid: int((labels == rid).sum()))


def _portals(labels, resolution, origin_x, origin_y, min_width_m):
    # type: (np.ndarray, float, float, float, float) -> List[Portal]
    """Every place two regions touch, as one portal per contiguous stretch.

    Two regions can touch in more than one place -- a ward with a door at each
    end of the corridor is the normal case -- so the boundary cells are split
    into connected runs and each run becomes its own opening. Collapsing them
    into one portal per pair would put its centre in the wall between them.
    """
    labels = np.asarray(labels)
    pairs = {}  # type: Dict[Tuple[int, int], np.ndarray]
    for shifted, axis in ((_shift_right(labels), 1), (_shift_down(labels), 0)):
        both = (labels != NO_REGION) & (shifted != NO_REGION)
        differ = both & (labels != shifted)
        if not differ.any():
            continue
        lo = np.minimum(labels, shifted)
        hi = np.maximum(labels, shifted)
        for a, b in set(zip(lo[differ].tolist(), hi[differ].tolist())):
            touching = differ & (lo == a) & (hi == b)
            key = (int(a), int(b))
            pairs[key] = touching if key not in pairs else (pairs[key] | touching)

    out = []  # type: List[Portal]
    next_id = 1
    for (a, b) in sorted(pairs):
        for run in connected_regions(pairs[(a, b)], connectivity=8):
            rows, cols = np.nonzero(run)
            height = rows.max() - rows.min()
            width_cells = cols.max() - cols.min()
            span = max(height, width_cells) + 1
            width = float(span * resolution)
            if width < min_width_m:
                continue
            # The MEDOID of the run, not its mean. A long or bent boundary --
            # the 24 m junction where this building's south hall meets its
            # corridor is both -- has a mean that lies off the boundary
            # entirely, so a centre taken there is not a point of the opening,
            # and stepping either side of it lands in the same region twice.
            mean_row, mean_col = rows.mean(), cols.mean()
            nearest = int(np.argmin((rows - mean_row) ** 2 + (cols - mean_col) ** 2))
            row, col = int(rows[nearest]), int(cols[nearest])
            centre_x = float(origin_x + (col + 0.5) * resolution)
            centre_y = float(origin_y + (row + 0.5) * resolution)
            yaw = _normal_yaw(labels, row, col, height >= width_cells, b)
            out.append(Portal(
                id=str(next_id),
                center=Pose2D(x=centre_x, y=centre_y, yaw=yaw),
                normal_yaw=yaw, width_m=width, between=(a, b)))
            next_id += 1
    return out


def _normal_yaw(labels, row, col, upright, towards):
    # type: (np.ndarray, int, int, bool, int) -> float
    """Which way the opening faces, pointing towards region ``towards``.

    Perpendicular to the run of boundary cells, because that is the axis a
    doorway is crossed along, and signed by looking one cell to each side to see
    which holds the far region. Carried on
    :class:`~sparx_agency.core.common.types.semantics.Portal2D.normal_yaw`, which
    is what ``EnterPortalBehavior`` builds its approach-then-cross waypoints
    from -- so a portal found here can be crossed by that behaviour unchanged.

    It means what it says only for an opening narrower than the regions it
    joins. Where one room simply runs into another over twenty metres the
    boundary is a line rather than a gap, and the direction you cross it is not
    a property it has; the value returned there is well-formed and not worth
    aiming by.
    """
    height, width = labels.shape
    # A run that is taller than it is wide lies in a vertical wall, so the way
    # through it is horizontal.
    step = (0, 1) if upright else (1, 0)
    forward = (min(max(row + step[0], 0), height - 1),
               min(max(col + step[1], 0), width - 1))
    if int(labels[forward]) == int(towards):
        return 0.0 if upright else math.pi / 2.0
    return math.pi if upright else -math.pi / 2.0


def _inside_centre(mask, resolution, origin_x, origin_y):
    # type: (np.ndarray, float, float, float) -> Tuple[float, float]
    """The region's centroid, moved to the nearest cell actually inside it.

    A corridor that bends, or a room around a pillar, has a centre of mass in
    the wall. The centre is used to aim the aircraft, so it has to be a place.
    """
    rows, cols = np.nonzero(mask)
    mean_row, mean_col = rows.mean(), cols.mean()
    nearest = int(np.argmin((rows - mean_row) ** 2 + (cols - mean_col) ** 2))
    return (float(origin_x + (cols[nearest] + 0.5) * resolution),
            float(origin_y + (rows[nearest] + 0.5) * resolution))
