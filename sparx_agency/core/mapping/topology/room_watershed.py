# core/mapping/topology/room_watershed.py
"""Grid-based room segmentation: clearance watershed, doors forced.

A drop-in alternative to
:func:`~sparx_agency.core.mapping.topology.room_segmentation.compute_rooms`
with the same return triple, written because the skeleton-cut pipeline
is UNSTABLE as exploration proceeds.

Why, measured
-------------
Segmenting a real captured FALCON BEV (413x200 @ 0.15 m, 7852 occupied
cells) while simulating growing coverage as a disc around the explored
centroid, largest room as a share of the segmented area::

    free cells   door-cut (1.6 m)      watershed (min_distance 2.0 m)
      19403      12 rooms  29%          5 rooms  35%
      35137      12 rooms  65%         14 rooms  28%
      48979      12 rooms  79%         14 rooms  26%
      57464      15 rooms  76%         19 rooms  23%

The door-cut decomposition COLLAPSES into one dominant room as coverage
grows; this one stays separated and improves. Two causes, and neither is
a tuning error:

1. The live BEV marks walls only where the drone actually observed
   them, so free space leaks between rooms at openings that are not in
   the pre-listed door set. 35 doors are listed for this building and
   only 11 carry a known width.
2. The medial axis of the explored region is ONE connected component,
   so cutting it at 35 doors cannot separate it — and portal widths in
   this building reach 24.75 m, which a 1.6 m disk cannot sever.

This module therefore stops deriving rooms from the skeleton's topology
and derives them from clearance geometry instead: every local maximum of
the distance-to-obstacle field seeds a room, and the watershed of that
field pushes the boundaries out to the narrow places between seeds.
A missing wall no longer merges two rooms — it only lowers the ridge
between them — so the decomposition degrades gracefully with coverage
rather than collapsing at a threshold.

Doors are still absolute: the user's model is "a room is a closed area
bounded by doors", so a listed door ALWAYS separates, whatever the
geometry says.

Where the cut lands matters when the door position is an estimate. A
surveyed door sits in its wall; a door *detected* from RGB-D sits where
the box's depth put it, and the Hanson ObjectNav recording of
2026-10-04 had one 0.6 m off its doorway, in the middle of a bedroom's
floor: the 0.75 m disk there missed the doorway it stood for and
severed the bedroom around its bed into two "rooms" (R0/R6) instead.
``door_snap_reach_m`` therefore moves each cut to the **choke** nearest
the door -- the medial-axis cell within reach where the passage is
narrowest and whose removal really parts the skeleton -- and sizes the
disk to that passage, so the cut spans the doorway and nothing beside
it. A door with no visible constriction within reach (walls not yet
observed, an archway wider than a door) keeps the plain disk carve.

The watershed cannot under-segment, but it OVER-segments: a room with
two wide spots either side of some furniture, or bent into an L, grows
two clearance peaks and comes out as two rooms. ``merge_dynamics_m``
repairs that, in
:mod:`~sparx_agency.core.mapping.topology.room_merge`. Same captured
BEV with the building's 35 listed doors carved, 43 basins, against a
ground truth of 20 rooms + 7 corridors = 27 regions::

    merge_dynamics_m  0.00(off)  0.30  0.50(default)  1.00  2.00
    rooms                   43     36             29    27     26
    largest room         10.7%  12.2%          12.2% 12.2%  12.4%

Merging by SADDLE WIDTH alone instead cascades the whole floor into one
region through the corridors (15 rooms, largest 82%); that measurement,
and why dynamics do not, are in ``room_merge``'s module docstring.

Coordinate convention, matching the sibling modules: grids are indexed
``[y, x]`` (row, col); door cells are ``(cx, cy)`` = (col, row) pairs.

Dependencies: numpy, scipy, scikit-image. Topology is a host-owned path
(never imported inside the Noetic FALCON container), so scipy and
skimage are allowed here — unlike a ``core`` module on a FALCON import
path, and unlike ``room_segmentation``, this module has no skimage-free
fallback and raises on import without it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.draw import line as draw_line
from skimage.feature import peak_local_max
from skimage.morphology import medial_axis
from skimage.segmentation import watershed

from sparx_agency.core.mapping.topology.room_merge import (
    merge_basins_by_dynamics,
)
from sparx_agency.core.mapping.topology.room_segmentation import (
    RoomStats,
    door_disk_mask,
    heal_free_mask,
)

# 8-connectivity for merging touching clearance peaks into one seed.
_EIGHT = np.ones((3, 3), np.uint8)
#: Four-connectivity, the flood's and the basin sides'.
_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


@dataclass(frozen=True)
class WatershedRoomParams:
    """Tuning knobs for :func:`segment_rooms_watershed`.

    Attributes:
        min_room_separation_m: Minimum distance between two room seeds.
            Peaks of the clearance field closer together than this are
            suppressed, so this is the smallest spacing at which two
            open areas can still read as two rooms. 2.0 m is the value
            behind the measured table in the module docstring.
        min_clearance_m: A seed must sit at least this far from the
            nearest non-free cell. 0.6 m rejects peaks inside doorways
            and corridor stubs, which would otherwise seed a "room"
            made of the doorway itself.
        min_room_cells: Rooms smaller than this many cells are
            discarded. 150 cells is 3.4 m^2 at 0.15 m — smaller than
            any real hospital room, larger than the fragments. Matches
            the flown node's parameter of the same name.
        door_cut_m: Radius of the disk carved out around each door cell
            to force a boundary there. 1.60 m, the value the flown node
            measured for its own cut on this 0.15 m BEV.
        merge_dynamics_m: Repair over-segmentation by merging adjacent
            basins whose DYNAMICS — how much clearance is lost from the
            shallower peak down to the saddle between them — fall below
            this. 0.50 m is the low end of a plateau: 0.50 through
            2.00 m all give 26-29 rooms on the captured BEV against a
            ground truth of 27, with the largest room fixed near 12%.
            0.0 disables the stage entirely and reproduces the raw
            watershed. Doors are never merged across, whatever this
            says. Full sweep in ``room_merge``'s module docstring.
        door_snap_reach_m: How far a door cell may be moved onto the
            nearest **choke** before the cut is carved (see the module
            docstring). 0.0 (the default, the surveyed-door case) carves
            the disk at the door cell itself. A detected door should
            allow about its own position error, 0.9 m on the ObjectNav
            maps.
        door_choke_min_m: The narrowest half-width a choke may have to
            stand for a door: a passage the body cannot pass is a notch
            between obstacles, not a doorway.
        door_choke_max_m: The widest. Beyond it the passage is open
            floor, and the door keeps its plain disk carve.
        threshold_snap_reach_m: How far a WALKED-THROUGH threshold cell
            (:func:`trail_thresholds`) may be moved onto a severing choke
            before it is carved; a threshold with no such choke within
            reach is not carved at all. 0 disables threshold carving.
        threshold_dip_max_m: The most clearance a trail cell may have to
            be a threshold: half the width of the widest passage read as
            a doorway (0.55 m -> 1.1 m), narrower than ``door_choke_max_m``
            because no door detection vouches for it.
        threshold_rise_m: How much more clearance the trail must reach on
            BOTH sides of the dip, within ``threshold_side_m`` of travel,
            for the dip to be a threshold between two wider spaces rather
            than a point of a uniformly narrow corridor.
        threshold_side_m: The trail length either side of a dip over which
            the rise is looked for.
        stub_min_cells: Instantiate a room for the floor BEYOND a cut
            that no clearance peak seeded -- the sliver of a room glimpsed
            through its doorway -- when it holds at least this many cells
            and borders unknown (``unknown_mask``). 0 (the default, the
            surveyed-BEV case) keeps the older behaviour, in which such a
            sliver is given to the nearest seeded room ACROSS the cut by
            :func:`_reclaim_carved` and the space beyond an open door is
            no room of its own until it grows a peak (the ObjectNav
            recordings of 2026-10-04 to 2026-10-21: the toilet seen through
            a door named the hallway a bathroom). A stub is exempt from
            ``min_room_cells``; it keeps its number as it grows (the
            registry matches by containment).
    """

    min_room_separation_m: float = 2.0
    min_clearance_m: float = 0.6
    min_room_cells: int = 150
    door_cut_m: float = 1.60
    merge_dynamics_m: float = 0.50
    door_snap_reach_m: float = 0.0
    door_choke_min_m: float = 0.25
    door_choke_max_m: float = 0.8
    threshold_snap_reach_m: float = 0.0
    threshold_dip_max_m: float = 0.55
    threshold_rise_m: float = 0.30
    threshold_side_m: float = 1.5
    stub_min_cells: int = 0


def segment_rooms_watershed(
    free_mask: np.ndarray,
    resolution: float,
    params: WatershedRoomParams = WatershedRoomParams(),
    door_cells: Iterable[Tuple[int, int]] = (),
    threshold_cells: Iterable[Tuple[int, int]] = (),
    unknown_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[RoomStats]]:
    """Segment free space into rooms by watershed of its clearance field.

    Pipeline: heal the free mask, distance-transform it to metres, carve
    a disk out of that mask at every door, seed one marker per local
    maximum of the clearance field, watershed the negated field, give
    the carved cells back to their nearest room, merge the basins the
    watershed split without cause, then drop the runts and relabel
    1..N.

    Signature-compatible with
    :func:`~sparx_agency.core.mapping.topology.room_segmentation.compute_rooms`
    in its return triple, so ``RoomRegistry``, ``room_stats`` and the
    payload/marker builders consume either segmenter unchanged.

    Args:
        free_mask: (H, W) bool, True where the grid reads free.
        resolution: Grid resolution in metres per cell. Every ``_m``
            parameter is converted through it.
        params: Tuning knobs.
        door_cells: Door positions as ``(cx, cy)`` cell pairs that must
            separate rooms whatever the geometry says. Out-of-bounds
            doors are ignored. May be empty — pure geometry then.
        threshold_cells: ``(cx, cy)`` cells of thresholds the robot has
            WALKED THROUGH (:func:`trail_thresholds`) or SEEN a doorway at
            on the way to an opening (:func:`sightline_thresholds`):
            carved like a snapped door when a severing choke lies within
            ``params.threshold_snap_reach_m``, and otherwise not at all --
            no plain disk, no barrier -- since nothing but the geometry
            vouches for them.
        unknown_mask: (H, W) bool, True where the grid is UNKNOWN; read
            with ``params.stub_min_cells`` to tell the floor glimpsed
            beyond a doorway (it borders unknown) from a nook a cut
            happened to part. None accepts any parted sliver.

    Returns:
        Tuple of:
            room_lbl: (H, W) int32 — 0 = not-a-room, 1..N = rooms
                (fresh labels, contiguous, not stable across ticks).
            skeleton: (H, W) bool — the medial axis of the healed free
                mask. It does NOT define the rooms here (that is the
                whole point of this module); it is returned because the
                RViz view draws it per-room as the "open space" spine
                and the operator explicitly wants the Voronoi skeleton
                visible. Restricted to the surviving rooms, as
                ``compute_rooms`` restricts its own.
            stats: One :class:`RoomStats` per room, in label order,
                with the same id/mask/n_cells/centroid_cells semantics
                as ``compute_rooms`` — centroid is ``(cx, cy)``.

    Raises:
        ValueError: If ``resolution`` is not strictly positive.
    """
    if not resolution > 0.0:
        raise ValueError("resolution must be > 0, got %r" % (resolution,))

    H, W = free_mask.shape
    empty_lbl = np.zeros((H, W), np.int32)
    empty_sk = np.zeros((H, W), bool)

    healed = heal_free_mask(free_mask)
    if not healed.any():
        return empty_lbl, empty_sk, []

    # Clearance in METRES to the nearest non-free cell. Computed on the
    # healed mask, NOT on the door-carved one, so a door disk does not
    # dent the field it is supposed to only fence.
    dt = distance_transform_edt(healed) * float(resolution)

    # FORCED DOOR BOUNDARIES. The carve is applied to the watershed MASK
    # rather than to the markers because a marker only proposes a room
    # centre — the watershed would happily flood straight through the
    # doorway between two markers and re-merge the rooms. Removing the
    # cells makes the doorway unfloodable, which is the only way to
    # guarantee separation independent of what the clearance field says.
    carve, barrier = door_carve_mask(healed, dt, door_cells, params, resolution,
                                     threshold_cells=threshold_cells, unknown_mask=unknown_mask)
    wmask = healed & ~carve
    if not wmask.any():
        return empty_lbl, empty_sk, []

    markers, n_markers = _clearance_markers(dt, wmask, params, resolution)
    if n_markers == 0:
        return empty_lbl, empty_sk, []

    labels = watershed(-dt, markers, mask=wmask)
    # THE FLOOR BEYOND A DOORWAY IS A ROOM OF ITS OWN (``stub_min_cells``, since 2026-10-21):
    # a sliver glimpsed through a door has no clearance peak, so no marker and no label, and
    # the reclaim below would hand it to the nearest seeded room -- ACROSS the cut that was
    # carved to keep them apart. Seeded here first, before the sides are read, so the merge
    # treats it as a basin parted by the cut and the reclaim finds it labelled already.
    stub_cells = None
    if params.stub_min_cells > 0 and carve.any():
        labels, stub_cells = _stub_rooms(labels, wmask, carve, unknown_mask, params.stub_min_cells)
    sides = _basin_sides(labels, wmask)
    labels = _reclaim_carved(labels, healed)

    # OVER-SEGMENTATION REPAIR, before the size floor rather than after:
    # a lobe the watershed split off a real room is often a runt, and
    # merging it back keeps its cells in the room instead of orphaning
    # them. Two records of where a door was survive the reclaim above:
    # ``barrier``, the plain disks (a pair whose border touches one is
    # never merged, as before), and ``sides``, the component of the carved
    # mask each basin was flooded in -- basins parted by ANY cut lie on
    # different sides and never merge, while two basins of one room beside
    # a snapped cut share a side and still may.
    labels = merge_basins_by_dynamics(labels, dt, params.merge_dynamics_m,
                                      barrier, basin_sides=sides)

    room_lbl, stats = _collect_rooms(labels, params.min_room_cells, stub_cells)
    if not stats:
        return empty_lbl, empty_sk, []

    # ``medial_axis`` breaks plateau ties with a PRNG seeded fresh on every
    # call unless one is supplied, so the spine visibly flickers between
    # otherwise identical ticks. Pin it. (``compute_rooms`` does not, and
    # there the skeleton DEFINES the rooms: five identical calls on the
    # captured BEV returned 11, 11, 11, 11 and 10 rooms.)
    skeleton = np.asarray(medial_axis(healed, rng=0), dtype=bool)
    return room_lbl, skeleton & (room_lbl > 0), stats


def door_carve_mask(
    healed: np.ndarray,
    dt: np.ndarray,
    door_cells: Iterable[Tuple[int, int]],
    params: WatershedRoomParams,
    resolution: float,
    threshold_cells: Iterable[Tuple[int, int]] = (),
    unknown_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The cells the watershed may not flood because a door stands there, and the merge barrier.

    With ``params.door_snap_reach_m == 0`` both are the plain disk of
    ``door_cut_m`` around every door cell. Otherwise each door is first
    moved onto the choke nearest it (:func:`snap_door_to_choke`) and the
    disk carved there is sized to the passage, ``clearance + 1 cell``, so
    it spans the doorway wall to wall and no more. Such a cut contributes
    nothing to the mask barrier -- the two rooms it parts lie on different
    sides of the carved mask, which :func:`segment_rooms_watershed` passes
    to the merge as ``basin_sides``, and the disk's protrusion into the
    rooms must not stop two basins of one room from merging. That holds
    only when the disk actually SEVERS the healed mask; since 2026-10-05 a
    snapped disk that leaves the floor around it in one piece (the medial
    axis is parted, the floor is not -- the clearance at a skeleton cell
    near the frontier is the distance to an unknown cell, so the disk is
    narrower than the passage) is replaced by the plain disk, barrier and
    all. Before, such a door had no separating power at all: the room
    behind it merged into the hallway on the ticks the snap found a choke
    and split off again under a new number on the ticks it did not -- the
    Ranchester door of 2026-10-05 whose room read R23, R26, R28 and R52.
    A door with no choke within reach keeps its plain disk for both.

    ``threshold_cells`` -- thresholds the robot walked through, found by
    :func:`trail_thresholds` -- are carved the snapped way only, within
    ``params.threshold_snap_reach_m``, and skipped when no severing choke
    is in reach: a plain disk on open floor would cut a room in two for
    nothing, and no detection vouches for a threshold the way it does for
    a door.

    Args:
        healed: (H, W) bool healed free mask.
        dt: (H, W) float clearance field in metres, computed on ``healed``.
        door_cells: Door positions as ``(cx, cy)`` cell pairs.
        params: Tuning knobs.
        resolution: Metres per cell.
        threshold_cells: Walked-through (or sighted) threshold cells, ``(cx, cy)``.
        unknown_mask: (H, W) bool, True where the grid is unknown; lets a
            choke whose far side is the short tail of a room glimpsed
            through its door count (:func:`snap_door_to_choke`), and the
            cut there be a line across the passage when the disk would
            swallow that tail (:func:`_severing_cut`).

    Returns:
        ``(carve, barrier)``: two (H, W) bool masks -- the cells carved out
        of the watershed mask, and the plain disks among them, across which
        basins are never merged (a subset of ``carve``).
    """
    cells = [(int(cx), int(cy)) for cx, cy in door_cells]
    thresholds = [(int(cx), int(cy)) for cx, cy in threshold_cells] if params.threshold_snap_reach_m > 0.0 else []
    shape = healed.shape
    plain_radius = int(round(params.door_cut_m / resolution))
    if not cells and not thresholds:
        empty = np.zeros(shape, bool)
        return empty, empty.copy()
    if not params.door_snap_reach_m > 0.0 and not thresholds:
        plain = door_disk_mask(shape, cells, plain_radius)
        return plain, plain.copy()
    fields = choke_fields(healed, resolution, unknown_mask, dt=dt)
    carve = np.zeros(shape, bool)
    barrier = np.zeros(shape, bool)
    for cell in cells:
        choke = fields.snap(cell, params) if params.door_snap_reach_m > 0.0 else None
        if choke is not None:
            cut = _severing_cut(healed, fields.nearest, choke, resolution)
            if cut is not None:
                carve |= cut
                continue
        plain = door_disk_mask(shape, [cell], plain_radius)
        carve |= plain
        barrier |= plain
    if thresholds:
        reach = replace(params, door_snap_reach_m=params.threshold_snap_reach_m)
        for cell in thresholds:
            choke = fields.snap(cell, reach)
            if choke is None:
                continue
            cut = _severing_cut(healed, fields.nearest, choke, resolution)
            if cut is not None:
                carve |= cut
    return carve, barrier


class ChokeFields:
    """The per-grid fields the choke snap reads, computed once for every door and threshold of a tick.

    Attributes:
        healed: The healed free mask the watershed floods.
        skeleton: Its medial axis.
        snap_dt: The clearance the chokes are ranked by -- to OBSERVED
            OBSTACLES with the unknown passable when ``unknown_mask`` is
            given (a sliver seen through a door is narrow because the
            unknown hems it in, and would otherwise out-narrow the doorway
            itself; the watershed still floods ``dt``, where the unknown is
            a wall), else to every non-free cell.
        leads_on: The band within :data:`LEADS_ON_CELLS` of the unknown, or None.
        obstacles: The observed obstacle cells, or None without the unknown.
        nearest: The nearest-non-free-cell index field (``distance_transform_edt(healed, return_indices=True)[1]``).
        resolution: Metres per cell.
    """

    def __init__(self, healed, resolution, unknown_mask=None, dt=None):
        self.healed = healed
        self.resolution = float(resolution)
        self.skeleton = np.asarray(medial_axis(healed, rng=0), dtype=bool)
        self.leads_on = None
        self.obstacles = None
        if unknown_mask is not None:
            unknown = np.asarray(unknown_mask, bool)
            self.leads_on = binary_dilation(unknown, structure=_FOUR.astype(bool), iterations=LEADS_ON_CELLS)
            self.snap_dt = distance_transform_edt(healed | unknown) * self.resolution
            self.obstacles = ~healed & ~unknown
        else:
            self.snap_dt = dt if dt is not None else distance_transform_edt(healed) * self.resolution
        self.nearest = distance_transform_edt(healed, return_indices=True)[1]

    def snap(self, cell, params):
        """:func:`snap_door_to_choke` for ``cell`` with these fields and ``params``'s reach."""
        return snap_door_to_choke(self.skeleton, self.snap_dt, cell, params, self.resolution, self.leads_on,
                                  self.healed, self.obstacles, self.nearest)


def choke_fields(healed_or_free: np.ndarray, resolution: float, unknown_mask: Optional[np.ndarray] = None,
                 dt: Optional[np.ndarray] = None, healed: bool = True) -> ChokeFields:
    """The :class:`ChokeFields` of a grid; ``healed=False`` heals ``healed_or_free`` first (a raw free mask)."""
    mask = np.asarray(healed_or_free, bool)
    if not healed:
        mask = heal_free_mask(mask)
    return ChokeFields(mask, resolution, unknown_mask, dt)


def _severing_cut(
    healed: np.ndarray,
    nearest: Tuple[np.ndarray, np.ndarray],
    choke: Tuple[Tuple[int, int], float],
    resolution: float,
) -> Optional[np.ndarray]:
    """The cut at a snapped choke that parts the floor: a line across the passage, else the passage-sized disk, else None.

    The LINE -- two cells thick, from the choke toward its nearest jamb
    and the opposite way, as long as the passage is wide -- is the fence
    the flood needs and no more (since 2026-10-21). The disk --
    ``clearance + 1 cell`` around the choke -- was the only cut before,
    and is kept for a passage the line does not seal (an irregular
    opening the line crosses at an angle). The disk failed exactly where
    the floor beyond the door is a sliver seen through it: it swallowed
    the sliver whole or left two crescents of it, so the door had no
    separating power or parted the sliver in two. ``nearest`` is the
    nearest-blocked-cell index field the line's direction is read from
    (:func:`_across_direction`).
    """
    (cx, cy), half_width = choke
    radius = int(np.ceil(half_width / resolution)) + 1
    line = _line_cut(healed.shape, (cx, cy), radius, _across_direction(nearest, (cx, cy)))
    if line is not None and _severs(healed, line, (cx, cy), radius):
        return line
    disk = door_disk_mask(healed.shape, [(cx, cy)], radius)
    if _severs(healed, disk, (cx, cy), radius):
        return disk
    return None


def _across_direction(nearest: Tuple[np.ndarray, np.ndarray], cell: Tuple[int, int]) -> Optional[np.ndarray]:
    """The unit direction from ``cell`` to its nearest blocked cell -- across the passage, toward a jamb -- or None.

    ``nearest`` is the index pair ``distance_transform_edt(..., return_indices=True)``
    gives: the row and column of the nearest blocked cell for every cell.
    None when the cell is itself blocked.
    """
    cx, cy = int(cell[0]), int(cell[1])
    iy, ix = int(nearest[0][cy, cx]), int(nearest[1][cy, cx])
    vector = np.array([ix - cx, iy - cy], float)
    norm = float(np.hypot(*vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _walled_both_sides(obstacles: np.ndarray, cell: Tuple[int, int], across: Optional[np.ndarray],
                       max_cells: int) -> bool:
    """Whether observed obstacles stand within ``max_cells`` on BOTH sides of the passage through ``cell``.

    A doorway is bounded by its jambs across the passage; the inside of a
    sliver seen through a door is bounded by the unknown on one side and
    opens into the room on the other, and reads narrow only because the
    unknown hems it in. Rays are cast from the cell both ways along
    ``across`` (:func:`_across_direction`), through free and unknown
    alike, until an obstacle cell. Without a direction the test is not
    made (True).
    """
    if across is None:
        return True
    H, W = obstacles.shape
    for sign in (1.0, -1.0):
        hit = False
        for step in range(1, int(max_cells) + 1):
            x = int(round(cell[0] + sign * step * across[0]))
            y = int(round(cell[1] + sign * step * across[1]))
            if not (0 <= x < W and 0 <= y < H):
                break
            if obstacles[y, x]:
                hit = True
                break
        if not hit:
            return False
    return True


def _line_cut(shape: Tuple[int, int], cell: Tuple[int, int], radius: int, across: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """A two-cell band across the passage at ``cell``: along ``across`` (toward the jambs), ``radius`` cells each way.

    None without a direction.
    """
    if across is None:
        return None
    cx, cy = int(cell[0]), int(cell[1])
    H, W = shape
    along = np.array([-across[1], across[0]])                     # the passage direction
    cut = np.zeros(shape, bool)
    for shift in (0.0, 1.0):                                   # two cells thick along the passage
        base = np.array([cx, cy], float) + shift * along
        start = np.rint(base - radius * across).astype(int)
        end = np.rint(base + radius * across).astype(int)
        rr, cc = draw_line(int(start[1]), int(start[0]), int(end[1]), int(end[0]))
        inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        cut[rr[inside], cc[inside]] = True
    return cut


def trail_thresholds(
    free_mask: np.ndarray,
    resolution: float,
    trail_cells: Sequence[Tuple[int, int]],
    params: WatershedRoomParams = WatershedRoomParams(),
) -> List[Tuple[int, int]]:
    """The doorways a trail walked through, read off the clearance along it.

    A doorway is a LOCAL narrowing of the floor: the clearance along the
    trail dips to ``params.threshold_dip_max_m`` or less and rises by at
    least ``params.threshold_rise_m`` above the dip on BOTH sides within
    ``params.threshold_side_m`` of travel. A uniformly narrow corridor
    never dips and so is never cut (the Allensville hallway: 0.30-0.50 m
    of clearance for forty actions); a door between two rooms does. The
    dip's trail cell is returned; :func:`door_carve_mask` snaps it to the
    severing choke within ``threshold_snap_reach_m`` and carves there, so
    the room beyond the door becomes a room of its own the tick the robot
    is through it, not when its floor has grown a clearance peak.

    Args:
        free_mask: (H, W) bool, True where the grid reads free.
        resolution: Metres per cell.
        trail_cells: The robot's recent cells, oldest first, ``(cx, cy)``;
            consecutive duplicates are harmless.
        params: Tuning knobs (the ``threshold_*`` fields).

    Returns:
        The ``(cx, cy)`` trail cells at the dips, in trail order; empty
        when the trail is short or never dips.
    """
    if not resolution > 0.0:
        raise ValueError("resolution must be > 0, got %r" % (resolution,))
    cells = []
    for cx, cy in trail_cells:
        cell = (int(cx), int(cy))
        if not cells or cells[-1] != cell:
            cells.append(cell)
    if len(cells) < 3:
        return []
    H, W = free_mask.shape
    healed = heal_free_mask(free_mask)
    if not healed.any():
        return []
    dt = distance_transform_edt(healed) * float(resolution)
    clearance = [float(dt[cy, cx]) if 0 <= cy < H and 0 <= cx < W else 0.0 for cx, cy in cells]
    # Travel distance along the trail, for the side windows.
    along = [0.0]
    for (ax, ay), (bx, by) in zip(cells[:-1], cells[1:]):
        along.append(along[-1] + float(np.hypot(bx - ax, by - ay)) * float(resolution))
    out = []
    last_along = None
    for i in range(1, len(cells) - 1):
        dip = clearance[i]
        if dip <= 0.0 or dip > params.threshold_dip_max_m:
            continue
        if clearance[i - 1] < dip or clearance[i + 1] < dip:
            continue                                    # not the bottom of the dip
        if last_along is not None and along[i] - last_along <= params.threshold_dip_max_m:
            continue                                    # the same doorway's plateau through its wall
        need = dip + params.threshold_rise_m
        before = [clearance[j] for j in range(i - 1, -1, -1) if along[i] - along[j] <= params.threshold_side_m]
        after = [clearance[j] for j in range(i + 1, len(cells)) if along[j] - along[i] <= params.threshold_side_m]
        if before and after and max(before) >= need and max(after) >= need:
            out.append(cells[i])
            last_along = along[i]
    return out


def _severs(healed: np.ndarray, disk: np.ndarray, cell: Tuple[int, int], radius: int) -> bool:
    """Whether carving ``disk`` out of ``healed`` leaves the floor just around it in two or more pieces.

    Four-connected like the flood: the ring two cells beyond the disk is
    read for the components of ``healed & ~disk`` it touches, and a disk
    whose ring lies in one component parts nothing by side.
    """
    components, _ = cc_label(healed & ~disk, structure=_FOUR)
    ring = door_disk_mask(healed.shape, [cell], radius + 2) & ~disk & healed
    touched = np.unique(components[ring])
    return int(np.count_nonzero(touched > 0)) >= 2


def _basin_sides(labels: np.ndarray, wmask: np.ndarray) -> np.ndarray:
    """The connected component of the floodable mask each basin was flooded in, indexed by label.

    Four-connected like the flood itself, so a basin never straddles two
    components; entry 0 is meaningless.
    """
    components, _ = cc_label(wmask, structure=_FOUR)
    n = int(labels.max()) if labels.size else 0
    sides = np.zeros(n + 1, np.int64)
    inside = (labels > 0) & wmask
    # One pass: every basin cell writes its component; a basin lies in one
    # component, so the last write is as good as any.
    sides[labels[inside]] = components[inside]
    return sides


def snap_door_to_choke(
    skeleton: np.ndarray,
    dt: np.ndarray,
    door_cell: Tuple[int, int],
    params: WatershedRoomParams,
    resolution: float,
    leads_on: Optional[np.ndarray] = None,
    healed: Optional[np.ndarray] = None,
    obstacles: Optional[np.ndarray] = None,
    nearest: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Tuple[Tuple[int, int], float]]:
    """The choke a detected door most plausibly stands in, or None.

    A choke is a medial-axis cell within ``door_snap_reach_m`` of the door
    whose clearance lies between ``door_choke_min_m`` and
    ``door_choke_max_m`` and whose disk -- ``clearance + 1 cell`` -- parts
    the skeleton into at least two pieces that each go on for at least
    :data:`CHOKE_SIDE_CELLS` cells (a dead-end notch is narrow too, but
    the stub beyond its mouth leads nowhere) -- or, with ``leads_on`` (the
    band of cells beside the unknown, since 2026-10-21), that reach it: the
    short tail of a room glimpsed through its door leads on into the
    unexplored, where a notch's stub does not. Among the candidates the narrowest wins,
    the nearer on a tie: a doorway is the tightest structural place a door
    can stand in, and a detected door's position error runs along the
    camera ray, so distance alone is the weaker cue. The rule is a
    heuristic over the observed geometry -- a furniture passage narrower
    than the doorway within reach of the door would be taken for it -- and
    on the fifteen recorded ObjectNav floors of 2026-10-04 it changed the
    one partition it was written for and left the others within a room of
    where they were.

    Args:
        skeleton: (H, W) bool medial axis of the healed free mask.
        dt: (H, W) float clearance field in metres.
        door_cell: ``(cx, cy)`` of the detected door.
        params: Tuning knobs (reach and the half-width band).
        resolution: Metres per cell.
        leads_on: Optional (H, W) bool -- cells within a few of the unknown;
            a skeleton piece that reaches them is a side of a passage
            however short it is.
        healed: Optional (H, W) bool healed free mask; with it a candidate
            must also have a cut that PARTS THE FLOOR (:func:`_severing_cut`)
            -- the test the carve applies anyway, brought forward so that a
            corner the obstacle distance reads as narrower than the doorway
            beside it (the jamb is near diagonally) does not win the snap
            and then carve nothing.
        obstacles: Optional (H, W) bool of OBSERVED obstacle cells; with it a
            candidate must have an obstacle within ``door_choke_max_m`` on
            both sides across the passage (:func:`_walled_both_sides`): a
            doorway has its jambs, the inside of a sliver glimpsed through
            it has the unknown on one side and the room on the other.
        nearest: The nearest-blocked-cell index field of ``healed``
            (``distance_transform_edt(healed, return_indices=True)[1]``),
            which the across-the-passage direction is read from; computed
            here when ``healed`` is given without it.

    Returns:
        ``((cx, cy), half_width_m)`` of the choke, or None when the door
        has no valid choke within reach.
    """
    H, W = skeleton.shape
    cx, cy = int(door_cell[0]), int(door_cell[1])
    reach = max(1, int(np.ceil(params.door_snap_reach_m / resolution)))
    y0, y1 = max(0, cy - reach), min(H, cy + reach + 1)
    x0, x1 = max(0, cx - reach), min(W, cx + reach + 1)
    ys, xs = np.nonzero(skeleton[y0:y1, x0:x1])
    if not len(xs):
        return None
    if nearest is None and (healed is not None or obstacles is not None):
        blocked = ~healed if healed is not None else np.asarray(obstacles, bool)
        nearest = distance_transform_edt(~blocked, return_indices=True)[1]
    ys, xs = ys + y0, xs + x0
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    within = d2 <= reach * reach
    best = None  # (clearance, distance, (x, y))
    for x, y, dist2 in zip(xs[within], ys[within], d2[within]):
        clearance = float(dt[y, x])
        if not params.door_choke_min_m <= clearance <= params.door_choke_max_m:
            continue
        if best is not None and clearance > best[0] + 1e-9:
            continue
        radius = int(np.ceil(clearance / resolution)) + 1
        if not _parts_skeleton(skeleton, (int(x), int(y)), radius, leads_on, healed):
            continue
        if obstacles is not None and not _walled_both_sides(
                obstacles, (int(x), int(y)), _across_direction(nearest, (int(x), int(y))),
                int(np.ceil(params.door_choke_max_m / resolution)) + 1):
            continue
        if healed is not None and _severing_cut(healed, nearest, ((int(x), int(y)), clearance), resolution) is None:
            continue
        candidate = (clearance, float(np.sqrt(dist2)), (int(x), int(y)))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    return best[2], best[0]


#: Skeleton cells a side of a choke must carry to count as a passage's side
#: rather than a dead-end stub (1.5 m at the ObjectNav resolution).
CHOKE_SIDE_CELLS = 15
#: How many cells from the unknown a skeleton piece may end and still be read as leading on into it.
LEADS_ON_CELLS = 3


def _parts_skeleton(skeleton: np.ndarray, cell: Tuple[int, int],
                    radius: int, leads_on: Optional[np.ndarray] = None,
                    healed: Optional[np.ndarray] = None) -> bool:
    """Whether carving a disk of ``radius`` at ``cell`` leaves >= 2 substantial skeleton pieces around it.

    The pieces are read off the skeleton outside the disk: the ring just
    beyond the disk is split into its connected groups, and each group
    must belong to a component of the cut skeleton at least
    :data:`CHOKE_SIDE_CELLS` long -- a notch's stub fails, a room's or a
    corridor's spine passes -- or, with ``leads_on``, one that reaches the
    unknown: the tail of a room glimpsed through its door is short and
    leads on, a notch's stub is short and does not. A sliver too shallow
    to leave any skeleton outside the disk counts as that far side when
    the disk itself reaches the unknown's band over floor (``healed``).
    """
    shape = skeleton.shape
    disk = door_disk_mask(shape, [cell], radius)
    outside = skeleton & ~disk
    ring = door_disk_mask(shape, [cell], radius + 2) & outside
    ring_groups, n_groups = cc_label(ring, structure=_EIGHT)
    disk_leads_on = bool(leads_on is not None and healed is not None
                         and (disk & healed & np.asarray(leads_on, bool)).any())
    if n_groups < 2 and not (n_groups == 1 and disk_leads_on):
        return False
    components, _ = cc_label(outside, structure=_EIGHT)
    sizes = np.bincount(components.ravel())
    reaching = None
    if leads_on is not None:
        reaching = set(np.unique(components[np.asarray(leads_on, bool) & outside]).tolist()) - {0}
    substantial, short_leading = set(), set()
    for group in range(1, n_groups + 1):
        member = components[ring_groups == group]
        member = member[member > 0]
        if not len(member):
            continue
        component = int(member[0])
        if sizes[component] >= CHOKE_SIDE_CELLS:
            substantial.add(component)
        elif reaching is not None and component in reaching:
            short_leading.add(component)
    # A passage has a substantial side at least; the other may be the short tail of a room glimpsed
    # through the door, leading on into the unknown -- or no tail at all, the disk reaching the unknown
    # over the sliver's floor. Two short tails that both lead on are the inside of such a sliver, not a
    # doorway through it.
    return len(substantial) >= 2 or (len(substantial) == 1 and (len(short_leading) >= 1 or disk_leads_on))


def _clearance_markers(
    dt: np.ndarray,
    wmask: np.ndarray,
    params: WatershedRoomParams,
    resolution: float,
) -> Tuple[np.ndarray, int]:
    """Seed one watershed marker per local maximum of the clearance field.

    Peaks are searched inside the door-carved mask, not the healed one,
    so a seed can never land in a cell the watershed is forbidden to
    flood — such a seed would be silently dropped and could cost a real
    room its only marker.

    ``exclude_border`` is off: the flown BEV is cropped tight around the
    explored region, so rooms routinely touch the grid edge and the
    default would delete their seeds.

    Args:
        dt: (H, W) float clearance field in metres.
        wmask: (H, W) bool mask the watershed may flood.
        params: Tuning knobs.
        resolution: Metres per cell.

    Returns:
        Tuple of the (H, W) int32 marker label image and the marker
        count. Touching peaks are merged with 8-connectivity, so a
        plateau of equal clearance seeds one room, not several.
    """
    sep_cells = params.min_room_separation_m / resolution
    min_distance = max(1, int(round(sep_cells)))
    coords = peak_local_max(
        dt,
        min_distance=min_distance,
        threshold_abs=float(params.min_clearance_m),
        labels=wmask.astype(np.int32),
        exclude_border=False,
    )
    seeds = np.zeros(dt.shape, bool)
    if len(coords):
        seeds[tuple(coords.T)] = True
    markers, n_markers = cc_label(seeds, structure=_EIGHT)
    return markers.astype(np.int32), int(n_markers)


def _reclaim_carved(labels: np.ndarray, healed: np.ndarray) -> np.ndarray:
    """Give every unlabelled healed free cell -- the door carve, and any seedless floor -- to its nearest labelled room.

    The carve is a fence, not a hole: leaving it unlabelled would strand
    every doorway cell outside any room, and the node's ``room_at_cell``
    lookup would then lose the drone exactly while it flies a doorway.
    The same assignment takes in floor no marker seeded -- the sliver of a
    room glimpsed through its door, with no clearance peak of its own --
    which therefore reads as part of the room across the door until it is
    big enough to seed a marker (the openings module values such a sliver
    as the room BEHIND the door, by the objects glimpsed through it).

    Args:
        labels: (H, W) int watershed labels, 0 in the carve and outside
            the mask.
        healed: (H, W) bool healed free mask.

    Returns:
        (H, W) int labels with every healed free cell assigned, unless
        no label exists at all.
    """
    gaps = healed & (labels == 0)
    if not gaps.any() or not (labels > 0).any():
        return labels
    _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
    out = labels.copy()
    out[gaps] = labels[iy[gaps], ix[gaps]]
    return out


def _collect_rooms(
    labels: np.ndarray,
    min_room_cells: int,
    stub_cells: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[RoomStats]]:
    """Drop rooms under the size floor and relabel the rest from 1.

    Args:
        labels: (H, W) int watershed labels, 0 = no room.
        min_room_cells: Size floor in cells.
        stub_cells: Optional (H, W) bool -- the cells :func:`_stub_rooms`
            seeded beyond a cut; a label holding any of them is kept
            whatever its size (its own floor was ``stub_min_cells``).

    Returns:
        Tuple of the (H, W) int32 compact label image and the
        :class:`RoomStats` list in label order.
    """
    out = np.zeros(labels.shape, np.int32)
    stats = []  # type: List[RoomStats]
    for k in np.unique(labels):
        if k == 0:
            continue
        m = labels == k
        n = int(m.sum())
        if n < min_room_cells and not (stub_cells is not None and bool((m & stub_cells).any())):
            continue
        nid = len(stats) + 1
        out[m] = nid
        ys, xs = np.where(m)
        stats.append(RoomStats(
            id=nid, mask=m, n_cells=n,
            centroid_cells=(float(xs.mean()), float(ys.mean())),
        ))
    return out, stats


def _stub_rooms(
    labels: np.ndarray,
    wmask: np.ndarray,
    carve: np.ndarray,
    unknown_mask: Optional[np.ndarray],
    stub_min_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seed a room on every seedless piece of floor a cut parted off that leads on into the unknown.

    A seedless piece is a four-connected component of the floodable mask
    with no watershed label -- no clearance peak of its own -- that
    touches the carve (it was parted by a door or a threshold, not merely
    disconnected floor seen through a window) and, when ``unknown_mask``
    is given, borders unknown cells: the unexplored area beyond the open
    door, not a nook the disk happened to cut off a room (that nook is
    still reclaimed into its room). Pieces under ``stub_min_cells`` are
    left to the reclaim as before.

    Returns:
        ``(labels, stub_cells)``: the labels with one fresh label per stub,
        and the bool mask of the cells so labelled.
    """
    seedless = wmask & (labels == 0)
    stub_cells = np.zeros(labels.shape, bool)
    if not seedless.any():
        return labels, stub_cells
    pieces, count = cc_label(seedless, structure=_FOUR)
    if count == 0:
        return labels, stub_cells
    beside_cut = binary_dilation(carve, structure=_EIGHT.astype(bool))
    beside_unknown = None if unknown_mask is None else binary_dilation(np.asarray(unknown_mask, bool), structure=_FOUR.astype(bool))
    out = labels.copy()
    next_label = int(labels.max()) + 1
    for k in range(1, count + 1):
        piece = pieces == k
        if int(piece.sum()) < int(stub_min_cells) or not bool((piece & beside_cut).any()):
            continue
        if beside_unknown is not None and not bool((piece & beside_unknown).any()):
            continue
        out[piece] = next_label
        stub_cells |= piece
        next_label += 1
    return out, stub_cells


def sightline_thresholds(
    free_mask: np.ndarray,
    resolution: float,
    line_cells: Sequence[Tuple[int, int]],
    params: WatershedRoomParams = WatershedRoomParams(),
    unknown_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """The doorways a line of sight passes on its way to an opening, read off the clearance along it.

    The counterpart of :func:`trail_thresholds` for a sightline: from the
    viewpoint (the first cell -- the agent's pose at the moment the
    opening was detected) to the opening's threshold (the last cell, at
    the edge of the mapped floor). A doorway is a local narrowing of the
    floor BETWEEN OBSTACLES: the distance to the nearest observed
    obstacle -- occupied cells; unknown is passable for this reading,
    since the unknown ahead would otherwise make every approach to a
    frontier a narrowing -- dips to ``params.threshold_dip_max_m`` or
    less, and rises by ``params.threshold_rise_m`` above the dip on the
    VIEWPOINT side within ``params.threshold_side_m`` of travel; on the
    far side it rises likewise, or the line ends within that distance
    (what lies beyond the door is the sliver the camera saw through it).
    Without ``unknown_mask`` the clearance is to every non-free cell, as
    the trail's. The dips are returned nearest the viewpoint first;
    :func:`door_carve_mask` snaps each to the severing choke within
    ``threshold_snap_reach_m`` and carves there -- a furniture gap the
    floor goes round severs nothing and is left alone -- so the room
    glimpsed through a door separates from the room it was seen from the
    tick it is seen, not when its floor has grown a clearance peak.

    Args:
        free_mask: (H, W) bool, True where the grid reads free.
        resolution: Metres per cell.
        line_cells: The sightline's cells, viewpoint first, ``(cx, cy)``.
        params: Tuning knobs (the ``threshold_*`` fields).
        unknown_mask: (H, W) bool, True where the grid is unknown.

    Returns:
        The ``(cx, cy)`` cells at the dips, in line order; empty when the
        line is short or never narrows.
    """
    if not resolution > 0.0:
        raise ValueError("resolution must be > 0, got %r" % (resolution,))
    healed = heal_free_mask(free_mask)
    if not healed.any():
        return []
    passable = healed if unknown_mask is None else (healed | np.asarray(unknown_mask, bool))
    dt = distance_transform_edt(passable) * float(resolution)
    return sightline_dips(dt, healed, resolution, line_cells, params)


def sightline_dips(
    clearance_field: np.ndarray,
    healed: np.ndarray,
    resolution: float,
    line_cells: Sequence[Tuple[int, int]],
    params: WatershedRoomParams = WatershedRoomParams(),
) -> List[Tuple[int, int]]:
    """:func:`sightline_thresholds` over a clearance field already computed (``ChokeFields.snap_dt``) and the healed mask."""
    cells = []
    for cx, cy in line_cells:
        cell = (int(cx), int(cy))
        if not cells or cells[-1] != cell:
            cells.append(cell)
    if len(cells) < 3:
        return []
    H, W = healed.shape
    dt = clearance_field
    clearance = [float(dt[cy, cx]) if 0 <= cy < H and 0 <= cx < W and healed[cy, cx] else 0.0 for cx, cy in cells]
    along = [0.0]
    for (ax, ay), (bx, by) in zip(cells[:-1], cells[1:]):
        along.append(along[-1] + float(np.hypot(bx - ax, by - ay)) * float(resolution))
    out = []
    last_along = None
    for i in range(1, len(cells) - 1):
        dip = clearance[i]
        if dip <= 0.0 or dip > params.threshold_dip_max_m:
            continue
        if clearance[i - 1] < dip or clearance[i + 1] < dip:
            continue                                    # not the bottom of the dip
        if last_along is not None and along[i] - last_along <= params.threshold_dip_max_m:
            continue                                    # the same doorway's plateau through its wall
        need = dip + params.threshold_rise_m
        before = [clearance[j] for j in range(i - 1, -1, -1) if along[i] - along[j] <= params.threshold_side_m]
        after = [clearance[j] for j in range(i + 1, len(cells)) if along[j] - along[i] <= params.threshold_side_m]
        ends_soon = along[-1] - along[i] <= params.threshold_side_m
        if before and max(before) >= need and ((after and max(after) >= need) or ends_soon):
            out.append(cells[i])
            last_along = along[i]
    return out
