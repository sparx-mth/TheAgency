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

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
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
    """

    min_room_separation_m: float = 2.0
    min_clearance_m: float = 0.6
    min_room_cells: int = 150
    door_cut_m: float = 1.60
    merge_dynamics_m: float = 0.50


def segment_rooms_watershed(
    free_mask: np.ndarray,
    resolution: float,
    params: WatershedRoomParams = WatershedRoomParams(),
    door_cells: Iterable[Tuple[int, int]] = (),
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
    carve = door_disk_mask((H, W), door_cells,
                           int(round(params.door_cut_m / resolution)))
    wmask = healed & ~carve
    if not wmask.any():
        return empty_lbl, empty_sk, []

    markers, n_markers = _clearance_markers(dt, wmask, params, resolution)
    if n_markers == 0:
        return empty_lbl, empty_sk, []

    labels = watershed(-dt, markers, mask=wmask)
    labels = _reclaim_carved(labels, healed)

    # OVER-SEGMENTATION REPAIR, before the size floor rather than after:
    # a lobe the watershed split off a real room is often a runt, and
    # merging it back keeps its cells in the room instead of orphaning
    # them. ``carve`` is passed as the barrier so a listed door is never
    # merged across — the carve itself is gone by now, reclaimed above,
    # and the mask is the only surviving record of where a door was.
    labels = merge_basins_by_dynamics(labels, dt, params.merge_dynamics_m,
                                      carve)

    room_lbl, stats = _collect_rooms(labels, params.min_room_cells)
    if not stats:
        return empty_lbl, empty_sk, []

    # ``medial_axis`` breaks plateau ties with a PRNG seeded fresh on every
    # call unless one is supplied, so the spine visibly flickers between
    # otherwise identical ticks. Pin it. (``compute_rooms`` does not, and
    # there the skeleton DEFINES the rooms: five identical calls on the
    # captured BEV returned 11, 11, 11, 11 and 10 rooms.)
    skeleton = np.asarray(medial_axis(healed, rng=0), dtype=bool)
    return room_lbl, skeleton & (room_lbl > 0), stats


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
    """Give the door-carved cells back to their nearest labelled room.

    The carve is a fence, not a hole: leaving it unlabelled would strand
    every doorway cell outside any room, and the node's ``room_at_cell``
    lookup would then lose the drone exactly while it flies a doorway.

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
) -> Tuple[np.ndarray, List[RoomStats]]:
    """Drop rooms under the size floor and relabel the rest from 1.

    Args:
        labels: (H, W) int watershed labels, 0 = no room.
        min_room_cells: Size floor in cells.

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
        if n < min_room_cells:
            continue
        nid = len(stats) + 1
        out[m] = nid
        ys, xs = np.where(m)
        stats.append(RoomStats(
            id=nid, mask=m, n_cells=n,
            centroid_cells=(float(xs.mean()), float(ys.mean())),
        ))
    return out, stats
