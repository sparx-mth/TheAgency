# core/mapping/topology/room_segmentation.py
"""Grid-based room segmentation: medial-axis skeleton cut at doors.

This is the pipeline that actually flew in the SJTU hospital sim (ported
from the old stack's ``semantic_mapper_node.py``). It works directly on
an occupancy grid:

    free mask ─► heal (binary closing + opening, 1 iteration each)
              ─► medial-axis skeleton (DT-ridge fallback without skimage)
              ─► punch a disk of ``door_cut_cells`` radius through the
                 skeleton at every discovered door cell
              ─► 8-connected label the cut skeleton
              ─► paint each free cell with the label of its nearest
                 skeleton pixel (EDT with ``return_indices``)
              ─► drop rooms under ``min_room_cells``

Relationship to the sibling ``room_separation.py``: that module is the
graph-based Gaussian-door-field implementation from MORE (Werby et al.,
2025) operating on a Voronoi navigation graph, and it is untested in
flight; this module is the grid-based pipeline that flew. Both split
rooms at doors — callers choose which to use.

Coordinate convention: grids are indexed ``[y, x]`` (row, col); door
cells are ``(cx, cy)`` = (col, row) pairs, matching the flown source.

Dependencies: numpy, scipy; scikit-image if available (topology is a
host-owned path, so scipy/skimage are allowed here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt
from scipy.ndimage import label as cc_label

try:
    from skimage.morphology import medial_axis
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover - venv ships skimage
    _HAS_SKIMAGE = False


# OccupancyGrid value semantics (nav_msgs standard, as flown):
# -1 unknown, 0..49 free, >=50 occupied.
UNKNOWN = -1
FREE_MAX = 49
OCC_MIN = 50


def heal_free_mask(free_mask: np.ndarray) -> np.ndarray:
    """Close pinhole noise in a free mask, then shave its slivers.

    One iteration of 3x3 binary closing followed by one of binary
    opening. Closing fills the single-cell holes a noisy depth map
    punches into free space (each of which would otherwise fork the
    medial axis around it); opening removes the one-cell whiskers that
    closing grows off wall faces.

    Shared by :func:`compute_rooms` and ``room_watershed``'s
    :func:`segment_rooms_watershed` so the two segmenters start from
    the identical mask and their outputs stay comparable cell for cell.

    Args:
        free_mask: (H, W) bool, True where the grid reads free.

    Returns:
        (H, W) bool healed mask. May be empty even when the input is
        not: opening deletes free regions thinner than one cell.
    """
    return binary_opening(binary_closing(free_mask, iterations=1),
                          iterations=1)


def door_disk_mask(
    shape: Tuple[int, int],
    door_cells: Iterable[Tuple[int, int]],
    radius_cells: int,
) -> np.ndarray:
    """Union of the disks that doors punch through a grid.

    Args:
        shape: ``(H, W)`` of the grid the disks are stamped into.
        door_cells: Door positions as ``(cx, cy)`` = (col, row) pairs.
            A door whose CENTRE is out of bounds is ignored entirely,
            even if its disk would have overlapped the grid — that is
            the flown behaviour and callers rely on it.
        radius_cells: Disk radius in cells; clamped to at least 1.

    Returns:
        (H, W) bool, True inside any door disk.
    """
    H, W = shape
    r = max(1, int(radius_cells))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    out = np.zeros((H, W), bool)
    for dcx, dcy in door_cells:
        if not (0 <= dcx < W and 0 <= dcy < H):
            continue
        y0, y1 = max(0, dcy - r), min(H, dcy + r + 1)
        x0, x1 = max(0, dcx - r), min(W, dcx + r + 1)
        dy0, dx0 = y0 - (dcy - r), x0 - (dcx - r)
        out[y0:y1, x0:x1] |= disk[dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)]
    return out


@dataclass(frozen=True)
class RoomSegmentationParams:
    """Tuning knobs for :func:`compute_rooms`.

    Attributes:
        door_cut_cells: Radius (cells) of the disk punched through the
            skeleton at each door. The flown default was 0.60 m at
            0.15 m/cell resolution, i.e. 4 cells.
        min_room_cells: Rooms smaller than this many cells are
            discarded. At 0.15 m resolution, 40 cells is roughly
            0.9 m^2 — small storage closets still qualify.
    """

    door_cut_cells: int = 4
    min_room_cells: int = 40


@dataclass(frozen=True)
class RoomStats:
    """One segmented room, in fresh (per-tick) label order.

    Attributes:
        id: Fresh label, 1..N, contiguous within one tick. Not stable
            across ticks — use ``RoomRegistry`` for persistent ids.
        mask: (H, W) bool membership mask.
        n_cells: Number of cells in the mask.
        centroid_cells: Mask centroid as ``(cx, cy)`` = (col, row).
    """

    id: int
    mask: np.ndarray
    n_cells: int
    centroid_cells: Tuple[float, float]


def compute_rooms(
    free_mask: np.ndarray,
    door_cells: Iterable[Tuple[int, int]],
    params: RoomSegmentationParams = RoomSegmentationParams(),
) -> Tuple[np.ndarray, np.ndarray, List[RoomStats]]:
    """Segment free space into rooms by cutting its skeleton at doors.

    The algorithm is agnostic about where the free mask came from.

    Args:
        free_mask: (H, W) bool, True where the grid reads free.
        door_cells: Discovered door positions as ``(cx, cy)`` cell
            pairs. Out-of-bounds doors are ignored.
        params: Tuning knobs.

    Returns:
        Tuple of:
            room_lbl: (H, W) int32 — 0 = not-a-room, 1..N = rooms
                (fresh labels, contiguous).
            skeleton: (H, W) bool — the cut skeleton restricted to
                surviving rooms.
            stats: One :class:`RoomStats` per room, in label order.
    """
    H, W = free_mask.shape
    empty_lbl = np.zeros((H, W), np.int32)
    empty_sk = np.zeros((H, W), bool)
    if not free_mask.any():
        return empty_lbl, empty_sk, []

    # Heal pinhole noise so the skeleton does not fork unnecessarily.
    fm = heal_free_mask(free_mask)
    if not fm.any():
        return empty_lbl, empty_sk, []

    # 1. Medial-axis skeleton of free space.
    skel = (medial_axis(fm).astype(bool) if _HAS_SKIMAGE
            else _ridge_fallback(fm))

    # 2. Punch a disk through the skeleton at every discovered door.
    sk_cut = skel & ~door_disk_mask((H, W), door_cells, params.door_cut_cells)

    # 3. 8-connected components on the cut skeleton. (4-connectivity
    #    would fragment the diagonal medial-axis ridges into hundreds
    #    of bogus components.)
    sk_lbl, n_lbl = cc_label(sk_cut, structure=np.ones((3, 3), np.uint8))
    if n_lbl == 0:
        return empty_lbl, sk_cut, []

    # 4. Paint each free cell with the label of its nearest skeleton
    #    pixel. That is the Voronoi diagram of the cut skeleton clipped
    #    to free space.
    _, (iy, ix) = distance_transform_edt(~sk_cut, return_indices=True)
    room_lbl = np.where(fm, sk_lbl[iy, ix], 0).astype(np.int32)

    # 5. Drop rooms smaller than the threshold; remap surviving ids to
    #    a contiguous 1..N for downstream.
    out = np.zeros_like(room_lbl)
    stats = []  # type: List[RoomStats]
    for k in range(1, n_lbl + 1):
        m = room_lbl == k
        n = int(m.sum())
        if n < params.min_room_cells:
            continue
        nid = len(stats) + 1
        out[m] = nid
        ys, xs = np.where(m)
        stats.append(RoomStats(
            id=nid, mask=m, n_cells=n,
            centroid_cells=(float(xs.mean()), float(ys.mean())),
        ))

    keep_sk = np.zeros_like(sk_cut)
    for s in stats:
        keep_sk |= sk_cut & s.mask
    return out, keep_sk, stats


def _ridge_fallback(fm: np.ndarray) -> np.ndarray:
    """DT-ridge skeleton fallback when scikit-image is not available.

    Marks interior cells whose distance-transform value is a local
    maximum along the x or y axis — a coarse but serviceable stand-in
    for the medial axis.

    Args:
        fm: (H, W) bool healed free mask.

    Returns:
        (H, W) bool skeleton.
    """
    dt = distance_transform_edt(fm)
    sk = np.zeros_like(fm, bool)
    c = dt[1:-1, 1:-1]
    rx = (c >= dt[1:-1, :-2]) & (c >= dt[1:-1, 2:])
    ry = (c >= dt[:-2, 1:-1]) & (c >= dt[2:, 1:-1])
    sk[1:-1, 1:-1] = (c > 0.5) & (rx | ry)
    return sk
