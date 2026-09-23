"""The geometry of "stay in this room": a keep-in box and its door seals.

Pure functions, no ROS, unit-tested in the plain ``.venv``. What they produce
crosses the bridge to ``room_confine_node`` and becomes FALCON's leased
confinement.

**Why a box AND door seals, when the box alone looks sufficient.** A room's
axis-aligned bounding box is not the room: an L-shaped ward's box covers the
corridor outside its own doorway, and two rooms off the same corridor have
overlapping boxes. So the box leaks, and it leaks exactly where the doors are.
Sealing each door with a small keep-out square closes the leak at the only
place the aircraft could actually use it.

**Why the box is grown, not shrunk, and then intersected with nothing.** The
room mask is the segmenter's opinion about explored free space; the aircraft is
0.63 m wide and the planner inflates by 0.4 m. A box drawn tight to the mask
fences the aircraft out of the last half metre of its own room, and -- worse --
can exclude the very cell the aircraft is standing on the moment it arrives,
which reads to the planner as "no legal position anywhere" and stops it dead.
Measured exactly that: a fence applied while the aircraft sat just outside the
box froze it until the lease expired.

**Why the aircraft's own position is always inside the result.** Same failure,
deliberately made impossible: :func:`confine_payload` takes the drone's pose
and unions a small box around it into the keep-in list. A fence the aircraft is
outside of is not a fence, it is a trap, and the planner cannot tell the
difference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The planner tests z against these too, and the BEV is a flat slab with no
#: height of its own. A tall box means "every altitude", which is what a
#: 2D room mask actually says.
Z_MIN = -1000.0
Z_MAX = 1000.0


def room_bbox(mask: np.ndarray, resolution: float,
              origin_xy: Tuple[float, float],
              margin_m: float = 0.5) -> Optional[Tuple[float, float, float, float]]:
    """The world-frame bounding box of a room mask, grown by ``margin_m``.

    Args:
        mask: ``(H, W)`` bool, True inside the room, row 0 at minimum y.
        resolution: Metres per cell.
        origin_xy: World position of cell ``(0, 0)``.
        margin_m: How far to grow the box on every side. Grown, never shrunk:
            see the module docstring.

    Returns:
        ``(xmin, ymin, xmax, ymax)`` in world metres, or None for an empty
        mask -- which is what a renumbered room looks like, and is a reason to
        send no fence at all rather than a fence around nothing.
    """
    cells = np.asarray(mask, dtype=bool)
    if not cells.any():
        return None
    ys, xs = np.nonzero(cells)
    res = float(resolution)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    # Cell (gx, gy) spans [ox + gx*res, ox + (gx+1)*res).
    xmin = ox + float(xs.min()) * res - float(margin_m)
    xmax = ox + (float(xs.max()) + 1.0) * res + float(margin_m)
    ymin = oy + float(ys.min()) * res - float(margin_m)
    ymax = oy + (float(ys.max()) + 1.0) * res + float(margin_m)
    return (xmin, ymin, xmax, ymax)


def box3(xmin: float, ymin: float, xmax: float,
         ymax: float) -> List[float]:
    """One flat 6-value box at every altitude, as the planner parses them."""
    return [float(xmin), float(ymin), Z_MIN, float(xmax), float(ymax), Z_MAX]


def door_seals(doors: Sequence[Mapping[str, Any]],
               room_id: int,
               half_m: float = 0.9) -> List[List[float]]:
    """A keep-out square at every door this room opens through.

    ``/scene_graph`` doors carry an ``xy`` and the room pairs they join, but no
    ORIENTATION, and the planner's test is an axis-aligned box -- so a seal is
    a square, slightly over-blocking into both rooms. That is acceptable only
    because the keep-in box does the real confining and these squares exist to
    close where it leaks; a square too small to cover the jambs is a fence with
    a hole in it, and the failure is silent. Err large.

    Args:
        doors: The ``doors`` list of a ``/scene_graph`` payload.
        room_id: Only doors that touch this room are sealed. Sealing a door
            on the far side of the building would fence a route the aircraft
            legitimately needs on its way OUT once the room is done.
        half_m: Half-side of the square. 0.9 m against the hospital's 0.93 m
            doorways.

    Returns:
        One flat 6-value box per door, possibly empty.
    """
    seals = []  # type: List[List[float]]
    for door in doors or []:
        try:
            rooms = [int(r) for r in (door.get("rooms") or [])]
        except (TypeError, ValueError):
            continue
        if int(room_id) not in rooms:
            continue
        try:
            x = float(door["xy"][0])
            y = float(door["xy"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        h = float(half_m)
        seals.append(box3(x - h, y - h, x + h, y + h))
    return seals


def confine_payload(room_id: int,
                    mask: np.ndarray,
                    resolution: float,
                    origin_xy: Tuple[float, float],
                    doors: Sequence[Mapping[str, Any]],
                    drone_xy: Optional[Tuple[float, float]],
                    lease_s: float,
                    margin_m: float = 0.5,
                    door_half_m: float = 0.9,
                    drone_halo_m: float = 1.5) -> Optional[Dict[str, Any]]:
    """The full ``/scene_graph/confine`` request for one room.

    Args:
        room_id: The room to confine FALCON to.
        mask: That room's boolean mask on the BEV lattice.
        resolution: Metres per cell.
        origin_xy: World position of cell ``(0, 0)``.
        doors: The scene graph's door list, for the seals.
        drone_xy: Where the aircraft is. A halo around it is unioned into the
            keep-in list so the fence can never exclude the aircraft's own
            position -- the difference between a fence and a trap.
        lease_s: How long the fence should hold before lapsing. The caller
            must keep republishing; see ``room_confine_node``.
        margin_m: How far to grow the room box.
        door_half_m: Half-side of each door seal.
        drone_halo_m: Half-side of the box unioned around the aircraft.

    Returns:
        The request dict, or None when the mask is empty -- send nothing
        rather than a fence around nothing.
    """
    bbox = room_bbox(mask, resolution, origin_xy, margin_m=margin_m)
    if bbox is None:
        return None
    keep_in = [box3(*bbox)]
    if drone_xy is not None:
        h = float(drone_halo_m)
        x, y = float(drone_xy[0]), float(drone_xy[1])
        keep_in.append(box3(x - h, y - h, x + h, y + h))
    return {
        "room_id": int(room_id),
        "lease_s": float(lease_s),
        "keep_in": keep_in,
        "keep_out": door_seals(doors, room_id, half_m=door_half_m),
    }


def release_payload(room_id: Optional[int] = None) -> Dict[str, Any]:
    """The request that lifts the fence at once.

    An optimisation, not a requirement: falling silent lifts it one lease
    later anyway. Worth sending because the aircraft has to fly OUT through a
    door this fence is sealing, and waiting a lease to do it is wasted
    mission time.
    """
    return {
        "room_id": None if room_id is None else int(room_id),
        "lease_s": 0.0,
        "keep_in": [],
        "keep_out": [],
    }
