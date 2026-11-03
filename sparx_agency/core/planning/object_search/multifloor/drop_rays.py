"""The cells a depth ray crossed below a line under the storey: a drop measured along the ray, not only at its end.

A depth pixel that lands below ``plane - drop_m`` is a drop where it lands
(``SpawnFloorGuard._mark_drops``). It is also a drop between: from the point
its ray sinks under a line below the plane to the point it hits, the ray
travels below that line without meeting a surface, so whatever surface lies
under those cells lies lower still. Nothing else measures them. The observed
map's free-space rays run toward obstacle points 0.15-0.93 m above the plane
and carve these very cells free, and a level camera never sees a flight's
treads that close (its lowest ray is 31.5 deg down, a flight's slope about the
same): 00835-q3zU7Yy5E5s/000006 (2026-10-31) walked a frontier route down a
flight whose every cell was free on the map and none a drop, 0.59 m off the
plane in two strides, then 18 actions of ``RETREAT``.

The stretch above the line is not marked: a ray there passes over a step as
readily as over a hole. The caller (``SpawnFloorGuard._mark_drops``) puts
the line at the storey's step reach (never above its drop line), so a
crossed cell's surface is deeper than any step the storey owns; draws
crossings only from a camera above the storey plane; and counts a crossed
cell as a drop only where no surface above the drop line was ever measured:
past a stairwell a ray can run on under the floor slab beyond it.
"""
from __future__ import annotations

import numpy as np

#: Samples per map cell along a ray: two keep a diagonal crossing from skipping a cell.
SAMPLES_PER_CELL = 2


def cells_crossed_below(camera_xyz, points, line_z, origin_xy, resolution, shape):
    """``(xs, ys)`` of the grid cells the rays to ``points`` crossed below ``line_z``.

    Args:
        camera_xyz: ``(3,)`` optical centre, world ENU.
        points: ``(N, 3)`` ray endpoints, all below ``line_z``.
        line_z: The line, metres; a ray sinking under it is marked from there on.
        origin_xy: The grid's world origin ``(x, y)``.
        resolution: Metres per cell.
        shape: The grid's ``(height, width)``.

    Returns:
        Two int arrays of in-bounds cell indices, with repeats; each ray's
        endpoint cell is included.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.zeros(0, int), np.zeros(0, int)
    camera = np.asarray(camera_xyz, dtype=float)
    drop = camera[2] - points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(drop > 0, (camera[2] - float(line_z)) / drop, 0.0)
    t0 = np.clip(t0, 0.0, 1.0)
    start = camera[None, :2] + t0[:, None] * (points[:, :2] - camera[None, :2])
    span = points[:, :2] - start
    counts = np.ceil(np.hypot(span[:, 0], span[:, 1]) * SAMPLES_PER_CELL / resolution).astype(int) + 1
    ray = np.repeat(np.arange(len(points)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    fraction = (np.arange(int(counts.sum())) - first) / np.maximum(np.repeat(counts, counts) - 1, 1)
    xy = start[ray] + fraction[:, None] * span[ray]
    xs = np.floor((xy[:, 0] - origin_xy[0]) / resolution).astype(int)
    ys = np.floor((xy[:, 1] - origin_xy[1]) / resolution).astype(int)
    inside = (xs >= 0) & (xs < shape[1]) & (ys >= 0) & (ys < shape[0])
    return xs[inside], ys[inside]
