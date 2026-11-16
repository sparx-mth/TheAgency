"""Which floor cells belong to which region, where to stand in a room, which way to look. numpy only; habitat axes (y up)."""
import collections
import math

import numpy as np

UNASSIGNED = -1
NEAREST_OBJECTS = 3
MAX_OBJECT_HEIGHT_ABOVE_FLOOR_M = 3.0


def assign_cells(cells_xyz, objects, max_gap_m):
    """Region of each floor cell: the majority among its nearest objects on the same floor, ``UNASSIGNED`` if none is within ``max_gap_m``."""
    centers = np.array([o["center"] for o in objects], dtype=float).reshape(-1, 3)
    regions = [o["region"] for o in objects]
    found = []
    for cell in np.asarray(cells_xyz, dtype=float):
        height = centers[:, 1] - cell[1]
        on_floor = np.flatnonzero((height >= 0) & (height <= MAX_OBJECT_HEIGHT_ABOVE_FLOOR_M))
        distance = np.hypot(centers[on_floor, 0] - cell[0], centers[on_floor, 2] - cell[2])
        nearest = on_floor[np.argsort(distance)[:NEAREST_OBJECTS]]
        if len(nearest) == 0 or distance.min() > max_gap_m:
            found.append(UNASSIGNED)
            continue
        found.append(collections.Counter(regions[i] for i in nearest).most_common(1)[0][0])
    return np.array(found)


def spread_points(points_xz, count):
    """Indices of up to ``count`` points: the one nearest the middle, then each time the point farthest from those chosen."""
    points = np.asarray(points_xz, dtype=float)
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    while len(chosen) < min(count, len(points)):
        gap = np.min(np.linalg.norm(points[:, None] - points[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(gap)))
    return chosen


def compass_yaws(count):
    return [2 * math.pi * i / count for i in range(count)]


def in_box_share(depth_m, camera_xz_yaw, box_xz, width, height, hfov_deg, margin_m=0.3, stride=8):
    """The share of the valid depth pixels whose floor-plan point lies inside the box (x low, z low, x high, z high) grown by ``margin_m``."""
    sampled = np.asarray(depth_m, dtype=float)[::stride, ::stride]
    columns = np.arange(0, width, stride)[:sampled.shape[1]]
    valid = np.isfinite(sampled) & (sampled > 0)
    sampled = np.where(valid, sampled, 0.0)
    metres = sampled[valid]
    sideways = ((columns[None, :] - width / 2) / ((width / 2) / math.tan(math.radians(hfov_deg) / 2)) * sampled)[valid]
    if not metres.size:
        return 0.0
    cam_x, cam_z, yaw = camera_xz_yaw
    ahead, right = (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))
    x = cam_x + metres * ahead[0] + sideways * right[0]
    z = cam_z + metres * ahead[1] + sideways * right[1]
    x0, z0, x1, z1 = box_xz
    return float(((x >= x0 - margin_m) & (x <= x1 + margin_m) & (z >= z0 - margin_m) & (z <= z1 + margin_m)).mean())
