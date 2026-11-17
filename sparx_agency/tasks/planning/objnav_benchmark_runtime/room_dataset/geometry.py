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


def visible_cells(depth_m, camera_xyz_yaw, cells_xyz, width, height, hfov_deg, tolerance_m=0.3):
    """Which floor cells lie in front of the camera: inside its horizontal field of view and not behind a nearer surface at the horizon row; no depth there counts as blocked."""
    cam_x, _, cam_z, yaw = camera_xyz_yaw
    cells = np.asarray(cells_xyz, dtype=float).reshape(-1, 3)
    ahead, right = (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))
    dx, dz = cells[:, 0] - cam_x, cells[:, 2] - cam_z
    forward = dx * ahead[0] + dz * ahead[1]
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    in_front = forward > 0.1
    column = np.round(width / 2 + focal * (dx * right[0] + dz * right[1]) / np.where(in_front, forward, 1.0)).astype(int)
    inside = in_front & (column >= 0) & (column < width)
    surface = np.asarray(depth_m, dtype=float)[height // 2, np.clip(column, 0, width - 1)]
    return inside & np.isfinite(surface) & (surface > 0) & (surface >= forward - tolerance_m)


def greedy_views(seen_masks, count, min_new, weights=None, repeat_credit=0.0):
    """Indices of up to ``count`` views, each adding the most weight; an element seen ``k`` times before counts ``repeat_credit ** k``. Stops below ``min_new``."""
    weights = np.ones(len(seen_masks[0])) if weights is None else np.asarray(weights, dtype=float)
    times_seen = np.zeros(len(weights))
    chosen = []
    while len(chosen) < count:
        credit = weights * repeat_credit ** times_seen
        gains = [-1.0 if i in chosen else float(credit[mask].sum()) for i, mask in enumerate(seen_masks)]
        best = int(np.argmax(gains))
        if gains[best] < min_new:
            break
        chosen.append(best)
        times_seen[seen_masks[best]] += 1
    return chosen


def object_coverage(detections_per_view, radius_m):
    """The distinct objects over all views (same class within ``radius_m`` is one object) and, per view, which of them it shows."""
    objects = []
    for detections in detections_per_view:
        for d in detections:
            if not any(o["cls"] == d["cls"] and math.dist(o["point"], d["point"]) <= radius_m for o in objects):
                objects.append(d)
    masks = [np.array([any(o["cls"] == d["cls"] and math.dist(o["point"], d["point"]) <= radius_m for d in detections) for o in objects], dtype=bool)
             for detections in detections_per_view]
    return objects, masks


def near_room(point_xz, room_cells_xyz, max_gap_m):
    """Whether a floor-plan point has a room cell within ``max_gap_m``: the object stands in the room."""
    cells = np.asarray(room_cells_xyz, dtype=float)
    return bool(np.hypot(cells[:, 0] - point_xz[0], cells[:, 2] - point_xz[1]).min() <= max_gap_m)


def near_share(depth_m, near_m):
    """The share of pixels closer than ``near_m`` or without depth (nearer than the sensor sees): high for a wall or door in the camera's face."""
    depth = np.asarray(depth_m, dtype=float)
    return float((~np.isfinite(depth) | (depth <= 0) | (depth < near_m)).mean())
