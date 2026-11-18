"""Choosing where a robot should look in a room from an occupancy grid, at a cost in actions: rays on the grid stand in for the depth image. numpy only; habitat x, z axes."""
import math
from dataclasses import dataclass

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import hits_in_room_share, near_share, visible_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import IN_ROOM_GAP_M, MIN_GAIN, NEAR_M, REPEAT_CREDIT, usable

FORWARD_STEP_M = 0.25
TURN_DEG = 30.0
RAY_STEP_FRACTION = 0.5


@dataclass(frozen=True)
class OccupancyGrid:
    free: np.ndarray
    origin_xz: tuple
    cell_m: float

    def is_free(self, x, z):
        col = np.floor((np.asarray(x) - self.origin_xz[0]) / self.cell_m).astype(int)
        row = np.floor((np.asarray(z) - self.origin_xz[1]) / self.cell_m).astype(int)
        inside = (row >= 0) & (row < self.free.shape[0]) & (col >= 0) & (col < self.free.shape[1])
        return inside & self.free[np.clip(row, 0, self.free.shape[0] - 1), np.clip(col, 0, self.free.shape[1] - 1)]


def ray_ranges(grid, camera_xz, yaw, bearings_rad, max_range_m):
    """Distance along each ray (``bearings_rad`` right of the heading) to the first cell that is not free; ``max_range_m`` if it never meets one."""
    bearings = np.asarray(bearings_rad, dtype=float)
    ahead, right = (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))
    dx = np.cos(bearings) * ahead[0] + np.sin(bearings) * right[0]
    dz = np.cos(bearings) * ahead[1] + np.sin(bearings) * right[1]
    step = grid.cell_m * RAY_STEP_FRACTION
    along = np.arange(0.0, max_range_m + step / 2, step)
    free = grid.is_free(camera_xz[0] + dx[:, None] * along[None, :], camera_xz[1] + dz[:, None] * along[None, :])
    blocked = ~free
    first = np.argmax(blocked, axis=1)
    return np.where(blocked.any(axis=1), along[first], max_range_m)


def synthetic_depth(grid, camera_xz, yaw, width, height, hfov_deg, max_range_m):
    """A depth image made from the grid: each column holds its ray's range as depth along the optical axis, and no depth where the ray finds nothing within range."""
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    bearings = np.arctan((np.arange(width) - width / 2) / focal)
    ranges = ray_ranges(grid, camera_xz, yaw, bearings, max_range_m)
    row = np.where(ranges < max_range_m, ranges * np.cos(bearings), 0.0)
    return np.tile(row, (height, 1))


def grid_candidates(grid, room_cells, poses, width, height, hfov_deg, max_range_m):
    """What each pose (floor cell, yaw) would see by rays on the grid, in the form ``view_selection.select`` reads."""
    found = []
    for cell, yaw in poses:
        depth = synthetic_depth(grid, (cell[0], cell[2]), yaw, width, height, hfov_deg, max_range_m)
        found.append({"cell": cell, "yaw": yaw, "near": near_share(depth, NEAR_M),
                      "cells": visible_cells(depth, (cell[0], cell[1], cell[2], yaw), room_cells, width, height, hfov_deg),
                      "inside": hits_in_room_share(depth, (cell[0], cell[2], yaw), room_cells[:, [0, 2]], width, hfov_deg, IN_ROOM_GAP_M)})
    return found


def heading_of(start_xz, end_xz):
    return math.atan2(-(end_xz[0] - start_xz[0]), -(end_xz[1] - start_xz[1]))


def turns_between(yaw_a, yaw_b):
    angle = abs((yaw_b - yaw_a + math.pi) % (2 * math.pi) - math.pi)
    return math.ceil(math.degrees(angle) / TURN_DEG - 1e-6)


def leg_actions(path_xz, start_yaw, end_yaw):
    """Actions to walk the path from ``start_yaw`` and end facing ``end_yaw``: a forward step per quarter metre and a turn per thirty degrees, bends included."""
    points = [path_xz[0]] + [p for a, p in zip(path_xz, path_xz[1:]) if math.dist(a, p) > 1e-6]
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    headings = [start_yaw] + [heading_of(a, b) for a, b in zip(points, points[1:])] + [end_yaw]
    return math.ceil(length / FORWARD_STEP_M - 1e-6) + sum(turns_between(a, b) for a, b in zip(headings, headings[1:]))


def entrance_pose(path_xz, in_room):
    """Where a path first reaches the room and the heading it arrives with; None if it never does."""
    for i, point in enumerate(path_xz):
        if in_room(point):
            before = path_xz[i - 1] if i > 0 else point
            after = path_xz[i + 1] if i == 0 and len(path_xz) > 1 else point
            return point, heading_of(before, point) if i > 0 else heading_of(point, after)
    return None


def plan_views(candidates, count, budget, cost_fn, min_new=MIN_GAIN, repeat_credit=REPEAT_CREDIT):
    """Up to ``count`` usable candidates in visiting order, each the one adding the most room floor among those whose leg fits the actions left (``budget`` None: no limit).

    ``cost_fn(candidate, index, previous_index)`` is the leg's actions; ``previous_index`` is None for the first leg.
    """
    pool = usable(candidates)
    if not pool:
        return []
    weights = np.full(len(candidates[pool[0]]["cells"]), 1.0 / len(candidates[pool[0]]["cells"]))
    times_seen = np.zeros(len(weights))
    chosen, previous, left = [], None, budget
    while len(chosen) < count:
        credit = weights * repeat_credit ** times_seen
        options = []
        for i in pool:
            if i in chosen:
                continue
            leg = cost_fn(candidates[i], i, previous)
            gain = float(credit[candidates[i]["cells"]].sum())
            if (left is None or leg <= left) and gain >= min_new:
                options.append((gain, -leg, -i, i, leg))
        if not options:
            break
        _, _, _, best, leg = max(options)
        chosen.append(best)
        times_seen[candidates[best]["cells"]] += 1
        previous = best
        left = None if left is None else left - leg
    return chosen
