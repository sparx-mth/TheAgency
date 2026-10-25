"""RGB-D support surfaces for stairs; no simulator geometry is consulted.

A candidate flight is a reachable run of measured support rising
``stair_min_rise_m`` within 5 m **whose height profile is a flight's**
(:func:`stair_profile`, since 2026-10-22): at least ``stair_min_risers``
risers of ``stair_riser_min_m`` to ``max_step_m`` each, every tread between
them ``stair_min_tread_m`` to ``stair_max_tread_m`` deep, climbing one way.
Before that any climbable chain of surfaces 0.4 m up qualified --
Corozal/000003 of the 25-episode campaign placed a "staircase" on a coffee
table and the sofa behind it, Darden/000000 one on a bed, each masking a
square metre or two of the room on the planning grid and excluding it from
the room partition. A table top, a sofa seat or a mattress is a tread half a
metre deep or more; the apron and edge of a table or a sofa's arm, seen
obliquely, rasterise as a ramp of one-cell levels -- risers of a tread's
size with no tread between them; and no piece of furniture is three
tread-sized risers tall.
"""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
from scipy.ndimage import label

from sparx_agency.core.planning.exploration.falcon.ordering import Deadline
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells
from sparx_agency.core.planning.object_search.world_model.stair_grid import build_support_grid


def _segments(path, tolerance_m=0.06):
    """Collapse a path's points into ``[(height, run_m), ...]`` -- one entry per level, with the distance walked on it.

    A level's run counts the step onto it, so one cell of a level is one
    cell's run (0.1 m at the ObjectNav resolution), two cells two: a tread
    two cells deep reads 0.2 m, a ramp of single cells 0.1 m each.
    """
    segments = []
    previous = None
    for point in path:
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        step = 0.0 if previous is None else math.hypot(x - previous[0], y - previous[1])
        if segments and abs(z - segments[-1][0]) <= tolerance_m:
            height, run, count = segments[-1]
            segments[-1] = ((height * count + z) / (count + 1), run + step, count + 1)
        else:
            segments.append((z, step, 1))
        previous = (x, y)
    return [(height, run) for height, run, _ in segments]


def stair_profile(path, sign, params):
    """``(risers, rise_m)`` of the longest flight-like stretch of ``path``, or None when there is none.

    A flight climbs one way (``sign`` +1 up, -1 down) in risers of
    ``stair_riser_min_m`` to ``max_step_m``, with a tread of
    ``stair_min_tread_m`` to ``stair_max_tread_m`` between consecutive
    risers; a level walked for longer than that is a floor, a landing or a
    piece of furniture, one shorter is the rasterised edge of something
    (a ramp of single cells), and either ends the stretch. The stretch must
    hold ``stair_min_risers`` risers and rise ``stair_min_rise_m`` in all.
    A riser smaller than the minimum is noise on one tread and merges into
    it; one larger than ``max_step_m`` is a cliff the route graph never
    joins.
    """
    levels = _segments(path)
    if len(levels) < 2:
        return None
    # Merge sub-riser steps (noise within one tread) into the tread before them.
    merged = [list(levels[0])]
    for height, run in levels[1:]:
        if abs(height - merged[-1][0]) < params.stair_riser_min_m:
            merged[-1][1] += run
        else:
            merged.append([height, run])
    best = None
    risers, rise, start_height = 0, 0.0, merged[0][0]
    for index in range(1, len(merged)):
        step = sign * (merged[index][0] - merged[index - 1][0])
        tread_ok = risers == 0 or params.stair_min_tread_m <= merged[index - 1][1] <= params.stair_max_tread_m
        if params.stair_riser_min_m <= step <= params.max_step_m and tread_ok:
            risers += 1
            rise = sign * (merged[index][0] - start_height)
        else:
            risers, rise, start_height = 0, 0.0, merged[index][0]   # a landing or furniture: the next stretch starts here
        if risers >= params.stair_min_risers and rise >= params.stair_min_rise_m and (best is None or rise > best[1]):
            best = (risers, float(rise))
    return best


class StairTerrain:
    """Bounded rolling, height-qualified surface memory and safe stair paths."""
    def __init__(self, params, resolution, body_radius, body_height, stride=6):
        self.params, self.resolution = params, resolution
        self.body_radius, self.body_height = body_radius, body_height
        self.stride = min(stride, 6)
        self.samples, self.obstacles, self.visited = {}, {}, set()
        self.world = self.routes = None
        self.candidates = []
        self._floor_world = None
        self._floor_height = None
        self._foot_height = None
        self._observed_base_height = None

    def update(self, observation, floor_world=None, floor_height=None):
        pose = observation.pose
        self._observed_base_height = pose.z
        self._floor_world, self._floor_height = floor_world, floor_height
        points, horizontal = self._surfaces(observation)
        self._foot_height = None
        support = points[horizontal]
        if len(support):
            distances = np.linalg.norm(support[:, :2] - np.array([pose.x, pose.y]), axis=1)
            near = support[(distances < self.body_radius + 0.05)
                           & (support[:, 2] <= pose.z + 0.05) & (support[:, 2] >= pose.z - 0.40)]
            if len(near) >= 6:
                self._foot_height = float(np.median(near[:, 2]))
        for point, normal in zip(points, horizontal):
            x, y, z = point
            key = (int(math.floor(x / self.resolution)), int(math.floor(y / self.resolution)), int(round(z / 0.10)))
            previous = self.samples.get(key)
            supported = bool(normal) or (previous is not None and previous[2] and abs(previous[0] - z) < 0.08)
            self.samples[key] = (float(z), observation.step, supported)
        radius = self.params.terrain_radius_m
        self.samples = {k: v for k, v in self.samples.items()
                        if abs(k[0] * self.resolution - pose.x) <= radius
                        and abs(k[1] * self.resolution - pose.y) <= radius
                        and abs(v[0] - pose.z) <= 2.0 and observation.step - v[1] <= 100}
        self._rasterize(pose)
        self.visited.add((int(math.floor(pose.x / 0.3)), int(math.floor(pose.y / 0.3)), int(round(self.support_height_m / 0.3))))
        return self.world

    def _surfaces(self, observation):
        depth = observation.depth_m[::self.stride, ::self.stride]
        camera, pose = observation.camera, observation.pose
        yy, xx = np.indices(depth.shape)
        k = camera.intrinsics
        optical = np.stack(((xx * self.stride - k.cx) / k.fx * depth,
                            (yy * self.stride - k.cy) / k.fy * depth, depth), axis=-1)
        valid = np.isfinite(depth) & (depth > camera.min_depth_m) & (depth < camera.max_depth_m)
        optical[~valid] = np.nan
        transform = world_T_camera_optical(pose, camera)
        points = optical @ transform[:3, :3].T + transform[:3, 3]
        with np.errstate(invalid="ignore"):
            normal = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
            length = np.linalg.norm(normal, axis=-1)
            horizontal = np.zeros(depth.shape, bool)
            # Image right cross image down points away from the camera:
            # visible upward treads have negative z, ceiling undersides positive.
            horizontal[1:-1, 1:-1] = (normal[..., 2] < -0.80 * length) & (length > 1e-6)
        return points[valid], horizontal[valid]

    def _rasterize(self, pose):
        build_support_grid(self, pose)
        self._candidate_paths(pose)

    def _candidate_paths(self, pose):
        self.candidates = []
        distances, _ = self.distances
        reachable = np.isfinite(distances)
        for sign in (-1, 1):
            rise = sign * (self.heights[self.routes.y, self.routes.x] - pose.z)
            useful = reachable & (rise >= self.params.stair_min_rise_m) & (distances * self.resolution < 5.0)
            indices = np.flatnonzero(useful)
            if not len(indices):
                continue
            scores = rise[indices] - distances[indices] * self.resolution * 0.12
            chosen = int(indices[np.argmax(scores)])
            cell = (int(self.routes.x[chosen]), int(self.routes.y[chosen]))
            path = self.path(cell)
            profile = stair_profile(path, sign, self.params)
            if len(path) >= 3 and profile is not None:
                self.candidates.append({"direction": sign, "path": path, "rise_m": float(rise[chosen]),
                                        "risers": profile[0], "profile_rise_m": profile[1]})

    def path(self, cell):
        cells = self.routes.path(self.start, cell, Deadline(1.0))
        return [(*self.world.grid_to_world(x, y), float(self.heights[y, x])) for x, y in cells]

    def plateau_area(self, pose):
        """Connected observed flat support near the base, not surveyed area."""
        if self.world is None or self._observed_base_height is None or abs(pose.z - self._observed_base_height) > self.params.stable_height_m:
            return 0.0
        x, y = self.world.world_to_grid(pose.x, pose.y)
        if not self.world.in_bounds(x, y) or not self.safe[y, x]:
            return 0.0
        flat = self.safe & (np.abs(self.heights - self.heights[y, x]) <= self.params.stable_height_m)
        components, _ = label(flat)
        current = components[y, x]
        return float(np.count_nonzero(components == current)) * self.resolution ** 2 if current else 0.0

    def stair_distance(self, pose):
        """Distance to reachable non-level support; walls and unknown are not stairs."""
        if self.world is None:
            return 0.0
        x, y = self.world.world_to_grid(pose.x, pose.y)
        if not self.world.in_bounds(x, y):
            return 0.0
        distances, _ = self.distances
        different = np.abs(self.heights[self.routes.y, self.routes.x] - self.heights[y, x]) > 0.16
        values = distances[different & np.isfinite(distances)] * self.resolution
        return float(values.min()) if len(values) else math.inf

    def continuation(self, pose, direction, source_height, guide=None):
        """Prefer fresh support in the climb direction, including flat landings."""
        distances, _ = self.distances
        support_height = self.heights[self.start[1], self.start[0]]
        candidates = []
        for i in np.flatnonzero(np.isfinite(distances)):
            distance = float(distances[i] * self.resolution)
            if not 0.65 <= distance <= 3.0:
                continue
            x, y = int(self.routes.x[i]), int(self.routes.y[i])
            wx, wy = self.world.grid_to_world(x, y)
            z = float(self.heights[y, x])
            if direction * (z - support_height) < -0.20:
                continue
            visited = (int(math.floor(wx / 0.3)), int(math.floor(wy / 0.3)), int(round(z / 0.3))) in self.visited
            score = 3.0 * direction * (z - source_height) + 0.12 * distance - 2.0 * visited
            if guide is not None and direction * (pose.z - guide[2]) < -0.15:
                score -= 1.5 * math.dist((wx, wy), guide[:2])
            candidates.append((score, -distance, x, y))
        for _, _, x, y in sorted(candidates, reverse=True)[:5]:
            path = self.path((x, y))
            if path:
                return path
        return []

    def steering_path(self, pose, path, action_spec):
        """Safe short segment on the actual discrete heading lattice."""
        if not path:
            return []
        step = action_spec.forward_step_m
        horizon = step + self.resolution
        goal = path[-1]
        remaining = self.routes.distances(self.world.world_to_grid(goal[0], goal[1]), Deadline(1.0))
        if remaining is None:
            return []
        options = []
        turns = int(round(180.0 / action_spec.turn_angle_deg))
        for turn in range(-turns, turns + 1):
            yaw = pose.yaw + turn * action_spec.turn_angle_rad
            facing = replace(pose, yaw=yaw)
            if not self.permits(facing, step) or not self.permits(facing, horizon):
                continue
            x, y = pose.x + horizon * math.cos(yaw), pose.y + horizon * math.sin(yaw)
            gx, gy = self.world.world_to_grid(x, y)
            z = float(self.heights[gy, gx])
            node = int(self.routes.ids[gy, gx])
            distance = remaining[0][node] if node >= 0 else np.inf
            if not np.isfinite(distance):
                continue
            score = -float(distance) * self.resolution - 0.025 * abs(turn)
            options.append((score, -abs(turn), x, y, z))
        if not options:
            return []
        _, _, x, y, z = max(options)
        return [(pose.x, pose.y, pose.z), (x, y, z)]

    def permits(self, pose, step_m):
        x, y = pose.x + step_m * math.cos(pose.yaw), pose.y + step_m * math.sin(pose.yaw)
        cells = line_cells(*self.world.world_to_grid(pose.x, pose.y), *self.world.world_to_grid(x, y))
        previous = None
        for cx, cy in cells:
            if not self.world.in_bounds(cx, cy) or not self.safe[cy, cx]:
                return False
            if previous is not None:
                px, py = previous
                if abs(self.heights[cy, cx] - self.heights[py, px]) > self.params.max_step_m:
                    return False
                if cx != px and cy != py and not (self.safe[cy, px] and self.safe[py, cx]):
                    return False
                if cx != px and cy != py and (abs(self.heights[cy, px] - self.heights[py, px]) > self.params.max_step_m
                                               or abs(self.heights[py, cx] - self.heights[py, px]) > self.params.max_step_m):
                    return False
            previous = (cx, cy)
        return True

    def blocked(self, pose, step_m):
        x = int(math.floor((pose.x + step_m * math.cos(pose.yaw)) / self.resolution))
        y = int(math.floor((pose.y + step_m * math.sin(pose.yaw)) / self.resolution))
        self.obstacles[(x, y, int(round(pose.z / 0.1)))] = True
