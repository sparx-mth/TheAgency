"""RGB-D support surfaces for stairs; no simulator geometry is consulted."""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
from scipy.ndimage import label

from sparx_agency.core.planning.exploration.falcon.ordering import Deadline
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_grid import build_support_grid


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
            levels = {int(round(p[2] / 0.12)) for p in path}
            if len(path) >= 3 and len(levels) >= 3:
                self.candidates.append({"direction": sign, "path": path, "rise_m": float(rise[chosen])})

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
