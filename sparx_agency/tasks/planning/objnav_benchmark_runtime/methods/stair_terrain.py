"""Local RGB-D support surfaces for stairs; no simulator geometry is consulted.

Organized depth normals distinguish treads from walls. Samples are rasterized
at measured heights, connected only across embodiment-sized steps, and pruned
by body-height obstacles. This transient height map is never a semantic floor
map: room/landmark state remains in the persistent floor contexts.
"""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter, label

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.falcon.ordering import Deadline
from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells


class StairTerrain:
    """Bounded rolling surface memory, with height-aware shortest paths."""

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
        # Surface heights are retained per (x,y,z-bin): overlapping storeys
        # cannot overwrite each other. Only nearby-height support is projected.
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
        self.visited.add((int(math.floor(pose.x / 0.3)), int(math.floor(pose.y / 0.3)), int(round(pose.z / 0.3))))
        self._rasterize(pose)
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
            normal = np.cross(points[1:-1, 2:] - points[1:-1, :-2],
                              points[2:, 1:-1] - points[:-2, 1:-1])
            length = np.linalg.norm(normal, axis=-1)
            horizontal = np.zeros(depth.shape, bool)
            # Image right × image down points away from the camera: a visible
            # upward-facing tread has negative z, a ceiling underside positive.
            horizontal[1:-1, 1:-1] = (normal[..., 2] < -0.80 * length) & (length > 1e-6)
        return points[valid], horizontal[valid]

    def _rasterize(self, pose):
        r = self.resolution
        half = int(math.ceil(self.params.terrain_radius_m / r))
        cx, cy = int(math.floor(pose.x / r)), int(math.floor(pose.y / r))
        self.offset = (cx - half, cy - half)
        n = 2 * half + 1
        heights = np.full((n, n), np.nan)
        # Keep the upper visible upward-facing support. Choosing the closest
        # height to the base can instead pick free space beneath a stair tread.
        for (x, y, _), (z, _, horizontal) in self.samples.items():
            x, y = x - self.offset[0], y - self.offset[1]
            if horizontal and 0 <= x < n and 0 <= y < n and abs(z - pose.z) <= 1.5:
                old = heights[y, x]
                if not np.isfinite(old) or z > old:
                    heights[y, x] = z
        planar_obstacles = np.zeros(heights.shape, bool)
        world = self._floor_world
        if world is not None and self._floor_height is not None and abs(pose.z - self._floor_height) < self.params.floor_match_m:
            yy, xx = np.indices(heights.shape)
            wx, wy = (xx + self.offset[0] + 0.5) * r, (yy + self.offset[1] + 0.5) * r
            gx = np.floor((wx - world.origin_x) / world.resolution).astype(int)
            gy = np.floor((wy - world.origin_y) / world.resolution).astype(int)
            inside = (gx >= 0) & (gx < world.width) & (gy >= 0) & (gy < world.height)
            close = (wx - pose.x) ** 2 + (wy - pose.y) ** 2 < 2.0 ** 2
            iy, ix = np.nonzero(inside & close)
            values = world.grid[gy[iy, ix], gx[iy, ix]]
            free = (values == world.values.free) & ~np.isfinite(heights[iy, ix])
            heights[iy[free], ix[free]] = self._floor_height - 0.05
            occupied = values == world.values.occupied
            planar_obstacles[iy[occupied], ix[occupied]] = True
        # Fill only sub-cell sampling holes supported on opposing sides.
        known = np.isfinite(heights)
        if known.any():
            low = minimum_filter(np.where(known, heights, np.inf), size=3)
            high = maximum_filter(np.where(known, heights, -np.inf), size=3)
            enclosed = np.zeros_like(known)
            enclosed[1:-1, 1:-1] = ((known[1:-1, :-2] & known[1:-1, 2:])
                                    | (known[:-2, 1:-1] & known[2:, 1:-1]))
            fill = ~known & enclosed & (high - low < self.params.max_step_m)
            heights[fill] = (high[fill] + low[fill]) * 0.5
        # The actual occupied base footprint is observed support, not a future
        # destination carved free. Navmesh bases can sit slightly above scans.
        yy, xx = np.indices(heights.shape)
        footprint = (xx - half) ** 2 + (yy - half) ** 2 <= (self.body_radius / r) ** 2
        foot = self._foot_height
        if foot is None:
            observed = heights[half, half]
            foot = float(observed) if np.isfinite(observed) and pose.z - 0.40 <= observed <= pose.z + 0.05 else pose.z - 0.05
        heights[footprint] = foot
        # A measured lower tread may have been an obstacle in the source
        # planar slab. Only retain that slab obstacle at near-level height.
        obstacles = planar_obstacles & (np.abs(heights - pose.z) < 0.08)
        for (x, y, _), (z, _, horizontal) in self.samples.items():
            x, y = x - self.offset[0], y - self.offset[1]
            if 0 <= x < n and 0 <= y < n and np.isfinite(heights[y, x]):
                above = z - heights[y, x]
                if 0.28 < above < self.body_height + 0.10:
                    obstacles[y, x] = True
        for x, y in self.obstacles:
            x, y = x - self.offset[0], y - self.offset[1]
            if 0 <= x < n and 0 <= y < n:
                obstacles[y, x] = True
        known = np.isfinite(heights)
        # Unknown support remains blocked. One-cell margin admits measured
        # narrow treads; wall clearance still uses the full physical radius.
        clearance = distance_transform_edt(~obstacles) * r if obstacles.any() else np.full_like(heights, np.inf)
        safe = known & ~obstacles & (clearance > self.body_radius)
        safe[half, half] = True
        data = np.full(heights.shape, -1, np.int8)
        data[safe] = 0
        data[obstacles] = 100
        self.heights, self.safe = heights, safe
        self.world = OccupancyGrid2D(data, OccupancyGrid2DParams(r, self.offset[0] * r, self.offset[1] * r, "world"),
                                     values=OccupancyValues(free=0, occupied=100, unknown=-1))
        self.routes = GridRoutes(np.where(safe, 1.0, np.inf), r)
        graph = self.routes.graph.tocoo()
        ay, ax = self.routes.y[graph.row], self.routes.x[graph.row]
        by, bx = self.routes.y[graph.col], self.routes.x[graph.col]
        valid = np.abs(heights[ay, ax] - heights[by, bx]) <= self.params.max_step_m
        # Diagonal sides must also be step-compatible, not just XY-free.
        valid &= np.abs(heights[ay, bx] - heights[ay, ax]) <= self.params.max_step_m
        valid &= np.abs(heights[by, ax] - heights[ay, ax]) <= self.params.max_step_m
        graph.data[~valid] = 0
        graph.eliminate_zeros()
        self.routes.graph = graph.tocsr()
        # Preserve tread-by-tread routes; planar any-angle shortening could
        # otherwise jump a height discontinuity at an overlapping stairwell.
        self.routes.shorten = lambda cells, deadline: tuple(cells)
        self.start = (half, half)
        self.distances = self.routes.distances(self.start, Deadline(1.0))
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
            if len(path) >= 3:
                self.candidates.append({"direction": sign, "path": path, "rise_m": float(rise[chosen])})

    def path(self, cell):
        cells = self.routes.path(self.start, cell, Deadline(1.0))
        return [(*self.world.grid_to_world(x, y), float(self.heights[y, x])) for x, y in cells]

    def plateau_area(self, pose):
        """Observed connected flat support near the current base, never GT area."""
        if self.world is None or self._observed_base_height is None or abs(pose.z - self._observed_base_height) > self.params.stable_height_m:
            return 0.0
        x, y = self.world.world_to_grid(pose.x, pose.y)
        if not self.world.in_bounds(x, y) or not self.safe[y, x]:
            return 0.0
        flat = self.safe & (np.abs(self.heights - self.heights[y, x]) <= self.params.stable_height_m)
        components, _ = label(flat)
        current = components[y, x]
        return float(np.count_nonzero(components == current)) * self.resolution ** 2 if current else 0.0

    def continuation(self, pose, direction, source_height):
        """Prefer fresh support in the climb direction, including flat landings."""
        distances, _ = self.distances
        candidates = []
        for i in np.flatnonzero(np.isfinite(distances)):
            distance = float(distances[i] * self.resolution)
            if not 0.65 <= distance <= 3.0:
                continue
            x, y = int(self.routes.x[i]), int(self.routes.y[i])
            wx, wy = self.world.grid_to_world(x, y)
            z = float(self.heights[y, x])
            if direction * (z - pose.z) < -0.20:
                continue
            visited = (int(math.floor(wx / 0.3)), int(math.floor(wy / 0.3)), int(round(z / 0.3))) in self.visited
            score = 3.0 * direction * (z - source_height) + 0.12 * distance - 2.0 * visited
            candidates.append((score, -distance, x, y))
        for _, _, x, y in sorted(candidates, reverse=True)[:5]:
            path = self.path((x, y))
            if path:
                return path
        return []

    def steering_path(self, pose, path, action_spec):
        """A short safe segment on the actual discrete heading lattice.

        A continuous grid path can be safe while its half-step lookahead cuts
        a narrow stair corner. The next step and its arrival margin are checked;
        the policy still sends geometry through the standard action converter.
        """
        if not path:
            return []
        step = action_spec.forward_step_m
        horizon = step + self.resolution
        goal = path[-1]
        goal_cell = self.world.world_to_grid(goal[0], goal[1])
        remaining = self.routes.distances(goal_cell, Deadline(1.0))
        if remaining is None:
            return []
        options = []
        turns = int(round(180.0 / action_spec.turn_angle_deg))
        for turn in range(-turns, turns + 1):
            yaw = pose.yaw + turn * action_spec.turn_angle_rad
            facing = replace(pose, yaw=yaw)
            if not self.permits(facing, step):
                continue
            if not self.permits(facing, horizon):
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
            previous = (cx, cy)
        return True

    def blocked(self, pose, step_m):
        x = int(math.floor((pose.x + step_m * math.cos(pose.yaw)) / self.resolution))
        y = int(math.floor((pose.y + step_m * math.sin(pose.yaw)) / self.resolution))
        self.obstacles[(x, y)] = True

