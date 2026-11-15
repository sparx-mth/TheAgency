"""FALCON IV-C / frontier_finder correspondence; independently written planar port.

Frontier boundaries update with every observation. Only changed cells and their
neighbours need boundary recomputation; CCL and PCA splits use the current mask.
Qualified candidates retain both predicted unknown gain and visible frontiers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import binary_dilation, label

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.falcon.travel import segment_clear
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.objnav.types.pose import AgentPose


@dataclass(frozen=True)
class Viewpoint:
    frontier: int
    cell: tuple
    yaw: float
    gain: int
    visible: int


class FrontierMemory:
    def __init__(self, params):
        self.params = params
        self.previous = self.boundary = None
        self.clusters = {}
        self._next_id = 0
        self.revision = 0
        self.updates = 0

    def update(self, grid):
        """Maintain free/unknown boundaries, then split large components by PCA."""
        free, unknown = grid == 0, grid == -1
        adjacent = binary_dilation(unknown, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        new = free & adjacent
        changed = np.ones(grid.shape, bool) if self.previous is None else grid != self.previous
        if changed.any():
            dirty = binary_dilation(changed)
            if self.boundary is None:
                self.boundary = new
            else:
                self.boundary[dirty] = new[dirty]
            self.previous = grid.copy()
            self.revision += 1
        self.updates += 1
        return self.boundary

    def in_scope(self, scope, resolution, deadline):
        labels, count = label(self.boundary & scope, structure=np.ones((3, 3), np.uint8))
        groups = []
        for k in range(1, count + 1):
            deadline.check()
            ys, xs = np.nonzero(labels == k)
            if len(xs) >= self.params.min_cluster_cells:
                self._split(np.column_stack((xs, ys)), resolution, groups)
        old, result, used = self.clusters, {}, set()
        for cells in groups:
            indices = set(map(tuple, cells.tolist()))
            matches = [(len(indices & previous), key) for key, previous in old.items() if key not in used]
            overlap, key = max(matches, default=(0, -1))
            if overlap == 0:
                key, self._next_id = self._next_id, self._next_id + 1
            used.add(key)
            result[key] = cells
        self.clusters = {key: set(map(tuple, cells.tolist())) for key, cells in result.items()}
        return result

    def _split(self, cells, resolution, output):
        delta = cells - cells.mean(axis=0)
        if len(cells) < 2 * self.params.min_cluster_cells or np.max(np.linalg.norm(delta, axis=1)) * resolution <= self.params.cluster_radius_m:
            output.append(cells)
            return
        _, axes = np.linalg.eigh(delta.T @ delta / len(cells))
        direction = axes[:, -1]
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        side = delta @ direction >= 0
        if side.all() or not side.any():
            output.append(cells)
            return
        for group in (cells[side], cells[~side]):
            self._split(group, resolution, output)


class CameraVisibility:
    """A body-height column proxy viewed through the actual registered camera.

    Known occupied columns occlude all rays (conservative 2.5D projection).
    Predicted unknown cells behind unknown cells are speculative information,
    not free evidence. Nothing here writes the observation map.
    """

    def __init__(self, world, camera, pitch, body_height, params):
        self.world, self.camera, self.pitch = world, camera, pitch
        self.height = min(camera.height_m, body_height)
        self.params = params

    def in_frustum(self, origin, yaw, cells):
        if len(cells) == 0:
            return np.zeros(0, bool)
        x, y = self.world.grid_to_world(*origin)
        pose = AgentPose(x, y, 0.0, yaw, camera_pitch=self.pitch)
        transform = np.linalg.inv(world_T_camera_optical(pose, self.camera))
        origin_xy = self.world.grid_to_world(0, 0)
        points = np.ones((len(cells), 4))
        points[:, :2] = np.asarray(cells) * self.world.resolution + origin_xy
        points[:, 2] = self.height
        optical = points @ transform.T
        z = optical[:, 2]
        k = self.camera.intrinsics
        valid = (z > self.camera.min_depth_m) & (z < self.camera.max_depth_m)
        u = k.fx * optical[:, 0] / np.maximum(z, 1e-8) + k.cx
        v = k.fy * optical[:, 1] / np.maximum(z, 1e-8) + k.cy
        return valid & (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)

    def visible_frontier(self, origin, yaw, cells):
        selected = cells[self.in_frustum(origin, yaw, cells)]
        free = self.world.grid == self.world.values.free
        return sum(segment_clear(free, origin, point) for point in selected)

    def gain(self, origin, yaw, scope):
        k, world = self.camera.intrinsics, self.world
        # Pinhole left/right angles, not an omnidirectional range sensor.
        angles = np.linspace(-math.atan((k.width - k.cx) / k.fx), math.atan(k.cx / k.fx), self.params.visibility_rays)
        unknown = set()
        for angle in angles:
            radius = self.camera.max_depth_m / max(0.1, math.cos(angle)) / world.resolution
            end = (int(round(origin[0] + radius * math.cos(yaw + angle))),
                   int(round(origin[1] + radius * math.sin(yaw + angle))))
            for cell in line_cells(*origin, *end):
                x, y = cell
                if not world.in_bounds(x, y) or world.grid[y, x] == world.values.occupied:
                    break
                if scope[y, x] and world.grid[y, x] == world.values.unknown:
                    unknown.add(cell)
        cells = np.array(sorted(unknown), dtype=int).reshape(-1, 2)
        return int(np.count_nonzero(self.in_frustum(origin, yaw, cells)))


def sample_viewpoints(clusters, world, safe, scope, start, camera, params, used, deadline,
                      start_yaw=0.0, turn_angle=math.pi / 6, arrival_m=0.2):
    """Circle samples → visible-frontier gate → gain z-score → top alternatives."""
    result, dormant = {}, 0
    for fid, cells in sorted(clusters.items()):
        deadline.check()
        center = cells.mean(axis=0)
        samples = {tuple(start)}
        for radius in params.candidate_radii_m:
            for angle in np.linspace(-math.pi, math.pi, params.candidate_angles, endpoint=False):
                sample = center + radius / world.resolution * np.array([math.cos(angle), math.sin(angle)])
                samples.add(tuple(map(int, np.rint(sample))))
        candidates = []
        for x, y in sorted(samples):
            deadline.check()
            if not world.in_bounds(x, y) or not safe[y, x] or not scope[y, x]:
                continue
            bearings = np.arctan2(cells[:, 1] - y, cells[:, 0] - x)
            yaw = math.atan2(float(np.sin(bearings).mean()), float(np.cos(bearings).mean()))
            yaw = normalize_angle(start_yaw + round(normalize_angle(yaw - start_yaw) / turn_angle) * turn_angle)
            if math.dist((x, y), start) * world.resolution <= arrival_m and abs(normalize_angle(yaw - start_yaw)) < turn_angle / 2:
                continue  # the current observation already is this discrete view
            if used((x, y), yaw):
                continue
            visible = camera.visible_frontier((x, y), yaw, cells)
            if visible == 0:
                continue
            gain = camera.gain((x, y), yaw, scope)
            if gain:
                candidates.append(Viewpoint(fid, (x, y), yaw, gain, visible))
        if not candidates:
            dormant += 1
            continue
        gains = np.array([view.gain for view in candidates])
        # Source config uses mean + z*sd with z=0. Keep ties, unlike source's
        # strict comparison + all-empty bailout (same accepted equal-gain set).
        cutoff = gains.mean() + params.viewpoint_z_score * gains.std()
        qualified = [v for v in candidates if v.gain >= cutoff]
        if not qualified:
            dormant += 1
            continue
        qualified.sort(key=lambda v: (-v.visible, -v.gain, v.cell, v.yaw))
        result[fid] = [v for v in qualified if v.visible >= qualified[0].visible * params.view_decay][:params.top_viewpoints]
    return result, dormant
