"""Observed-only RGB-D mapping using the existing core log-odds integrator."""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.mapping.costmap.depth_to_grid import update_grid_from_depth
from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridConfig, LogOddsGridCostmap
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical, backproject_depth


class ObservedMap:
    """A fresh map per episode, centred on its first observed position.

    This is the method's map, never a benchmark scoring map. Height filtering is
    relative to the first observed floor, matching the current planar search
    method. Running out of map is a method error, never a licence to ask the
    simulator for scene bounds. The initial footprint uses only known pose.
    """

    def __init__(self, size_m=80.0, resolution_m=0.1, stride=8, *, body_height_m):
        if not math.isfinite(body_height_m) or body_height_m <= 0:
            raise ValueError("body_height_m must be positive and finite")
        self.grid = LogOddsGridCostmap(LogOddsGridConfig(
            size_m=size_m, resolution_m=resolution_m))
        self.stride = stride
        self.body_height_m = body_height_m
        self._anchor = None
        self.floor_revision = 0

    def update(self, observation):
        pose, camera = observation.pose, observation.camera
        if self._anchor is None:
            self._anchor = pose.z
            self.grid.origin_x = pose.x - self.grid.cfg.size_m / 2
            self.grid.origin_y = pose.y - self.grid.cfg.size_m / 2
        if abs(pose.z - self._anchor) > 0.6:
            # Do not overlay two storeys in one XY map. This remains a local
            # 2.5D ground-robot adaptation, not full FALCON 3D exploration.
            self.grid.reset()
            self._anchor = pose.z
            self.floor_revision += 1
        margin = camera.max_depth_m
        if (abs(pose.x - self.grid.origin_x - self.grid.cfg.size_m / 2)
                > self.grid.cfg.size_m / 2 - margin
                or abs(pose.y - self.grid.origin_y - self.grid.cfg.size_m / 2)
                > self.grid.cfg.size_m / 2 - margin):
            raise ValueError("Observed map exhausted; increase method map_size_m")
        k = camera.intrinsics
        matrix = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
        update_grid_from_depth(
            self.grid, observation.depth_m, matrix,
            world_T_camera_optical(pose, camera),
            z_min_world=self._anchor + 0.15,
            z_max_world=self._anchor + self.body_height_m + 0.05,
            depth_min_m=camera.min_depth_m, depth_max_m=camera.max_depth_m,
            downsample=self.stride, stamp_sec=float(observation.step),
            raytrace=True, raytrace_stride=1)
        spec, probabilities = self.grid.get_grid()
        # A visible floor sample is free evidence at its OWN cell, not an
        # unobserved free ray under/through furniture. Never erase an occupied
        # column: robot-height obstacle evidence takes precedence.
        points = backproject_depth(observation.depth_m, camera, pose, self.stride)
        floor = points[np.abs(points[:, 2] - self._anchor) <= 0.10]
        free_mask = np.zeros(probabilities.shape, dtype=bool)
        if len(floor):
            xs = np.floor((floor[:, 0] - spec.origin_x) / spec.resolution_m).astype(int)
            ys = np.floor((floor[:, 1] - spec.origin_y) / spec.resolution_m).astype(int)
            inside = (xs >= 0) & (xs < spec.width) & (ys >= 0) & (ys < spec.height)
            free_mask[ys[inside], xs[inside]] = True
            free_mask &= probabilities < 65
            self.grid.apply_free_mask(free_mask)
            spec, probabilities = self.grid.get_grid()
        # Explicit ternary encoding; raw probabilities are not occupied=100.
        data = np.full(probabilities.shape, -1, dtype=np.int8)
        data[(probabilities >= 0) & (probabilities <= 45)] = 0
        data[probabilities >= 65] = 100
        gx = int((pose.x - spec.origin_x) / spec.resolution_m)
        gy = int((pose.y - spec.origin_y) / spec.resolution_m)
        data[gy, gx] = 0  # the observed base itself is reachable
        return OccupancyGrid2D(
            data, OccupancyGrid2DParams(spec.resolution_m, spec.origin_x,
                                        spec.origin_y, "world"),
            values=OccupancyValues(free=0, occupied=100, unknown=-1))

    def notify_blocked(self, observation, forward_step_m):
        """Record a failed forward step as local evidence, not a GT wall lookup."""
        p = observation.pose
        ahead = [[p.x + forward_step_m * math.cos(p.yaw),
                  p.y + forward_step_m * math.sin(p.yaw)]]
        self.grid.update_from_points_xy(np.repeat(ahead, 20, axis=0))


