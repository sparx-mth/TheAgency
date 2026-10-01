"""Stable target standoff selection; visibility is not a route-validity test."""
from __future__ import annotations

import math


class TargetApproachPath:
    """Own the projected footpoint while the existing route memory owns A*."""

    def __init__(self, policy):
        self.policy = policy
        self.goal_xyz = self.reference_xyz = None
        self.refinements = 0

    def command(self, target, obs, world):
        p = self.policy
        if self.reference_xyz is not None and math.dist(self.reference_xyz[:2], target.xyz[:2]) > target.settings.refine_distance_m:
            self.goal_xyz = None
            self.refinements += 1
            p.route_memory.clear("target_refined")
            p._route = p._goal = None
        if self.goal_xyz is not None:
            # Do not generate a different bearing-relative endpoint every frame.
            # _navigate still checks collisions, blocked motion and path progress.
            command = p._navigate(obs, world, self.goal_xyz[:2], "target_closing")
            if command is not None:
                return command
            self.goal_xyz = None
        limit = target.settings.terminal_distance_m - p.converter_params.goal_tolerance_m
        radius = max(p.settings.body_radius_m, limit - world.resolution)
        bearing = math.atan2(obs.pose.y - target.xyz[1], obs.pose.x - target.xyz[0])
        for offset in (0, 30, -30, 60, -60, 90, -90, 180):
            angle = bearing + math.radians(offset)
            goal = (target.xyz[0] + radius * math.cos(angle), target.xyz[1] + radius * math.sin(angle), obs.pose.z)
            if p.target_projector is not None:
                goal = p.target_projector(goal)
                if goal is None:
                    continue
            if abs(goal[2] - obs.pose.z) > 0.2 or math.dist(goal[:2], target.xyz[:2]) > limit:
                continue
            command = p._navigate(obs, world, goal[:2], "target_closing")
            if command is not None and math.dist(command.waypoints[-1], target.xyz[:2]) <= limit:
                self.goal_xyz, self.reference_xyz = tuple(goal), tuple(target.xyz)
                return command
        return None
