"""Ground-safe FALCON execution boundary; hypothetical CP edges never execute."""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.exploration.falcon.travel import segment_clear
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid


class GroundMotion:
    def __init__(self, policy):
        self.policy = policy
        self.scope = None

    def update(self, observation, world):
        self.world = world
        p = self.policy
        fields = p.planner.fields_for(world)
        self.topology_cost = assemble_cost_grid(fields, p.planner_params, p.settings.body_radius_m)[0]
        cost = self.topology_cost.copy()
        cost[fields.soft_clearance <= p.settings.body_radius_m] = np.inf
        x, y = world.world_to_grid(observation.pose.x, observation.pose.y)
        if world.in_bounds(x, y) and world.grid[y, x] == world.values.free:
            cost[y, x] = 1.0  # actual occupied base, not a snapped destination
        self.cost = cost
        return cost

    def path_collides(self, world, points, passable_start=None):
        safe = np.isfinite(self.cost)
        if self.scope is not None:
            safe &= self.scope
        cells = [world.world_to_grid(p.x, p.y) for p in points]
        return any(not segment_clear(safe, a, b) for a, b in zip(cells, cells[1:]))

    def permits(self, observation, action, local=False):
        if action != DiscreteAction.MOVE_FORWARD:
            return True
        pose = observation.pose
        after = apply_action(pose, action, self.policy.episode.action_spec)
        safe = np.isfinite(self.cost)
        if local and self.scope is not None:
            safe &= self.scope
        # Check the continuous segment at sub-cell intervals as well as its
        # no-corner-cut grid supercover. No alternate or larger action is sent.
        fractions = np.linspace(0, 1, max(3, int(self.policy.episode.action_spec.forward_step_m / self.world.resolution * 4) + 1))
        cells = [self.world.world_to_grid(pose.x + t * (after.x - pose.x), pose.y + t * (after.y - pose.y)) for t in fractions]
        return all(segment_clear(safe, a, b) for a, b in zip(cells, cells[1:]))
