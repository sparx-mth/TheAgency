"""
Naive exploration policy: random local goal selection.

This is intentionally *not* an action-level controller (no TURN/FORWARD).
It returns a short-horizon goal Pose2D in world coordinates that is:
- within bounds
- in FREE space (and optionally not UNKNOWN)

The caller can pass this goal into a planner (e.g., A* 2D) to produce a path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.interfaces.exploration import (
    ExplorationContext,
    ExplorationDecision,
    ExplorationPolicy,
)


@dataclass(frozen=True)
class RandomWalkParams:
    """
    Parameters for random-walk exploration.

    Attributes:
        max_radius_cells: Sample candidates within this Manhattan radius (in grid cells).
        n_tries: Number of random samples per step before giving up.
        avoid_unknown: If True, sample only FREE cells (not UNKNOWN).
    """
    max_radius_cells: int = 6
    n_tries: int = 50
    avoid_unknown: bool = True


@dataclass
class RandomWalkPolicy:
    """
    Random exploration policy that proposes a nearby goal in free space.

    This policy:
    - Samples a random FREE cell around the robot.
    - Outputs it as Pose2D in world.

    It does not compute paths and does not track state beyond params.
    """
    name: str = field(default="random_walk", init=False)
    params: RandomWalkParams = field(default_factory=RandomWalkParams)

    def step(self, ctx: ExplorationContext, world: Any) -> ExplorationDecision:
        if not isinstance(world, OccupancyGrid2D):
            return ExplorationDecision(goal=None, path=None, info={"error": "world must be OccupancyGrid2D"})

        gx0, gy0 = world.world_to_grid(ctx.pose.x, ctx.pose.y)

        def is_ok(gx: int, gy: int) -> bool:
            if not world.in_bounds(gx, gy):
                return False
            if self.params.avoid_unknown and world.is_unknown(gx, gy):
                return False
            return world.is_free(gx, gy)

        r = max(1, int(self.params.max_radius_cells))
        for _ in range(max(1, int(self.params.n_tries))):
            dx = random.randint(-r, r)
            dy = random.randint(-r, r)
            if abs(dx) + abs(dy) > r:
                continue
            gx, gy = gx0 + dx, gy0 + dy
            if not is_ok(gx, gy):
                continue

            wx, wy = world.grid_to_world(gx, gy)
            return ExplorationDecision(
                goal=Pose2D(wx, wy, 0.0),
                path=None,
                info={"policy": "random_walk", "goal_grid": (gx, gy)},
            )

        return ExplorationDecision(goal=None, path=None, info={"policy": "random_walk", "status": "no_candidate_found"})
