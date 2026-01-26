"""
Wall-following navigation behavior.

This module implements WallFollowBehavior, a reactive behavior that navigates
the robot while maintaining a wall on a specified side at a target clearance
distance.

Algorithm:
    The behavior samples candidate directions around the robot's current
    heading, evaluates each for wall proximity on the desired side, and
    selects the direction that best maintains the target clearance.

Usage:
    >>> behavior = WallFollowBehavior(side="right", clearance_m=0.3)
    >>> ctx = BehaviorContext(robot_id=1, pose=pose, options={"world": costmap})
    >>> output = behavior.step(ctx)
    >>> navigate_to(output.subgoal)

See Also:
    - GoToPoseBehavior: For goal-directed navigation
    - ExploreRoomBehavior: For frontier-based exploration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Any, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment.costmap2d import Costmap2D
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D

from ..interfaces.context import BehaviorContext
from ..interfaces.output import BehaviorOutput, BehaviorStatus
from ..utils.world_adapters import costmap_from_occupancy_grid


@dataclass
class WallFollowBehavior:
    """
    Reactive wall-following navigation.

    Produces local subgoals that keep a wall on the specified side at
    approximately the target clearance distance. This is a reactive
    behavior that does not require a planner or goal.

    Requirements:
        - ctx.options["world"] must be Costmap2D or OccupancyGrid2D

    Attributes:
        name: Behavior identifier ("wall_follow").
        side: Which side to keep the wall on ("right" or "left").
            Defaults to "right".
        lookahead_m: Distance to project subgoals ahead of the robot.
            Defaults to 0.6m.
        clearance_m: Target distance to maintain from the wall.
            Defaults to 0.25m.
        sample_angles_deg: Tuple of angles (degrees) to sample around
            the robot's heading. Defaults to (-60, -30, 0, 30, 60).
        max_sample_range_m: Maximum raycast distance for wall detection.
            Defaults to 2.0m.

    Algorithm:
        1. For each sample angle, compute a candidate subgoal
        2. Check if the candidate is in free space
        3. Raycast to the side to measure wall distance
        4. Score candidates by how close the wall distance is to clearance_m
        5. Return the highest-scoring candidate as the subgoal

    Example:
        >>> # Follow wall on the right at 30cm clearance
        >>> behavior = WallFollowBehavior(
        ...     side="right",
        ...     clearance_m=0.30,
        ...     lookahead_m=0.5
        ... )
        >>>
        >>> while not done:
        ...     output = behavior.step(ctx)
        ...     if output.status == BehaviorStatus.FAILURE:
        ...         print("No valid wall-follow direction")
        ...         break
        ...     drive_to(output.subgoal)

    Output Contract:
        - RUNNING + subgoal: Valid wall-following direction found
        - FAILURE: No free candidate directions or no costmap available

    Note:
        This behavior runs indefinitely (always RUNNING) until externally
        stopped. It is typically composed with other behaviors or stopped
        by a coordinator when a condition is met.
    """

    name: str = field(default="wall_follow", init=False)
    side: str = "right"
    lookahead_m: float = 0.6
    clearance_m: float = 0.25
    sample_angles_deg: Tuple[int, ...] = (-60, -30, 0, 30, 60)
    max_sample_range_m: float = 2.0

    # Internal state
    _initialized: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        """Reset behavior state."""
        self._initialized = False

    def step(self, ctx: BehaviorContext, *, planner: Optional[Any] = None) -> BehaviorOutput:
        """
        Compute the next wall-following subgoal.

        Args:
            ctx: Behavior context with robot pose and world representation.
                Required: ctx.options["world"] must be Costmap2D or OccupancyGrid2D.
            planner: Unused. Wall-following is purely reactive.

        Returns:
            BehaviorOutput with one of:
            - status=RUNNING + subgoal for the next waypoint
            - status=FAILURE if no valid direction found
        """
        world = ctx.options.get("world")
        costmap = self._ensure_costmap(world)

        if costmap is None:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "WallFollowBehavior requires Costmap2D or OccupancyGrid2D in ctx.options['world']"},
            )

        subgoal = self._pick_wall_follow_subgoal(ctx.pose, costmap)
        if subgoal is None:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "no_valid_wall_follow_subgoal"},
            )

        return BehaviorOutput(
            status=BehaviorStatus.RUNNING,
            subgoal=subgoal,
            info={"mode": "subgoal", "side": self.side},
        )

    @staticmethod
    def _ensure_costmap(world: Any) -> Optional[Costmap2D]:
        """
        Convert world representation to Costmap2D if needed.

        Args:
            world: World representation (Costmap2D, OccupancyGrid2D, or other).

        Returns:
            Costmap2D instance, or None if conversion not possible.
        """
        if isinstance(world, Costmap2D):
            return world
        if isinstance(world, OccupancyGrid2D):
            return costmap_from_occupancy_grid(world, unknown_is_occupied=True)
        return None

    def _pick_wall_follow_subgoal(
        self, pose: Pose2D, costmap: Costmap2D
    ) -> Optional[Pose2D]:
        """
        Select the best wall-following subgoal.

        Samples candidate directions, filters by free space, and scores
        by proximity to target wall clearance.

        Args:
            pose: Current robot pose.
            costmap: Cost map for obstacle checking.

        Returns:
            Best subgoal Pose2D, or None if no valid candidates.
        """
        base_yaw = pose.yaw
        best_subgoal: Optional[Pose2D] = None
        best_score = float("-inf")

        for deg in self.sample_angles_deg:
            # Compute candidate direction
            yaw = base_yaw + np.deg2rad(float(deg))
            goal_x = pose.x + cos(yaw) * float(self.lookahead_m)
            goal_y = pose.y + sin(yaw) * float(self.lookahead_m)

            # Skip if candidate is not in free space
            if not self._is_free_world(costmap, goal_x, goal_y):
                continue

            # Score by wall clearance on desired side
            score = self._side_clearance_score(pose, yaw, costmap)
            if score > best_score:
                best_score = score
                best_subgoal = Pose2D(goal_x, goal_y, yaw)

        return best_subgoal

    def _side_clearance_score(
        self, pose: Pose2D, yaw: float, costmap: Costmap2D
    ) -> float:
        """
        Score a direction by wall clearance on the target side.

        Higher scores indicate wall distance closer to target clearance.

        Args:
            pose: Current robot pose.
            yaw: Candidate heading direction.
            costmap: Cost map for raycasting.

        Returns:
            Score value (higher is better). Returns large negative value
            if no wall detected.
        """
        # Compute perpendicular direction to the target side
        side_sign = -1.0 if self.side == "right" else 1.0
        side_yaw = yaw + side_sign * (pi / 2.0)

        # Raycast to find wall distance
        wall_distance = self._raycast_distance(
            costmap, pose.x, pose.y, side_yaw, max_range=float(self.max_sample_range_m)
        )

        if wall_distance is None:
            return -1e9

        # Score: penalize deviation from target clearance
        target = float(self.clearance_m)
        return -abs(wall_distance - target)

    @staticmethod
    def _is_free_world(costmap: Costmap2D, x: float, y: float) -> bool:
        """Check if a world coordinate is in free space."""
        grid_x, grid_y = costmap.world_to_grid(x, y)
        return costmap.is_free(grid_x, grid_y)

    @staticmethod
    def _raycast_distance(
        costmap: Costmap2D,
        x: float,
        y: float,
        yaw: float,
        *,
        max_range: float,
    ) -> Optional[float]:
        """
        Cast a ray and return distance to first obstacle.

        Simple grid-based raycast that steps along the ray until hitting
        an occupied cell or going out of bounds.

        Args:
            costmap: Cost map for obstacle checking.
            x: Ray origin X coordinate (world frame).
            y: Ray origin Y coordinate (world frame).
            yaw: Ray direction (radians).
            max_range: Maximum raycast distance.

        Returns:
            Distance to obstacle in meters, or max_range if no obstacle
            within range. Returns None only on unexpected errors.
        """
        step = costmap.resolution * 0.5
        if step <= 0:
            step = 0.05

        distance = 0.0
        while distance <= max_range:
            # Compute query point
            query_x = x + cos(yaw) * distance
            query_y = y + sin(yaw) * distance

            # Convert to grid coordinates
            grid_x, grid_y = costmap.world_to_grid(query_x, query_y)

            # Check bounds and occupancy
            if not costmap.in_bounds(grid_x, grid_y):
                return distance
            if costmap.is_occupied(grid_x, grid_y):
                return distance

            distance += step

        return max_range