"""
Room-constrained frontier exploration behavior.

This module implements ExploreRoomBehavior, which explores unknown areas
within a room by navigating to frontier cells while respecting portal
(doorway) boundaries.

Frontier Exploration:
    Frontiers are the boundaries between known-free and unknown space.
    This behavior iteratively selects the best reachable frontier and
    navigates to it until no frontiers remain.

Room Constraints:
    Portals (doorways) can be marked as "forbidden" to constrain exploration
    to a single room. The behavior will not select frontiers near forbidden
    portals.

Usage:
    >>> behavior = ExploreRoomBehavior(max_frontier_distance_m=15.0)
    >>> ctx = BehaviorContext(
    ...     robot_id=1,
    ...     pose=current_pose,
    ...     options={
    ...         "world": occupancy_grid,
    ...         "forbidden_portals": [door1, door2]
    ...     }
    ... )
    >>> output = behavior.step(ctx, planner=my_planner)

See Also:
    - GoToPoseBehavior: For goal-directed navigation
    - EnterPortalBehavior: For crossing room boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.common.types.semantics import Portal2D
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.exploration.frontier import extract_frontiers
from sparx_agency.core.planning.interfaces.planner import PlanRequest

from ..interfaces.context import BehaviorContext
from ..interfaces.output import BehaviorOutput, BehaviorStatus
from ..utils.path_utils import trim_path_prefix


@dataclass
class ExploreRoomBehavior:
    """
    Frontier-based exploration constrained by room boundaries.

    Explores unknown space by navigating to frontier cells (boundaries
    between known-free and unknown areas). Forbidden portals define
    room boundaries that the robot should not cross.

    Requirements:
        - ctx.options["world"] must be an OccupancyGrid2D
        - Optional: ctx.options["forbidden_portals"] as List[Portal2D]

    Attributes:
        name: Behavior identifier ("explore_room").
        no_frontier_patience: Number of consecutive steps with no valid
            frontier before declaring exploration complete. Defaults to 8.
        max_frontier_distance_m: Maximum distance to consider a frontier.
            Frontiers beyond this distance are ignored. Defaults to 20.0m.

    Algorithm:
        1. Extract all frontiers from the occupancy grid
        2. Filter out frontiers near forbidden portals
        3. Filter out frontiers beyond max_frontier_distance_m
        4. Select best reachable frontier (by path cost if planner provided,
           else by Euclidean distance)
        5. Return path or subgoal to selected frontier

    Completion Conditions:
        - SUCCESS: No valid frontiers for `no_frontier_patience` consecutive steps
        - FAILURE: World is not an OccupancyGrid2D

    Example:
        >>> behavior = ExploreRoomBehavior(
        ...     no_frontier_patience=10,
        ...     max_frontier_distance_m=15.0
        ... )
        >>> ctx = BehaviorContext(
        ...     robot_id=1,
        ...     pose=robot_pose,
        ...     options={
        ...         "world": grid,
        ...         "forbidden_portals": room_boundaries
        ...     }
        ... )
        >>>
        >>> while True:
        ...     output = behavior.step(ctx, planner=astar)
        ...     if output.status == BehaviorStatus.SUCCESS:
        ...         print("Room fully explored!")
        ...         break
        ...     execute(output.path)

    Note:
        Frontier extraction is performed each step, which may be expensive
        for large grids. Consider caching or throttling in the coordinator.
    """

    name: str = field(default="explore_room", init=False)
    no_frontier_patience: int = 8
    max_frontier_distance_m: float = 20.0

    # Internal state
    _no_frontier_counter: int = field(default=0, init=False, repr=False)
    _active_goal: Optional[Pose2D] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Reset exploration state for a new task."""
        self._no_frontier_counter = 0
        self._active_goal = None

    def step(self, ctx: BehaviorContext, *, planner: Optional[Any] = None) -> BehaviorOutput:
        """
        Compute the next exploration target.

        Args:
            ctx: Behavior context with robot pose and occupancy grid.
                Required: ctx.options["world"] must be OccupancyGrid2D.
                Optional: ctx.options["forbidden_portals"] for room constraints.
            planner: Optional path planner. If provided, selects frontiers
                by path cost; otherwise uses Euclidean distance.

        Returns:
            BehaviorOutput with one of:
            - status=SUCCESS if exploration complete (no frontiers)
            - status=RUNNING + path if frontier reachable with planner
            - status=RUNNING + subgoal if frontier selected without planner
            - status=FAILURE if world type is invalid
        """
        world = ctx.options.get("world")

        # Validate world type
        if not isinstance(world, OccupancyGrid2D):
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "ExploreRoomBehavior requires OccupancyGrid2D in ctx.options['world']"},
            )

        grid: OccupancyGrid2D = world
        forbidden_portals: List[Portal2D] = ctx.options.get("forbidden_portals", [])

        # Extract frontier candidates
        frontiers: Set[Pose2D] = extract_frontiers(grid)
        if not frontiers:
            return self._handle_no_frontiers("no_frontiers")

        # Filter frontiers by distance and forbidden portal proximity
        candidates = self._filter_candidates(ctx.pose, list(frontiers), forbidden_portals)
        if not candidates:
            return self._handle_no_frontiers("no_valid_frontiers")

        # Select and navigate to best frontier
        if planner is None:
            return self._select_nearest_frontier(ctx.pose, candidates)

        return self._select_best_path_frontier(ctx, candidates, planner, grid)

    def _handle_no_frontiers(self, status: str) -> BehaviorOutput:
        """Handle case when no valid frontiers are available."""
        self._no_frontier_counter += 1

        if self._no_frontier_counter >= int(self.no_frontier_patience):
            return BehaviorOutput(
                status=BehaviorStatus.SUCCESS,
                info={"status": status},
            )

        return BehaviorOutput(
            status=BehaviorStatus.RUNNING,
            info={"status": f"{status}_yet", "counter": self._no_frontier_counter},
        )

    def _filter_candidates(
        self,
        pose: Pose2D,
        frontiers: List[Pose2D],
        forbidden_portals: List[Portal2D],
    ) -> List[Pose2D]:
        """Filter frontiers by distance and forbidden portal proximity."""
        candidates = []
        for f in frontiers:
            # Skip frontiers too far away
            if pose.distance_to(f) > float(self.max_frontier_distance_m):
                continue
            # Skip frontiers near forbidden portals
            if self._is_near_forbidden(f, forbidden_portals, radius_m=0.25):
                continue
            candidates.append(f)
        return candidates

    def _select_nearest_frontier(
        self, pose: Pose2D, candidates: List[Pose2D]
    ) -> BehaviorOutput:
        """Select nearest frontier when no planner is available."""
        goal = min(candidates, key=lambda p: pose.distance_to(p))
        self._active_goal = goal
        self._no_frontier_counter = 0

        return BehaviorOutput(
            status=BehaviorStatus.RUNNING,
            subgoal=goal,
            info={"mode": "subgoal_only"},
        )

    def _select_best_path_frontier(
        self,
        ctx: BehaviorContext,
        candidates: List[Pose2D],
        planner: Any,
        grid: OccupancyGrid2D,
    ) -> BehaviorOutput:
        """Select best frontier by path cost using the planner."""
        # Sort candidates by Euclidean distance, try nearest first
        candidates.sort(key=lambda p: ctx.pose.distance_to(p))
        top_candidates = candidates[:min(25, len(candidates))]

        best_path: Optional[Path2D] = None
        best_goal: Optional[Pose2D] = None
        best_cost = float("inf")

        for goal in top_candidates:
            try:
                req = PlanRequest(start=ctx.pose, goal=goal)
                res = planner.plan(req, grid)
            except Exception:
                continue

            if not getattr(res, "ok", False) or res.path is None:
                continue
            if not isinstance(res.path, Path2D):
                continue

            path = trim_path_prefix(res.path, ctx.pose)
            cost = path.length()

            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_goal = goal

        if best_path is None or best_goal is None:
            return self._handle_no_frontiers("no_reachable_frontiers")

        self._active_goal = best_goal
        self._no_frontier_counter = 0

        return BehaviorOutput(
            status=BehaviorStatus.RUNNING,
            path=best_path,
            info={
                "frontier_goal": (best_goal.x, best_goal.y),
                "path_cost": best_cost,
            },
        )

    @staticmethod
    def _is_near_forbidden(
        pose: Pose2D, portals: List[Portal2D], *, radius_m: float
    ) -> bool:
        """Check if a pose is within radius of any forbidden portal."""
        for portal in portals:
            if pose.distance_to(portal.center) <= float(radius_m):
                return True
        return False