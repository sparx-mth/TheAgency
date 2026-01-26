"""
Goal-directed navigation behavior.

This module implements GoToPoseBehavior, which navigates the robot to a
specified goal pose. It supports both planner-assisted and subgoal-only
modes of operation.

Usage:
    >>> behavior = GoToPoseBehavior(goal_tolerance_m=0.1)
    >>> ctx = BehaviorContext(robot_id=1, pose=current, goal=target, world=grid)
    >>> output = behavior.step(ctx, planner=my_planner)

Modes:
    - With planner: Returns a Path2D for the tracker to follow
    - Without planner: Returns subgoal for coordinator to plan

See Also:
    - ExploreRoomBehavior: For frontier-based exploration
    - EnterPortalBehavior: For doorway traversal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.common.types.planning import PlanResult, PlanStatus
from sparx_agency.core.planning.interfaces.planner import PlanRequest

from ..interfaces.context import BehaviorContext
from ..interfaces.output import BehaviorOutput, BehaviorStatus
from ..utils.path_utils import trim_path_prefix


@dataclass
class GoToPoseBehavior:
    """
    Navigate to a specific goal pose.

    A fundamental navigation behavior that drives the robot to a target
    pose. Supports two modes of operation:

    1. **With planner**: Computes a full path using the injected planner
       and returns it for the tracker to follow.

    2. **Without planner**: Returns the goal as a subgoal, delegating
       path planning to the coordinator.

    Requirements:
        - ctx.goal must be set to the target Pose2D

    Attributes:
        name: Behavior identifier ("go_to_pose").
        goal_tolerance_m: Distance threshold for considering the goal
            reached. Defaults to 0.05 meters.

    Example:
        >>> behavior = GoToPoseBehavior(goal_tolerance_m=0.1)
        >>> ctx = BehaviorContext(
        ...     robot_id=1,
        ...     pose=Pose2D(0, 0, 0),
        ...     options={"goal": Pose2D(5, 3, 1.57)}
        ... )
        >>>
        >>> # With planner
        >>> output = behavior.step(ctx, planner=astar_planner)
        >>> assert output.path is not None
        >>>
        >>> # Without planner
        >>> output = behavior.step(ctx, planner=None)
        >>> assert output.subgoal is not None

    Output Contract:
        - SUCCESS: Robot within goal_tolerance_m of goal
        - RUNNING + path: Planner succeeded, follow the path
        - RUNNING + subgoal: No planner, coordinator should plan
        - FAILURE: Goal not set or planning failed

    Note:
        This behavior does not handle dynamic obstacles or replanning.
        The coordinator should monitor for path invalidation and re-invoke
        step() as needed.
    """

    name: str = field(default="go_to_pose", init=False)
    goal_tolerance_m: float = 0.05

    def reset(self) -> None:
        """Reset behavior state (no-op for this stateless behavior)."""
        pass

    def step(self, ctx: BehaviorContext, *, planner: Optional[Any] = None) -> BehaviorOutput:
        """
        Compute the next navigation output towards the goal.

        Args:
            ctx: Behavior context containing robot pose and goal.
                Required: ctx.options["goal"] must be set.
            planner: Optional path planner instance. If provided, behavior
                will compute a full path. Must implement:
                planner.plan(PlanRequest, world) -> PlanResult

        Returns:
            BehaviorOutput with one of:
            - status=SUCCESS if already at goal
            - status=RUNNING + path if planner succeeded
            - status=RUNNING + subgoal if no planner provided
            - status=FAILURE if goal missing or planning failed
        """
        goal: Optional[Pose2D] = ctx.options.get("goal")
        if goal is None:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "GoToPoseBehavior requires ctx.options['goal']"},
            )

        # Check if already at goal
        if ctx.pose.distance_to(goal) <= float(self.goal_tolerance_m):
            return BehaviorOutput(
                status=BehaviorStatus.SUCCESS,
                info={"status": "already_at_goal"},
            )

        # Subgoal-only mode: delegate planning to coordinator
        if planner is None:
            return BehaviorOutput(
                status=BehaviorStatus.RUNNING,
                subgoal=goal,
                info={"mode": "subgoal_only"},
            )

        # Planner mode: compute full path
        world = ctx.options.get("world")
        try:
            req = PlanRequest(start=ctx.pose, goal=goal)
            res: PlanResult = planner.plan(req, world)
        except Exception as e:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": f"planner_exception: {e}"},
            )

        if not getattr(res, "ok", False) or res.path is None:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={
                    "error": "planning_failed",
                    "plan_status": getattr(res, "status", PlanStatus.ERROR),
                    "message": getattr(res, "message", ""),
                },
            )

        if not isinstance(res.path, Path2D):
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "planner did not return Path2D"},
            )

        # Trim already-traversed prefix from path
        path = trim_path_prefix(res.path, ctx.pose)

        return BehaviorOutput(
            status=BehaviorStatus.RUNNING,
            path=path,
            info={
                "planner_status": getattr(res, "status", PlanStatus.SUCCESS),
                "message": getattr(res, "message", ""),
            },
        )