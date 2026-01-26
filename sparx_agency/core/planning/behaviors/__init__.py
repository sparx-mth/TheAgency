"""
Modular behavior framework for autonomous robot navigation.

This package provides composable navigation behaviors that produce paths,
subgoals, or control commands for a coordinator to execute.

Behaviors:
    - GoToPoseBehavior: Goal-directed navigation
    - ExploreRoomBehavior: Frontier-based room exploration
    - EnterPortalBehavior: Doorway/threshold traversal
    - WallFollowBehavior: Wall-following navigation

Interfaces:
    - Behavior: Protocol for all behaviors
    - BehaviorContext: Input container
    - BehaviorOutput: Output container with status
    - BehaviorStatus: Lifecycle states (RUNNING, SUCCESS, FAILURE)
    - Portal2D: Semantic feature for doorways/thresholds

Registry:
    - BehaviorRegistry: Lookup behaviors by name

Example:
    >>> from sparx_agency.core.planning.behaviors import (
    ...     GoToPoseBehavior,
    ...     BehaviorContext,
    ...     BehaviorRegistry,
    ... )
    >>>
    >>> behavior = GoToPoseBehavior()
    >>> ctx = BehaviorContext(robot_id=1, pose=current, goal=target, world=grid)
    >>> output = behavior.step(ctx, planner=my_planner)
"""

from .interfaces import (
    Behavior,
    BehaviorContext,
    BehaviorDecision,
    BehaviorOutput,
    BehaviorStatus,
    Portal2D,
)
from .algorithmic import (
    EnterPortalBehavior,
    ExploreRoomBehavior,
    GoToPoseBehavior,
    WallFollowBehavior,
)
from .registry import BehaviorRegistry

__all__ = [
    # Interfaces
    "Behavior",
    "BehaviorContext",
    "BehaviorDecision",
    "BehaviorOutput",
    "BehaviorStatus",
    "Portal2D",
    # Behaviors
    "EnterPortalBehavior",
    "ExploreRoomBehavior",
    "GoToPoseBehavior",
    "WallFollowBehavior",
    # Registry
    "BehaviorRegistry",
]