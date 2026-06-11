"""
Behavior interface definitions.

This module defines the base protocol and data structures for implementing
navigation behaviors in the sparx_agency planning system.

Behaviors are stateful components that produce navigation outputs (goals, paths,
or control commands) given a context. They follow a simple lifecycle:
    1. Instantiate with configuration parameters
    2. Call reset() before each new task
    3. Call step() repeatedly until done/failure

Example:
    >>> behavior = MyBehavior(param=value)
    >>> behavior.reset()
    >>> while True:
    ...     output = behavior.step(ctx, world)
    ...     if output.done:
    ...         break
    ...     execute(output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from sparx_agency.core.common.types import Path2D, Pose2D


@dataclass(frozen=True)
class BehaviorDecision:
    """
    Generic behavior output container.

    A behavior's step() method returns this to communicate its decision
    to the coordinator. Typically, exactly one of `goal` or `path` is set.

    Attributes:
        goal: Target pose for the coordinator to plan towards. Preferred output
            when the behavior delegates path planning to the coordinator.
        path: Pre-computed path if the behavior performed planning internally.
            The coordinator should track this path directly.
        done: Terminal flag. True indicates the behavior has completed (success
            or failure). The coordinator should transition to the next behavior.
        info: Diagnostic metadata for logging, debugging, and visualization.
            Common keys: "status", "error", "phase", "distance_remaining".

    Note:
        This class is deprecated in favor of BehaviorOutput which provides
        richer status information. Kept for backward compatibility.
    """

    goal: Optional[Pose2D] = None
    path: Optional[Path2D] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Behavior(Protocol):
    """
    Protocol defining the behavior interface.

    All navigation behaviors must implement this protocol. The protocol
    defines a minimal contract that allows behaviors to be composed and
    managed by a coordinator.

    Attributes:
        name: Unique identifier for the behavior. Used for logging and
            registry lookup.

    Methods:
        reset: Reinitialize internal state for a new task.
        step: Compute the next navigation decision.

    Example:
        >>> class MyBehavior:
        ...     name: str = "my_behavior"
        ...
        ...     def reset(self) -> None:
        ...         self._state = initial_state()
        ...
        ...     def step(self, ctx: Any, world: Any) -> BehaviorDecision:
        ...         return BehaviorDecision(goal=compute_goal(ctx, world))
    """

    name: str

    def reset(self) -> None:
        """
        Reset internal state for a new task.

        Called by the coordinator before starting a new navigation task.
        Implementations should clear any cached state, counters, or phase
        tracking variables.
        """
        ...

    def step(self, ctx: Any, world: Any) -> BehaviorDecision:
        """
        Compute the next navigation decision.

        Called repeatedly by the coordinator until the behavior signals
        completion via `done=True` in the returned decision.

        Args:
            ctx: Behavior context containing robot state, goal, and features.
            world: World representation (e.g., OccupancyGrid2D, Costmap2D).

        Returns:
            BehaviorDecision containing the navigation output and status.
        """
        ...