"""
Behavior output definitions.

This module defines the output structures returned by behavior step() methods.
The output communicates the behavior's decision and status to the coordinator.

Output Hierarchy:
    - BehaviorStatus: Lifecycle state (RUNNING, SUCCESS, FAILURE)
    - BehaviorOutput: Full output container with navigation commands

Output Types (mutually exclusive in typical usage):
    - path: Pre-planned trajectory for the tracker to follow
    - subgoal: Intermediate waypoint for the coordinator to plan towards
    - control: Direct control command (rare; bypasses planning)

Example:
    >>> output = BehaviorOutput(
    ...     status=BehaviorStatus.RUNNING,
    ...     subgoal=Pose2D(x=5.0, y=3.0, yaw=0.0),
    ...     info={"phase": "approach", "distance": 2.5}
    ... )
    >>> if output.ok:
    ...     coordinator.plan_to(output.subgoal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from sparx_agency.core.common.types import ControlCommand, Path2D, Pose2D


class BehaviorStatus(str, Enum):
    """
    Behavior lifecycle status.

    Indicates the current state of a behavior's execution. The coordinator
    uses this to determine whether to continue stepping the behavior or
    transition to another.

    Values:
        RUNNING: Behavior is actively working towards its objective.
            Continue calling step().
        SUCCESS: Behavior completed successfully. Transition to next behavior.
        FAILURE: Behavior failed and cannot continue. Handle error or retry.

    Example:
        >>> if output.status == BehaviorStatus.SUCCESS:
        ...     coordinator.transition_to_next_behavior()
        >>> elif output.status == BehaviorStatus.FAILURE:
        ...     coordinator.handle_failure(output.info.get("error"))
    """

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class BehaviorOutput:
    """
    Immutable container for behavior step() output.

    Encapsulates the behavior's navigation decision and execution status.
    The coordinator interprets this output to drive robot motion.

    Output Semantics:
        - Exactly one of (path, subgoal, control) is typically set
        - path: Behavior computed a full trajectory; tracker should follow it
        - subgoal: Behavior wants to reach this pose; coordinator should plan
        - control: Direct velocity command; bypass planning (use sparingly)
        - None of the above: Behavior is waiting or has no motion command

    Attributes:
        status: Current lifecycle state. See BehaviorStatus.
        subgoal: Intermediate goal pose for the coordinator to plan towards.
            Used when the behavior delegates path planning.
        path: Pre-computed path for the tracker to follow directly.
            Used when the behavior has an internal planner.
        control: Direct control command (linear/angular velocity).
            Rare; used for reactive behaviors that bypass planning.
        info: Diagnostic metadata dictionary. Common keys:
            - "error": Error message on FAILURE
            - "phase": Current behavior phase (e.g., "approach", "cross")
            - "distance": Distance to goal/target
            - "mode": Operating mode ("subgoal_only", "with_planner")

    Properties:
        ok: True if status is RUNNING or SUCCESS (non-failure).

    Example:
        >>> # Exploration behavior returning a frontier goal
        >>> output = BehaviorOutput(
        ...     status=BehaviorStatus.RUNNING,
        ...     subgoal=frontier_pose,
        ...     info={"frontier_id": 42, "distance": 3.2}
        ... )
        >>>
        >>> # Goal-directed behavior with pre-planned path
        >>> output = BehaviorOutput(
        ...     status=BehaviorStatus.RUNNING,
        ...     path=planned_path,
        ...     info={"path_length": 5.4}
        ... )
        >>>
        >>> # Behavior failure
        >>> output = BehaviorOutput(
        ...     status=BehaviorStatus.FAILURE,
        ...     info={"error": "no_valid_path", "attempts": 3}
        ... )
    """

    status: BehaviorStatus
    subgoal: Optional[Pose2D] = None
    path: Optional[Path2D] = None
    control: Optional[ControlCommand] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """
        Check if the behavior is in a non-failure state.

        Returns:
            True if status is RUNNING or SUCCESS, False if FAILURE.
        """
        return self.status in (BehaviorStatus.RUNNING, BehaviorStatus.SUCCESS)