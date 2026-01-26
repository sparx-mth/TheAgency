"""
Behavior context container.

This module defines the BehaviorContext dataclass, which encapsulates all
inputs required by navigation behaviors. The context provides a clean
separation between behavior logic and data acquisition.

Design Philosophy:
    - Immutable: Context is frozen to prevent accidental mutation
    - Minimal: Core fields only; extensions via `options` dict
    - Decoupled: Behaviors receive data, don't fetch it

Example:
    >>> ctx = BehaviorContext(
    ...     robot_id=1,
    ...     pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
    ...     portals=[portal1, portal2],
    ...     options={"max_speed": 0.5}
    ... )
    >>> output = behavior.step(ctx, world)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Set

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.common.types.semantics import Portal2D


@dataclass(frozen=True)
class BehaviorContext:
    """
    Immutable container for behavior inputs.

    Provides the minimal set of inputs needed by navigation behaviors.
    Task-specific or behavior-specific parameters should be passed via
    the `options` dictionary.

    Attributes:
        robot_id: Unique identifier for the robot. Used for multi-robot
            coordination and logging.
        pose: Current robot pose in the map frame. Must be updated each
            step by the coordinator.
        map_frame_id: TF frame ID for the map coordinate system.
            Defaults to "map".
        portals: Sequence of known portals (doorways, thresholds) in the
            environment. Provided by perception or semantic mapping.
        forbidden_portal_ids: Set of portal IDs that the robot should not
            traverse. Used for room-constrained exploration.
        options: Free-form dictionary for task-specific parameters.
            Common keys include:
            - "goal": Target Pose2D for goal-directed behaviors
            - "max_speed": Speed limit (m/s)
            - "timeout": Maximum execution time (s)
            - "tolerance": Goal tolerance (m)

    Example:
        >>> ctx = BehaviorContext(
        ...     robot_id=0,
        ...     pose=robot.get_pose(),
        ...     portals=perception.get_portals(),
        ...     forbidden_portal_ids={"door_3", "door_7"},
        ...     options={"goal": target_pose}
        ... )

    Note:
        The context is intentionally lightweight. Heavy data like occupancy
        grids should be passed separately as the `world` parameter to
        behavior.step().
    """

    robot_id: int
    pose: Pose2D
    map_frame_id: str = "map"

    # Semantic constraints from perception/task layer
    portals: Sequence[Portal2D] = field(default_factory=tuple)
    forbidden_portal_ids: Set[str] = field(default_factory=set)

    # Extension point for task-specific parameters
    options: Dict[str, Any] = field(default_factory=dict)