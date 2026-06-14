"""
Semantic feature definitions.

This module defines data structures for semantic features in the environment
that behaviors can reason about. These features are external inputs provided
by perception or semantic mapping systems—behaviors do not detect them.

Supported Features:
    - Portal2D: Doorways, gates, thresholds, and other traversable boundaries

Design Philosophy:
    - Features are immutable value objects
    - Features carry enough metadata for behavior decision-making
    - Features are detection-agnostic (behaviors don't care how they were found)

Example:
    >>> portal = Portal2D(
    ...     id="door_kitchen",
    ...     center=Pose2D(x=5.0, y=3.0, yaw=1.57),
    ...     normal_yaw=1.57,
    ...     width_m=0.9,
    ...     tags={"type": "doorway", "room": "kitchen"}
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from sparx_agency.core.common.types import Pose2D


@dataclass(frozen=True)
class Portal2D:
    """
    A traversable boundary in the environment.

    Represents doorways, gates, thresholds, or any other traversable
    boundary between regions. Portals are provided by external systems
    (perception, semantic maps) and used by behaviors for navigation
    decisions.

    The portal coordinate system:
        - center: The midpoint of the portal opening
        - normal_yaw: Direction pointing "outward" from one side
          (perpendicular to the portal plane)

    Attributes:
        id: Unique identifier for the portal. Used for tracking and
            forbidden portal filtering. Should be stable across frames.
        center: Pose at the center of the portal opening. The yaw component
            may be set to the crossing direction or left as 0.
        normal_yaw: Orientation of the portal normal (radians). Points
            perpendicular to the portal plane, indicating the "outward"
            direction. Used to compute approach/cross waypoints.
            None if orientation is unknown.
        width_m: Width of the portal opening in meters. Used for
            collision-aware path planning. None if unknown.
        tags: Free-form metadata dictionary. Common keys:
            - "type": Portal type ("doorway", "gate", "threshold")
            - "room": Associated room name
            - "traversable": "true" or "false"
            - "direction": "bidirectional", "entry_only", "exit_only"

    Example:
        >>> # Kitchen doorway facing north
        >>> portal = Portal2D(
        ...     id="door_001",
        ...     center=Pose2D(x=5.0, y=3.0, yaw=0.0),
        ...     normal_yaw=math.pi / 2,  # Pointing north
        ...     width_m=0.85,
        ...     tags={"type": "doorway", "room": "kitchen"}
        ... )
        >>>
        >>> # Approach from the south, cross to the north
        >>> approach_point = portal.center.x, portal.center.y - 0.5
        >>> cross_point = portal.center.x, portal.center.y + 0.5

    Note:
        Behaviors should not modify portals. If portal state changes
        (e.g., door closes), the perception system should provide
        updated portal objects.
    """

    id: str
    center: Pose2D
    normal_yaw: Optional[float] = None
    width_m: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)