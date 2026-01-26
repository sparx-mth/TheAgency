"""
Portal traversal behavior.

This module implements EnterPortalBehavior, which navigates the robot
through doorways, gates, or other traversable boundaries (portals).

Traversal Phases:
    1. Approach: Navigate to a point before the portal
    2. Cross: Navigate through and past the portal

Portal Geometry:
    The behavior uses the portal's normal vector to compute approach and
    cross waypoints. The approach point is offset opposite to the normal,
    and the cross point is offset along the normal.

Usage:
    >>> behavior = EnterPortalBehavior(approach_offset_m=0.4)
    >>> ctx = BehaviorContext(
    ...     robot_id=1,
    ...     pose=current_pose,
    ...     portals=[kitchen_door],
    ...     options={"target_portal_id": "door_1"}
    ... )
    >>> output = behavior.step(ctx)

See Also:
    - ExploreRoomBehavior: For room-constrained exploration
    - GoToPoseBehavior: For general goal navigation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Any, List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.common.types.semantics import Portal2D

from ..interfaces.context import BehaviorContext
from ..interfaces.output import BehaviorOutput, BehaviorStatus


@dataclass
class EnterPortalBehavior:
    """
    Navigate through a portal (doorway/threshold).

    Guides the robot through a two-phase traversal: approach the portal
    from one side, then cross to the other side. Portals are provided
    externally via the context; this behavior does not detect them.

    Requirements:
        - ctx.portals: Sequence[Portal2D] of available portals
        - Optional: ctx.options["target_portal_id"]: str to select specific portal

    Attributes:
        name: Behavior identifier ("enter_portal").
        approach_offset_m: Distance to stop before the portal during
            approach phase. Defaults to 0.35m.
        cross_offset_m: Distance to travel past the portal during
            cross phase. Defaults to 0.55m.
        success_radius_m: Distance threshold for phase transitions.
            Defaults to 0.20m.

    Traversal Phases:
        1. **Approach**: Robot navigates to a point `approach_offset_m` meters
           before the portal (opposite the normal direction).
        2. **Cross**: Robot navigates through the portal to a point
           `cross_offset_m` meters past it (along the normal direction).

    Portal Selection:
        - If `target_portal_id` is provided, selects that specific portal
        - Otherwise, selects the nearest portal to the robot

    Example:
        >>> behavior = EnterPortalBehavior(
        ...     approach_offset_m=0.4,
        ...     cross_offset_m=0.6,
        ...     success_radius_m=0.15
        ... )
        >>> ctx = BehaviorContext(
        ...     robot_id=1,
        ...     pose=robot_pose,
        ...     portals=[door1, door2],
        ...     options={"target_portal_id": "door1"}
        ... )
        >>>
        >>> while True:
        ...     output = behavior.step(ctx)
        ...     if output.status == BehaviorStatus.SUCCESS:
        ...         print("Portal crossed!")
        ...         break
        ...     navigate_to(output.subgoal)

    Output Contract:
        - RUNNING + subgoal: Navigate to approach or cross point
        - SUCCESS: Robot has crossed the portal
        - FAILURE: No portals provided or target portal not found

    Note:
        If portal.normal_yaw is None, the behavior returns the portal center
        as both approach and cross targets. The coordinator/planner must
        determine how to cross in this case.
    """

    name: str = field(default="enter_portal", init=False)
    approach_offset_m: float = 0.35
    cross_offset_m: float = 0.55
    success_radius_m: float = 0.20

    # Internal state
    _phase: str = field(default="approach", init=False, repr=False)
    _active_portal_id: Optional[str] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Reset traversal state for a new portal crossing."""
        self._phase = "approach"
        self._active_portal_id = None

    def step(self, ctx: BehaviorContext, *, planner: Optional[Any] = None) -> BehaviorOutput:
        """
        Compute the next traversal waypoint.

        Args:
            ctx: Behavior context containing robot pose and portal features.
                Required: ctx.portals as Sequence[Portal2D].
                Optional: ctx.options["target_portal_id"] as str.
            planner: Unused. Portal traversal returns subgoals only;
                the coordinator is responsible for path planning.

        Returns:
            BehaviorOutput with one of:
            - status=RUNNING + subgoal for approach/cross waypoint
            - status=SUCCESS when portal has been crossed
            - status=FAILURE if no portals or target not found
        """
        # Get available portals
        portals: Sequence[Portal2D] = ctx.portals
        if not portals:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "no portals provided"},
            )

        # Select target portal
        target_id: Optional[str] = ctx.options.get("target_portal_id")
        portal = self._select_portal(ctx.pose, list(portals), target_id)

        if portal is None:
            return BehaviorOutput(
                status=BehaviorStatus.FAILURE,
                info={"error": "target portal not found"},
            )

        # Initialize or update active portal tracking
        if self._active_portal_id is None:
            self._active_portal_id = portal.id

        # Handle portal change (e.g., from external update)
        if portal.id != self._active_portal_id:
            self._active_portal_id = portal.id
            self._phase = "approach"

        # Compute approach and cross waypoints
        approach, cross = self._compute_waypoints(portal)

        # Phase: Approach
        if self._phase == "approach":
            if ctx.pose.distance_to(approach) <= self.success_radius_m:
                self._phase = "cross"
            else:
                return BehaviorOutput(
                    status=BehaviorStatus.RUNNING,
                    subgoal=approach,
                    info={"portal_id": portal.id, "phase": "approach"},
                )

        # Phase: Cross
        if ctx.pose.distance_to(cross) <= self.success_radius_m:
            return BehaviorOutput(
                status=BehaviorStatus.SUCCESS,
                info={"portal_id": portal.id, "phase": "done"},
            )

        return BehaviorOutput(
            status=BehaviorStatus.RUNNING,
            subgoal=cross,
            info={"portal_id": portal.id, "phase": "cross"},
        )

    @staticmethod
    def _select_portal(
        pose: Pose2D, portals: List[Portal2D], target_id: Optional[str]
    ) -> Optional[Portal2D]:
        """
        Select the target portal.

        Args:
            pose: Current robot pose for nearest-portal selection.
            portals: List of available portals.
            target_id: Specific portal ID to select, or None for nearest.

        Returns:
            Selected Portal2D, or None if target_id not found.
        """
        if target_id:
            for p in portals:
                if p.id == target_id:
                    return p
            return None

        # No target specified: select nearest portal
        if not portals:
            return None

        return min(portals, key=lambda p: pose.distance_to(p.center))

    def _compute_waypoints(self, portal: Portal2D) -> Tuple[Pose2D, Pose2D]:
        """
        Compute approach and cross waypoints for the portal.

        Args:
            portal: Target portal with center and optional normal_yaw.

        Returns:
            Tuple of (approach_pose, cross_pose). If normal_yaw is None,
            both poses are the portal center.
        """
        center = portal.center

        if portal.normal_yaw is None:
            # Without orientation, return center for both waypoints
            # Coordinator/planner must determine crossing direction
            return center, center

        # Compute unit normal vector
        nx = cos(portal.normal_yaw)
        ny = sin(portal.normal_yaw)

        # Approach: offset opposite to normal (before the portal)
        approach_x = center.x - nx * float(self.approach_offset_m)
        approach_y = center.y - ny * float(self.approach_offset_m)
        approach = Pose2D(approach_x, approach_y, portal.normal_yaw)

        # Cross: offset along normal (after the portal)
        cross_x = center.x + nx * float(self.cross_offset_m)
        cross_y = center.y + ny * float(self.cross_offset_m)
        cross = Pose2D(cross_x, cross_y, portal.normal_yaw)

        return approach, cross