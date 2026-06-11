"""
Emergency-stop / reflex layer for immediate collision avoidance.

This module provides a simple bubble-based emergency stop mechanism that operates
independently of trajectory planning. While TrajectorySafetyChecker validates a
short prefix of a reference trajectory, EmergencyStop checks for imminent
collisions in the robot's immediate vicinity.

This separation of concerns ensures that:
    - TrajectorySafetyChecker: Validates planned paths (proactive safety)
    - EmergencyStop: Provides a hard safety gate (reactive safety)

Classes:
    EmergencyStopParams: Configuration for the emergency stop bubble.
    EmergencyStopResult: Result of an emergency stop check.
    EmergencyStop: The emergency stop checker class.

Example:
    >>> from sparx_agency.core.planning.safety.emergency_stop import (
    ...     EmergencyStop,
    ...     EmergencyStopParams,
    ... )
    >>>
    >>> estop = EmergencyStop(EmergencyStopParams(radius_m=0.25))
    >>> result = estop.check(robot_state, local_map)
    >>> if result.should_stop:
    ...     halt_all_motors()

Notes:
    - This module is intentionally minimal and can be extended as needed.
    - The emergency stop is independent of any trajectory or path planning.
    - It should be checked at high frequency in the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sparx_agency.core.common.types import State3D

from .types import SafetyStatus, UnknownPolicy
from sparx_agency.core.planning.safety.adapters import query_tube


@dataclass(frozen=True)
class EmergencyStopParams:
    """
    Configuration parameters for the emergency stop safety bubble.

    The emergency stop checks a spherical region around the robot's current
    position. If any obstacle is detected within this bubble, an emergency
    stop is triggered.

    Attributes:
        radius_m: Radius (meters) of the safety bubble around the robot.
            Should be at least as large as the robot's physical radius plus
            stopping distance at current velocity. Default: 0.20.
        unknown_policy: How to treat unknown cells within the bubble.
            Default: UnknownPolicy.BLOCK (conservative - stop on unknown).

    Example:
        >>> params = EmergencyStopParams(
        ...     radius_m=0.30,  # 30cm safety bubble
        ...     unknown_policy=UnknownPolicy.BLOCK,
        ... )

    Notes:
        - The radius should account for:
            - Robot physical dimensions
            - Braking distance at maximum speed
            - Sensor uncertainty / latency
        - Using BLOCK for unknown_policy is recommended for safety-critical
          applications.
    """

    radius_m: float = 0.20
    unknown_policy: UnknownPolicy = UnknownPolicy.BLOCK


@dataclass(frozen=True)
class EmergencyStopResult:
    """
    Result of an emergency stop check.

    This dataclass encapsulates the outcome of checking the safety bubble
    around the robot.

    Attributes:
        should_stop: True if an emergency stop should be triggered, False
            if the area is clear.
        status: The underlying SafetyStatus from the tube query.
        message: Human-readable description of the result. Default: "".

    Example:
        >>> result = estop.check(state, local_map)
        >>> if result.should_stop:
        ...     print(f"EMERGENCY STOP: {result.message}")
        ...     print(f"Status: {result.status}")
    """

    should_stop: bool
    status: SafetyStatus
    message: str = ""


class EmergencyStop:
    """
    Simple bubble-based emergency stop gate.

    This class provides a reactive safety layer that checks for obstacles
    in the immediate vicinity of the robot. Unlike trajectory validation,
    which looks ahead along a planned path, emergency stop checks a fixed
    bubble around the current position.

    The emergency stop should be checked at high frequency (e.g., 50-100 Hz)
    in the control loop, independently of trajectory planning.

    Attributes:
        None publicly exposed. Use check() method to perform safety checks.

    Example:
        >>> from sparx_agency.core.planning.safety.emergency_stop import (
        ...     EmergencyStop,
        ...     EmergencyStopParams,
        ... )
        >>>
        >>> # Create with custom parameters
        >>> estop = EmergencyStop(EmergencyStopParams(
        ...     radius_m=0.25,
        ...     unknown_policy=UnknownPolicy.BLOCK,
        ... ))
        >>>
        >>> # Or use defaults
        >>> estop = EmergencyStop()
        >>>
        >>> # In control loop:
        >>> result = estop.check(robot_state, local_map)
        >>> if result.should_stop:
        ...     actuator.emergency_stop()
        ...     logger.warn(f"E-stop triggered: {result.message}")

    Notes:
        - This is a stateless checker - each call is independent.
        - The check is fast (single tube query) and suitable for high-frequency use.
        - For trajectory-aware safety checking, use TrajectorySafetyChecker instead.
    """

    def __init__(self, params: Optional[EmergencyStopParams] = None) -> None:
        """
        Initialize the emergency stop checker.

        Args:
            params: Configuration parameters. If None, default parameters
                are used (20cm radius, BLOCK unknown policy).
        """
        self._p = params or EmergencyStopParams()

    def check(self, state: State3D, local_map: Any) -> EmergencyStopResult:
        """
        Check if an emergency stop should be triggered.

        This method queries the local map for obstacles within the safety
        bubble centered at the robot's current position.

        Args:
            state: Current robot state containing at minimum the pose (x, y, z).
            local_map: The local map for collision checking. Supported types:
                - Costmap2D
                - OccupancyGrid2D
                - VoxelMap3D (duck-typed)

        Returns:
            EmergencyStopResult with:
                - should_stop: True if obstacle detected (stop immediately)
                - status: The underlying SafetyStatus
                - message: Description ("clear" or "imminent collision risk")

        Example:
            >>> result = estop.check(robot_state, voxel_map)
            >>> if result.should_stop:
            ...     motor_controller.halt()
            ...     alert_system.trigger(result.message)
            >>> else:
            ...     continue_normal_operation()

        Notes:
            - Only SafetyStatus.CLEAR results in should_stop=False
            - All other statuses (BLOCKED, UNKNOWN with BLOCK policy,
              OUT_OF_BOUNDS) trigger an emergency stop
        """
        p = self._p
        status, _ = query_tube(
            local_map=local_map,
            x=state.pose.x,
            y=state.pose.y,
            z=state.pose.z,
            radius_m=p.radius_m,
            unknown_policy=p.unknown_policy,
        )

        if status == SafetyStatus.CLEAR:
            return EmergencyStopResult(
                should_stop=False,
                status=status,
                message="clear",
            )

        return EmergencyStopResult(
            should_stop=True,
            status=status,
            message="imminent collision risk",
        )