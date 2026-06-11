"""
Type definitions for the safety validation system.

This module defines the core data types used throughout the safety package,
including enums for policies and statuses, as well as dataclasses for
configuration parameters and results.

Classes:
    UnknownPolicy: Enum defining how to handle unknown cells in occupancy maps.
    SafetyStatus: Enum representing the outcome of safety checks.
    TrajectorySafetyParams: Configuration parameters for trajectory validation.
    SafetyCheckResult: Result container for trajectory safety checks.

Example:
    >>> from sparx_agency.core.planning.safety.types import (
    ...     TrajectorySafetyParams,
    ...     UnknownPolicy,
    ... )
    >>> params = TrajectorySafetyParams(
    ...     lookahead_distance_m=3.0,
    ...     tube_radius_m=0.3,
    ...     unknown_policy=UnknownPolicy.WARN,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class UnknownPolicy(str, Enum):
    """
    Policy for handling UNKNOWN cells in maps that support them.

    This enum determines how the safety checker treats cells with unknown
    occupancy status (e.g., unexplored regions in OccupancyGrid2D).

    Attributes:
        BLOCK: Treat unknown cells as obstacles (conservative/safe).
        ALLOW: Treat unknown cells as free space (permissive/risky).
        WARN: Report UNKNOWN status if no hard collision is found,
              allowing the caller to decide how to proceed.

    Example:
        >>> policy = UnknownPolicy.WARN
        >>> if policy == UnknownPolicy.BLOCK:
        ...     print("Unknown regions are treated as blocked")
    """

    BLOCK = "block"
    ALLOW = "allow"
    WARN = "warn"


class SafetyStatus(str, Enum):
    """
    Outcome status for a safety check on a corridor or tube.

    This enum represents the result of checking whether a position or
    trajectory segment is safe for navigation.

    Attributes:
        CLEAR: The checked region is free of obstacles.
        BLOCKED: An obstacle was detected in the checked region.
        UNKNOWN: The region contains unknown cells (only returned when
                 using UnknownPolicy.WARN and no hard collision found).
        OUT_OF_BOUNDS: The query position is outside the map boundaries.

    Example:
        >>> status = SafetyStatus.CLEAR
        >>> if status == SafetyStatus.CLEAR:
        ...     print("Safe to proceed")
    """

    CLEAR = "clear"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass(frozen=True)
class TrajectorySafetyParams:
    """
    Configuration parameters for real-time trajectory safety checking.

    This dataclass defines all tunable parameters for the trajectory safety
    checker, including lookahead distances, sampling rates, and the safety
    tube dimensions.

    Attributes:
        lookahead_distance_m: Maximum arc-length distance (meters) to check
            along the trajectory from the current position. Default: 5.0.
        lookahead_time_s: Maximum time horizon (seconds) to check along the
            trajectory. If None, only distance-based cutoff is used. Default: 1.5.
        sample_dt_s: Time step (seconds) for sampling trajectory points.
            Smaller values increase accuracy but also computation. Default: 0.05.
        tube_radius_m: Base radius (meters) of the safety tube around the
            trajectory. Should account for the robot's physical radius. Default: 0.25.
        tube_extra_m: Additional margin (meters) added to tube_radius_m to
            account for tracking error or extra safety margin. Default: 0.0.
        unknown_policy: Policy for handling unknown cells in occupancy maps.
            Default: UnknownPolicy.WARN.
        max_samples: Maximum number of trajectory samples to check. Prevents
            excessive computation on long trajectories. Default: 600.

    Notes:
        - The effective tube radius is: tube_radius_m + tube_extra_m
        - The checker stops at whichever horizon is reached first:
          lookahead_distance_m or lookahead_time_s
        - For a drone, tube_radius_m should include the propeller sweep radius
          plus any desired safety margin

    Example:
        >>> params = TrajectorySafetyParams(
        ...     lookahead_distance_m=3.0,
        ...     lookahead_time_s=2.0,
        ...     tube_radius_m=0.30,
        ...     tube_extra_m=0.05,
        ...     unknown_policy=UnknownPolicy.BLOCK,
        ... )
    """

    lookahead_distance_m: float = 5.0
    lookahead_time_s: Optional[float] = 1.5
    sample_dt_s: float = 0.05

    tube_radius_m: float = 0.25
    tube_extra_m: float = 0.0

    unknown_policy: UnknownPolicy = UnknownPolicy.WARN
    max_samples: int = 600


@dataclass(frozen=True)
class SafetyCheckResult:
    """
    Result container for a trajectory safety check.

    This dataclass holds all information about the outcome of a safety check,
    including the status, diagnostic messages, and location of any detected
    obstacles.

    Attributes:
        status: The overall safety status (CLEAR, BLOCKED, UNKNOWN, or OUT_OF_BOUNDS).
        message: Human-readable description of the result. Default: "".
        progress_t: The trajectory time parameter at the robot's current position
            (i.e., where the check started along the trajectory). Default: None.
        first_hit_s: Arc-length distance (meters) from the starting position to
            the first detected obstacle. None if no obstacle found. Default: None.
        first_hit_point: World coordinates (x, y, z) of the first detected
            obstacle. None if no obstacle found. Default: None.

    Example:
        >>> result = SafetyCheckResult(
        ...     status=SafetyStatus.BLOCKED,
        ...     message="Obstacle detected at s≈2.5m",
        ...     progress_t=1.2,
        ...     first_hit_s=2.5,
        ...     first_hit_point=(3.0, 4.0, 1.0),
        ... )
        >>> if result.status != SafetyStatus.CLEAR:
        ...     print(f"Warning: {result.message}")
    """

    status: SafetyStatus
    message: str = ""
    progress_t: Optional[float] = None

    first_hit_s: Optional[float] = None
    first_hit_point: Optional[Tuple[float, float, float]] = None