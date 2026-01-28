"""
Real-time trajectory safety validation.

This module provides the TrajectorySafetyChecker class, which validates a short
prefix (lookahead corridor) of a reference trajectory against a local map. It
is designed for real-time use in the control loop to detect imminent collisions
before they occur.

Classes:
    TrajectorySafetyChecker: Main class for validating trajectory safety.

Dependencies:
    - State3D, Trajectory from sparx_agency.core.common.types
    - SafetyCheckResult, SafetyStatus, TrajectorySafetyParams, UnknownPolicy from .types
    - nearest_index_xyz, sample_trajectory from .trajectory_sampling
    - query_tube from .adapters

Example:
    >>> from sparx_agency.core.planning.safety import (
    ...     TrajectorySafetyChecker,
    ...     TrajectorySafetyParams,
    ...     SafetyStatus,
    ... )
    >>>
    >>> checker = TrajectorySafetyChecker(TrajectorySafetyParams(
    ...     lookahead_distance_m=3.0,
    ...     tube_radius_m=0.30,
    ... ))
    >>> result = checker.check(current_state, trajectory, local_map)
    >>> if result.status != SafetyStatus.CLEAR:
    ...     trigger_replanning()
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from sparx_agency.core.common.types import State3D, Trajectory

from .types import SafetyCheckResult, SafetyStatus, TrajectorySafetyParams, UnknownPolicy
from .trajectory_sampling import nearest_index_xyz, sample_trajectory
from sparx_agency.core.planning.safety.adapters import query_tube


class TrajectorySafetyChecker:
    """
    Real-time validator for a short trajectory prefix (lookahead corridor).

    This checker is designed to run in the control loop to validate that the
    upcoming portion of a reference trajectory is collision-free. It samples
    the trajectory at discrete points and checks a safety tube around each
    sample against the local map.

    The validation process:
        1. Sample the trajectory at fixed time intervals
        2. Find the robot's current position along the trajectory (progress)
        3. Iterate forward from current position up to lookahead limits
        4. At each sample, query a safety tube against the map
        5. Return on first collision or after completing the lookahead

    Attributes:
        params (TrajectorySafetyParams): Configuration parameters for the checker.

    Example:
        >>> params = TrajectorySafetyParams(
        ...     lookahead_distance_m=5.0,
        ...     lookahead_time_s=2.0,
        ...     tube_radius_m=0.25,
        ...     unknown_policy=UnknownPolicy.WARN,
        ... )
        >>> checker = TrajectorySafetyChecker(params)
        >>>
        >>> # In control loop:
        >>> result = checker.check(robot_state, planned_trajectory, local_map)
        >>> if result.status == SafetyStatus.BLOCKED:
        ...     print(f"Collision at {result.first_hit_point}")
        ...     initiate_emergency_stop()
    """

    def __init__(self, params: TrajectorySafetyParams | None = None) -> None:
        """
        Initialize the trajectory safety checker.

        Args:
            params: Configuration parameters. If None, default parameters
                are used (see TrajectorySafetyParams for defaults).
        """
        self._p = params or TrajectorySafetyParams()

    @property
    def params(self) -> TrajectorySafetyParams:
        """
        Get the current configuration parameters.

        Returns:
            The TrajectorySafetyParams instance used by this checker.
        """
        return self._p

    def check(
        self,
        state: State3D,
        trajectory: Trajectory,
        local_map: Any,
    ) -> SafetyCheckResult:
        """
        Validate the trajectory prefix for collisions.

        This method samples the trajectory, locates the robot's current
        progress, and checks each subsequent sample within the lookahead
        horizon against the local map.

        Args:
            state: Current robot state (pose and optionally velocity).
            trajectory: The reference trajectory to validate.
            local_map: The local map for collision checking. Supported types:
                - Costmap2D
                - OccupancyGrid2D
                - VoxelMap3D (duck-typed)

        Returns:
            SafetyCheckResult containing:
                - status: CLEAR, BLOCKED, UNKNOWN, or OUT_OF_BOUNDS
                - message: Human-readable description
                - progress_t: Trajectory time at robot's current position
                - first_hit_s: Arc-length to first collision (if any)
                - first_hit_point: World coordinates of collision (if any)

        Example:
            >>> result = checker.check(state, trajectory, costmap)
            >>> if result.status == SafetyStatus.CLEAR:
            ...     continue_execution()
            >>> elif result.status == SafetyStatus.BLOCKED:
            ...     print(f"Obstacle at s={result.first_hit_s:.2f}m")
            ...     replan_trajectory()
            >>> elif result.status == SafetyStatus.UNKNOWN:
            ...     reduce_speed()  # Proceed with caution

        Notes:
            - The effective tube radius is: tube_radius_m + tube_extra_m
            - Checking stops at whichever horizon is reached first:
              lookahead_distance_m or lookahead_time_s
            - If the trajectory has fewer than 2 samples, returns BLOCKED
              with an appropriate message
        """
        p = self._p
        tube_r = max(0.0, p.tube_radius_m + p.tube_extra_m)

        # Sample the trajectory
        pts = sample_trajectory(trajectory, dt=p.sample_dt_s, max_samples=p.max_samples)
        if len(pts) < 2:
            return SafetyCheckResult(
                SafetyStatus.BLOCKED,
                "Trajectory sampling produced <2 points",
            )

        # Find current progress along trajectory
        x0, y0, z0 = state.pose.x, state.pose.y, state.pose.z
        i0 = nearest_index_xyz(pts, x0, y0, z0)
        t0 = pts[i0].t

        checked_s = 0.0
        unknown_seen = False

        prev = pts[i0]
        for i in range(i0, len(pts)):
            pt = pts[i]

            # Check time horizon
            if p.lookahead_time_s is not None and (pt.t - t0) > p.lookahead_time_s:
                break

            # Check distance horizon
            if checked_s > p.lookahead_distance_m:
                break

            # Update arc-length (3D distance; 2D maps ignore z in adapter)
            if i > i0:
                dx = pt.x - prev.x
                dy = pt.y - prev.y
                dz = pt.z - prev.z
                checked_s += sqrt(dx * dx + dy * dy + dz * dz)
                prev = pt

            # Query safety tube at this sample
            status, saw_unknown = query_tube(
                local_map=local_map,
                x=pt.x,
                y=pt.y,
                z=pt.z,
                radius_m=tube_r,
                unknown_policy=p.unknown_policy,
            )
            unknown_seen = unknown_seen or saw_unknown

            if status == SafetyStatus.CLEAR:
                continue

            if status == SafetyStatus.UNKNOWN and p.unknown_policy == UnknownPolicy.WARN:
                # Keep scanning: we still want to early-fail on hard collisions
                continue

            # BLOCKED / OUT_OF_BOUNDS or UNKNOWN with BLOCK policy
            return SafetyCheckResult(
                status=status,
                message=f"Trajectory unsafe at s≈{checked_s:.2f}m",
                progress_t=t0,
                first_hit_s=checked_s,
                first_hit_point=(pt.x, pt.y, pt.z),
            )

        # Completed lookahead without hard collision
        if unknown_seen and p.unknown_policy == UnknownPolicy.WARN:
            return SafetyCheckResult(
                status=SafetyStatus.UNKNOWN,
                message="UNKNOWN encountered in lookahead corridor",
                progress_t=t0,
            )

        return SafetyCheckResult(
            SafetyStatus.CLEAR,
            "Trajectory prefix is clear",
            progress_t=t0,
        )