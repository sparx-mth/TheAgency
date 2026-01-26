"""
Safety and validation utilities for planning and control.

This package provides real-time safety validation tools for autonomous systems,
including trajectory validation against local maps and emergency stop mechanisms.

Main Components:
    - **TrajectorySafetyChecker**: Validates a short prefix (lookahead corridor)
      of a reference trajectory against a local map. Designed for use in the
      control loop to detect upcoming collisions.

    - **Map Adapters**: Unified interface for querying safety tubes across
      different map representations (Costmap2D, OccupancyGrid2D, VoxelMap3D).
      See the `adapters` subpackage.

    - **EmergencyStop**: Independent bubble-based collision check for immediate
      vicinity. Provides a hard safety gate separate from trajectory validation.
      See `emergency_stop` module.

Classes:
    TrajectorySafetyChecker: Main trajectory validation class.

Enums:
    SafetyStatus: Outcome of a safety check (CLEAR, BLOCKED, UNKNOWN, OUT_OF_BOUNDS).
    UnknownPolicy: How to treat unknown map cells (BLOCK, ALLOW, WARN).

Dataclasses:
    TrajectorySafetyParams: Configuration for trajectory safety checking.
    SafetyCheckResult: Result container for safety checks.

Example:
    Basic trajectory safety checking::

        from sparx_agency.core.planning.safety import (
            TrajectorySafetyChecker,
            TrajectorySafetyParams,
            SafetyStatus,
            UnknownPolicy,
        )

        # Configure the checker
        params = TrajectorySafetyParams(
            lookahead_distance_m=5.0,
            lookahead_time_s=2.0,
            tube_radius_m=0.30,
            unknown_policy=UnknownPolicy.WARN,
        )
        checker = TrajectorySafetyChecker(params)

        # In control loop
        result = checker.check(robot_state, trajectory, local_map)

        if result.status == SafetyStatus.CLEAR:
            continue_execution()
        elif result.status == SafetyStatus.BLOCKED:
            trigger_replanning()
        elif result.status == SafetyStatus.UNKNOWN:
            reduce_speed()  # Proceed with caution

    Combined with emergency stop::

        from sparx_agency.core.planning.safety.emergency_stop import (
            EmergencyStop,
            EmergencyStopParams,
        )

        estop = EmergencyStop(EmergencyStopParams(radius_m=0.25))

        # High-frequency safety loop
        estop_result = estop.check(robot_state, local_map)
        if estop_result.should_stop:
            halt_motors_immediately()

Architecture Notes:
    The safety package is designed with separation of concerns:

    1. **Trajectory Validation** (proactive): Looks ahead along planned paths
       to detect future collisions, allowing time for replanning.

    2. **Emergency Stop** (reactive): Checks immediate vicinity independent
       of any trajectory, providing a hard safety gate.

    3. **Map Adapters**: Abstract away map representation details, allowing
       the safety logic to work with different map types (2D grids, 3D voxels,
       cost maps with clearance fields).

    The recommended integration pattern is:
    - Run trajectory validation at planning rate (e.g., 10-20 Hz)
    - Run emergency stop at control rate (e.g., 50-100 Hz)
    - Use appropriate unknown policies based on operational requirements

See Also:
    - adapters: Map-specific query implementations
    - emergency_stop: Independent collision avoidance
    - trajectory_sampling: Utilities for sampling trajectories
"""

from .types import (
    SafetyStatus,
    UnknownPolicy,
    TrajectorySafetyParams,
    SafetyCheckResult,
)
from .trajectory_checker import TrajectorySafetyChecker

__all__ = [
    "SafetyStatus",
    "UnknownPolicy",
    "TrajectorySafetyParams",
    "SafetyCheckResult",
    "TrajectorySafetyChecker",
]