"""
Local planning types (scoped to core.planning.local_planners).

Alignment rules with core.planning.safety:
- SafetyStatus / SafetyCheckResult are the single source of truth for
  "is the current reference safe?" questions.
- The local planner focuses on "produce an alternative short-horizon
  trajectory" and returns its own local planning status.
- Horizon and safety tube parameters are reused from TrajectorySafetyParams
  to avoid duplicated configuration knobs.

This module intentionally reuses shared types from core.common.types:
- motion.State3D
- planning.Path2D / Path3D / Trajectory
- control.KinematicLimits / ControlCommand
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

from sparx_agency.core.common.types.control import ControlCommand, KinematicLimits
from sparx_agency.core.common.types.motion import State3D
from sparx_agency.core.common.types.planning import Path2D, Path3D, Trajectory
from sparx_agency.core.planning.safety.types import SafetyCheckResult, TrajectorySafetyParams


class LocalPlanStatus(str, Enum):
    """
    Outcome of a local planning attempt.

    Notes:
        - "blocked" is not a local-planner status; it is reported by the safety checker.
        - If the reference is blocked and a detour cannot be found, return NO_SOLUTION.
    """
    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    TIMEOUT = "timeout"
    ERROR = "error"


class LocalFailureReason(str, Enum):
    """
    Compact reason codes for debugging / telemetry.

    Keep this stable; algorithm-specific details belong in artifacts.
    """
    START_IN_COLLISION = "start_in_collision"
    REFERENCE_TOO_SHORT = "reference_too_short"
    GOAL_UNREACHABLE = "goal_unreachable"
    DYNAMICS_INFEASIBLE = "dynamics_infeasible"
    OPTIMIZATION_DIVERGED = "optimization_diverged"
    INTERNAL_ERROR = "internal_error"


# The local planner can accept either a geometric reference path (2D/3D)
# or a time-parameterized reference trajectory (e.g., smoother output).
LocalReference = Union[Path2D, Path3D, Trajectory]


@runtime_checkable
class CollisionChecker(Protocol):
    """
    Minimal collision checking contract for local planners.

    This avoids committing to a map format (2D occupancy, 3D voxels, ESDF, etc.).
    Implement adapters around your existing safety_maps/collision_maps dispatchers.

    The planner typically needs:
    - point collision test (inflated by safety tube radius)
    - coarse segment collision test for pruning
    """

    def is_position_free(self, x: float, y: float, z: float, radius_m: float) -> bool:
        """Return True if a sphere (radius_m) at position is collision-free."""
        ...

    def is_segment_free(
        self,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        radius_m: float,
        step_m: float = 0.1,
    ) -> bool:
        """
        Return True if the swept segment is collision-free.

        Implementations may discretize along the segment with step_m.
        """
        ...


@dataclass(frozen=True)
class LocalPlanInput:
    """
    Input to a local planner.

    Attributes:
        state: Current drone state (pose + twist).
        reference: Global/smoothed reference to follow (path or trajectory).
        safety_params: Shared horizon + tube configuration (reused from safety checker).
        limits: Kinematic constraints used by sampling/optimization.
        collision: CollisionChecker for local map + dynamic obstacles (already fused).
        last_safety: Optional recent SafetyCheckResult for the reference.
            - If provided and indicates a hit, algorithms can use first_hit_* as a hint.
        frame_id: Frame for all geometry (default: "map").
        metadata: Optional extras (predicted obstacles, preferred altitude band, etc.).
    """
    state: State3D
    reference: LocalReference
    safety_params: TrajectorySafetyParams
    limits: KinematicLimits
    collision: CollisionChecker

    last_safety: Optional[SafetyCheckResult] = None
    frame_id: str = "map"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def inflation_radius_m(self) -> float:
        """Effective collision inflation radius used by both safety and local planning."""
        return self.safety_params.tube_radius_m + self.safety_params.tube_extra_m

    @property
    def horizon_distance_m(self) -> float:
        """Distance horizon (meters) reused from safety params."""
        return self.safety_params.lookahead_distance_m

    @property
    def horizon_time_s(self) -> Optional[float]:
        """Time horizon (seconds) reused from safety params."""
        return self.safety_params.lookahead_time_s


@dataclass(frozen=True)
class LocalPlanOutput:
    """
    Output of a local planning attempt.

    Preferred output is a short time-parameterized trajectory. In emergencies,
    the manager above the planner may choose to apply a fallback command.

    Attributes:
        status: Local planning status.
        trajectory: Short horizon trajectory if successful, else None.
        fallback: Optional immediate safe control command (e.g., hover/stop).
        reason: Optional structured failure reason.
        message: Human-readable message.
        artifacts: Algorithm-specific debug payload.
    """
    status: LocalPlanStatus
    trajectory: Optional[Trajectory] = None
    fallback: Optional[ControlCommand] = None
    reason: Optional[LocalFailureReason] = None
    message: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if a valid trajectory is provided."""
        return self.status == LocalPlanStatus.SUCCESS and self.trajectory is not None
