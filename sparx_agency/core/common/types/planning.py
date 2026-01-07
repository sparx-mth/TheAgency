# core/common/types/planning.py
"""
Planning-related types.

Includes geometric paths, time-parameterized trajectory interfaces, and plan results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
from .geometry import Pose2D
from .primitives import _assert_finite


# ---------------------------------------------------------------------
# Path (planner output)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Path2D:
    """
    Geometric path (NO velocities).
    """
    points: Tuple[Pose2D, ...]
    frame_id: str = "map"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("Path2D must contain at least 2 points")

    def length(self) -> float:
        total = 0.0
        for a, b in zip(self.points[:-1], self.points[1:]):
            total += a.distance_to(b)
        return total

    def start(self) -> Pose2D:
        return self.points[0]

    def goal(self) -> Pose2D:
        return self.points[-1]


# ---------------------------------------------------------------------
# Trajectory (smoother output)
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    z: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    yaw: Optional[float] = None
    yaw_rate: Optional[float] = None

    s: Optional[float] = None
    curvature: Optional[float] = None

    def __post_init__(self):
        _assert_finite("TrajectoryPoint.t", self.t)
        _assert_finite("TrajectoryPoint.x", self.x)
        _assert_finite("TrajectoryPoint.y", self.y)
        _assert_finite("TrajectoryPoint.z", self.z)


@runtime_checkable
class Trajectory(Protocol):
    """
    Time-parameterized trajectory.
    """

    @property
    def total_time(self) -> float: ...

    @property
    def start(self) -> Tuple[float, float, float]: ...

    @property
    def end(self) -> Tuple[float, float, float]: ...

    def sample(self, t: float) -> TrajectoryPoint: ...

    def sample_by_time(self, dt: float) -> List[TrajectoryPoint]: ...


# ---------------------------------------------------------------------
# Planner result
# ---------------------------------------------------------------------

class PlanStatus(str, Enum):
    SUCCESS = "success"
    NO_PATH = "no_path"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    ERROR = "error"


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    path: Optional[Path2D] = None
    message: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PlanStatus.SUCCESS and self.path is not None
