"""Planning-related types: paths, trajectories, and plan results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .geometry import Pose2D
from .primitives import _assert_finite


@dataclass(frozen=True)
class Path2D:
    """
    Geometric path without time parameterization.

    Output of planners, input to smoothers.

    Attributes:
        points: Sequence of 2D poses defining the path.
        frame_id: Coordinate frame identifier.
        metadata: Optional algorithm-specific data.
    """
    points: Tuple[Pose2D, ...]
    frame_id: str = "map"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Path2D requires at least 2 points")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def start(self) -> Pose2D:
        return self.points[0]

    @property
    def goal(self) -> Pose2D:
        return self.points[-1]

    def length(self) -> float:
        """Total path length in meters."""
        return sum(
            a.distance_to(b)
            for a, b in zip(self.points[:-1], self.points[1:])
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    """
    Single point on a time-parameterized trajectory.

    Core fields (t, x, y, z) are required. Velocities, accelerations,
    and auxiliary fields (yaw, curvature, arc length) are optional.
    """
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

    s: Optional[float] = None  # Arc length from start
    curvature: Optional[float] = None  # Path curvature at this point

    def __post_init__(self) -> None:
        _assert_finite("TrajectoryPoint.t", self.t)
        _assert_finite("TrajectoryPoint.x", self.x)
        _assert_finite("TrajectoryPoint.y", self.y)
        _assert_finite("TrajectoryPoint.z", self.z)

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    @property
    def acceleration(self) -> Tuple[float, float, float]:
        return (self.ax, self.ay, self.az)


@runtime_checkable
class Trajectory(Protocol):
    """
    Time-parameterized trajectory protocol.

    Implemented by smoother outputs. Supports both continuous
    sampling and discrete iteration.
    """

    @property
    def total_time(self) -> float:
        """Duration from first to last sample."""
        ...

    @property
    def start(self) -> Tuple[float, float, float]:
        """Starting position (x, y, z)."""
        ...

    @property
    def end(self) -> Tuple[float, float, float]:
        """Ending position (x, y, z)."""
        ...

    def sample(self, t: float) -> TrajectoryPoint:
        """Sample trajectory at time t (with interpolation if needed)."""
        ...

    def sample_by_time(self, dt: float) -> List[TrajectoryPoint]:
        """Sample trajectory at uniform time intervals."""
        ...


class PlanStatus(str, Enum):
    """Planning outcome status."""
    SUCCESS = "success"
    NO_PATH = "no_path"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class PlanResult:
    """
    Planner output containing status and optional path.

    Attributes:
        status: Outcome of planning attempt.
        path: Resulting path if successful, None otherwise.
        message: Human-readable status message.
        artifacts: Algorithm-specific debug data.
    """
    status: PlanStatus
    path: Optional[Path2D] = None
    message: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if planning succeeded with a valid path."""
        return self.status == PlanStatus.SUCCESS and self.path is not None
