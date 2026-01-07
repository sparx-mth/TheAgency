# core/common/types/__init__.py

from .primitives import (
    Number, Coord2D, Coord3D, Index2D, Index3D,
    Vec2, Vec3,
)

from .geometry import (
    Pose2D, Pose3D,
    normalize_angle,
)

from .motion import (
    Twist2D, Twist3D,
    Accel2D, Accel3D,
    State2D, State3D,
)

from .planning import (
    Path2D,
    TrajectoryPoint, Trajectory,
    PlanStatus, PlanResult,
)

from .control import (
    ControlMode, ControlCommand,
    KinematicLimits,
)
