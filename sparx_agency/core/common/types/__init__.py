"""Core data types for robotics planning, control, and perception."""
from .primitives import Number, Coord2D, Coord3D, Index2D, Index3D, Vec2, Vec3
from .geometry import normalize_angle, Pose2D, Pose3D
from .perception import Intrinsics, PoseSE3, RGBFrame, DepthFrame, PointCloud, Observation
from .motion import Twist2D, Twist3D, Accel2D, Accel3D, State2D, State3D
from .planning import Path2D, Path3D, TrajectoryPoint, Trajectory, PlanStatus, PlanResult
from .control import ControlMode, ControlCommand, KinematicLimits

__all__ = [
    # Primitives
    "Number", "Coord2D", "Coord3D", "Index2D", "Index3D", "Vec2", "Vec3",
    # Geometry
    "normalize_angle", "Pose2D", "Pose3D",
    # Perception
    "Intrinsics", "PoseSE3", "RGBFrame", "DepthFrame", "PointCloud", "Observation",
    # Motion
    "Twist2D", "Twist3D", "Accel2D", "Accel3D", "State2D", "State3D",
    # Planning
    "Path2D", "Path3D", "TrajectoryPoint", "Trajectory", "PlanStatus", "PlanResult",
    # Control
    "ControlMode", "ControlCommand", "KinematicLimits",
]