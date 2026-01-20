import numpy as np

from sparx_agency.core.common.types import PoseSE3
from sparx_agency.robots.common.spatial_math import euler_to_rot, quat_to_rot

from nav_msgs.msg import Odometry


def uav_state_to_pose_se3(msg) -> PoseSE3:
    # 1. Position (Translation)
    # Using relative altitude if position X,Y are 0
    t = np.array([
        float(msg.position.x),
        float(msg.position.y),
        float(msg.position.z)
    ], dtype=np.float32)

    # 2. Rotation Matrix
    # msg.azimuth is Yaw
    R_matrix = euler_to_rot(
        roll=float(msg.roll),
        pitch=float(msg.pitch),
        yaw=float(msg.azimuth)
    )

    return PoseSE3(R=R_matrix, t=t)


def odom_to_pose_se3(odom: Odometry) -> PoseSE3:
    p = odom.pose.pose.position
    o = odom.pose.pose.orientation
    R = quat_to_rot(float(o.x), float(o.y), float(o.z), float(o.w))
    t = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)
    return PoseSE3(R=R, t=t)
