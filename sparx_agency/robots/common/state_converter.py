import numpy as np

from sparx_agency.core.common.types import PoseSE3, Intrinsics, Pose3D, State3D, Twist3D
from sparx_agency.robots.common.spatial_math import euler_to_rot_zyx, quat_to_rot

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo



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
    R_matrix = euler_to_rot_zyx(
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


def cam_info_to_intrinsics(ci: CameraInfo) -> Intrinsics:
    fx = float(ci.k[0])
    fy = float(ci.k[4])
    cx = float(ci.k[2])
    cy = float(ci.k[5])
    return Intrinsics(
        width=int(ci.width),
        height=int(ci.height),
        fx=fx, fy=fy, cx=cx, cy=cy
    )

def costmap_to_occupancygrid(costmap, stamp, frame_id: str) -> OccupancyGrid:
    """
    Convert ROS-free costmap (ProbabilisticGridCostmap) into nav_msgs/OccupancyGrid.
    """
    spec, grid = costmap.get_grid()  # GridSpec + (H,W) int8
    msg = OccupancyGrid()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.info.resolution = float(spec.resolution_m)
    msg.info.width = int(spec.width)
    msg.info.height = int(spec.height)

    msg.info.origin.position.x = float(spec.origin_x)
    msg.info.origin.position.y = float(spec.origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.x = 0.0
    msg.info.origin.orientation.y = 0.0
    msg.info.origin.orientation.z = 0.0
    msg.info.origin.orientation.w = 1.0

    msg.data = np.array(grid, dtype=np.int8).flatten().tolist()

    return msg

