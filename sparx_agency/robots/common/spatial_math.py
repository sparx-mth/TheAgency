# Compatibility shim — module was moved to core/common/spatial_math.
# Re-export everything so existing imports keep working.
from sparx_agency.core.common.spatial_math import *  # noqa: F401, F403
from sparx_agency.core.common.spatial_math import (
    euler_to_rot_zyx,
    rpy_deg_to_rot,
    rot_y_deg,
    rpy_to_transform,
    quat_to_rot,
    quat_to_transform,
    rot_to_quat,
    rot_to_rpy,
    quat_msg_to_rpy_deg,
    quat_to_yaw,
    yaw_to_quat,
    pose_xyz_yaw_to_T,
    transform_to_pose,
    fov_to_intrinsics,
    yaml_to_intrinsics,
    open3d_pose_to_ros_pose,
    points_open3d_to_ros,
)
