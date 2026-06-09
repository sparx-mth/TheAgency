import math

import numpy as np
import yaml

from sparx_agency.core.common.types import Intrinsics


def euler_to_rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Manual conversion from Euler angles (Roll, Pitch, Yaw/Azimuth) to 3x3 Rotation Matrix.
    Uses the ZYX convention (standard for aircraft/drones).
    """
    # Precompute sines and cosines
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # ZYX rotation matrix components
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=np.float32)

    return R


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Returns 3x3 rotation matrix
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float32)


def get_euler(q):
    # Quaternion to Euler (Roll, Pitch, Yaw)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def yaw_to_quaternion(yaw_rad: float):
    """Quaternion for yaw-only rotation around Z."""
    half = 0.5 * yaw_rad
    qz = math.sin(half)
    qw = math.cos(half)
    return 0.0, 0.0, qz, qw



def intrinsics_from_fov(width: int, height: int, hfov_deg: float, vfov_deg: float) -> Intrinsics:
    """Converts FOV angles to a pinhole camera intrinsic matrix."""
    hfov = math.radians(float(hfov_deg))
    vfov = math.radians(float(vfov_deg))

    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = (height / 2.0) / math.tan(vfov / 2.0)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # Return the dataclass object, not a raw tuple
    return Intrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy)
    )


def load_intrinsics_from_yaml(yaml_path: str) -> Intrinsics:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return Intrinsics(
        fx=float(data['fx']),
        fy=float(data['fy']),
        cx=float(data['cx']),
        cy=float(data['cy']),
        width=int(data['image_width']),
        height=int(data['image_height'])
    )

def rpy_deg_to_R_base(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Rotation matrix for roll/pitch/yaw defined about *base axes*:
      roll  around X (forward)
      pitch around Y (left)
      yaw   around Z (up)

    Returns R (3x3) such that: p_rot = R @ p
    """
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    # right-handed, axes: X forward, Y left, Z up
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float32)


def pose_xyz_yaw_to_T(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """
    Create a 4x4 homogeneous transform matrix from position (x,y,z) and yaw (rad).
    Assumes:
      - yaw around +Z axis
      - right-handed coordinate system
    """
    c = math.cos(yaw)
    s = math.sin(yaw)

    T = np.eye(4, dtype=np.float32)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c

    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z

    return T


def open3d_pose_to_ros_pose(T_o3d: np.ndarray) -> np.ndarray:
    """
    Convert Open3D camera-coordinate trajectory to ROS/RViz-friendly coordinates.

    Open3D/camera:
      x = right
      y = down
      z = forward

    ROS/RViz:
      x = forward
      y = left
      z = up
    """
    R_o3d_to_ros = np.array([
        [0.0, 0.0, 1.0],  # ros x = o3d z
        [-1.0, 0.0, 0.0],  # ros y = -o3d x
        [0.0, -1.0, 0.0],  # ros z = -o3d y
    ], dtype=np.float64)

    T_ros = np.eye(4, dtype=np.float64)
    T_ros[:3, :3] = R_o3d_to_ros @ T_o3d[:3, :3]
    T_ros[:3, 3] = R_o3d_to_ros @ T_o3d[:3, 3]

    return T_ros


def points_open3d_to_ros(points_o3d: np.ndarray, initial_height_m: float = 0.0) -> np.ndarray:
    """
    Convert points from Open3D/camera-world coordinates to ROS/RViz coordinates.

    Open3D:
      x = right
      y = down
      z = forward

    ROS:
      x = forward
      y = left
      z = up
    """
    R = np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)

    pts_ros = (R @ points_o3d.T).T
    pts_ros[:, 2] += float(initial_height_m)
    return pts_ros


