import math

import numpy as np
import yaml

from sparx_agency.core.common.types import Intrinsics


# ── Rotation matrix builders ──────────────────────────────────────────────────

def euler_to_rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY angles (rad) → 3×3 rotation matrix, ZYX convention (R = Rz@Ry@Rx)."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ],
    ], dtype=np.float32)


def rpy_deg_to_rot(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """RPY angles (degrees) → 3×3 rotation matrix, ZYX convention."""
    r, p, y = np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0,   0  ], [0, cr, -sr], [0,   sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0,  sp ], [0, 1,   0 ], [-sp,  0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0 ], [sy, cy,  0], [0,    0,  1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def rot_y_deg(deg: float) -> np.ndarray:
    """Rotation matrix around Y axis (degrees). Axes: X=forward, Y=left, Z=up."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float32)


def rpy_to_transform(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY angles (rad) → 4×4 homogeneous rotation matrix, ZYX convention."""
    sr, cr = math.sin(roll),  math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw),   math.cos(yaw)
    Rx = np.array([[1, 0,   0  ], [0, cr, -sr], [0,   sr, cr]], dtype=float)
    Ry = np.array([[cp, 0,  sp ], [0, 1,   0 ], [-sp,  0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0 ], [sy, cy,  0], [0,    0,  1]], dtype=float)
    M = np.eye(4, dtype=float)
    M[:3, :3] = Rz @ Ry @ Rx
    return M


# ── Quaternion ↔ rotation matrix ──────────────────────────────────────────────

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (separate floats, xyzw) → 3×3 rotation matrix (no normalisation)."""
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)    ],
        [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)    ],
        [2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)],
    ], dtype=np.float32)


def quat_to_transform(q) -> np.ndarray:
    """Quaternion [x,y,z,w] (iterable) → normalised 4×4 homogeneous rotation matrix."""
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        return np.eye(4, dtype=float)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    M = np.eye(4, dtype=float)
    M[:3, :3] = np.array([
        [1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)    ],
        [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)    ],
        [2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)],
    ])
    return M


def rot_to_quat(M: np.ndarray) -> list:
    """3×3 or 4×4 rotation matrix → quaternion [x, y, z, w] (Shepperd method)."""
    R = M[:3, :3]
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


# ── Euler / RPY extraction ────────────────────────────────────────────────────

def rot_to_rpy(R: np.ndarray) -> tuple:
    """3×3 rotation matrix → (roll, pitch, yaw) in radians, ZYX convention."""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = math.atan2( R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def quat_msg_to_rpy_deg(q) -> tuple:
    """ROS quaternion message (has .w .x .y .z) → (roll, pitch, yaw) in degrees."""
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ── Pose / transform helpers ──────────────────────────────────────────────────

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Quaternion (xyzw) → yaw angle in radians (rotation around Z axis)."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def yaw_to_quat(yaw_rad: float) -> tuple:
    """Yaw angle (rad) → quaternion (qx=0, qy=0, qz, qw) for a pure Z-axis rotation."""
    half = 0.5 * yaw_rad
    return 0.0, 0.0, math.sin(half), math.cos(half)


def pose_xyz_yaw_to_T(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """(x, y, z, yaw_rad) → 4×4 homogeneous transform (yaw rotation around +Z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4, dtype=np.float32)
    T[0, 0] =  c;  T[0, 1] = -s;  T[0, 3] = x
    T[1, 0] =  s;  T[1, 1] =  c;  T[1, 3] = y
    T[2, 3] = z
    return T


def transform_to_pose(M: np.ndarray) -> tuple:
    """4×4 homogeneous transform → ((x, y, z), (qx, qy, qz, qw))."""
    x, y, z = float(M[0, 3]), float(M[1, 3]), float(M[2, 3])
    qx, qy, qz, qw = rot_to_quat(M)
    return (x, y, z), (qx, qy, qz, qw)


# ── Intrinsics helpers ────────────────────────────────────────────────────────

def fov_to_intrinsics(width: int, height: int, hfov_deg: float, vfov_deg: float) -> Intrinsics:
    """FOV angles (degrees) + image size → pinhole Intrinsics dataclass."""
    hfov = math.radians(float(hfov_deg))
    vfov = math.radians(float(vfov_deg))
    fx = (width  / 2.0) / math.tan(hfov / 2.0)
    fy = (height / 2.0) / math.tan(vfov / 2.0)
    return Intrinsics(
        width=int(width), height=int(height),
        fx=float(fx), fy=float(fy),
        cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
    )


def yaml_to_intrinsics(yaml_path: str) -> Intrinsics:
    """Simple YAML with raw fx/fy/cx/cy fields → Intrinsics dataclass."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Intrinsics(
        fx=float(data["fx"]), fy=float(data["fy"]),
        cx=float(data["cx"]), cy=float(data["cy"]),
        width=int(data["image_width"]), height=int(data["image_height"]),
    )


# ── Open3D ↔ ROS coordinate conversions ──────────────────────────────────────

def open3d_pose_to_ros_pose(T_o3d: np.ndarray) -> np.ndarray:
    """4×4 pose in Open3D camera coords (x=right,y=down,z=fwd) → ROS (x=fwd,y=left,z=up)."""
    R = np.array([
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    T_ros = np.eye(4, dtype=np.float64)
    T_ros[:3, :3] = R @ T_o3d[:3, :3]
    T_ros[:3, 3]  = R @ T_o3d[:3, 3]
    return T_ros


def points_open3d_to_ros(points_o3d: np.ndarray, initial_height_m: float = 0.0) -> np.ndarray:
    """Nx3 points in Open3D coords (x=right,y=down,z=fwd) → ROS (x=fwd,y=left,z=up)."""
    R = np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)
    pts_ros = (R @ points_o3d.T).T
    pts_ros[:, 2] += float(initial_height_m)
    return pts_ros


# ── Backward-compatible aliases (keep old names importable) ──────────────────
euler_matrix            = rpy_to_transform
quaternion_to_matrix    = quat_to_transform
quaternion_from_matrix  = rot_to_quat
rpy_from_rotation_matrix = rot_to_rpy
matrix_to_pose          = transform_to_pose
get_euler               = quat_msg_to_rpy_deg
yaw_to_quaternion       = yaw_to_quat
rpy_deg_to_R_base       = rpy_deg_to_rot
rot_y                   = rot_y_deg
load_intrinsics_from_yaml = yaml_to_intrinsics
intrinsics_from_fov     = fov_to_intrinsics