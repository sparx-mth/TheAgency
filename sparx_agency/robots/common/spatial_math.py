import numpy as np

def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
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
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
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
