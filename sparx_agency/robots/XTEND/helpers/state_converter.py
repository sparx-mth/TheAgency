import numpy as np

from sparx_agency.core.common.types import Pose3D, Twist3D, State3D, PoseSE3
from sparx_agency.robots.common.spatial_math import euler_to_rot_zyx


def xtend_robot_block_to_pose_se3(robot_block: dict) -> PoseSE3:
    """
    Convert XTEND ROBOT_STATUS 'robot' dict into PoseSE3.

    Expected fields (from your probe):
      robot_block["local_telemetry"] -> x,y,z (meters)
      robot_block["telemetry"]["details"]["bearing"] -> yaw (rad)
    """
    x, y, z, yaw = xtend_robot_block(robot_block)
    # roll/pitch not provided in the messages you showed, assume 0
    R_matrix = euler_to_rot_zyx(roll=0.0, pitch=0.0, yaw=yaw)

    t = np.array([x, y, z], dtype=np.float32)
    return PoseSE3(R=R_matrix, t=t)


def xtend_robot_block_to_state3d(robot_block: dict) -> State3D:
    """
    Convert XTEND ROBOT_STATUS 'robot' dict into your generic State3D.
    Uses pose + yaw. Twist is set to zeros until you parse HIGH_FREQUENCY_ROBOT_TELEMETRY.
    """

    x, y, z, yaw = xtend_robot_block(robot_block)
    pose = Pose3D(x=x, y=y, z=z, yaw=yaw)
    twist = Twist3D(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
    return State3D(pose=pose, twist=twist)


def xtend_robot_block(robot_block: dict):
    lt = robot_block.get("local_telemetry", {}) or {}
    tel = robot_block.get("telemetry", {}) or {}
    details = tel.get("details", {}) or {}

    x = float(lt.get("x", 0.0))
    y = float(lt.get("y", 0.0))
    z = float(lt.get("z", 0.0))
    yaw = float(details.get("bearing", 0.0))

    return x, y, z, yaw

def xtend_extract_robot_block(msg: dict, robot_uid: str) -> dict | None:
    """
    From a WS message with command=ROBOT_STATUS, return the robot dict for robot_uid.
    """
    content = msg.get("content", {}) or {}
    robots = content.get("robots", [])
    if not isinstance(robots, list):
        return None
    for r in robots:
        if isinstance(r, dict) and r.get("robot_uid") == robot_uid:
            return r
    return None
