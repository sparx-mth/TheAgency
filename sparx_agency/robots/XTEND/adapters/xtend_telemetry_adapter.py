# robots/XTEND/adapters/xtend_telemetry_adapter.py
from dataclasses import dataclass
from typing import Any, Optional

from sparx_agency.core.common.types import PoseSE3
from sparx_agency.robots.common.state_converter import (
    xtend_extract_robot_block,
    xtend_robot_block_to_pose_se3,
)

@dataclass
class XtendTelemetry:
    pose: PoseSE3
    yaw_rad: float
    stamp_utc: Optional[str] = None


class XtendTelemetryAdapter:
    def __init__(self, robot_uid: str):
        self.robot_uid = robot_uid
        self.last: Optional[XtendTelemetry] = None

    def handle_message(self, msg: dict[str, Any]) -> None:
        header = msg.get("header", {}) or {}
        cmd = header.get("command")
        if cmd != "ROBOT_STATUS":
            return

        robot = xtend_extract_robot_block(msg, self.robot_uid)
        if robot is None:
            return

        # pose
        pose = xtend_robot_block_to_pose_se3(robot)

        # yaw (keep explicit because useful/debuggable)
        yaw = (
            (robot.get("telemetry", {}) or {})
            .get("details", {}) or {}
        ).get("bearing")
        if yaw is None:
            yaw = 0.0

        self.last = XtendTelemetry(
            pose=pose,
            yaw_rad=float(yaw),
            stamp_utc=header.get("timestamp"),
        )

    def get_pose_se3(self) -> Optional[PoseSE3]:
        return None if self.last is None else self.last.pose
