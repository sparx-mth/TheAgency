# robots/XTEND/adapters/xtend_telemetry_adapter.py
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class XtendTelemetry:
    yaw_rad: float
    stamp_utc: Optional[str] = None


class XtendTelemetryAdapter:
    def __init__(self, robot_uid: str):
        self.robot_uid = robot_uid
        self.last: Optional[XtendTelemetry] = None

    def handle_message(self, msg: dict[str, Any]) -> None:
        header = msg.get("header", {})
        content = msg.get("content", {})
        cmd = header.get("command")

        if cmd != "ROBOT_STATUS":
            return

        robots = content.get("robots", [])
        for r in robots:
            if r.get("robot_uid") != self.robot_uid:
                continue
            bearing = (
                r.get("telemetry", {})
                 .get("details", {})
                 .get("bearing")
            )
            if bearing is None:
                return
            self.last = XtendTelemetry(yaw_rad=float(bearing), stamp_utc=header.get("timestamp"))
            return
