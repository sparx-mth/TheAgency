# robots/XTEND/adapters/xtend_robot_adapter.py
from typing import Any, Optional

from sparx_agency.core.common.types import Intrinsics, Observation, PoseSE3

from .xtend_ws_client import XtendWsClient, XtendWsConfig
from .xtend_control_adapter import XtendControlAdapter, XtendControlConfig
from .xtend_telemetry_adapter import XtendTelemetryAdapter
from .xtend_video_adapter import XtendVideoAdapter

# Temporary: reuse the working GStreamer RtspProbe from your probe script
from sparx_agency.robots.XTEND.get_xtend_probe import RtspProbe  # :contentReference[oaicite:1]{index=1}


class XtendRobotAdapter:
    def __init__(
        self,
        host: str,
        port: int,
        robot_uid: str,
        frequency_hz: float = 30.0,
        rtsp_uri: str = "rtsp://192.0.0.15:8556/osd_snapshot",
        rtsp_latency_ms: int = 0,
        intrinsics: Optional[Intrinsics] = None,
        frame_id: str = "xtend_camera",
    ):
        self.robot_uid = robot_uid
        self.telemetry = XtendTelemetryAdapter(robot_uid=robot_uid)

        async def _on_message(msg: dict[str, Any]) -> None:
            self.telemetry.handle_message(msg)

        self.ws = XtendWsClient(
            XtendWsConfig(host=host, port=port, frequency_hz=frequency_hz),
            on_message=_on_message,
        )
        self.control = XtendControlAdapter(self.ws, XtendControlConfig(robot_uid=robot_uid))

        self.rtsp = RtspProbe(uri=rtsp_uri, latency_ms=rtsp_latency_ms)
        self.video = XtendVideoAdapter(self.rtsp, intrinsics=intrinsics, frame_id=frame_id)

    async def start(self) -> None:
        await self.ws.start()
        self.rtsp.start()
        # optional "wake up"
        await self.control.hover()

    async def stop(self) -> None:
        self.rtsp.stop()
        await self.ws.stop()

    def getPoseSE3(self) -> Optional[PoseSE3]:
        return self.telemetry.get_pose_se3()

    def getLatestObservation(self) -> Optional[Observation]:
        pose = self.getPoseSE3()
        return self.video.get_latest_observation(pose_map_base=pose)

