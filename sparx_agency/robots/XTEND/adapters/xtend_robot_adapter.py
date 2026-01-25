# robots/XTEND/adapters/xtend_robot_adapter.py
import asyncio
from typing import Any, Optional

from .xtend_ws_client import XtendWsClient, XtendWsConfig
from .xtend_control_adapter import XtendControlAdapter, XtendControlConfig
from .xtend_telemetry_adapter import XtendTelemetryAdapter


class XtendRobotAdapter:
    def __init__(
        self,
        host: str,
        port: int,
        robot_uid: str,
        frequency_hz: float = 30.0,
    ):
        self.telemetry = XtendTelemetryAdapter(robot_uid=robot_uid)

        async def _on_message(msg: dict[str, Any]) -> None:
            self.telemetry.handle_message(msg)

        self.ws = XtendWsClient(
            XtendWsConfig(host=host, port=port, frequency_hz=frequency_hz),
            on_message=_on_message,
        )
        self.control = XtendControlAdapter(
            self.ws,
            XtendControlConfig(robot_uid=robot_uid),
        )

    async def start(self) -> None:
        await self.ws.start()
        # Ensure we publish at least one message quickly (some servers "wake up" on first packet)
        await self.control.hover()

    async def stop(self) -> None:
        await self.ws.stop()
