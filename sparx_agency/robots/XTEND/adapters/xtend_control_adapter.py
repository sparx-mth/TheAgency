# robots/XTEND/adapters/xtend_control_adapter.py
import asyncio
from dataclasses import dataclass
from typing import Any

from .xtend_ws_client import XtendWsClient, utc_iso


@dataclass
class XtendControlConfig:
    robot_uid: str
    pilot_station_uid: str = "gcu12345678"
    user_uid: str = "user12345"
    controller_type: int = 1  # mnf


class XtendControlAdapter:
    """
    High-level control -> updates the outbound VIRTUAL_CONTROLLER message.
    Message shape matches your example.
    """

    def __init__(self, ws: XtendWsClient, cfg: XtendControlConfig):
        self.ws = ws
        self.cfg = cfg

        # Matches the example fields.
        self._buttons = [0, 0, 0, 0, 0, 0]
        self._axes = [0, 0, 0, 0, 0]

        self._lock = asyncio.Lock()

    async def _push(self) -> None:
        async with self._lock:
            content: dict[str, Any] = {
                "robot_uid": self.cfg.robot_uid,
                "pilot_station_uid": self.cfg.pilot_station_uid,
                "user_uid": self.cfg.user_uid,
                "type": self.cfg.controller_type,
                "buttons": list(self._buttons),
                "axes": list(self._axes),
            }

        payload = {
            "header": {"timestamp": utc_iso(), "command": "VIRTUAL_CONTROLLER"},
            "content": content,
        }
        await self.ws.set_outbound_payload(payload)

    async def hover(self) -> None:
        async with self._lock:
            self._buttons = [0, 0, 0, 0, 0, 0]
            self._axes = [0, 0, 0, 0, 0]
        await self._push()

    async def arm(self) -> None:
        # Implements same pulse pattern as your script.
        async with self._lock:
            self._buttons[0] = 1
        await self._push()
        await asyncio.sleep(0.1)

        async with self._lock:
            self._buttons[0] = 0
        await self._push()
        await asyncio.sleep(0.1)

        async with self._lock:
            self._buttons[0] = 1
        await self._push()
        await asyncio.sleep(0.3)

        async with self._lock:
            self._buttons[0] = 0
        await self._push()

    async def disarm(self) -> None:
        async with self._lock:
            self._buttons[0] = 1
        await self._push()
        await asyncio.sleep(0.1)

        async with self._lock:
            self._buttons[0] = 0
        await self._push()
        await asyncio.sleep(0.1)

        async with self._lock:
            self._buttons[0] = 1
        await self._push()
        await asyncio.sleep(0.1)

        async with self._lock:
            self._buttons[0] = 0
        await self._push()

    async def takeoff(self, seconds: float = 3.1) -> None:
        async with self._lock:
            self._axes[1] = 1000
        await self._push()
        await asyncio.sleep(seconds)

        async with self._lock:
            self._axes[1] = 0
        await self._push()

    async def land(self, seconds: float = 3.1) -> None:
        async with self._lock:
            self._buttons[3] = 1
        await self._push()
        await asyncio.sleep(seconds)

        async with self._lock:
            self._buttons[3] = 0
        await self._push()

    async def set_xy_yaw_trigger(self, x: int, y: int, trigger: int, yaw: int) -> None:
        """
        Direct axes control:
          axes[0] = lateral
          axes[1] = vertical
          axes[2] = trigger forward/back
          axes[3] = marker horizontal (used as yaw in your script)
          axes[4] = marker vertical
        """
        async with self._lock:
            self._axes[0] = int(x)
            self._axes[1] = int(y)
            self._axes[2] = int(trigger)
            self._axes[3] = int(yaw)
        await self._push()
