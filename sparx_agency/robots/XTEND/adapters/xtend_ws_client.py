# robots/XTEND/adapters/xtend_ws_client.py
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import websockets


JsonDict = dict[str, Any]
OnMessageCb = Callable[[JsonDict], Awaitable[None]]


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class XtendWsConfig:
    host: str
    port: int
    frequency_hz: float = 30.0
    reconnect_backoff_s: float = 1.0

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"


class XtendWsClient:
    """
    Thin WebSocket transport:
    - Maintains connection with reconnect
    - Runs a send loop at fixed rate (publishes latest outbound payload)
    - Runs a receive loop (dispatches JSON messages to a callback)
    """

    def __init__(self, cfg: XtendWsConfig, on_message: Optional[OnMessageCb] = None):
        self.cfg = cfg
        self.on_message = on_message

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._stop = asyncio.Event()

        self._outbound_lock = asyncio.Lock()
        self._outbound_payload: Optional[JsonDict] = None

        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._stop.clear()
        self._tasks = [asyncio.create_task(self._run())]

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def set_outbound_payload(self, payload: JsonDict) -> None:
        async with self._outbound_lock:
            self._outbound_payload = payload

    async def _run(self) -> None:
        backoff = self.cfg.reconnect_backoff_s
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.cfg.uri) as ws:
                    self._ws = ws
                    send_task = asyncio.create_task(self._send_loop())
                    recv_task = asyncio.create_task(self._recv_loop())

                    done, pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for p in pending:
                        p.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                    # If one task crashed, raise to trigger reconnect
                    for d in done:
                        exc = d.exception()
                        if exc is not None:
                            raise exc

            except asyncio.CancelledError:
                return
            except Exception:
                await asyncio.sleep(backoff)

    async def _send_loop(self) -> None:
        interval = 1.0 / max(self.cfg.frequency_hz, 1e-6)
        while not self._stop.is_set():
            if self._ws is None:
                await asyncio.sleep(interval)
                continue

            async with self._outbound_lock:
                payload = self._outbound_payload

            if payload is not None:
                msg = json.dumps(payload)
                await self._ws.send(msg)

            await asyncio.sleep(interval)

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            if self._stop.is_set():
                return
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if self.on_message is not None:
                await self.on_message(data)
