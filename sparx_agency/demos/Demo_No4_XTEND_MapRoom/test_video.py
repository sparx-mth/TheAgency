#!/usr/bin/env python3
import asyncio, json
from datetime import datetime, timezone
import websockets

HOST = "192.0.0.15"
PORT = 8000

def utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def api_time():
    return datetime.now(timezone.utc).isoformat()

async def main():
    uri = f"ws://{HOST}:{PORT}"
    async with websockets.connect(uri) as ws:
        print("connected", uri)

        api_version = "3.4.0"

        # Envelope A: header/content (like VIRTUAL_CONTROLLER)
        reqs_a = [
            {"header": {"timestamp": utc_iso(), "command": "GET_PILOT_STATION_VIDEO_STREAM"}, "content": {"data": 0}},
            {"header": {"timestamp": utc_iso(), "command": "GET_PILOT_STATION_VIDEO_STREAM"}, "content": {"data": 1}},
            {"header": {"timestamp": utc_iso(), "command": "GET_ROBOT_VIDEO_STREAMS"}, "content": {}},
        ]

        # Envelope B: command/content + content.header.version/time (like the HTML)
        reqs_b = [
            {"command": "GET_PILOT_STATION_VIDEO_STREAM", "content": {"data": 0, "header": {"version": api_version, "time": api_time()}}},
            {"command": "GET_PILOT_STATION_VIDEO_STREAM", "content": {"data": 1, "header": {"version": api_version, "time": api_time()}}},
            {"command": "GET_ROBOT_VIDEO_STREAMS", "content": {"header": {"version": api_version, "time": api_time()}}},
        ]

        for r in (reqs_a + reqs_b):
            await ws.send(json.dumps(r))

        # Read for a few seconds
        t_end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < t_end:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            cmd_a = msg.get("command")
            cmd_b = msg.get("header", {}).get("command")
            cmd = cmd_a or cmd_b

            # print everything briefly so we see if responses exist
            if cmd in ("GET_PILOT_STATION_VIDEO_STREAM", "GET_ROBOT_VIDEO_STREAMS"):
                print("\n=== RESPONSE", cmd, "===")
                print(json.dumps(msg, indent=2))

            # debug: show what commands we actually get
            # comment out once it works
            # print("[RECV] cmd=", cmd)

asyncio.run(main())