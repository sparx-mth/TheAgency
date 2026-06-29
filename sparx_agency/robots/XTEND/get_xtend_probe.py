#!/usr/bin/env python3
import argparse
import asyncio
import json
import threading
import time
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import websockets

import gi

from sparx_agency.robots.XTEND.helpers.state_converter import xtend_extract_robot_block, xtend_robot_block_to_pose_se3, \
    xtend_robot_block_to_state3d

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)



def utc_iso() -> str:
    # Good enough for logging; XTEND examples used ISO-ish timestamps.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

async def send_vc_loop(
    ws,
    robot_uid: str,
    frequency_hz: float,
    pilot_station_uid: str = "gcu12345678",
    user_uid: str = "user12345",
    controller_type: int = 1,
):
    """Send neutral VIRTUAL_CONTROLLER messages at a fixed rate (heartbeat)."""
    interval = 1.0 / max(frequency_hz, 1e-6)

    payload = {
        "header": {"timestamp": utc_iso(), "command": "VIRTUAL_CONTROLLER"},
        "content": {
            "robot_uid": robot_uid,
            "pilot_station_uid": pilot_station_uid,
            "user_uid": user_uid,
            "type": controller_type,
            "buttons": [0, 0, 0, 0, 0, 0],
            "axes": [0, 0, 0, 0, 0],
        },
    }

    while True:
        payload["header"]["timestamp"] = utc_iso()
        await ws.send(json.dumps(payload))
        await asyncio.sleep(interval)



# -----------------------------
# RTSP Video Probe
# -----------------------------
def build_gst_rtsp_pipeline(uri: str, latency_ms: int = 0) -> str:
    # Similar to your gst-launch:
    # rtspsrc ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink
    return (
        f"rtspsrc location={uri} latency={latency_ms} ! "
        f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class RtspProbe:
    """
    RTSP -> BGR frames via GStreamer appsink (no OpenCV backend needed).
    get_latest() returns (frame_bgr, stamp_sec).
    """

    def __init__(self, uri: str, latency_ms: int = 0):
        self.uri = uri
        self.latency_ms = latency_ms

        self._stop_evt = threading.Event()
        self._th: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_stamp: float = 0.0

        self.fps: float = 0.0
        self._fps_counter = 0
        self._fps_last_t = time.time()

        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink = None

    def start(self) -> None:
        # Force TCP to avoid UDP issues on some links.
        pipeline_str = (
            f"rtspsrc location={self.uri} latency={self.latency_ms} protocols=tcp ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
        )

        pipeline = Gst.parse_launch(pipeline_str)
        appsink = pipeline.get_by_name("appsink")
        if appsink is None:
            raise RuntimeError("Failed to create appsink element")

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Failed to set GStreamer pipeline to PLAYING")

        self._pipeline = pipeline
        self._appsink = appsink

        self._stop_evt.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        self._th = None

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)

        self._pipeline = None
        self._appsink = None

    def _loop(self) -> None:
        assert self._appsink is not None
        while not self._stop_evt.is_set():
            sample = self._appsink.emit("pull-sample")
            if sample is None:
                time.sleep(0.01)
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)
            width = int(s.get_value("width"))
            height = int(s.get_value("height"))

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue

            try:
                data = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame_bgr = data.reshape((height, width, 3))
            finally:
                buf.unmap(mapinfo)

            t = time.time()
            with self._lock:
                self._latest_bgr = frame_bgr.copy()
                self._latest_stamp = t

            # FPS estimate
            self._fps_counter += 1
            if (t - self._fps_last_t) >= 1.0:
                self.fps = self._fps_counter / (t - self._fps_last_t)
                self._fps_counter = 0
                self._fps_last_t = t

    def get_latest(self) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy(), float(self._latest_stamp)



# -----------------------------
# Telemetry Probe (WebSocket)
# -----------------------------
def summarize_dict_keys(d: Any, depth: int = 2, prefix: str = "") -> list[str]:
    """
    Returns "key paths" up to a given depth, so you can see structure without dumping huge JSON.
    """
    out = []
    if not isinstance(d, dict) or depth <= 0:
        return out
    for k, v in d.items():
        path = f"{prefix}{k}"
        out.append(path)
        if isinstance(v, dict):
            out.extend(summarize_dict_keys(v, depth - 1, prefix=path + "."))
    return out


def pretty_json(obj: Any, limit_chars: int = 6000) -> str:
    s = json.dumps(obj, indent=2, ensure_ascii=False)
    if len(s) > limit_chars:
        return s[:limit_chars] + "\n... (truncated)"
    return s


class XtendProbe:
    def __init__(
        self,
        ws_uri: str,
        robot_uid: str,
        mode: str = "both",
        frequency_hz: float = 10.0,
        raw_dump_seconds: float = 0.0,
        print_robot_status_only: bool = False,
    ):
        self._printed_schema = False
        self._printed_hf_once = False
        self.ws_uri = ws_uri
        self.robot_uid = robot_uid
        self.mode = mode
        self.frequency_hz = frequency_hz
        self.raw_dump_seconds = raw_dump_seconds
        self.print_robot_status_only = print_robot_status_only

        self.msg_type_counts = Counter()
        self.last_robot_status: Optional[Dict[str, Any]] = None
        self.last_robot_status_t: float = 0.0

        self._start_t = time.time()

    async def run(self) -> None:
        print(f"[WS] Connecting to {self.ws_uri}")
        async with websockets.connect(self.ws_uri) as ws:
            send_task = None
            if self.mode in ("send", "both"):
                send_task = asyncio.create_task(send_vc_loop(ws, self.robot_uid, self.frequency_hz))
            print("[WS] Connected.")

            # We are NOT sending controller commands in this probe.
            # If the server requires periodic messages to keep telemetry flowing,
            # we can add a keepalive later.
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    header = msg.get("header", {})
                    cmd = header.get("command", "UNKNOWN")
                    self.msg_type_counts[cmd] += 1

                    now = time.time()
                    should_raw_dump = (self.raw_dump_seconds > 0.0) and ((now - self._start_t) <= self.raw_dump_seconds)

                    if should_raw_dump and (not self.print_robot_status_only or cmd == "ROBOT_STATUS"):
                        print("\n" + "=" * 80)
                        print(f"[RAW] {utc_iso()} command={cmd}")
                        print(pretty_json(msg))
                        print("=" * 80)

                    # Special handling: ROBOT_STATUS -> extract the robot block and summarize keys
                    if cmd == "ROBOT_STATUS":
                        robot_block = xtend_extract_robot_block(msg, self.robot_uid)
                        if not self._printed_schema:
                            self._printed_schema = True
                            print("[XTEND] Robot block keys:", sorted(robot_block.keys()))
                            print("[XTEND] Key paths depth=3:")
                            for kp in summarize_dict_keys(robot_block, depth=3):
                                print(" -", kp)

                        if robot_block is not None:
                            pose_se3 = xtend_robot_block_to_pose_se3(robot_block)
                            state3d = xtend_robot_block_to_state3d(robot_block)
                            self.last_robot_status = robot_block
                            self.last_robot_status_t = now

                    await asyncio.sleep(0)  # yield

                    if cmd == "HIGH_FREQUENCY_ROBOT_TELEMETRY":
                        # print one example structure once
                        if not self._printed_hf_once:
                            print(pretty_json(msg))
                            self._printed_hf_once = True
            finally:
                if send_task is not None:
                    send_task.cancel()
                    await asyncio.gather(send_task, return_exceptions=True)


    def _extract_robot_block(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = msg.get("content", {})
        robots = content.get("robots", [])
        if not isinstance(robots, list):
            return None
        for r in robots:
            if isinstance(r, dict) and r.get("robot_uid") == self.robot_uid:
                return r
        return None

    def print_summary(self) -> None:
        # Prints a compact snapshot of what we learned so far
        print("\n" + "-" * 80)
        print("[SUMMARY] Message type counts:")
        for k, v in self.msg_type_counts.most_common():
            print(f"  {k}: {v}")

        if self.last_robot_status is None:
            # print(f"\n[SUMMARY] No ROBOT_STATUS for robot_uid={self.robot_uid} yet.")
            # print("-" * 80)
            return

        r = self.last_robot_status

        # 1. Extract Local Telemetry (x, y, z)
        local_telemetry = r.get("local_telemetry", {})
        x = local_telemetry.get("x", 0.0)
        y = local_telemetry.get("y", 0.0)
        z = local_telemetry.get("z", 0.0)

        # 2. Extract Bearing/Yaw
        telemetry = r.get("telemetry", {})
        details = telemetry.get("details", {})
        bearing = details.get("bearing", 0.0)

        # 3. Print the combined "Pose" data
        print(f"\n[SUMMARY] Latest Pose for {self.robot_uid}:")
        print(f"  Local Position -> X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")
        print(f"  Bearing (rad)  -> {bearing:.4f}")

        print("-" * 80)




# -----------------------------
# Main
# -----------------------------
async def main_async(args: argparse.Namespace) -> None:
    ws_uri = f"ws://{args.host}:{args.port}"
    probe = XtendProbe(
        ws_uri=ws_uri,
        robot_uid=args.robot_uid,
        mode=args.mode,
        frequency_hz=args.frequency_hz,

        raw_dump_seconds=args.raw_dump_seconds,
        print_robot_status_only=args.robot_status_only,
    )

    rtsp = RtspProbe(uri=args.rtsp_uri, latency_ms=args.rtsp_latency_ms)
    rtsp.start()

    stop_evt = asyncio.Event()

    async def ws_task():
        try:
            await probe.run()
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[WS] Error: {e}")
            # stop_evt.set()

    async def ui_task():
        last_summary_t = time.time()
        while not stop_evt.is_set():
            # Print once per second: video status + telemetry summary hint
            now = time.time()
            if now - last_summary_t >= 1.0:
                last_summary_t = now
                latest = rtsp.get_latest()
                if latest is None:
                    print(f"[VIDEO] no frames yet | fps={rtsp.fps:.1f}")
                else:
                    frame, stamp = latest
                    h, w = frame.shape[:2]
                    age_ms = (time.time() - stamp) * 1000.0
                    print(f"[VIDEO] {w}x{h} | fps={rtsp.fps:.1f} | frame_age_ms={age_ms:.0f}")

                # Also print compact telemetry summary once per second
                probe.print_summary()

            if args.show_video:
                out = rtsp.get_latest()
                if out is not None:
                    frame, _ = out
                    cv2.imshow("XTEND RTSP", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop_evt.set()
                        break

            await asyncio.sleep(0.02)

    t1 = asyncio.create_task(ws_task())
    t2 = asyncio.create_task(ui_task())

    try:
        await stop_evt.wait()
    finally:
        t1.cancel()
        t2.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)
        rtsp.stop()
        if args.show_video:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XTEND probe: RTSP video + WS telemetry dump")
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--robot-uid", default="drndfb3eeb1")  # drone 42B drnb177ede2, drone 36B drndfb3eeb1
    p.add_argument("--frequency-hz", type=float, default=10.0)
    p.add_argument("--mode", choices=["send", "listen", "both"], default="both")

    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")
    p.add_argument("--rtsp-latency-ms", type=int, default=0)

    p.add_argument("--show-video", action="store_true", help="Open a cv2 window (press q to quit)")
    p.add_argument("--raw-dump-seconds", type=float, default=5.0, help="Print full raw JSON for first N seconds (0 disables)")
    p.add_argument("--robot-status-only", action="store_true", help="During raw dump, print only ROBOT_STATUS messages")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
