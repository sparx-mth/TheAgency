#!/usr/bin/env python3
"""Standalone camera-capture acceptance check for Robotican's new Orin-NX
architecture.

Until now the only Rooster video path in this repo was network RTP/H264
into a GStreamer appsink (see rooster_video_adapter.py's VideoStreamManager),
because the camera lived on a separate host from the compute. An early block
diagram from Robotican showed the camera wired directly by USB into the same
Jetson Orin NX that runs the backend and the models -- but that diagram is
NOT confirmed final, so this script does not assume either transport is
"the" answer. Run --mode v4l2 first; if there's no USB camera device, or the
stream still turns out to come through Rooster's own video relay, --mode gst
covers that instead. Whichever one actually works IS the answer -- that's
the point of testing it rather than building against the diagram.

This script does NOT depend on ROS2, rclpy, GStreamer, cv_bridge or any of
VideoStreamManager's other heavy dependencies (several of which -- the
AprilTag task in particular -- hardcode onboard-only paths like
/home/rooster/sparx_agency/...). It is meant to run standalone, before any
ROS2 graph is even up, as the very first "is the camera even there" check
on a freshly-delivered unit.

Two capture modes:
  --mode v4l2 (default): open the USB camera directly via OpenCV/V4L2.
  --mode gst: the RTP/H264-over-UDP GStreamer pipeline (same caps/depay/
      decode chain as VideoStreamManager), for the case the stream still
      arrives via Rooster's own video relay rather than a local USB device.

Usage:
    python3 orin_nx_camera_capture_test.py --device /dev/video0 \\
        --width 1280 --height 720 --fps 30 --duration-sec 10 \\
        --out-dir /tmp/orin_nx_camera_test
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def open_v4l2(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_gst_rtp(host_ip: str, port: int, width: int, height: int) -> cv2.VideoCapture:
    """Same pipeline shape as VideoStreamManager's GStreamer path, minus the
    ROS2/appsink-callback plumbing -- OpenCV's own GStreamer backend pulls
    frames directly."""
    pipeline = (
        f"udpsrc address={host_ip} port={port} buffer-size=5242880 do-timestamp=true "
        "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
        "rtpjitterbuffer latency=100 drop-on-latency=true ! "
        "rtph264depay ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "decodebin ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def run_capture_check(
    cap: cv2.VideoCapture,
    duration_sec: float,
    out_dir: Path,
    expect_width: int,
    expect_height: int,
) -> dict:
    if not cap.isOpened():
        return {"opened": False}

    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    fail_count = 0
    resolutions_seen = set()
    frame_timestamps = []
    first_frame = None
    last_frame = None

    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        now = time.monotonic()
        if not ok or frame is None:
            fail_count += 1
            continue
        frame_count += 1
        frame_timestamps.append(now)
        resolutions_seen.add((frame.shape[1], frame.shape[0]))
        if first_frame is None:
            first_frame = frame.copy()
        last_frame = frame

    if first_frame is not None:
        cv2.imwrite(str(out_dir / "first_frame.jpg"), first_frame)
    if last_frame is not None:
        cv2.imwrite(str(out_dir / "last_frame.jpg"), last_frame)

    inter_frame_gaps = [
        b - a for a, b in zip(frame_timestamps, frame_timestamps[1:])
    ]
    measured_fps = frame_count / duration_sec if duration_sec > 0 else 0.0
    max_gap = max(inter_frame_gaps) if inter_frame_gaps else float("nan")
    mean_gap = statistics.mean(inter_frame_gaps) if inter_frame_gaps else float("nan")

    resolution_ok = (
        len(resolutions_seen) == 1
        and next(iter(resolutions_seen)) == (expect_width, expect_height)
    )

    return {
        "opened": True,
        "frame_count": frame_count,
        "fail_count": fail_count,
        "measured_fps": measured_fps,
        "resolutions_seen": resolutions_seen,
        "resolution_ok": resolution_ok,
        "mean_gap_sec": mean_gap,
        "max_gap_sec": max_gap,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orin-NX camera acceptance check: open the camera, capture "
        "for N seconds, report measured FPS/resolution/dropped-frame stats."
    )
    p.add_argument("--mode", choices=["v4l2", "gst"], default="v4l2",
                    help="v4l2 = direct USB camera (expected on the new architecture). "
                         "gst = legacy RTP/H264-over-UDP fallback.")
    p.add_argument("--device", default="/dev/video0",
                    help="V4L2 device path (--mode v4l2 only).")
    p.add_argument("--host-ip", default="127.0.0.1",
                    help="RTP source address to listen on (--mode gst only).")
    p.add_argument("--port", type=int, default=5001,
                    help="RTP UDP port (--mode gst only, matches Rooster's video_handler port).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30,
                    help="Requested capture FPS (--mode v4l2 only; gst is source-driven).")
    p.add_argument("--duration-sec", type=float, default=10.0)
    p.add_argument("--min-fps", type=float, default=None,
                    help="Pass threshold for measured FPS. Defaults to 80%% of --fps.")
    p.add_argument("--out-dir", default="/tmp/orin_nx_camera_test",
                    help="Where to save first/last captured frame for visual inspection.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    min_fps = args.min_fps if args.min_fps is not None else 0.8 * args.fps

    print(f"[camera-test] mode        : {args.mode}")
    if args.mode == "v4l2":
        print(f"[camera-test] device      : {args.device}")
        cap = open_v4l2(args.device, args.width, args.height, args.fps)
    else:
        print(f"[camera-test] source      : udp {args.host_ip}:{args.port}")
        cap = open_gst_rtp(args.host_ip, args.port, args.width, args.height)

    print(f"[camera-test] duration_sec : {args.duration_sec}")
    print(f"[camera-test] min_fps      : {min_fps:.1f}")

    result = run_capture_check(
        cap, args.duration_sec, Path(args.out_dir), args.width, args.height
    )
    cap.release()

    if not result["opened"]:
        print("[camera-test] FAIL: could not open capture source at all.")
        return 1

    print(
        f"[camera-test] frames captured : {result['frame_count']} "
        f"(read failures: {result['fail_count']})"
    )
    print(f"[camera-test] measured FPS    : {result['measured_fps']:.2f}")
    print(f"[camera-test] resolutions seen: {sorted(result['resolutions_seen'])}")
    print(
        f"[camera-test] inter-frame gap : mean={result['mean_gap_sec'] * 1000:.1f}ms "
        f"max={result['max_gap_sec'] * 1000:.1f}ms"
    )
    print(f"[camera-test] sample frames saved to: {args.out_dir}")

    passed = (
        result["frame_count"] > 0
        and result["measured_fps"] >= min_fps
        and result["resolution_ok"]
    )

    if not result["resolution_ok"]:
        print(
            f"[camera-test] WARNING: resolution mismatch or unstable resolution "
            f"(expected {args.width}x{args.height}, saw {sorted(result['resolutions_seen'])})"
        )

    print(f"[camera-test] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
