#!/usr/bin/env python3
"""fake_udp_stream.py

Reads JPEGs from a directory in sorted order, loops them, and sends each frame
as a UDP/RTP-H264 stream — exactly the format rooster_frame_dir_publisher.py
expects from a real Rooster drone.

Use for testing the frames pipeline without a live drone:
  ./run_fake_stream.sh --frames-dir /path/to/jpgs [--fps 5] [--port 5001]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


def build_pipeline(host: str, port: int, width: int, height: int, fps: int) -> tuple:
    pipeline_str = (
        f"appsrc name=src is-live=true block=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast ! "
        "rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port}"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("src")
    if appsrc is None:
        raise RuntimeError("Failed to get appsrc from pipeline")
    return pipeline, appsrc


def main():
    p = argparse.ArgumentParser(
        description="Loop JPEGs from a directory as a UDP/RTP-H264 stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames-dir", required=True, help="Directory containing JPEG frames")
    p.add_argument("--glob", default="*.jpg", help="Glob pattern to select frames")
    p.add_argument("--host", default="127.0.0.1", help="UDP destination host")
    p.add_argument("--port", type=int, default=5001, help="UDP destination port")
    p.add_argument("--fps", type=int, default=5, help="Playback frames per second")
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    frame_files = sorted(frames_dir.glob(args.glob))
    if not frame_files:
        print(f"ERROR: no files matching '{args.glob}' in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    first = cv2.imread(str(frame_files[0]))
    if first is None:
        print(f"ERROR: cannot read {frame_files[0]}", file=sys.stderr)
        sys.exit(1)
    height, width = first.shape[:2]

    print(
        f"fake_udp_stream: {len(frame_files)} frames  {width}x{height}  "
        f"{args.fps} fps  →  {args.host}:{args.port}"
    )

    pipeline, appsrc = build_pipeline(args.host, args.port, width, height, args.fps)
    pipeline.set_state(Gst.State.PLAYING)

    frame_duration_ns = int(1e9 / args.fps)
    pts = 0
    period = 1.0 / args.fps

    try:
        while True:
            for fpath in frame_files:
                frame = cv2.imread(str(fpath))
                if frame is None:
                    print(f"WARN: skipping unreadable frame {fpath.name}", file=sys.stderr)
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))

                buf = Gst.Buffer.new_wrapped(frame.tobytes())
                buf.pts = pts
                buf.duration = frame_duration_ns
                pts += frame_duration_ns

                ret = appsrc.emit("push-buffer", buf)
                if ret != Gst.FlowReturn.OK:
                    print(f"ERROR: push-buffer returned {ret}", file=sys.stderr)
                    return

                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        appsrc.emit("end-of-stream")
        pipeline.set_state(Gst.State.NULL)
        print("fake_udp_stream: stopped")


if __name__ == "__main__":
    main()