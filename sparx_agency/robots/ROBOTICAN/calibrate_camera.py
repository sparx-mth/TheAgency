#!/usr/bin/env python3
"""Interactive chessboard camera calibration for the ROBOTICAN drone camera.

Connects to the drone's live H264/RTP video stream using the same GStreamer
appsink pattern as examples/src/video_stream.py, lets the operator capture
chessboard views interactively, then runs OpenCV's calibrateCamera and writes
a YAML in the schema used by tasks/localization/config/front_camera_calib.yaml.

Works against either real hardware or the SPHERA simulator - both expose the
same `SetVideoMode` service (see adapters/sphera_ros2_ingestor.py). --width/
--height are only the *requested* resolution passed to that service; the
actual frame size is auto-detected from the negotiated GStreamer caps on
every frame, since the simulator isn't guaranteed to honor the request (e.g.
sphera_ros2_ingestor.py requests 640x360 while the real drone's tested preset
is 540x360) and the calibration output must match whatever was really
captured.

`--pattern-cols`/`--pattern-rows` are the number of squares on the board
(e.g. a 9x6 board), not inner corners; the script converts internally.

Usage:
    python3 calibrate_camera.py --host-ip 192.168.131.24 --drone-id R2

Controls (in the preview window):
    c        capture the current frame (only works once a board is detected)
    q / ESC  finish capturing and run calibration
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

from sparx_agency.robots.ROBOTICAN.helpers.rooster_payload import RoosterPayload
from sparx_agency.robots.common.chessboard_camera_calibration import (
    calibrate_camera,
    find_chessboard_corners,
    make_object_points,
    save_calibration_yaml,
)

Gst.init(None)

DEFAULT_MIN_CAPTURES = 8


class RoosterVideoGrabber(Node):
    """Owns the drone video stream and exposes the latest decoded BGR frame."""

    def __init__(self, drone_id: str, host_ip: str, port: int, requested_width: int, requested_height: int):
        super().__init__("rooster_camera_calibration")
        # Only the resolution requested from SetVideoMode - not necessarily what
        # actually arrives (the simulator doesn't have to honor it). Used solely
        # as a placeholder size before the first real frame is seen.
        self.requested_width = requested_width
        self.requested_height = requested_height

        self.payload = RoosterPayload(
            self,
            drone_id,
            video_host=host_ip,
            video_port=port,
            video_width=requested_width,
            video_height=requested_height,
        )
        self.create_timer(1.0, self.payload.publish_gcs_keep_alive)
        self.create_timer(2.0, self._ensure_video_on)

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None  # actual (width, height), from caps
        self._frame_lock = threading.Lock()

        pipeline_str = (
            f"udpsrc port={port} buffer-size=5242880 do-timestamp=true "
            "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=100 drop-on-latency=true ! "
            "rtph264depay ! queue leaky=downstream max-size-buffers=1 ! decodebin ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self.get_logger().info(f"Starting GStreamer pipeline:\n{pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        appsink = self.pipeline.get_by_name("sink")
        if appsink is None:
            raise RuntimeError("Failed to create GStreamer appsink from pipeline")
        appsink.connect("new-sample", self._on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _ensure_video_on(self):
        if not self.payload.video_on:
            self.payload.set_video(True)

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        info = GstVideo.VideoInfo()
        if not info.from_caps(sample.get_caps()):
            self.get_logger().warn("Failed to read video caps, dropping frame")
            return Gst.FlowReturn.ERROR
        width, height = info.width, info.height

        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            expected = width * height * 3
            if len(map_info.data) < expected:
                return Gst.FlowReturn.OK
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

        with self._frame_lock:
            if self._frame_size is not None and self._frame_size != (width, height):
                self.get_logger().warn(
                    f"Stream resolution changed {self._frame_size} -> {(width, height)}"
                )
            self._frame_size = (width, height)
            self._latest_frame = frame
        return Gst.FlowReturn.OK

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Actual (width, height) seen on the stream, or None before the first frame."""
        with self._frame_lock:
            return self._frame_size

    def destroy_node(self):
        self.payload.set_video(False)
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def _waiting_frame(width: int, height: int, message: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame, message, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
        cv2.LINE_AA,
    )
    return frame


def run_capture_loop(
    node: RoosterVideoGrabber,
    pattern_size: Tuple[int, int],
    square_size_m: float,
    min_captures: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Shows a live preview and lets the operator capture chessboard views.

    Returns the parallel (object_points, image_points) lists ready for
    calibrate_camera().
    """
    objp = make_object_points(pattern_size, square_size_m)
    object_points: List[np.ndarray] = []
    image_points: List[np.ndarray] = []
    window = "ROBOTICAN camera calibration"

    while True:
        frame = node.get_latest_frame()
        if frame is None:
            display = _waiting_frame(
                node.requested_width, node.requested_height, "Waiting for video stream..."
            )
            corners = None
        else:
            display = frame.copy()
            corners = find_chessboard_corners(frame, pattern_size)
            if corners is not None:
                cv2.drawChessboardCorners(display, pattern_size, corners, True)

        status = f"captured {len(object_points)} (need >= {min_captures})  [c]apture  [q]uit"
        cv2.putText(
            display, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("c") and corners is not None:
            object_points.append(objp)
            image_points.append(corners)
            print(f"Captured view {len(object_points)}")

    cv2.destroyAllWindows()
    return object_points, image_points


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--drone-id", default="R2", help="Drone ID (R1/R2/R3...)")
    parser.add_argument(
        "--host-ip", required=True, help="This machine's IP the drone should stream video to"
    )
    parser.add_argument("--port", type=int, default=5001, help="UDP port for the video stream")
    parser.add_argument("--width", type=int, default=540, help="Stream width (drone's 3:2 preset)")
    parser.add_argument("--height", type=int, default=360, help="Stream height (drone's 3:2 preset)")
    parser.add_argument(
        "--pattern-cols", type=int, default=9, help="Chessboard squares across (not inner corners)"
    )
    parser.add_argument(
        "--pattern-rows", type=int, default=6, help="Chessboard squares down (not inner corners)"
    )
    parser.add_argument("--square-size-cm", type=float, default=2.5, help="Chessboard square edge length")
    parser.add_argument("--min-captures", type=int, default=DEFAULT_MIN_CAPTURES)
    parser.add_argument(
        "--out", default=None, help="Output calibration YAML path (default: config/camera_rooster_calib_<w>_<h>.yaml)"
    )
    args = parser.parse_args()

    pattern_size = (args.pattern_cols - 1, args.pattern_rows - 1)
    square_size_m = args.square_size_cm / 100.0

    rclpy.init()
    node = RoosterVideoGrabber(args.drone_id, args.host_ip, args.port, args.width, args.height)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        object_points, image_points = run_capture_loop(
            node, pattern_size, square_size_m, args.min_captures
        )
        frame_size = node.get_frame_size()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if frame_size is None:
        raise RuntimeError(
            "No video frames were ever received - check the stream is reaching "
            f"{args.host_ip}:{args.port} (SetVideoMode request, firewall, ROS_DOMAIN_ID)."
        )
    if len(object_points) < args.min_captures:
        raise RuntimeError(
            f"Only captured {len(object_points)} views, need at least {args.min_captures}. "
            "Re-run and capture more chessboard poses covering the frame edges/corners and tilts."
        )

    width, height = frame_size
    if (width, height) != (args.width, args.height):
        print(
            f"Note: requested {args.width}x{args.height} but the stream actually delivered "
            f"{width}x{height} - calibrating and saving at the actual size."
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).parent / "config" / f"camera_rooster_calib_{width}_{height}.yaml"
    )

    result = calibrate_camera(object_points, image_points, (width, height))
    print(f"RMS reprojection error: {result.rms_reprojection_error:.4f} px")
    print(f"Per-view errors (px): {[round(e, 3) for e in result.per_view_errors]}")

    save_calibration_yaml(out_path, width, height, result.camera_matrix, result.dist_coeffs)
    print(f"Saved calibration to {out_path}")


if __name__ == "__main__":
    main()
