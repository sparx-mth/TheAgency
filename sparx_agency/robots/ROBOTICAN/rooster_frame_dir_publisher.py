#!/usr/bin/env python3
"""rooster_frame_dir_publisher.py

Decodes the ROBOTICAN/Rooster drone's UDP/RTP-H264 video stream, saves each
frame as a JPEG under --out-dir, and publishes each file's absolute path on
--path-topic (std_msgs/String, "{path} {sec} {nanosec}") for downstream nodes
(depth_processor_node.py, localization_node.py) to consume — the same
file-path convention used for XTEND (online_nav_bridge_dir_publisher.py).

Frames are always written to disk AND their path published — never
publish-only. This is required for both the 360-degree sweep capture and any
future FALCON bridging, where RGB/depth cross the ROS1/ROS2 bridge as
file-path strings rather than raw sensor_msgs/Image.

Cage removal (BarInpainter) runs here, once, before the frame is saved: every
downstream consumer reads this same saved JPEG (depth_processor_node.py via
frame_path, the YOLO detector via rgb_topic), so cleaning at the source means
neither has to duplicate the logic or risk drifting out of sync with it — see
bar_inpainter.py for the two cage artifacts it handles and why.

Single responsibility: video decode -> cage removal -> frame file -> path
topic. It does not touch FCU commands, SetVideoMode, AprilTag, or TF — those
stay owned by RoosterCommandUnitNode / localization_node.py respectively.

Frames are published at the drone's native resolution (540x360) with no
crop/resize — the DA3 model and its camera calibration are sized to match
ROBOTICAN's own frames directly, so no aspect-ratio workaround is needed.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sparx_agency.robots.common.image_utils import BadFrameGuard
from sparx_agency.robots.ROBOTICAN.bar_inpainter import BarInpainter

Gst.init(None)

_DEFAULT_CAGE_MASK = Path(__file__).resolve().parent / "config" / "cage_static_mask.npy"


class UdpH264FrameGrabber:
    """Decodes a UDP/RTP-H264 stream to BGR frames via a GStreamer appsink.

    Pipeline mirrors the one already proven working in
    robots/ROBOTICAN/adapters/rooster_video_adapter.py's VideoStreamManager,
    stripped of the AprilTag/TF/FCU logic that class bundles alongside it.
    """

    def __init__(self, port: int):
        self._lock = threading.Lock()
        self._latest = None  # (frame_bgr, monotonic_stamp)

        pipeline_str = (
            f"udpsrc port={port} buffer-size=5242880 do-timestamp=true "
            "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=100 drop-on-latency=true ! "
            "rtph264depay ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "decodebin ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        appsink = self._pipeline.get_by_name("sink")
        if appsink is None:
            raise RuntimeError("Failed to get appsink from GStreamer pipeline")
        appsink.connect("new-sample", self._on_new_sample)

    def start(self):
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set GStreamer pipeline to PLAYING")

    def stop(self):
        self._pipeline.set_state(Gst.State.NULL)

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        if not width or not height:
            return Gst.FlowReturn.ERROR

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            expected = width * height * 3
            if len(map_info.data) < expected:
                return Gst.FlowReturn.ERROR
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

        with self._lock:
            self._latest = (frame, time.time())
        return Gst.FlowReturn.OK

    def get_latest(self):
        with self._lock:
            return self._latest


class RoosterFrameDirPublisher(Node):
    def __init__(self, args):
        super().__init__("rooster_frame_dir_publisher")

        self.out_dir = Path(args.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_clear_on_start:
            removed = 0
            for f in self.out_dir.glob("frame_*"):
                if f.suffix in (".jpg", ".tmp"):
                    f.unlink(missing_ok=True)
                    removed += 1
            if removed:
                self.get_logger().info(f"cleared {removed} old frame file(s) from {self.out_dir}")

        self.path_topic = args.path_topic
        self.jpeg_quality = int(args.jpeg_quality)
        self.max_frames_kept = max(0, int(args.max_frames_kept))

        self.drop_bad_frames = not args.no_drop_bad_frames
        self.frame_guard = BadFrameGuard(
            mean_min=args.bad_frame_mean_min,
            std_min=args.bad_frame_std_min,
            sample_step=args.bad_frame_sample_step,
            log_every=args.bad_frame_log_every,
            prefix="rooster_dir_pub",
        )
        self.bar_inpainter = None if args.no_cage_clean else BarInpainter(args.cage_mask_path)

        self._frame_seq = 0
        self.path_pub = self.create_publisher(String, self.path_topic, 10)

        keep_alive_topic = f"/{args.rooster_id}/gcs_keep_alive"
        self._keep_alive_pub = self.create_publisher(Bool, keep_alive_topic, 10)
        self.create_timer(1.0, self._on_keep_alive)

        self.grabber = UdpH264FrameGrabber(port=args.port)
        self.grabber.start()

        period = 1.0 / max(args.frequency, 1e-6)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"rooster_frame_dir_publisher ready\n"
            f"  udp port:    {args.port}\n"
            f"  out_dir:     {self.out_dir}\n"
            f"  path topic:  {self.path_topic}"
        )

    def _on_keep_alive(self):
        msg = Bool()
        msg.data = True
        self._keep_alive_pub.publish(msg)

    def _on_timer(self):
        out = self.grabber.get_latest()
        if out is None:
            return
        frame, _stamp = out

        if self.drop_bad_frames and not self.frame_guard.should_pass(frame):
            return

        if self.bar_inpainter is not None:
            frame = self.bar_inpainter.process(frame)

        self._frame_seq += 1
        final_path = self.out_dir / f"frame_{self._frame_seq:08d}.jpg"
        tmp_path = final_path.with_suffix(".tmp")

        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok or buf is None:
            self.get_logger().warn(f"imencode failed for seq {self._frame_seq}")
            return
        tmp_path.write_bytes(buf.tobytes())
        # Atomic rename: subscribers only ever see a fully written file.
        tmp_path.rename(final_path)

        if self.max_frames_kept > 0:
            existing = sorted(self.out_dir.glob("frame_*.jpg"))
            for old in existing[: max(0, len(existing) - self.max_frames_kept)]:
                old.unlink(missing_ok=True)

        # Stamp is taken after the rename so it reflects when the frame
        # became available. Consumers parse with rsplit(" ", 2).
        ros_stamp = self.get_clock().now().to_msg()
        msg = String()
        msg.data = f"{final_path} {ros_stamp.sec} {ros_stamp.nanosec}"
        self.path_pub.publish(msg)

    def destroy_node(self):
        self.grabber.stop()
        super().destroy_node()


def parse_args():
    p = argparse.ArgumentParser(
        description="Decode ROBOTICAN's UDP/RTP-H264 stream to JPEGs on disk, publishing each path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rooster-id", default="R1", help="Used to build default --path-topic.")
    p.add_argument("--port", type=int, default=5001,
                   help="UDP port to receive H264/RTP on — must match RoosterCommandUnitNode's video_port.")
    p.add_argument("--out-dir", default="/tmp/rooster_frames")
    p.add_argument("--path-topic", default="", help="Defaults to /<rooster-id>/rgb_frame_path.")
    p.add_argument("--frequency", type=float, default=10.0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--no-clear-on-start", action="store_true", help="Keep existing frames from a previous run")
    p.add_argument("--max-frames-kept", type=int, default=30,
                   help="Rolling window: delete oldest frames beyond this count (0=keep all)")

    p.add_argument("--no-drop-bad-frames", action="store_true")
    p.add_argument("--bad-frame-mean-min", type=float, default=2.0)
    p.add_argument("--bad-frame-std-min", type=float, default=1.0)
    p.add_argument("--bad-frame-sample-step", type=int, default=16)
    p.add_argument("--bad-frame-log-every", type=int, default=30)

    p.add_argument("--cage-mask-path", default=str(_DEFAULT_CAGE_MASK),
                    help="Static cage-arc mask (BarInpainter); the moving crossbar is detected fresh per frame regardless.")
    p.add_argument("--no-cage-clean", action="store_true",
                    help="Skip cage removal entirely (debugging only — every consumer reads the raw frame).")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.path_topic:
        args.path_topic = f"/{args.rooster_id}/rgb_frame_path"

    rclpy.init()
    node = RoosterFrameDirPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
