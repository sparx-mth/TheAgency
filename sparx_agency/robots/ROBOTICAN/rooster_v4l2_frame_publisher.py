#!/usr/bin/env python3
"""rooster_v4l2_frame_publisher.py

Drop-in replacement for rooster_frame_dir_publisher.py on the new Orin-NX
architecture: reads the onboard camera via local V4L2 instead of decoding
a wirelessly transmitted UDP/RTP-H264 stream. Same rgb_frame_path topic,
wire format, and frame processing as the original, so DA3/NanoOwl/vLLM
need no changes.

Also publishes (path, frame_id, capture instant) as JSON on a second,
additive --sync-topic, for the IMU/video sync module. Capture instant is
time.monotonic() taken at cap.read(), not the legacy topic's ROS-clock
stamp -- that one reflects disk-write time, not capture time, which is
the wrong clock to align IMU samples against.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from sparx_agency.robots.common.image_utils import BadFrameGuard
from sparx_agency.robots.ROBOTICAN.bar_inpainter import BarInpainter

_DEFAULT_CAGE_MASK = Path(__file__).resolve().parent / "config" / "cage_static_mask.npy"


class V4L2FrameGrabber:
    """Stamps each frame with time.monotonic() + frame_id at cap.read() --
    the only point that stamp is taken, so it's the true capture instant
    no matter how late _on_timer later picks the frame up.
    """

    def __init__(self, device: str, width: int, height: int, fps: float):
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps

        self._lock = threading.Lock()
        self._latest = None  # (frame_bgr, monotonic_stamp, frame_id)
        self._frame_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open V4L2 device {device!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._cap.release()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            stamp = time.monotonic()
            if not ok or frame is None:
                continue
            with self._lock:
                self._frame_id += 1
                self._latest = (frame, stamp, self._frame_id)

    def get_latest(self):
        with self._lock:
            return self._latest


class RoosterV4L2FramePublisher(Node):
    def __init__(self, args):
        super().__init__("rooster_v4l2_frame_publisher")

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
        self.sync_topic = args.sync_topic
        self.jpeg_quality = int(args.jpeg_quality)
        self.max_frames_kept = max(0, int(args.max_frames_kept))

        self.drop_bad_frames = not args.no_drop_bad_frames
        self.frame_guard = BadFrameGuard(
            mean_min=args.bad_frame_mean_min,
            std_min=args.bad_frame_std_min,
            sample_step=args.bad_frame_sample_step,
            log_every=args.bad_frame_log_every,
            prefix="rooster_v4l2_pub",
        )
        self.bar_inpainter = None if args.no_cage_clean else BarInpainter(args.cage_mask_path)

        self._saved_seq = 0
        self._last_published_frame_id = 0
        self.path_pub = self.create_publisher(String, self.path_topic, 10)
        self.sync_pub = self.create_publisher(String, self.sync_topic, 10)

        keep_alive_topic = f"/{args.rooster_id}/gcs_keep_alive"
        self._keep_alive_pub = self.create_publisher(Bool, keep_alive_topic, 10)
        self.create_timer(1.0, self._on_keep_alive)

        self.grabber = V4L2FrameGrabber(
            device=args.device, width=args.width, height=args.height, fps=args.capture_fps)
        self.grabber.start()

        period = 1.0 / max(args.frequency, 1e-6)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"rooster_v4l2_frame_publisher ready\n"
            f"  device:      {args.device}\n"
            f"  out_dir:     {self.out_dir}\n"
            f"  path topic:  {self.path_topic}\n"
            f"  sync topic:  {self.sync_topic}"
        )

    def _on_keep_alive(self):
        msg = Bool()
        msg.data = True
        self._keep_alive_pub.publish(msg)

    def _on_timer(self):
        out = self.grabber.get_latest()
        if out is None:
            return
        frame, capture_monotonic, frame_id = out
        if frame_id == self._last_published_frame_id:
            return  # camera hasn't produced a new frame since our last tick
        self._last_published_frame_id = frame_id

        if self.drop_bad_frames and not self.frame_guard.should_pass(frame):
            return

        if self.bar_inpainter is not None:
            frame = self.bar_inpainter.process(frame)

        self._saved_seq += 1
        final_path = self.out_dir / f"frame_{self._saved_seq:08d}.jpg"
        tmp_path = final_path.with_suffix(".tmp")

        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok or buf is None:
            self.get_logger().warn(f"imencode failed for seq {self._saved_seq}")
            return
        tmp_path.write_bytes(buf.tobytes())
        # Atomic rename: subscribers only ever see a fully written file.
        tmp_path.rename(final_path)

        if self.max_frames_kept > 0:
            existing = sorted(self.out_dir.glob("frame_*.jpg"))
            for old in existing[: max(0, len(existing) - self.max_frames_kept)]:
                old.unlink(missing_ok=True)

        ros_stamp = self.get_clock().now().to_msg()
        legacy_msg = String()
        legacy_msg.data = f"{final_path} {ros_stamp.sec} {ros_stamp.nanosec}"
        self.path_pub.publish(legacy_msg)

        sync_msg = String()
        sync_msg.data = json.dumps({
            "path": str(final_path),
            "frame_id": frame_id,
            "capture_monotonic": capture_monotonic,
        })
        self.sync_pub.publish(sync_msg)

    def destroy_node(self):
        self.grabber.stop()
        super().destroy_node()


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture Robotican's onboard camera via V4L2 (new Orin-NX architecture) "
                    "to JPEGs on disk, publishing each path -- drop-in replacement for "
                    "rooster_frame_dir_publisher.py's GStreamer/UDP version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rooster-id", default="R1", help="Used to build default --path-topic/--sync-topic.")
    p.add_argument("--device", default="/dev/HD_CAMERA",
                   help="V4L2 device path -- see orin_nx_raw_flight_test.py's --camera-device for the same device.")
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--capture-fps", type=float, default=30.0, help="Requested V4L2 capture rate.")
    p.add_argument("--out-dir", default="/tmp/rooster_frames")
    p.add_argument("--path-topic", default="", help="Defaults to /<rooster-id>/rgb_frame_path.")
    p.add_argument("--sync-topic", default="", help="Defaults to /<rooster-id>/rgb_frame_sync.")
    p.add_argument("--frequency", type=float, default=10.0, help="Publish/save rate (decoupled from --capture-fps).")
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
                    help="Skip cage removal entirely (debugging only -- every consumer reads the raw frame).")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.path_topic:
        args.path_topic = f"/{args.rooster_id}/rgb_frame_path"
    if not args.sync_topic:
        args.sync_topic = f"/{args.rooster_id}/rgb_frame_sync"

    rclpy.init()
    node = RoosterV4L2FramePublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
