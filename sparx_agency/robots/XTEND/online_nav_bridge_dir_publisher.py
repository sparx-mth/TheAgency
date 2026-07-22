#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.logging import LoggingSeverity
from std_msgs.msg import String

from sparx_agency.robots.XTEND.xtend_online_bridge_base import OnlineXtendBridgeBase
from sparx_agency.robots.XTEND.xtend_rtsp_image_publisher import LatestFrameGrabber
from sparx_agency.robots.common.image_utils import BadFrameGuard, pad_width_center, center_crop_resize


class OnlineNavBridgeDirPublisher(OnlineXtendBridgeBase):
    """
    Online XTEND bridge that saves preprocessed RTSP frames to a directory
    and publishes each frame's absolute path via std_msgs/String.

    The path is published only after os.rename() completes (atomic on Linux/POSIX),
    so any subscriber that reads the file on receipt of the topic message is
    guaranteed to see a fully written JPEG.

    Write flow per frame:
        imwrite -> frame_XXXXXXXX.tmp  (blocking, completes before we continue)
        rename  -> frame_XXXXXXXX.jpg  (atomic: only visible when complete)
        publish -> String("/abs/path/frame_XXXXXXXX.jpg")
    """

    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        rtsp_uri: str,
        *,
        out_dir: str | Path = "./frames",
        path_topic: str = "/xtend/rgb_frame_path",
        frame_id: str = "xtend_camera",
        backend: str = "gstreamer",
        jpeg_quality: int = 90,
        clear_on_start: bool = True,
        max_frames_kept: int = 30,
        preprocess_mode: str = "resize",
        pad_to_width: int = 728,
        crop_width: int = 540,
        crop_height: int = 420,
        output_width: int = 504,
        output_height: int = 294,
        drop_bad_frames: bool = True,
        bad_frame_mean_min: float = 2.0,
        bad_frame_std_min: float = 1.0,
        bad_frame_sample_step: int = 16,
        bad_frame_log_every: int = 30,
        telemetry_topic: str = "/xtend/local_telemetry",
        bearing_topic: str = "/xtend/bearing",
        telemetry_frame_id: str = "odom",
        telemetry_child_frame_id: str = "xtend_camera",
        log_dir: str | Path = "./xtend_dir_publisher_logs",
    ):
        super().__init__(
            host=host,
            port=port,
            frequency=frequency,
            robot_uid=robot_uid,
            telemetry_topic=telemetry_topic,
            bearing_topic=bearing_topic,
            telemetry_frame_id=telemetry_frame_id,
            telemetry_child_frame_id=telemetry_child_frame_id,
            log_dir=log_dir,
        )

        if preprocess_mode not in ("pad", "crop_resize", "resize"):
            raise ValueError(
                f"preprocess_mode must be 'pad', 'crop_resize', or 'resize', got: {preprocess_mode!r}"
            )

        self.rtsp_uri = rtsp_uri
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames_kept = max(0, int(max_frames_kept))

        if clear_on_start:
            removed = 0
            for f in self.out_dir.glob("frame_*"):
                if f.suffix in (".jpg", ".tmp"):
                    f.unlink(missing_ok=True)
                    removed += 1
            if removed:
                print(f"[dir_pub] cleared {removed} old frame file(s) from {self.out_dir}")

        self.path_topic = path_topic
        self.frame_id = frame_id
        self.jpeg_quality = int(jpeg_quality)
        self.preprocess_mode = preprocess_mode
        self.pad_to_width = int(pad_to_width)
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)

        self.drop_bad_frames = bool(drop_bad_frames)
        self.frame_guard = BadFrameGuard(
            mean_min=bad_frame_mean_min,
            std_min=bad_frame_std_min,
            sample_step=bad_frame_sample_step,
            log_every=bad_frame_log_every,
            prefix="dir_pub",
        )

        self._frame_seq = 0

        self.path_pub = self.ros_node.create_publisher(String, self.path_topic, 10)

        self.grabber = LatestFrameGrabber(rtsp_uri, backend=backend)
        self.grabber.start()

        print(f"[dir_pub] RTSP:          {self.rtsp_uri}")
        print(f"[dir_pub] out_dir:       {self.out_dir}")
        print(f"[dir_pub] path topic:    {self.path_topic}")
        print(f"[dir_pub] preprocess:    {self.preprocess_mode}")
        if self.preprocess_mode == "pad":
            print(f"[dir_pub] pad_to_width:  {self.pad_to_width}")
        elif self.preprocess_mode == "resize":
            print(f"[dir_pub] resize -> {self.output_width}x{self.output_height}")
        else:
            print(f"[dir_pub] crop {self.crop_width}x{self.crop_height} -> {self.output_width}x{self.output_height}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.preprocess_mode == "pad":
            return pad_width_center(frame, self.pad_to_width)
        if self.preprocess_mode == "resize":
            return cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_AREA,
            )
        return center_crop_resize(
            frame,
            crop_width=self.crop_width,
            crop_height=self.crop_height,
            output_width=self.output_width,
            output_height=self.output_height,
        )

    def should_publish_frame(self, frame) -> bool:
        if not self.drop_bad_frames:
            return True
        return self.frame_guard.should_pass(frame)

    async def frame_save_loop(self):
        sleep_time = 1.0 / max(self.frequency, 1e-6)
        print("[dir_pub] frame save loop active")

        while True:
            frame, _stamp = self.grabber.get_latest()

            if not self.should_publish_frame(frame):
                await asyncio.sleep(sleep_time)
                continue

            try:
                frame = self._preprocess(frame)
            except ValueError as exc:
                print(f"[dir_pub][preprocess] {exc}")
                await asyncio.sleep(sleep_time)
                continue

            self._frame_seq += 1
            final_path = self.out_dir / f"frame_{self._frame_seq:08d}.jpg"
            tmp_path = final_path.with_suffix(".tmp")

            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok or buf is None:
                print(f"[dir_pub] imencode failed for seq {self._frame_seq}")
                await asyncio.sleep(sleep_time)
                continue
            tmp_path.write_bytes(buf.tobytes())

            # Atomic rename: subscribers only see the file once it is complete
            tmp_path.rename(final_path)

            # Rolling cleanup: delete oldest frames beyond the keep window.
            # Sort by name (zero-padded seq = lexicographic = chronological).
            if self.max_frames_kept > 0:
                existing = sorted(self.out_dir.glob("frame_*.jpg"))
                for old in existing[: max(0, len(existing) - self.max_frames_kept)]:
                    old.unlink(missing_ok=True)

            # Stamp is taken AFTER the rename so it reflects when the frame became available.
            # Format: "{abs_path} {sec} {nanosec}" — consumers parse with rsplit(" ", 2).
            ros_stamp = self.ros_node.get_clock().now().to_msg()
            msg = String()
            msg.data = f"{final_path} {ros_stamp.sec} {ros_stamp.nanosec}"
            self.path_pub.publish(msg)

            await asyncio.sleep(sleep_time)

    def create_extra_tasks(self):
        return [asyncio.create_task(self.frame_save_loop())]

    def on_shutdown(self):
        self.grabber.stop()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frequency", type=float, default=10.0)
    p.add_argument("--robot-uid", default="drndfb3eeb1")
    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")

    p.add_argument("--out-dir", default="./frames")
    p.add_argument("--path-topic", default="/xtend/rgb_frame_path")
    p.add_argument("--no-clear-on-start", action="store_true", help="Keep existing frames from a previous run")
    p.add_argument("--max-frames-kept", type=int, default=30, help="Rolling window: delete oldest frames beyond this count (0=keep all)")
    p.add_argument("--frame-id", default="xtend_camera")
    p.add_argument("--backend", choices=["ffmpeg", "gstreamer", "default"], default="gstreamer")
    p.add_argument("--jpeg-quality", type=int, default=90)

    p.add_argument(
        "--preprocess-mode",
        choices=["pad", "crop_resize", "resize"],
        default="resize",
    )
    p.add_argument("--pad-to-width", type=int, default=728)
    p.add_argument("--crop-width", type=int, default=540)
    p.add_argument("--crop-height", type=int, default=420)
    p.add_argument("--output-width", type=int, default=504)
    p.add_argument("--output-height", type=int, default=294)

    p.add_argument("--no-drop-bad-frames", action="store_true")
    p.add_argument("--bad-frame-mean-min", type=float, default=2.0)
    p.add_argument("--bad-frame-std-min", type=float, default=1.0)
    p.add_argument("--bad-frame-sample-step", type=int, default=16)
    p.add_argument("--bad-frame-log-every", type=int, default=30)

    p.add_argument("--telemetry-topic", default="/xtend/local_telemetry")
    p.add_argument("--bearing-topic", default="/xtend/bearing")
    p.add_argument("--telemetry-frame-id", default="odom")
    p.add_argument("--telemetry-child-frame-id", default="xtend_camera")
    p.add_argument("--log-dir", default="./xtend_dir_publisher_logs")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging on the bridge's ROS logger")
    return p.parse_args()


async def async_main():
    args = parse_args()
    rclpy.init()

    bridge = OnlineNavBridgeDirPublisher(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        robot_uid=args.robot_uid,
        rtsp_uri=args.rtsp_uri,
        out_dir=args.out_dir,
        path_topic=args.path_topic,
        frame_id=args.frame_id,
        backend=args.backend,
        jpeg_quality=args.jpeg_quality,
        clear_on_start=not args.no_clear_on_start,
        max_frames_kept=args.max_frames_kept,
        preprocess_mode=args.preprocess_mode,
        pad_to_width=args.pad_to_width,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        output_width=args.output_width,
        output_height=args.output_height,
        drop_bad_frames=not args.no_drop_bad_frames,
        bad_frame_mean_min=args.bad_frame_mean_min,
        bad_frame_std_min=args.bad_frame_std_min,
        bad_frame_sample_step=args.bad_frame_sample_step,
        bad_frame_log_every=args.bad_frame_log_every,
        log_dir=args.log_dir,
    )

    if args.debug:
        bridge.ros_node.get_logger().set_level(LoggingSeverity.DEBUG)

    await bridge.run_bridge()


def main():
    try:
        import asyncio
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[main] stopped by user")


if __name__ == "__main__":
    main()