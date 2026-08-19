#!/usr/bin/env python3
"""Record a run: drone camera + N1's top-down route + System-1/2 FPS, to MP4.

Subscribes only to topics -- the camera, the odometry, the committed and full N1
routes, and the `/simple_drone/n1/info` status the policy node publishes (action,
System-1/System-2 FPS, the S2 pixel goal). Every pixel is drawn by the ROS-free
:mod:`~sparx_agency.tasks.planning.sjtu_internvla_n1.recording` helpers, so this
node is thin: pull the latest of each, compose, write a frame at a fixed rate.

Writes with OpenCV's own `mp4v` encoder (no system ffmpeg needed). Pair it with
`ros2 bag record` in `scripts/record_run.sh` for a lossless copy of every topic.

CPU-only, like every node in this stack.
"""
from __future__ import annotations

import json
import os
import threading
from math import atan2

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

# cv_bridge is convenient but absent from the SJTU sim's Humble image, where this
# recorder runs so the camera is native (host<->container DDS drops large
# messages). Decode manually when it is missing; the front camera is rgb8.
try:
    from cv_bridge import CvBridge
    _CV_BRIDGE = CvBridge()
except Exception:  # noqa: BLE001
    _CV_BRIDGE = None


def _imgmsg_to_bgr(msg):
    """sensor_msgs/Image -> HxWx3 BGR, without requiring cv_bridge."""
    if _CV_BRIDGE is not None:
        try:
            return _CV_BRIDGE.imgmsg_to_cv2(msg, "bgr8")
        except Exception:  # noqa: BLE001
            pass
    enc = (msg.encoding or "rgb8").lower()
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "rgb8" else img)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(buf.reshape(msg.height, msg.width, -1)[:, :, :3])

from sparx_agency.tasks.planning.sjtu_internvla_n1.recording import (
    OverlayInfo,
    TopDownRenderer,
    compose,
    draw_camera_panel,
)


def _yaw_from_quat(q):
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _path_xy(msg):
    return np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses],
                    dtype=float) if msg.poses else None


class N1RunRecorderNode(Node):
    """Compose the drone camera and N1's route into a recorded MP4."""

    def __init__(self):
        super().__init__("n1_run_recorder_node")
        self.declare_parameter("config_file", "")
        self.declare_parameter("output", "")
        cfg = _load_config(self.get_parameter("config_file").value)
        topics = cfg.get("topics", {})
        rec = cfg.get("recorder", {})

        rgb_topic = topics.get("rgb", "/simple_drone/front/image_raw")
        self._rgb_compressed = topics.get("rgb_type", "raw") == "compressed"
        odom_topic = topics.get("odom", "/simple_drone/odom")
        traj_topic = topics.get("trajectory", "/simple_drone/n1/trajectory")
        full_topic = topics.get("trajectory_full", "/simple_drone/n1/trajectory_full")
        info_topic = topics.get("info", "/simple_drone/n1/info")

        self._panel_w = int(rec.get("panel_width", 640))
        self._panel_h = int(rec.get("panel_height", 480))
        self._record_fps = float(rec.get("fps", 10.0))
        output = self.get_parameter("output").value or rec.get(
            "output", "/tmp/sjtu_n1/run.mp4")
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        self._output = output

        self._bridge = _CV_BRIDGE
        self._lock = threading.Lock()
        self._frame = None
        self._pose = None
        self._committed = None
        self._full = None
        self._info = OverlayInfo()
        self._topdown = TopDownRenderer(size=(self._panel_w, self._panel_h))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output, fourcc, self._record_fps,
                                       (self._panel_w * 2, self._panel_h), True)
        if not self._writer.isOpened():
            raise RuntimeError("could not open VideoWriter at %s" % (output,))
        self._frames_written = 0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        rgb_type = CompressedImage if self._rgb_compressed else Image
        self.create_subscription(rgb_type, rgb_topic, self._on_rgb, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(Path, traj_topic, self._on_committed, latched)
        self.create_subscription(Path, full_topic, self._on_full, latched)
        self.create_subscription(String, info_topic, self._on_info, latched)

        self.create_timer(1.0 / max(1e-3, self._record_fps), self._record)
        self.get_logger().info(
            "recording drone camera + N1 route to %s (%dx%d @ %.0f fps)"
            % (output, self._panel_w * 2, self._panel_h, self._record_fps))

    # ── subscriptions ────────────────────────────────────────────────
    def _on_rgb(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                frame = _imgmsg_to_bgr(msg)
            with self._lock:
                self._frame = frame
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("rgb decode failed: %s" % (exc,))

    def _on_odom(self, msg):
        p = msg.pose.pose
        pose = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))
        with self._lock:
            self._pose = pose
        self._topdown.add_pose(pose[0], pose[1])

    def _on_committed(self, msg):
        with self._lock:
            self._committed = _path_xy(msg)

    def _on_full(self, msg):
        with self._lock:
            self._full = _path_xy(msg)

    def _on_info(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        pg = d.get("pixel_goal")
        pgf = d.get("pixel_goal_frame")
        with self._lock:
            self._info = OverlayInfo(
                instruction=d.get("instruction", ""),
                action=d.get("action") or "",
                status="STOP" if d.get("stop") else "navigating",
                s1_fps=d.get("s1_fps"), s2_fps=d.get("s2_fps"),
                s1_ms=d.get("s1_ms"), s2_ms=d.get("s2_ms"),
                pixel_goal=(int(pg[0]), int(pg[1])) if pg else None,
                pixel_goal_frame=(int(pgf[0]), int(pgf[1])) if pgf else None)

    # ── the record loop ──────────────────────────────────────────────
    def _record(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            pose = self._pose
            committed = None if self._committed is None else self._committed.copy()
            full = None if self._full is None else self._full.copy()
            info = self._info
        if frame is None:
            return
        left = draw_camera_panel(frame, info, (self._panel_w, self._panel_h))
        right = self._topdown.render(pose, committed, full)
        self._writer.write(compose(left, right))
        self._frames_written += 1

    def destroy_node(self):
        try:
            if getattr(self, "_writer", None):
                self._writer.release()
                self.get_logger().info(
                    "wrote %d frames to %s" % (self._frames_written, self._output))
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = N1RunRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

