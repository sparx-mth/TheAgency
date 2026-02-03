#!/usr/bin/env python3
"""
Overlay Recorder Node — Side-by-Side Display

Shows two panels:
  LEFT:  S2 inference frame with red pixel-goal circle (from bridge waypoint_image topic)
  RIGHT: Live video stream with text overlays (instruction, action, status, FPS)

Subscribes to:
- RGB image (sensor_msgs/Image or sensor_msgs/CompressedImage)
- Instruction (std_msgs/String)
- Action (std_msgs/String)
- Status (std_msgs/String)
- Waypoint image (sensor_msgs/Image) — the S2 frame annotated by bridge_node

This is simulator-agnostic (Gazebo/Sphera/anything) as long as ROS2 topics exist.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


@dataclass
class TextState:
    instruction: str = ""
    action: str = ""
    status: str = ""


def _wrap_text(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if len(nxt) <= max_chars:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class OverlayRecorder(Node):
    def __init__(self):
        super().__init__("internnav_overlay_recorder")

        # -------- Params --------
        self.declare_parameter("rgb_topic", "/camera/rgb/image_raw")
        self.declare_parameter("rgb_type", "raw")  # "raw" | "compressed"
        self.declare_parameter("instruction_topic", "/navigation/instruction")
        self.declare_parameter("action_topic", "/navigation/action")
        self.declare_parameter("status_topic", "/navigation/status")
        self.declare_parameter("waypoint_image_topic", "")  # empty = auto-derive from action_topic

        self.declare_parameter("output", "/tmp/internnav_overlay.mp4")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("target_width", 640)
        self.declare_parameter("target_height", 480)
        self.declare_parameter("show_preview", False)

        # overlay look
        self.declare_parameter("max_chars_per_line", 48)
        self.declare_parameter("max_lines", 3)
        self.declare_parameter("draw_fps", True)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.rgb_type = self.get_parameter("rgb_type").value
        self.instruction_topic = self.get_parameter("instruction_topic").value
        self.action_topic = self.get_parameter("action_topic").value
        self.status_topic = self.get_parameter("status_topic").value

        wp_img_topic = self.get_parameter("waypoint_image_topic").value
        if not wp_img_topic:
            # Derive: /R1/navigation/action -> /R1/navigation/waypoint_image
            wp_img_topic = self.action_topic.rsplit('/', 1)[0] + "/waypoint_image"
        self.waypoint_image_topic = wp_img_topic

        self.output_path = self.get_parameter("output").value
        self.fps = float(self.get_parameter("fps").value)
        self.tw = int(self.get_parameter("target_width").value)
        self.th = int(self.get_parameter("target_height").value)
        self.show_preview = bool(self.get_parameter("show_preview").value)

        self.max_chars_per_line = int(self.get_parameter("max_chars_per_line").value)
        self.max_lines = int(self.get_parameter("max_lines").value)
        self.draw_fps = bool(self.get_parameter("draw_fps").value)

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        # -------- ROS --------
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.text = TextState()
        self.last_s2_frame = None  # latest S2 inference frame with red circle

        if self.rgb_type.lower() == "compressed":
            self.rgb_sub = self.create_subscription(
                CompressedImage, self.rgb_topic, self._rgb_cb_compressed, qos_img
            )
        else:
            self.rgb_sub = self.create_subscription(
                Image, self.rgb_topic, self._rgb_cb_raw, qos_img
            )

        self.inst_sub = self.create_subscription(String, self.instruction_topic, self._inst_cb, 10)
        self.action_sub = self.create_subscription(String, self.action_topic, self._action_cb, 10)
        self.status_sub = self.create_subscription(String, self.status_topic, self._status_cb, 10)

        # S2 waypoint image from bridge_node (Image with red circle already drawn)
        self.waypoint_image_sub = self.create_subscription(
            Image, self.waypoint_image_topic, self._waypoint_image_cb, qos_img
        )

        # -------- Video --------
        # Side-by-side: total width = 2 * target_width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.tw * 2, self.th), True)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.output_path}")

        self.last_frame_t = None
        self.ema_fps = None

        self.get_logger().info(f"Recording to: {self.output_path}  (side-by-side {self.tw*2}x{self.th})")
        self.get_logger().info(f"RGB: {self.rgb_topic} ({self.rgb_type}) | inst: {self.instruction_topic}")
        self.get_logger().info(f"action: {self.action_topic} | status: {self.status_topic}")
        self.get_logger().info(f"S2 waypoint image: {self.waypoint_image_topic}")

    def destroy_node(self):
        try:
            if hasattr(self, "writer") and self.writer:
                self.writer.release()
        except Exception:
            pass
        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()

    # -------- Text callbacks --------
    def _inst_cb(self, msg: String):
        self.text.instruction = msg.data.strip()

    def _action_cb(self, msg: String):
        self.text.action = msg.data.strip()

    def _status_cb(self, msg: String):
        self.text.status = msg.data.strip()

    # -------- S2 waypoint image callback --------
    def _waypoint_image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame.shape[1] != self.tw or frame.shape[0] != self.th:
                frame = cv2.resize(frame, (self.tw, self.th), interpolation=cv2.INTER_AREA)
            self.last_s2_frame = frame
            self.get_logger().info("S2 frame received", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"Waypoint image error: {e}")

    # -------- Image callbacks --------
    def _rgb_cb_raw(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._handle_frame(frame)
        except Exception as e:
            self.get_logger().error(f"RGB(raw) error: {e}")

    def _rgb_cb_compressed(self, msg: CompressedImage):
        try:
            npbuf = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
            if frame is None:
                return
            self._handle_frame(frame)
        except Exception as e:
            self.get_logger().error(f"RGB(compressed) error: {e}")

    def _handle_frame(self, frame_bgr: np.ndarray):
        # resize to deterministic output size
        if frame_bgr.shape[1] != self.tw or frame_bgr.shape[0] != self.th:
            frame_bgr = cv2.resize(frame_bgr, (self.tw, self.th), interpolation=cv2.INTER_AREA)

        # compute FPS
        now = time.time()
        if self.last_frame_t is not None:
            inst_fps = 1.0 / max(1e-6, (now - self.last_frame_t))
            if self.ema_fps is None:
                self.ema_fps = inst_fps
            else:
                self.ema_fps = 0.9 * self.ema_fps + 0.1 * inst_fps
        self.last_frame_t = now

        # RIGHT panel: live stream with text overlays (unchanged from original)
        right = self._draw_overlay(frame_bgr, now)

        # LEFT panel: S2 inference frame with red circle, or placeholder
        left = self._get_s2_panel()

        # Combine side-by-side: [S2 GOAL | LIVE STREAM]
        combined = np.hstack((left, right))

        self.writer.write(combined)

        if self.show_preview:
            cv2.imshow("InternNav  |  S2 Goal  |  Live Stream", combined)
            cv2.waitKey(1)

    def _get_s2_panel(self) -> np.ndarray:
        """Return the left panel: last S2 frame or a dark placeholder."""
        if self.last_s2_frame is not None:
            panel = self.last_s2_frame.copy()
        else:
            panel = np.zeros((self.th, self.tw, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(panel, "Waiting for S2...", (self.tw // 2 - 130, self.th // 2),
                        font, 0.7, (100, 100, 100), 2, cv2.LINE_AA)

        # Panel label top-left
        cv2.rectangle(panel, (0, 0), (210, 32), (0, 0, 0), -1)
        cv2.putText(panel, "S2 PIXEL GOAL", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return panel

    def _draw_overlay(self, img: np.ndarray, now_ts: float) -> np.ndarray:
        out = img.copy()

        # background box
        pad = 10
        box_h = 110
        cv2.rectangle(out, (0, 0), (self.tw, box_h), (0, 0, 0), thickness=-1)

        # text layout
        y = 28
        line_gap = 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2

        status = self.text.status
        action = self.text.action
        inst = self.text.instruction

        # line 1: status + action + time
        tstr = time.strftime("%H:%M:%S", time.localtime(now_ts))
        l1 = f"STATUS: {status or '-'}   ACTION: {action or '-'}   TIME: {tstr}"
        cv2.putText(out, l1, (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_gap

        # line 2..: instruction wrapped
        wrapped = _wrap_text(inst, self.max_chars_per_line)[: self.max_lines]
        if not wrapped:
            wrapped = ["-"]
        cv2.putText(out, "INSTR:", (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        x0 = pad + 80
        for i, line in enumerate(wrapped):
            cv2.putText(out, line, (x0, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            y += line_gap

        # bottom-right FPS
        if self.draw_fps and self.ema_fps is not None:
            fps_txt = f"{self.ema_fps:.1f} FPS"
            (tw, th), _ = cv2.getTextSize(fps_txt, font, scale, thick)
            cv2.putText(out, fps_txt, (self.tw - tw - 10, self.th - 12),
                        font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        return out


def main():
    rclpy.init()
    node = OverlayRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()