# sparx_agency/robots/common/ros2/tag_azimuth_node.py
from __future__ import annotations

import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32

import tf2_ros
from tf2_ros import TransformException

from sparx_agency.core.localization.tag_azimuth_estimator import TagAzimuthEstimator
from sparx_agency.core.localization.types.tag_azimuth import TagBearingObservation


class TagAzimuthNode(Node):
    """
    ROS2 adapter:
    - Reads tag configuration from YAML
    - Looks up TF transforms camera_frame -> tag_frame for known tags
    - Builds observations (tx,tz) and passes to core estimator
    - Publishes /<robot_ns>/camera_azimuth (Float32 degrees 0..360)
    """

    def __init__(self):
        super().__init__("tag_azimuth_node")

        # --------------------
        # Parameters (generic)
        # --------------------
        self.declare_parameter("robot_ns", "")  # "" or "R1" or "/R1" etc.
        self.declare_parameter("camera_frame", "")  # explicit camera frame
        self.declare_parameter("tag_family", "36h11")
        self.declare_parameter("tag_config_path", "")  # YAML file path
        self.declare_parameter("publish_topic", "")  # optional override
        self.declare_parameter("timer_hz", 5.0)
        self.declare_parameter("history_len", 20)
        self.declare_parameter("max_time_diff_sec", 1.0)

        self.declare_parameter("candidate_tag_frames", [
            "tag{family}:{id}",
            "tag_{id}",
            "tag{id}",
        ])

        self.robot_ns = self._clean_ns(self.get_parameter("robot_ns").value)
        self.tag_family = str(self.get_parameter("tag_family").value).strip()

        camera_frame = str(self.get_parameter("camera_frame").value).strip()
        self.camera_frame = camera_frame if camera_frame else self._default_camera_frame()

        # Load YAML config for tags (id -> wall azimuth)
        cfg_path = str(self.get_parameter("tag_config_path").value).strip()
        if not cfg_path:
            raise RuntimeError("tag_config_path param is required (YAML with tag azimuth mapping)")
        self.tag_config_deg = self._load_tag_config(cfg_path)

        history_len = int(self.get_parameter("history_len").value)
        max_dt = float(self.get_parameter("max_time_diff_sec").value)
        self.estimator = TagAzimuthEstimator(
            tag_config_deg=self.tag_config_deg,
            max_history=history_len,
            max_time_diff_sec=max_dt,
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher topic
        publish_topic = str(self.get_parameter("publish_topic").value).strip()
        if publish_topic:
            topic = publish_topic
        else:
            # default: /<robot_ns>/camera_azimuth  OR /camera_azimuth if robot_ns empty
            topic = f"/{self.robot_ns}/camera_azimuth" if self.robot_ns else "/camera_azimuth"

        self.pub = self.create_publisher(Float32, topic, 10)

        # Candidate frame templates
        self.candidate_templates = list(self.get_parameter("candidate_tag_frames").value)

        # Timer
        hz = float(self.get_parameter("timer_hz").value)
        period = 1.0 / max(hz, 0.1)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(f"TagAzimuthNode started")
        self.get_logger().info(f"robot_ns='{self.robot_ns}' camera_frame='{self.camera_frame}' tag_family='{self.tag_family}'")
        self.get_logger().info(f"known_tag_ids={sorted(self.tag_config_deg.keys())}")
        self.get_logger().info(f"publishing to: {topic}")

    # --------------------
    # Helpers
    # --------------------
    def _clean_ns(self, s: str) -> str:
        s = str(s).strip()
        if s.startswith("/"):
            s = s[1:]
        return s.rstrip("/")

    def _default_camera_frame(self) -> str:
        # Generic fallback; you should override via param if your TF differs.
        # If you want robot-specific naming: set camera_frame param in launch.
        return f"{self.robot_ns}_camera" if self.robot_ns else "camera"

    def _load_tag_config(self, path: str) -> Dict[int, float]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tag_config_path does not exist: {path}")

        with p.open("r") as f:
            data = yaml.safe_load(f)

        # Allow two formats:
        # 1) { tags: {10: 0.0, 11: 90.0, ...} }
        # 2) { 10: 0.0, 11: 90.0, ... }
        tags = data.get("tags", data)

        out: Dict[int, float] = {}
        for k, v in tags.items():
            out[int(k)] = float(v)

        if not out:
            raise ValueError(f"No tags found in config: {path}")
        return out

    def _candidate_frames_for_id(self, tag_id: int) -> List[str]:
        frames = []
        for templ in self.candidate_templates:
            frames.append(
                templ.format(family=self.tag_family, id=tag_id)
            )
        return frames

    def _lookup_camera_to_tag(self, tag_frame: str):
        return self.tf_buffer.lookup_transform(
            self.camera_frame,  # target
            tag_frame,          # source
            rclpy.time.Time(),  # latest
        )

    # --------------------
    # Main loop
    # --------------------
    def _on_timer(self):
        observations: List[TagBearingObservation] = []
        last_err: Optional[Exception] = None

        for tag_id in self.estimator.known_tag_ids:
            transform = None
            for frame in self._candidate_frames_for_id(tag_id):
                try:
                    transform = self._lookup_camera_to_tag(frame)
                    break
                except TransformException as e:
                    last_err = e
                    continue

            if transform is None:
                continue

            t = transform.transform.translation
            tx = float(t.x)
            tz = float(t.z)

            # If tz is ~0, atan2 might still be defined, but can be noisy.
            # We keep it and let core handle edge cases.
            observations.append(TagBearingObservation(tag_id=tag_id, tx=tx, tz=tz))

        if not observations:
            self.get_logger().warn(
                f"No tags visible. Last TF error: {last_err}",
                throttle_duration_sec=2.0,
            )
            return

        now = self.get_clock().now()
        stamp_sec = now.nanoseconds / 1e9

        result = self.estimator.update(observations, stamp_sec=stamp_sec)
        if result is None:
            self.get_logger().warn("Estimator returned None (no usable observations).")
            return

        yaw_deg, best_tag = result

        msg = Float32()
        msg.data = float(yaw_deg)
        self.pub.publish(msg)

        self.get_logger().info(
            f"Azimuth: {yaw_deg:.1f}° (Best tag: {best_tag})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TagAzimuthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
