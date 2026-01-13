# sparx_agency/tasks/localization/nodes/tag_triangulation_node.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Int32MultiArray

from apriltag_msgs.msg import AprilTagDetectionArray

from sparx_agency.core.localization.tag_triangulation import (
    TagWorldPose,
    TagObservation,
    estimate_camera_pose_from_tags,
    matrix_to_pose,
)


class TagTriangulationNode(Node):
    """
    ROS2 node that:
    - Subscribes to AprilTag detections
    - Looks up TF (camera -> tag) at detection stamp (fallback latest)
    - Uses known tag world poses (from YAML) to estimate camera pose in world
    - Publishes PoseStamped + used tag ids (+ optional timing debug)

    This node is platform-agnostic: adapt to any drone via params:
      - camera_frame
      - detections_topic
      - tag_map_path
      - tag_family
      - tag_frame_templates
      - publish_policy
    """

    def __init__(self):
        super().__init__("tag_triangulation_node")

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter("use_sim_time", True)

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("detections_topic", "/detections")

        self.declare_parameter("tag_family", "36h11")
        self.declare_parameter("tag_map_path", "")  # YAML file for tag poses in world

        # how to form TF frame names for tags
        self.declare_parameter("tag_frame_templates", [
            "tag{family}:{id}",
            "tag_{id}",
            "tag{id}",
        ])

        # Publish behavior:
        #  - "new_tag_only": publish only when a new tag appears compared to previous frame
        #  - "always": publish on every detection message (if pose can be estimated)
        self.declare_parameter("publish_policy", "new_tag_only")

        # Fuse method (keep same behavior as your demo)
        self.declare_parameter("fuse_method", "avg_translation_keep_first_rotation")

        # Outputs
        self.declare_parameter("pose_topic", "/tag_pose")
        self.declare_parameter("ids_topic", "/tag_pose_ids")
        self.declare_parameter("timing_topic", "/pose_publish_time")  # optional; empty to disable

        # Read params
        self.world_frame = str(self.get_parameter("world_frame").value).strip()
        self.camera_frame = str(self.get_parameter("camera_frame").value).strip()
        if not self.camera_frame:
            raise RuntimeError("camera_frame param is required (must exist in /tf)")

        self.detections_topic = str(self.get_parameter("detections_topic").value).strip()
        self.tag_family = str(self.get_parameter("tag_family").value).strip()

        tag_map_path = str(self.get_parameter("tag_map_path").value).strip()
        if not tag_map_path:
            raise RuntimeError("tag_map_path param is required (YAML file with tag world poses)")
        self.tag_map: Dict[int, TagWorldPose] = self._load_tag_map(tag_map_path)

        self.tag_frame_templates: List[str] = list(self.get_parameter("tag_frame_templates").value)
        self.publish_policy = str(self.get_parameter("publish_policy").value).strip().lower()
        self.fuse_method = str(self.get_parameter("fuse_method").value).strip()

        self.pose_topic = str(self.get_parameter("pose_topic").value).strip()
        self.ids_topic = str(self.get_parameter("ids_topic").value).strip()
        self.timing_topic = str(self.get_parameter("timing_topic").value).strip()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.ids_pub = self.create_publisher(Int32MultiArray, self.ids_topic, 10)
        self.timing_pub = None
        if self.timing_topic:
            self.timing_pub = self.create_publisher(Int32MultiArray, self.timing_topic, 10)

        # Subscriber
        self.sub = self.create_subscription(
            AprilTagDetectionArray,
            self.detections_topic,
            self.detections_cb,
            10,
        )

        # state for "new tag only"
        self.last_seen_ids: Set[int] = set()

        self.get_logger().info("TagTriangulationNode started.")
        self.get_logger().info(f"camera_frame='{self.camera_frame}', world_frame='{self.world_frame}', topic='{self.detections_topic}'")
        self.get_logger().info(f"known tag ids from map: {sorted(list(self.tag_map.keys()))}")
        self.get_logger().info(f"publish_policy='{self.publish_policy}', fuse_method='{self.fuse_method}'")

    # ------------------------
    # YAML map loading
    # ------------------------
    def _load_tag_map(self, path: str) -> Dict[int, TagWorldPose]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tag_map_path not found: {path}")

        with p.open("r") as f:
            data = yaml.safe_load(f)

        # Expected:
        # tags:
        #   14: { xyz: [...], rpy: [...] }
        # or directly at root {14: {...}}
        tags = data.get("tags", data)

        out: Dict[int, TagWorldPose] = {}
        for k, v in tags.items():
            tid = int(k)
            xyz = tuple(float(x) for x in v["xyz"])
            rpy = tuple(float(x) for x in v["rpy"])
            out[tid] = TagWorldPose(xyz=xyz, rpy=rpy)

        if not out:
            raise ValueError(f"No tags loaded from: {path}")
        return out

    # ------------------------
    # Detection parsing helpers
    # ------------------------
    @staticmethod
    def get_detection_id(det) -> Optional[int]:
        try:
            if hasattr(det.id, "__len__"):
                return int(det.id[0])
            return int(det.id)
        except Exception:
            return None

    def candidate_tag_frames(self, tag_id: int) -> List[str]:
        return [t.format(family=self.tag_family, id=tag_id) for t in self.tag_frame_templates]

    def safe_lookup_cam_to_tag(self, tag_frame: str, detection_stamp) -> Optional[TransformStamped]:
        """
        Try stamped TF first (sync to detection time), then latest TF.
        Also tries with/without leading slashes.
        """
        cam_frames = [self.camera_frame, f"/{self.camera_frame.lstrip('/')}"]
        tag_frames = [tag_frame, f"/{tag_frame.lstrip('/')}"]

        t_meas = Time.from_msg(detection_stamp)

        # 1) stamped TF
        for cf in cam_frames:
            for tf_ in tag_frames:
                try:
                    return self.tf_buffer.lookup_transform(cf, tf_, t_meas)
                except Exception:
                    pass

        # 2) latest TF
        for cf in cam_frames:
            for tf_ in tag_frames:
                try:
                    return self.tf_buffer.lookup_transform(cf, tf_, rclpy.time.Time())
                except Exception:
                    pass

        return None

    @staticmethod
    def transform_to_matrix(t: TransformStamped) -> np.ndarray:
        q = t.transform.rotation
        tr = t.transform.translation
        # quaternion -> matrix
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)

        # normalize
        norm = (x*x + y*y + z*z + w*w) ** 0.5
        if norm == 0.0:
            M = np.eye(4, dtype=float)
        else:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z

            R = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
                [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
                [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
            ], dtype=float)

            M = np.eye(4, dtype=float)
            M[:3, :3] = R

        M[0, 3] = float(tr.x)
        M[1, 3] = float(tr.y)
        M[2, 3] = float(tr.z)
        return M

    # ------------------------
    # Main callback
    # ------------------------
    def detections_cb(self, msg: AprilTagDetectionArray):
        if not msg.detections:
            self.last_seen_ids = set()
            return

        current_ids: Set[int] = set()
        for det in msg.detections:
            tid = self.get_detection_id(det)
            if tid is not None:
                current_ids.add(tid)

        if not current_ids:
            self.last_seen_ids = set()
            return

        # publish policy gate
        if self.publish_policy == "new_tag_only":
            new_ids = current_ids - self.last_seen_ids
            if not new_ids:
                self.last_seen_ids = current_ids
                return
            self.get_logger().info(f"[NEW TAG EVENT] new_ids={sorted(list(new_ids))}")

        detection_stamp = msg.header.stamp

        # Build observations (only for tags we know in tag_map)
        observations: List[TagObservation] = []
        used_for_tf: Set[int] = set()

        for tag_id in sorted(list(current_ids)):
            if tag_id not in self.tag_map:
                continue

            # Try multiple possible tag frame names
            tf_found = None
            tf_frame_used = None
            for tag_frame in self.candidate_tag_frames(tag_id):
                tf_found = self.safe_lookup_cam_to_tag(tag_frame, detection_stamp)
                if tf_found is not None:
                    tf_frame_used = tag_frame
                    break

            if tf_found is None:
                self.get_logger().warn(
                    f"TF lookup failed for tag_id={tag_id} (camera='{self.camera_frame}')",
                    throttle_duration_sec=2.0,
                )
                continue

            cam_T_tag = self.transform_to_matrix(tf_found)
            observations.append(TagObservation(tag_id=tag_id, cam_T_tag=cam_T_tag))
            used_for_tf.add(tag_id)

        if not observations:
            self.get_logger().warn("No valid tag observations -> not publishing.")
            self.last_seen_ids = current_ids
            return

        est = estimate_camera_pose_from_tags(
            observations=observations,
            tag_map=self.tag_map,
            fuse_method=self.fuse_method,
        )

        if est is None:
            self.get_logger().warn("Estimator returned None -> not publishing.")
            self.last_seen_ids = current_ids
            return

        (x, y, z), (qx, qy, qz, qw) = matrix_to_pose(est.world_T_cam)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.world_frame
        pose_msg.header.stamp = detection_stamp
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        self.pose_pub.publish(pose_msg)

        ids_msg = Int32MultiArray()
        ids_msg.data = sorted(list(set(est.used_tag_ids)))
        self.ids_pub.publish(ids_msg)

        # optional timing debug: [meas_sec, meas_ns, pub_sec, pub_ns]
        if self.timing_pub is not None:
            publish_time = self.get_clock().now().to_msg()
            time_msg = Int32MultiArray()
            time_msg.data = [
                int(detection_stamp.sec),
                int(detection_stamp.nanosec),
                int(publish_time.sec),
                int(publish_time.nanosec),
            ]
            self.timing_pub.publish(time_msg)

        self.get_logger().info(
            f"[PUBLISH] pose=({x:.2f}, {y:.2f}, {z:.2f}), used_ids={sorted(list(set(est.used_tag_ids)))}"
        )

        self.last_seen_ids = current_ids


def main(args=None):
    rclpy.init(args=args)
    node = TagTriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
