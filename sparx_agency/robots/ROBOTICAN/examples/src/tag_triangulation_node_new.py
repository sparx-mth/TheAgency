#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
from apriltag_msgs.msg import AprilTagDetectionArray

from std_msgs.msg import Int32MultiArray

from sparx_agency.robots.common.spatial_math import (
    rpy_to_transform,
    quat_to_transform,
    rot_to_quat,
    transform_to_pose,
)

# ===================  NODE  =================== #

class TagTriangulationNode(Node):

    def __init__(self):
        super().__init__("tag_triangulation_node")

        self.world_frame = "world"
        self.camera_frame = "/R1/camera/image_raw"  # must exist in /tf

        # Known tag poses in the world frame
        self.tag_world_poses = {
            "tag36h11:14": {"xyz": (11.195, 0.0, 1.5), "rpy": (0.0, -1.5708, 0.0)},
            "tag36h11:15": {"xyz": (11.495, 1.0, 1.5), "rpy": (0.0, -1.5708, 0.0)},
        }

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, "/tag_pose", 10)
        self.ids_pub = self.create_publisher(Int32MultiArray, "/tag_pose_ids", 10)

        # Subscriber to detections
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            "/detections",
            self.detections_cb,
            10,
        )

        # Keep last seen IDs to detect "new tag only" events
        self.last_seen_ids = set()

        self.get_logger().info("TagTriangulationNode started (NEW TAG only).")

    # ---------------------- HELPERS ---------------------- #

    @staticmethod
    def get_detection_id(det):
        """
        apriltag_msgs sometimes publishes det.id as an array (e.g., [14]).
        This helper safely extracts a single int.
        """
        try:
            # If id is array-like
            if hasattr(det.id, "__len__"):
                return int(det.id[0])
            return int(det.id)
        except Exception:
            return None

    def safe_lookup_cam_to_tag(self, tag_frame, detection_stamp):
        """
        Try stamped TF first (sync to detection time), then latest TF.
        Also tries with/without leading slashes for robustness.
        Returns TransformStamped or None.
        """
        cam_frames = [self.camera_frame, f"/{self.camera_frame.lstrip('/')}"]
        tag_frames = [tag_frame, f"/{tag_frame.lstrip('/')}"]

        t_meas = rclpy.time.Time.from_msg(detection_stamp)

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

    # ---------------------- CALLBACK ---------------------- #

    def detections_cb(self, msg: AprilTagDetectionArray):

        # No detections -> reset history
        if not msg.detections:
            self.last_seen_ids = set()
            return

        # Current IDs in this frame
        current_ids = set()
        for det in msg.detections:
            det_id = self.get_detection_id(det)
            if det_id is not None:
                current_ids.add(det_id)

        if not current_ids:
            self.last_seen_ids = set()
            return

        # New tag event = tag that was not seen before
        new_ids = current_ids - self.last_seen_ids
        if not new_ids:
            self.last_seen_ids = current_ids
            return

        self.get_logger().info(f"[NEW TAG EVENT] new_ids={new_ids}")

        # Compute camera pose in world for valid tags
        cam_poses_world = []
        detection_stamp = msg.header.stamp

        for tag_id in current_ids:
            tag_frame = f"tag36h11:{tag_id}"  # normalize: no leading "/"

            if tag_frame not in self.tag_world_poses:
                self.get_logger().warn(
                    f"Tag {tag_frame} detected but no world pose defined."
                )
                continue

            xyz = self.tag_world_poses[tag_frame]["xyz"]
            rpy = self.tag_world_poses[tag_frame]["rpy"]
            world_T_tag = self.pose_to_matrix(xyz, rpy)

            try:
                cam_T_tag_tf = self.safe_lookup_cam_to_tag(tag_frame, detection_stamp)
                if cam_T_tag_tf is None:
                    self.get_logger().warn(
                        f"TF lookup failed (stamped+latest) for {self.camera_frame} -> {tag_frame}"
                    )
                    continue

                cam_T_tag = self.transform_to_matrix(cam_T_tag_tf)
                tag_T_cam = np.linalg.inv(cam_T_tag)
                world_T_cam = world_T_tag @ tag_T_cam
                cam_poses_world.append(world_T_cam)

            except Exception as e:
                self.get_logger().warn(
                    f"Failed processing tag {tag_frame}: {e}"
                )
                continue

        if not cam_poses_world:
            self.get_logger().warn("No valid TF/world poses -> not publishing.")
            self.last_seen_ids = current_ids
            return

        # If only one tag -> use it. If multiple -> average translation only.
        if len(cam_poses_world) == 1:
            world_T_cam_avg = cam_poses_world[0]
        else:
            translations = np.array([M[:3, 3] for M in cam_poses_world])
            avg_t = np.mean(translations, axis=0)

            # Keep rotation from first tag for simplicity
            R = cam_poses_world[0][:3, :3]

            world_T_cam_avg = np.eye(4)
            world_T_cam_avg[:3, :3] = R
            world_T_cam_avg[:3, 3] = avg_t

        (x, y, z), (qx, qy, qz, qw) = transform_to_pose(world_T_cam_avg)

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
        ids_msg.data = sorted(list(current_ids))
        self.ids_pub.publish(ids_msg)


        self.get_logger().info(
            f"[PUBLISH] pose=({x:.2f}, {y:.2f}, {z:.2f}), ids={current_ids}"
        )

        self.last_seen_ids = current_ids

    # ---------------------- MATRIX HELPERS ---------------------- #

    @staticmethod
    def transform_to_matrix(t: TransformStamped):
        q = t.transform.rotation
        tr = t.transform.translation
        M = quat_to_transform([q.x, q.y, q.z, q.w])
        M[0, 3] = tr.x
        M[1, 3] = tr.y
        M[2, 3] = tr.z
        return M

    @staticmethod
    def pose_to_matrix(xyz, rpy):
        x, y, z = xyz
        roll, pitch, yaw = rpy
        M = rpy_to_transform(roll, pitch, yaw)
        M[0, 3] = x
        M[1, 3] = y
        M[2, 3] = z
        return M


# ===================  MAIN  =================== #

def main():
    rclpy.init()
    node = TagTriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
