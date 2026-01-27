#!/usr/bin/env python3
from __future__ import annotations

import math
import csv
from collections import deque
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import Vector3Stamped, Pose, PoseStamped
from rclpy.duration import Duration


def norm_frame(frame: str) -> str:
    # TF2 in ROS2: frame_ids must NOT start with '/'
    if frame is None:
        return ""
    return frame.lstrip("/")


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # yaw from quaternion (Z axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate_xy(vx: float, vy: float, yaw: float) -> Tuple[float, float]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (cy * vx - sy * vy, sy * vx + cy * vy)


class FlowDepthPoseEvalNode(Node):
    """
    Integrates /flow_depth/velocity into a pose estimate and compares to /simple_drone/gt_pose.

    Outputs:
      - Publishes PoseStamped estimate to publish_pose_topic
      - Optional CSV: merged rows (est + gt + err)
      - Optional TUM trajectories:
          * est_tum_path:  t x y z qx qy qz qw (identity quat)
          * gt_tum_path:   t x y z qx qy qz qw (from GT pose quat)

    Notes:
      - GT topic is geometry_msgs/Pose (no header). We write GT with the SAME timestamp
        as the velocity sample, because that's what we compare against.
      - TF lookup uses the velocity message timestamp (better for bag replay).
    """

    def __init__(self):
        super().__init__("flow_depth_pose_eval_node")


        # params
        self.declare_parameter("vel_topic", "/flow_depth/velocity")
        self.declare_parameter("gt_pose_topic", "/simple_drone/gt_pose")
        self.declare_parameter("target_frame", "/simple_drone/odom")  # compare/integrate in this frame
        self.declare_parameter("publish_pose_topic", "/flow_depth/pose_est")
        self.declare_parameter("init_from_gt", True)     # start est at first GT pose
        self.declare_parameter("csv_path", "")           # e.g. /tmp/pose_eval.csv (optional)
        self.declare_parameter("print_every_sec", 1.0)

        # TUM outputs for evo
        self.declare_parameter("est_tum_path", "")       # e.g. /tmp/est_tum.txt
        self.declare_parameter("gt_tum_path", "")        # e.g. /tmp/gt_tum.txt
        self.declare_parameter("flush_every_n", 200)     # flush files every N writes (0 disables)

        vel_topic = self.get_parameter("vel_topic").get_parameter_value().string_value
        gt_topic = self.get_parameter("gt_pose_topic").get_parameter_value().string_value
        self.target_frame = norm_frame(self.get_parameter("target_frame").get_parameter_value().string_value)
        pose_topic = self.get_parameter("publish_pose_topic").get_parameter_value().string_value
        self.init_from_gt = self.get_parameter("init_from_gt").get_parameter_value().bool_value
        self.csv_path = self.get_parameter("csv_path").get_parameter_value().string_value
        self.print_every_sec = float(self.get_parameter("print_every_sec").get_parameter_value().double_value)

        self.est_tum_path = self.get_parameter("est_tum_path").get_parameter_value().string_value
        self.gt_tum_path = self.get_parameter("gt_tum_path").get_parameter_value().string_value
        self.flush_every_n = int(self.get_parameter("flush_every_n").get_parameter_value().integer_value)

        self.get_logger().info(f"[PoseEval] vel: {vel_topic}")
        self.get_logger().info(f"[PoseEval] gt_pose: {gt_topic} (Pose without header)")
        self.get_logger().info(f"[PoseEval] target_frame: {self.target_frame}")
        self.get_logger().info(f"[PoseEval] publish pose: {pose_topic}")
        if self.est_tum_path:
            self.get_logger().info(f"[PoseEval] est_tum_path: {self.est_tum_path}")
        if self.gt_tum_path:
            self.get_logger().info(f"[PoseEval] gt_tum_path: {self.gt_tum_path}")

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # state
        self.have_est = False
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_z = 0.0
        self.last_vel_stamp = None  # builtin_interfaces/Time

        self.gt_pose_latest: Optional[Pose] = None  # last received GT pose

        # error stats
        self.err_hist = deque(maxlen=5000)
        self.last_print_time = self.get_clock().now()

        # CSV (merged)
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t_sec", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z", "err_m"])

        # TUM files
        self.est_tum_file = open(self.est_tum_path, "w", encoding="utf-8") if self.est_tum_path else None
        self.gt_tum_file = open(self.gt_tum_path, "w", encoding="utf-8") if self.gt_tum_path else None
        self._tum_write_count = 0

        # pubs/subs
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.gt_sub = self.create_subscription(Pose, gt_topic, self.gt_pose_cb, qos_profile_sensor_data)
        self.vel_sub = self.create_subscription(Vector3Stamped, vel_topic, self.vel_cb, qos_profile_sensor_data)

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        if self.est_tum_file:
            self.est_tum_file.close()
        if self.gt_tum_file:
            self.gt_tum_file.close()
        super().destroy_node()

    # -------- callbacks --------
    def gt_pose_cb(self, msg: Pose):
        # store latest GT pose
        self.gt_pose_latest = msg

        # initialize estimate to GT at first message (optional)
        if self.init_from_gt and (not self.have_est):
            self.est_x = float(msg.position.x)
            self.est_y = float(msg.position.y)
            self.est_z = float(msg.position.z)
            self.have_est = True
            self.get_logger().info("[PoseEval] Initialized estimated pose from first GT pose.")

    def vel_cb(self, msg: Vector3Stamped):
        """Callback for velocity messages."""
        # Need GT for comparison (and possibly init)
        if self.gt_pose_latest is None and self.init_from_gt:
            return

        # for first velocity message, only store stamp (no integration)
        if self.last_vel_stamp is None:
            self.last_vel_stamp = msg.header.stamp
            return

        # dt from velocity stamps
        dt_ns = (msg.header.stamp.sec - self.last_vel_stamp.sec) * 1_000_000_000 + \
                (msg.header.stamp.nanosec - self.last_vel_stamp.nanosec)
        dt = float(dt_ns) * 1e-9
        self.last_vel_stamp = msg.header.stamp

        # skip crazy dt (bag pauses / jumps)
        if dt <= 0.0 or dt > 1.0:
            return

        # Ensure we have an estimate
        if not self.have_est:
            self.have_est = True  # start at origin if not init_from_gt

        # velocity in camera frame
        v_cam_x = float(msg.vector.x)
        v_cam_y = float(msg.vector.y)
        v_cam_z = float(msg.vector.z)

        # Transform velocity to target frame using TF at the SAME timestamp as msg
        try:
            source_frame = norm_frame(msg.header.frame_id)
            target_frame = self.target_frame

            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                msg.header.stamp,  # IMPORTANT: timestamped TF lookup (better for bag replay)
            )

            q = tf.transform.rotation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            v_x, v_y = rotate_xy(v_cam_x, v_cam_y, yaw)
            v_z = v_cam_z

        except TransformException as e:
            self.get_logger().warn(f"[PoseEval] TF lookup failed: {e}")
            return

        # Integrate position
        self.est_x += v_x * dt
        self.est_y += v_y * dt
        self.est_z += v_z * dt

        # Publish estimated pose
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target_frame
        out.pose.position.x = float(self.est_x)
        out.pose.position.y = float(self.est_y)
        out.pose.position.z = float(self.est_z)
        out.pose.orientation.w = 1.0  # unknown -> identity
        self.pose_pub.publish(out)

        # time in seconds for logs/files
        t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Write EST TUM
        if self.est_tum_file is not None:
            self.est_tum_file.write(
                f"{t_sec:.9f} {self.est_x:.9f} {self.est_y:.9f} {self.est_z:.9f} 0.0 0.0 0.0 1.0\n"
            )

        # Compare to GT + Write GT TUM
        if self.gt_pose_latest is not None:
            gx = float(self.gt_pose_latest.position.x)
            gy = float(self.gt_pose_latest.position.y)
            gz = float(self.gt_pose_latest.position.z)

            if self.gt_tum_file is not None:
                gq = self.gt_pose_latest.orientation
                self.gt_tum_file.write(
                    f"{t_sec:.9f} {gx:.9f} {gy:.9f} {gz:.9f} {gq.x:.9f} {gq.y:.9f} {gq.z:.9f} {gq.w:.9f}\n"
                )

            err = math.sqrt((self.est_x - gx) ** 2 + (self.est_y - gy) ** 2 + (self.est_z - gz) ** 2)
            self.err_hist.append(err)

            # CSV
            if self.csv_writer is not None:
                self.csv_writer.writerow([t_sec, self.est_x, self.est_y, self.est_z, gx, gy, gz, err])

        # Flush periodically (safer if interrupted)
        if self.flush_every_n > 0 and (self.est_tum_file or self.gt_tum_file or self.csv_file):
            self._tum_write_count += 1
            if self._tum_write_count % self.flush_every_n == 0:
                try:
                    if self.est_tum_file:
                        self.est_tum_file.flush()
                    if self.gt_tum_file:
                        self.gt_tum_file.flush()
                    if self.csv_file:
                        self.csv_file.flush()
                except Exception:
                    pass

        # Periodic print
        now = self.get_clock().now()
        if (now - self.last_print_time).nanoseconds * 1e-9 >= self.print_every_sec:
            self.last_print_time = now
            if len(self.err_hist) > 0:
                last = self.err_hist[-1]
                rms = math.sqrt(sum(e * e for e in self.err_hist) / len(self.err_hist))
                self.get_logger().info(f"[PoseEval] err_last={last:.3f} m, err_rms({len(self.err_hist)})={rms:.3f} m")


def main():
    rclpy.init()
    node = FlowDepthPoseEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
