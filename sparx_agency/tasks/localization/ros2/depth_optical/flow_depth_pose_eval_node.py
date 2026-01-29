#!/usr/bin/env python3
from __future__ import annotations

import math
import csv
import numpy as np
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

def rotate_vector_3d(v: np.ndarray, q) -> np.ndarray:
    """
    Rotates a 3D vector by a quaternion (x, y, z, w).
    This accounts for full 3D orientation (Roll, Pitch, Yaw).
    """
    # Extract quaternion components
    x, y, z, w = q.x, q.y, q.z, q.w
    
    # Quaternion rotation formula: v' = v + 2 * cross(q_vec, cross(q_vec, v) + w * v)
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    
    return v + 2.0 * (w * uv + uuv)

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
            self.gt_pose_latest = msg
            if self.init_from_gt and not self.have_est:
                self.est_x, self.est_y, self.est_z = msg.position.x, msg.position.y, msg.position.z
                self.have_est = True
                self.get_logger().info("Initialized estimate from Ground Truth.")

    def vel_cb(self, msg: Vector3Stamped):
        if self.gt_pose_latest is None and self.init_from_gt:
            return

        if self.last_vel_stamp is None:
            self.last_vel_stamp = msg.header.stamp
            return

        # Calculate Delta Time (dt)
        t_now = rclpy.time.Time.from_msg(msg.header.stamp)
        t_prev = rclpy.time.Time.from_msg(self.last_vel_stamp)
        dt = (t_now - t_prev).nanoseconds * 1e-9
        self.last_vel_stamp = msg.header.stamp

        if dt <= 0.0 or dt > 1.0:
            return

        # 3D Rotation using TF
        try:
            source_frame = norm_frame(msg.header.frame_id)
            # Lookup transform from drone frame to world (odom) frame
            tf = self.tf_buffer.lookup_transform(self.target_frame, source_frame, msg.header.stamp)
            
            # Rotate the velocity vector from Body Frame to World Frame
            v_body = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
            v_world = rotate_vector_3d(v_body, tf.transform.rotation)
            
            v_x, v_y, v_z = v_world
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Integrate velocity to update position
        if not self.have_est: self.have_est = True
        self.est_x += v_x * dt
        self.est_y += v_y * dt
        self.est_z += v_z * dt

        # Publish Estimated Pose
        est_msg = PoseStamped()
        est_msg.header = msg.header
        est_msg.header.frame_id = self.target_frame
        est_msg.pose.position.x, est_msg.pose.position.y, est_msg.pose.position.z = self.est_x, self.est_y, self.est_z
        est_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(est_msg)

        # Logging and Comparison
        t_sec = t_now.nanoseconds * 1e-9
        if self.est_tum_file:
            self.est_tum_file.write(f"{t_sec:.9f} {self.est_x:.6f} {self.est_y:.6f} {self.est_z:.6f} 0 0 0 1\n")

        if self.gt_pose_latest:
            gx, gy, gz = self.gt_pose_latest.position.x, self.gt_pose_latest.position.y, self.gt_pose_latest.position.z
            err = math.sqrt((self.est_x - gx)**2 + (self.est_y - gy)**2 + (self.est_z - gz)**2)
            self.err_hist.append(err)
            
            if self.csv_writer:
                self.csv_writer.writerow([t_sec, self.est_x, self.est_y, self.est_z, gx, gy, gz, err])
            
            if self.gt_tum_file:
                go = self.gt_pose_latest.orientation
                self.gt_tum_file.write(f"{t_sec:.9f} {gx:.6f} {gy:.6f} {gz:.6f} {go.x} {go.y} {go.z} {go.w}\n")

        # Periodically print stats
        if (self.get_clock().now() - self.last_print_time).nanoseconds * 1e-9 > self.print_every_sec:
            if self.err_hist:
                rms = math.sqrt(sum(e*e for e in self.err_hist)/len(self.err_hist))
                self.get_logger().info(f"Dist Error: {self.err_hist[-1]:.3f}m | RMS: {rms:.3f}m")
            self.last_print_time = self.get_clock().now()

    def destroy_node(self):
        for f in [self.csv_file, self.est_tum_file, self.gt_tum_file]:
            if f: f.close()
        super().destroy_node()

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
