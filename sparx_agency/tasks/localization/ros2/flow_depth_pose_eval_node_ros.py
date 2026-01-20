#!/usr/bin/env python3
from __future__ import annotations

import math
import csv
from collections import deque
from typing import Optional, Tuple
from rclpy.time import Time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Vector3Stamped, Pose, PoseStamped
import tf2_ros
from tf2_ros import TransformException

def norm_frame(frame: str) -> str:
    # TF2 in ROS2: frame_ids must NOT start with '/'
    if frame is None:
        return ""
    return frame.lstrip("/")

def quat_to_yaw(qx, qy, qz, qw) -> float:
    # yaw from quaternion (Z axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate_xy(vx, vy, yaw) -> Tuple[float, float]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (cy * vx - sy * vy, sy * vx + cy * vy)


class FlowDepthPoseEvalNode(Node):
    """
    Integrates /flow_depth/velocity into a pose estimate and compares to /simple_drone/gt_pose.

    Key idea:
      - velocity arrives in camera frame (front_cam_link)
      - convert velocity to target_frame (default: odom) using TF at the same timestamp
      - integrate to position
      - compare to GT pose (last received Pose)
    """

    def __init__(self):
        super().__init__("flow_depth_pose_eval_node")

        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        # params
        self.declare_parameter("vel_topic", "/flow_depth/velocity")
        self.declare_parameter("gt_pose_topic", "/simple_drone/gt_pose")
        self.declare_parameter("target_frame", "/simple_drone/odom")  # compare/integrate in this frame
        self.declare_parameter("publish_pose_topic", "/flow_depth/pose_est")
        self.declare_parameter("init_from_gt", True)     # start est at first GT pose
        self.declare_parameter("csv_path", "")           # e.g. /tmp/pose_eval.csv (optional)
        self.declare_parameter("print_every_sec", 1.0)

        vel_topic = self.get_parameter("vel_topic").get_parameter_value().string_value
        gt_topic = self.get_parameter("gt_pose_topic").get_parameter_value().string_value
        self.target_frame = norm_frame(self.get_parameter("target_frame").get_parameter_value().string_value)
        pose_topic = self.get_parameter("publish_pose_topic").get_parameter_value().string_value
        self.init_from_gt = self.get_parameter("init_from_gt").get_parameter_value().bool_value
        self.csv_path = self.get_parameter("csv_path").get_parameter_value().string_value
        self.print_every_sec = float(self.get_parameter("print_every_sec").get_parameter_value().double_value)

        self.get_logger().info(f"[PoseEval] vel: {vel_topic}")
        self.get_logger().info(f"[PoseEval] gt_pose: {gt_topic} (Pose without header)")
        self.get_logger().info(f"[PoseEval] target_frame: {self.target_frame}")
        self.get_logger().info(f"[PoseEval] publish pose: {pose_topic}")

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # state
        self.have_est = False
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_z = 0.0
        self.last_vel_stamp = None  # builtin_interfaces/Time

        self.gt_pose_latest: Optional[Pose] = None
        self.gt_pose_time_latest = None  # node clock time when received (since Pose has no stamp)

        # error stats
        self.err_hist = deque(maxlen=5000)  # store recent errors
        self.last_print_time = self.get_clock().now()

        # CSV
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t_sec", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z", "err_m"])

        # pubs/subs
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)

        self.gt_sub = self.create_subscription(Pose, gt_topic, self.gt_pose_cb, qos_profile_sensor_data)
        self.vel_sub = self.create_subscription(Vector3Stamped, vel_topic, self.vel_cb, qos_profile_sensor_data)
    

    
    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()

    # -------- callbacks --------
    def gt_pose_cb(self, msg: Pose):
        self.gt_pose_latest = msg
        self.gt_pose_time_latest = self.get_clock().now()

        # initialize estimate to GT at first message (optional)
        if self.init_from_gt and (not self.have_est):
            self.est_x = float(msg.position.x)
            self.est_y = float(msg.position.y)
            self.est_z = float(msg.position.z)
            self.have_est = True
            self.get_logger().info("[PoseEval] Initialized estimated pose from first GT pose.")

    def vel_cb(self, msg: Vector3Stamped):
        # Need GT for comparison (and possibly init)
        if self.gt_pose_latest is None and self.init_from_gt:
            return

        # dt from velocity stamps
        if self.last_vel_stamp is None:
            self.last_vel_stamp = msg.header.stamp
            return

        dt_ns = (msg.header.stamp.sec - self.last_vel_stamp.sec) * 1_000_000_000 + \
                (msg.header.stamp.nanosec - self.last_vel_stamp.nanosec)
        dt = float(dt_ns) * 1e-9
        self.last_vel_stamp = msg.header.stamp

        if dt <= 0.0 or dt > 1.0:
            # skip crazy dt (bag pauses / jumps)
            return

        # Make sure we have an estimate to integrate
        if not self.have_est:
            # if not init_from_gt, start at origin
            self.have_est = True

        # Transform velocity to target_frame using TF rotation at that time
        v_cam_x = float(msg.vector.x)
        v_cam_y = float(msg.vector.y)
        v_cam_z = float(msg.vector.z)

        try:
            source_frame = norm_frame(msg.header.frame_id)
            target_frame = self.target_frame

            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time()
            )
            # Extract yaw from rotation and rotate XY (we only use planar here)
            q = tf.transform.rotation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            v_x, v_y = rotate_xy(v_cam_x, v_cam_y, yaw)
            # z handling: keep as-is (often 0 anyway)
            v_z = v_cam_z

        except TransformException as e:
            # If TF missing, cannot compare/integrate correctly
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
        # orientation unknown here (we’re only integrating translation)
        out.pose.orientation.w = 1.0
        self.pose_pub.publish(out)

        # Compare to GT (last received)
        if self.gt_pose_latest is not None:
            gx = float(self.gt_pose_latest.position.x)
            gy = float(self.gt_pose_latest.position.y)
            gz = float(self.gt_pose_latest.position.z)
            err = math.sqrt((self.est_x - gx) ** 2 + (self.est_y - gy) ** 2 + (self.est_z - gz) ** 2)
            self.err_hist.append(err)

            # CSV
            if self.csv_writer is not None:
                t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                self.csv_writer.writerow([t_sec, self.est_x, self.est_y, self.est_z, gx, gy, gz, err])

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
