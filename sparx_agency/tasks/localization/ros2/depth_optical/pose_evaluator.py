#!/usr/bin/env python3
from __future__ import annotations

import math
import csv
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


@dataclass
class GTSample:
    t: Time
    pose: Pose


class PoseEvaluatorNode(Node):
    """
    Compares /flow_depth/pose_est to /simple_drone/gt_pose.
    Outputs evaluation to CSV and TUM formats.
    """
    def __init__(self):
        super().__init__("pose_evaluator_node")

        self.declare_parameter("est_pose_topic", "/flow_depth/pose_est")
        self.declare_parameter("gt_pose_topic", "/simple_drone/gt_pose")
        self.declare_parameter("depth_topic", "/depth_anything/depth")

        self.declare_parameter("gt_queue_size", 5000)
        self.declare_parameter("gt_max_time_diff", 1.00)

        self.declare_parameter("csv_path", "")
        self.declare_parameter("est_tum_path", "")
        self.declare_parameter("gt_tum_path", "")
        self.declare_parameter("depth_comparison_csv", "")
        self.declare_parameter("flush_every_n", 200)
        self.declare_parameter("print_every_sec", 1.0)

        est_topic = self.get_parameter("est_pose_topic").value
        gt_topic = self.get_parameter("gt_pose_topic").value
        depth_topic = self.get_parameter("depth_topic").value

        self.gt_queue_size = int(self.get_parameter("gt_queue_size").value)
        self.gt_max_time_diff = float(self.get_parameter("gt_max_time_diff").value)
        self.print_every_sec = float(self.get_parameter("print_every_sec").value)
        
        self.csv_path = self.get_parameter("csv_path").value
        self.est_tum_path = self.get_parameter("est_tum_path").value
        self.gt_tum_path = self.get_parameter("gt_tum_path").value
        self.depth_comparison_csv = self.get_parameter("depth_comparison_csv").value
        self.flush_every_n = int(self.get_parameter("flush_every_n").value)

        self.get_logger().info(f"[Eval] est_pose: {est_topic}")
        self.get_logger().info(f"[Eval] gt_pose: {gt_topic}")
        self.get_logger().info(f"[Eval] depth: {depth_topic}")

        # State
        self.gt_queue: Deque[GTSample] = deque(maxlen=self.gt_queue_size)
        self.err_hist = deque(maxlen=5000)
        self.last_print_time = self.get_clock().now()
        
        # Latest depth for comparison
        self.latest_depth = None
        self.bridge = CvBridge()

        # Files setup
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t_sec", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z", "err_m", "gt_dt"])

        self.depth_comparison_file = None
        self.depth_comparison_writer = None
        if self.depth_comparison_csv:
            self.depth_comparison_file = open(self.depth_comparison_csv, "w", newline="", encoding="utf-8")
            self.depth_comparison_writer = csv.writer(self.depth_comparison_file)
            self.depth_comparison_writer.writerow(["t_sec", "center_depth", "min_depth", "max_depth", "mean_depth", "std_depth"])

        self.est_tum_file = open(self.est_tum_path, "w", encoding="utf-8") if self.est_tum_path else None
        self.gt_tum_file = open(self.gt_tum_path, "w", encoding="utf-8") if self.gt_tum_path else None
        self._tum_write_count = 0

        # Pubs / Subs
        self.gt_sub = self.create_subscription(Pose, gt_topic, self.gt_pose_cb, qos_profile_sensor_data)
        self.est_sub = self.create_subscription(PoseStamped, est_topic, self.est_pose_cb, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_cb, qos_profile_sensor_data)

    def destroy_node(self):
        for f in [self.csv_file, self.est_tum_file, self.gt_tum_file, self.depth_comparison_file]:
            if f:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
        super().destroy_node()

    def time_to_sec(self, t: Time) -> float:
        return t.nanoseconds * 1e-9

    def gt_pose_cb(self, msg: Pose):
        t = self.get_clock().now()
        self.gt_queue.append(GTSample(t=t, pose=msg))

    def depth_cb(self, msg: Image):
        """Callback to capture depth map for statistics"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth: {e}")

    def find_closest_gt(self, t: Time) -> Tuple[Optional[Pose], float]:
        if not self.gt_queue:
            return None, float("inf")
        best_pose, best_dt = None, float("inf")
        for s in self.gt_queue:
            dt = abs((t - s.t).nanoseconds) * 1e-9
            if dt < best_dt:
                best_dt = dt
                best_pose = s.pose
        return best_pose, best_dt

    def est_pose_cb(self, msg: PoseStamped):
        t_est = Time.from_msg(msg.header.stamp)
        t_sec = self.time_to_sec(t_est)
        
        est_x, est_y, est_z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

        gt_pose, gt_dt = self.find_closest_gt(t_est)
        have_good_gt = (gt_pose is not None and gt_dt <= self.gt_max_time_diff)

        # Write TUM
        if self.est_tum_file:
            self.est_tum_file.write(f"{t_sec:.9f} {est_x:.6f} {est_y:.6f} {est_z:.6f} 0 0 0 1\n")

        if self.gt_tum_file and have_good_gt:
            go = gt_pose.orientation
            self.gt_tum_file.write(
                f"{t_sec:.9f} {gt_pose.position.x:.6f} {gt_pose.position.y:.6f} {gt_pose.position.z:.6f} "
                f"{go.x:.9f} {go.y:.9f} {go.z:.9f} {go.w:.9f}\n"
            )

        self._tum_write_count += 1
        if self.flush_every_n > 0 and (self._tum_write_count % self.flush_every_n) == 0:
            for f in [self.est_tum_file, self.gt_tum_file, self.csv_file, self.depth_comparison_file]:
                if f: f.flush()
        
        # Write depth statistics
        if self.depth_comparison_writer and self.latest_depth is not None:
            valid_depth = self.latest_depth[np.isfinite(self.latest_depth) & (self.latest_depth > 0)]
            h, w = self.latest_depth.shape
            center_depth = self.latest_depth[h//2, w//2] if np.isfinite(self.latest_depth[h//2, w//2]) else -1.0
            if len(valid_depth) > 0:
                self.depth_comparison_writer.writerow([
                    t_sec,
                    center_depth,
                    np.min(valid_depth),
                    np.max(valid_depth),
                    np.mean(valid_depth),
                    np.std(valid_depth)
                ])

        # CSV and Error Calculation
        if have_good_gt:
            gx, gy, gz = gt_pose.position.x, gt_pose.position.y, gt_pose.position.z
            err = math.sqrt((est_x - gx) ** 2 + (est_y - gy) ** 2 + (est_z - gz) ** 2)
            self.err_hist.append(err)

            if self.csv_writer:
                self.csv_writer.writerow([t_sec, est_x, est_y, est_z, gx, gy, gz, err, gt_dt])
        else:
            if self.csv_writer:
                self.csv_writer.writerow([t_sec, est_x, est_y, est_z, "", "", "", "", gt_dt])

        # Print stats
        if (self.get_clock().now() - self.last_print_time).nanoseconds * 1e-9 > self.print_every_sec:
            if self.err_hist:
                rms = math.sqrt(sum(e * e for e in self.err_hist) / len(self.err_hist))
                self.get_logger().info(f"[Eval] last_err={self.err_hist[-1]:.3f}m | RMS={rms:.3f}m | gt_dt={gt_dt:.3f}s")
            else:
                self.get_logger().info(f"[Eval] running... (no valid GT yet) gt_dt={gt_dt:.3f}s")
            self.last_print_time = self.get_clock().now()


def main():
    rclpy.init()
    node = PoseEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()