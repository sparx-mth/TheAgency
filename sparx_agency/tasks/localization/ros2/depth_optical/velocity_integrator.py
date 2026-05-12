#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import Vector3Stamped, Pose, PoseStamped


def norm_frame(frame: str) -> str:
    if frame is None:
        return ""
    return frame.lstrip("/")


def rotate_vector_3d(v: np.ndarray, q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    q_vec = np.array([x, y, z], dtype=np.float64)
    v = v.astype(np.float64)
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (w * uv + uuv)


@dataclass
class GTSample:
    t: Time
    pose: Pose


class VelocityIntegratorNode(Node):
    """
    Integrates /flow_depth/velocity into pose.
    Handles TF rotations and optionally initializes the starting pose from GT.
    """
    def __init__(self):
        super().__init__("velocity_integrator_node")

        self.declare_parameter("vel_topic", "/flow_depth/velocity")
        self.declare_parameter("gt_pose_topic", "/simple_drone/gt_pose")
        self.declare_parameter("target_frame", "simple_drone/odom")
        self.declare_parameter("publish_pose_topic", "/flow_depth/pose_est")
        
        self.declare_parameter("init_from_gt", True)
        self.declare_parameter("min_dt", 1e-3)
        self.declare_parameter("max_dt", 2.0)
        self.declare_parameter("gt_max_time_diff", 1.05)
        
        self.declare_parameter("tf_cache_sec", 20.0)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("tf_fallback_to_latest", True)

        vel_topic = self.get_parameter("vel_topic").value
        gt_topic = self.get_parameter("gt_pose_topic").value
        self.target_frame = norm_frame(self.get_parameter("target_frame").value)
        pose_topic = self.get_parameter("publish_pose_topic").value

        self.init_from_gt = bool(self.get_parameter("init_from_gt").value)
        self.min_dt = float(self.get_parameter("min_dt").value)
        self.max_dt = float(self.get_parameter("max_dt").value)
        self.gt_max_time_diff = float(self.get_parameter("gt_max_time_diff").value)

        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.tf_fallback_to_latest = bool(self.get_parameter("tf_fallback_to_latest").value)

        self.get_logger().info(f"[Integrator] vel: {vel_topic}")
        self.get_logger().info(f"[Integrator] target_frame: {self.target_frame}")
        self.get_logger().info(f"[Integrator] publish pose: {pose_topic}")

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=float(self.get_parameter("tf_cache_sec").value)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.have_est = False
        self.est_x, self.est_y, self.est_z = 0.0, 0.0, 0.0
        
        # --- Variables to track total distance ---
        self.start_x, self.start_y, self.start_z = 0.0, 0.0, 0.0
        self.path_length = 0.0 # Accumulated distance step-by-step
        # -----------------------------------------

        self.last_vel_time: Optional[Time] = None

        # GT queue for initialization only
        self.gt_queue: Deque[GTSample] = deque(maxlen=5000)
        self._gt_warn_counter = 0

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Pubs / Subs
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, image_qos)
        self.vel_sub = self.create_subscription(Vector3Stamped, vel_topic, self.vel_cb, image_qos)
        
        if self.init_from_gt:
            self.gt_sub = self.create_subscription(Pose, gt_topic, self.gt_pose_cb, image_qos)

    def gt_pose_cb(self, msg: Pose):
        if not self.have_est:
            self.gt_queue.append(GTSample(t=self.get_clock().now(), pose=msg))

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

    def lookup_tf(self, target: str, source: str, t: Time):
        timeout = Duration(seconds=self.tf_timeout_sec)
        try:
            return self.tf_buffer.lookup_transform(target, source, t, timeout=timeout)
        except TransformException as e:
            if self.tf_fallback_to_latest and "extrapolation into the future" in str(e).lower():
                try:
                    return self.tf_buffer.lookup_transform(target, source, Time(), timeout=timeout)
                except TransformException:
                    raise
            raise

    def vel_cb(self, msg: Vector3Stamped):
        t_vel_stamp = Time.from_msg(msg.header.stamp)
        t_vel_arrival = self.get_clock().now()

        # Init from GT if required
        if not self.have_est and self.init_from_gt:
            gt_pose, gt_dt = self.find_closest_gt(t_vel_arrival)
            if gt_pose is None or gt_dt > self.gt_max_time_diff:
                self._gt_warn_counter += 1
                if self._gt_warn_counter % 50 == 0:
                    self.get_logger().warn(f"[Integrator] Waiting for GT near first vel. closest_dt={gt_dt:.3f}s")
                return
            self.est_x, self.est_y, self.est_z = gt_pose.position.x, gt_pose.position.y, gt_pose.position.z
            
            # --- Save starting position ---
            self.start_x, self.start_y, self.start_z = self.est_x, self.est_y, self.est_z
            # ------------------------------
            
            self.have_est = True
            self.get_logger().info(f"[Integrator] Initialized estimate from GT (dt={gt_dt:.3f}s). Start pos: ({self.start_x:.2f}, {self.start_y:.2f}, {self.start_z:.2f})")

        if self.last_vel_time is None:
            self.last_vel_time = t_vel_stamp
            if not self.init_from_gt:
                self.have_est = True
                # --- If not using GT, save starting position as 0,0,0 ---
                self.start_x, self.start_y, self.start_z = 0.0, 0.0, 0.0
            return

        dt = (t_vel_stamp - self.last_vel_time).nanoseconds * 1e-9
        self.last_vel_time = t_vel_stamp

        if not (self.min_dt <= dt <= self.max_dt):
            return

        # Transform velocity
        try:
            source_frame = norm_frame(msg.header.frame_id)
            if not source_frame:
                return
            tf = self.lookup_tf(self.target_frame, source_frame, t_vel_stamp)
            v_body = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float64)
            v_world = rotate_vector_3d(v_body, tf.transform.rotation)
            v_x, v_y, v_z = float(v_world[0]), float(v_world[1]), float(v_world[2])
        except TransformException as e:
            self.get_logger().warn(f"[Integrator] TF lookup failed: {e}")
            return

        # Integrate
        dx = v_x * dt
        dy = v_y * dt
        dz = v_z * dt
        
        self.est_x += dx
        self.est_y += dy
        self.est_z += dz
        
        # --- Accumulate step-by-step path length ---
        step_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        self.path_length += step_distance
        # -------------------------------------------

        # Publish
        est_msg = PoseStamped()
        est_msg.header.stamp = self.get_clock().now().to_msg()
        est_msg.header.frame_id = self.target_frame
        est_msg.pose.position.x = self.est_x
        est_msg.pose.position.y = self.est_y
        est_msg.pose.position.z = self.est_z
        est_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(est_msg)

    def print_final_stats(self):
        """Prints the total distance traveled when the node shuts down."""
        if not self.have_est:
            print("\n[Integrator] No velocity data was integrated.")
            return

        # Calculate straight-line distance from start to finish
        straight_line_dist = math.sqrt(
            (self.est_x - self.start_x)**2 + 
            (self.est_y - self.start_y)**2 + 
            (self.est_z - self.start_z)**2
        )

        # Using standard Python print() to avoid ROS logging errors during shutdown
        print("\n=========================================")
        print("          INTEGRATION SUMMARY            ")
        print("=========================================")
        print(f"Start Position: ({self.start_x:.3f}, {self.start_y:.3f}, {self.start_z:.3f})")
        print(f"Final Position: ({self.est_x:.3f}, {self.est_y:.3f}, {self.est_z:.3f})")
        print(f"Straight-Line Displacement: {straight_line_dist:.3f} meters")
        print(f"Total Path Length Accumulated: {self.path_length:.3f} meters")
        print("=========================================\n")

def main():
    rclpy.init()
    node = VelocityIntegratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Standard print ignores the ROS shutdown state
        print("\n[Integrator] Shutting down... calculating final distance.")
    finally:
        node.print_final_stats()
        node.destroy_node()
        # Prevent double-shutdown error in newer ROS 2 versions
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()