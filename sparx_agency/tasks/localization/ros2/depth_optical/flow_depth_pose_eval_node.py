#!/usr/bin/env python3
from __future__ import annotations

import math
import csv
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import Vector3Stamped, Pose, PoseStamped


def norm_frame(frame: str) -> str:
    # TF2 in ROS2: frame_ids must NOT start with '/'
    if frame is None:
        return ""
    return frame.lstrip("/")


def rotate_vector_3d(v: np.ndarray, q) -> np.ndarray:
    """
    Rotate 3D vector by quaternion (x,y,z,w).
    """
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


class FlowDepthPoseEvalNode(Node):
    """
    Integrates /flow_depth/velocity into pose and compares to /simple_drone/gt_pose.

    Key fixes vs your version:
      1) GT Pose has no header -> we timestamp it on receipt (sim time) and keep a small queue.
         For each velocity sample, we pick the closest GT by time (within gt_max_time_diff).
      2) Start from GT so est/gt begin at same location (init_from_gt).
      3) TF extrapolation: try exact time, if "future" -> fallback to latest TF (Time()).
      4) Remove duplicate destroy_node and add safe file flushing.

    Outputs:
      - PoseStamped estimate on publish_pose_topic
      - CSV merged rows (optional)
      - TUM est/gt (optional) for evo
    """

    def __init__(self):
        super().__init__("flow_depth_pose_eval_node")

        # ---- params ----
        #self.declare_parameter("use_sim_time", True)

        self.declare_parameter("vel_topic", "/flow_depth/velocity")
        self.declare_parameter("gt_pose_topic", "/simple_drone/gt_pose")

        # compare/integrate in this frame
        self.declare_parameter("target_frame", "simple_drone/odom")

        self.declare_parameter("publish_pose_topic", "/flow_depth/pose_est")

        # init/alignment
        self.declare_parameter("init_from_gt", True)         # start est at first GT pose (closest to first vel)
        self.declare_parameter("write_initial_tum", True)    # write a first line immediately when initialized

        # dt limits
        self.declare_parameter("min_dt", 1e-3)
        self.declare_parameter("max_dt", 2.0)

        # GT association limits
        self.declare_parameter("gt_queue_size", 5000)
        self.declare_parameter("gt_max_time_diff", 1.00)     # seconds (match gt to vel)
        self.declare_parameter("gt_print_warn_every_n", 50)

        # output
        self.declare_parameter("csv_path", "")
        self.declare_parameter("print_every_sec", 1.0)

        # TUM outputs for evo
        self.declare_parameter("est_tum_path", "")
        self.declare_parameter("gt_tum_path", "")
        self.declare_parameter("flush_every_n", 200)

        # TF behavior
        self.declare_parameter("tf_cache_sec", 20.0)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("tf_fallback_to_latest", True)

        # ---- read params ----
        vel_topic = self.get_parameter("vel_topic").value
        gt_topic = self.get_parameter("gt_pose_topic").value
        self.target_frame = norm_frame(self.get_parameter("target_frame").value)
        pose_topic = self.get_parameter("publish_pose_topic").value

        self.init_from_gt = bool(self.get_parameter("init_from_gt").value)
        self.write_initial_tum = bool(self.get_parameter("write_initial_tum").value)

        self.min_dt = float(self.get_parameter("min_dt").value)
        self.max_dt = float(self.get_parameter("max_dt").value)

        self.gt_queue_size = int(self.get_parameter("gt_queue_size").value)
        self.gt_max_time_diff = float(self.get_parameter("gt_max_time_diff").value)
        self.gt_print_warn_every_n = int(self.get_parameter("gt_print_warn_every_n").value)

        self.csv_path = self.get_parameter("csv_path").value
        self.print_every_sec = float(self.get_parameter("print_every_sec").value)

        self.est_tum_path = self.get_parameter("est_tum_path").value
        self.gt_tum_path = self.get_parameter("gt_tum_path").value
        self.flush_every_n = int(self.get_parameter("flush_every_n").value)

        self.tf_cache_sec = float(self.get_parameter("tf_cache_sec").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.tf_fallback_to_latest = bool(self.get_parameter("tf_fallback_to_latest").value)

        self.get_logger().info(f"[PoseEval] vel: {vel_topic}")
        self.get_logger().info(f"[PoseEval] gt_pose: {gt_topic} (Pose no header -> timestamp on receipt)")
        self.get_logger().info(f"[PoseEval] target_frame: {self.target_frame}")
        self.get_logger().info(f"[PoseEval] publish pose: {pose_topic}")
        self.get_logger().info(f"[PoseEval] init_from_gt={self.init_from_gt} gt_max_time_diff={self.gt_max_time_diff}s")
        if self.est_tum_path:
            self.get_logger().info(f"[PoseEval] est_tum_path: {self.est_tum_path}")
        if self.gt_tum_path:
            self.get_logger().info(f"[PoseEval] gt_tum_path: {self.gt_tum_path}")

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=self.tf_cache_sec))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- state ----
        self.have_est = False
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_z = 0.0

        self.last_vel_time: Optional[Time] = None  # rclpy Time

        # GT queue (time-stamped on receipt)
        self.gt_queue: Deque[GTSample] = deque(maxlen=self.gt_queue_size)
        self._gt_warn_counter = 0

        # error stats
        self.err_hist = deque(maxlen=5000)
        self.last_print_time = self.get_clock().now()

        # ---- CSV ----
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t_sec", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z", "err_m", "gt_dt"])

        # ---- TUM ----
        self.est_tum_file = open(self.est_tum_path, "w", encoding="utf-8") if self.est_tum_path else None
        self.gt_tum_file = open(self.gt_tum_path, "w", encoding="utf-8") if self.gt_tum_path else None
        self._tum_write_count = 0

        # ---- pubs/subs ----
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.gt_sub = self.create_subscription(Pose, gt_topic, self.gt_pose_cb, qos_profile_sensor_data)
        self.vel_sub = self.create_subscription(Vector3Stamped, vel_topic, self.vel_cb, qos_profile_sensor_data)

    # ---------------- lifecycle ----------------
    def destroy_node(self):
        for f in [self.csv_file, self.est_tum_file, self.gt_tum_file]:
            if f:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
        super().destroy_node()

    # ---------------- helpers ----------------
    def now_time(self) -> Time:
        # rclpy Time (uses sim time if enabled)
        return self.get_clock().now()

    def time_to_sec(self, t: Time) -> float:
        return t.nanoseconds * 1e-9

    def find_closest_gt(self, t: Time) -> Tuple[Optional[Pose], float]:
        """
        Returns (closest_gt_pose, dt_sec). dt_sec is abs time difference.
        """
        if not self.gt_queue:
            return None, float("inf")

        best_pose = None
        best_dt = float("inf")
        for s in self.gt_queue:
            dt = abs((t - s.t).nanoseconds) * 1e-9
            if dt < best_dt:
                best_dt = dt
                best_pose = s.pose
        return best_pose, best_dt

    def maybe_init_from_gt(self, t_vel: Time):
        if self.have_est or not self.init_from_gt:
            return

        gt_pose, gt_dt = self.find_closest_gt(t_vel)
        if gt_pose is None or gt_dt > self.gt_max_time_diff:
            self._gt_warn_counter += 1
            if (self._gt_warn_counter % self.gt_print_warn_every_n) == 0:
                self.get_logger().warn(
                    f"[PoseEval] Waiting for GT near first vel. "
                    f"gt_queue={len(self.gt_queue)} closest_dt={gt_dt:.3f}s (need <= {self.gt_max_time_diff:.3f}s)"
                )
            return

        self.est_x, self.est_y, self.est_z = gt_pose.position.x, gt_pose.position.y, gt_pose.position.z
        self.have_est = True

        # Write initial TUM line so both trajectories start same place (important for evo visuals)
        if self.write_initial_tum:
            t_sec = self.time_to_sec(t_vel)
            if self.est_tum_file:
                self.est_tum_file.write(
                    f"{t_sec:.9f} {self.est_x:.6f} {self.est_y:.6f} {self.est_z:.6f} 0 0 0 1\n"
                )
            if self.gt_tum_file:
                go = gt_pose.orientation
                self.gt_tum_file.write(
                    f"{t_sec:.9f} {gt_pose.position.x:.6f} {gt_pose.position.y:.6f} {gt_pose.position.z:.6f} "
                    f"{go.x:.9f} {go.y:.9f} {go.z:.9f} {go.w:.9f}\n"
                )
            self._tum_write_count += 1

        self.get_logger().info(f"[PoseEval] Initialized estimate from GT (dt={gt_dt:.3f}s).")

    def lookup_tf(self, target: str, source: str, t: Time):
        """
        Try TF at time t. If extrapolation future and enabled, fallback to latest.
        """
        timeout = Duration(seconds=self.tf_timeout_sec)
        try:
            return self.tf_buffer.lookup_transform(target, source, t, timeout=timeout)
        except TransformException as e:
            msg = str(e)
            if self.tf_fallback_to_latest and "extrapolation into the future" in msg.lower():
                # fallback to latest available TF
                try:
                    return self.tf_buffer.lookup_transform(target, source, Time(), timeout=timeout)
                except TransformException:
                    raise
            raise

    # ---------------- callbacks ----------------
    def gt_pose_cb(self, msg: Pose):
        # Timestamp GT on receipt time (sim time). This is the best we can do without header.
        t = self.now_time()
        self.gt_queue.append(GTSample(t=t, pose=msg))

    def vel_cb(self, msg: Vector3Stamped):

        t_vel = Time.from_msg(msg.header.stamp)  # velocity sample time

        # init if needed
        self.maybe_init_from_gt(t_vel)
        if self.init_from_gt and not self.have_est:
            return

        # dt
        if self.last_vel_time is None:
            self.last_vel_time = t_vel
            return

        dt = (t_vel - self.last_vel_time).nanoseconds * 1e-9
        self.last_vel_time = t_vel

        if not (self.min_dt <= dt <= self.max_dt):
            return

        # rotate velocity into target frame
        try:
            source_frame = norm_frame(msg.header.frame_id)
            if not source_frame:
                return

            tf = self.lookup_tf(self.target_frame, source_frame, t_vel)
            v_body = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float64)
            v_world = rotate_vector_3d(v_body, tf.transform.rotation)
            v_x, v_y, v_z = float(v_world[0]), float(v_world[1]), float(v_world[2])
        except TransformException as e:
            self.get_logger().warn(f"[PoseEval] TF lookup failed: {e}")
            return

        # integrate
        if not self.have_est:
            self.have_est = True
        self.est_x += v_x * dt
        self.est_y += v_y * dt
        self.est_z += v_z * dt

        # publish
        est_msg = PoseStamped()
        est_msg.header.stamp = msg.header.stamp
        est_msg.header.frame_id = self.target_frame
        est_msg.pose.position.x = self.est_x
        est_msg.pose.position.y = self.est_y
        est_msg.pose.position.z = self.est_z
        est_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(est_msg)

        # find closest GT for this vel time
        gt_pose, gt_dt = self.find_closest_gt(t_vel)
        have_good_gt = (gt_pose is not None and gt_dt <= self.gt_max_time_diff)

        # write TUM
        t_sec = self.time_to_sec(t_vel)
        if self.est_tum_file:
            self.est_tum_file.write(f"{t_sec:.9f} {self.est_x:.6f} {self.est_y:.6f} {self.est_z:.6f} 0 0 0 1\n")

        if self.gt_tum_file and have_good_gt:
            go = gt_pose.orientation
            self.gt_tum_file.write(
                f"{t_sec:.9f} {gt_pose.position.x:.6f} {gt_pose.position.y:.6f} {gt_pose.position.z:.6f} "
                f"{go.x:.9f} {go.y:.9f} {go.z:.9f} {go.w:.9f}\n"
            )

        self._tum_write_count += 1
        if self.flush_every_n > 0 and (self._tum_write_count % self.flush_every_n) == 0:
            if self.est_tum_file:
                self.est_tum_file.flush()
            if self.gt_tum_file:
                self.gt_tum_file.flush()
            if self.csv_file:
                self.csv_file.flush()

        # error + CSV
        if have_good_gt:
            gx, gy, gz = gt_pose.position.x, gt_pose.position.y, gt_pose.position.z
            err = math.sqrt((self.est_x - gx) ** 2 + (self.est_y - gy) ** 2 + (self.est_z - gz) ** 2)
            self.err_hist.append(err)

            if self.csv_writer:
                self.csv_writer.writerow([t_sec, self.est_x, self.est_y, self.est_z, gx, gy, gz, err, gt_dt])
        else:
            # still optionally write CSV with empty GT
            if self.csv_writer:
                self.csv_writer.writerow([t_sec, self.est_x, self.est_y, self.est_z, "", "", "", "", gt_dt])

        # periodic stats
        if (self.get_clock().now() - self.last_print_time).nanoseconds * 1e-9 > self.print_every_sec:
            if self.err_hist:
                rms = math.sqrt(sum(e * e for e in self.err_hist) / len(self.err_hist))
                self.get_logger().info(
                    f"[PoseEval] last_err={self.err_hist[-1]:.3f}m | RMS={rms:.3f}m | gt_dt={gt_dt:.3f}s | dt={dt:.3f}s"
                )
            else:
                self.get_logger().info(f"[PoseEval] running... (no valid GT yet) gt_dt={gt_dt:.3f}s")
            self.last_print_time = self.get_clock().now()


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
