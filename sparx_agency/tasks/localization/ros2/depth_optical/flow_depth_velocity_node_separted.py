#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
import json
import math
import os
import yaml

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped, Twist, PoseStamped
from cv_bridge import CvBridge

from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker


class FlowDepthVelocitySeparatedNode(Node):
    """
    SPARX ROS2 node - separated RGB and Depth version.

    Main idea:
      - RGB callback runs at RGB rate.
      - Depth callback runs at Depth rate.
      - Depth maps are stored in a small queue.
      - For each RGB frame, we choose the closest Depth map by timestamp.
      - If the closest Depth is too old, we skip this RGB frame.

    This is useful when RGB is fast and Depth is slower.
    """

    def __init__(self):
        super().__init__("flow_depth_velocity_node_separated")

        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time",
                rclpy.parameter.Parameter.Type.BOOL,
                True
            )
        ])

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")

        self.declare_parameter(
            "camera_config_yaml",
            "/home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml"
        )

        self.declare_parameter("output_topic", "/flow_depth/velocity")
        self.declare_parameter("pose_est_topic", "/flow_depth/pose_est")

        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 70)

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("depth_scale", 1.0)

        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "xtend_camera")

        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        self.declare_parameter("depth_ema_alpha", 0.15)

        # Important for separated RGB/Depth mode:
        # If RGB uses a depth map older than this threshold, skip the RGB frame.
        self.declare_parameter("max_depth_age_sec", 0.25)

        # Number of recent depth maps to keep.
        self.declare_parameter("depth_queue_size", 20)

        self.declare_parameter(
            "json_out_path",
            "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/estimated_trajectory.json"
        )

        # -------------------------
        # Read parameters
        # -------------------------
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.pose_est_topic = self.get_parameter("pose_est_topic").get_parameter_value().string_value

        self.camera_config_yaml = self.get_parameter("camera_config_yaml").get_parameter_value().string_value

        self.max_corners = int(self.get_parameter("max_corners").get_parameter_value().integer_value)
        self.min_corners = int(self.get_parameter("min_corners").get_parameter_value().integer_value)

        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)
        self.depth_scale = float(self.get_parameter("depth_scale").get_parameter_value().double_value)

        self.show_debug = bool(self.get_parameter("show_debug").get_parameter_value().bool_value)
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        lk_win = int(self.get_parameter("lk_win").get_parameter_value().integer_value)
        lk_levels = int(self.get_parameter("lk_levels").get_parameter_value().integer_value)

        self.ema_alpha = float(self.get_parameter("depth_ema_alpha").get_parameter_value().double_value)
        self.max_depth_age_sec = float(self.get_parameter("max_depth_age_sec").get_parameter_value().double_value)
        depth_queue_size = int(self.get_parameter("depth_queue_size").get_parameter_value().integer_value)

        self.json_out_path = self.get_parameter("json_out_path").get_parameter_value().string_value

        # -------------------------
        # State
        # -------------------------
        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.last_smoothed_depth = None
        self.depth_queue = deque(maxlen=depth_queue_size)

        self.center_depth = 0.0

        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_vz = 0.0
        self.vel_alpha = 0.2

        self.latest_gt_vx = 0.0
        self.latest_gt_vy = 0.0
        self.latest_gt_vz = 0.0

        self.debug_path = [(0.0, 0.0)]
        self.current_distance = 0.0
        self.trajectory_history = []

        self.rgb_frames_count = 0
        self.depth_frames_count = 0
        self.skipped_no_depth = 0
        self.skipped_old_depth = 0

        # -------------------------
        # Load camera intrinsics
        # -------------------------
        self.load_camera_intrinsics_from_yaml(self.camera_config_yaml)

        # -------------------------
        # Optical Flow tracker
        # -------------------------
        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

        # -------------------------
        # Publishers / Subscribers
        # -------------------------
        self.vel_pub = self.create_publisher(Vector3Stamped, self.output_topic, 10)

        self.rgb_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )

        self.create_subscription(
            Twist,
            "/simple_drone/gt_vel",
            self.gt_vel_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_est_topic,
            self.pose_est_callback,
            10
        )

        self.get_logger().info("==============================================")
        self.get_logger().info("[FlowDepthSeparated] Started")
        self.get_logger().info(f"[FlowDepthSeparated] RGB topic: {self.image_topic}")
        self.get_logger().info(f"[FlowDepthSeparated] Depth topic: {self.depth_topic}")
        self.get_logger().info(f"[FlowDepthSeparated] Output topic: {self.output_topic}")
        self.get_logger().info(f"[FlowDepthSeparated] max_depth_age_sec: {self.max_depth_age_sec:.3f}")
        self.get_logger().info(f"[FlowDepthSeparated] depth_queue_size: {depth_queue_size}")
        self.get_logger().info("==============================================")

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    # ---------------------------------------------------------
    # Camera intrinsics
    # ---------------------------------------------------------
    def load_camera_intrinsics_from_yaml(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Camera YAML not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        if "projection_matrix" in cfg and "data" in cfg["projection_matrix"]:
            P = cfg["projection_matrix"]["data"]

            if len(P) < 12 or P[0] == 0.0:
                raise ValueError(f"Invalid projection_matrix in YAML: {yaml_path}")

            self.fx = float(P[0])
            self.fy = float(P[5])
            self.cx = float(P[2])
            self.cy = float(P[6])

            self.get_logger().info(
                f"[Camera YAML] Loaded intrinsics from projection_matrix: "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            return

        required = ["fx", "fy", "cx", "cy"]
        if all(k in cfg for k in required):
            self.fx = float(cfg["fx"])
            self.fy = float(cfg["fy"])
            self.cx = float(cfg["cx"])
            self.cy = float(cfg["cy"])

            self.get_logger().info(
                f"[Camera YAML] Loaded raw intrinsics: "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            return

        raise ValueError(
            f"Could not find projection_matrix or fx/fy/cx/cy in YAML: {yaml_path}"
        )

    # ---------------------------------------------------------
    # Depth handling
    # ---------------------------------------------------------
    def smooth_depth_map(self, raw_depth_map: np.ndarray) -> np.ndarray:
        """
        EMA smoothing over depth frames.
        This runs only when a new depth frame arrives.
        """
        raw_depth_map = raw_depth_map.astype(np.float32)

        if self.last_smoothed_depth is None:
            self.last_smoothed_depth = raw_depth_map
        else:
            if self.last_smoothed_depth.shape != raw_depth_map.shape:
                self.get_logger().warn(
                    "[Depth] Shape changed, resetting depth EMA buffer"
                )
                self.last_smoothed_depth = raw_depth_map
            else:
                self.last_smoothed_depth = (
                    self.ema_alpha * raw_depth_map
                    + (1.0 - self.ema_alpha) * self.last_smoothed_depth
                )

        return self.last_smoothed_depth.astype(np.float32)

    def depth_callback(self, depth_msg: Image):
        """
        Runs at Depth rate.
        Converts depth image, smooths it, and stores it in a queue.
        """
        self.depth_frames_count += 1

        try:
            depth_cv = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="32FC1"
            )
        except Exception as e:
            self.get_logger().error(f"[Depth] Convert failed: {e}")
            return

        depth_raw = np.asarray(depth_cv, dtype=np.float32)
        depth_map = self.smooth_depth_map(depth_raw)

        depth_t = self.stamp_to_sec(depth_msg.header.stamp)

        self.depth_queue.append({
            "stamp": depth_msg.header.stamp,
            "time": depth_t,
            "depth": depth_map.copy(),
        })

        self.update_center_depth(depth_map)

        if self.depth_frames_count % 30 == 0:
            self.get_logger().info(
                f"[Depth] received={self.depth_frames_count}, "
                f"queue={len(self.depth_queue)}, "
                f"center_depth={self.center_depth:.3f} m"
            )

    def update_center_depth(self, depth_map: np.ndarray):
        if self.cx is None or self.cy is None:
            return

        H, W = depth_map.shape[:2]

        u_idx = int(round(self.cx))
        v_idx = int(round(self.cy))

        if not (0 <= u_idx < W and 0 <= v_idx < H):
            return

        half_side = 2

        y0 = max(0, v_idx - half_side)
        y1 = min(H, v_idx + half_side + 1)
        x0 = max(0, u_idx - half_side)
        x1 = min(W, u_idx + half_side + 1)

        center_region = depth_map[y0:y1, x0:x1]
        valid_center_depths = center_region[
            np.isfinite(center_region) & (center_region > 0)
        ]

        if valid_center_depths.size > 0:
            self.center_depth = float(np.mean(valid_center_depths) * self.depth_scale)

    def get_closest_depth(self, rgb_stamp):
        """
        Finds the closest depth map in the queue for a given RGB timestamp.
        Returns:
            depth_map, depth_age_sec
        """
        if len(self.depth_queue) == 0:
            return None, None

        rgb_t = self.stamp_to_sec(rgb_stamp)

        best_item = min(
            self.depth_queue,
            key=lambda item: abs(item["time"] - rgb_t)
        )

        depth_age_sec = abs(best_item["time"] - rgb_t)

        if depth_age_sec > self.max_depth_age_sec:
            return None, depth_age_sec

        return best_item["depth"], depth_age_sec

    # ---------------------------------------------------------
    # RGB handling
    # ---------------------------------------------------------
    def rgb_callback(self, rgb_msg: Image):
        """
        Runs at RGB rate.
        Does NOT wait for a new depth frame.
        Uses the closest available depth frame from the depth queue.
        """
        self.rgb_frames_count += 1

        if self.fx is None or self.fy is None:
            return

        depth_map, depth_age_sec = self.get_closest_depth(rgb_msg.header.stamp)

        if depth_map is None:
            if depth_age_sec is None:
                self.skipped_no_depth += 1
                if self.skipped_no_depth % 30 == 1:
                    self.get_logger().warn(
                        "[RGB] No depth received yet, skipping RGB frame"
                    )
            else:
                self.skipped_old_depth += 1
                if self.skipped_old_depth % 30 == 1:
                    self.get_logger().warn(
                        f"[RGB] Closest depth is too old: "
                        f"{depth_age_sec * 1000.0:.1f} ms > "
                        f"{self.max_depth_age_sec * 1000.0:.1f} ms. "
                        f"Skipping RGB frame."
                    )
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"[RGB] Convert failed: {e}")
            return

        # Optical flow is calculated on RGB frames.
        flow_res = self.tracker.process(frame, rgb_msg.header.stamp)
        if flow_res is None:
            return

        if flow_res.dt <= 0.0:
            self.get_logger().warn(f"[RGB] Invalid optical flow dt={flow_res.dt}")
            return

        vx_mps, vy_mps, vz_mps, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old,
            flow_res.good_new,
            depth_map,
            flow_res.dt
        )

        # Low-pass filtering for velocity stability.
        vx_mps = self.vel_alpha * vx_mps + (1.0 - self.vel_alpha) * self.prev_vx
        vy_mps = self.vel_alpha * vy_mps + (1.0 - self.vel_alpha) * self.prev_vy
        vz_mps = self.vel_alpha * vz_mps + (1.0 - self.vel_alpha) * self.prev_vz

        self.prev_vx = vx_mps
        self.prev_vy = vy_mps
        self.prev_vz = vz_mps

        # Dead-band to remove tiny noise.
        if abs(vx_mps) < 0.02:
            vx_mps = 0.0
        if abs(vy_mps) < 0.2:
            vy_mps = 0.0
        if abs(vz_mps) < 0.2:
            vz_mps = 0.0

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = rgb_msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(vx_mps)
        vel_msg.vector.y = float(vy_mps)
        vel_msg.vector.z = float(vz_mps)

        self.vel_pub.publish(vel_msg)

        if self.rgb_frames_count % 60 == 0:
            self.get_logger().info(
                f"[RGB] frames={self.rgb_frames_count}, "
                f"depth_age={depth_age_sec * 1000.0:.1f} ms, "
                f"vx={vx_mps:.3f}, vy={vy_mps:.3f}, vz={vz_mps:.3f}, "
                f"used={n_used}"
            )

        if self.show_debug:
            self.draw_visual_debug(
                frame=frame,
                good_old=flow_res.good_old,
                good_new=flow_res.good_new,
                depth_map=depth_map,
                vx_mps=vx_mps,
                vy_mps=vy_mps,
                vz_mps=vz_mps,
                n_used=n_used,
                depth_age_sec=depth_age_sec,
            )

    # ---------------------------------------------------------
    # Velocity calculation
    # ---------------------------------------------------------
    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float):
        """
        Returns velocity in m/s.

        Internal LS model estimates camera-frame velocity.
        Final returned convention follows your existing code:
            return Vz_o, -Vx_o, -Vy_o
        """
        H, W = depth_map.shape[:2]

        du = (good_new[:, 0] - good_old[:, 0]) / dt
        dv = (good_new[:, 1] - good_old[:, 1]) / dt

        u_int = np.rint(good_new[:, 0]).astype(np.int32)
        v_int = np.rint(good_new[:, 1]).astype(np.int32)

        valid = (
            (u_int >= 0) & (u_int < W) &
            (v_int >= 0) & (v_int < H)
        )

        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0

        Z = np.zeros_like(du, dtype=np.float32)
        Z[valid] = depth_map[v_int[valid], u_int[valid]] * self.depth_scale

        valid = (
            valid &
            np.isfinite(Z) &
            (Z > self.min_depth) &
            (Z < self.max_depth)
        )

        nv = int(np.sum(valid))
        if nv < 8:
            return 0.0, 0.0, 0.0, 0

        u_c_v = good_old[valid, 0].astype(np.float64) - self.cx
        v_c_v = good_old[valid, 1].astype(np.float64) - self.cy

        Zv = Z[valid].astype(np.float64)
        du_v = du[valid].astype(np.float64)
        dv_v = dv[valid].astype(np.float64)

        A = np.zeros((2 * nv, 3), dtype=np.float64)
        B = np.zeros((2 * nv,), dtype=np.float64)

        A[0::2, 0] = -self.fx
        A[0::2, 2] = u_c_v
        B[0::2] = du_v * Zv

        A[1::2, 1] = -self.fy
        A[1::2, 2] = v_c_v
        B[1::2] = dv_v * Zv

        # Weight points closer to image center.
        dist_sq = u_c_v**2 + v_c_v**2
        denom = 0.05 * (self.cx**2 + self.cy**2)
        weights = np.exp(-dist_sq / denom)

        W_vec = np.zeros((2 * nv,), dtype=np.float64)
        W_vec[0::2] = weights
        W_vec[1::2] = weights

        sqrt_w = np.sqrt(W_vec)

        try:
            vel, *_ = np.linalg.lstsq(
                A * sqrt_w[:, np.newaxis],
                B * sqrt_w,
                rcond=None
            )
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, nv

        Vx_o = float(vel[0])
        Vy_o = float(vel[1])
        Vz_o = float(vel[2])

        return Vz_o, -Vx_o, -Vy_o, nv

    # ---------------------------------------------------------
    # Pose / GT callbacks
    # ---------------------------------------------------------
    def gt_vel_callback(self, msg: Twist):
        self.latest_gt_vx = msg.linear.x
        self.latest_gt_vy = msg.linear.y
        self.latest_gt_vz = msg.linear.z

    def pose_est_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        self.debug_path.append((x, y))
        self.current_distance = math.sqrt(x**2 + y**2 + z**2)

        self.trajectory_history.append({
            "image": f"frame_{len(self.trajectory_history):06d}.jpg",
            "pose": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "yaw": 0.0,
            }
        })

    # ---------------------------------------------------------
    # Debug drawing
    # ---------------------------------------------------------
    def draw_visual_debug(
        self,
        frame,
        good_old,
        good_new,
        depth_map,
        vx_mps,
        vy_mps,
        vz_mps,
        n_used,
        depth_age_sec,
    ):
        vis = frame.copy()

        self.draw_debug(vis, good_old, good_new, depth_map)
        self.draw_minimap(vis, self.debug_path)

        vel_txt = (
            f"vx={vx_mps:.3f} vy={vy_mps:.3f} "
            f"vz={vz_mps:.3f} used={n_used}"
        )
        cv2.putText(
            vis,
            vel_txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        depth_txt = f"Depth age: {depth_age_sec * 1000.0:.1f} ms"
        cv2.putText(
            vis,
            depth_txt,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        center_depth_txt = f"Center depth: {self.center_depth:.2f} m"
        cv2.putText(
            vis,
            center_depth_txt,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 0),
            2
        )

        dist_txt = f"Distance: {self.current_distance:.2f} m"
        cv2.putText(
            vis,
            dist_txt,
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Flow+Depth Velocity - Separated RGB/Depth", vis)
        cv2.waitKey(1)

    def draw_debug(self, vis_bgr, good_old, good_new, depth_map):
        H, W = depth_map.shape[:2]

        for (u2, v2), (u1, v1) in zip(good_new, good_old):
            x2 = int(round(u2))
            y2 = int(round(v2))
            x1 = int(round(u1))
            y1 = int(round(v1))

            if not (0 <= x2 < W and 0 <= y2 < H):
                continue

            z = float(depth_map[y2, x2])
            if np.isfinite(z):
                cv2.arrowedLine(
                    vis_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                    tipLength=0.3
                )

    def draw_minimap(self, vis, path):
        if len(path) < 2:
            return

        map_size = 200
        margin = 15
        h, w = vis.shape[:2]

        overlay = vis.copy()
        cv2.rectangle(
            overlay,
            (w - map_size - margin, margin),
            (w - margin, margin + map_size),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        span = max(max_x - min_x, max_y - min_y, 1.0)
        scale = (map_size - 40) / span

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        screen_cx = w - margin - map_size // 2
        screen_cy = margin + map_size // 2

        pts = []
        for px, py in path:
            mx = int(screen_cx - (py - cy) * scale)
            my = int(screen_cy - (px - cx) * scale)
            pts.append((mx, my))

        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], (0, 255, 255), 2)

        cv2.circle(vis, pts[0], 4, (0, 255, 0), -1)
        cv2.circle(vis, pts[-1], 6, (0, 0, 255), -1)

        cv2.putText(
            vis,
            "Trajectory Map",
            (w - map_size - margin + 10, margin + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------
    def destroy_node(self):
        if self.trajectory_history:
            try:
                with open(self.json_out_path, "w", encoding="utf-8") as f:
                    json.dump(self.trajectory_history, f, indent=4)

                self.get_logger().info(
                    f"[JSON] Saved {len(self.trajectory_history)} poses "
                    f"to {self.json_out_path}"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to save JSON: {e}")

        super().destroy_node()


def main():
    rclpy.init()

    node = FlowDepthVelocitySeparatedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()