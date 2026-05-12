#!/usr/bin/env python3
from __future__ import annotations
from turtle import stamp
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped, Twist, PoseStamped 
from cv_bridge import CvBridge
from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque
import csv
import os
import math  
import yaml


class FlowDepthVelocityNode(Node):
    """
    SPARX ROS2 node:
      - Subscribes to RGB + Depth 
      - Loads camera intrinsics from YAML config
      - WLS Optical Flow + Depth Smoothing
      - Publishes velocity
      - Listens to pose updates from VelocityIntegrator to draw the minimap
    """

    def __init__(self):
        super().__init__("flow_depth_velocity_node")

        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        # params
        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")
        
        # Read the camera_config_yaml parameter to get the camera_info 
        self.declare_parameter(
            "camera_config_yaml",
            "/home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml"
        )
        # read the camera_info from the topic instead of from the YAML
        #self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("output_topic", "/flow_depth/velocity")
        self.declare_parameter("pose_est_topic", "/flow_depth/pose_est") 

        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 70) 

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("use_depth_norm", False)
        self.declare_parameter("depth_scale", 1.0) 
        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "xtend_camera")
        
        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        self.declare_parameter("depth_median_window", 5)
        self.declare_parameter("depth_ema_alpha", 0.15) # EMA alpha for depth smoothing, between 0 and 1. Higher means more smoothing but more lag.
        self.declare_parameter("csv_filename", "/tmp/zone_velocities_log_no_imu.csv")
        
        self.declare_parameter("json_out_path", "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/estimated_trajectory.json")
        self.json_out_path = self.get_parameter("json_out_path").get_parameter_value().string_value

        self.declare_parameter("max_rgb_queue_size", 100)
        self.declare_parameter("max_sync_dt", 0.05)  # seconds, e.g. 50 ms

        self.trajectory_history = []

        self.median_window = int(self.get_parameter("depth_median_window").get_parameter_value().integer_value)
        self.ema_alpha = float(self.get_parameter("depth_ema_alpha").get_parameter_value().double_value)
        
        self.last_smoothed_depth = None
        self.center_depth = 0.0
        
        # === Variables for the Real-Time Minimap ===
        self.pose_origin = None
        self.debug_path = [(0.0, 0.0)]  # Start at origin
        self.current_distance = 0.0     # This will track the total distance traveled from the start point (0,0)
        # ===========================================

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        #caminfo_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        pose_est_topic = self.get_parameter("pose_est_topic").get_parameter_value().string_value

        self.max_corners = int(self.get_parameter("max_corners").get_parameter_value().integer_value)
        self.min_corners = int(self.get_parameter("min_corners").get_parameter_value().integer_value)
        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)
        self.use_depth_norm = bool(self.get_parameter("use_depth_norm").get_parameter_value().bool_value)
        self.depth_scale = float(self.get_parameter("depth_scale").get_parameter_value().double_value)
        self.show_debug = bool(self.get_parameter("show_debug").get_parameter_value().bool_value)
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        self.max_rgb_queue_size = int(
            self.get_parameter("max_rgb_queue_size").get_parameter_value().integer_value
        )

        self.max_sync_dt = float(
            self.get_parameter("max_sync_dt").get_parameter_value().double_value
        )

        self.rgb_queue = deque(maxlen=self.max_rgb_queue_size)

        # === Debug counters for manual RGB-Depth synchronization ===
        self.rgb_count = 0
        self.depth_count = 0
        self.matched_count = 0
        self.dropped_depth_count = 0
        self.dropped_rgb_count = 0
        self.last_sync_debug_time = self.get_clock().now()


        self.latest_gt_vx = 0.0
        self.latest_gt_vy = 0.0
        self.latest_gt_vz = 0.0 

        self.prev_vx, self.prev_vy, self.prev_vz = 0.0, 0.0, 0.0
        self.vel_alpha = 0.2

        lk_win = int(self.get_parameter("lk_win").get_parameter_value().integer_value)
        lk_levels = int(self.get_parameter("lk_levels").get_parameter_value().integer_value)
        self.csv_filename = self.get_parameter("csv_filename").get_parameter_value().string_value
        
        self.get_logger().info(f"[FlowDepth] RGB: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {depth_topic}")

        self.bridge = CvBridge()
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        camera_config_yaml = self.get_parameter("camera_config_yaml").get_parameter_value().string_value
        self.load_camera_intrinsics_from_yaml(camera_config_yaml)

        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )


        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, image_qos)
        #self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)
        self.create_subscription(Twist, '/simple_drone/gt_vel', self.gt_vel_callback, image_qos)

        self.pose_sub = self.create_subscription(PoseStamped, pose_est_topic, self.pose_est_callback, image_qos)
        # ------------------------------------------------

        self.rgb_sub = self.create_subscription(
            Image,
            image_topic,
            self.rgb_callback,
            image_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            image_qos
        )


    
    def pose_est_callback(self, msg: PoseStamped):
        # unpack the position from the PoseStamped message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z  
        
        self.debug_path.append((x, y))
        
        # Calculate the distance from the start point (0,0) to the current position (x,y) and update current_distance
        self.current_distance = math.sqrt(x**2 + y**2 + z**2)

        self.trajectory_history.append({
            "image": f"frame_{len(self.trajectory_history):06d}.jpg", 
            "pose": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "yaw": 0.0  
            }
        })
    # ---------------------------------


    def load_camera_intrinsics_from_yaml(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Camera YAML not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Prefer projection_matrix P, because this is the effective pinhole model
        # after rectification / undistortion.
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

        # Fallback to direct fx/fy/cx/cy
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

    def smooth_depth_map(self, raw_depth_map: np.ndarray) -> np.ndarray:
        if self.last_smoothed_depth is None:
            self.last_smoothed_depth = raw_depth_map
        else:
            # Apply Exponential Moving Average (EMA) smoothing
            self.last_smoothed_depth = (self.ema_alpha * raw_depth_map) + \
                                       ((1.0 - self.ema_alpha) * self.last_smoothed_depth)
        return self.last_smoothed_depth.astype(np.float32)

    """
    def caminfo_callback(self, msg: CameraInfo):
        # Try to extract intrinsics from the CameraInfo message
        P = msg.p
        if len(P) < 12 or P[0] == 0.0:
            K = msg.k
            self.fx = float(K[0])
            self.fy = float(K[4])
            self.cx = float(K[2])
            self.cy = float(K[5])
        else:
            self.fx = float(P[0])
            self.fy = float(P[5])
            self.cx = float(P[2])
            self.cy = float(P[6])
            
        if not hasattr(self, '_caminfo_logged'):
            self.get_logger().info(f"[CamInfo] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            self._caminfo_logged = True
    """

    def gt_vel_callback(self, msg):
        self.latest_gt_vx = msg.linear.x
        self.latest_gt_vy = msg.linear.y
        self.latest_gt_vz = msg.linear.z
    
    def stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def rgb_callback(self, msg: Image):
        """
        Store incoming RGB messages in a queue.
        Depth callback will later find the RGB image with the closest timestamp.
        """

        self.rgb_count += 1

        rgb_t = self.stamp_to_sec(msg.header.stamp)

        queue_size_before = len(self.rgb_queue)
        self.rgb_queue.append(msg)
        queue_size_after = len(self.rgb_queue)

        # If deque is full, append automatically removes the oldest message.
        if queue_size_before == self.max_rgb_queue_size and queue_size_after == self.max_rgb_queue_size:
            self.dropped_rgb_count += 1

        self.get_logger().debug(
            f"[RGB Queue] received_rgb={self.rgb_count} "
            f"stamp={rgb_t:.9f} "
            f"queue_size={queue_size_after}/{self.max_rgb_queue_size} "
            f"auto_dropped_old_rgb={self.dropped_rgb_count}"
        )

    def depth_callback(self, depth_msg: Image):
        """
        When a depth frame arrives:
        1. Search the RGB queue.
        2. Find the RGB message with the closest timestamp.
        3. If the time difference is small enough, process RGB + Depth together.
        """

        self.depth_count += 1
        depth_t = self.stamp_to_sec(depth_msg.header.stamp)

        self.get_logger().debug(
            f"[Depth] received_depth={self.depth_count} "
            f"stamp={depth_t:.9f} "
            f"rgb_queue_size={len(self.rgb_queue)}"
        )

        if len(self.rgb_queue) == 0:
            self.dropped_depth_count += 1
            self.get_logger().warn(
                f"[Sync] Depth arrived but RGB queue is empty | "
                f"depth_count={self.depth_count} "
                f"dropped_depth={self.dropped_depth_count}"
            )
            return

        best_idx = None
        best_dt = None
        best_rgb_t = None

        oldest_rgb_t = self.stamp_to_sec(self.rgb_queue[0].header.stamp)
        newest_rgb_t = self.stamp_to_sec(self.rgb_queue[-1].header.stamp)

        for i, rgb_msg in enumerate(self.rgb_queue):
            rgb_t = self.stamp_to_sec(rgb_msg.header.stamp)
            dt = abs(depth_t - rgb_t)

            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = i
                best_rgb_t = rgb_t

        if best_idx is None or best_dt is None:
            self.dropped_depth_count += 1
            self.get_logger().warn(
                f"[Sync] Could not find RGB candidate | "
                f"depth_stamp={depth_t:.9f} "
                f"queue_size={len(self.rgb_queue)}"
            )
            return

        self.get_logger().debug(
            f"[Sync Search] depth_stamp={depth_t:.9f} "
            f"oldest_rgb={oldest_rgb_t:.9f} "
            f"newest_rgb={newest_rgb_t:.9f} "
            f"best_rgb={best_rgb_t:.9f} "
            f"best_idx={best_idx} "
            f"best_dt={best_dt*1000:.2f} ms "
            f"max_allowed={self.max_sync_dt*1000:.2f} ms"
        )

        if best_dt > self.max_sync_dt:
            self.dropped_depth_count += 1

            removed_old_rgb = 0

            # Remove RGB messages that are much older than this depth
            while len(self.rgb_queue) > 0:
                rgb_t = self.stamp_to_sec(self.rgb_queue[0].header.stamp)

                if rgb_t < depth_t - self.max_sync_dt:
                    self.rgb_queue.popleft()
                    removed_old_rgb += 1
                    self.dropped_rgb_count += 1
                else:
                    break

            self.get_logger().warn(
                f"[Sync FAIL] No close RGB for Depth | "
                f"depth_stamp={depth_t:.9f} "
                f"best_rgb_stamp={best_rgb_t:.9f} "
                f"best_dt={best_dt*1000:.2f} ms "
                f"max_sync_dt={self.max_sync_dt*1000:.2f} ms "
                f"removed_old_rgb={removed_old_rgb} "
                f"queue_size_after_cleanup={len(self.rgb_queue)} "
                f"dropped_depth={self.dropped_depth_count}"
            )

            return

        rgb_msg = self.rgb_queue[best_idx]

        # Remove matched RGB and all older RGB frames so we do not reuse the same RGB
        removed_rgb = best_idx + 1
        for _ in range(removed_rgb):
            self.rgb_queue.popleft()

        self.matched_count += 1

        self.get_logger().info(
            f"[Sync OK] match={self.matched_count} "
            f"rgb_count={self.rgb_count} "
            f"depth_count={self.depth_count} "
            f"rgb_stamp={best_rgb_t:.9f} "
            f"depth_stamp={depth_t:.9f} "
            f"dt={best_dt*1000:.2f} ms "
            f"removed_rgb={removed_rgb} "
            f"queue_size_after={len(self.rgb_queue)}"
        )

        # Print summary every ~2 seconds
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_sync_debug_time).nanoseconds * 1e-9

        if elapsed_sec >= 2.0:
            self.get_logger().info(
                f"[Sync Summary] "
                f"rgb_received={self.rgb_count} "
                f"depth_received={self.depth_count} "
                f"matched={self.matched_count} "
                f"dropped_depth={self.dropped_depth_count} "
                f"dropped_rgb={self.dropped_rgb_count} "
                f"rgb_queue_size={len(self.rgb_queue)}"
            )
            self.last_sync_debug_time = now

        self.sync_callback(rgb_msg, depth_msg)

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None or self.fy is None:
            return

        try:
            # Convert ROS Image messages to OpenCV format
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Convert failed: {e}")
            return
        
        # Apply smoothing to the depth map
        depth_map = self.smooth_depth_map(np.asarray(depth_cv, dtype=np.float32))


        # Extract the depth value at the center of the image for use in velocity scaling and as a fallback depth estimate
        half_side = 2 
        u_idx = int(self.cx)
        v_idx = int(self.cy)

        center_region = depth_map[v_idx-half_side : v_idx+half_side+1, 
                                u_idx-half_side : u_idx+half_side+1]

        valid_center_depths = center_region[np.isfinite(center_region) & (center_region > 0)]
        if valid_center_depths.size > 0:
            self.center_depth = np.mean(valid_center_depths) * self.depth_scale

        # Run the optical flow tracker and compute velocity
        flow_res = self.tracker.process(frame, rgb_msg.header.stamp)
        if flow_res is None:
            return

        # Compute velocity from flow and depth
        vx_mps, vy_mps, vz_mps, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old, flow_res.good_new, depth_map, flow_res.dt
        )

        # Apply simple low-pass filtering to the velocity estimates to reduce noise
        vx_mps = self.vel_alpha * vx_mps + (1 - self.vel_alpha) * self.prev_vx
        vy_mps = self.vel_alpha * vy_mps + (1 - self.vel_alpha) * self.prev_vy
        vz_mps = self.vel_alpha * vz_mps + (1 - self.vel_alpha) * self.prev_vz

        self.prev_vx, self.prev_vy, self.prev_vz = vx_mps, vy_mps, vz_mps


        if abs(vx_mps) < 0.02: vx_mps = 0.0
        if abs(vy_mps) < 0.2: vy_mps = 0.0
        if abs(vz_mps) < 0.2: vz_mps = 0.0

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = rgb_msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x, vel_msg.vector.y, vel_msg.vector.z = float(vx_mps), float(vy_mps), float(vz_mps)
        self.vel_pub.publish(vel_msg)

        # Visual Debug
        if self.show_debug:
            vis = frame.copy()
            self.draw_debug(vis, flow_res.good_old, flow_res.good_new, depth_map)
            
            # Draw the new Minimap
            self.draw_minimap(vis, self.debug_path)
            
            vel_txt = f"vx={vx_mps:.3f} vy={vy_mps:.3f} vz={vz_mps:.3f} used={n_used}"
            cv2.putText(vis, vel_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            
            dist_txt = f"Distance: {self.current_distance:.2f} m"
            (tw, th), _ = cv2.getTextSize(dist_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(vis, (8, 35), (12 + tw, 45 + th), (0, 0, 0), -1) 
            cv2.putText(vis, dist_txt, (10, 40 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

          #  wall_dist_txt = f"Wall Dist (Center): {self.center_depth:.2f} m"
          #  cv2.putText(vis, wall_dist_txt, (10, 80 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)

           # cv2.drawMarker(vis, (u_idx, v_idx), (255, 100, 0), cv2.MARKER_CROSS, 15, 2)
            # --------------------------------------

            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1) 

    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float):
        # Returns velocity in m/s in the camera frame (vx forward, vy right, vz down)

        # H and W of the depth map for bounds checking
        H, W = depth_map.shape[:2]
         
        # du and dv are the pixel displacements divided by dt to get pixel velocities
        du = (good_new[:, 0] - good_old[:, 0]) / dt
        dv = (good_new[:, 1] - good_old[:, 1]) / dt

        # Convert good_new to integer pixel coordinates for depth lookup
        u_int = np.rint(good_new[:, 0]).astype(np.int32)
        v_int = np.rint(good_new[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        if not np.any(valid): return 0.0, 0.0, 0.0, 0

        # Get the depth values at the new feature locations
        Z = np.zeros_like(du, dtype=np.float32)
        Z[valid] = depth_map[v_int[valid], u_int[valid]] * self.depth_scale
        valid = valid & np.isfinite(Z) & (Z > self.min_depth) & (Z < self.max_depth)

        nv = int(np.sum(valid))
        if nv < 8: return 0.0, 0.0, 0.0, 0

        u_c_v = good_old[valid, 0].astype(np.float64) - self.cx
        v_c_v = good_old[valid, 1].astype(np.float64) - self.cy
        Zv, du_v, dv_v = Z[valid].astype(np.float64), du[valid].astype(np.float64), dv[valid].astype(np.float64)

        A = np.zeros((2 * nv, 3), dtype=np.float64)
        B = np.zeros((2 * nv,),   dtype=np.float64)
        A[0::2, 0] = -self.fx;  A[0::2, 2] = u_c_v;  B[0::2] = du_v * Zv
        A[1::2, 1] = -self.fy;  A[1::2, 2] = v_c_v;  B[1::2] = dv_v * Zv

        # Weights based on distance from the center of the image (optional, can help reduce noise from features near the edges)
        dist_sq = u_c_v**2 + v_c_v**2
        weights = np.exp(-dist_sq / (0.05 * (self.cx**2 + self.cy**2)))
        W_vec = np.zeros((2 * nv,), dtype=np.float64)
        W_vec[0::2], W_vec[1::2] = weights, weights
        
        sqrt_w = np.sqrt(W_vec)
        try:
            vel, *_ = np.linalg.lstsq(A * sqrt_w[:, np.newaxis], B * sqrt_w, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, nv

        # vel is in the camera frame with vx forward, vy right, vz down. We want to return it as vx forward, vy left, vz up, so we need to negate vy and vz.
        Vx_o, Vy_o, Vz_o = float(vel[0]), float(vel[1]), float(vel[2])
        return Vz_o, -Vx_o, -Vy_o, nv

    def draw_debug(self, vis_bgr, good_old, good_new, depth_map):
        H, W = depth_map.shape[:2]
        for (u2, v2), (u1, v1) in zip(good_new, good_old):
            x2, y2, x1, y1 = int(round(u2)), int(round(v2)), int(round(u1)), int(round(v1))
            if not (0 <= x2 < W and 0 <= y2 < H): continue
            z = float(depth_map[y2, x2])
            if np.isfinite(z):
                cv2.arrowedLine(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2, tipLength=0.3)

    # ==========================================
    # Minimap Drawing Function
    # ==========================================
    def draw_minimap(self, vis, path):
        if len(path) < 2:
            return
            
        map_size = 200
        margin = 15
        h, w = vis.shape[:2]

        # Draw semi-transparent background at top-right corner
        overlay = vis.copy()
        cv2.rectangle(overlay, (w - map_size - margin, margin), (w - margin, margin + map_size), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        # Dynamic Scaling (Find min/max bounds of the path)
        xs, ys = [p[0] for p in path], [p[1] for p in path]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        span = max(max_x - min_x, max_y - min_y, 1.0)
        scale = (map_size - 40) / span  # Leave 20px inner padding
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

        # Screen center for the minimap box
        screen_cx = w - margin - map_size // 2
        screen_cy = margin + map_size // 2

        # Convert path points to screen coordinates
        pts = []
        for (px, py) in path:
            mx = int(screen_cx - (py - cy) * scale)
            my = int(screen_cy - (px - cx) * scale)
            pts.append((mx, my))

        # Draw the trajectory lines
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], (0, 255, 255), 2)  # Yellow path

        # Draw the start point (Green Dot)
        cv2.circle(vis, pts[0], 4, (0, 255, 0), -1)
        
        # Draw current position (Red Dot)
        cv2.circle(vis, pts[-1], 6, (0, 0, 255), -1)

        # Add Title
        cv2.putText(vis, "Trajectory Map", (w - map_size - margin + 10, margin + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    def destroy_node(self):
            if self.trajectory_history:
                try:
                    import json
                    with open(self.json_out_path, 'w', encoding='utf-8') as f:
                        json.dump(self.trajectory_history, f, indent=4)
                    self.get_logger().info(f"[JSON] Saved {len(self.trajectory_history)} poses to {self.json_out_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to save JSON: {e}")
                    
            super().destroy_node()

def main():
    rclpy.init()
    node = FlowDepthVelocityNode()
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