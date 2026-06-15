#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped, Twist, PoseStamped 
from cv_bridge import CvBridge
from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker
from sparx_agency.robots.common.image_utils import _finite_mask
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# --- Added imports for concurrency and queue management ---
import threading
from collections import deque
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import os
import math  
import yaml
import json
from sparx_agency.robots.common.helpers import valid_depth_mask

class FlowDepthVelocityNode(Node):
    """
    SPARX ROS2 node:
      - Subscribes to RGB + Depth 
      - Runs Optical Flow continuously on RGB frames (High Frequency)
      - Late-binds flow results with Depth map based on timestamps (Low Frequency)
      - Publishes 3D velocity
    """

    def __init__(self):
        super().__init__("flow_depth_velocity_node")

        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        # Node Parameters
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

        self.declare_parameter("depth_median_window", 5)
        self.declare_parameter("depth_ema_alpha", 0.15) 
        
        self.declare_parameter("json_out_path", "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/estimated_trajectory.json")
        self.json_out_path = self.get_parameter("json_out_path").get_parameter_value().string_value

        # Queue Management Parameters
        self.declare_parameter("max_rgb_queue_size", 100)
        self.declare_parameter("max_sync_dt", 0.05)  # Max allowed time difference in seconds (50 ms)

        # Internal Variables
        self.trajectory_history = []
        self.median_window = int(self.get_parameter("depth_median_window").get_parameter_value().integer_value)
        self.ema_alpha = float(self.get_parameter("depth_ema_alpha").get_parameter_value().double_value)
        self.last_smoothed_depth = None
        self.center_depth = 0.0
        
        # Minimap Variables
        self.pose_origin = None
        self.debug_path = [(0.0, 0.0)]  
        self.current_distance = 0.0     

        # Get parameter values
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        pose_est_topic = self.get_parameter("pose_est_topic").get_parameter_value().string_value

        self.max_corners = int(self.get_parameter("max_corners").get_parameter_value().integer_value)
        self.min_corners = int(self.get_parameter("min_corners").get_parameter_value().integer_value)
        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)
        self.depth_scale = float(self.get_parameter("depth_scale").get_parameter_value().double_value)
        self.show_debug = bool(self.get_parameter("show_debug").get_parameter_value().bool_value)
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.max_rgb_queue_size = int(self.get_parameter("max_rgb_queue_size").get_parameter_value().integer_value)
        self.max_sync_dt = float(self.get_parameter("max_sync_dt").get_parameter_value().double_value)

        # --- Queue and Concurrency Setup ---
        self.rgb_queue = deque(maxlen=self.max_rgb_queue_size)
        self.queue_lock = threading.Lock() # Prevents race conditions between callbacks

        # Callback groups allow the callbacks to execute in parallel threads
        self.rgb_cb_group = MutuallyExclusiveCallbackGroup()
        self.depth_cb_group = MutuallyExclusiveCallbackGroup()

        # Debug Counters
        self.rgb_count = 0
        self.depth_count = 0
        self.matched_count = 0
        self.dropped_depth_count = 0
        self.dropped_rgb_count = 0
        self.last_sync_debug_time = self.get_clock().now()

        # Velocity Variables
        self.prev_vx, self.prev_vy, self.prev_vz = 0.0, 0.0, 0.0
        self.vel_alpha = 0.2

        lk_win = int(self.get_parameter("lk_win").get_parameter_value().integer_value)
        lk_levels = int(self.get_parameter("lk_levels").get_parameter_value().integer_value)

        self.get_logger().info(f"[FlowDepth] RGB Topic: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth Topic: {depth_topic}")

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
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, image_qos)
        self.create_subscription(Twist, '/simple_drone/gt_vel', self.gt_vel_callback, image_qos)
        self.pose_sub = self.create_subscription(PoseStamped, pose_est_topic, self.pose_est_callback, image_qos)
        
        # Subscriptions using Callback Groups
        self.rgb_sub = self.create_subscription(
            Image, image_topic, self.rgb_callback, image_qos, callback_group=self.rgb_cb_group
        )

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, image_qos, callback_group=self.depth_cb_group
        )

    def stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def load_camera_intrinsics_from_yaml(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Camera YAML not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        if "projection_matrix" in cfg and "data" in cfg["projection_matrix"]:
            P = cfg["projection_matrix"]["data"]
            if len(P) < 12 or P[0] == 0.0:
                raise ValueError(f"Invalid projection_matrix in YAML: {yaml_path}")

            self.fx, self.fy = float(P[0]), float(P[5])
            self.cx, self.cy = float(P[2]), float(P[6])
            self.get_logger().info(f"[Camera YAML] Loaded from P: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            return

        required = ["fx", "fy", "cx", "cy"]
        if all(k in cfg for k in required):
            self.fx, self.fy = float(cfg["fx"]), float(cfg["fy"])
            self.cx, self.cy = float(cfg["cx"]), float(cfg["cy"])
            self.get_logger().info(f"[Camera YAML] Loaded raw intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            return

        raise ValueError(f"Could not find projection_matrix or fx/fy/cx/cy in YAML: {yaml_path}")

    def smooth_depth_map(self, raw_depth_map: np.ndarray) -> np.ndarray:
        if self.last_smoothed_depth is None:
            self.last_smoothed_depth = raw_depth_map
        else:
            self.last_smoothed_depth = (self.ema_alpha * raw_depth_map) + ((1.0 - self.ema_alpha) * self.last_smoothed_depth)
        return self.last_smoothed_depth.astype(np.float32)

    def gt_vel_callback(self, msg):
        pass # Optional logic for GT comparison

    def pose_est_callback(self, msg: PoseStamped):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z  
        self.debug_path.append((x, y))
        self.current_distance = math.sqrt(x**2 + y**2 + z**2)
        self.trajectory_history.append({
            "image": f"frame_{len(self.trajectory_history):06d}.jpg", 
            "pose": {"x": float(x), "y": float(y), "z": float(z), "yaw": 0.0}
        })

    # ==========================================
    # Callbacks
    # ==========================================

    def rgb_callback(self, msg: Image):
        """
        Runs constantly at high frequency (e.g., 30Hz).
        Executes Optical Flow continuously to maintain track stability, 
        then stores the flow results in the queue for later binding with Depth.
        """
        if self.fx is None or self.fy is None:
            return

        self.rgb_count += 1
        rgb_t = self.stamp_to_sec(msg.header.stamp)

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        # Run Optical Flow Tracker on EVERY incoming frame
        flow_res = self.tracker.process(frame, msg.header.stamp)
        if flow_res is None:
            return

        # Store the optical flow results in a dictionary
        queue_item = {
            'stamp_sec': rgb_t,
            'ros_stamp': msg.header.stamp,
            'good_old': flow_res.good_old,
            'good_new': flow_res.good_new,
            'dt': flow_res.dt,
            'frame': frame if self.show_debug else None # Save frame only if debugging is active
        }

        # Lock the queue before modifying it
        with self.queue_lock:
            self.rgb_queue.append(queue_item)

    def depth_callback(self, depth_msg: Image):
        """
        Runs at lower frequency (e.g., 5Hz).
        Finds the closest flow result from the queue, extracts it, 
        and calculates the 3D velocity.
        """
        self.depth_count += 1
        depth_t = self.stamp_to_sec(depth_msg.header.stamp)

        matched_flow_item = None

        # --- Scope to lock and search the queue ---
        with self.queue_lock:
            if len(self.rgb_queue) == 0:
                self.dropped_depth_count += 1
                return

            best_idx = None
            best_dt = float('inf')

            # Find the closest timestamp match in the queue
            for i, item in enumerate(self.rgb_queue):
                dt = abs(depth_t - item['stamp_sec'])
                if dt < best_dt:
                    best_dt = dt
                    best_idx = i

            # If the closest match is still too far apart in time, discard the depth frame
            if best_idx is None or best_dt > self.max_sync_dt:
                self.dropped_depth_count += 1
                
                # Cleanup frames that are too old to prevent memory bloat
                while len(self.rgb_queue) > 0 and self.rgb_queue[0]['stamp_sec'] < depth_t - self.max_sync_dt:
                    self.rgb_queue.popleft()
                    self.dropped_rgb_count += 1
                return

            # Match found! Extract it.
            matched_flow_item = self.rgb_queue[best_idx]
            
            # Remove the matched frame and all older frames from the queue
            for _ in range(best_idx + 1):
                self.rgb_queue.popleft()

        # --- Queue lock released. Now perform heavy operations ---
        self.matched_count += 1

        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return
        
        depth_map = self.smooth_depth_map(np.asarray(depth_cv, dtype=np.float32))

        # Extract center depth for reference/debug
        u_idx, v_idx = int(self.cx), int(self.cy)
        half_side = 2 
        if 0 <= v_idx-half_side and v_idx+half_side+1 <= depth_map.shape[0] and \
           0 <= u_idx-half_side and u_idx+half_side+1 <= depth_map.shape[1]:
            center_region = depth_map[v_idx-half_side : v_idx+half_side+1, u_idx-half_side : u_idx+half_side+1]
            valid_center_depths = center_region[_finite_mask(center_region)]
            if valid_center_depths.size > 0:
                self.center_depth = np.mean(valid_center_depths) * self.depth_scale

        # Compute velocity using matched flow results and the new depth map
        vx_mps, vy_mps, vz_mps, n_used = self.velocity_from_flow_and_depth(
            matched_flow_item['good_old'], 
            matched_flow_item['good_new'], 
            depth_map, 
            matched_flow_item['dt']
        )

        # Apply low-pass filter
        vx_mps = self.vel_alpha * vx_mps + (1 - self.vel_alpha) * self.prev_vx
        vy_mps = self.vel_alpha * vy_mps + (1 - self.vel_alpha) * self.prev_vy
        vz_mps = self.vel_alpha * vz_mps + (1 - self.vel_alpha) * self.prev_vz
        self.prev_vx, self.prev_vy, self.prev_vz = vx_mps, vy_mps, vz_mps

        # Deadband thresholding
        if abs(vx_mps) < 0.02: vx_mps = 0.0
        if abs(vy_mps) < 0.2: vy_mps = 0.0
        if abs(vz_mps) < 0.2: vz_mps = 0.0

        # Publish the velocity 
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = matched_flow_item['ros_stamp'] # Use original RGB timestamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x, vel_msg.vector.y, vel_msg.vector.z = float(vx_mps), float(vy_mps), float(vz_mps)
        self.vel_pub.publish(vel_msg)

        # Print debug summary occasionally
        now = self.get_clock().now()
        if (now - self.last_sync_debug_time).nanoseconds * 1e-9 >= 5.0:
            self.get_logger().info(
                f"[Sync Stats] Matched: {self.matched_count} | Dropped Depth: {self.dropped_depth_count} | Queue Size: {len(self.rgb_queue)}"
            )
            self.last_sync_debug_time = now

        # Debug Visualization
        if self.show_debug and matched_flow_item['frame'] is not None:
            vis = matched_flow_item['frame'].copy()
            self.draw_debug(vis, matched_flow_item['good_old'], matched_flow_item['good_new'], depth_map)
            self.draw_minimap(vis, self.debug_path)
            
            cv2.putText(vis, f"vx={vx_mps:.3f} vy={vy_mps:.3f} vz={vz_mps:.3f} used={n_used}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            dist_txt = f"Distance: {self.current_distance:.2f} m"
            (tw, th), _ = cv2.getTextSize(dist_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(vis, (8, 35), (12 + tw, 45 + th), (0, 0, 0), -1) 
            cv2.putText(vis, dist_txt, (10, 40 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1) 

    # ==========================================
    # Velocity Calculation & Drawing
    # ==========================================

    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float):
        H, W = depth_map.shape[:2]
         
        du = (good_new[:, 0] - good_old[:, 0]) / dt
        dv = (good_new[:, 1] - good_old[:, 1]) / dt

        u_int = np.rint(good_new[:, 0]).astype(np.int32)
        v_int = np.rint(good_new[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        if not np.any(valid): return 0.0, 0.0, 0.0, 0

        Z = np.zeros_like(du, dtype=np.float32)
        Z[valid] = depth_map[v_int[valid], u_int[valid]] * self.depth_scale
        valid = valid & valid_depth_mask(Z, min_depth=self.min_depth, max_depth=self.max_depth)

        nv = int(np.sum(valid))
        if nv < 8: return 0.0, 0.0, 0.0, 0

        u_c_v = good_old[valid, 0].astype(np.float64) - self.cx
        v_c_v = good_old[valid, 1].astype(np.float64) - self.cy
        Zv, du_v, dv_v = Z[valid].astype(np.float64), du[valid].astype(np.float64), dv[valid].astype(np.float64)

        A = np.zeros((2 * nv, 3), dtype=np.float64)
        B = np.zeros((2 * nv,),   dtype=np.float64)
        A[0::2, 0] = -self.fx;  A[0::2, 2] = u_c_v;  B[0::2] = du_v * Zv
        A[1::2, 1] = -self.fy;  A[1::2, 2] = v_c_v;  B[1::2] = dv_v * Zv

        dist_sq = u_c_v**2 + v_c_v**2
        weights = np.exp(-dist_sq / (0.05 * (self.cx**2 + self.cy**2)))
        W_vec = np.zeros((2 * nv,), dtype=np.float64)
        W_vec[0::2], W_vec[1::2] = weights, weights
        
        sqrt_w = np.sqrt(W_vec)
        try:
            vel, *_ = np.linalg.lstsq(A * sqrt_w[:, np.newaxis], B * sqrt_w, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, nv

        # Convert to typical odometry frame (vx forward, vy left, vz up)
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

    def draw_minimap(self, vis, path):
        if len(path) < 2:
            return
            
        map_size = 200
        margin = 15
        h, w = vis.shape[:2]

        overlay = vis.copy()
        cv2.rectangle(overlay, (w - map_size - margin, margin), (w - margin, margin + map_size), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        xs, ys = [p[0] for p in path], [p[1] for p in path]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        span = max(max_x - min_x, max_y - min_y, 1.0)
        scale = (map_size - 40) / span  
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

        screen_cx = w - margin - map_size // 2
        screen_cy = margin + map_size // 2

        pts = []
        for (px, py) in path:
            mx = int(screen_cx - (py - cy) * scale)
            my = int(screen_cy - (px - cx) * scale)
            pts.append((mx, my))

        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], (0, 255, 255), 2)  

        cv2.circle(vis, pts[0], 4, (0, 255, 0), -1)
        cv2.circle(vis, pts[-1], 6, (0, 0, 255), -1)
        cv2.putText(vis, "Trajectory Map", (w - map_size - margin + 10, margin + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def destroy_node(self):
        if self.trajectory_history:
            try:
                with open(self.json_out_path, 'w', encoding='utf-8') as f:
                    json.dump(self.trajectory_history, f, indent=4)
                self.get_logger().info(f"[JSON] Saved {len(self.trajectory_history)} poses to {self.json_out_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to save JSON: {e}")
                
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FlowDepthVelocityNode()
    
    # Critical: Use MultiThreadedExecutor to run callbacks in parallel
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    cv2.destroyAllWindows()
    
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    main()