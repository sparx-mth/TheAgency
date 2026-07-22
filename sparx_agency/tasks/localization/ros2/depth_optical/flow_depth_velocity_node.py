#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped, Twist, PoseStamped 
from cv_bridge import CvBridge
from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker
from sparx_agency.robots.common.image_utils import _finite_mask
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pathlib import Path

import message_filters
import csv
import os
import math  
import yaml
from sparx_agency.robots.common.helpers import valid_depth_mask


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
        
        self.declare_parameter(
            "camera_config_yaml",
            str(Path.home() / "GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml")
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
        
        self.declare_parameter("log_vel_csv_path", "/tmp/velocity_log.csv")
        self.declare_parameter("log_pose_csv_path", "/tmp/pose_log.csv")
        
        self.declare_parameter(
            "json_out_path",
            str(Path.home() / "GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/estimated_trajectory.json"),
        )
        
        self.json_out_path = self.get_parameter("json_out_path").get_parameter_value().string_value

        self.vel_csv_path = self.get_parameter("log_vel_csv_path").get_parameter_value().string_value
        self.pose_csv_path = self.get_parameter("log_pose_csv_path").get_parameter_value().string_value

        # Initialize velocity CSV file
        self.vel_log_file = open(self.vel_csv_path, 'w', newline='')
        self.vel_csv_writer = csv.writer(self.vel_log_file)
        self.vel_csv_writer.writerow([
            'timestamp', 'dt', 
           # 'raw_vx', 'raw_vy', 'raw_vz', 
           # 'filtered_vx', 'filtered_vy', 'filtered_vz', 
            'pub_vx', 'pub_vy', 'pub_vz', 
            'features_used'
        ])

        # Initialize pose CSV file
        self.pose_log_file = open(self.pose_csv_path, 'w', newline='')
        self.pose_csv_writer = csv.writer(self.pose_log_file)
        self.pose_csv_writer.writerow(['timestamp', 'x', 'y', 'z'])

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
            reliability=ReliabilityPolicy.RELIABLE,
        )


        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, image_qos)
        #self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)
        self.create_subscription(Twist, '/simple_drone/gt_vel', self.gt_vel_callback, image_qos)

        self.pose_sub = self.create_subscription(PoseStamped, pose_est_topic, self.pose_est_callback, image_qos)
        # ------------------------------------------------

        self.rgb_sub = message_filters.Subscriber(self, Image, image_topic,qos_profile=image_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic,qos_profile=image_qos)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=50, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)


    
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

        # Log pose to CSV
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.pose_csv_writer.writerow([
            f"{stamp_sec:.6f}", 
            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"
        ])
        self.pose_log_file.flush()
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

        valid_center_depths = center_region[_finite_mask(center_region)]
        if valid_center_depths.size > 0:
            self.center_depth = np.mean(valid_center_depths) * self.depth_scale

        # Run the optical flow tracker and compute velocity
        flow_res = self.tracker.process(frame, rgb_msg.header.stamp)
        if flow_res is None:
            return

        # Compute velocity from flow and depth
        raw_vx_mps, raw_vy_mps, raw_vz_mps, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old, flow_res.good_new, depth_map, flow_res.dt
        )

        # Apply simple low-pass filtering to the velocity estimates to reduce noise
        filtered_vx_mps = self.vel_alpha * raw_vx_mps + (1 - self.vel_alpha) * self.prev_vx
        filtered_vy_mps = self.vel_alpha * raw_vy_mps + (1 - self.vel_alpha) * self.prev_vy
        filtered_vz_mps = self.vel_alpha * raw_vz_mps + (1 - self.vel_alpha) * self.prev_vz

        self.prev_vx, self.prev_vy, self.prev_vz = filtered_vx_mps, filtered_vy_mps, filtered_vz_mps

        # Copy for publishing
        vx_mps, vy_mps, vz_mps = filtered_vx_mps, filtered_vy_mps, filtered_vz_mps

        if abs(vx_mps) < 0.02: vx_mps = 0.0
        if abs(vy_mps) < 0.2: vy_mps = 0.0
        if abs(vz_mps) < 0.2: vz_mps = 0.0

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = rgb_msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x, vel_msg.vector.y, vel_msg.vector.z = float(vx_mps), float(vy_mps), float(vz_mps)
        self.vel_pub.publish(vel_msg)

        # Log velocity to CSV
        rgb_stamp_sec = float(rgb_msg.header.stamp.sec) + float(rgb_msg.header.stamp.nanosec) * 1e-9
        self.vel_csv_writer.writerow([
            f"{rgb_stamp_sec:.6f}", f"{flow_res.dt:.4f}",
           # f"{raw_vx_mps:.4f}", f"{raw_vy_mps:.4f}", f"{raw_vz_mps:.4f}",
           # f"{filtered_vx_mps:.4f}", f"{filtered_vy_mps:.4f}", f"{filtered_vz_mps:.4f}",
            f"{vx_mps:.4f}", f"{vy_mps:.4f}", f"{vz_mps:.4f}",
            n_used
        ])
        self.vel_log_file.flush()  # Ensure data is written to disk promptly

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
            if hasattr(self, 'vel_log_file') and not self.vel_log_file.closed:
                self.vel_log_file.close()
                self.get_logger().info(f"[CSV] Saved velocity log to {self.vel_csv_path}")
            
            if hasattr(self, 'pose_log_file') and not self.pose_log_file.closed:
                self.pose_log_file.close()
                self.get_logger().info(f"[CSV] Saved pose log to {self.pose_csv_path}")

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