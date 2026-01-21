#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge


class FlowDepthVelocityNode(Node):
    """
    SPARX ROS2 node:
      - subscribes to RGB + Depth map (32FC1) + CameraInfo
      - tracks features with LK optical flow
      - samples depth for each tracked point
      - converts pixel flow to meters/sec using per-point depth:
            vx = Z * (du/dt)/fx
            vy = Z * (dv/dt)/fy
      - publishes robust (median) velocity
    """

    def __init__(self):
        super().__init__("flow_depth_velocity_node")

        # sim time
        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        # params
        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")
        self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("output_topic", "/flow_depth/velocity")

        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 30)

        self.declare_parameter("min_depth", 0.05)   # reject 0 / invalid
        self.declare_parameter("max_depth", 50.0)   # reject crazy values
        self.declare_parameter("use_depth_norm", False)  # if depth is 0..1 relative, keep as-is

        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_link")

        #  LK params
        self.declare_parameter("lk_win", 21) # window size
        self.declare_parameter("lk_levels", 3) # pyramid levels

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.max_corners = int(self.get_parameter("max_corners").get_parameter_value().integer_value)
        self.min_corners = int(self.get_parameter("min_corners").get_parameter_value().integer_value)

        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)
        self.use_depth_norm = bool(self.get_parameter("use_depth_norm").get_parameter_value().bool_value)

        self.show_debug = bool(self.get_parameter("show_debug").get_parameter_value().bool_value)
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        lk_win = int(self.get_parameter("lk_win").get_parameter_value().integer_value)
        lk_levels = int(self.get_parameter("lk_levels").get_parameter_value().integer_value)

        # log params
        self.get_logger().info(f"[FlowDepth] RGB: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] Pub: {output_topic}")

        self.bridge = CvBridge()

        # state
        self.prev_gray = None
        self.prev_pts = None
        self.prev_stamp = None

        self.latest_depth = None
        self.latest_depth_stamp = None

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # LK params
        self.lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), 
        )

        # pubs/subs
        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, 10)

        self.rgb_sub = self.create_subscription(Image, image_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)

    # ---------- callbacks ----------
    def caminfo_callback(self, msg: CameraInfo):
        K = msg.k
        self.fx = float(K[0])
        self.fy = float(K[4])
        self.cx = float(K[2])
        self.cy = float(K[5])

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        self.latest_depth = np.asarray(depth, dtype=np.float32)
        self.latest_depth_stamp = msg.header.stamp

    def rgb_callback(self, msg: Image):
        # Need cam intrinsics + depth
        if self.fx is None or self.fy is None:
            return
        if self.latest_depth is None:
            return

        # Convert RGB
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        # Grayscale for LK
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Init the first frame 
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self.detect_features(gray)
            self.prev_stamp = msg.header.stamp
            return

        # dt
        curr_stamp = msg.header.stamp
        dt_ns = (curr_stamp.sec - self.prev_stamp.sec) * 1_000_000_000 + \
                (curr_stamp.nanosec - self.prev_stamp.nanosec)
        dt = float(dt_ns) * 1e-9
        if dt <= 0.0: # if no time elapsed, skip frame
            self.prev_gray = gray
            self.prev_stamp = curr_stamp
            return

        # Refresh features if needed
        if self.prev_pts is None or len(self.prev_pts) < self.min_corners: # redetect if too few points
            self.prev_pts = self.detect_features(self.prev_gray)
            if self.prev_pts is None: # still no points found--skip frame
                self.prev_gray = gray
                self.prev_stamp = curr_stamp
                return

        # LK optical flow
        # next_pts: [N,1,2] float32 the new positions of input features in the second image
        # st: [N,1] uint8 status vector (1=found, 0=not found)
        # err: [N,1] float32 error vector "how good the flow for the feature is"
        
        next_pts, st, err = cv2.calcOpticalFlowPyrLK( 
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        if next_pts is None or st is None: # flow failed -- skip frame
            self.prev_gray = gray
            self.prev_pts = None
            self.prev_stamp = curr_stamp
            return

        # Select good points
        good_new = next_pts[st == 1]  # tracked points only
        good_old = self.prev_pts[st == 1] # corresponding old points
        if len(good_new) == 0: # no good points -- skip frame
            self.prev_gray = gray
            self.prev_pts = None
            self.prev_stamp = curr_stamp
            return

        # Compute per-point metric velocities using depth sampling
        vx_mps, vy_mps, n_used = self.velocity_from_flow_and_depth(good_old, good_new, self.latest_depth, dt)

        # Publish (robust median over points)
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(vx_mps)
        vel_msg.vector.y = float(vy_mps)
        vel_msg.vector.z = 0.0
        self.vel_pub.publish(vel_msg)

        # Debug visualization
        if self.show_debug:
            vis = frame.copy() # create a copy to draw on
            self.draw_debug(vis, good_old, good_new, self.latest_depth) # draw flow+depth
            txt = f"vx={vx_mps:.3f} m/s vy={vy_mps:.3f} m/s used={n_used}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1)

        # Update state for next frame
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.prev_stamp = curr_stamp

    # ---------- core helpers ----------

    def detect_features(self, gray):
        """Detect good features to track in the given grayscale image."""
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners, # maximum number of corners to return
            qualityLevel=0.01, # minimal quality level of image corners
            minDistance=7, # minimum possible Euclidean distance between the returned corners
            blockSize=7,  # size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        )
        return pts

    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float):
        """
        good_old/new: [N,2] float pixel coords
        depth_map: [H,W] float32 depth (relative or scaled)
        dt: seconds

        Returns:
          (vx_mps, vy_mps, n_used) as robust medians over valid points
        """
        H, W = depth_map.shape[:2] # height, width of depth map

        # pixel velocities px/s of the tracked points in the optical flow
        du = (good_new[:, 0] - good_old[:, 0]) / dt  # px/s
        dv = (good_new[:, 1] - good_old[:, 1]) / dt  # px/s

        # sample depth at new locations
        u = np.rint(good_new[:, 0]).astype(np.int32)
        v = np.rint(good_new[:, 1]).astype(np.int32)

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(valid):
            return 0.0, 0.0, 0

        Z = np.zeros_like(du, dtype=np.float32)
        Z[valid] = depth_map[v[valid], u[valid]]

        # filter bad depth
        valid = valid & np.isfinite(Z)

        if not self.use_depth_norm:
            valid = valid & (Z > self.min_depth) & (Z < self.max_depth)

        if not np.any(valid):
            return 0.0, 0.0, 0

        # Convert to m/s using per-point depth
        vx = Z[valid] * (du[valid] / self.fx)
        vy = Z[valid] * (dv[valid] / self.fy)

        # Robust summary
        vx_mps = float(np.median(vx))
        vy_mps = float(np.median(vy))
        return vx_mps, vy_mps, int(np.sum(valid))

    def draw_debug(self, vis_bgr, good_old, good_new, depth_map):
        H, W = depth_map.shape[:2]
        # simple depth->color normalization for lines
        d = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        dmin, dmax = float(np.min(d)), float(np.max(d))
        den = max(dmax - dmin, 1e-6)

        for (u2, v2), (u1, v1) in zip(good_new, good_old):
            x2, y2 = int(round(u2)), int(round(v2))
            x1, y1 = int(round(u1)), int(round(v1))
            if not (0 <= x2 < W and 0 <= y2 < H):
                continue
            z = float(depth_map[y2, x2])
            if not np.isfinite(z):
                continue
            t = (z - dmin) / den  # 0..1
            # map t to a visible color (no need to be perfect)
            #color = (int(255 * (1 - t)), int(255 * t), 128)
            color = (0, 0, 255)
            cv2.arrowedLine(vis_bgr, (x1, y1), (x2, y2), color, 2, tipLength=0.3)


def main():
    rclpy.init()
    node = FlowDepthVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
