#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge
from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker

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
        self.declare_parameter("max_depth", 30.0)   # reject crazy values
        self.declare_parameter("use_depth_norm", False)  # if depth is 0..1 relative, keep as-is
        self.declare_parameter("depth_scale", 0.4)  # Calibrated for Depth Anything V2 + Gazebo
        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_link")

        #  LK params (Lucas-Kanade optical flow)
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
        self.depth_scale = float(self.get_parameter("depth_scale").get_parameter_value().double_value)

        self.show_debug = bool(self.get_parameter("show_debug").get_parameter_value().bool_value)
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        max_corners = int(self.get_parameter("max_corners").value)
        min_corners = int(self.get_parameter("min_corners").value)
        lk_win = int(self.get_parameter("lk_win").get_parameter_value().integer_value)
        lk_levels = int(self.get_parameter("lk_levels").get_parameter_value().integer_value)

        # log params
        self.get_logger().info(f"[FlowDepth] RGB: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] Pub: {output_topic}")

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_stamp = None

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # tracker module (fixed)
        self.tracker = OpticalFlowTracker(
            max_corners=max_corners,
            min_corners=min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

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
        self.fx = float(K[0]) # focal length x in pixels
        self.fy = float(K[4]) # focal length y in pixels
        self.cx = float(K[2]) # principal point x in pixels
        self.cy = float(K[5]) # principal point y in pixels

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            # 32FC1 means 32-bit float, single channel (depth in meters or normalized)
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

        flow_res = self.tracker.process(frame, msg.header.stamp)
        if flow_res is None:
            return

        # compute velocity from flow + depth (3-DOF least-squares)
        vx_mps, vy_mps, vz_mps, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old, flow_res.good_new, self.latest_depth, flow_res.dt
        )

        # Publish in front_cam_link frame (Xl=forward, Yl=left, Zl=up).
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(vx_mps)
        vel_msg.vector.y = float(vy_mps)
        vel_msg.vector.z = float(vz_mps)
        self.vel_pub.publish(vel_msg)

        # debug visualization of flow vectors colored by depth
        if self.show_debug:
            vis = frame.copy()
            self.draw_debug(vis, flow_res.good_old, flow_res.good_new, self.latest_depth)
            txt = f"vx={vx_mps:.3f} vy={vy_mps:.3f} vz={vz_mps:.3f} used={n_used}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1)


    # ---------- core helpers ----------

    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float):
        """
        Solve for the 3-DOF camera translational velocity in the optical frame
        using a least-squares formulation of the brightness-constancy equation.

        Full optical-flow model (per point i):
            du_i * Z_i = -fx * Vx_optical  +  u_c_i * Vz_optical
            dv_i * Z_i = -fy * Vy_optical  +  v_c_i * Vz_optical

        Written as A @ V = B with V = [Vx_optical, Vy_optical, Vz_optical]:
            A[2i,   :] = [-fx,   0,  u_c_i]
            A[2i+1, :] = [  0, -fy,  v_c_i]
            B[2i  ]    = du_i * Z_i
            B[2i+1]    = dv_i * Z_i

        The simplified formula (vz = Z*dv/fy) is a degenerate case that drops
        the u_c*Vz and v_c*Vz terms, contaminating lateral/vertical estimates
        with forward motion when features are not symmetric around the image centre.

        Output in front_cam_link frame (Xl=forward, Yl=left, Zl=up):
            Vx_link =  Vz_optical   (forward)
            Vy_link = -Vx_optical   (leftward)
            Vz_link = -Vy_optical   (upward)

        Returns (Vx_link, Vy_link, Vz_link, n_used).
        """
        MIN_POINTS = 8
        H, W = depth_map.shape[:2]

        du = (good_new[:, 0] - good_old[:, 0]) / dt  # px/s
        dv = (good_new[:, 1] - good_old[:, 1]) / dt  # px/s

        u_idx = np.rint(good_new[:, 0]).astype(np.int32)
        v_idx = np.rint(good_new[:, 1]).astype(np.int32)

        valid = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
        if int(np.sum(valid)) < MIN_POINTS:
            return 0.0, 0.0, 0.0, 0

        Z = np.zeros(len(du), dtype=np.float64)
        Z[valid] = depth_map[v_idx[valid], u_idx[valid]] * self.depth_scale  # Apply scale here

        valid = valid & np.isfinite(Z)
        if not self.use_depth_norm:
            valid = valid & (Z > self.min_depth) & (Z < self.max_depth)

        n = int(np.sum(valid))
        if n < MIN_POINTS:
            return 0.0, 0.0, 0.0, 0

        u_c = good_new[valid, 0].astype(np.float64) - self.cx
        v_c = good_new[valid, 1].astype(np.float64) - self.cy
        Zv  = Z[valid]
        duv = du[valid].astype(np.float64)
        dvv = dv[valid].astype(np.float64)

        # Build the 2N×3 linear system
        A = np.zeros((2 * n, 3), dtype=np.float64)
        B = np.zeros((2 * n,),   dtype=np.float64)
        A[0::2, 0] = -self.fx;  A[0::2, 2] = u_c;  B[0::2] = duv * Zv
        A[1::2, 1] = -self.fy;  A[1::2, 2] = v_c;  B[1::2] = dvv * Zv

        try:
            vel, *_ = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, n

        Vx_o, Vy_o, Vz_o = float(vel[0]), float(vel[1]), float(vel[2])

        # Rotate optical → front_cam_link
        Vx_link =  Vz_o   # forward
        Vy_link = -Vx_o   # leftward
        Vz_link = -Vy_o   # upward
        return Vx_link, Vy_link, Vz_link, n

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
