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


class FlowDepthEssentialVelocityNode(Node):
    """
    ROS2 node:
      - subscribes to RGB + Depth map + CameraInfo
      - tracks features via OpticalFlowTracker (LK)
      - estimates Essential Matrix (RANSAC) -> R, t_dir
      - estimates scale using depth (reprojection search) -> s
      - outputs translation velocity v = (s * t_dir) / dt  (m/s if depth metric)
    """

    def __init__(self):
        super().__init__("flow_depth_essential_velocity_node")

        # sim time
        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        # topics
        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")
        self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("output_topic", "/flow_depth/velocity")

        # tracking params
        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 30)
        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        # depth params
        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 50.0)
        self.declare_parameter("use_depth_norm", False)

        # essential matrix params
        self.declare_parameter("ransac_thresh_px", 1.5)      # pixel threshold
        self.declare_parameter("ransac_prob", 0.999)
        self.declare_parameter("min_inliers", 30)

        # scale search params (tune if needed)
        self.declare_parameter("scale_min", 0.0)
        self.declare_parameter("scale_max", 0.5)            # meters per frame (if metric depth)
        self.declare_parameter("scale_steps", 51)           # coarse search steps

        # debug
        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_link")

        # read params
        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        caminfo_topic = self.get_parameter("camera_info_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.max_corners = int(self.get_parameter("max_corners").value)
        self.min_corners = int(self.get_parameter("min_corners").value)
        lk_win = int(self.get_parameter("lk_win").value)
        lk_levels = int(self.get_parameter("lk_levels").value)

        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.use_depth_norm = bool(self.get_parameter("use_depth_norm").value)

        self.ransac_thresh_px = float(self.get_parameter("ransac_thresh_px").value)
        self.ransac_prob = float(self.get_parameter("ransac_prob").value)
        self.min_inliers = int(self.get_parameter("min_inliers").value)

        self.scale_min = float(self.get_parameter("scale_min").value)
        self.scale_max = float(self.get_parameter("scale_max").value)
        self.scale_steps = int(self.get_parameter("scale_steps").value)

        self.show_debug = bool(self.get_parameter("show_debug").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)

        # if depth is normalized, default scale search to [0..1] "relative units"
        if self.use_depth_norm:
            self.scale_min = 0.0
            self.scale_max = 1.0

        self.get_logger().info(f"[Essential] RGB: {image_topic}")
        self.get_logger().info(f"[Essential] Depth: {depth_topic}")
        self.get_logger().info(f"[Essential] CamInfo: {caminfo_topic}")
        self.get_logger().info(f"[Essential] Pub: {output_topic}")

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_stamp = None

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # tracker
        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

        # pubs/subs
        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, 10)
        self.rgb_sub = self.create_subscription(Image, image_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)

    # ---------------- callbacks ----------------
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
        if self.fx is None or self.fy is None:
            return
        if self.latest_depth is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        flow_res = self.tracker.process(frame, msg.header.stamp)
        if flow_res is None:
            return

        good_old = flow_res.good_old.astype(np.float64)
        good_new = flow_res.good_new.astype(np.float64)
        dt = float(flow_res.dt)
        if dt <= 0.0 or len(good_old) < self.min_inliers:
            return

        # Estimate R,t_dir via Essential Matrix (RANSAC)
        pose = self.estimate_pose_essential(good_old, good_new)
        if pose is None:
            return
        R, t_dir, inliers = pose

        # Estimate scale s using depth + reprojection
        s = self.estimate_scale_by_reprojection(R, t_dir, good_old[inliers], good_new[inliers], self.latest_depth)
        if s is None:
            return

        # translation (per frame), then velocity
        trans = s * t_dir  # (meters or relative units)
        v = trans / dt

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(v[0])
        vel_msg.vector.y = float(v[1])
        vel_msg.vector.z = float(v[2])
        self.vel_pub.publish(vel_msg)

        if self.show_debug:
            vis = frame.copy()
            txt = f"inliers={int(inliers.sum())} s={s:.3f} v=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            self.draw_inlier_flow(vis, good_old[inliers], good_new[inliers])
            cv2.imshow("Essential+Depth Velocity", vis)
            cv2.waitKey(1)

    # ---------------- core: Essential pose ----------------
    def K_mat(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def estimate_pose_essential(self, pts1: np.ndarray, pts2: np.ndarray):
        """
        pts1, pts2: [N,2] pixel coords (float64)
        Returns: (R [3,3], t_dir [3], inliers_mask [N] bool) or None
        """
        K = self.K_mat()
        
        # Estimate Essential Matrix via RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_thresh_px
        )
        if E is None or mask is None:
            return None

        inliers = mask.ravel().astype(bool)
        if int(inliers.sum()) < self.min_inliers:
            return None

        # Recover pose from Essential Matrix
        n_in, R, t, _ = cv2.recoverPose(E, pts1[inliers], pts2[inliers], K)
        if n_in < self.min_inliers:
            return None

        t_dir = t.reshape(3).astype(np.float64)

        # normalize direction for safety
        n = np.linalg.norm(t_dir)
        if n < 1e-9:
            return None
        t_dir /= n
        return R.astype(np.float64), t_dir, inliers

    # ---------------- core: scale from depth ----------------
    def estimate_scale_by_reprojection(self, R, t_dir, pts1, pts2, depth_map):
        """
        Estimate scalar s such that P2 ≈ R P1 + s t_dir minimizes reprojection error.
        - pts1, pts2: [N,2] inlier pixel coords
        - depth_map: [H,W] depth
        Returns s (float) or None
        """
        if len(pts1) < self.min_inliers:
            return None

        H, W = depth_map.shape[:2]

        # sample depth at pts1 (you can also sample at pts2; pts1 is typical)
        u1 = np.rint(pts1[:, 0]).astype(np.int32)
        v1 = np.rint(pts1[:, 1]).astype(np.int32)

        valid = (u1 >= 0) & (u1 < W) & (v1 >= 0) & (v1 < H)
        if not np.any(valid):
            return None

        Z = np.zeros(len(pts1), dtype=np.float64)
        Z[valid] = depth_map[v1[valid], u1[valid]]

        valid = valid & np.isfinite(Z)
        if not self.use_depth_norm:
            valid = valid & (Z > self.min_depth) & (Z < self.max_depth)

        pts1 = pts1[valid]
        pts2 = pts2[valid]
        Z = Z[valid]

        if len(pts1) < self.min_inliers:
            return None

        # backproject pts1 -> 3D (camera1 frame)
        X = (pts1[:, 0] - self.cx) * Z / self.fx
        Y = (pts1[:, 1] - self.cy) * Z / self.fy
        P1 = np.stack([X, Y, Z], axis=1)  # [N,3]

        # precompute rotated points
        RP1 = (R @ P1.T).T  # [N,3]

        # search scale
        scales = np.linspace(self.scale_min, self.scale_max, self.scale_steps, dtype=np.float64)

        best_s = None
        best_err = np.inf

        for s in scales:
            P2 = RP1 + (s * t_dir.reshape(1, 3))  # [N,3]

            # reject points that go behind camera
            z2 = P2[:, 2]
            ok = z2 > 1e-6
            if ok.sum() < self.min_inliers:
                continue

            u_proj = (P2[ok, 0] * self.fx / z2[ok]) + self.cx
            v_proj = (P2[ok, 1] * self.fy / z2[ok]) + self.cy

            # reprojection error in pixels
            err = (u_proj - pts2[ok, 0])**2 + (v_proj - pts2[ok, 1])**2
            med = float(np.median(err))
            if med < best_err:
                best_err = med
                best_s = float(s)

        # sanity: if reprojection still huge, skip update
        if best_s is None or best_err > (10.0**2):  # median > 10px^2 -> tune if needed
            return None

        return best_s

    # ---------------- debug drawing ----------------
    def draw_inlier_flow(self, vis_bgr, pts1, pts2):
        for (u1, v1), (u2, v2) in zip(pts1, pts2):
            x1, y1 = int(round(u1)), int(round(v1))
            x2, y2 = int(round(u2)), int(round(v2))
            cv2.arrowedLine(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2, tipLength=0.25)


def main():
    rclpy.init()
    node = FlowDepthEssentialVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
