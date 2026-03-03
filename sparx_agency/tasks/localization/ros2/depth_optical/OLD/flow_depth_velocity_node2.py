#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped, Pose
from cv_bridge import CvBridge

from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker


@dataclass
class DepthFrame:
    stamp: Time
    depth: np.ndarray  # float32 HxW


@dataclass
class PoseFrame:
    stamp: Time
    p: np.ndarray  # (3,) float64


def stamp_to_time(stamp) -> Time:
    return Time(seconds=float(stamp.sec) + float(stamp.nanosec) * 1e-9)


def time_diff_sec(a: Time, b: Time) -> float:
    return abs((a - b).nanoseconds) * 1e-9


def robust_median_mad(values: np.ndarray, z_thresh: float = 3.5) -> Tuple[float, int]:
    if values.size == 0:
        return 0.0, 0
    med = np.median(values)
    abs_dev = np.abs(values - med)
    mad = np.median(abs_dev) + 1e-9
    z = 0.6745 * (values - med) / mad
    keep = np.abs(z) <= z_thresh
    if not np.any(keep):
        return float(med), int(values.size)
    return float(np.median(values[keep])), int(np.sum(keep))


class FlowDepthVelocityNode(Node):
    """
    - RGB + depth + caminfo
    - LK optical flow
    - Solve translation-only model (vx, vy, vz) in camera optical frame (x right, y down, z forward)
    - Optional: calibrate depth scale using /simple_drone/gt_pose (Pose without header -> use receipt time)
    """

    def __init__(self):
        super().__init__("flow_depth_velocity_node")

        # sim time
        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        # topics
        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")
        self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("gt_pose_topic", "/simple_drone/gt_pose")
        self.declare_parameter("output_topic", "/flow_depth/velocity")

        # feature/LK params
        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 30)
        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        # depth params
        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 50.0)
        self.declare_parameter("use_depth_norm", False)

        # base depth_scale (static) + dynamic GT scale
        self.declare_parameter("depth_scale", 1.0)
        self.declare_parameter("enable_gt_scale", True)
        self.declare_parameter("gt_pose_buffer_size", 200)
        self.declare_parameter("scale_window", 50)           # keep last N alpha values
        self.declare_parameter("min_gt_dt", 1e-3)
        self.declare_parameter("max_gt_dt", 0.2)
        self.declare_parameter("min_speed_for_scale", 0.02)  # m/s ignore near-static

        # sync & robustness params
        self.declare_parameter("depth_buffer_size", 30)
        self.declare_parameter("max_depth_time_diff", 1.00)
        self.declare_parameter("min_dt", 1e-3)
        self.declare_parameter("max_dt", 0.2)
        self.declare_parameter("max_flow_px_per_s", 1500.0)
        self.declare_parameter("mad_z_thresh", 3.5)

        # debug
        self.declare_parameter("show_debug", False)
        self.declare_parameter("log_every_n", 30)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_optical_frame")

        # read params
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.caminfo_topic = str(self.get_parameter("camera_info_topic").value)
        self.gt_pose_topic = str(self.get_parameter("gt_pose_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.max_corners = int(self.get_parameter("max_corners").value)
        self.min_corners = int(self.get_parameter("min_corners").value)
        self.lk_win = int(self.get_parameter("lk_win").value)
        self.lk_levels = int(self.get_parameter("lk_levels").value)

        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.use_depth_norm = bool(self.get_parameter("use_depth_norm").value)

        self.depth_buffer_size = int(self.get_parameter("depth_buffer_size").value)
        self.max_depth_time_diff = float(self.get_parameter("max_depth_time_diff").value)

        self.min_dt = float(self.get_parameter("min_dt").value)
        self.max_dt = float(self.get_parameter("max_dt").value)
        self.max_flow_px_per_s = float(self.get_parameter("max_flow_px_per_s").value)
        self.mad_z_thresh = float(self.get_parameter("mad_z_thresh").value)

        self.show_debug = bool(self.get_parameter("show_debug").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)

        self.base_depth_scale = float(self.get_parameter("depth_scale").value)

        self.enable_gt_scale = bool(self.get_parameter("enable_gt_scale").value)
        self.gt_pose_buffer_size = int(self.get_parameter("gt_pose_buffer_size").value)
        self.scale_window = int(self.get_parameter("scale_window").value)
        self.min_gt_dt = float(self.get_parameter("min_gt_dt").value)
        self.max_gt_dt = float(self.get_parameter("max_gt_dt").value)
        self.min_speed_for_scale = float(self.get_parameter("min_speed_for_scale").value)

        # current dynamic scale (starts at base)
        self.depth_scale_dyn = self.base_depth_scale
        self.alpha_hist: Deque[float] = deque(maxlen=max(1, self.scale_window))

        # log
        self.get_logger().info(f"[FlowDepth] RGB: {self.image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {self.depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {self.caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] GT Pose: {self.gt_pose_topic} (Pose no header -> using receipt time)")
        self.get_logger().info(f"[FlowDepth] Pub: {self.output_topic}")
        self.get_logger().info(f"[FlowDepth] enable_gt_scale={self.enable_gt_scale} scale_window={self.scale_window}")

        self.bridge = CvBridge()

        # camera intrinsics
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # buffers
        self.depth_buf: Deque[DepthFrame] = deque(maxlen=self.depth_buffer_size)
        self.gt_pose_buf: Deque[PoseFrame] = deque(maxlen=self.gt_pose_buffer_size)

        # tracker
        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=self.lk_win,
            lk_levels=self.lk_levels,
        )

        # pubs/subs
        self.vel_pub = self.create_publisher(Vector3Stamped, self.output_topic, 10)

        self.rgb_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.caminfo_sub = self.create_subscription(CameraInfo, self.caminfo_topic, self.caminfo_callback, qos_profile_sensor_data)

        # GT pose has no header -> QoS not critical
        self.gt_pose_sub = self.create_subscription(Pose, self.gt_pose_topic, self.gt_pose_callback, 10)

        self._dbg_counter = 0

    # ---------- callbacks ----------
    def caminfo_callback(self, msg: CameraInfo):
        K = msg.k
        self.fx = float(K[0])
        self.fy = float(K[4])
        self.cx = float(K[2])
        self.cy = float(K[5])

    def gt_pose_callback(self, msg: Pose):
        # Pose has no stamp -> take receipt time (sim time)
        t = self.get_clock().now()
        p = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float64)
        self.gt_pose_buf.append(PoseFrame(stamp=t, p=p))

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return
        depth_np = np.asarray(depth, dtype=np.float32)
        t = stamp_to_time(msg.header.stamp)
        self.depth_buf.append(DepthFrame(stamp=t, depth=depth_np))

    def rgb_callback(self, msg: Image):
        if self.fx is None or self.fy is None:
            return
        if len(self.depth_buf) == 0:
            return

        t_rgb = stamp_to_time(msg.header.stamp)
        depth_map, t_depth, dt_depth = self.get_depth_closest_to(t_rgb)
        if depth_map is None:
            return

        if dt_depth > self.max_depth_time_diff:
            if (self._dbg_counter % self.log_every_n) == 0:
                self.get_logger().warn(f"[FlowDepth] depth too old/new: |t_rgb - t_depth| = {dt_depth:.3f}s (skip)")
            return

        # Apply current dynamic scale
        depth_map = (self.depth_scale_dyn * depth_map).astype(np.float32)

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        flow_res = self.tracker.process(frame, msg.header.stamp)
        if flow_res is None:
            return

        dt = float(flow_res.dt)
        if not (self.min_dt <= dt <= self.max_dt):
            if (self._dbg_counter % self.log_every_n) == 0:
                self.get_logger().warn(f"[FlowDepth] bad dt={dt:.6f}s (skip)")
            return

        vx, vy, vz, n_used, n_kept = self.velocity_from_flow_and_depth(
            flow_res.good_old,
            flow_res.good_new,
            depth_map,
            dt=dt,
        )

        # Optionally update scale from GT
        v_est_norm = float(np.linalg.norm([vx, vy, vz]))
        v_gt = self.try_get_gt_speed_now()

        if self.enable_gt_scale and (v_gt is not None) and (v_est_norm > 1e-6) and (v_gt >= self.min_speed_for_scale):
            alpha = float(v_gt / v_est_norm)

            # reject crazy alphas
            if np.isfinite(alpha) and (0.01 <= alpha <= 100.0):
                self.alpha_hist.append(alpha)
                self.depth_scale_dyn = self.base_depth_scale * float(np.median(np.array(self.alpha_hist, dtype=np.float64)))

        # Publish velocity (camera optical frame)
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(vx)
        vel_msg.vector.y = float(vy)
        vel_msg.vector.z = float(vz)
        self.vel_pub.publish(vel_msg)

        if (self._dbg_counter % self.log_every_n) == 0:
            self.get_logger().info(
                f"[FlowDepth] used={n_used} kept={n_kept} "
                f"v_est=({vx:.3f},{vy:.3f},{vz:.3f}) |v|={v_est_norm:.3f} "
                f"v_gt={(-1.0 if v_gt is None else v_gt):.3f} "
                f"depth_scale_dyn={self.depth_scale_dyn:.4f} "
                f"|rgb-depth|={dt_depth:.3f}s"
            )

        self._dbg_counter += 1

        if self.show_debug:
            vis = frame.copy()
            self.draw_debug(vis, flow_res.good_old, flow_res.good_new, depth_map)
            txt = f"v=({vx:.2f},{vy:.2f},{vz:.2f}) scale={self.depth_scale_dyn:.3f} dTD={dt_depth:.3f}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1)

    # ---------- GT speed from pose buffer ----------
    def try_get_gt_speed_now(self) -> Optional[float]:
        """
        gt_pose is Pose without header.
        We compute speed from last two received poses using receipt time (sim time).
        """
        if len(self.gt_pose_buf) < 2:
            return None
        a = self.gt_pose_buf[-2]
        b = self.gt_pose_buf[-1]
        dt = time_diff_sec(a.stamp, b.stamp)
        if not (self.min_gt_dt <= dt <= self.max_gt_dt):
            return None
        dp = b.p - a.p
        v = float(np.linalg.norm(dp) / dt)
        return v

    # ---------- sync helper ----------
    def get_depth_closest_to(self, t_rgb: Time) -> Tuple[Optional[np.ndarray], Optional[Time], float]:
        if len(self.depth_buf) == 0:
            return None, None, float("inf")
        best = None
        best_dt = float("inf")
        for df in self.depth_buf:
            d = time_diff_sec(df.stamp, t_rgb)
            if d < best_dt:
                best_dt = d
                best = df
        return best.depth if best else None, best.stamp if best else None, best_dt

    # ---------- core helpers ----------
    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float):
        if good_old is None or good_new is None or good_old.shape[0] < 8:
            return 0.0, 0.0, 0.0, 0, 0

        H, W = depth_map.shape[:2]

        u_new = good_new[:, 0]
        v_new = good_new[:, 1]

        du = (good_new[:, 0] - good_old[:, 0]) / dt
        dv = (good_new[:, 1] - good_old[:, 1]) / dt

        # reject insane flow
        flow_mag = np.sqrt(du * du + dv * dv)
        keep_flow = np.isfinite(flow_mag) & (flow_mag < self.max_flow_px_per_s)
        if not np.any(keep_flow):
            return 0.0, 0.0, 0.0, 0, 0

        u_idx = np.rint(u_new).astype(np.int32)
        v_idx = np.rint(v_new).astype(np.int32)

        valid = keep_flow & (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0, 0

        Z = depth_map[v_idx[valid], u_idx[valid]].astype(np.float64)

        validZ = np.isfinite(Z)
        if not self.use_depth_norm:
            validZ = validZ & (Z > self.min_depth) & (Z < self.max_depth)

        if int(np.sum(validZ)) < 8:
            return 0.0, 0.0, 0.0, int(np.sum(valid)), 0

        curr_u = u_new[valid][validZ].astype(np.float64)
        curr_v = v_new[valid][validZ].astype(np.float64)
        curr_du = du[valid][validZ].astype(np.float64)
        curr_dv = dv[valid][validZ].astype(np.float64)
        curr_Z = Z[validZ]

        n = curr_Z.shape[0]
        A = np.zeros((2 * n, 3), dtype=np.float64)
        B = np.zeros((2 * n, 1), dtype=np.float64)

        # du * Z = -fx*vx + (u-cx)*vz
        # dv * Z = -fy*vy + (v-cy)*vz
        A[0::2, 0] = -self.fx
        A[0::2, 2] = (curr_u - self.cx)
        B[0::2, 0] = curr_du * curr_Z

        A[1::2, 1] = -self.fy
        A[1::2, 2] = (curr_v - self.cy)
        B[1::2, 0] = curr_dv * curr_Z

        try:
            vel, *_ = np.linalg.lstsq(A, B, rcond=None)
            vx, vy, vz = vel.flatten()
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, n, 0

        return float(vx), float(vy), float(vz), n, n

    def draw_debug(self, vis_bgr, good_old, good_new, depth_map):
        H, W = depth_map.shape[:2]
        for (u2, v2), (u1, v1) in zip(good_new, good_old):
            x2, y2 = int(round(u2)), int(round(v2))
            x1, y1 = int(round(u1)), int(round(v1))
            if not (0 <= x2 < W and 0 <= y2 < H):
                continue
            z = float(depth_map[y2, x2])
            if not np.isfinite(z):
                continue
            cv2.arrowedLine(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2, tipLength=0.3)


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
