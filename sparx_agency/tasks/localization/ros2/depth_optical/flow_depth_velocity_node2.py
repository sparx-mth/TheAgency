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
from geometry_msgs.msg import Vector3Stamped, Twist
from cv_bridge import CvBridge

from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker


@dataclass
class DepthFrame:
    stamp: Time
    depth: np.ndarray  # float32 HxW


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
    Flow+Depth velocity (no Essential) + auto depth-scale calibration using GT velocity magnitude.

    Outputs Vector3Stamped in `camera_frame` (optical convention):
      x = right, y = down, z = forward

    Calibration:
      - Reads /simple_drone/gt_vel (geometry_msgs/msg/Twist)
      - Computes alpha = median( |v_gt| / |v_depth_raw| ) over a sliding window
      - Applies alpha to depth (equivalently scales velocity)
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
        self.declare_parameter("output_topic", "/flow_depth/velocity")

        # GT velocity topic (Twist)
        self.declare_parameter("gt_vel_topic", "/simple_drone/gt_vel")

        # feature/LK params
        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 30)
        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        # depth params
        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 50.0)
        self.declare_parameter("use_depth_norm", False)
        self.declare_parameter("depth_scale_init", 1.0)   # initial alpha

        # sync & robustness params
        self.declare_parameter("depth_buffer_size", 30)
        self.declare_parameter("max_depth_time_diff", 1.00)
        self.declare_parameter("min_dt", 1e-3)
        self.declare_parameter("max_dt", 0.2)
        self.declare_parameter("max_flow_px_per_s", 1500.0)
        self.declare_parameter("mad_z_thresh", 3.5)

        # Calibration params
        self.declare_parameter("enable_gt_calib", True)
        self.declare_parameter("gt_max_time_diff", 0.25)        # sec (rgb vs gt)
        self.declare_parameter("calib_window", 120)             # number of alpha samples
        self.declare_parameter("calib_min_samples", 30)         # before "locking in"
        self.declare_parameter("calib_vmin", 0.05)              # m/s ignore too small GT
        self.declare_parameter("calib_vmax", 5.0)               # m/s ignore crazy GT
        self.declare_parameter("calib_vdepth_min", 1e-3)        # ignore near-zero estimate
        self.declare_parameter("alpha_clip_min", 0.01)
        self.declare_parameter("alpha_clip_max", 100.0)

        # debug
        self.declare_parameter("show_debug", False)
        self.declare_parameter("log_every_n", 30)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_optical_frame")

        # read params
        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.caminfo_topic = self.get_parameter("camera_info_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.gt_vel_topic = self.get_parameter("gt_vel_topic").value

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

        self.enable_gt_calib = bool(self.get_parameter("enable_gt_calib").value)
        self.gt_max_time_diff = float(self.get_parameter("gt_max_time_diff").value)
        self.calib_window = int(self.get_parameter("calib_window").value)
        self.calib_min_samples = int(self.get_parameter("calib_min_samples").value)
        self.calib_vmin = float(self.get_parameter("calib_vmin").value)
        self.calib_vmax = float(self.get_parameter("calib_vmax").value)
        self.calib_vdepth_min = float(self.get_parameter("calib_vdepth_min").value)
        self.alpha_clip_min = float(self.get_parameter("alpha_clip_min").value)
        self.alpha_clip_max = float(self.get_parameter("alpha_clip_max").value)

        self.show_debug = bool(self.get_parameter("show_debug").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)
        self.camera_frame = self.get_parameter("camera_frame").value

        # depth scale (alpha)
        self.alpha = float(self.get_parameter("depth_scale_init").value)
        self.alpha_buf: Deque[float] = deque(maxlen=self.calib_window)

        # log
        self.get_logger().info(f"[FlowDepth] RGB: {self.image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {self.depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {self.caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] GT vel: {self.gt_vel_topic} (Twist)")
        self.get_logger().info(f"[FlowDepth] Pub: {self.output_topic}")
        self.get_logger().info(f"[FlowDepth] depth_buf={self.depth_buffer_size} max_depth_dt={self.max_depth_time_diff}s")
        self.get_logger().info(f"[FlowDepth] alpha_init={self.alpha:.6f} enable_gt_calib={self.enable_gt_calib}")

        self.bridge = CvBridge()

        # camera intrinsics
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # depth buffer
        self.depth_buf: Deque[DepthFrame] = deque(maxlen=self.depth_buffer_size)

        # latest gt velocity
        self.gt_vel_vec: Optional[np.ndarray] = None  # (3,)
        self.gt_vel_stamp: Optional[Time] = None

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

        if self.enable_gt_calib:
            self.gt_sub = self.create_subscription(Twist, self.gt_vel_topic, self.gt_vel_callback, qos_profile_sensor_data)

        self._dbg_counter = 0

    # ---------- callbacks ----------
    def caminfo_callback(self, msg: CameraInfo):
        K = msg.k
        self.fx = float(K[0])
        self.fy = float(K[4])
        self.cx = float(K[2])
        self.cy = float(K[5])

    def gt_vel_callback(self, msg: Twist):
        self.gt_vel_vec = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z],
            dtype=np.float64
        )
        # Twist has no header stamp; use node time as "best effort"
        # In sim/bag this is still okay if callbacks are roughly in time.
        self.gt_vel_stamp = self.get_clock().now()

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

        # Apply current alpha to depth (turn "relative depth" to metric-like)
        depth_map = (self.alpha * depth_map).astype(np.float32, copy=False)

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

        # --- update alpha from GT (magnitude) ---
        if self.enable_gt_calib:
            self.try_update_alpha_from_gt(vx, vy, vz)

        # publish (camera optical frame convention)
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(vx)
        vel_msg.vector.y = float(vy)
        vel_msg.vector.z = float(vz)
        self.vel_pub.publish(vel_msg)

        if (self._dbg_counter % self.log_every_n) == 0:
            vnorm = float(np.linalg.norm([vx, vy, vz]))
            alpha_n = len(self.alpha_buf)
            self.get_logger().info(
                f"[FlowDepth] used={n_used} kept={n_kept} "
                f"v=({vx:.3f},{vy:.3f},{vz:.3f}) |v|={vnorm:.3f} "
                f"dt={dt:.3f}s |rgb-depth|={dt_depth:.3f}s alpha={self.alpha:.5f} (n={alpha_n})"
            )

        self._dbg_counter += 1

        if self.show_debug:
            vis = frame.copy()
            self.draw_debug(vis, flow_res.good_old, flow_res.good_new, depth_map)
            txt = f"v=({vx:.3f},{vy:.3f},{vz:.3f}) alpha={self.alpha:.3f}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1)

    # ---------- calibration helper ----------
    def try_update_alpha_from_gt(self, vx: float, vy: float, vz: float):
        if self.gt_vel_vec is None:
            return
        if self.gt_vel_stamp is None:
            return

        # Twist has no stamp, so we can't do exact time sync; best-effort gate:
        # if callbacks drift a lot, this still stays roughly okay.
        # You can disable this gate by setting gt_max_time_diff very large.
        dt_gt = abs((self.get_clock().now() - self.gt_vel_stamp).nanoseconds) * 1e-9
        if dt_gt > self.gt_max_time_diff:
            return

        vgt = float(np.linalg.norm(self.gt_vel_vec))
        vdepth = float(np.linalg.norm([vx, vy, vz]))

        if not (self.calib_vmin <= vgt <= self.calib_vmax):
            return
        if vdepth < self.calib_vdepth_min:
            return

        alpha_i = vgt / vdepth
        if not np.isfinite(alpha_i):
            return

        alpha_i = float(np.clip(alpha_i, self.alpha_clip_min, self.alpha_clip_max))
        self.alpha_buf.append(alpha_i)

        if len(self.alpha_buf) >= self.calib_min_samples:
            new_alpha = float(np.median(np.array(self.alpha_buf, dtype=np.float64)))
            if np.isfinite(new_alpha) and new_alpha > 0:
                self.alpha = new_alpha

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
        """
        Solve least squares for (vx, vy, vz) from optical flow constraints:
          du * Z = -fx * vx + (u-cx) * vz
          dv * Z = -fy * vy + (v-cy) * vz

        Returns vx, vy, vz in *camera optical convention* (x right, y down, z forward).
        """
        if good_old is None or good_new is None or good_old.shape[0] < 8:
            return 0.0, 0.0, 0.0, 0, 0

        H, W = depth_map.shape[:2]
        u_new = good_new[:, 0]
        v_new = good_new[:, 1]

        du = (good_new[:, 0] - good_old[:, 0]) / dt
        dv = (good_new[:, 1] - good_old[:, 1]) / dt

        # reject insane flow early
        flow_mag = np.sqrt(du * du + dv * dv)
        keep_flow = np.isfinite(flow_mag) & (flow_mag < self.max_flow_px_per_s)

        u_idx = np.rint(u_new).astype(np.int32)
        v_idx = np.rint(v_new).astype(np.int32)

        valid = keep_flow & (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0, 0

        Z = depth_map[v_idx[valid], u_idx[valid]].astype(np.float64)

        keep = np.isfinite(Z)
        if not self.use_depth_norm:
            keep = keep & (Z > self.min_depth) & (Z < self.max_depth)

        if np.sum(keep) < 8:
            return 0.0, 0.0, 0.0, int(np.sum(valid)), int(np.sum(keep))

        curr_u = u_new[valid][keep].astype(np.float64)
        curr_v = v_new[valid][keep].astype(np.float64)
        curr_du = du[valid][keep].astype(np.float64)
        curr_dv = dv[valid][keep].astype(np.float64)
        curr_Z = Z[keep].astype(np.float64)

        n = int(curr_Z.size)

        A = np.zeros((2 * n, 3), dtype=np.float64)
        B = np.zeros((2 * n, 1), dtype=np.float64)

        # fill A,B vectorized (faster + less bugs)
        # row 0..n-1 for du
        A[0:n, 0] = -self.fx
        A[0:n, 1] = 0.0
        A[0:n, 2] = (curr_u - self.cx)
        B[0:n, 0] = curr_du * curr_Z

        # row n..2n-1 for dv
        A[n:2*n, 0] = 0.0
        A[n:2*n, 1] = -self.fy
        A[n:2*n, 2] = (curr_v - self.cy)
        B[n:2*n, 0] = curr_dv * curr_Z

        try:
            vel, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
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
