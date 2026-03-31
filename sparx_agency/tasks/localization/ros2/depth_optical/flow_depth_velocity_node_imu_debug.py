#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sparx_agency.tasks.localization.common.optical_flow_tracker import OpticalFlowTracker


class FlowDepthVelocityNode(Node):

    def __init__(self):
        super().__init__("flow_depth_velocity_node")

        # sim time
        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time",
                rclpy.parameter.Parameter.Type.BOOL,
                True
            )
        ])

        # ---------------- Parameters ----------------
        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")
        self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("imu_topic", "/simple_drone/imu/out")
        self.declare_parameter("output_topic", "/flow_depth/velocity")

        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 30)

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 30.0)
        self.declare_parameter("use_depth_norm", False)
        self.declare_parameter("depth_scale", 0.4)

        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_link")

        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        self.declare_parameter("log_every_n_frames", 10)
        self.declare_parameter("min_ls_points", 8)

        # message_filters sync params
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop_sec", 0.03)  # 30 ms

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.max_corners = int(self.get_parameter("max_corners").get_parameter_value().integer_value)
        self.min_corners = int(self.get_parameter("min_corners").get_parameter_value().integer_value)

        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)
        self.use_depth_norm = bool(self.get_parameter("use_depth_norm").get_parameter_value().bool_value)
        self.depth_scale = float(self.get_parameter("depth_scale").get_parameter_value().double_value)

        self.show_debug = bool(self.get_parameter("show_debug").get_parameter_value().bool_value)
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        lk_win = int(self.get_parameter("lk_win").get_parameter_value().integer_value)
        lk_levels = int(self.get_parameter("lk_levels").get_parameter_value().integer_value)

        self.log_every_n_frames = int(
            self.get_parameter("log_every_n_frames").get_parameter_value().integer_value
        )
        self.min_ls_points = int(
            self.get_parameter("min_ls_points").get_parameter_value().integer_value
        )

        self.sync_queue_size = int(
            self.get_parameter("sync_queue_size").get_parameter_value().integer_value
        )
        self.sync_slop_sec = float(
            self.get_parameter("sync_slop_sec").get_parameter_value().double_value
        )

        # ---------------- Logs ----------------
        self.get_logger().info(f"[FlowDepth] RGB: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] IMU: {imu_topic}")
        self.get_logger().info(f"[FlowDepth] Pub: {output_topic}")
        self.get_logger().info(
            f"[FlowDepth] Sync queue={self.sync_queue_size}, slop={self.sync_slop_sec:.3f}s"
        )

        # ---------------- State ----------------
        self.bridge = CvBridge()

        self.frame_counter = 0

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # last IMU sample
        self.latest_gyro = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.latest_imu_stamp = None

        # tracker
        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

        # publisher
        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, 10)

        # regular ROS subscriptions
        self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 50)

        # message_filters subscribers for synchronized RGB+Depth
        self.rgb_sub_sync = Subscriber(self, Image, image_topic)
        self.depth_sub_sync = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub_sync, self.depth_sub_sync],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

    # ---------- helpers ----------
    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def should_log(self) -> bool:
        return self.frame_counter % max(1, self.log_every_n_frames) == 0

    # ---------- callbacks ----------
    def caminfo_callback(self, msg: CameraInfo):
        K = msg.k
        self.fx = float(K[0])
        self.fy = float(K[4])
        self.cx = float(K[2])
        self.cy = float(K[5])

        if not hasattr(self, "_caminfo_logged"):
            self.get_logger().info(
                f"[CamInfo] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            self.get_logger().info(f"[CamInfo] K matrix: {list(K)}")
            self._caminfo_logged = True

    def imu_callback(self, msg: Imu):
        roll_rate = msg.angular_velocity.x
        pitch_rate = msg.angular_velocity.y
        yaw_rate = msg.angular_velocity.z

        # keep the same mapping you had before
        self.latest_gyro[0] = -pitch_rate
        self.latest_gyro[1] = -yaw_rate
        self.latest_gyro[2] = roll_rate
        self.latest_imu_stamp = msg.header.stamp

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None or self.fy is None:
            return

        self.frame_counter += 1

        rgb_t = self.stamp_to_sec(rgb_msg.header.stamp)
        depth_t = self.stamp_to_sec(depth_msg.header.stamp)
        depth_age_ms = (rgb_t - depth_t) * 1000.0

        if self.latest_imu_stamp is not None:
            imu_t = self.stamp_to_sec(self.latest_imu_stamp)
            imu_age_ms = (rgb_t - imu_t) * 1000.0
            gyro_for_frame = self.latest_gyro.copy()
        else:
            imu_age_ms = float("nan")
            gyro_for_frame = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        if self.should_log():
            self.get_logger().info(
                f"[SYNC] frame={self.frame_counter} rgb_t={rgb_t:.6f} "
                f"depth_age_ms={depth_age_ms:.2f} imu_age_ms={imu_age_ms:.2f}"
            )
            if self.latest_imu_stamp is None:
                self.get_logger().warn("[SYNC] No IMU received yet, using zero gyro")

        # convert RGB
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        # convert Depth
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            depth_map = np.asarray(depth, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        flow_res = self.tracker.process(frame, rgb_msg.header.stamp)
        if flow_res is None:
            return

        if self.should_log():
            self.get_logger().info(
                f"[FLOW] frame={self.frame_counter} tracked_points={flow_res.n_used} dt={flow_res.dt:.4f}s"
            )

        vx_mps, vy_mps, vz_mps, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old,
            flow_res.good_new,
            depth_map,
            flow_res.dt,
            gyro_for_frame,
        )

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = rgb_msg.header.stamp
        vel_msg.header.frame_id = self.camera_frame
        vel_msg.vector.x = float(vx_mps)
        vel_msg.vector.y = float(vy_mps)
        vel_msg.vector.z = float(vz_mps)
        self.vel_pub.publish(vel_msg)

        if self.show_debug:
            vis = frame.copy()
            self.draw_debug(vis, flow_res.good_old, flow_res.good_new, depth_map)
            txt = f"vx={vx_mps:.3f} vy={vy_mps:.3f} vz={vz_mps:.3f} used={n_used}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1)

    # ---------- core helpers ----------
    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float, gyro: np.ndarray):
        H, W = depth_map.shape[:2]

        # total pixel velocities
        du_total = (good_new[:, 0] - good_old[:, 0]) / dt
        dv_total = (good_new[:, 1] - good_old[:, 1]) / dt

        n_total = len(good_old)
        flow_mag_total = np.sqrt(du_total**2 + dv_total**2)
        flow_mag_med = float(np.median(flow_mag_total)) if n_total > 0 else 0.0
        flow_mag_p90 = float(np.percentile(flow_mag_total, 90)) if n_total > 0 else 0.0

        # rotational compensation from gyro
        wx, wy, wz = gyro

        u = good_old[:, 0]
        v = good_old[:, 1]

        u_c = u - self.cx
        v_c = v - self.cy

        du_rot = (u_c * v_c / self.fx) * wx - (self.fx + u_c**2 / self.fx) * wy + v_c * wz
        dv_rot = (self.fy + v_c**2 / self.fy) * wx - (u_c * v_c / self.fy) * wy - u_c * wz

        du_trans = du_total - du_rot
        dv_trans = dv_total - dv_rot

        # sample depth at new point locations
        u_int = np.rint(good_new[:, 0]).astype(np.int32)
        v_int = np.rint(good_new[:, 1]).astype(np.int32)

        valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        n_in_bounds = int(np.sum(valid_bounds))
        if not np.any(valid_bounds):
            return 0.0, 0.0, 0.0, 0

        Z = np.zeros_like(du_total, dtype=np.float32)
        Z[valid_bounds] = depth_map[v_int[valid_bounds], u_int[valid_bounds]] * self.depth_scale

        valid = valid_bounds & np.isfinite(Z) & (Z > self.min_depth) & (Z < self.max_depth)
        n_depth_valid = int(np.sum(valid))

        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0

        nv = int(np.sum(valid))
        if nv < self.min_ls_points:
            if self.should_log():
                self.get_logger().info(
                    f"[QUALITY] n_total={n_total} in_bounds={n_in_bounds} "
                    f"depth_valid={n_depth_valid} ls_points={nv} < min_ls_points={self.min_ls_points}"
                )
            return 0.0, 0.0, 0.0, 0

        u_c_v = good_old[valid, 0].astype(np.float64) - self.cx
        v_c_v = good_old[valid, 1].astype(np.float64) - self.cy
        Zv = Z[valid].astype(np.float64)
        du_v = du_trans[valid].astype(np.float64)
        dv_v = dv_trans[valid].astype(np.float64)

        z_med = float(np.median(Zv)) if nv > 0 else 0.0
        z_min = float(np.min(Zv)) if nv > 0 else 0.0
        z_max = float(np.max(Zv)) if nv > 0 else 0.0

        A = np.zeros((2 * nv, 3), dtype=np.float64)
        B = np.zeros((2 * nv,), dtype=np.float64)

        A[0::2, 0] = -self.fx
        A[0::2, 2] = u_c_v
        B[0::2] = du_v * Zv

        A[1::2, 1] = -self.fy
        A[1::2, 2] = v_c_v
        B[1::2] = dv_v * Zv

        try:
            vel, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            self.get_logger().warn("[LS] np.linalg.lstsq failed")
            return 0.0, 0.0, 0.0, nv

        pred = A @ vel
        res = B - pred
        rmse_res = float(np.sqrt(np.mean(res**2))) if len(res) > 0 else 0.0
        cond_A = float(np.linalg.cond(A)) if A.shape[0] >= A.shape[1] else float("inf")
        s_min = float(np.min(s)) if len(s) > 0 else 0.0
        s_max = float(np.max(s)) if len(s) > 0 else 0.0

        Vx_o, Vy_o, Vz_o = float(vel[0]), float(vel[1]), float(vel[2])

        du_total_mean = np.mean(du_total[valid]) if nv > 0 else 0.0
        dv_total_mean = np.mean(dv_total[valid]) if nv > 0 else 0.0
        du_rot_mean = np.mean(du_rot[valid]) if nv > 0 else 0.0
        dv_rot_mean = np.mean(dv_rot[valid]) if nv > 0 else 0.0
        du_trans_mean = np.mean(du_trans[valid]) if nv > 0 else 0.0
        dv_trans_mean = np.mean(dv_trans[valid]) if nv > 0 else 0.0

        if self.should_log():
            self.get_logger().info(
                f"[QUALITY] n_total={n_total} in_bounds={n_in_bounds} depth_valid={n_depth_valid} "
                f"ls_points={nv} flow_med={flow_mag_med:.3f} flow_p90={flow_mag_p90:.3f} "
                f"z_med={z_med:.3f} z_range=[{z_min:.3f},{z_max:.3f}]"
            )

            self.get_logger().info(
                f"[ROT] total=({du_total_mean:.3f},{dv_total_mean:.3f}) "
                f"rot=({du_rot_mean:.3f},{dv_rot_mean:.3f}) "
                f"trans=({du_trans_mean:.3f},{dv_trans_mean:.3f}) "
                f"gyro=({gyro[0]:.4f},{gyro[1]:.4f},{gyro[2]:.4f})"
            )

            self.get_logger().info(
                f"[LS] rank={rank} cond={cond_A:.2e} rmse_res={rmse_res:.4f} "
                f"s_min={s_min:.4e} s_max={s_max:.4e}"
            )

            self.get_logger().info(
                f"[SUMMARY] frame={self.frame_counter} "
                f"tracked={n_total} ls_points={nv} "
                f"Vopt=({Vx_o:.3f},{Vy_o:.3f},{Vz_o:.3f}) "
                f"Vout=({Vz_o:.3f},{-Vx_o:.3f},{-Vy_o:.3f})"
            )

        # mapping to front_cam_link
        return Vz_o, -Vx_o, -Vy_o, nv

    def draw_debug(self, vis_bgr, good_old, good_new, depth_map):
        H, W = depth_map.shape[:2]
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
            _t = (z - dmin) / den
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