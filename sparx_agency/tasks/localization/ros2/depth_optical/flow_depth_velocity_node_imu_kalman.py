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
from sensor_msgs.msg import Imu
import message_filters


class KalmanVelocityFilter:
    def __init__(self):
        # state: [vx, vy, vz]
        self.x = np.zeros((3, 1), dtype=np.float64)

        # covariance
        self.P = np.eye(3, dtype=np.float64) * 1.0

        # dynamics: constant velocity
        self.F = np.eye(3, dtype=np.float64)

        # measurement matrix
        self.H = np.eye(3, dtype=np.float64)

        # tunable noises
        self.Q = np.eye(3, dtype=np.float64) * 0.02
        self.R = np.eye(3, dtype=np.float64) * 0.15

        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        z = z.reshape(3, 1)

        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.flatten()

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(3, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.x.flatten()

    def step(self, z: np.ndarray):
        if not self.initialized:
            return self.update(z)
        self.predict()
        return self.update(z)


class FlowDepthVelocityNode(Node):

    def __init__(self):
        super().__init__("flow_depth_velocity_node")

        self.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True
        )])

        self.declare_parameter("image_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("depth_topic", "/depth_anything/depth")
        self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("output_topic", "/flow_depth/velocity")

        self.declare_parameter("max_corners", 300)
        self.declare_parameter("min_corners", 70)

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 30.0)
        self.declare_parameter("use_depth_norm", False)
        self.declare_parameter("depth_scale", 0.4)
        self.declare_parameter("show_debug", False)
        self.declare_parameter("camera_frame", "simple_drone/front_cam_link")
        self.declare_parameter("imu_topic", "/simple_drone/imu/out")

        self.declare_parameter("lk_win", 21)
        self.declare_parameter("lk_levels", 3)

        # Kalman params
        self.declare_parameter("use_kalman", True)
        self.declare_parameter("kalman_q", 0.02)
        self.declare_parameter("kalman_r", 0.15)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value

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

        self.use_kalman = bool(self.get_parameter("use_kalman").get_parameter_value().bool_value)
        self.kalman_q = float(self.get_parameter("kalman_q").get_parameter_value().double_value)
        self.kalman_r = float(self.get_parameter("kalman_r").get_parameter_value().double_value)

        self.get_logger().info(f"[FlowDepth] RGB: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] Pub: {output_topic}")
        self.get_logger().info(f"[Kalman] enabled={self.use_kalman} Q={self.kalman_q} R={self.kalman_r}")

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_stamp = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

        self.latest_gyro = np.array([0.0, 0.0, 0.0])

        self.kalman_filter = KalmanVelocityFilter()
        self.kalman_filter.Q = np.eye(3, dtype=np.float64) * self.kalman_q
        self.kalman_filter.R = np.eye(3, dtype=np.float64) * self.kalman_r

        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, 10)

        self.rgb_sub = message_filters.Subscriber(self, Image, image_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)

        self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.imu_sub_sync = message_filters.Subscriber(self, Imu, imu_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.imu_sub_sync],
            queue_size=50,
            slop=0.15
        )
        self.ts.registerCallback(self.sync_callback)

    def caminfo_callback(self, msg: CameraInfo):
        K = msg.k
        self.fx = float(K[0])
        self.fy = float(K[4])
        self.cx = float(K[2])
        self.cy = float(K[5])

        if not hasattr(self, '_caminfo_logged'):
            self.get_logger().info(
                f"[CamInfo] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            self._caminfo_logged = True

    def imu_callback(self, msg: Imu):
        roll_rate = msg.angular_velocity.x
        pitch_rate = msg.angular_velocity.y
        yaw_rate = msg.angular_velocity.z
        self.latest_gyro[0] = -pitch_rate
        self.latest_gyro[1] = -yaw_rate
        self.latest_gyro[2] = roll_rate

    def sync_callback(self, rgb_msg: Image, depth_msg: Image, imu_msg: Imu):
        if self.fx is None or self.fy is None:
            return

        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            depth_map = np.asarray(depth_cv, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        roll_rate = imu_msg.angular_velocity.x
        pitch_rate = imu_msg.angular_velocity.y
        yaw_rate = imu_msg.angular_velocity.z
        gyro_sync = np.array([
            -pitch_rate,
            -yaw_rate,
            roll_rate
        ], dtype=np.float64)

        flow_res = self.tracker.process(frame, rgb_msg.header.stamp)
        if flow_res is None:
            return

        vx_raw, vy_raw, vz_raw, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old,
            flow_res.good_new,
            depth_map,
            flow_res.dt,
            gyro_sync
        )

        vx_mps, vy_mps, vz_mps = vx_raw, vy_raw, vz_raw

        if self.use_kalman and n_used > 0:
            z = np.array([vx_raw, vy_raw, vz_raw], dtype=np.float64)
            vx_mps, vy_mps, vz_mps = self.kalman_filter.step(z)

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
            txt = (
                f"raw=({vx_raw:.3f},{vy_raw:.3f},{vz_raw:.3f}) "
                f"filt=({vx_mps:.3f},{vy_mps:.3f},{vz_mps:.3f}) "
                f"used={n_used}"
            )
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow("Flow+Depth Velocity", vis)
            cv2.waitKey(1)

    def velocity_from_flow_and_depth(self, good_old, good_new, depth_map, dt: float, gyro: np.ndarray):
        H, W = depth_map.shape[:2]

        du_total = (good_new[:, 0] - good_old[:, 0]) / dt
        dv_total = (good_new[:, 1] - good_old[:, 1]) / dt

        wx, wy, wz = gyro

        u = good_old[:, 0]
        v = good_old[:, 1]
        u_c = u - self.cx
        v_c = v - self.cy

        du_rot = (u_c * v_c / self.fx) * wx - (self.fx + u_c**2 / self.fx) * wy + v_c * wz
        dv_rot = (self.fy + v_c**2 / self.fy) * wx - (u_c * v_c / self.fy) * wy - u_c * wz

        du_trans = du_total - du_rot
        dv_trans = dv_total - dv_rot

        u_int = np.rint(good_new[:, 0]).astype(np.int32)
        v_int = np.rint(good_new[:, 1]).astype(np.int32)

        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0

        Z = np.zeros_like(du_total, dtype=np.float32)
        Z[valid] = depth_map[v_int[valid], u_int[valid]] * self.depth_scale

        valid = valid & np.isfinite(Z) & (Z > self.min_depth) & (Z < self.max_depth)
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0

        MIN_POINTS = 8
        nv = int(np.sum(valid))
        if nv < MIN_POINTS:
            return 0.0, 0.0, 0.0, 0

        u_c_v = good_old[valid, 0].astype(np.float64) - self.cx
        v_c_v = good_old[valid, 1].astype(np.float64) - self.cy
        Zv = Z[valid].astype(np.float64)
        du_v = du_trans[valid].astype(np.float64)
        dv_v = dv_trans[valid].astype(np.float64)

        A = np.zeros((2 * nv, 3), dtype=np.float64)
        B = np.zeros((2 * nv,), dtype=np.float64)

        weight = 1.0 / (Zv + 1e-6)

        A[0::2, 0] = -self.fx * weight
        A[0::2, 2] = u_c_v * weight
        B[0::2] = du_v * Zv * weight

        A[1::2, 1] = -self.fy * weight
        A[1::2, 2] = v_c_v * weight
        B[1::2] = dv_v * Zv * weight

        try:
            vel, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, nv

        Vx_o, Vy_o, Vz_o = float(vel[0]), float(vel[1]), float(vel[2])

        return Vz_o, -Vx_o, -Vy_o, nv

    def draw_debug(self, vis_bgr, good_old, good_new, depth_map):
        H, W = depth_map.shape[:2]
        for (u2, v2), (u1, v1) in zip(good_new, good_old):
            x2, y2 = int(round(u2)), int(round(v2))
            x1, y1 = int(round(u1)), int(round(v1))
            if not (0 <= x2 < W and 0 <= y2 < H):
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