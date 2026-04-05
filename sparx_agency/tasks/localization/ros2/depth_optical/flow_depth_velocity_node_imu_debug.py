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

class FlowDepthVelocityNode(Node):

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
        self.declare_parameter("imu_topic", "/simple_drone/imu/out")
        
        #  LK params (Lucas-Kanade optical flow)
        self.declare_parameter("lk_win", 21) # window size
        self.declare_parameter("lk_levels", 3) # pyramid levels

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

        # log params
        self.get_logger().info(f"[FlowDepth] RGB: {image_topic}")
        self.get_logger().info(f"[FlowDepth] Depth: {depth_topic}")
        self.get_logger().info(f"[FlowDepth] CamInfo: {caminfo_topic}")
        self.get_logger().info(f"[FlowDepth] Pub: {output_topic}")

        self.bridge = CvBridge()

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # tracker module (fixed)
        self.tracker = OpticalFlowTracker(
            max_corners=self.max_corners,
            min_corners=self.min_corners,
            lk_win=lk_win,
            lk_levels=lk_levels,
        )

        # LK params
        self.lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), 
        )

        # IMU state
        self.latest_gyro = np.array([0.0, 0.0, 0.0])

        # pubs/subs
        self.vel_pub = self.create_publisher(Vector3Stamped, output_topic, 10)
        
        # Message Filters for synchronization
        self.rgb_sub = message_filters.Subscriber(self, Image, image_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)

        self.caminfo_sub = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_callback, 10)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)

        # Approximate Time Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=50,
            slop=0.15
        )
        self.ts.registerCallback(self.sync_callback)

    # ---------- callbacks ----------
    def caminfo_callback(self, msg: CameraInfo):
        K = msg.k
        self.fx = float(K[0]) # focal length x in pixels
        self.fy = float(K[4]) # focal length y in pixels
        self.cx = float(K[2]) # principal point x in pixels
        self.cy = float(K[5]) # principal point y in pixels
        
        # Log camera intrinsics (only once)
        if not hasattr(self, '_caminfo_logged'):
            self.get_logger().info(f"[CamInfo] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
            self.get_logger().info(f"[CamInfo] K matrix: {list(K)}")
            self._caminfo_logged = True

    def imu_callback(self, msg: Imu):
        roll_rate = msg.angular_velocity.x  
        pitch_rate = msg.angular_velocity.y 
        yaw_rate = msg.angular_velocity.z   
        self.latest_gyro[0] = -pitch_rate
        self.latest_gyro[1] = -yaw_rate
        self.latest_gyro[2] = roll_rate

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        # ----------------------------------------------------------------------
        # TIMING DEBUG PRINTS
        # ----------------------------------------------------------------------
        rgb_t = rgb_msg.header.stamp.sec + (rgb_msg.header.stamp.nanosec * 1e-9)
        depth_t = depth_msg.header.stamp.sec + (depth_msg.header.stamp.nanosec * 1e-9)
        time_diff = abs(rgb_t - depth_t)

        # Initialize counter for throttling logs
        if not hasattr(self, '_sync_log_count'):
            self._sync_log_count = 0
        
        self._sync_log_count += 1
        
        # Print every 10 frames to avoid spamming the console
        if self._sync_log_count % 10 == 0:
            self.get_logger().info(
                f"[SYNC DEBUG] RGB t: {rgb_t:.4f} | Depth t: {depth_t:.4f} | Diff: {time_diff:.4f} sec"
            )
            if time_diff > 0.05:
                self.get_logger().warn(f"[SYNC WARNING] Time difference > 50ms ({time_diff:.4f} sec). Check simulator rate.")
        # ----------------------------------------------------------------------

        if self.fx is None or self.fy is None:
            return

        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return
        
        depth_map = np.asarray(depth_cv, dtype=np.float32)

        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB convert failed: {e}")
            return

        flow_res = self.tracker.process(frame, rgb_msg.header.stamp)
        if flow_res is None:
            return

        vx_mps, vy_mps, vz_mps, n_used = self.velocity_from_flow_and_depth(
            flow_res.good_old,
            flow_res.good_new,
            depth_map,
            flow_res.dt,
            self.latest_gyro
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
        """
        good_old/new: [N,2] float pixel coords
        depth_map: [H,W] float32 depth (relative or scaled)
        dt: seconds

        Returns:
          (vx_mps, vy_mps, n_used) as robust medians over valid points
        """
        H, W = depth_map.shape[:2] # height, width of depth map

        # pixel velocities px/s of the tracked points in the optical flow
        du_total = (good_new[:, 0] - good_old[:, 0]) / dt
        dv_total = (good_new[:, 1] - good_old[:, 1]) / dt

        # compensate for rotational flow from gyro
        wx, wy, wz = gyro

        # compute expected pixel flow from rotation (assuming small angles and focal length in pixels)
        u = good_old[:, 0]
        v = good_old[:, 1]
        
        # centered pixel coordinates
        u_c = u - self.cx
        v_c = v - self.cy

        du_rot = (u_c * v_c / self.fx) * wx - (self.fx + u_c**2 / self.fx) * wy + v_c * wz
        dv_rot = (self.fy + v_c**2 / self.fy) * wx - (u_c * v_c / self.fy) * wy - u_c * wz

        # translational flow is total flow minus rotational flow
        du_trans = du_total - du_rot
        dv_trans = dv_total - dv_rot

        # sample depth at new locations
        u_int = np.rint(good_new[:, 0]).astype(np.int32)
        v_int = np.rint(good_new[:, 1]).astype(np.int32)

        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        if not np.any(valid):
            return 0.0, 0.0, 0

        Z = np.zeros_like(du_total, dtype=np.float32)
        Z[valid] = depth_map[v_int[valid], u_int[valid]] * self.depth_scale  # sample depth map and apply scale

        # filter bad depth
        valid = valid & np.isfinite(Z) & (Z > self.min_depth) & (Z < self.max_depth)

        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0

        MIN_POINTS = 8
        nv = int(np.sum(valid))
        if nv < MIN_POINTS:
            return 0.0, 0.0, 0.0, 0

        # Use old-point coordinates for u_c, v_c (same as where rotation was computed)
        u_c_v = good_old[valid, 0].astype(np.float64) - self.cx
        v_c_v = good_old[valid, 1].astype(np.float64) - self.cy
        Zv    = Z[valid].astype(np.float64)
        du_v  = du_trans[valid].astype(np.float64)
        dv_v  = dv_trans[valid].astype(np.float64)

        A = np.zeros((2 * nv, 3), dtype=np.float64)
        B = np.zeros((2 * nv,),   dtype=np.float64)
        A[0::2, 0] = -self.fx;  A[0::2, 2] = u_c_v;  B[0::2] = du_v * Zv
        A[1::2, 1] = -self.fy;  A[1::2, 2] = v_c_v;  B[1::2] = dv_v * Zv

        try:
            vel, *_ = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, nv

        Vx_o, Vy_o, Vz_o = float(vel[0]), float(vel[1]), float(vel[2])
        
        # Debug: Check rotational vs translational components
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count % 50 == 0:  # Print every 50 frames
            du_rot_mean = np.mean(du_rot[valid]) if len(du_rot) > 0 else 0
            dv_rot_mean = np.mean(dv_rot[valid]) if len(dv_rot) > 0 else 0
            du_trans_mean = np.mean(du_trans[valid]) if len(du_trans) > 0 else 0
            dv_trans_mean = np.mean(dv_trans[valid]) if len(dv_trans) > 0 else 0
            self.get_logger().info(f"[LSTSQ DEBUG] du_rot={du_rot_mean:.3f} dv_rot={dv_rot_mean:.3f} | du_trans={du_trans_mean:.3f} dv_trans={dv_trans_mean:.3f}")
            self.get_logger().info(f"[LSTSQ DEBUG] Vx_o={Vx_o:.3f} Vy_o={Vy_o:.3f} Vz_o={Vz_o:.3f} | Output: Vx={Vz_o:.3f} Vy={-Vx_o:.3f} Vz={-Vy_o:.3f}")
            self.get_logger().info(f"[LSTSQ DEBUG] Gyro: {self.latest_gyro}")
        
        return Vz_o, -Vx_o, -Vy_o, nv

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