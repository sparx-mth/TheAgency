#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import message_filters
import numpy as np
import tf2_ros

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, PointStamped
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32
import math

from sparx_agency.core.mapping.costmap.potential_mapper import PotentialMapper, PotentialMapperConfig
from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel


class PotentialMapperNode(Node):
    def __init__(self):
        super().__init__('potential_mapper_node')

        # 1. Parameters


        self.declare_parameter('engine_path', '/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine')
        self.declare_parameter('config_yaml', '/home/daphnaa/GIT/TheAgency/sparx_agency/tasks/mapping/config/simple_drone_front_cam.yaml')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')

        self.declare_parameter('size_m', 6.5)  # From demo_depth
        self.declare_parameter('show_gui', True)
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.click_target = None
        self.last_click_pixel = None
        self.last_point_cloud = None

        # 2. Initialize Core Logic
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Perception & Mapping
        engine_path = self.get_parameter('engine_path').value
        yaml_path = self.get_parameter('config_yaml').value

        self.depth_model = DA3TensorRTModel(engine_path, yaml_path)
        self.mapper = PotentialMapper(PotentialMapperConfig(resolution_m=0.05, size_m=20.0))

        self.MAP_SIZE = self.get_parameter('size_m').value
        self.PANE_W = 320
        self.PANE_H = 320
        self.click_target = None
        self.latest_frame = None
        self.target_u = None

        # 3. Subscribers (Synchronized)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


        self.sub_image = message_filters.Subscriber(self, Image, '/simple_drone/front/image_raw', qos_profile=qos)
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/simple_drone/front/camera_info', qos_profile=qos)

        self.sub_target_pixel = self.create_subscription(
            PointStamped,
            '/planner_target_pixel',
            self.target_pixel_callback,
            10
        )

        self.sub_clicked_point = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )
        # TimeSynchronizer ensures we pair the image with the correct metadata
        self.ts = message_filters.TimeSynchronizer([self.sub_image, self.sub_info], 10)
        self.ts.registerCallback(self.image_callback)

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        # 4. Publishers
        self.pub_grid = self.create_publisher(OccupancyGrid, '/map_local', qos_profile=qos)
        self.pub_nav_vector = self.create_publisher(
            Vector3Stamped,
            '/local_nav_vector',
            10
        )

        self.pub_nav_heading = self.create_publisher(
            Float32,
            '/local_nav_heading',
            10
        )
        self.pub_pf_debug = self.create_publisher(Image, '/potential_field_debug', 10)
        self.pub_depth_debug = self.create_publisher(Image, '/depth_debug', 10)
        self.get_logger().info("Potential Mapper Node Started with Synchronized Callbacks.")

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = self.get_parameter('base_frame').value
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

        self.last_pose = None
        self.last_time = None
        self.latest_rgb = None
        self.display_rgb = None

        if self.get_parameter('show_gui').value:
            cv2.namedWindow("Sparx Click Interface", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sparx Click Interface", 960, 540)
            cv2.setMouseCallback("Sparx Click Interface", self.on_rgb_click)
            cv2.waitKey(1)
            self.create_timer(0.033, self.cv_refresh_callback)

    def get_odometry_delta(self, target_time):
        try:
            # 1. Lookup current transform (Odom -> Base)
            trans = self.tf_buffer.lookup_transform(
                self.get_parameter('odom_frame').value,
                self.get_parameter('base_frame').value,
                target_time,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )

            curr_pos = trans.transform.translation
            curr_rot = trans.transform.rotation

            # 2. Convert Quaternion to Yaw (Euler)
            import math
            siny_cosp = 2 * (curr_rot.w * curr_rot.z + curr_rot.x * curr_rot.y)
            cosy_cosp = 1 - 2 * (curr_rot.y * curr_rot.y + curr_rot.z * curr_rot.z)
            curr_yaw = math.atan2(siny_cosp, cosy_cosp)

            if self.last_pose is None:
                self.last_pose = (curr_pos.x, curr_pos.y, curr_yaw)
                return 0.0, 0.0, 0.0

            # 3. Calculate Deltas in Global Frame
            dx = curr_pos.x - self.last_pose[0]
            dy = curr_pos.y - self.last_pose[1]
            dyaw = curr_yaw - self.last_pose[2]

            # 4. Transform Global Delta to Robot-Local (Fwd/Left)
            # PotentialMapper expects deltas relative to the robot's heading
            cos_y = math.cos(self.last_pose[2])
            sin_y = math.sin(self.last_pose[2])

            delta_fwd = dx * cos_y + dy * sin_y
            delta_left = -dx * sin_y + dy * cos_y
            delta_yaw_deg = math.degrees(dyaw)

            # Update for next frame
            self.last_pose = (curr_pos.x, curr_pos.y, curr_yaw)

            return delta_fwd, delta_left, delta_yaw_deg

        except Exception as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")
            return 0.0, 0.0, 0.0

    def image_callback(self, img_msg, info_msg):
        # A. Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        # Save for GUI display and click reference
        self.latest_rgb = cv_image
        self.display_rgb = cv_image.copy()
        # B. Run Perception (DepthAnything V3)
        # Returns (H, W, 3) point cloud in Camera Frame
        depth_map, point_cloud = self.depth_model.infer_all(cv_image)
        self.last_point_cloud = point_cloud.copy()
        # C. Get Odometry Delta for Grid Warping
        target_time = rclpy.time.Time()
        df, dl, dy = self.get_odometry_delta(target_time=target_time)

        # D. Publish OccupancyGrid for RViz
        if self.click_target:
            self.mapper.set_goal(*self.click_target)
        elif self.target_u is not None:
            local_target = self.get_local_target_from_pixel()
            self.get_logger().info(f"Target pixel: u={self.target_u:.1f}, v={self.target_v:.1f}")
            if local_target: self.mapper.set_goal(*local_target)

        # E. Update Mapper (Using the stabilized Tanh + Numba logic)
        self.mapper.update(
            point_cloud,
            delta_fwd_m=df,
            delta_left_m=dl,
            delta_yaw_deg=dy
        )

        self.publish_occupancy_grid(img_msg.header)
        self.publish_local_nav(img_msg.header)
        self.publish_potential_field_debug(img_msg.header)
        self.publish_depth_debug(depth_map, img_msg.header)

    def target_pixel_callback(self, msg):
        self.target_u = float(msg.point.x)
        self.target_v = float(msg.point.y)
        self.get_logger().debug(f"Target pixel: u={self.target_u:.1f}, v={self.target_v:.1f}")

    def clicked_point_callback(self, msg: PointStamped):
        """
        RViz Publish Point callback.
        Converts clicked world point into local target in base frame:
          forward = x
          left = y
        """
        try:
            base_frame = self.get_parameter('base_frame').value

            if msg.header.frame_id == base_frame:
                x_b = float(msg.point.x)
                y_b = float(msg.point.y)
            else:
                tf = self.tf_buffer.lookup_transform(
                    base_frame,
                    msg.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )

                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                q = tf.transform.rotation

                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )

                # clicked point in source frame
                px = float(msg.point.x)
                py = float(msg.point.y)

                # transform 2D point into base frame
                x_b = tx + math.cos(yaw) * px - math.sin(yaw) * py
                y_b = ty + math.sin(yaw) * px + math.cos(yaw) * py

            # local convention: forward=x, left=y
            self.click_target = (x_b, y_b)
            self.get_logger().info(
                f"Clicked target in {base_frame}: fwd={x_b:.2f}, left={y_b:.2f}"
            )

        except Exception as e:
            self.get_logger().warn(f"Failed to transform clicked point: {e}")

    def get_local_target_from_pixel(self) -> tuple[float, float] | None:
        """
        Convert target pixel (u,v) into a local attractive direction (forward, left).

        We do not estimate metric distance here.
        We only create a normalized steering direction:
          - forward always positive
          - left depends on horizontal pixel offset
        """
        if self.target_u is None or self.target_v is None:
            return None

        intr = self.depth_model.intrinsics
        if intr is None:
            return None

        # Pixel -> normalized image ray
        x_img = (self.target_u - intr.cx) / intr.fx
        # y_img available if later you want vertical control:
        # y_img = (self.target_v - intr.cy) / intr.fy

        # Attractive direction in your local convention:
        # forward = x axis of command
        # left    = y axis of command
        v_fwd = 1.0
        v_left = float(x_img)

        norm = math.sqrt(v_fwd * v_fwd + v_left * v_left)
        if norm < 1e-6:
            return 1.0, 0.0

        return v_fwd / norm, v_left / norm

    def get_active_attractive_direction(self) -> np.ndarray:
        """
        Return the currently active attractive direction as a normalized [fwd, left] vector.
        Priority:
          1. metric click target from RGB point cloud click
          2. planner target pixel converted to local direction
          3. straight ahead
        """
        if self.click_target is not None:
            fwd, left = self.click_target
            v = np.array([float(fwd), float(left)], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                return v / n

        local_target = self.get_local_target_from_pixel()
        if local_target is not None:
            v = np.array(local_target, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                return v / n

        return np.array([1.0, 0.0], dtype=np.float32)

    def publish_occupancy_grid(self, header):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.get_parameter('base_frame').value

        # Map metadata
        res = self.mapper.cfg.resolution_m
        n = self.mapper._n
        grid_msg.info.resolution = res
        grid_msg.info.width = n
        grid_msg.info.height = n
        grid_msg.info.origin.position.x = 0.0  # Origin is robot center
        grid_msg.info.origin.position.y = - (n // 2) * res
        grid_msg.info.origin.position.z = 0.0

        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.info.origin.orientation.x = 0.0
        grid_msg.info.origin.orientation.y = 0.0
        grid_msg.info.origin.orientation.z = 0.0

        # Convert M_nav [0..1] to ROS [0..100]
        # Use -1 for NaNs (Unknown)
        # Use the probability map (0 to 1) and scale to ROS (0 to 100)
        nav_data = self.mapper.get_potential_map()
        ros_data = (np.nan_to_num(nav_data, nan=0.0) * 100).astype(np.int8)

        # # Flip so OpenCV-style becomes map-style / Rviz-style
        ros_data = np.flip(ros_data)

        grid_msg.data = ros_data.flatten().tolist()
        # self.get_logger().info(f"Publishing Occupancy Grid with unique values: {np.unique(grid_msg.data)}")
        self.pub_grid.publish(grid_msg)

    def publish_local_nav(self, header):
        n = self.mapper._n
        cr = n - 1
        cc = n // 2

        grad_total = self.mapper.get_total_gradient()
        if grad_total is None:
            return

        v = np.array(grad_total[cr, cc], dtype=np.float32)

        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v = v / norm
        else:
            v = np.array([0.0, 0.0], dtype=np.float32)

        v_fwd = float(v[0])
        v_left = float(v[1])
        heading = math.atan2(v_left, v_fwd)

        vec_msg = Vector3Stamped()
        vec_msg.header.stamp = self.get_clock().now().to_msg()
        vec_msg.header.frame_id = self.get_parameter('base_frame').value
        vec_msg.vector.x = v_fwd
        vec_msg.vector.y = v_left
        vec_msg.vector.z = 0.0
        self.pub_nav_vector.publish(vec_msg)

        heading_msg = Float32()
        heading_msg.data = float(heading)
        self.pub_nav_heading.publish(heading_msg)

    def publish_potential_field_debug(self, header):
        nav = self.mapper.get_nav_map()
        grad_total = self.mapper.get_total_gradient()

        if nav is None or grad_total is None:
            return

        # display convention:
        # robot at bottom-center
        # forward-positive -> up
        # left-positive -> left
        nav_disp = np.flipud(nav)
        grad_disp = np.flipud(grad_total).copy()

        h, w = nav_disp.shape
        cr = h - 1
        cc = w // 2

        # ------------------------------------------------------------
        # clean bright background instead of heatmap
        # ------------------------------------------------------------
        vis = np.full((h, w, 3), 245, dtype=np.uint8)

        def metric_to_display(fwd, left):
            fwd = float(np.clip(fwd, 0.0, self.mapper.cfg.size_m))
            half = 0.5 * self.mapper.cfg.size_m
            left = float(np.clip(left, -half, half))
            y_px = int((1.0 - fwd / self.mapper.cfg.size_m) * (h - 1))
            x_px = int((0.5 - left / self.mapper.cfg.size_m) * (w - 1))
            return x_px, y_px

        # ------------------------------------------------------------
        # draw obstacles clearly
        # ------------------------------------------------------------
        occ_mask = nav_disp > self.mapper.cfg.occ_thresh
        kernel = np.ones((3, 3), np.uint8)
        occ_vis = cv2.dilate(occ_mask.astype(np.uint8), kernel)

        vis[occ_vis > 0] = (40, 40, 40)

        # ------------------------------------------------------------
        # total field arrows across whole map
        # ------------------------------------------------------------
        step = max(4, h // 30)
        scale = 14.0

        for r in range(0, h, step):
            for c in range(0, w, step):
                # don't draw inside obstacle cells
                if occ_mask[r, c]:
                    continue

                v_fwd = float(grad_disp[r, c, 0])
                v_left = float(grad_disp[r, c, 1])
                mag = math.hypot(v_fwd, v_left)
                if mag < 1e-4:
                    continue

                # normalize for display only
                v_fwd /= mag
                v_left /= mag

                dx = int(-scale * v_left)  # left-positive -> screen left
                dy = int(-scale * v_fwd)  # forward-positive -> screen up

                cv2.arrowedLine(
                    vis,
                    (c, r),
                    (c + dx, r + dy),
                    (255, 20, 20),  # dark arrows
                    1,
                    tipLength=0.35
                )

        # ------------------------------------------------------------
        # target marker
        # ------------------------------------------------------------
        if self.click_target is not None:
            click_fwd, click_left = self.click_target
            px, py = metric_to_display(click_fwd, click_left)

            cv2.circle(vis, (px, py), 10, (220, 0, 220), -1)  # magenta goal
            cv2.putText(
                vis, "Goal", (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 0, 120), 1
            )

        # ------------------------------------------------------------
        # obstacle point marker if you want to show raw click separately
        # (currently same as click_target)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # robot marker
        # ------------------------------------------------------------
        cv2.circle(vis, (cc, cr), 7, (0, 180, 255), -1)

        # robot local total direction
        v_robot = np.array(grad_disp[cr, cc], dtype=np.float32)
        nrm = float(np.linalg.norm(v_robot))
        if nrm > 1e-6:
            v_robot /= nrm
            dx = int(-35 * float(v_robot[1]))
            dy = int(-35 * float(v_robot[0]))
            cv2.arrowedLine(
                vis,
                (cc, cr),
                (cc + dx, cr + dy),
                (0, 120, 255),
                1,
                tipLength=0.3
            )

        # ------------------------------------------------------------
        # enlarge for easier viewing
        # ------------------------------------------------------------
        vis_big = cv2.resize(vis, (640, 640), interpolation=cv2.INTER_NEAREST)

        msg = self.bridge.cv2_to_imgmsg(vis_big, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('base_frame').value
        self.pub_pf_debug.publish(msg)

    def publish_depth_debug(self, depth_map, header):
        depth_clean = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = header.frame_id
        self.pub_depth_debug.publish(msg)

    def on_rgb_click(self, event, u, v, flags, param):
        """
        Translates pixel (u,v) to metric (fwd, left) using camera intrinsics.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.last_point_cloud is None:
                return

            p = self.last_point_cloud[v, u]  # (x, y, z)

            if not np.isfinite(p).all():
                return

            fwd = float(p[2])
            left = float(p[0])

            self.click_target = (fwd, left)
            # Store pixel for visual feedback in the window
            self.last_click_pixel = (u, v)
            self.get_logger().info(f"Pixel ({u}, {v}) -> Metric Fwd: {fwd:.2f}m, Left: {left:.2f}m")

    def cv_refresh_callback(self):
        """Refreshes window and draws the target point."""
        if self.display_rgb is not None:
            vis_img = self.display_rgb.copy()

            if self.last_click_pixel is not None:
                cv2.circle(vis_img, self.last_click_pixel, 2, (0, 0, 255), -1)
                cv2.putText(
                    vis_img,
                    "Target Set",
                    (self.last_click_pixel[0] + 10, self.last_click_pixel[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

            cv2.imshow("Sparx Click Interface", vis_img)

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PotentialMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()