#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import message_filters
import numpy as np
import tf2_ros

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

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

        # 2. Initialize Core Logic
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Perception & Mapping
        engine_path = self.get_parameter('engine_path').value
        yaml_path = self.get_parameter('config_yaml').value

        self.depth_model = DA3TensorRTModel(engine_path, yaml_path)
        self.mapper = PotentialMapper(PotentialMapperConfig(resolution_m=0.05, size_m=20.0))

        # 3. Subscribers (Synchronized)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sub_image = message_filters.Subscriber(self, Image, '/simple_drone/front/image_raw', qos_profile=qos)
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/simple_drone/front/camera_info', qos_profile=qos)

        # TimeSynchronizer ensures we pair the image with the correct metadata
        self.ts = message_filters.TimeSynchronizer([self.sub_image, self.sub_info], 10)
        self.ts.registerCallback(self.image_callback)

        # 4. Publishers
        self.pub_grid = self.create_publisher(OccupancyGrid, '/map_local', 10)

        self.get_logger().info("Potential Mapper Node Started with Synchronized Callbacks.")

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

        self.last_pose = None
        self.last_time = None

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

        # B. Run Perception (DepthAnything V3)
        # Returns (H, W, 3) point cloud in Camera Frame
        _, point_cloud = self.depth_model.infer_all(cv_image)

        # C. Get Odometry Delta for Grid Warping
        target_time = rclpy.time.Time()
        df, dl, dy = self.get_odometry_delta(target_time=target_time)

        # D. Update Mapper (Using the stabilized Tanh + Numba logic)
        self.mapper.update(
            point_cloud,
            delta_fwd_m=df,
            delta_left_m=dl,
            delta_yaw_deg=dy
        )

        # E. Publish OccupancyGrid for RViz
        self.publish_occupancy_grid(img_msg.header)

    def publish_occupancy_grid(self, header):
        grid_msg = OccupancyGrid()
        grid_msg.header = header
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
        nav_data = self.mapper._M_nav.copy()
        mask_unknown = np.isnan(nav_data)

        ros_data = (nav_data * 100).astype(np.int8)
        ros_data[mask_unknown] = -1

        grid_msg.data = ros_data.flatten().tolist()
        self.get_logger().info(f"Publishing Occupancy Grid with unique values: {np.unique(grid_msg.data)}")
        self.pub_grid.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PotentialMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()