import struct

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from sparx_agency.core.mapping.costmap.integrated_map import IntegratedMap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.robots.common.spatial_math import rot_y_deg, fov_to_intrinsics


class OccupancyNode(Node):
    def __init__(self):
        super().__init__('occupancy_node')
        self.cfg = ProbabilisticGridConfig()
        self.map = IntegratedMap(self.cfg)

        # 1. Build Intrinsics
        self.width = 640
        self.height = 480
        self.hfov = 130.0
        self.vfov = 90.0
        self.intr = fov_to_intrinsics(self.width, self.height, self.hfov, self.vfov)

        # 2. Parameters
        self.declare_parameter('pitch_deg', 0.0)
        self.declare_parameter('sensor_z', 1.0)
        self.declare_parameter('accumulate', True)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(PointCloud2, '/video/point_cloud', self.cb, sensor_qos)
        self.pub_2d = self.create_publisher(OccupancyGrid, '/map_2d', 10)
        self.pub_transformed_cloud = self.create_publisher(PointCloud2, '/video/point_cloud_transformed', 10)

    def cb(self, msg):
        # 1. Read 'rgb' and 'xyz'
        pts_struct = point_cloud2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        pts_np = np.array(list(pts_struct))
        if pts_np.size == 0: return

        # 2. Correct Axis Mapping (Flipping Z to put floor at the bottom)
        world_x = pts_np['z']  # Depth -> Forward
        world_y = -pts_np['x']  # Horizontal -> Lateral
        world_z = pts_np['y']  # Vertical Down -> Vertical Up (Negative flips it)

        pts_w_local = np.stack([world_x, world_y, world_z], axis=1).astype(np.float32)

        # 3. Apply Pitch (Rotation around Y)
        # If the floor and ceiling were swapped, the rotation was likely
        # tilting the wrong way. Ensure the sign of pitch matches your camera tilt.
        pitch_rad = np.radians(self.get_parameter('pitch_deg').value)
        sz = self.get_parameter('sensor_z').value

        # R rotates the points around the lateral Y axis
        R = rot_y_deg(-pitch_rad)
        pts_transformed = (pts_w_local @ R.T)

        # 4. Lift to sensor height
        pts_transformed[:, 2] += sz

        # 5. Republish
        self.republish_rgb_cloud(pts_transformed, pts_np['rgb'], msg.header)

        # 6. Update Map
        t = np.array([0.0, 0.0, sz], np.float32)
        self.map.update(pts_transformed, t, accumulate=True)
        self.publish_viz(msg.header.frame_id)

    def republish_rgb_cloud(self, pts, packed_colors, original_header):
        # Use the exact fields found in your incoming message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1),  # Match offset 16
        ]

        header = Header()
        header.stamp = original_header.stamp
        header.frame_id = original_header.frame_id

        merged_data = []
        for i in range(len(pts)):
            merged_data.append([
                float(pts[i, 0]),
                float(pts[i, 1]),
                float(pts[i, 2]),
                float(packed_colors[i])
            ])

        cloud_msg = point_cloud2.create_cloud(header, fields, merged_data)
        self.pub_transformed_cloud.publish(cloud_msg)

    def publish_viz(self, frame_id):
        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.info.resolution = self.map.res
        grid.info.width = self.map.width
        grid.info.height = self.map.height
        grid.info.origin.position.x = self.map.origin_x
        grid.info.origin.position.y = self.map.origin_y
        data = np.full((self.map.height, self.map.width), -1, dtype=np.int8)
        lo, seen, _ = self.map.get_viz_data()
        data[seen & (lo <= 0)] = 20
        data[seen & (lo > 0.5)] = 100
        grid.data = data.flatten().tolist()
        self.pub_2d.publish(grid)

    def republish_cloud(self, pts, original_header):
        header = Header()
        header.stamp = original_header.stamp
        header.frame_id = original_header.frame_id  # Usually 'map' or 'odom'

        # Create and publish the new cloud message
        cloud_msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.pub_transformed_cloud.publish(cloud_msg)


def main():
    rclpy.init()
    rclpy.spin(OccupancyNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()