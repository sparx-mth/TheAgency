import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from sparx_agency.core.mapping.costmap.integrated_map import IntegratedMap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.robots.common.spatial_math import rot_y


class OccupancyNode(Node):
    def __init__(self):
        super().__init__('occupancy_node')
        self.cfg = ProbabilisticGridConfig()
        self.map = IntegratedMap(self.cfg)

        self.declare_parameter('pitch_deg', 30.0)
        self.declare_parameter('sensor_z', 10.0)
        self.declare_parameter('accumulate', True)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(PointCloud2, '/camera/point_cloud', self.cb, sensor_qos)
        self.pub_2d = self.create_publisher(OccupancyGrid, '/map_2d', 10)
        self.pub_3d = self.create_publisher(Marker, '/map_3d', 10)

    def cb(self, msg):
        # 1. Standard reading of raw camera points (Z=forward, X=left, Y=up)
        pts_struct = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts_np = np.array(list(pts_struct))
        if pts_np.size == 0: return

        # Extract based on your observation
        raw_x = pts_np['x']  # Left
        raw_y = pts_np['y']  # Up
        raw_z = pts_np['z']  # Forward

        # 2. Map to World Frame: X=Forward, Y=Left, Z=Up
        world_x = raw_z
        world_y = raw_x
        world_z = raw_y

        pts_w_local = np.stack([world_x, world_y, world_z], axis=1)

        # 3. Apply Pitch and Height (from your map_from_image logic)
        pitch_rad = np.radians(self.get_parameter('pitch_deg').value)
        sz = self.get_parameter('sensor_z').value

        # Rotate around Y to handle camera tilt and add height offset
        R = rot_y(-pitch_rad)
        t = np.array([0.0, 0.0, sz], np.float32)
        pts_w = (pts_w_local @ R.T) + t

        # 4. Update the map
        self.map.update(pts_w, t, accumulate=self.get_parameter('accumulate').value)
        self.publish_viz(msg.header.frame_id)

    def publish_viz(self, frame_id):
        # 2D Grid Setup
        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.info.resolution = self.map.res
        grid.info.width = self.map.width
        grid.info.height = self.map.height
        grid.info.origin.position.x = self.map.origin_x
        grid.info.origin.position.y = self.map.origin_y

        # Create the data array using your probabilistic thresholds
        data = np.full((self.map.width, self.map.height), -1, dtype=np.int8)
        data[self.map._seen_mask & (self.map._lo <= 0)] = 20  # Light Gray (Free)
        data[self.map._seen_mask & (self.map._lo > 0.5)] = 100  # Black (Occupied)

        grid.data = data.flatten().tolist()
        self.pub_2d.publish(grid)

        # 3D Voxels
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.CUBE_LIST
        marker.scale.x = marker.scale.y = marker.scale.z = self.map.res
        marker.color.r, marker.color.a = 1.0, 0.6

        from geometry_msgs.msg import Point
        # for (gy, gx, gz_idx) in voxels:
        #     p = Point()
        #     p.x = float(gy * self.map.res + self.map.origin_x)
        #     p.y = float(gx * self.map.res + self.map.origin_y)
        #     p.z = float(gz_idx * self.map.res)
        #     marker.points.append(p)
        # self.pub_3d.publish(marker)


def main():
    rclpy.init()
    rclpy.spin(OccupancyNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()