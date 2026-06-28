#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as Rot

from sparx_agency.core.mapping.costmap.integrated_map import IntegratedMap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig


class OccupancyNode(Node):
    def __init__(self):
        super().__init__('occupancy_node')
        self.cfg = ProbabilisticGridConfig()
        self.map = IntegratedMap(self.cfg)

        self.declare_parameter('sensor_z', 1.0)
        self.declare_parameter('accumulate', True)
        self.declare_parameter('pointcloud_topic', '/xtend/pointcloud')
        self.declare_parameter('pose_topic', '/xtend/april_tag_pose')
        self.declare_parameter('cloud_out_topic', '/xtend/pointcloud_world')
        self.declare_parameter('occupancy_topic', '/xtend/occupancy_grid')

        self._world_R: np.ndarray | None = None  # 3x3 rotation world_T_body
        self._world_t: np.ndarray | None = None  # 3, translation world_T_body

        best_effort_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        cloud_topic   = str(self.get_parameter('pointcloud_topic').value)
        pose_topic    = str(self.get_parameter('pose_topic').value)
        cloud_out     = str(self.get_parameter('cloud_out_topic').value)
        occupancy_out = str(self.get_parameter('occupancy_topic').value)

        self.create_subscription(PointCloud2, cloud_topic, self._cloud_cb, best_effort_qos)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10)
        # Use RELIABLE for world-frame cloud so RViz (which subscribes RELIABLE by default) receives it.
        self.pub_cloud = self.create_publisher(PointCloud2, cloud_out, 10)
        self.pub_occ   = self.create_publisher(OccupancyGrid, occupancy_out, 10)

        self.get_logger().info(f"cloud in:  {cloud_topic}")
        self.get_logger().info(f"pose in:   {pose_topic}")
        self.get_logger().info(f"cloud out: {cloud_out}")
        self.get_logger().info(f"occ out:   {occupancy_out}")

    def _pose_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        p = msg.pose.position
        self._world_R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        self._world_t = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        pts_raw = self._read_xyz(msg)
        if pts_raw is None or len(pts_raw) == 0:
            return

        # Camera optical (X=right, Y=down, Z=forward) → body (X=forward, Y=left, Z=up)
        pts_body = np.stack([
             pts_raw[:, 2],   # body X = cam Z (forward)
            -pts_raw[:, 0],   # body Y = -cam X (left)
            -pts_raw[:, 1],   # body Z = -cam Y (up)
        ], axis=1).astype(np.float32)

        sensor_z = float(self.get_parameter('sensor_z').value)

        if self._world_R is not None:
            pts_world  = (pts_body @ self._world_R.T) + self._world_t
            sensor_pos = self._world_t.copy()
        else:
            pts_world  = pts_body.copy()
            pts_world[:, 2] += sensor_z
            sensor_pos = np.array([0.0, 0.0, sensor_z], dtype=np.float32)

        self._publish_cloud(pts_world, msg.header.stamp)

        accumulate = bool(self.get_parameter('accumulate').value)
        self.map.update(pts_world, sensor_pos, accumulate=accumulate)
        self._publish_occupancy(msg.header.stamp)

    def _read_xyz(self, msg: PointCloud2) -> np.ndarray | None:
        if msg.width == 0:
            return None
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        return pts[valid]

    def _publish_cloud(self, pts: np.ndarray, stamp) -> None:
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.is_dense = True
        msg.data = pts.astype(np.float32).tobytes()
        self.pub_cloud.publish(msg)

    def _publish_occupancy(self, stamp) -> None:
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = stamp
        grid.info.resolution = self.map.res
        grid.info.width = self.map.width
        grid.info.height = self.map.height
        grid.info.origin.position.x    = self.map.origin_x
        grid.info.origin.position.y    = self.map.origin_y
        grid.info.origin.orientation.w = 1.0  # identity quaternion (ROS2 default is 0,0,0,0 — invalid)
        data = np.full((self.map.height, self.map.width), -1, dtype=np.int8)
        lo, seen, _ = self.map.get_viz_data()
        data[seen & (lo <= 0)]   = 20
        data[seen & (lo > 0.5)]  = 100
        grid.data = data.flatten().tolist()
        self.pub_occ.publish(grid)


def main():
    rclpy.init()
    rclpy.spin(OccupancyNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
