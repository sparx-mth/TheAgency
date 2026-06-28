import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
import numpy as np
import time
import struct
import threading
import sys
import os

# Adjust path to find the module if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sparx_agency.core.mapping.ros_node import DepthToCostmapNode

class MockCloudPublisher(Node):
    def __init__(self):
        super().__init__('mock_cloud_pub')
        self.pub = self.create_publisher(PointCloud2, '/camera/point_cloud', 10)
        self.timer = self.create_timer(1.0, self.publish_mock_cloud)
        
    def publish_mock_cloud(self):
        # Create a synthetic cloud
        # 1. Floor points: Y=0 (or close to 0 if camera is at 0)
        # Wait, node logic assumes camera is at some height?
        # Node uses "generate_costmap_from_points" which does "Fit Floor Plane".
        # If we provide perfect plane:
        # Camera Frame: Y down, Z forward.
        # Floor should be Y positive (below camera).
        # Let's put floor at Y = 1.0 meter (Camera is 1m up).
        
        # Grid of floor points
        x = np.linspace(-2, 2, 20)
        z = np.linspace(1, 5, 20)
        xv, zv = np.meshgrid(x, z)
        yv = np.ones_like(xv) * 1.0 # Floor at Y=1
        
        floor_pts = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
        
        # Obstacle: Wall at Z=3, X in [-1, 1], Y from 0 to 1
        ox = np.linspace(-1, 1, 10)
        oy = np.linspace(0, 1, 10)
        oxv, oyv = np.meshgrid(ox, oy)
        ozv = np.ones_like(oxv) * 3.0
        
        obs_pts = np.stack([oxv.flatten(), oyv.flatten(), ozv.flatten()], axis=-1)
        
        all_pts = np.vstack([floor_pts, obs_pts]).astype(np.float32)
        
        # Create PointCloud2 msg manually
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        
        msg.height = 1
        msg.width = len(all_pts)
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = all_pts.tobytes()
        
        self.pub.publish(msg)
        self.get_logger().info("Published mock cloud")

class TestSubscriber(Node):
    def __init__(self):
        super().__init__('test_subscriber')
        self.sub = self.create_subscription(OccupancyGrid, '/output/costmap_2d', self.callback, 10)
        self.received = False
        
    def callback(self, msg):
        self.get_logger().info(f"Received Costmap! Size: {msg.info.width}x{msg.info.height}")
        # Validate content?
        # Expect some occupied cells (100)
        data = np.array(msg.data)
        occupied_count = np.sum(data == 100)
        self.get_logger().info(f"Occupied cells: {occupied_count}")
        
        if occupied_count > 0:
            self.get_logger().info("TEST PASSED: Obstacles detected.")
            self.received = True
            # We can exit?
            raise SystemExit(0) # Proper exit

def main():
    rclpy.init()
    
    # Create Nodes
    mock_pub = MockCloudPublisher()
    dut = DepthToCostmapNode() # Device Under Test
    test_sub = TestSubscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(mock_pub)
    executor.add_node(dut)
    executor.add_node(test_sub)
    
    print("Starting Test...", flush=True)
    
    try:
        executor.spin()
    except SystemExit:
        print("Test Finished.")
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
