import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
import cv2
import numpy as np
import sys
import os
import struct

# Ensure we can import core modules
# Assuming this script is run as a module or installed package, imports should work.
# If running directly from source tree without install, might need path hacking, 
# but let's assume standard python path usage for now or that the user runs it via python -m.
from sparx_agency.core.mapping.costmap.create_costmap_from_image import ImageToCostmap, CostmapConfig

class DepthToCostmapNode(Node):
    def __init__(self):
        super().__init__('depth_to_costmap_node')
        
        # Parameters
        self.declare_parameter('input_pointcloud_topic', '/camera/point_cloud')
        self.declare_parameter('input_depth_topic', '/camera/depth')
        self.declare_parameter('input_camera_info_topic', '/camera/camera_info')
        self.declare_parameter('output_costmap_topic', '/output/costmap_2d')
        self.declare_parameter('output_cloud_topic', '/output/obstacle_cloud')

        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('obstacle_height', 0.1)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('map_size_m', 10.0)
        
        # Get parameters
        input_depth = self.get_parameter('input_depth_topic').value
        input_cam_info = self.get_parameter('input_camera_info_topic').value
        input_points = self.get_parameter('input_pointcloud_topic').value
        out_costmap = self.get_parameter('output_costmap_topic').value
        out_cloud = self.get_parameter('output_cloud_topic').value
        
        # Config
        self.cfg = CostmapConfig()
        self.cfg.grid_res = self.get_parameter('resolution').value
        self.cfg.max_range = self.get_parameter('max_range').value
        self.cfg.obstacle_height = self.get_parameter('obstacle_height').value
        self.cfg.max_height = self.get_parameter('max_height').value
        self.cfg.map_size_m = self.get_parameter('map_size_m').value
        
        # Mapper
        self.mapper = ImageToCostmap(self.cfg)
        self.camera_info_received = False
        
        # Subs/Pubs
        self.sub_info = self.create_subscription(
            CameraInfo,
            input_cam_info,
            self.camera_info_callback,
            10
        )
        
        self.sub_depth = self.create_subscription(
            Image,
            input_depth,
            self.depth_callback,
            10
        )

        self.sub_points = self.create_subscription(
            PointCloud2,
            input_points,
            self.pointcloud_callback,
            10
        )
        
        self.pub_costmap = self.create_publisher(OccupancyGrid, out_costmap, 10)
        self.pub_cloud = self.create_publisher(PointCloud2, out_cloud, 10)
        
        self.get_logger().info(f"DepthToCostmapNode started. Listening on {input_depth} and {input_points}")

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            K = np.array(msg.k).reshape(3,3)
            self.cfg.fx = K[0,0]
            self.cfg.fy = K[1,1]
            self.cfg.cx = K[0,2]
            self.cfg.cy = K[1,2]
            
            self.mapper.cfg = self.cfg 
            
            self.get_logger().info(f"Received CameraInfo: fx={self.cfg.fx}, fy={self.cfg.fy}, cx={self.cfg.cx}, cy={self.cfg.cy}")
            self.camera_info_received = True

    def pointcloud_callback(self, msg: PointCloud2):
        """
        Callback for direct PointCloud2 input.
        If received, we process this INSTEAD of depth? Or in addition?
        Typically one or the other is active.
        """
        try:
            # Convert PointCloud2 to Numpy
            # Simple manual parser for "x, y, z" float32
            # Assuming fields x, y, z exist and are standard.
            
            fmt_full = ''
            field_names = []
            offset_map = {}
            for field in msg.fields:
                field_names.append(field.name)
                offset_map[field.name] = field.offset
            
            if 'x' not in offset_map or 'y' not in offset_map or 'z' not in offset_map:
                self.get_logger().warn("PointCloud2 missing x,y,z fields")
                return
                
            # Efficient read? 
            # msg.data is bytes. point_step is stride.
            # Using np.frombuffer is fastest if packed and little endian.
            # If standard float32 x,y,z (maybe also rgb).
            
            # Let's hope it's standard.
            # point_step = msg.point_step
            # This is complex to do robustly without sensor_msgs_py.
            # We'll try a simplified struct iteration or numpy view if possible.
            
            # If width*height is huge, python loop is slow.
            # Try numpy view:
            dtype_list = []
            # We need to construct a dtype compatible with point_step
            # This is hard if padding exists.
            
            # Use a safe implementation: manually extract x,y,z with defaults
            # (Assuming user has correct x,y,z float32 at offsets)
            
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # We want to view 3 float32s at specific offsets per point.
            # Stride = msg.point_step
            
            count = msg.width * msg.height
            if len(raw_data) != count * msg.point_step:
                # self.get_logger().warn("Data length mismatch")
                pass # continued
            
            # Reshape to (count, point_step)
            # This allows slicing columns
            points_bytes = raw_data.reshape(count, msg.point_step)
            
            x_off = offset_map['x']
            y_off = offset_map['y']
            z_off = offset_map['z']
            
            # Extract bytes for each channel
            # We assume little-endian float32 (standard in ROS usually)
            # If big-endian, msg.is_bigendian would be True.
            
            xs_bytes = points_bytes[:, x_off:x_off+4]
            ys_bytes = points_bytes[:, y_off:y_off+4]
            zs_bytes = points_bytes[:, z_off:z_off+4]
            
            # Combine to (N, 3, 4) bytes? No using frombuffer on disconnected
            # Copy to contiguous?
            # Or just view?
            # xs = xs_bytes.view(dtype=np.float32) ... works if contiguous.
            # But sliced numpy array is not always contiguous in memory?
            # We can force copy: np.ascontiguousarray
            
            xs = np.frombuffer(np.ascontiguousarray(xs_bytes), dtype=np.float32)
            ys = np.frombuffer(np.ascontiguousarray(ys_bytes), dtype=np.float32)
            zs = np.frombuffer(np.ascontiguousarray(zs_bytes), dtype=np.float32)
            
            points_3d = np.stack([xs, ys, zs], axis=-1) # (N, 3)
            
            # Process
            result = self.mapper.generate_costmap_from_points(points_3d)
            
            if "costmap" in result:
                self.publish_costmap(result, msg.header)
                if "obstacle_points" in result:
                    self.publish_pointcloud(result["obstacle_points"], msg.header)
                    
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud_callback: {e}")

    def depth_callback(self, msg: Image):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for camera info before processing depth...", throttle_duration_sec=2.0)
            return

        try:
            # Manual conversion to avoid cv_bridge dependency issues if environment is minimal,
            # though usually cv_bridge is standard. Let's try manual decoding for float32.
            # encoding "32FC1"
            if msg.encoding == '32FC1':
                dtype = np.float32
                channels = 1
                itemsize = 4
            elif msg.encoding == '16UC1':
                dtype = np.uint16
                channels = 1
                itemsize = 2
            else:
                self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}")
                return

            # Buffer to numpy
            # msg.data is array.array or bytes depending on middleware? In rclpy it's usually bytes/array.
            # Using np.frombuffer
            img_data = np.frombuffer(msg.data, dtype=dtype)
            depth_img = img_data.reshape((msg.height, msg.width))
            
            # If 16UC1, convert to meters (millimeters usually)
            if dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
                
            # Run Mapping
            # Create a dummy RGB or pass None if mapper handles it (currently mapper signature asks for rgb)
            # Checking existing code: generate_costmap(self, rgb: np.ndarray, depth: np.ndarray)
            # It uses RGB for nothing critical? "rgb" var is unused in "generate_costmap" except passed to "masker.get_clean_mask" if we used it,
            # but in "generate_costmap" it calls "masker.depth_to_points" directly.
            # Wait, checking code...
            # Line 48: def generate_costmap(self, rgb: np.ndarray, depth: np.ndarray) -> dict:
            # It calls:  points = self.masker.depth_to_points(depth)
            # It DOES NOT use rgb anywhere else in generate_costmap logic shown.
            # So we can pass None or a dummy.
            
            # Pass dummy RGB
            dummy_rgb = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
            
            result = self.mapper.generate_costmap(dummy_rgb, depth_img)
            
            if "costmap" in result:
                # 1. Publish 2D Costmap
                self.publish_costmap(result, msg.header)
                
                # 2. Publish 3D Pointcloud
                if "obstacle_points" in result:
                    self.publish_pointcloud(result["obstacle_points"], msg.header)
                    
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")
            import traceback
            traceback.print_exc()

    def publish_costmap(self, result, header):
        costmap_img = result["costmap"] # H x W x 3 (BGR) or just 8-bit?
        # Logic in create_costmap returns 3 channel visual image:
        # costmap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        # We need generic OccupancyGrid (int8, 0-100, -1 unknown)
        
        # Convert visual costmap to occupancy
        # Red=[0,0,255] is occupied -> 100
        # Green=[0,255,0] is free -> 0
        # Black=[0,0,0] is unknown -> -1
        
        # Extract channels
        # If created by cv2, it's BGR. Red is last channel. Green is middle.
        b, g, r = cv2.split(costmap_img)
        
        occupied_mask = (r == 255)
        free_mask = (g == 255)
        
        # Flatten to 1D
        # OccupancyGrid data is row-major, starting from (0,0).
        # Standard map origin (0,0) is usually bottom-left.
        # Image (0,0) is top-left.
        # We need to flip vertically to match standard map orientation if we assume map origin is bottom-left?
        # In create_costmap, we did:
        #   rob_u = int((-min_x) / res)
        #   rob_v = int((-min_z) / res)
        #   cv2.circle(costmap, (rob_u, rob_v) ...)
        # Z increased downwards in image V?
        # In `generate_costmap`:
        #   vs = ((zs - min_z) / res).astype(np.int32)
        #   min_z is usually near 0. zs increases forward.
        #   So V increases as Z increases (Forward).
        #   Usually "Up" in image is "Forward" in map?
        #   If V increases down, and Z increases forward, then Down is Forward.
        #   Standard OccupancyGrid:
        #   Origin is bottom-left. Data is row by row.
        #   If we want "Forward" to be "Up" in visualization (standard map),
        #   we should flip the image so that Z-max is at the TOP (row 0? or row H?)
        #   Wait, ROS Map:
        #   index = y * width + x.
        #   (0,0) is bottom-left. +Y is up.
        #   If we want robot to look UP (+Y), then Z (forward) should map to Y.
        
        # Current image:
        # V (row) ~ Z (forward). So as Z increases, V increases (goes down image).
        # So "Forward" is DOWN in the generated image.
        # We want "Forward" to be UP (+Y) in the map.
        # So we must FLIP the image vertically.
        
        occ_grid_img = np.full(costmap_img.shape[:2], -1, dtype=np.int8)
        occ_grid_img[occupied_mask] = 100
        occ_grid_img[free_mask] = 0
        
        # Flip to make Forward (Z+) point UP
        occ_grid_img = cv2.flip(occ_grid_img, 0)
        
        grid_msg = OccupancyGrid()
        grid_msg.header = header
        grid_msg.header.frame_id = "base_link" # Or whatever the aligned local frame is
        
        h, w = occ_grid_img.shape
        grid_msg.info.width = w
        grid_msg.info.height = h
        grid_msg.info.resolution = result["resolution"]
        
        # Set Origin
        # original min_x, min_z were the bounds.
        # After flip, the bottom-left of the image corresponds to:
        #   X = min_x (left column)
        #   Y (visual) = max_z (bottom row before flip? No.)
        #   Let's trace.
        #   Pre-flip: Top-Left (0,0) -> (min_x, min_z). Bottom-Right (W,H) -> (max_x, max_z).
        #   Post-flip: Top-Left -> (min_x, max_z). Bottom-Left (0,0 index) -> (min_x, min_z).
        #   So data[0] (first cell) corresponds to Bottom-Left of image -> (min_x, min_z).
        #   So origin is (min_x, min_z).
        
        origin_x, origin_z = result["origin"]
        
        # However, our "aligned" points are in a frame where Y is UP (floor normal).
        # Z is forward. X is right.
        # Costmap frame (OccupancyGrid) is 2D: X (right), Y (up/forward).
        # So our Z maps to Map's Y. Our X maps to Map's X.
        
        grid_msg.info.origin.position.x = float(origin_x)
        grid_msg.info.origin.position.y = float(origin_z)
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0 # Identity
        
        grid_msg.data = occ_grid_img.flatten().tolist()
        
        self.pub_costmap.publish(grid_msg)

    def publish_pointcloud(self, points_3d, header):
        # points_3d is (N, 3) float32
        # Construct PointCloud2 manually or via helper
        
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = "base_link" # Aligned frame
        
        msg.height = 1
        msg.width = points_3d.shape[0]
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points_3d.shape[0]
        msg.is_dense = True
        
        # Ensure float32
        points_3d = points_3d.astype(np.float32)
        msg.data = points_3d.tobytes()
        
        self.pub_cloud.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
