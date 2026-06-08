import traceback
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs_py import point_cloud2
from nav_msgs.msg import  OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from fcu_driver_interfaces.msg import UAVState

from sparx_agency.core.common.types import PoseSE3, Observation, RGBFrame
from sparx_agency.robots.common.adapters.ros2.uav_state_adapter import UAVStateToPoseAdapter
from sparx_agency.robots.common.helpers import depth_to_vis_u8

from sparx_agency.robots.common.spatial_math import fov_to_intrinsics
from sparx_agency.robots.common.state_converter import costmap_to_occupancygrid
from sparx_agency.robots.common.txt_utils import  stamp_to_sec
from sparx_agency.robots.common.image_utils import ros_image_to_rgb_np, numpy_to_image_msg


class MappingTask(Node):
    def __init__(
        self,
        pipeline,
        drone_id: str = "R1",
        process_period_sec: float = 1.5,
        width: int = 640,
        height: int = 360,
        hfov_deg: float = 130.0,
        vfov_deg: float = 90.0,
    ):
        super().__init__("mapping_task")
        self.rgb_frame = None
        self.pipeline = pipeline
        self.drone_id = drone_id

        self.intr = fov_to_intrinsics(width, height, hfov_deg, vfov_deg)
        self.pose_adapter = UAVStateToPoseAdapter()

        self.last_img: Optional[Image] | None = None
        self.last_state: Optional[PoseSE3] | None = None
        self.last_processed_stamp = None  # (sec, nanosec)

        self.create_subscription(Image, f"/{drone_id}/camera/image_raw", self.image_cb, 10)
        self.create_subscription(UAVState, f"/{drone_id}/fcu/state", self.state_cb, 10)

        self.create_timer(float(process_period_sec), self._tick)

        # publishers
        self.map_frame = "map"  # Or get from parameter
        self.cam_frame = f"{drone_id}_camera"  # Match your URDF/Bag

        self.pub_depth_raw = self.create_publisher(Image, "depth/raw", 10)
        self.pub_depth_vis = self.create_publisher(Image, "depth/vis", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, "cloud/global", 10)
        self.pub_occ = self.create_publisher(OccupancyGrid, "costmap/occupancy", 10)
        self.get_logger().info(
            f"Bag mode (no CameraInfo): W={width} H={height} HFOV={hfov_deg} VFOV={vfov_deg} "
            f"-> fx={self.intr.fx:.1f} fy={self.intr.fy:.1f} cx={self.intr.cx:.1f} cy={self.intr.cy:.1f}. "
            f"Processing every {process_period_sec:.2f}s"
        )

    def image_cb(self, msg: Image):
        self.get_logger().debug(f"Image received: {msg.header.stamp}")
        self.last_img = msg
        rgb = ros_image_to_rgb_np(msg)
        stamp_sec = stamp_to_sec(msg.header.stamp)
        cam_frame : str = msg.header.frame_id

        self.get_logger().debug(
            f"Image shape: {rgb.shape} frame_id={cam_frame} stamp={stamp_sec:.3f} sec={stamp_sec:.0f}"
        )
        self.rgb_frame = RGBFrame(image=rgb, stamp_sec=stamp_sec, frame_id=cam_frame)

    def state_cb(self, msg: UAVState):
        self.get_logger().debug(f"State received: {msg.header.stamp}")
        self.pose_adapter.update(msg)

    def _tick(self):
        self.get_logger().info("tick")

        if self.rgb_frame is None:
            self.get_logger().warn("No image received yet.")
            return

        stamp = self.rgb_frame.stamp_sec
        if self.last_processed_stamp == stamp:
            return

        try:
            obs = Observation(
                rgb=self.rgb_frame,
                intrinsics=self.intr,
                pose_map_base=self.pose_adapter.get_pose(),  # bag mode: local update
                depth=None,
                cloud=None,
            )

            self.pipeline.step(obs)
            self.last_processed_stamp = self.rgb_frame.stamp_sec

        except Exception as e:
            self.get_logger().error(f"pipeline.step failed: {e}")

        depth_m = getattr(self.pipeline, "last_depth", None)
        cloud_global = getattr(self.pipeline, "last_cloud_global", None)

        # We use our utility to get the time for the message headers
        ros_stamp = self.last_img.header.stamp

        # 2. Publish Depth Visualization
        if depth_m is not None:
            # Raw Depth (32FC1)
            depth_msg = numpy_to_image_msg(
                depth_m.astype(np.float32),
                frame_id=self.cam_frame,
                stamp=ros_stamp,
                encoding="32fc1"
            )
            self.pub_depth_raw.publish(depth_msg)

            # Visual Depth (Mono8)
            vis_u8 = depth_to_vis_u8(
                depth_m,
                clip_min=self.pipeline.cfg.range_min,
                clip_max=self.pipeline.cfg.range_max
            )
            vis_msg = numpy_to_image_msg(
                vis_u8,
                frame_id=self.cam_frame,
                stamp=ros_stamp,
                encoding="mono8"
            )
            self.pub_depth_vis.publish(vis_msg)

        # 3. Publish Global PointCloud
        if cloud_global is not None:
            header = Header()
            header.stamp = ros_stamp
            header.frame_id = self.map_frame  # Use 'map' or 'odom'

            pts = cloud_global.astype(np.float32)
            # cloud_global is already in world frame thanks to the pipeline
            cloud_msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
            self.pub_cloud.publish(cloud_msg)

        # 4. Publish Occupancy Grid (The Costmap)
        try:
            # Convert the internal costmap to a ROS OccupancyGrid
            occ_msg = costmap_to_occupancygrid(
                self.pipeline.costmap,
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.map_frame
            )

            self.pub_occ.publish(occ_msg)


        except Exception as e:
            self.get_logger().error(f"Costmap publish failed: {e}")
