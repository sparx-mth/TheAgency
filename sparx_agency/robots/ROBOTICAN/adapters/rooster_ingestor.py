from __future__ import annotations
import numpy as np

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from fcu_driver_interfaces.msg import UAVState

from sparx_agency.core.common.types import Observation, RGBFrame, Intrinsics
from sparx_agency.robots.common.txt_utils import stamp_to_sec
from sparx_agency.robots.common.perception_converter import rgbframe_from_bgr, rgbframe_from_ros_image
from sparx_agency.robots.common.state_converter import uav_state_to_pose_se3
from sparx_agency.robots.common.state_converter import costmap_to_occupancygrid



class RoosterIngestor(Node):
    def __init__(self, pipeline, drone_id: str, intrinsics: Intrinsics, process_hz: float = 2.0):
        super().__init__("rooster_ingestor")
        self.pipeline = pipeline
        self.drone_id = drone_id
        self.intrinsics = intrinsics

        self.trigger_srv = f"/{drone_id}/trigger_capture"
        self.image_topic = f"/{drone_id}/camera/image_raw"
        self.state_topic = f"/{drone_id}/fcu/state"

        self.trigger_client = self.create_client(Trigger, self.trigger_srv)
        self.latest_img: Image | None = None
        self.latest_state: UAVState | None = None

        self.last_img_t = -1.0
        self.last_state_t = -1.0

        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(UAVState, self.state_topic, self._state_cb, 10)

        self.pub_occ = self.create_publisher(OccupancyGrid, "costmap/occupancy", 10)

        period = 1.0 / max(process_hz, 1e-6)
        self.create_timer(period, self._tick)

    def _image_cb(self, msg: Image) -> None:
        self.latest_img = msg

    def _state_cb(self, msg: UAVState) -> None:
        self.latest_state = msg

    def _tick(self) -> None:
        # Request a fresh pair
        if self.trigger_client.service_is_ready():
            self.trigger_client.call_async(Trigger.Request())

        # Process only if both exist
        if self.latest_img is None or self.latest_state is None:
            return

        img_t = stamp_to_sec(self.latest_img.header.stamp)
        st_t = stamp_to_sec(self.latest_state.header.stamp)

        # Process only if both are new
        if img_t <= self.last_img_t or st_t <= self.last_state_t:
            return

        try:
            rgb = rgbframe_from_ros_image(self.latest_img, frame_id=f"{self.drone_id}_camera")
            pose = uav_state_to_pose_se3(self.latest_state)

            obs = Observation(intrinsics=self.intrinsics, pose_map_base=pose, rgb=rgb)
            self.pipeline.step(obs)

            # Mark consumed
            self.last_img_t = img_t
            self.last_state_t = st_t

            # Export costmap right here (bag-pattern)
            occ = costmap_to_occupancygrid(
                self.pipeline.costmap,
                stamp=self.latest_img.header.stamp,  # align with frame time
                frame_id="map",
            )
            self.pub_occ.publish(occ)

        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")