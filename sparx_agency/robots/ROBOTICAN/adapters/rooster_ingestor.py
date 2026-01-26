from __future__ import annotations
import numpy as np

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from fcu_driver_interfaces.msg import UAVState

from sparx_agency.core.common.types import Observation, RGBFrame, Intrinsics
from sparx_agency.robots.common.perception_converter import rgbframe_from_bgr
from sparx_agency.robots.common.state_converter import uav_state_to_pose_se3
from sparx_agency.robots.common.state_converter import costmap_to_occupancygrid


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def bgr_from_ros_image(msg: Image) -> np.ndarray:
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = (msg.encoding or "").lower()

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == w * 3:
        img = buf.reshape((h, w, 3))
    else:
        img = buf.reshape((h, step))[:, : w * 3].reshape((h, w, 3))

    if enc == "bgr8":
        return img.copy()
    if enc == "rgb8":
        return img[:, :, ::-1].copy()  # rgb->bgr
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def rgbframe_from_ros_image(msg: Image, frame_id: str) -> RGBFrame:
    bgr = bgr_from_ros_image(msg)
    rgb = bgr[:, :, ::-1]
    return RGBFrame(image=rgb.astype(np.uint8), stamp_sec=stamp_to_sec(msg.header.stamp), frame_id=frame_id)


class RoosterIngestor(Node):
    def __init__(self, pipeline, drone_id: str, intrinsics: Intrinsics, process_hz: float = 2.0):
        super().__init__("rooster_ingestor")
        self.pipeline = pipeline
        self.costmap = pipeline.costmap
        self.drone_id = drone_id
        self.intrinsics = intrinsics

        self.trigger_srv = f"/{drone_id}/trigger_capture"
        self.image_topic = f"/{drone_id}/camera/image_raw"
        self.state_topic = f"/{drone_id}/fcu/state"

        self.trigger_client = self.create_client(Trigger, self.trigger_srv)
        self.latest_img: Image | None = None
        self.latest_state: UAVState | None = None

        self._latest_bgr = None
        self._latest_stamp_sec = None
        self._latest_pose_msg = None

        self.last_img_t = -1.0
        self.last_state_t = -1.0

        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(UAVState, self.state_topic, self._state_cb, 10)

        period = 1.0 / max(process_hz, 1e-6)
        self.create_timer(period, self._tick)

        self.pub_occ = self.create_publisher(OccupancyGrid, "costmap/occupancy", 10)
        self.timer = self.create_timer(1.0 / process_hz, self._on_timer)

    def _image_cb(self, msg: Image) -> None:
        self.latest_img = msg

    def _state_cb(self, msg: UAVState) -> None:
        self.latest_state = msg

    def _tick(self) -> None:
        # request a fresh pair
        if self.trigger_client.service_is_ready():
            self.trigger_client.call_async(Trigger.Request())

        # process only if both are new
        if self.latest_img is None or self.latest_state is None:
            return

        img_t = stamp_to_sec(self.latest_img.header.stamp)
        st_t = stamp_to_sec(self.latest_state.header.stamp)
        if img_t <= self.last_img_t or st_t <= self.last_state_t:
            return

        try:
            rgb = rgbframe_from_ros_image(self.latest_img, frame_id=f"{self.drone_id}_camera")
            pose = uav_state_to_pose_se3(self.latest_state)

            obs = Observation(intrinsics=self.intrinsics, pose_map_base=pose, rgb=rgb)
            self.pipeline.step(obs)

            self.last_img_t = img_t
            self.last_state_t = st_t
            # TODO: fix this
            self._latest_bgr = bgr
            self._latest_stamp_sec = stamp_sec
            self._latest_pose_msg = msg


        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")

    def _on_timer(self):
        if self._latest_bgr is None or self._latest_pose_msg is None or self.intrinsics is None:
            return

        # pose conversion: pick the right one for your msg type
        pose = uav_state_to_pose_se3(self._latest_pose_msg)  # or odom_to_pose_se3(...)

        rgb = rgbframe_from_bgr(self._latest_bgr, stamp_sec=float(self._latest_stamp_sec), frame_id=self.frame_id)
        obs = Observation(intrinsics=self.intrinsics, pose_map_base=pose, rgb=rgb)

        self.pipeline.step(obs)

        stamp_msg = self.get_clock().now().to_msg()
        occ = costmap_to_occupancygrid(self.pipeline.costmap, stamp=stamp_msg, frame_id="map")
        self.pub_occ.publish(occ)
