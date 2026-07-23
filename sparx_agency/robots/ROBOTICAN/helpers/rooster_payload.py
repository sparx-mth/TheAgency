# rooster_payload.py
"""Battery telemetry and video-stream control for one Rooster drone.

Split out of rooster_unit.py to keep flight control (arm/disarm/takeoff/
land/move) separate from payload concerns (battery, camera). RoosterUnit
composes one of these per drone the same way it owns every other piece of
FCU-facing I/O.
"""

from __future__ import annotations

from typing import Callable, Optional

from rclpy.node import Node
from std_msgs.msg import Bool
from fcu_driver_interfaces.msg import Battery
from video_handler_interfaces.srv import SetVideoMode


class RoosterPayload:
    """Owns battery telemetry and video-stream on/off for one Rooster ID."""

    def __init__(
        self,
        node: Node,
        rooster_id: str,
        video_host: str = "127.0.0.1",
        video_port: int = 5001,
        video_width: int = 540,
        video_height: int = 360,
    ):
        self.id = rooster_id
        self.node = node
        self.video_host = str(video_host)
        self.video_port = int(video_port)
        # 540x360 (3:2) matches the drone's documented "low" preset (ICD
        # 6.2.4) - the camera's native aspect is 3:2, not the 16:9 we used
        # to derive height from width, which was distorting the stream.
        self.video_width = int(video_width)
        self.video_height = int(video_height)

        self.battery_voltage = 0.0
        self.battery_percentage = 0.0
        self.video_on = False

        self.gcs_keep_alive_pub = node.create_publisher(
            Bool, f"/{rooster_id}/gcs_keep_alive", 10)
        self.battery_sub = node.create_subscription(
            Battery, f"/{rooster_id}/fcu/battery", self._battery_cb, 10)
        self.video_client = node.create_client(
            SetVideoMode, f"/{rooster_id}/video_handler/set_video_mode")

    def _battery_cb(self, msg: Battery):
        self.battery_voltage = msg.voltage
        self.battery_percentage = msg.percentage

    def publish_gcs_keep_alive(self):
        """Required by video_handler - the video stream turns off if this
        stops arriving for 5s, independent of the flight-control keep_alive."""
        msg = Bool()
        msg.data = True
        self.gcs_keep_alive_pub.publish(msg)

    def set_video(self, on: bool, on_done: Optional[Callable[[bool], None]] = None):
        if not self.video_client.service_is_ready():
            self.node.get_logger().warn(f"[{self.id}] video_handler service not ready.")
            if on_done:
                on_done(False)
            return
        req = SetVideoMode.Request()
        req.camera_id = 0
        req.playing = bool(on)
        req.host = self.video_host
        req.port = self.video_port
        req.resolution_width = self.video_width
        req.resolution_height = self.video_height
        req.recording = False
        req.bitrate = SetVideoMode.Request.BITRATE_1500000
        req.fps = 0
        future = self.video_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception as e:
                self.node.get_logger().error(f"[{self.id}] set_video_mode error: {e}")
                if on_done:
                    on_done(False)
                return
            if resp.success:
                self.video_on = bool(on)
                self.node.get_logger().info(f"[{self.id}] Video {'on' if on else 'off'}.")
            else:
                self.node.get_logger().warn(f"[{self.id}] set_video_mode refused: {resp.message}")
            if on_done:
                on_done(resp.success)

        future.add_done_callback(_done)
