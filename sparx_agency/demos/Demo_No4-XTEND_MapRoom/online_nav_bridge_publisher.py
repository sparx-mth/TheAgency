import asyncio
import rclpy
import json
import websockets
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

# Base automation class
from sparx_agency.robots.XTEND.automation import ControllerAutomation
from sparx_agency.robots.XTEND.xtend_rtsp_image_publisher import LatestFrameGrabber


# class LatestFrameGrabber:
#     """Thread-safe RTSP grabber for high-performance frame retrieval."""
#
#     def __init__(self, uri: str, backend: str = "gstreamer"):
#         self.uri = uri
#         self.backend = backend
#         self.lock = threading.Lock()
#         self.latest_frame = None
#         self.latest_stamp = 0.0
#         self.running = False
#         self.thread = None
#         self.cap = None
#
#     def start(self):
#         self.running = True
#         self.cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG if self.backend == "ffmpeg" else cv2.CAP_ANY)
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.thread = threading.Thread(target=self._loop, daemon=True)
#         self.thread.start()
#
#     def _loop(self):
#         while self.running:
#             ok, frame = self.cap.read()
#             if ok and frame is not None:
#                 with self.lock:
#                     self.latest_frame = frame
#                     self.latest_stamp = time.time()
#             else:
#                 time.sleep(0.01)
#
#     def get_latest(self):
#         with self.lock:
#             if self.latest_frame is None:
#                 return None, 0.0
#             return self.latest_frame.copy(), self.latest_stamp
#
#     def stop(self):
#         self.running = False
#         if self.thread:
#             self.thread.join(timeout=2.0)
#         if self.cap:
#             self.cap.release()


class OnlineNavBridgePublisher(ControllerAutomation):
    def __init__(self, host, port, frequency, robot_uid, rtsp_uri):
        super().__init__(host, port, frequency, robot_uid)

        # Event loop and Queue for Nav Commands
        self.loop = asyncio.get_event_loop()
        self.cmd_queue = asyncio.Queue()
        self.bridge = CvBridge()

        # ROS 2 Setup
        self.ros_node = rclpy.create_node('drone_bridge_publisher_node')

        # Publisher: FPV Camera
        self.image_pub = self.ros_node.create_publisher(
            Image,
            '/xtend/image_raw',
            qos_profile_sensor_data
        )

        # Subscriber: Navigation Commands
        self.subscription = self.ros_node.create_subscription(
            String,
            '/drone/cmd_nav',
            self.ros_callback,
            10
        )

        # Performance Frame Grabber
        self.grabber = LatestFrameGrabber(rtsp_uri, backend='gstreamer')
        self.grabber.start()

    def ros_callback(self, msg):
        """Moves ROS messages into the Asyncio loop safely."""
        try:
            data = json.loads(msg.data)
            self.loop.call_soon_threadsafe(self.cmd_queue.put_nowait, data)
        except Exception as e:
            self.ros_node.get_logger().error(f"Failed to parse command: {e}")

    async def image_publish_loop(self):
        """Independent loop to publish frames at 30Hz with cropping."""
        # Define crop parameters from your config
        crop_left, crop_top = 108, 70
        crop_width, crop_height = 504, 280
        sleep_time = 1.0 / max(self.frequency, 1e-6)
        print("✓ Image Publisher Active with Cropping")
        while True:
            frame, _ = self.grabber.get_latest()
            if frame is not None:
                # Apply the crop slicing: frame[y:y+h, x:x+w]
                cropped_frame = frame[crop_top:crop_top + crop_height,
                crop_left:crop_left + crop_width]

                msg = self.bridge.cv2_to_imgmsg(cropped_frame, encoding="bgr8")
                msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                msg.header.frame_id = "xtend_camera"

                self.image_pub.publish(msg)

            await asyncio.sleep(sleep_time)

    async def dynamic_executor(self):
        """Processes navigation commands while ignoring stale inputs."""
        print("✓ Navigation Executor Active")
        while True:
            command = await self.cmd_queue.get()

            action = command.get("action")
            thrust = command.get("value", 500)
            duration = command.get("duration", 0)

            # Execution (including built-in API stabilization)
            if action == "forward":
                await self.move_forward(duration=duration, value=thrust)
            elif action == "backward":
                await self.move_backward(duration=duration, value=thrust)
            elif action == "left":
                await self.move_left(duration=duration, value=thrust)
            elif action == "right":
                await self.move_right(duration=duration, value=thrust)
            elif action == "takeoff":
                await self.takeoff()
            elif action == "land":
                await self.land()
                self.cmd_queue.task_done()
                break

            self.cmd_queue.task_done()

            # Flush Logic: Ignore any commands that arrived during the move
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                    self.cmd_queue.task_done()
                except asyncio.QueueEmpty:
                    break

    async def run_bridge(self):
        """Main entry point to run all concurrent tasks."""
        ros_thread = asyncio.to_thread(rclpy.spin, self.ros_node)

        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"✓ Connected to Drone at {self.uri}")
                await asyncio.gather(
                    self.send_message(websocket),
                    self.receive_message(websocket),
                    self.dynamic_executor(),
                    self.image_publish_loop(),
                    ros_thread
                )
        finally:
            self.grabber.stop()
            self.ros_node.destroy_node()
            rclpy.shutdown()


async def main():
    rclpy.init()
    bridge = OnlineNavBridgePublisher(
        host="192.0.0.15",
        port=8000,
        frequency=15.0,
        robot_uid="drnb177ede2",
        rtsp_uri="rtsp://192.0.0.15:8510/active_drone_fpv"
    )
    await bridge.run_bridge()


if __name__ == "__main__":
    asyncio.run(main())