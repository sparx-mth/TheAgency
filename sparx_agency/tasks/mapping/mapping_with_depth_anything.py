import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from fcu_driver_interfaces.msg import UAVState
from geometry_msgs.msg import Pose


class MappingTask(Node):
    def __init__(self, pipeline):
        super().__init__('mapping_task')
        self.pipeline = pipeline
        self.drone_id = "R1"

        self.trigger_client = self.create_client(Trigger, f'/{self.drone_id}/trigger_capture')

        # Listen for the specific triggered output
        self.create_subscription(Image, f'/{self.drone_id}/camera/image_raw', self.image_cb, 10)
        self.create_subscription(UAVState, f'/{self.drone_id}/fcu/state', self.state_cb, 10)

        self.ready_img = None
        self.ready_state = None

    def image_cb(self, msg): self.ready_img = msg

    def state_cb(self, msg): self.ready_state = msg

    def request_new_frame(self):
        """Ask the PC to publish one new image + state."""
        if not self.trigger_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service /{self.drone_id}/trigger_capture not available")
            return

        fut = self.trigger_client.call_async(Trigger.Request())
        fut.add_done_callback(self._on_trigger_done)

    def _on_trigger_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"Trigger call failed: {e}")
            return

        if not resp.success:
            self.get_logger().warn(f"Trigger rejected: {resp.message}")
            return

        # Trigger accepted: wait briefly for the next image/state to arrive
        self._wait_deadline_ns = self.get_clock().now().nanoseconds + int(0.5e9)  # 0.5s
        self._poll_for_ready()

    def _poll_for_ready(self):
        if self.ready_img is not None and self.ready_state is not None:
            self._process_ready_pair()
            return

        if self.get_clock().now().nanoseconds > getattr(self, "_wait_deadline_ns", 0):
            self.get_logger().warn("Triggered capture, but image/state did not arrive before timeout")
            return

        # poll again shortly
        self.create_timer(0.02, self._poll_for_ready, callback_group=None)  # 20 ms

    def _process_ready_pair(self):
        pose = Pose()
        pose.position.x = self.ready_state.position.x
        pose.position.y = self.ready_state.position.y
        pose.position.z = self.ready_state.position.z

        self.pipeline.process_frame(self.ready_img, pose)

        self.ready_img = None
        self.ready_state = None 