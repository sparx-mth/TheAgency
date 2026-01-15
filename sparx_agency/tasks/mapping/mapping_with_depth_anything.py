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
        self.drone_id = "R2"

        self.trigger_client = self.create_client(Trigger, f'/{self.drone_id}/trigger_capture')

        # Listen for the specific triggered output
        self.create_subscription(Image, f'/{self.drone_id}/on_demand/image', self.image_cb, 10)
        self.create_subscription(UAVState, f'/{self.drone_id}/on_demand/state', self.state_cb, 10)

        self.ready_img = None
        self.ready_state = None

    def image_cb(self, msg): self.ready_img = msg

    def state_cb(self, msg): self.ready_state = msg

    def request_new_frame(self):
        """Call this to 'ask' the PC for a new image + odom"""
        self.trigger_client.call_async(Trigger.Request()).add_done_callback(self.process_if_ready)

    def process_if_ready(self, future):
        # Ensure the Jetson has received both topics from the PC
        if self.ready_img and self.ready_state:
            # Map UAVState position and azimuth to standard Pose
            pose = Pose()
            pose.position.x = self.ready_state.position.x
            pose.position.y = self.ready_state.position.y
            pose.position.z = self.ready_state.position.z
            # (Note: Use your azimuth-to-quaternion logic here)

            self.pipeline.process_frame(self.ready_img, pose)

            # Reset buffers for next 'ask'
            self.ready_img = None
            self.ready_state = None