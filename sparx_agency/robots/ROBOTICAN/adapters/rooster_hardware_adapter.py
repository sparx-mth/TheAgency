import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from fcu_driver_interfaces.msg import UAVState
from sparx_agency.robots.ROBOTICAN.adapters.sphera_ros2_ingestor import SpheraRos2Ingestor


class RoosterHardwareAdapter(Node):
    def __init__(self):
        super().__init__('rooster_hardware_adapter')

        # drone_id can be 'R2' for both Sphera or Rooster
        self.drone_id = self.declare_parameter('drone_id', 'R2').value

        # The ingestor manages hardware-specific GStreamer and KeepAlive
        self.ingestor = SpheraRos2Ingestor(pipeline=None, drone_id=self.drone_id)

        # On-demand publishers for the Jetson
        self.img_pub = self.create_publisher(Image, f'/{self.drone_id}/on_demand/image', 10)
        self.state_pub = self.create_publisher(UAVState, f'/{self.drone_id}/on_demand/state', 10)

        # Trigger service for the Jetson to "ask" for data
        self.srv = self.create_service(Trigger, f'/{self.drone_id}/trigger_capture', self.trigger_callback)

        # Initialize hardware stream
        self.ingestor.activate_video_hardware()

    def trigger_callback(self, request, response):
        # Retrieve the latest synchronized data from internal buffers
        frame = self.ingestor.get_latest_frame()
        state = self.ingestor.get_latest_state()

        if frame and state:
            self.img_pub.publish(frame)
            self.state_pub.publish(state)
            response.success = True
            response.message = "Successfully published coupled data pair"
        else:
            response.success = False
            response.message = "Hardware buffers are currently empty"
        return response


def main():
    rclpy.init()
    node = RoosterHardwareAdapter()
    rclpy.spin(node)
    rclpy.shutdown()