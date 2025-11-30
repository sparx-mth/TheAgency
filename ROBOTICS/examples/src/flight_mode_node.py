#!/usr/bin/python3

import rclpy
from rclpy.node import Node, Parameter
from rooster_handler_interfaces.msg import KeepAlive


class FlightMode(Node):
    def __init__(self):
        super().__init__('flight_mode_node')
        self.id = "R1"
        self.flight_mode_node = self.create_publisher(KeepAlive, f'/{self.id}/keep_alive', 10)
        self.get_logger().info("Start Sending Requests")
        self.send_reqs()

    def send_reqs (self):
        self.flight_mode = KeepAlive()
        self.flight_mode.is_active = True
        self.flight_mode.requested_flight_mode = 1
        # LIGHT_MODE_NONE=0
        # FLIGHT_MODE_GROUND_ROLL=1
        # FLIGHT_MODE_MANUAL=2
        # FLIGHT_MODE_POSITION=3
        # FLIGHT_MODE_ALTITUDE=4
        # FLIGHT_MODE_ACRO=5
        # FLIGHT_MODE_STABILIZED=6
        # FLIGHT_MODE_TAKEOFF=7
        # FLIGHT_MODE_LAND=8
        self.flight_mode.command_reboot = False
        self.timer = self.create_timer(1.0, self.change_flight_mode)

    def change_flight_mode(self):
        self.flight_mode_node.publish(self.flight_mode)


def main(args=None):
    rclpy.init(args=args)
    flight_mode_node = FlightMode()
    rclpy.spin(flight_mode_node)
    flight_mode_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()