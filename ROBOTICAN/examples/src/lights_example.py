#!/usr/bin/python3

import signal
import rclpy
import time
from rclpy.node import Node, Parameter
from io_driver_interfaces.msg import LedsData
from io_driver_interfaces.srv import ChangeLedState
from std_srvs.srv import SetBool

'''
This code demonstrate drone lED control
'''
class IODriver(Node):
    def __init__(self):
        super().__init__('io_driver_node')
        self.led_type = 0
        self.led_power = 0
        self.led_power = 0
        self.previous_led_type = 0

        self.id = "R1"
        self.led_state = self.create_client(ChangeLedState,f'/{self.id}/io_driver/change_led_state')
        self.toggle_dazzle_mode_srv = self.create_client(SetBool, f'/{self.id}/io_driver/dazzle')
        self.get_logger().info("Start Sending Requests")
        self.timer = self.create_timer(4, self.set_leds)
        self.step  = 0
        self.set_leds()

    def set_leds (self):
        change_led_state_request = ChangeLedState.Request()
        if self.step == 0:
            self.get_logger().info(f"Step#{self.step} Turn dazzle mode")
            self.dazzle_mode = SetBool.Request()
            self.dazzle_mode.data = False
            future = self.toggle_dazzle_mode_srv.call_async(self.dazzle_mode)
            future.add_done_callback(self.dazzle_response_callback)

        if self.step < 8:
            if self.step < 5:
                change_led_state_request.type = ChangeLedState.Request.LED_TYPE_DAY
            else:
                change_led_state_request.type = ChangeLedState.Request.LED_TYPE_IR

            self.led_power = (self.led_power + 10) % 40
            change_led_state_request.power = self.led_power

            self.get_logger().info(f"Step#{self.step} Request light state {change_led_state_request}")
            future = self.led_state.call_async(change_led_state_request)
            future.add_done_callback(self.change_led_state_response_callback)

        if self.step == 8:
            self.get_logger().info(f"Step#{self.step} Turn dazzle mode")
            self.dazzle_mode = SetBool.Request()
            self.dazzle_mode.data = True
            future = self.toggle_dazzle_mode_srv.call_async(self.dazzle_mode)
            future.add_done_callback(self.dazzle_response_callback)
        if self.step == 9:
            self.step = 0


    def get_led_state(self):
        self.get_logger(f"Led State\n type: {self.led_type}\n , power{self.led_power}")

    def dazzle_response_callback(self,response):
        self.step = self.step  + 1 if self.step < 9 else 0
        response = response.result()
        self.get_logger().info(f"Dazzle mode response {response.success} go to step #{self.step}")

    def change_led_state_response_callback(self,response):
        self.step = self.step  + 1 if self.step < 9 else 0
        response = response.result()
        self.get_logger().info(f"change led state {response.success} go to step #{self.step}")


def main(args=None):
    rclpy.init(args=args)
    io_driver_node = IODriver()
    rclpy.spin(io_driver_node)
    io_driver_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()