#!/usr/bin/python3

import rclpy
from rclpy.node import Node, Parameter
from std_srvs.srv import SetBool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState
import time

'''
This code demonstrate sequence:
1. Set throttle 0
2. Arm
3. Takoff + Flight Up
4. Flight forward
'''
class Flying(Node):
    def __init__(self):
        super().__init__('flying_node')
        self.id = "R1"
        self.z = 0.0
        self.mode_num = KeepAlive.FLIGHT_MODE_MANUAL
        self.arm_state = False
        self.flight_mode = None

        self.force_arm = self.create_client(SetBool,f'/{self.id}/fcu/command/force_arm')


        self.manual_control = ManualControl()
        self.manual_control.x = 0.0
        self.manual_control.y = 0.0
        self.manual_control.z = 0.0
        self.manual_control.r = 0.0
        self.manual_control.buttons = 0
        self.manual_control_pub = self.create_publisher(ManualControl, f'/{self.id}/manual_control', 10)
        self.timer_arm = self.create_timer(1 / 40, self.publish_manual_control)

        self.state_node_sub = self.create_subscription(RoosterState, f'/{self.id}/state', self.drone_state_cb, 10)

        self.flight_mode_pub = self.create_publisher(KeepAlive, f'/{self.id}/keep_alive', 10)
        self.keep_alive_timer = self.create_timer(1, self.publish_keep_alive)

        self.step = 1
        self.step_timer = self.create_timer(1, self.step_timer_cb)
        self.takeoff_start_time = None

    def drone_state_cb(self, msg):
        if  self.arm_state != msg.armed or self.flight_mode != msg.flight_mode:
            print(f"state armed:{msg.armed} flight_mode:{msg.flight_mode}]")
            self.arm_state = msg.armed
            self.flight_mode = msg.flight_mode


    def step_timer_cb(self):
        if 1 == self.step:
            print("Set throttle 0")
            self.step = 2
        if 2 == self.step:
            print("Request Arm")
            self.arm_req = SetBool.Request()
            self.arm_req.data = True
            future = self.force_arm.call_async(self.arm_req)
            def arm_callback(response):
                result = response.result()
                if result.success:
                    print("Armed, can set throttle up")
                    self.step = 3
                else:
                    print("Arm failed")
            future.add_done_callback(arm_callback)
        if 3 == self.step:
            self.manual_control.z = 600.0
            self.takeoff_start_time = time.time()
            print(f"Set manual control to UP {self.manual_control}")
            self.step = 4
        if 4 == self.step and (time.time() - self.takeoff_start_time) > 2:
            self.manual_control.z = 568.0
            self.manual_control.x = 100.0
            print(f"Set manual control to Slow forward {self.manual_control}")


    def publish_keep_alive (self):
        print("keep alive")
        keep_alive_msg = KeepAlive()
        keep_alive_msg.is_active = True
        keep_alive_msg.requested_flight_mode = self.mode_num
        keep_alive_msg.command_reboot = False
        self.flight_mode_pub.publish(keep_alive_msg)


    def publish_manual_control(self):
        self.manual_control_pub.publish(self.manual_control)

def main(args=None):
    rclpy.init(args=args)
    flying_node = Flying()
    rclpy.spin(flying_node)
    flying_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()