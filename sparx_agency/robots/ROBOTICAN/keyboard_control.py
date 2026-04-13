#!/usr/bin/env python3
import sys
import select
import termios
import tty
import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive

MSG = """
Control Your Drone!
---------------------------
Moving around:
        w
    a   s    d
        x

t/l: takeoff/land
q/e: increase/decrease speed
A/D: rotate left/right
r/f: rise/fall

CTRL-C to quit
---------------------------
"""

class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.declare_parameter('drone_id', 'R1')
        self.declare_parameter('step_xy', 50.0)
        self.declare_parameter('step_z', 50.0)
        self.declare_parameter('step_yaw', 80.0)
        self.declare_parameter('max_cmd', 1000.0)
        self.declare_parameter('publish_hz', 20.0)

        self.drone_id = self.get_parameter('drone_id').value
        self.step_xy = float(self.get_parameter('step_xy').value)
        self.step_z = float(self.get_parameter('step_z').value)
        self.step_yaw = float(self.get_parameter('step_yaw').value)
        self.max_cmd = float(self.get_parameter('max_cmd').value)

        self.cmd = ManualControl()
        self.cmd.x = 0.0
        self.cmd.y = 0.0
        self.cmd.z = 0.0
        self.cmd.r = 0.0
        self.cmd.buttons = 0

        self.manual_pub = self.create_publisher(
            ManualControl, f'/{self.drone_id}/manual_control', 10
        )
        self.keep_alive_pub = self.create_publisher(
            KeepAlive, f'/{self.drone_id}/keep_alive', 10
        )
        self.gcs_keep_alive_pub = self.create_publisher(
            Bool, f'/{self.drone_id}/gcs_keep_alive', 10
        )

        self.arm_client = self.create_client(
            SetBool, f'/{self.drone_id}/fcu/command/force_arm'
        )

        hz = float(self.get_parameter('publish_hz').value)
        self.create_timer(1.0 / hz, self.publish_manual)
        self.create_timer(1.0, self.publish_keep_alive)

        self.get_logger().info(f'Keyboard teleop for {self.drone_id}')
        print(MSG)

    def clamp(self, v: float) -> float:
        return max(-self.max_cmd, min(self.max_cmd, v))

    def publish_manual(self):
        self.manual_pub.publish(self.cmd)

    def publish_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = KeepAlive.FLIGHT_MODE_MANUAL
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

        gcs = Bool()
        gcs.data = True
        self.gcs_keep_alive_pub.publish(gcs)

    def arm_and_takeoff(self):
        if not self.arm_client.service_is_ready():
            self.get_logger().warn('arm service not ready')
            return

        req = SetBool.Request()
        req.data = True
        future = self.arm_client.call_async(req)

        def done_cb(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info('Armed')
                    self.cmd.z = 600.0
                else:
                    self.get_logger().warn(f'Arm failed: {resp.message}')
            except Exception as e:
                self.get_logger().error(f'Arm call failed: {e}')

        future.add_done_callback(done_cb)

    def land(self):
        self.cmd.x = 0.0
        self.cmd.y = 0.0
        self.cmd.r = 0.0
        self.cmd.z = 400.0
        self.get_logger().info('Landing command sent')

    def stop(self):
        self.cmd.x = 0.0
        self.cmd.y = 0.0
        self.cmd.z = 0.0
        self.cmd.r = 0.0

    def handle_key(self, key: str):
        if key == 'w':
            self.cmd.x = self.clamp(self.cmd.x + self.step_xy)
        elif key == 'x':
            self.cmd.x = self.clamp(self.cmd.x - self.step_xy)
        elif key == 'a':
            self.cmd.y = self.clamp(self.cmd.y - self.step_xy)
        elif key == 'd':
            self.cmd.y = self.clamp(self.cmd.y + self.step_xy)
        elif key == 'r':
            self.cmd.z = self.clamp(self.cmd.z + self.step_z)
        elif key == 'f':
            self.cmd.z = self.clamp(self.cmd.z - self.step_z)
        elif key == 'A':
            self.cmd.r = self.clamp(self.cmd.r + self.step_yaw)
        elif key == 'D':
            self.cmd.r = self.clamp(self.cmd.r - self.step_yaw)
        elif key == 'q':
            self.step_xy *= 1.1
            self.step_z *= 1.1
            self.step_yaw *= 1.1
            self.get_logger().info(f'Speed scale up: xy={self.step_xy:.1f}')
        elif key == 'e':
            self.step_xy *= 0.9
            self.step_z *= 0.9
            self.step_yaw *= 0.9
            self.get_logger().info(f'Speed scale down: xy={self.step_xy:.1f}')
        elif key == 's':
            self.stop()
        elif key == 't':
            self.arm_and_takeoff()
        elif key == 'l':
            self.land()

def keyboard_loop(node: KeyboardTeleopNode):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while rclpy.ok():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                key = sys.stdin.read(1)
                node.handle_key(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()

    th = threading.Thread(target=keyboard_loop, args=(node,), daemon=True)
    th.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()