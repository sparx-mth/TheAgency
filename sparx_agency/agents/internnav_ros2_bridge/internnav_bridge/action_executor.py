#!/usr/bin/env python3
"""
Action Executor - Converts discrete actions to timed velocity commands.

Use this when your simulation needs continuous velocity but the model outputs discrete actions.
"""

import time
import threading
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ActionState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"


class ActionExecutor(Node):
    """Converts discrete actions to timed velocity commands."""

    def __init__(self):
        super().__init__('action_executor')

        # Parameters
        self.declare_parameter('action_topic', '/navigation/action')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('forward_velocity', 0.25)
        self.declare_parameter('forward_distance', 0.25)
        self.declare_parameter('turn_velocity', 0.5236)  # 30 deg/s
        self.declare_parameter('turn_angle', 0.5236)  # 30 deg
        self.declare_parameter('control_rate', 20.0)

        action_topic = self.get_parameter('action_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.forward_vel = self.get_parameter('forward_velocity').value
        self.forward_dist = self.get_parameter('forward_distance').value
        self.turn_vel = self.get_parameter('turn_velocity').value
        self.turn_angle = self.get_parameter('turn_angle').value
        control_rate = self.get_parameter('control_rate').value

        # State
        self.state = ActionState.IDLE
        self.lock = threading.Lock()
        self.target_twist = Twist()
        self.exec_start = 0.0
        self.exec_duration = 0.0

        # Action aliases
        self.aliases = {
            'forward': 'MOVE_FORWARD', 'move_forward': 'MOVE_FORWARD', 'ahead': 'MOVE_FORWARD',
            'left': 'TURN_LEFT', 'turn_left': 'TURN_LEFT',
            'right': 'TURN_RIGHT', 'turn_right': 'TURN_RIGHT',
            'stop': 'STOP', 'done': 'STOP', 'hover': 'STOP',
        }

        # ROS interfaces
        self.action_sub = self.create_subscription(String, action_topic, self._action_cb, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.control_timer = self.create_timer(1.0 / control_rate, self._control_loop)

        self.get_logger().info(f"ActionExecutor: {action_topic} -> {cmd_vel_topic}")

    def _action_cb(self, msg: String):
        action = msg.data.strip().lower()
        normalized = self.aliases.get(action, action.upper())

        with self.lock:
            if self.state == ActionState.IDLE:
                self._start_action(normalized)

    def _start_action(self, action: str):
        self.state = ActionState.EXECUTING
        self.exec_start = time.time()

        if action == 'MOVE_FORWARD':
            self.target_twist.linear.x = self.forward_vel
            self.target_twist.angular.z = 0.0
            self.exec_duration = self.forward_dist / self.forward_vel
        elif action == 'TURN_LEFT':
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = self.turn_vel
            self.exec_duration = self.turn_angle / self.turn_vel
        elif action == 'TURN_RIGHT':
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = -self.turn_vel
            self.exec_duration = self.turn_angle / self.turn_vel
        else:  # STOP or unknown
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = 0.0
            self.exec_duration = 0.1

        self.get_logger().info(f"Executing {action} for {self.exec_duration:.2f}s")

    def _control_loop(self):
        with self.lock:
            if self.state == ActionState.IDLE:
                self._stop()
                return

            if time.time() - self.exec_start >= self.exec_duration:
                self._stop()
                self.state = ActionState.IDLE
            else:
                self.cmd_vel_pub.publish(self.target_twist)

    def _stop(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutor()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()