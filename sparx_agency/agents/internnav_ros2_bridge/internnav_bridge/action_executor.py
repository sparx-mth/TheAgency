#!/usr/bin/env python3
"""
Action Executor Node

This node subscribes to discrete navigation actions and converts them
to timed velocity commands, executing each action to completion before
accepting the next one.

This is useful when your simulation expects continuous velocity commands
but the InternNav model outputs discrete actions.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import math
import time
import threading
from enum import Enum
from typing import Optional, Dict

from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ActionState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"


class ActionExecutor(Node):
    """
    Converts discrete navigation actions to timed velocity commands.
    
    This node:
    1. Subscribes to discrete action commands (e.g., "forward", "left", "right", "stop")
    2. Executes each action by publishing velocity commands for a specified duration
    3. Publishes status updates during execution
    """
    
    def __init__(self):
        super().__init__('action_executor')
        
        # Declare parameters
        self.declare_parameter('action_topic', '/navigation/action')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('status_topic', '/action_executor/status')
        
        # Movement parameters
        self.declare_parameter('forward_velocity', 0.25)  # m/s
        self.declare_parameter('forward_distance', 0.25)  # meters
        self.declare_parameter('turn_velocity', 0.5236)  # rad/s (30 deg/s)
        self.declare_parameter('turn_angle', 0.5236)  # radians (30 degrees)
        self.declare_parameter('control_rate', 20.0)  # Hz
        
        # Get parameters
        action_topic = self.get_parameter('action_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        status_topic = self.get_parameter('status_topic').value
        
        self.forward_velocity = self.get_parameter('forward_velocity').value
        self.forward_distance = self.get_parameter('forward_distance').value
        self.turn_velocity = self.get_parameter('turn_velocity').value
        self.turn_angle = self.get_parameter('turn_angle').value
        self.control_rate = self.get_parameter('control_rate').value
        
        # State
        self.state = ActionState.IDLE
        self.current_action: Optional[str] = None
        self.action_lock = threading.Lock()
        self.action_queue = []
        
        # Action mapping (maps incoming action strings to internal actions)
        self.action_aliases = {
            # Forward
            'forward': 'MOVE_FORWARD',
            'move_forward': 'MOVE_FORWARD',
            'ahead': 'MOVE_FORWARD',
            'straight': 'MOVE_FORWARD',
            # Left
            'left': 'TURN_LEFT',
            'turn_left': 'TURN_LEFT',
            # Right
            'right': 'TURN_RIGHT',
            'turn_right': 'TURN_RIGHT',
            # Stop
            'stop': 'STOP',
            'done': 'STOP',
            'finish': 'STOP',
        }
        
        # Callback groups
        self.action_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Subscribers
        self.action_sub = self.create_subscription(
            String,
            action_topic,
            self._action_callback,
            10,
            callback_group=self.action_callback_group
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_loop,
            callback_group=self.control_callback_group
        )
        
        # Execution state
        self.execution_start_time: float = 0.0
        self.execution_duration: float = 0.0
        self.target_twist = Twist()
        
        self.get_logger().info(f"Action Executor initialized")
        self.get_logger().info(f"  Action topic: {action_topic}")
        self.get_logger().info(f"  Cmd vel topic: {cmd_vel_topic}")
        self.get_logger().info(f"  Forward: {self.forward_velocity} m/s, {self.forward_distance} m")
        self.get_logger().info(f"  Turn: {math.degrees(self.turn_velocity)} deg/s, {math.degrees(self.turn_angle)} deg")
        
    def _action_callback(self, msg: String):
        """Handle incoming action commands."""
        action = msg.data.strip().lower()
        
        # Normalize action
        normalized = self.action_aliases.get(action, action.upper())
        
        self.get_logger().info(f"Received action: {action} -> {normalized}")
        
        with self.action_lock:
            # If idle, start executing immediately
            if self.state == ActionState.IDLE:
                self._start_action(normalized)
            else:
                # Queue the action (or replace - depending on desired behavior)
                self.action_queue.append(normalized)
                self.get_logger().info(f"Action queued (queue size: {len(self.action_queue)})")
                
    def _start_action(self, action: str):
        """Start executing an action."""
        self.current_action = action
        self.state = ActionState.EXECUTING
        self.execution_start_time = time.time()
        
        # Calculate target velocity and duration
        if action == 'MOVE_FORWARD':
            self.target_twist.linear.x = self.forward_velocity
            self.target_twist.angular.z = 0.0
            self.execution_duration = self.forward_distance / self.forward_velocity
            
        elif action == 'TURN_LEFT':
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = self.turn_velocity
            self.execution_duration = self.turn_angle / self.turn_velocity
            
        elif action == 'TURN_RIGHT':
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = -self.turn_velocity
            self.execution_duration = self.turn_angle / self.turn_velocity
            
        elif action == 'STOP':
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = 0.0
            self.execution_duration = 0.1  # Brief stop
            
        else:
            self.get_logger().warn(f"Unknown action: {action}, stopping")
            self.target_twist.linear.x = 0.0
            self.target_twist.angular.z = 0.0
            self.execution_duration = 0.1
            
        self.get_logger().info(f"Executing {action} for {self.execution_duration:.2f}s")
        self._publish_status(f"executing:{action}")
        
    def _control_loop(self):
        """Main control loop - publishes velocity commands."""
        with self.action_lock:
            if self.state == ActionState.IDLE:
                # Publish zero velocity when idle
                self._publish_zero_velocity()
                return
                
            if self.state == ActionState.EXECUTING:
                # Check if action is complete
                elapsed = time.time() - self.execution_start_time
                
                if elapsed >= self.execution_duration:
                    # Action complete
                    self._complete_action()
                else:
                    # Continue executing
                    self.cmd_vel_pub.publish(self.target_twist)
                    
    def _complete_action(self):
        """Complete the current action and start next if queued."""
        self.get_logger().info(f"Action {self.current_action} completed")
        self._publish_status(f"completed:{self.current_action}")
        
        # Stop the robot
        self._publish_zero_velocity()
        
        # Check for queued actions
        if self.action_queue:
            next_action = self.action_queue.pop(0)
            self._start_action(next_action)
        else:
            self.state = ActionState.IDLE
            self.current_action = None
            self._publish_status("idle")
            
    def _publish_zero_velocity(self):
        """Publish zero velocity command."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
    def _publish_status(self, status: str):
        """Publish executor status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        

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
