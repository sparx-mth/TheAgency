#!/usr/bin/env python3
"""
InternNav Bridge Node

ROS2 bridge connecting simulations to the InternNav model server.
"""

import json
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from .types import ActionType, BridgeState
from .model_client import ModelClient
from .config import load_config


class InternNavBridge(Node):
    """Main ROS2 bridge node."""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__('internnav_bridge')

        # Parameters
        self.declare_parameter('config_path', '')
        config_path = config_path or self.get_parameter('config_path').value

        # Load config
        self.config = load_config(config_path, self.get_logger())

        # State
        self.state = BridgeState()
        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        # Model client
        server_cfg = self.config['bridge']['server']
        self.client = ModelClient(
            host=server_cfg['host'],
            port=server_cfg['port'],
            timeout=server_cfg.get('timeout_sec', 30.0),
            protocol=server_cfg.get('protocol', 'http'),
            logger=self.get_logger()
        )

        # Initialize model connection
        if self.client.check_health():
            self.get_logger().info("Model server is healthy")
            model_cfg = self.config.get('model', {})
            self.client.init_agent(
                model_name=model_cfg.get('variant', 'InternVLA-N1'),
                ckpt_path=model_cfg.get('ckpt_path', ''),
                model_settings=model_cfg.get('model_settings', {})
            )
        else:
            self.get_logger().warn("Model server not responding, will retry on inference")

        # Callback groups
        self.input_cb_group = ReentrantCallbackGroup()
        self.inference_cb_group = MutuallyExclusiveCallbackGroup()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()

        # Inference timer
        rate = self.config['bridge']['control'].get('inference_rate', 4.0)
        self.inference_timer = self.create_timer(
            1.0 / rate, self._inference_callback, callback_group=self.inference_cb_group
        )

        self._log_config()

    def _setup_subscribers(self):
        """Setup input subscribers."""
        inputs = self.config['inputs']
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # RGB
        if inputs['rgb'].get('enabled', True):
            rgb_cfg = inputs['rgb']
            msg_type = CompressedImage if 'Compressed' in rgb_cfg.get('msg_type', '') else Image
            self.rgb_sub = self.create_subscription(
                msg_type, rgb_cfg['topic'], self._rgb_callback, qos,
                callback_group=self.input_cb_group
            )
            self.get_logger().info(f"Subscribed to RGB: {rgb_cfg['topic']}")

        # Depth
        if inputs.get('depth', {}).get('enabled', False):
            self.depth_sub = self.create_subscription(
                Image, inputs['depth']['topic'], self._depth_callback, qos,
                callback_group=self.input_cb_group
            )
            self.get_logger().info(f"Subscribed to Depth: {inputs['depth']['topic']}")

        # Instruction
        if inputs.get('instruction', {}).get('enabled', True):
            inst_cfg = inputs['instruction']
            self.instruction_sub = self.create_subscription(
                String, inst_cfg['topic'], self._instruction_callback, 1,
                callback_group=self.input_cb_group
            )
            self.state.current_instruction = inst_cfg.get('default', '')
            self.get_logger().info(f"Subscribed to Instruction: {inst_cfg['topic']}")

        # Odometry
        if inputs.get('odometry', {}).get('enabled', False):
            self.odom_sub = self.create_subscription(
                Odometry, inputs['odometry']['topic'], self._odom_callback, qos,
                callback_group=self.input_cb_group
            )

    def _setup_publishers(self):
        """Setup output publishers."""
        outputs = self.config['outputs']

        # Discrete action
        if outputs.get('discrete', {}).get('enabled', True):
            self.action_pub = self.create_publisher(String, outputs['discrete']['topic'], 1)
            self.get_logger().info(f"Publishing actions to: {outputs['discrete']['topic']}")

        # Continuous velocity
        if outputs.get('continuous', {}).get('enabled', False):
            self.cmd_vel_pub = self.create_publisher(Twist, outputs['continuous']['topic'], 1)
            self.get_logger().info(f"Publishing velocities to: {outputs['continuous']['topic']}")

        # Feedback
        if outputs.get('feedback', {}).get('enabled', True):
            self.feedback_pub = self.create_publisher(String, outputs['feedback']['topic'], 1)

        # Status
        if outputs.get('status', {}).get('enabled', True):
            self.status_pub = self.create_publisher(String, outputs['status']['topic'], 1)

    # === Callbacks ===

    def _rgb_callback(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            with self.lock:
                self.state.current_rgb = image
                self.state.rgb_timestamp = time.time()
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def _depth_callback(self, msg):
        try:
            with self.lock:
                self.state.current_depth = self.cv_bridge.imgmsg_to_cv2(msg)
                self.state.depth_timestamp = time.time()
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def _instruction_callback(self, msg: String):
        with self.lock:
            self.state.current_instruction = msg.data
            self.state.is_navigating = True
        self.get_logger().info(f"Instruction: {msg.data}")

    def _odom_callback(self, msg: Odometry):
        with self.lock:
            self.state.current_odometry = {
                'position': {'x': msg.pose.pose.position.x, 'y': msg.pose.pose.position.y,
                             'z': msg.pose.pose.position.z},
                'orientation': {'x': msg.pose.pose.orientation.x, 'y': msg.pose.pose.orientation.y,
                                'z': msg.pose.pose.orientation.z, 'w': msg.pose.pose.orientation.w}
            }

    def _inference_callback(self):
        """Run model inference."""
        control = self.config['bridge']['control']
        current_time = time.time()

        # Check timing
        if current_time - self.state.last_inference_time < control.get('min_inference_interval', 0.1):
            return

        with self.lock:
            if self.state.current_rgb is None:
                return
            if not control.get('continuous_inference', False) and not self.state.is_navigating:
                return
            payload = self._prepare_payload()

        if payload is None:
            return

        # Run inference
        result = self.client.step(payload['rgb'], payload['instruction'], payload.get('depth'))

        if not result.success:
            self._publish_status("error")
            return

        self._process_result(result)
        self.state.last_inference_time = current_time

    def _prepare_payload(self) -> Optional[Dict]:
        """Prepare inference payload."""
        model_cfg = self.config.get('model', {}).get('input_format', {})
        target_w = model_cfg.get('target_width', 640)
        target_h = model_cfg.get('target_height', 480)

        # Process RGB
        rgb = self.state.current_rgb
        if rgb.shape[1] != target_w or rgb.shape[0] != target_h:
            rgb = cv2.resize(rgb, (target_w, target_h))
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        payload = {'rgb': rgb, 'instruction': self.state.current_instruction}

        # Process depth if available
        if self.config['inputs'].get('depth', {}).get('enabled') and self.state.current_depth is not None:
            depth = self.state.current_depth
            if depth.shape[1] != target_w or depth.shape[0] != target_h:
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
            payload['depth'] = depth.astype(np.float32)

        return payload

    def _process_result(self, result):
        """Process inference result and publish."""
        action_str = result.action
        action = self._parse_action(action_str)
        outputs = self.config['outputs']

        # Discrete action
        if outputs.get('discrete', {}).get('enabled', True):
            mapping = outputs['discrete'].get('action_mapping', {})
            mapped = mapping.get(action.value, action_str)
            msg = String()
            msg.data = mapped
            self.action_pub.publish(msg)
            self.get_logger().debug(f"Action: {mapped}")

        # Continuous velocity
        if outputs.get('continuous', {}).get('enabled', False):
            self._publish_velocity(action)

        # Feedback
        if outputs.get('feedback', {}).get('enabled', True):
            feedback = {'action': action.value, 'timestamp': time.time(), 'inference_ms': result.inference_time_ms}
            msg = String()
            msg.data = json.dumps(feedback)
            self.feedback_pub.publish(msg)

        # Status
        if action == ActionType.STOP:
            self.state.is_navigating = False
            self._publish_status("completed")
        else:
            self._publish_status("navigating")

    def _parse_action(self, action_str: str) -> ActionType:
        """Parse action string to ActionType."""
        upper = action_str.upper().strip()
        for at in ActionType:
            if at.value in upper:
                return at
        if 'FORWARD' in upper:
            return ActionType.MOVE_FORWARD
        if 'LEFT' in upper:
            return ActionType.TURN_LEFT
        if 'RIGHT' in upper:
            return ActionType.TURN_RIGHT
        if 'STOP' in upper or 'DONE' in upper:
            return ActionType.STOP
        return ActionType.UNKNOWN

    def _publish_velocity(self, action: ActionType):
        """Publish velocity command."""
        cfg = self.config['outputs']['continuous']
        vel_map = cfg.get('action_to_velocity', {})
        vel = vel_map.get(action.value, {'linear_x': 0.0, 'angular_z': 0.0})
        limits = cfg.get('limits', {})

        msg = Twist()
        msg.linear.x = max(-limits.get('max_linear', 1.0), min(limits.get('max_linear', 1.0), vel.get('linear_x', 0.0)))
        msg.angular.z = max(-limits.get('max_angular', 1.0),
                            min(limits.get('max_angular', 1.0), vel.get('angular_z', 0.0)))
        self.cmd_vel_pub.publish(msg)

    def _publish_status(self, status: str):
        """Publish status."""
        if hasattr(self, 'status_pub'):
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)

    def _log_config(self):
        """Log configuration summary."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("InternNav Bridge Configuration:")
        self.get_logger().info(
            f"  Server: {self.config['bridge']['server']['host']}:{self.config['bridge']['server']['port']}")
        self.get_logger().info(f"  RGB: {self.config['inputs']['rgb']['topic']}")
        self.get_logger().info(f"  Action: {self.config['outputs']['discrete']['topic']}")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)

    import sys
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    node = InternNavBridge(config_path)
    executor = MultiThreadedExecutor(num_threads=4)
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