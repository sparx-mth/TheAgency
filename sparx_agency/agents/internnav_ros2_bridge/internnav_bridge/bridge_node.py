#!/usr/bin/env python3
"""
InternNav Bridge Node for Rooster Platform - GROUND ROLL ONLY

ROS2 bridge connecting simulations to the InternNav model server.
Uses ManualControl for ground rolling - NO FLYING.

Key differences from flying mode:
- Stabilization uses z=0 (no throttle)
- Movement uses x for forward, r for turning
- z is only used minimally to keep wheels engaged (not for lift)
"""

import json
import math
import threading
import time
from typing import Dict, Optional
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from .types import ActionType, BridgeState
from .model_client import ModelClient
from .config import load_config

# Try to import Rooster-specific messages
try:
    from fcu_driver_interfaces.msg import ManualControl, UAVState
    from rooster_handler_interfaces.msg import KeepAlive
    from rooster_manager_interfaces.msg import RoosterState

    ROOSTER_MSGS_AVAILABLE = True
except ImportError:
    ROOSTER_MSGS_AVAILABLE = False
    ManualControl = None
    UAVState = None
    KeepAlive = None
    RoosterState = None

# Arm service
try:
    from std_srvs.srv import SetBool

    STD_SRVS_AVAILABLE = True
except ImportError:
    STD_SRVS_AVAILABLE = False
    SetBool = None


class ActionState(Enum):
    """State of action execution."""
    IDLE = "idle"
    EXECUTING = "executing"


class InternNavBridge(Node):
    """Main ROS2 bridge node - GROUND ROLL mode only."""

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

        # Action execution state (for timed actions)
        self.action_state = ActionState.IDLE
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.current_manual_control = None

        # Rooster/UAV state (from UAVState message)
        self.arm_state = False
        self.airborne = False
        self.flight_mode = None
        self.is_ready = False
        self.uav_state_received = False

        # ARM control
        self.arm_requested = False
        self.arm_in_progress = False
        self.last_arm_attempt = 0.0
        self.arm_retry_interval = 2.0

        # GROUND ROLL - No stabilization throttle needed!
        self.stabilize_time = 1.0  # Just wait 1s for arm to settle
        self.arm_time = 0.0
        self.is_stabilized = False

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
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_arm_service()

        # Inference timer
        rate = self.config['bridge']['control'].get('inference_rate', 4.0)
        self.inference_timer = self.create_timer(
            1.0 / rate, self._inference_callback, callback_group=self.inference_cb_group
        )

        self._log_config()

    def _setup_arm_service(self):
        """Setup the force_arm service client."""
        if not STD_SRVS_AVAILABLE:
            self.get_logger().error("std_srvs not available! Cannot arm drone.")
            self.force_arm_client = None
            return

        rgb_topic = self.config['inputs']['rgb']['topic']
        parts = rgb_topic.split('/')
        rooster_id = parts[1] if len(parts) > 1 else "R1"

        arm_service = f"/{rooster_id}/fcu/command/force_arm"
        self.force_arm_client = self.create_client(SetBool, arm_service)
        self.get_logger().info(f"Created ARM service client: {arm_service}")

    def _request_arm(self, arm: bool = True) -> bool:
        """Request drone to arm via force_arm service."""
        if self.force_arm_client is None:
            self.get_logger().error("ARM service client not available!")
            return False

        if not self.force_arm_client.service_is_ready():
            self.get_logger().warn("ARM service not ready, waiting...")
            if not self.force_arm_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("ARM service timeout!")
                return False

        self.get_logger().info(f"Calling force_arm service (arm={arm})...")
        request = SetBool.Request()
        request.data = arm

        future = self.force_arm_client.call_async(request)
        self.arm_in_progress = True
        self.last_arm_attempt = time.time()
        return True

    def _setup_subscribers(self):
        """Setup input subscribers."""
        inputs = self.config['inputs']
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

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

        # UAVState subscription
        if ROOSTER_MSGS_AVAILABLE:
            rgb_topic = self.config['inputs']['rgb']['topic']
            parts = rgb_topic.split('/')
            rooster_id = parts[1] if len(parts) > 1 else "R1"

            uav_state_topic = f"/{rooster_id}/fcu/state"
            self.uav_state_sub = self.create_subscription(
                UAVState, uav_state_topic, self._uav_state_callback, 10,
                callback_group=self.input_cb_group
            )
            self.get_logger().info(f"Subscribed to UAVState: {uav_state_topic}")

        # RoosterState (backwards compat)
        outputs = self.config['outputs']
        if outputs.get('state', {}).get('enabled', False) and ROOSTER_MSGS_AVAILABLE:
            state_topic = outputs['state'].get('topic', '/R1/state')
            self.rooster_state_sub = self.create_subscription(
                RoosterState, state_topic, self._rooster_state_callback, 10,
                callback_group=self.input_cb_group
            )
            self.get_logger().info(f"Subscribed to Rooster State: {state_topic}")

    def _setup_publishers(self):
        """Setup output publishers."""
        outputs = self.config['outputs']

        # Discrete action
        if outputs.get('discrete', {}).get('enabled', True):
            self.action_pub = self.create_publisher(String, outputs['discrete']['topic'], 1)
            self.get_logger().info(f"Publishing actions to: {outputs['discrete']['topic']}")

        # Continuous velocity (Twist)
        if outputs.get('continuous', {}).get('enabled', False):
            self.cmd_vel_pub = self.create_publisher(Twist, outputs['continuous']['topic'], 1)
            self.get_logger().info(f"Publishing velocities to: {outputs['continuous']['topic']}")

        # ManualControl output
        if outputs.get('manual_control', {}).get('enabled', False):
            if not ROOSTER_MSGS_AVAILABLE:
                self.get_logger().error("ManualControl enabled but fcu_driver_interfaces not available!")
            else:
                mc_cfg = outputs['manual_control']
                self.manual_control_pub = self.create_publisher(
                    ManualControl, mc_cfg['topic'], 10
                )
                self.get_logger().info(f"Publishing ManualControl to: {mc_cfg['topic']}")

                mc_rate = mc_cfg.get('publish_rate_hz', 40.0)
                self.manual_control_timer = self.create_timer(
                    1.0 / mc_rate, self._manual_control_callback,
                    callback_group=self.control_cb_group
                )

        # KeepAlive publisher
        if outputs.get('keep_alive', {}).get('enabled', False):
            if not ROOSTER_MSGS_AVAILABLE:
                self.get_logger().error("KeepAlive enabled but rooster_handler_interfaces not available!")
            else:
                ka_cfg = outputs['keep_alive']
                self.keep_alive_pub = self.create_publisher(
                    KeepAlive, ka_cfg['topic'], 10
                )
                self.get_logger().info(f"Publishing KeepAlive to: {ka_cfg['topic']}")

                ka_rate = ka_cfg.get('publish_rate_hz', 1.0)
                self.keep_alive_timer = self.create_timer(
                    1.0 / ka_rate, self._keep_alive_callback,
                    callback_group=self.control_cb_group
                )

        # Feedback
        if outputs.get('feedback', {}).get('enabled', True):
            self.feedback_pub = self.create_publisher(String, outputs['feedback']['topic'], 1)

        # Status
        if outputs.get('status', {}).get('enabled', True):
            self.status_pub = self.create_publisher(String, outputs['status']['topic'], 1)

    # === Input Callbacks ===

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
            self.arm_requested = True
            self.is_stabilized = False
        self.get_logger().info(f"Instruction: {msg.data}")

        if not self.arm_state:
            self._request_arm(True)

    def _odom_callback(self, msg: Odometry):
        with self.lock:
            self.state.current_odometry = {
                'position': {
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.pose.orientation.x,
                    'y': msg.pose.pose.orientation.y,
                    'z': msg.pose.pose.orientation.z,
                    'w': msg.pose.pose.orientation.w
                }
            }

    def _uav_state_callback(self, msg: 'UAVState'):
        """Handle UAVState updates."""
        with self.lock:
            was_armed = self.arm_state
            self.arm_state = msg.armed
            self.airborne = msg.airborne
            self.uav_state_received = True

            if msg.armed and not was_armed:
                self.arm_time = time.time()
                self.is_stabilized = False
                self.get_logger().info("✓ ARMED! Waiting for stabilization...")
            elif not msg.armed and was_armed:
                self.get_logger().warn("DISARMED!")
                self.is_stabilized = False

        if was_armed != msg.armed:
            self.get_logger().info(f"UAVState - armed: {msg.armed}, airborne: {msg.airborne}")

    def _rooster_state_callback(self, msg: 'RoosterState'):
        """Handle Rooster state updates."""
        if self.flight_mode != msg.flight_mode:
            self.get_logger().info(f"Rooster state - flight_mode: {msg.flight_mode}, ready: {msg.is_ready}")
            self.flight_mode = msg.flight_mode
            self.is_ready = msg.is_ready

    # === Control Callbacks ===

    def _keep_alive_callback(self):
        """Publish KeepAlive for GROUND_ROLL mode."""
        if not hasattr(self, 'keep_alive_pub'):
            return

        ka_cfg = self.config['outputs']['keep_alive']

        msg = KeepAlive()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.is_active = ka_cfg.get('is_active', True)
        msg.requested_flight_mode = 1  # GROUND_ROLL = 1
        msg.command_reboot = False

        self.keep_alive_pub.publish(msg)

        # Re-arm if needed
        current_time = time.time()
        if self.arm_requested and not self.arm_state:
            if current_time - self.last_arm_attempt > self.arm_retry_interval:
                self.get_logger().info("Re-attempting ARM...")
                self._request_arm(True)

    def _manual_control_callback(self):
        """Publish ManualControl - GROUND ROLL only (no lift)."""
        if not hasattr(self, 'manual_control_pub'):
            return

        current_time = time.time()

        with self.lock:
            # Stabilization: just wait, NO throttle (z=0)
            if self.arm_state and not self.is_stabilized:
                elapsed_since_arm = current_time - self.arm_time
                if elapsed_since_arm < self.stabilize_time:
                    # Send zeros during stabilization - NO LIFT
                    msg = self._get_stop_control()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    self.manual_control_pub.publish(msg)
                    return
                else:
                    if not self.is_stabilized:
                        self.get_logger().info("✓ Stabilization complete, ready for ground roll")
                        self.is_stabilized = True

            # Check action execution
            if self.action_state == ActionState.EXECUTING:
                elapsed = current_time - self.action_start_time
                if elapsed >= self.action_duration:
                    self.action_state = ActionState.IDLE
                    self.current_manual_control = None
                    self.get_logger().debug("Action complete")

            # Determine what to publish
            if self.action_state == ActionState.EXECUTING and self.current_manual_control is not None:
                msg = self.current_manual_control
            else:
                # IDLE: send zeros (ground roll doesn't need constant throttle)
                msg = self._get_stop_control()

            msg.header.stamp = self.get_clock().now().to_msg()
            self.manual_control_pub.publish(msg)

    def _get_stop_control(self) -> 'ManualControl':
        """Get a stop ManualControl message - ALL ZEROS for ground mode."""
        msg = ManualControl()
        msg.header = Header()
        msg.x = 0.0
        msg.y = 0.0
        msg.z = 0.0  # NO THROTTLE - ground roll only!
        msg.r = 0.0
        msg.buttons = 0
        return msg

    # === Inference ===

    def _inference_callback(self):
        """Run model inference."""
        control = self.config['bridge']['control']
        current_time = time.time()

        if current_time - self.state.last_inference_time < control.get('min_inference_interval', 0.1):
            return

        with self.lock:
            if self.action_state == ActionState.EXECUTING:
                return

            if self.state.current_rgb is None:
                return
            if not control.get('continuous_inference', False) and not self.state.is_navigating:
                return

            if self.arm_requested and (not self.arm_state or not self.is_stabilized):
                return

            payload = self._prepare_payload()

        if payload is None:
            return

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

        rgb = self.state.current_rgb
        if rgb.shape[1] != target_w or rgb.shape[0] != target_h:
            rgb = cv2.resize(rgb, (target_w, target_h))
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        payload = {'rgb': rgb, 'instruction': self.state.current_instruction}

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

        # Discrete action (for logging)
        if outputs.get('discrete', {}).get('enabled', True):
            mapping = outputs['discrete'].get('action_mapping', {})
            mapped = mapping.get(action.value, action_str)
            msg = String()
            msg.data = mapped
            self.action_pub.publish(msg)
            self.get_logger().info(f"Action: {mapped} (raw: {action_str}) [armed={self.arm_state}]")

        # ManualControl output
        if outputs.get('manual_control', {}).get('enabled', False) and ROOSTER_MSGS_AVAILABLE:
            if not self.arm_state:
                self.get_logger().warn("Cannot execute action - not armed!")
                self._request_arm(True)
                return
            self._execute_manual_control_action(action)

        elif outputs.get('continuous', {}).get('enabled', False):
            self._publish_velocity(action)

        # Feedback
        if outputs.get('feedback', {}).get('enabled', True):
            feedback = {
                'action': action.value,
                'timestamp': time.time(),
                'inference_ms': result.inference_time_ms,
                'armed': self.arm_state
            }
            msg = String()
            msg.data = json.dumps(feedback)
            self.feedback_pub.publish(msg)

        # Status
        if action == ActionType.STOP:
            self.state.is_navigating = False
            self._publish_status("completed")
        else:
            self._publish_status("navigating")

    def _execute_manual_control_action(self, action: ActionType):
        """Execute action using ManualControl - GROUND ROLL."""
        mc_cfg = self.config['outputs']['manual_control']
        action_mapping = mc_cfg.get('action_mapping', {})

        action_params = action_mapping.get(action.value, action_mapping.get('STOP', {}))

        # Create ManualControl message
        msg = ManualControl()
        msg.header = Header()
        msg.x = float(action_params.get('x', 0.0))
        msg.y = float(action_params.get('y', 0.0))
        msg.z = float(action_params.get('z', 0.0))  # Use config value
        msg.r = float(action_params.get('r', 0.0))
        msg.buttons = int(action_params.get('buttons', 0))

        duration = action_params.get('duration_sec', 0.5)

        with self.lock:
            self.current_manual_control = msg
            self.action_start_time = time.time()
            self.action_duration = duration
            self.action_state = ActionState.EXECUTING

        self.get_logger().info(
            f"Executing {action.value}: x={msg.x}, y={msg.y}, z={msg.z}, r={msg.r} for {duration:.2f}s"
        )

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
        """Publish velocity command (Twist)."""
        cfg = self.config['outputs']['continuous']
        vel_map = cfg.get('action_to_velocity', {})
        vel = vel_map.get(action.value, {'linear_x': 0.0, 'angular_z': 0.0})
        limits = cfg.get('limits', {})

        msg = Twist()
        msg.linear.x = max(
            -limits.get('max_linear', 1.0),
            min(limits.get('max_linear', 1.0), vel.get('linear_x', 0.0))
        )
        msg.angular.z = max(
            -limits.get('max_angular', 1.0),
            min(limits.get('max_angular', 1.0), vel.get('angular_z', 0.0))
        )
        self.cmd_vel_pub.publish(msg)

    def _publish_status(self, status: str):
        """Publish status."""
        if hasattr(self, 'status_pub'):
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)

    def _log_config(self):
        """Log configuration summary."""
        outputs = self.config['outputs']
        self.get_logger().info("=" * 60)
        self.get_logger().info("InternNav Bridge - GROUND ROLL MODE (No Flying)")
        self.get_logger().info(
            f"  Server: {self.config['bridge']['server']['host']}:"
            f"{self.config['bridge']['server']['port']}"
        )
        self.get_logger().info(f"  RGB: {self.config['inputs']['rgb']['topic']}")

        if outputs.get('manual_control', {}).get('enabled'):
            self.get_logger().info(f"  ManualControl: {outputs['manual_control']['topic']}")
            self.get_logger().info(f"  KeepAlive: {outputs['keep_alive']['topic']}")

        self.get_logger().info(f"  Rooster msgs available: {ROOSTER_MSGS_AVAILABLE}")
        self.get_logger().info(f"  ARM via: force_arm service")
        self.get_logger().info(f"  Flight Mode: GROUND_ROLL (1)")
        self.get_logger().info("=" * 60)


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