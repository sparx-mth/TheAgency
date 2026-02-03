#!/usr/bin/env python3
"""
InternNav Bridge Node for Rooster Platform - GROUND ROLL MODE

ROS2 bridge connecting simulations to the InternNav model server.
STOP action does NOT terminate navigation - only explicit commands do.
"""

import json
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

# Optional Rooster messages
try:
    from fcu_driver_interfaces.msg import ManualControl, UAVState
    from rooster_handler_interfaces.msg import KeepAlive
    from rooster_manager_interfaces.msg import RoosterState
    ROOSTER_MSGS_AVAILABLE = True
except ImportError:
    ROOSTER_MSGS_AVAILABLE = False
    ManualControl = UAVState = KeepAlive = RoosterState = None

try:
    from std_srvs.srv import SetBool
    STD_SRVS_AVAILABLE = True
except ImportError:
    STD_SRVS_AVAILABLE = False
    SetBool = None


class ActionState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"


class NavigationStatus(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    PAUSED = "paused"
    COMPLETED_SUCCESS = "success"
    COMPLETED_FAILURE = "failure"


class InternNavBridge(Node):
    """Main ROS2 bridge node - GROUND ROLL mode."""

    GOAL_KEYWORDS = ["goal reached", "arrived", "destination", "finished", "complete", "done navigating"]
    FAILURE_KEYWORDS = ["cannot reach", "unreachable", "blocked", "failed", "stuck"]

    def __init__(self, config_path: Optional[str] = None):
        super().__init__('internnav_bridge')
        self.declare_parameter('config_path', '')
        config_path = config_path or self.get_parameter('config_path').value

        self.config = load_config(config_path, self.get_logger())
        self.state = BridgeState()
        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        # Navigation state
        self.nav_status = NavigationStatus.IDLE
        self.consecutive_stops = 0
        self.last_action = None

        # Action execution
        self.action_state = ActionState.IDLE
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.current_manual_control = None

        # Handheld mode: no ARM, no keep_alive, no motor commands — just inference + discrete actions
        self.handheld_mode = self.config['bridge']['control'].get('handheld', False)

        # UAV state
        self.arm_state = False
        self.arm_requested = False
        self.last_arm_attempt = 0.0
        self.arm_time = 0.0
        self.is_stabilized = False
        self.stabilize_time = 1.0

        # Model client
        server_cfg = self.config['bridge']['server']
        self.client = ModelClient(
            host=server_cfg['host'], port=server_cfg['port'],
            timeout=server_cfg.get('timeout_sec', 30.0),
            protocol=server_cfg.get('protocol', 'http'),
            logger=self.get_logger()
        )

        if self.client.check_health():
            self.get_logger().info("Model server is healthy")
            model_cfg = self.config.get('model', {})
            self.client.init_agent(
                model_name=model_cfg.get('variant', 'InternVLA-N1'),
                ckpt_path=model_cfg.get('ckpt_path', ''),
                model_settings=model_cfg.get('model_settings', {})
            )
        else:
            self.get_logger().warn("Model server not responding")

        # Callback groups
        self.input_cb_group = ReentrantCallbackGroup()
        self.inference_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        self._setup_subscribers()
        self._setup_publishers()
        if not self.handheld_mode:
            self._setup_arm_service()
        else:
            self.force_arm_client = None

        rate = self.config['bridge']['control'].get('inference_rate', 4.0)
        self.inference_timer = self.create_timer(1.0 / rate, self._inference_callback,
                                                  callback_group=self.inference_cb_group)
        self._log_config()

    def _setup_arm_service(self):
        if not STD_SRVS_AVAILABLE:
            self.force_arm_client = None
            return
        rgb_topic = self.config['inputs']['rgb']['topic']
        rooster_id = rgb_topic.split('/')[1] if len(rgb_topic.split('/')) > 1 else "R1"
        self.force_arm_client = self.create_client(SetBool, f"/{rooster_id}/fcu/command/force_arm")

    def _request_arm(self, arm: bool = True) -> bool:
        if not self.force_arm_client or not self.force_arm_client.service_is_ready():
            return False
        request = SetBool.Request()
        request.data = arm
        self.force_arm_client.call_async(request)
        self.last_arm_attempt = time.time()
        return True

    def _setup_subscribers(self):
        inputs = self.config['inputs']
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        if inputs['rgb'].get('enabled', True):
            rgb_cfg = inputs['rgb']
            msg_type = CompressedImage if 'Compressed' in rgb_cfg.get('msg_type', '') else Image
            self.rgb_sub = self.create_subscription(msg_type, rgb_cfg['topic'],
                                                     self._rgb_callback, qos,
                                                     callback_group=self.input_cb_group)

        if inputs.get('depth', {}).get('enabled'):
            self.depth_sub = self.create_subscription(Image, inputs['depth']['topic'],
                                                       self._depth_callback, qos,
                                                       callback_group=self.input_cb_group)

        if inputs.get('instruction', {}).get('enabled', True):
            inst_cfg = inputs['instruction']
            self.instruction_sub = self.create_subscription(String, inst_cfg['topic'],
                                                             self._instruction_callback, 1,
                                                             callback_group=self.input_cb_group)
            self.state.current_instruction = inst_cfg.get('default', '')

        if inputs.get('odometry', {}).get('enabled'):
            self.odom_sub = self.create_subscription(Odometry, inputs['odometry']['topic'],
                                                      self._odom_callback, qos,
                                                      callback_group=self.input_cb_group)

        if ROOSTER_MSGS_AVAILABLE and not self.handheld_mode:
            rgb_topic = self.config['inputs']['rgb']['topic']
            rooster_id = rgb_topic.split('/')[1] if len(rgb_topic.split('/')) > 1 else "R1"
            self.uav_state_sub = self.create_subscription(UAVState, f"/{rooster_id}/fcu/state",
                                                           self._uav_state_callback, 10,
                                                           callback_group=self.input_cb_group)

        nav_topic = inputs.get('nav_control', {}).get('topic', '/navigation/control')
        self.nav_control_sub = self.create_subscription(String, nav_topic,
                                                         self._nav_control_callback, 1,
                                                         callback_group=self.input_cb_group)

    def _setup_publishers(self):
        outputs = self.config['outputs']

        if outputs.get('discrete', {}).get('enabled', True):
            self.action_pub = self.create_publisher(String, outputs['discrete']['topic'], 1)

        if outputs.get('continuous', {}).get('enabled'):
            self.cmd_vel_pub = self.create_publisher(Twist, outputs['continuous']['topic'], 1)

        if outputs.get('manual_control', {}).get('enabled') and ROOSTER_MSGS_AVAILABLE and not self.handheld_mode:
            mc_cfg = outputs['manual_control']
            self.manual_control_pub = self.create_publisher(ManualControl, mc_cfg['topic'], 10)
            mc_rate = mc_cfg.get('publish_rate_hz', 40.0)
            self.manual_control_timer = self.create_timer(1.0 / mc_rate, self._manual_control_callback,
                                                          callback_group=self.control_cb_group)

        if outputs.get('keep_alive', {}).get('enabled') and ROOSTER_MSGS_AVAILABLE and not self.handheld_mode:
            ka_cfg = outputs['keep_alive']
            self.keep_alive_pub = self.create_publisher(KeepAlive, ka_cfg['topic'], 10)
            ka_rate = ka_cfg.get('publish_rate_hz', 1.0)
            self.keep_alive_timer = self.create_timer(1.0 / ka_rate, self._keep_alive_callback,
                                                      callback_group=self.control_cb_group)

        if outputs.get('feedback', {}).get('enabled', True):
            self.feedback_pub = self.create_publisher(String, outputs['feedback']['topic'], 1)

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
            self.get_logger().error(f"RGB error: {e}")

    def _depth_callback(self, msg):
        try:
            with self.lock:
                self.state.current_depth = self.cv_bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")

    def _instruction_callback(self, msg: String):
        instruction = msg.data.strip()
        instruction_lower = instruction.lower()

        with self.lock:
            if any(kw in instruction_lower for kw in self.GOAL_KEYWORDS):
                self.nav_status = NavigationStatus.COMPLETED_SUCCESS
                self.state.is_navigating = False
                self._publish_status("success")
                return

            if any(kw in instruction_lower for kw in self.FAILURE_KEYWORDS):
                self.nav_status = NavigationStatus.COMPLETED_FAILURE
                self.state.is_navigating = False
                self._publish_status("failure")
                return

            self.state.current_instruction = instruction
            self.state.is_navigating = True
            self.nav_status = NavigationStatus.NAVIGATING
            self.consecutive_stops = 0
            if not self.handheld_mode:
                self.arm_requested = True
                self.is_stabilized = False

        self.get_logger().info(f"Instruction: {instruction}")
        if not self.handheld_mode and not self.arm_state:
            self._request_arm(True)

    def _nav_control_callback(self, msg: String):
        command = msg.data.strip().lower()
        with self.lock:
            if command in ["stop", "end", "finish", "done"]:
                self.nav_status = NavigationStatus.COMPLETED_SUCCESS
                self.state.is_navigating = False
                self._publish_status("success")
            elif command == "pause":
                self.nav_status = NavigationStatus.PAUSED
            elif command == "resume" and self.nav_status == NavigationStatus.PAUSED:
                self.nav_status = NavigationStatus.NAVIGATING
                self.state.is_navigating = True
            elif command == "reset":
                self.nav_status = NavigationStatus.IDLE
                self.state.is_navigating = False
                self.consecutive_stops = 0

    def _odom_callback(self, msg: Odometry):
        with self.lock:
            self.state.current_odometry = {
                'position': {'x': msg.pose.pose.position.x, 'y': msg.pose.pose.position.y,
                            'z': msg.pose.pose.position.z},
                'orientation': {'x': msg.pose.pose.orientation.x, 'y': msg.pose.pose.orientation.y,
                               'z': msg.pose.pose.orientation.z, 'w': msg.pose.pose.orientation.w}
            }

    def _uav_state_callback(self, msg):
        with self.lock:
            was_armed = self.arm_state
            self.arm_state = msg.armed
            if msg.armed and not was_armed:
                self.arm_time = time.time()
                self.is_stabilized = False
                self.get_logger().info("ARMED!")

    def _keep_alive_callback(self):
        if not hasattr(self, 'keep_alive_pub'):
            return
        ka_cfg = self.config['outputs']['keep_alive']
        msg = KeepAlive()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.is_active = ka_cfg.get('is_active', True)
        msg.requested_flight_mode = 1  # GROUND_ROLL
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

        if self.arm_requested and not self.arm_state:
            if time.time() - self.last_arm_attempt > 2.0:
                self._request_arm(True)

    def _manual_control_callback(self):
        if not hasattr(self, 'manual_control_pub'):
            return
        current_time = time.time()

        with self.lock:
            if self.arm_state and not self.is_stabilized:
                if current_time - self.arm_time < self.stabilize_time:
                    msg = self._get_stop_control()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    self.manual_control_pub.publish(msg)
                    return
                self.is_stabilized = True
                self.get_logger().info("Stabilized")

            if self.action_state == ActionState.EXECUTING:
                if current_time - self.action_start_time >= self.action_duration:
                    self.action_state = ActionState.IDLE
                    self.current_manual_control = None

            if self.action_state == ActionState.EXECUTING and self.current_manual_control:
                msg = self.current_manual_control
            else:
                msg = self._get_stop_control()

            msg.header.stamp = self.get_clock().now().to_msg()
            self.manual_control_pub.publish(msg)

    def _get_stop_control(self):
        msg = ManualControl()
        msg.header = Header()
        msg.x = msg.y = msg.z = msg.r = 0.0
        msg.buttons = 0
        return msg

    # === Inference ===
    def _inference_callback(self):
        control = self.config['bridge']['control']
        current_time = time.time()

        if current_time - self.state.last_inference_time < control.get('min_inference_interval', 0.1):
            return

        with self.lock:
            if self.action_state == ActionState.EXECUTING or self.state.current_rgb is None:
                return
            if self.nav_status in [NavigationStatus.COMPLETED_SUCCESS,
                                   NavigationStatus.COMPLETED_FAILURE, NavigationStatus.IDLE]:
                if not control.get('continuous_inference'):
                    return
            if not self.handheld_mode and self.arm_requested and (not self.arm_state or not self.is_stabilized):
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
        action_str = result.action
        action = self._parse_action(action_str)
        outputs = self.config['outputs']

        if action == ActionType.STOP:
            self.consecutive_stops += 1
        else:
            self.consecutive_stops = 0
        self.last_action = action

        if outputs.get('discrete', {}).get('enabled', True):
            mapping = outputs['discrete'].get('action_mapping', {})
            mapped = mapping.get(action.value, action_str)
            msg = String()
            msg.data = mapped
            self.action_pub.publish(msg)
            if self.handheld_mode:
                self.get_logger().info(f"Action: {mapped} [handheld]")
            else:
                self.get_logger().info(f"Action: {mapped} [armed={self.arm_state}]")

        if not self.handheld_mode and outputs.get('manual_control', {}).get('enabled') and ROOSTER_MSGS_AVAILABLE:
            if not self.arm_state:
                self._request_arm(True)
                return
            self._execute_manual_control_action(action)
        elif not self.handheld_mode and outputs.get('continuous', {}).get('enabled'):
            self._publish_velocity(action)

        if outputs.get('feedback', {}).get('enabled', True):
            feedback = {'action': action.value, 'timestamp': time.time(),
                       'inference_ms': result.inference_time_ms, 'armed': self.arm_state}
            msg = String()
            msg.data = json.dumps(feedback)
            self.feedback_pub.publish(msg)

        self._publish_status("paused" if action == ActionType.STOP else "navigating")

    def _execute_manual_control_action(self, action: ActionType):
        mc_cfg = self.config['outputs']['manual_control']
        action_mapping = mc_cfg.get('action_mapping', {})
        action_params = action_mapping.get(action.value, action_mapping.get('STOP', {}))

        msg = ManualControl()
        msg.header = Header()
        msg.x = float(action_params.get('x', 0.0))
        msg.y = float(action_params.get('y', 0.0))
        msg.z = float(action_params.get('z', 0.0))
        msg.r = float(action_params.get('r', 0.0))
        msg.buttons = int(action_params.get('buttons', 0))

        duration = action_params.get('duration_sec', 0.5)

        with self.lock:
            self.current_manual_control = msg
            self.action_start_time = time.time()
            self.action_duration = duration
            self.action_state = ActionState.EXECUTING

        self.get_logger().info(f"Exec {action.value}: x={msg.x}, z={msg.z}, r={msg.r} for {duration:.2f}s")

    def _parse_action(self, action_str: str) -> ActionType:
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
        return ActionType.STOP if 'STOP' in upper or 'DONE' in upper else ActionType.UNKNOWN

    def _publish_velocity(self, action: ActionType):
        cfg = self.config['outputs']['continuous']
        vel_map = cfg.get('action_to_velocity', {})
        vel = vel_map.get(action.value, {'linear_x': 0.0, 'angular_z': 0.0})
        limits = cfg.get('limits', {})

        msg = Twist()
        msg.linear.x = max(-limits.get('max_linear', 1.0),
                          min(limits.get('max_linear', 1.0), vel.get('linear_x', 0.0)))
        msg.angular.z = max(-limits.get('max_angular', 1.0),
                           min(limits.get('max_angular', 1.0), vel.get('angular_z', 0.0)))
        self.cmd_vel_pub.publish(msg)

    def _publish_status(self, status: str):
        if hasattr(self, 'status_pub'):
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)

    def _log_config(self):
        outputs = self.config['outputs']
        self.get_logger().info("=" * 50)
        mode = "HANDHELD (no ARM/motors)" if self.handheld_mode else "GROUND ROLL"
        self.get_logger().info(f"InternNav Bridge - {mode} MODE")
        self.get_logger().info(f"  Server: {self.config['bridge']['server']['host']}:"
                               f"{self.config['bridge']['server']['port']}")
        self.get_logger().info(f"  RGB: {self.config['inputs']['rgb']['topic']}")
        if outputs.get('manual_control', {}).get('enabled'):
            self.get_logger().info(f"  ManualControl: {outputs['manual_control']['topic']}")
        self.get_logger().info(f"  Rooster msgs: {ROOSTER_MSGS_AVAILABLE}")
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