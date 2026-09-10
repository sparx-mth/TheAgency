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

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# The model contract (wire protocol + action vocabulary) is ROS-free and lives in
# core; only the ROS plumbing and the YAML loader are local to this package.
from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient
from sparx_agency.core.planning.vlas.internvla_n1.types import (
    NON_TERMINAL_IDLE_INDICES,
    ActionType,
    BridgeState,
)

from .config import load_config

# Optional Rooster messages
# Arming, the KeepAlive heartbeat, the post-arm settling window and the
# hold-then-idle ManualControl latch are Rooster R1 concerns, not InternVLA-N1
# concerns, so they live with the robot. This bridge only maps a discrete VLN
# action onto axis values via the YAML `action_mapping` table.
from sparx_agency.robots.ROBOTICAN.adapters.rooster_manual_control import (
    ManualAxes,
    RoosterManualControl,
)

try:
    from rooster_manager_interfaces.msg import RoosterState
    ROOSTER_MSGS_AVAILABLE = True
except ImportError:
    ROOSTER_MSGS_AVAILABLE = False
    RoosterState = None



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
        self.inference_step = 0  # monotonic step counter for log traceability

        # Action execution
        self.action_state = ActionState.IDLE
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.last_inference_rgb = None  # frame sent to model, for waypoint annotation
        self.last_waypoint = None       # (x, y) from S2
        self.last_waypoint_rgb = None   # the RGB frame that the waypoint corresponds to
        self.waypoint_age = 0           # how many inference steps since last fresh waypoint
        self.max_waypoint_age = 12      # clear stale waypoint after this many steps

        # Handheld mode: no ARM, no keep_alive, no motor commands — just inference + discrete actions
        self.handheld_mode = self.config['bridge']['control'].get('handheld', False)

        # Arming / settling now belong to the platform adapter (see
        # robots/ROBOTICAN/adapters/rooster_manual_control.py). Only the
        # navigation-level intent stays here.
        self.arm_requested = False

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
        self._setup_actuation()

        rate = self.config['bridge']['control'].get('inference_rate', 4.0)
        self.inference_timer = self.create_timer(1.0 / rate, self._inference_callback,
                                                  callback_group=self.inference_cb_group)
        self._log_config()

    def _setup_actuation(self):
        """Attach the Rooster R1 actuation adapter, unless in handheld mode.

        `handheld` is the existing de-platforming switch: inference + discrete
        actions only, no arming, no heartbeat, no motor commands. It is also what
        lets this node run on a dev box, so the adapter is simply never created.
        """
        self.rooster = None
        if self.handheld_mode:
            return
        outputs = self.config['outputs']
        mc_cfg = outputs.get('manual_control', {})
        ka_cfg = outputs.get('keep_alive', {})
        if not (mc_cfg.get('enabled') or ka_cfg.get('enabled')):
            return
        rgb_topic = self.config['inputs']['rgb']['topic']
        rooster_id = rgb_topic.split('/')[1] if len(rgb_topic.split('/')) > 1 else "R1"
        self.rooster = RoosterManualControl(
            self,
            rooster_id=rooster_id,
            manual_control_topic=mc_cfg.get('topic'),
            keep_alive_topic=ka_cfg.get('topic'),
            publish_rate_hz=mc_cfg.get('publish_rate_hz', 40.0),
            keep_alive_rate_hz=ka_cfg.get('publish_rate_hz', 1.0),
            flight_mode=ka_cfg.get('requested_flight_mode', 1),
            stabilize_s=self.config['bridge']['control'].get('stabilize_sec', 1.0),
            callback_group=self.control_cb_group,
        )
        self.rooster.attach()

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

        if outputs.get('feedback', {}).get('enabled', True):
            self.feedback_pub = self.create_publisher(String, outputs['feedback']['topic'], 1)

        if outputs.get('status', {}).get('enabled', True):
            self.status_pub = self.create_publisher(String, outputs['status']['topic'], 1)

        # Waypoint visualization publishers (derived from discrete action topic)
        base_ns = outputs.get('discrete', {}).get('topic', '/navigation/action').rsplit('/', 1)[0]
        self.waypoint_pub = self.create_publisher(String, f"{base_ns}/waypoint", 1)
        self.waypoint_image_pub = self.create_publisher(Image, f"{base_ns}/waypoint_image", 1)
        self.get_logger().info(f"Waypoint topics: {base_ns}/waypoint, {base_ns}/waypoint_image")

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

        self.get_logger().info(f"Instruction: {instruction}")
        if self.rooster is not None and not self.rooster.armed:
            self.rooster.request_arm()

    def _nav_control_callback(self, msg: String):
        command = msg.data.strip().lower()
        with self.lock:
            if command in ["stop", "end", "finish", "done"]:
                self.nav_status = NavigationStatus.COMPLETED_SUCCESS
                self.state.is_navigating = False
                self._clear_waypoint_state()
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
                self._clear_waypoint_state()

    def _clear_waypoint_state(self):
        """Clear all waypoint tracking state."""
        self.last_waypoint = None
        self.last_waypoint_rgb = None
        self.waypoint_age = 0

    def _odom_callback(self, msg: Odometry):
        with self.lock:
            self.state.current_odometry = {
                'position': {'x': msg.pose.pose.position.x, 'y': msg.pose.pose.position.y,
                            'z': msg.pose.pose.position.z},
                'orientation': {'x': msg.pose.pose.orientation.x, 'y': msg.pose.pose.orientation.y,
                               'z': msg.pose.pose.orientation.z, 'w': msg.pose.pose.orientation.w}
            }

    def _expire_action(self):
        """Clear EXECUTING once the action's duration has elapsed.

        The adapter republishes and then idles the ManualControl frame on its own
        timer; this only tracks the *navigation* state, which gates inference
        (we do not re-infer while an action is still being executed).
        """
        with self.lock:
            if (self.action_state == ActionState.EXECUTING
                    and time.time() - self.action_start_time >= self.action_duration):
                self.action_state = ActionState.IDLE

    # === Inference ===
    def _inference_callback(self):
        control = self.config['bridge']['control']
        current_time = time.time()
        self._expire_action()

        if current_time - self.state.last_inference_time < control.get('min_inference_interval', 0.1):
            return

        with self.lock:
            if self.action_state == ActionState.EXECUTING or self.state.current_rgb is None:
                return
            if self.nav_status in [NavigationStatus.COMPLETED_SUCCESS,
                                   NavigationStatus.COMPLETED_FAILURE, NavigationStatus.IDLE]:
                if not control.get('continuous_inference'):
                    return
            if self.arm_requested and self.rooster is not None and not self.rooster.ready:
                # Armed-and-settled is a precondition for acting on a prediction.
                # request_arm() is rate-limited inside the adapter, so calling it
                # every inference tick is safe.
                self.rooster.request_arm()
                return

            payload = self._prepare_payload()

        if payload is None:
            return

        self.last_inference_rgb = payload['rgb'].copy()
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
        action = self._action_from_result(result)
        action_str = result.action
        outputs = self.config['outputs']

        # -1 (NO_ACTION) and 5 (LOOK_DOWN) are not decisions. The agent emits
        # them while System 2 tilts the camera and whenever System 1 returns an
        # empty action list, and they mean "ask me again", not "the task is
        # done". Acting on them -- which is what any unknown-index-to-STOP
        # fallback does -- brakes the robot mid-route and, under manual control,
        # actually sends STOP's axes, because that is the action_mapping entry
        # the lookup falls back to. Measured over five hospital flights:
        # System 2 said STOP zero times, the agent emitted -1 seventeen times.
        idle = result.action_index in NON_TERMINAL_IDLE_INDICES

        if action == ActionType.STOP:
            self.consecutive_stops += 1
        elif not idle:
            self.consecutive_stops = 0
        if not idle:
            self.last_action = action
        self.inference_step += 1
        step = self.inference_step

        if outputs.get('discrete', {}).get('enabled', True):
            mapping = outputs['discrete'].get('action_mapping', {})
            mapped = mapping.get(action.value, action_str)
            msg = String()
            msg.data = mapped
            self.action_pub.publish(msg)

            # Single consolidated log line per inference step
            wp = result.waypoint
            armed = self.rooster.armed if self.rooster is not None else False
            mode = "handheld" if self.handheld_mode else f"armed={armed}"
            wp_str = f"goal=({wp[0]},{wp[1]})" if wp else "goal=none"
            self.get_logger().info(f"[step {step}] {mapped} | {wp_str} [{mode}]")

        if idle:
            pass    # No decision to actuate; whatever is already running stands.
        elif self.rooster is not None and outputs.get('manual_control', {}).get('enabled'):
            if not self.rooster.armed:
                self.rooster.request_arm()
                return
            self._execute_manual_control_action(action)
        elif not self.handheld_mode and outputs.get('continuous', {}).get('enabled'):
            self._publish_velocity(action)

        if outputs.get('feedback', {}).get('enabled', True):
            feedback = {'action': action.value, 'timestamp': time.time(),
                       'inference_ms': result.inference_time_ms,
                       'armed': self.rooster.armed if self.rooster is not None else False}
            if result.waypoint:
                feedback['waypoint'] = list(result.waypoint)
            msg = String()
            msg.data = json.dumps(feedback)
            self.feedback_pub.publish(msg)

        # --- Publish S2 waypoint and annotated image ---
        self._publish_waypoint(result)

        if not idle:
            self._publish_status("paused" if action == ActionType.STOP else "navigating")

    def _execute_manual_control_action(self, action: ActionType):
        mc_cfg = self.config['outputs']['manual_control']
        action_mapping = mc_cfg.get('action_mapping', {})
        action_params = action_mapping.get(action.value, action_mapping.get('STOP', {}))

        duration = action_params.get('duration_sec', 0.5)
        axes = ManualAxes(
            x=float(action_params.get('x', 0.0)),
            y=float(action_params.get('y', 0.0)),
            z=float(action_params.get('z', 0.0)),
            r=float(action_params.get('r', 0.0)),
            buttons=int(action_params.get('buttons', 0)),
            hold_s=duration,
        )
        if self.rooster is None or not self.rooster.send(axes):
            return

        with self.lock:
            self.action_start_time = time.time()
            self.action_duration = duration
            self.action_state = ActionState.EXECUTING

        self.get_logger().info(
            f"Exec {action.value}: x={axes.x}, z={axes.z}, r={axes.r} for {duration:.2f}s")

    @staticmethod
    def _action_from_result(result) -> ActionType:
        """The action the server chose, off the index ``ModelClient`` decoded.

        ``ModelClient._parse_response`` has already turned the wire's action
        index into an :class:`ActionType` name via ``INDEX_TO_ACTION``, or into
        ``"UNKNOWN"`` when the index is one it has never heard of. Re-deriving
        it here by matching substrings of that name only creates a second,
        looser vocabulary that can disagree with the first -- and the disagreement
        lands on STOP, which is the one answer that ends a run.
        """
        try:
            return ActionType(result.action)
        except ValueError:
            return ActionType.UNKNOWN

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

    def _publish_waypoint(self, result):
        """Publish S2 waypoint data and annotated image with waypoint circle."""
        wp = result.waypoint

        if wp:
            self.last_waypoint = wp
            self.last_waypoint_rgb = self.last_inference_rgb.copy() if self.last_inference_rgb is not None else None
            self.waypoint_age = 0
        else:
            self.waypoint_age += 1
            self.get_logger().debug(f"No waypoint this step (age={self.waypoint_age})")

            # Clear stale waypoint after too many steps without a fresh one
            if self.waypoint_age > self.max_waypoint_age:
                if self.last_waypoint is not None:
                    self.get_logger().info("Clearing stale waypoint")
                self.last_waypoint = None
                self.last_waypoint_rgb = None

        # Always publish waypoint JSON (null if no active waypoint)
        active_wp = wp or self.last_waypoint
        wp_msg = String()
        wp_msg.data = json.dumps({'x': active_wp[0], 'y': active_wp[1], 'action': result.action} if active_wp else
                                 {'x': None, 'y': None, 'action': result.action})
        self.waypoint_pub.publish(wp_msg)

        # Publish annotated image: use the S2 frame (when waypoint was set), not the current frame
        if active_wp:
            # Prefer the frame from when the waypoint was captured
            frame = self.last_waypoint_rgb if self.last_waypoint_rgb is not None else self.last_inference_rgb
            if frame is not None:
                annotated = self._draw_waypoint(frame.copy(), active_wp, is_fresh=(wp is not None))
                try:
                    img_msg = self.cv_bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    self.waypoint_image_pub.publish(img_msg)
                except Exception as e:
                    self.get_logger().error(f"Waypoint image publish error: {e}")
        elif self.last_inference_rgb is not None:
            # No waypoint at all — publish current frame without circle so the panel stays alive
            try:
                img_msg = self.cv_bridge.cv2_to_imgmsg(self.last_inference_rgb.copy(), encoding='bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                self.waypoint_image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Waypoint image publish error: {e}")

    @staticmethod
    def _draw_waypoint(img: np.ndarray, waypoint: tuple, is_fresh: bool = True) -> np.ndarray:
        """Draw S2 pixel goal on inference frame — bold red circle (dimmed if stale)."""
        x, y = int(waypoint[0]), int(waypoint[1])
        h, w = img.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        if is_fresh:
            color = (0, 0, 255)       # bright red
            outer_color = (0, 0, 180)  # dark red
            label = f"S2 GOAL ({x},{y})"
        else:
            color = (0, 100, 200)      # dimmed orange
            outer_color = (0, 80, 140)
            label = f"S2 GOAL ({x},{y}) [stale]"

        # Bold outer ring
        cv2.circle(img, (x, y), 28, outer_color, 2, cv2.LINE_AA)
        # Main circle
        cv2.circle(img, (x, y), 20, color, 3, cv2.LINE_AA)
        # Center dot
        cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)

        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        lx = max(0, min(x - tw // 2, w - tw))
        ly = max(th + 4, y - 36)
        cv2.rectangle(img, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 5), (0, 0, 0), -1)
        cv2.putText(img, label, (lx, ly), font, 0.55, color, 2, cv2.LINE_AA)

        return img

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