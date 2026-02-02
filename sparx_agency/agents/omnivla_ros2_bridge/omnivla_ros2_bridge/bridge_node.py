#!/usr/bin/env python3
"""
OmniVLA ROS2 Bridge Node — Omni-Modal Navigation

Subscribes to camera + goal topics, runs OmniVLA inference,
publishes velocity commands (Twist and/or ManualControl).

Goal modalities (set any combination via ROS2 topics):
  • Language instruction  →  /R1/navigation/instruction      (String)
  • Goal image            →  /R1/navigation/goal_image       (Image)
  • Goal pose             →  /R1/navigation/goal_pose        (Float32MultiArray, 4 values)

Modality is auto-detected from which goals are currently active.
Navigation starts when ANY goal is published, stops on "done"/"reset".
"""

import sys
import time
import threading
import json
from typing import Optional

import yaml
import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from .omnivla_model import OmniVLAModel
from .visualizer import draw_trajectory

# ── Optional Rooster messages ─────────────────────────────────────────
try:
    from fcu_driver_interfaces.msg import ManualControl
    from rooster_handler_interfaces.msg import KeepAlive
    HAS_ROOSTER = True
except ImportError:
    HAS_ROOSTER = False
    ManualControl = KeepAlive = None

try:
    from std_srvs.srv import SetBool
except ImportError:
    SetBool = None


# ══════════════════════════════════════════════════════════════════════
class OmniVLABridge(Node):
    """ROS2 ↔ OmniVLA bridge with omni-modal goal support."""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__("omnivla_bridge")
        self.declare_parameter("config_path", "")
        config_path = config_path or self.get_parameter("config_path").value

        # ── Load YAML config ──────────────────────────────────────────
        cfg = self._load_yaml(config_path)
        self.cfg = cfg

        # ── Shared state (lock-protected) ─────────────────────────────
        self.lock = threading.Lock()
        self.current_rgb: Optional[np.ndarray] = None
        self.navigating: bool = False
        self.arm_state = False
        self.cv_bridge = CvBridge()

        # ── Goal slots — each can be set/cleared independently ────────
        self.goal_instruction: Optional[str] = None
        self.goal_image_pil: Optional[PILImage.Image] = None
        self.goal_pose: Optional[np.ndarray] = None      # shape (4,)

        # Load goal image from file if configured
        gi_path = cfg["inputs"].get("goal_image", {}).get("file_path", "")
        if gi_path and gi_path.strip():
            self.goal_image_pil = PILImage.open(gi_path).convert("RGB")
            self.get_logger().info(f"Loaded goal image from file: {gi_path}")

        # ── Load OmniVLA model ────────────────────────────────────────
        mcfg = cfg["model"]
        self.get_logger().info("Loading OmniVLA model …")
        self.model = OmniVLAModel(
            vla_path=mcfg["vla_path"],
            resume_step=mcfg["resume_step"],
            device=mcfg.get("device", "cuda:0"),
            max_linear=mcfg.get("max_linear_vel", 0.3),
            max_angular=mcfg.get("max_angular_vel", 0.3),
            waypoint_index=mcfg.get("waypoint_index", 4),
        )
        self.get_logger().info("OmniVLA model ready ✓")

        # ── Callback groups ───────────────────────────────────────────
        self.input_cg = ReentrantCallbackGroup()
        self.infer_cg = MutuallyExclusiveCallbackGroup()
        self.ctrl_cg  = MutuallyExclusiveCallbackGroup()

        # ── Subscribers ───────────────────────────────────────────────
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        inp = cfg["inputs"]

        # Camera (always required)
        rgb_cfg = inp["rgb"]
        msg_cls = CompressedImage if "Compressed" in rgb_cfg.get("msg_type", "") else Image
        self.create_subscription(msg_cls, rgb_cfg["topic"], self._on_rgb, qos_img,
                                 callback_group=self.input_cg)

        # Goal: language instruction
        inst_cfg = inp["instruction"]
        self.create_subscription(String, inst_cfg["topic"], self._on_instruction, 1,
                                 callback_group=self.input_cg)

        # Goal: goal image
        gi_cfg = inp.get("goal_image", {})
        if gi_cfg.get("enabled"):
            self.create_subscription(Image, gi_cfg["topic"], self._on_goal_image,
                                     qos_img, callback_group=self.input_cg)

        # Goal: goal pose (Float32MultiArray with 4 values)
        gp_cfg = inp.get("goal_pose", {})
        if gp_cfg.get("enabled"):
            self.create_subscription(Float32MultiArray, gp_cfg["topic"],
                                     self._on_goal_pose, 1,
                                     callback_group=self.input_cg)

        # Navigation control (stop / pause / resume / reset)
        nav_cfg = inp.get("nav_control", {})
        if nav_cfg.get("enabled"):
            self.create_subscription(String, nav_cfg["topic"], self._on_nav_control, 1,
                                     callback_group=self.input_cg)

        # ── Publishers ────────────────────────────────────────────────
        out = cfg["outputs"]

        if out.get("twist", {}).get("enabled"):
            self.twist_pub = self.create_publisher(Twist, out["twist"]["topic"], 1)

        if out.get("manual_control", {}).get("enabled") and HAS_ROOSTER:
            mc = out["manual_control"]
            self.mc_pub = self.create_publisher(ManualControl, mc["topic"], 10)
            self.create_timer(1.0 / mc.get("publish_rate_hz", 50.0),
                              self._mc_tick, callback_group=self.ctrl_cg)
            self._mc_msg: Optional[ManualControl] = None
            self._mc_expire: float = 0.0

        if out.get("keep_alive", {}).get("enabled") and HAS_ROOSTER:
            ka = out["keep_alive"]
            self.ka_pub = self.create_publisher(KeepAlive, ka["topic"], 10)
            self._ka_cfg = ka
            self.create_timer(1.0 / ka.get("publish_rate_hz", 1.0),
                              self._ka_tick, callback_group=self.ctrl_cg)

        if out.get("feedback", {}).get("enabled"):
            self.fb_pub = self.create_publisher(String, out["feedback"]["topic"], 1)
        if out.get("status", {}).get("enabled"):
            self.st_pub = self.create_publisher(String, out["status"]["topic"], 1)

        # Arm service (Rooster)
        self._arm_client = None
        if SetBool and HAS_ROOSTER:
            rid = rgb_cfg["topic"].split("/")[1] if "/" in rgb_cfg["topic"] else "R1"
            self._arm_client = self.create_client(SetBool, f"/{rid}/fcu/command/force_arm")

        if HAS_ROOSTER:
            from fcu_driver_interfaces.msg import UAVState
            rid = rgb_cfg["topic"].split("/")[1] if "/" in rgb_cfg["topic"] else "R1"
            self.create_subscription(UAVState, f"/{rid}/fcu/state",
                                     self._on_uav_state, 10, callback_group=self.input_cg)

        # ── Inference timer ───────────────────────────────────────────
        rate = cfg["bridge"].get("inference_rate", 3.0)
        self.create_timer(1.0 / rate, self._inference_tick,
                          callback_group=self.infer_cg)

        # ── Visualization ─────────────────────────────────────────────
        vis_cfg = cfg.get("visualization", {})
        self._vis_enabled = vis_cfg.get("enabled", False)
        self._vis_save_dir = vis_cfg.get("save_dir", "")
        self._vis_count = 0

        if self._vis_enabled:
            vis_topic = vis_cfg.get("topic", "/R1/navigation/trajectory_viz")
            self.viz_pub = self.create_publisher(Image, vis_topic, 1)
            self.get_logger().info(f"Visualization enabled → {vis_topic}")
            if self._vis_save_dir:
                import os
                os.makedirs(self._vis_save_dir, exist_ok=True)
                self.get_logger().info(f"  Saving frames to: {self._vis_save_dir}")

        self._log_summary()

    # ══════════════════════════════════════════════════════════════════
    #  GOAL STATE
    # ══════════════════════════════════════════════════════════════════
    def _has_any_goal(self) -> bool:
        """True if at least one goal modality is active."""
        return (self.goal_instruction is not None
                or self.goal_image_pil is not None
                or self.goal_pose is not None)

    def _clear_all_goals(self):
        self.goal_instruction = None
        self.goal_image_pil = None
        self.goal_pose = None
        self.navigating = False

    def _active_goals_str(self) -> str:
        parts = []
        if self.goal_instruction is not None:
            parts.append("language")
        if self.goal_image_pil is not None:
            parts.append("image")
        if self.goal_pose is not None:
            parts.append("pose")
        return "+".join(parts) if parts else "none"

    # ══════════════════════════════════════════════════════════════════
    #  CALLBACKS — GOALS
    # ══════════════════════════════════════════════════════════════════
    def _on_instruction(self, msg: String):
        text = msg.data.strip()
        with self.lock:
            if text == "" or text.lower() == "clear":
                self.goal_instruction = None
                self.get_logger().info("Goal cleared: language")
            else:
                self.goal_instruction = text
                self.navigating = True
                self.get_logger().info(f"Goal set: language = '{text}'")
                self.get_logger().info(f"  Active goals: {self._active_goals_str()}")
        self._request_arm()

    def _on_goal_image(self, msg: Image):
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.goal_image_pil = PILImage.fromarray(rgb)
                self.navigating = True
            self.get_logger().info(f"Goal set: image ({bgr.shape[1]}x{bgr.shape[0]})")
            self.get_logger().info(f"  Active goals: {self._active_goals_str()}")
            self._request_arm()
        except Exception as e:
            self.get_logger().error(f"Goal image error: {e}")

    def _on_goal_pose(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.shape[0] != 4:
            self.get_logger().error(
                f"Goal pose must have 4 values [rel_y, -rel_x, cos_h, sin_h], got {data.shape[0]}"
            )
            return
        with self.lock:
            self.goal_pose = data
            self.navigating = True
        self.get_logger().info(
            f"Goal set: pose = [{data[0]:.2f}, {data[1]:.2f}, {data[2]:.2f}, {data[3]:.2f}]"
        )
        self.get_logger().info(f"  Active goals: {self._active_goals_str()}")
        self._request_arm()

    # ══════════════════════════════════════════════════════════════════
    #  CALLBACKS — CONTROL & CAMERA
    # ══════════════════════════════════════════════════════════════════
    def _on_rgb(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.current_rgb = img
        except Exception as e:
            self.get_logger().error(f"RGB error: {e}")

    def _on_nav_control(self, msg: String):
        cmd = msg.data.strip().lower()
        with self.lock:
            if cmd in ("stop", "done", "end", "finish"):
                self._clear_all_goals()
                self._pub_status("idle")
                self.get_logger().info("Navigation stopped — all goals cleared")
            elif cmd == "pause":
                self.navigating = False
                self._pub_status("paused")
            elif cmd == "resume":
                if self._has_any_goal():
                    self.navigating = True
                    self._pub_status("navigating")
                else:
                    self.get_logger().warn("Cannot resume — no goals set")
            elif cmd == "reset":
                self._clear_all_goals()
                self._pub_status("idle")
                self.get_logger().info("Reset — all goals cleared")
            elif cmd == "clear_language":
                self.goal_instruction = None
                self.get_logger().info("Cleared: language goal")
            elif cmd == "clear_image":
                self.goal_image_pil = None
                self.get_logger().info("Cleared: image goal")
            elif cmd == "clear_pose":
                self.goal_pose = None
                self.get_logger().info("Cleared: pose goal")

    def _on_uav_state(self, msg):
        with self.lock:
            self.arm_state = msg.armed

    # ══════════════════════════════════════════════════════════════════
    #  INFERENCE LOOP
    # ══════════════════════════════════════════════════════════════════
    def _inference_tick(self):
        with self.lock:
            if not self.navigating or self.current_rgb is None:
                return
            if not self._has_any_goal():
                return
            rgb_bgr     = self.current_rgb.copy()
            instruction = self.goal_instruction
            goal_img    = self.goal_image_pil
            goal_pose   = self.goal_pose.copy() if self.goal_pose is not None else None

        # Convert BGR→RGB→PIL
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # Run model
        t0 = time.time()
        try:
            lin, ang, wps, modality = self.model.predict(
                current_image=pil_img,
                instruction=instruction,
                goal_image=goal_img,
                goal_pose=goal_pose,
            )
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return
        dt_ms = (time.time() - t0) * 1000

        self.get_logger().info(
            f"v={lin:.3f}  w={ang:.3f}  mod={modality}  ({dt_ms:.0f}ms)"
        )

        self._publish_velocity(lin, ang)
        self._pub_feedback(lin, ang, dt_ms, modality)
        self._pub_status("navigating")

        # ── Visualization ─────────────────────────────────────────────
        if self._vis_enabled:
            save_path = None
            if self._vis_save_dir:
                save_path = f"{self._vis_save_dir}/{self._vis_count:06d}.jpg"
            self._vis_count += 1

            try:
                viz_bgr = draw_trajectory(
                    current_image=pil_img,
                    waypoints=wps,
                    linear_vel=lin,
                    angular_vel=ang,
                    modality=modality,
                    goal_image=goal_img,
                    goal_pose=goal_pose,
                    waypoint_index=self.cfg["model"].get("waypoint_index", 4),
                    save_path=save_path,
                )
                # Publish as ROS2 Image
                if hasattr(self, "viz_pub"):
                    viz_msg = self.cv_bridge.cv2_to_imgmsg(viz_bgr, encoding="bgr8")
                    self.viz_pub.publish(viz_msg)
            except Exception as e:
                self.get_logger().warn(f"Visualization error: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  OUTPUT PUBLISHING
    # ══════════════════════════════════════════════════════════════════
    def _publish_velocity(self, lin: float, ang: float):
        out = self.cfg["outputs"]

        # Twist
        if hasattr(self, "twist_pub"):
            tw = Twist()
            tw.linear.x = lin
            tw.angular.z = ang
            self.twist_pub.publish(tw)

        # ManualControl (Rooster ground-roll)
        # x: forward/backward  (-1000 .. 1000)
        # y: not used in roll mode
        # z: TILT              (-1000 .. 1000)
        # r: yaw rotation      (-1000 .. 1000)
        if hasattr(self, "mc_pub") and HAS_ROOSTER:
            mc_cfg = out["manual_control"]
            if not self.arm_state:
                self._request_arm()
                return

            msg = ManualControl()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.y = 0.0
            msg.buttons = 0

            is_moving = abs(lin) > 0.01 or abs(ang) > 0.05

            if not is_moving:
                msg.x = 0.0
                msg.z = float(mc_cfg.get("stop_tilt", -1000.0))
                msg.r = 0.0
            elif abs(lin) < 0.01:
                msg.x = 0.0
                msg.z = float(mc_cfg.get("turn_tilt", 250.0))
                msg.r = float(np.clip(ang * mc_cfg.get("angular_scale", 2000.0),
                                      -1000.0, 1000.0))
            else:
                msg.x = float(np.clip(lin * mc_cfg.get("linear_scale", 800.0),
                                      -1000.0, 1000.0))
                msg.z = float(np.clip(lin * mc_cfg.get("tilt_scale", 1300.0),
                                      0.0, mc_cfg.get("max_tilt", 500.0)))
                msg.r = float(np.clip(ang * mc_cfg.get("angular_scale", 2000.0),
                                      -1000.0, 1000.0))

            dur = mc_cfg.get("duration_sec", 0.3)
            with self.lock:
                self._mc_msg = msg
                self._mc_expire = time.time() + dur

    def _mc_tick(self):
        if not hasattr(self, "mc_pub"):
            return
        with self.lock:
            if self._mc_msg and time.time() < self._mc_expire:
                msg = self._mc_msg
            else:
                msg = ManualControl()
                msg.header = Header()
                msg.x = 0.0
                msg.y = 0.0
                msg.z = float(self.cfg["outputs"]["manual_control"].get("stop_tilt", -1000.0))
                msg.r = 0.0
                msg.buttons = 0
                self._mc_msg = None
        msg.header.stamp = self.get_clock().now().to_msg()
        self.mc_pub.publish(msg)

    def _ka_tick(self):
        if not hasattr(self, "ka_pub"):
            return
        msg = KeepAlive()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.is_active = self._ka_cfg.get("is_active", True)
        msg.requested_flight_mode = self._ka_cfg.get("requested_flight_mode", 1)
        msg.command_reboot = False
        self.ka_pub.publish(msg)

    def _request_arm(self):
        if self._arm_client and self._arm_client.service_is_ready():
            req = SetBool.Request()
            req.data = True
            self._arm_client.call_async(req)

    # ── Feedback ──────────────────────────────────────────────────────
    def _pub_feedback(self, lin, ang, ms, modality):
        if not hasattr(self, "fb_pub"):
            return
        msg = String()
        msg.data = json.dumps({
            "linear": lin, "angular": ang,
            "inference_ms": ms, "modality": modality,
            "t": time.time(),
        })
        self.fb_pub.publish(msg)

    def _pub_status(self, status):
        if not hasattr(self, "st_pub"):
            return
        msg = String()
        msg.data = status
        self.st_pub.publish(msg)

    # ── Config ────────────────────────────────────────────────────────
    @staticmethod
    def _load_yaml(path: str) -> dict:
        if path and path.strip():
            with open(path) as f:
                return yaml.safe_load(f)
        raise RuntimeError("Config path required.  --config <path.yaml>")

    def _log_summary(self):
        i, o = self.cfg["inputs"], self.cfg["outputs"]
        self.get_logger().info("═" * 50)
        self.get_logger().info("OmniVLA Bridge — Omni-Modal Navigation")
        self.get_logger().info(f"  Camera:      {i['rgb']['topic']}")
        self.get_logger().info(f"  Instruction: {i['instruction']['topic']}")
        gi = i.get("goal_image", {})
        if gi.get("enabled"):
            self.get_logger().info(f"  Goal image:  {gi['topic']}")
        gp = i.get("goal_pose", {})
        if gp.get("enabled"):
            self.get_logger().info(f"  Goal pose:   {gp['topic']}")
        if o.get("twist", {}).get("enabled"):
            self.get_logger().info(f"  Twist out:   {o['twist']['topic']}")
        if o.get("manual_control", {}).get("enabled") and HAS_ROOSTER:
            self.get_logger().info(f"  ManualCtrl:  {o['manual_control']['topic']}")
        vis = self.cfg.get("visualization", {})
        if vis.get("enabled"):
            self.get_logger().info(f"  Trajectory:  {vis.get('topic', '/R1/navigation/trajectory_viz')}")
        self.get_logger().info(f"  Rooster:     {HAS_ROOSTER}")
        self.get_logger().info("═" * 50)


# ══════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    config_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
    node = OmniVLABridge(config_path)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()