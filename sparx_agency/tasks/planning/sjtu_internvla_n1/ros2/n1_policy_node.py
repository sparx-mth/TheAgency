#!/usr/bin/env python3
"""InternVLA-N1 policy node: instruction + camera -> a committed world route.

This is the "trajectory that comes out of N1" half of the SJTU flight stack, and
it is the exact shape the NavDP node has in FALCON: subscribe the camera and the
pose, ask the policy for a body-frame trajectory, **anchor that trajectory in the
world at the pose it was asked from, and fly it as a route** rather than
re-inferring every frame. Only the last part is this node's own logic; the
anchoring and the commit discipline are
:class:`~sparx_agency.core.planning.vlas.common.plan_commit.executor.PlanCommitExecutor`,
shared with NavDP, and the policy is the uniform
:class:`~sparx_agency.core.planning.vlas.internvla_n1.policy.InternVLAN1Policy`.

The committed route is published as a plain ``nav_msgs/Path`` in the world frame.
A separate follower node pursues it and produces ``/cmd_vel`` -- the same split
as FALCON's ``navdp_click_node`` -> ``waypoint_follower_node``, so the policy
never touches the airframe and the follower never talks to a GPU.

**The aircraft stands still while the model thinks, and turns when it says turn.**
Those are the two things that make a dual-system VLA flyable rather than merely
connected, and both were absent:

* System 2 needs seconds. An aircraft that keeps flying through them shows the
  model a frame from where it *was*, and then anchors the answering route there
  too -- so a 3 m curve begins metres behind the drone and the pursuit's first
  job is to fly backwards to the start of it. Holding costs flight time and buys
  the only property the stack is judged on: the observation, the decision and
  the anchor are one place. It is also the regime the policy was trained in --
  VLN-CE takes every observation from a standstill.
* A discrete TURN action is a **rotation**, and rendering it as a short bent
  waypoint lets a holonomic tracker satisfy it by crabbing sideways. The model
  asks to look somewhere and the aircraft shuffles 0.25 m instead, sees the same
  wall, and asks again. Turns are handed to the follower as a heading
  (``/n1/yaw_goal``) and flown as a slow rotation that ends stopped.

The model runs on the GPU behind an HTTP server; **this node is CPU-only** and
must be, so the network keeps the whole card. It imports no torch and sets
``CUDA_VISIBLE_DEVICES=""`` for its own process as a belt-and-braces guard.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time

# Keep this process off the GPU: the policy runs on the server, everything here
# is numpy on the CPU, and the card belongs to the network alone.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy._rclpy_pybind11 import RCLError
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, String

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.core.planning.safety.depth_proximity_brake import (
    DepthProximityBrakeConfig,
    freer_side,
)
from sparx_agency.core.planning.vlas.common.plan_commit.executor import (
    CommitSpec,
    PlanCommitExecutor,
)
from sparx_agency.core.planning.vlas.common.turn_in_place import (
    TurnInPlace,
    describe as describe_turn,
    turn_spec_from_config,
)
from sparx_agency.core.planning.vlas.interfaces.goals import LanguageGoal
from sparx_agency.core.planning.vlas.interfaces.policy import PolicyObservation
from sparx_agency.core.planning.vlas.registry import default_vla_registry
from sparx_agency.tasks.planning.sjtu_internvla_n1.recording import FpsMeter


def _polyline_length(xy):
    """Arc length of an (N, 2) polyline, metres. Zero for fewer than two points."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _yaw_from_quat(q):
    """Yaw (radians, CCW from +x) from a geometry_msgs quaternion."""
    return yaw_from_quaternion((q.x, q.y, q.z, q.w))


def _load_config(path):
    """Read the SJTU/N1 binding YAML, or return an empty dict if unset."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


class N1PolicyNode(Node):
    """Drive InternVLA-N1 and publish its committed route as a world path."""

    def __init__(self):
        super().__init__("n1_policy_node")
        self.declare_parameter("config_file", "")
        cfg = _load_config(self.get_parameter("config_file").value)

        topics = cfg.get("topics", {})
        frames = cfg.get("frames", {})
        server = cfg.get("server", {})
        camera = cfg.get("camera", {})
        pp = cfg.get("policy_params", {})
        commit = cfg.get("commit", {})

        self._world_frame = frames.get("world", "world")
        self._rgb_topic = topics.get("rgb", "/simple_drone/front/image_raw")
        self._rgb_compressed = topics.get("rgb_type", "raw") == "compressed"
        self._depth_topic = topics.get("depth", "/simple_drone/front_depth/depth/image_raw")
        self._depth_enabled = bool(topics.get("depth_enabled", True))
        self._odom_topic = topics.get("odom", "/simple_drone/odom")
        self._instruction_topic = topics.get("instruction", "/simple_drone/navigation/instruction")
        self._path_topic = topics.get("trajectory", "/simple_drone/n1/trajectory")
        self._full_path_topic = topics.get("trajectory_full", "/simple_drone/n1/trajectory_full")
        self._hold_topic = topics.get("hold", "/simple_drone/n1/hold")
        self._yaw_goal_topic = topics.get("yaw_goal", "/simple_drone/n1/yaw_goal")
        self._blocked_topic = topics.get("blocked", "/simple_drone/n1/blocked")

        self._instruction = pp.get("default_instruction", "explore the warehouse")
        self._inference_rate = float(pp.get("inference_rate_hz", 4.0))
        self._control_rate = float(cfg.get("follower", {}).get("control_rate_hz", 20.0))

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._pose = None  # (x, y, z, yaw)
        self._speed = 0.0      # |horizontal velocity|, m/s, from odom
        self._yaw_rate = 0.0   # rad/s, from odom
        self._blocked = False  # the follower's depth reflex allows no forward speed

        # The policy: built through the registry so the string name is the only
        # coupling, exactly as an arbiter would build any VLA.
        model_settings = self._model_settings(camera, pp)
        self._policy = default_vla_registry().create(
            "internvla_n1",
            host=server.get("host", "127.0.0.1"),
            port=int(server.get("port", 8087)),
            timeout_s=float(server.get("timeout_sec", 30.0)),
            step_m=float(pp.get("step_m", 0.25)),
            turn_deg=float(pp.get("turn_deg", 15.0)),
            model_settings=model_settings,
            logger=self.get_logger(),
        )

        self._executor = PlanCommitExecutor(CommitSpec(
            fraction=float(commit.get("fraction", 0.5)),
            lookahead_m=float(commit.get("lookahead_m", 1.0)),
            arrive_radius_m=float(commit.get("arrive_radius_m", 0.15)),
            min_commit_m=float(commit.get("min_commit_m", 0.20)),
            max_commit_s=float(commit.get("max_commit_s", 8.0)),
            # NOT OPTIONAL, AND THEY WERE MISSING. Without these two the spec
            # falls back to a FLAT `max_commit_s`, so every commitment -- a 2 m
            # curve and a 0.25 m step alike -- sits out the full ceiling before
            # it can be replaced. The YAML has documented them, and the twelve
            # second stalls they were meant to remove, since before the node
            # read them: measured in the hospital, five separate twelve-second
            # stalls in a ninety-second flight, which was most of the time the
            # aircraft spent stationary. Found by scripts/dry_run.py.
            expected_speed_mps=float(commit.get("expected_speed_mps", 0.0)),
            commit_grace_s=float(commit.get("commit_grace_s", 2.0)),
            max_deviation_m=float(commit.get("max_deviation_m", 1.5)),
            min_period_s=float(commit.get("min_period_s", max(0.1, 1.0 / self._inference_rate))),
        ))
        foll = cfg.get("follower", {})

        # ── stop, look, think ────────────────────────────────────────
        # The whole flight is a sequence of stationary observations, which is
        # what the policy was trained on. `hold_to_think: false` restores the
        # fly-while-thinking behaviour, for comparison and nothing else.
        self._hold_to_think = bool(pp.get("hold_to_think", True))
        self._settle_speed = float(pp.get("settle_speed_mps", 0.05))
        self._settle_yaw_rate = float(pp.get("settle_yaw_rate_rad_s", 0.05))
        self._settle_s = float(pp.get("settle_s", 0.3))
        self._settle_timeout_s = float(pp.get("settle_timeout_s", 3.0))
        self._held = False
        self._hold_since = None
        self._settle_since = None
        self._deciding = False
        self._think_since = 0.0
        self._phase = "waiting"
        # The last decision's status, republished every status tick with the
        # live phase folded in. A decision now lasts seconds, so a status that
        # is only published when one is taken leaves the overlay reporting
        # "thinking" for the whole of the flight that follows it -- and a
        # motionless aircraft that is thinking looks exactly like a motionless
        # aircraft that is wedged.
        self._info = {}
        self._last_status_s = 0.0
        self._status_period_s = float(pp.get("status_period_s", 0.2))

        # ── a turn is a rotation ─────────────────────────────────────
        # `rotate` flies a discrete turn as a rotation that ends stopped;
        # `crab` restores the old behaviour of flying the bent step it is
        # rendered as. The pair exists so the two can be compared on the same
        # aircraft; `rotate` is what the model means.
        self._turn_mode = str(pp.get("discrete_turn_mode", "rotate")).lower()
        self._turn = TurnInPlace(turn_spec_from_config(foll.get("turn", {}) or {}))
        self._min_commit_m = float(commit.get("min_commit_m", 0.20))
        self._pivot_min_rad = float(np.deg2rad(float(pp.get("pivot_min_deg", 7.5))))
        self._pivot_min_reach_m = float(pp.get("pivot_min_reach_m", 0.05))
        self._turns = 0
        self._turns_flown = 0

        # ── the blocked-forward escape ───────────────────────────────
        # Not a planner. The depth reflex in the follower can pin the aircraft
        # at zero forward speed indefinitely, and a policy that cannot see that
        # will keep asking to fly forward from an identical stationary frame --
        # a loop that this stack has already lost most of a flight to.
        brake_cfg = cfg.get("brake", {})
        self._escape_after = int(pp.get("blocked_escape_after", 3))
        self._escape_turn_rad = float(np.deg2rad(float(pp.get("blocked_escape_deg", 45.0))))
        self._blocked_forward = 0
        self._escapes = 0
        self._brake_cfg = DepthProximityBrakeConfig(
            fx=float(camera.get("fx", 390.642735)), fy=float(camera.get("fy", 390.642735)),
            cx=float(camera.get("cx", 300.0)), cy=float(camera.get("cy", 300.0)),
            corridor_halfheight_m=float(brake_cfg.get("corridor_halfheight_m", 0.35)),
            min_valid_m=float(brake_cfg.get("min_valid_m", 0.15)),
            stride=int(brake_cfg.get("stride", 4)))

        self._last_inference_s = 0.0
        self._commits = 0
        # Look-down state. System 2 asks for a lower view of the scene and then
        # computes its pixel goal in that frame; this airframe cannot tilt its
        # camera, so it DIPS instead and keeps using the forward camera. The
        # sequence is: request the dip, wait until the aircraft is actually
        # down there, send exactly one frame from the low altitude (the one the
        # agent is waiting for), then climb back.
        self._dip_state = "none"   # none | descending | holding
        self._dip_since = 0.0
        self._dip_m = abs(float(pp.get("look_down_dip_m", 0.5)))
        self._dip_tol_m = float(pp.get("look_down_tolerance_m", 0.15))
        self._dip_timeout_s = float(pp.get("look_down_timeout_s", 6.0))
        # The follower owns the cruise altitude; the dip is expressed relative
        # to it, so this node has to read the same number out of the same file.
        self._cruise_alt = float(cfg.get("follower", {}).get("target_altitude_m", 1.2))
        self._decisions = 0
        self._curve_decisions = 0
        self._min_inference_interval = 1.0 / max(1e-3, self._inference_rate)

        # TWO CALLBACK GROUPS, AND THE REASON IS THE HTTP CALL. `_tick` blocks
        # inside `policy.step()` for as long as System 2 takes -- seconds. On a
        # single-threaded executor that stops EVERYTHING in this node for the
        # duration: no odometry is fused, so the settle gate and the rotation
        # tracker are reading a pose from before the aircraft stopped, and no
        # status is published, so a recording shows the last phase before the
        # think and a viewer cannot tell a five-second decision from a hang.
        #
        # `_tick` gets a mutually exclusive group so it can never re-enter
        # itself mid-inference; everything else shares a reentrant one and keeps
        # running while it waits.
        self._tick_group = MutuallyExclusiveCallbackGroup()
        self._live_group = ReentrantCallbackGroup()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        rgb_type = CompressedImage if self._rgb_compressed else Image
        self.create_subscription(rgb_type, self._rgb_topic, self._on_rgb, sensor_qos,
                                 callback_group=self._live_group)
        if self._depth_enabled:
            self.create_subscription(Image, self._depth_topic, self._on_depth, sensor_qos,
                                     callback_group=self._live_group)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, sensor_qos,
                                 callback_group=self._live_group)
        self.create_subscription(String, self._instruction_topic, self._on_instruction, 1,
                                 callback_group=self._live_group)
        self.create_subscription(Bool, self._blocked_topic, self._on_blocked, latched,
                                 callback_group=self._live_group)

        self._path_pub = self.create_publisher(Path, self._path_topic, latched)
        self._full_path_pub = self.create_publisher(Path, self._full_path_topic, latched)
        self._info_topic = topics.get("info", "/simple_drone/n1/info")
        self._info_pub = self.create_publisher(String, self._info_topic, latched)
        self._alt_offset_pub = self.create_publisher(
            Float32, topics.get("altitude_offset", "/simple_drone/n1/altitude_offset"), 1)
        self._hold_pub = self.create_publisher(Bool, self._hold_topic, 1)
        self._yaw_goal_pub = self.create_publisher(Float32, self._yaw_goal_topic, 1)

        # FPS meters: the model reports per-step System-1 / System-2 durations
        # (the trajectory-patched agent), which these turn into a smoothed rate.
        self._s1_fps = FpsMeter()
        self._s2_fps = FpsMeter()
        self._last_fps_log_s = 0.0
        self._cam_w = int(camera.get("width", 600))
        self._cam_h = int(camera.get("height", 600))

        self.create_timer(1.0 / max(1e-3, self._control_rate), self._tick,
                          callback_group=self._tick_group)
        # The status publisher is its own timer on the live group, so the
        # overlay keeps saying THINKING (and counting the seconds) while the
        # tick above is blocked in the model server.
        self.create_timer(max(0.05, self._status_period_s), self._status_tick,
                          callback_group=self._live_group)
        self.create_timer(1.5, self._init_once,  # one-shot server init after startup
                          callback_group=self._live_group)
        self._initialised = False

        self.get_logger().info(
            "n1_policy_node up: rgb=%s depth=%s odom=%s -> path=%s (server %s:%s), "
            "instruction=%r" % (self._rgb_topic, self._depth_topic if self._depth_enabled else "off",
                                self._odom_topic, self._path_topic,
                                server.get("host", "127.0.0.1"), server.get("port", 8087),
                                self._instruction))
        self.get_logger().info(
            "decisions are taken from a standstill: hold_to_think=%s (settle "
            "<= %.2f m/s for %.1f s, %.1f s timeout); discrete turns flown as "
            "%s -- %s"
            % (self._hold_to_think, self._settle_speed, self._settle_s,
               self._settle_timeout_s, self._turn_mode, describe_turn(self._turn.spec)))

    # ── setup ────────────────────────────────────────────────────────
    @staticmethod
    def _model_settings(camera, policy_params=None):
        """Server model settings: the SJTU camera model, and the S2 cadence.

        The FALCON/SJTU stack's single most expensive bug was a wrong camera
        intrinsic; the server projects its pixel goal with these, so they are
        passed explicitly rather than left at the 640x480 default.

        ``sys2_max_forward_step`` and ``sys1_continuous_only`` are the two knobs
        that trade decision rate for
        *continuity*. Measured over 168 System-2 replies: 92.7% of look-downs
        are followed by pixel coordinates, and coordinates are the only branch
        that runs System 1 -- so every System-2 call is very nearly one chance
        at a continuous trajectory. Firing System 2 more often (a lower value)
        therefore buys more curves per metre flown, and costs decision rate,
        because System 2 is ~98.5% of the per-decision budget. The agent reads
        it off ``model_settings`` with a default of 8, and ``ModelCfg`` allows
        extra keys, so it travels on ``/agent/init`` without an upstream patch.
        """
        settings = {}
        if policy_params and policy_params.get("sys2_max_forward_step") is not None:
            settings["sys2_max_forward_step"] = int(policy_params["sys2_max_forward_step"])
        # THE CONTINUOUS SWITCH (server PATCH 7). With it on, the agent stops
        # queueing the discretisation of a curve it has already handed over, so
        # every System-1 step returns a fresh curve instead of three stale
        # 0.25 m stubs. It also changes what `sys2_max_forward_step` counts --
        # System-1 runs rather than executed action steps -- which is why the
        # two are set together in the YAML.
        if policy_params and policy_params.get("sys1_continuous_only") is not None:
            settings["sys1_continuous_only"] = bool(policy_params["sys1_continuous_only"])
        if not camera:
            return settings
        fx = float(camera.get("fx", 390.642735))
        fy = float(camera.get("fy", 390.642735))
        cx = float(camera.get("cx", 300.0))
        cy = float(camera.get("cy", 300.0))
        settings.update({
            "camera_intrinsic": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "width": int(camera.get("width", 600)),
            "height": int(camera.get("height", 600)),
        })
        if "hfov_deg" in camera:
            settings["hfov"] = float(camera["hfov_deg"])
        return settings

    def _init_once(self):
        if self._initialised:
            return
        self._initialised = True
        try:
            self._policy.reset()
            self.get_logger().info("InternVLA-N1 server agent initialised")
        except Exception as exc:  # noqa: BLE001 - report and keep trying on inference
            self.get_logger().warn("policy reset failed (will retry on step): %s" % (exc,))

    # ── subscriptions ────────────────────────────────────────────────
    def _on_rgb(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                import cv2
                bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                rgb = bgr[:, :, ::-1]
            else:
                rgb = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            with self._lock:
                self._rgb = np.ascontiguousarray(rgb)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("rgb decode failed: %s" % (exc,))

    def _on_depth(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg)  # 32FC1 metres
            with self._lock:
                self._depth = np.asarray(depth, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("depth decode failed: %s" % (exc,))

    def _on_odom(self, msg):
        p = msg.pose.pose
        t = msg.twist.twist
        # The speed matters as much as the pose here: an observation is only
        # worth taking once the aircraft has actually stopped, and this airframe
        # translates by tilting, so it coasts well past the command going to
        # zero. The horizontal magnitude is frame-independent, which is why this
        # does not care whether the plugin reports twist in body or world axes.
        speed = float(np.hypot(t.linear.x, t.linear.y))
        with self._lock:
            self._pose = (p.position.x, p.position.y, p.position.z,
                          _yaw_from_quat(p.orientation))
            self._speed = speed
            self._yaw_rate = float(t.angular.z)

    def _on_blocked(self, msg):
        with self._lock:
            self._blocked = bool(msg.data)

    def _on_instruction(self, msg):
        text = msg.data.strip()
        if text:
            self._instruction = text
            self.get_logger().info("instruction: %r" % (text,))

    # ── the loop ─────────────────────────────────────────────────────
    def _tick(self):
        if not rclpy.ok():
            # The context is being torn down. Publishing into it raises
            # RCLError out of the timer callback, the node exits 1, and the
            # launch file's `on_exit=Shutdown()` reports a normal teardown as
            # "this node died" -- which makes the one alarm that exists for a
            # node dying at import useless.
            return
        with self._lock:
            pose = self._pose
            speed, yaw_rate = self._speed, self._yaw_rate
            rgb = None if self._rgb is None else self._rgb.copy()
            depth = None if self._depth is None else self._depth.copy()
            instruction = self._instruction
            blocked = self._blocked
        if pose is None:
            return

        now = time.time()

        # A ROTATION IS IN FLIGHT. Nothing else happens until the aircraft has
        # turned and stopped, because the frame at the end of the turn is the
        # entire reason the model asked for it. The same TurnInPlace the
        # follower flies, fed the same odometry -- this copy reads only `done`,
        # so the two can never disagree about whether the turn has happened.
        if self._turn.active:
            cmd = self._turn.update(pose[3], now, yaw_rate)
            if not cmd.done:
                self._phase = "turning"
                return
            self._turns_flown += 1
            self.get_logger().info(
                "turn %d finished at %.1f deg%s"
                % (self._turns_flown, np.degrees(pose[3]),
                   " (TIMED OUT -- the aircraft did not turn)" if cmd.timed_out else ""))

        # A look-down is in flight: hold off inference until the aircraft is
        # actually down at the dipped altitude, so the ONE frame the agent is
        # waiting for is genuinely the lower view it asked for. Sending the
        # cruise-altitude frame instead is what the stack did before, and the
        # model then computed its pixel goal in a frame it believed was tilted.
        if self._dip_state == "descending":
            # One-sided. A symmetric window goes true the instant the aircraft
            # crosses INTO the band on the way down, which on a 0.5 m dip with
            # a 0.15 m tolerance sends the frame at 0.35 m of descent -- 70% of
            # the dip, and the lower view is the entire point of doing it.
            reached = pose[2] <= (self._cruise_alt - self._dip_m) + 0.02
            if reached:
                self._dip_state = "holding"
                # Restart the settle window. The hold has been on since before
                # the descent, and without this the timeout below fires the
                # moment the aircraft arrives and reports "never settled" about
                # an aircraft that has been stationary the whole time.
                self._hold_since = None
                self._settle_since = None
                self.get_logger().info("look-down: at %.2f m, sending the low frame" % pose[2])
            elif now - self._dip_since > self._dip_timeout_s:
                # It never got there -- blocked, or the follower is not tracking.
                # Send the frame anyway rather than wedging the episode; the
                # agent is blocked waiting for it.
                self._dip_state = "holding"
                self._hold_since = None
                self._settle_since = None
                self.get_logger().warn(
                    "look-down: never reached %.2f m (at %.2f m after %.1f s); "
                    "sending the frame from here"
                    % (self._cruise_alt - self._dip_m, pose[2], now - self._dip_since))
            else:
                self._phase = "dipping"
                return

        tick = self._executor.tick(pose[0], pose[1], now)
        # The agent is blocked waiting for the look-down frame, so the commit
        # executor does not get a vote on whether to step. A flag, not a `tick =
        # None` -- the tick object is read again further down and nulling it
        # simply moved the problem to a line that had no reason to expect it.
        forced = self._dip_state == "holding"
        # `_deciding` LATCHES the decision. Once a replan reason has been seen
        # the aircraft is committed to stopping and asking, however many control
        # steps that takes -- otherwise the settle below would release the hold
        # on the first tick that the executor happened not to re-raise a reason,
        # and the aircraft would creep forward through its own observation.
        if not self._deciding and not forced and tick.replan_reason is None:
            self._phase = "flying"
            self._set_hold(False)
            return
        self._deciding = True
        if rgb is None or now - self._last_inference_s < self._min_inference_interval:
            return

        # STOP BEFORE LOOKING. System 2 takes seconds, and until this was here
        # the aircraft flew through every one of them: the frame the model
        # reasoned about was metres behind the aircraft by the time the answer
        # arrived, and the route was anchored at the pose the frame was taken
        # from -- so a 3 m curve started 2 m behind the drone and the pursuit
        # spent its first second flying backwards to reach the start of it.
        # Holding costs flight time and buys the one thing the whole stack is
        # judged on: the observation, the answer and the anchor are the same
        # place.
        if self._hold_to_think:
            self._set_hold(True)
            if not self._settled(speed, yaw_rate, now):
                self._phase = "settling"
                return

        self._phase = "thinking"
        self._think_since = now
        if not forced:
            self._executor.mark_attempt(now)
        self._last_inference_s = now
        result = self._policy.step(
            PolicyObservation(rgb=rgb, depth_m=depth, altitude_m=pose[2]),
            LanguageGoal(instruction=instruction))

        if result.metadata.get("transport_failed"):
            # Keep the current commitment; the server dropped a frame. Say so,
            # though: a silent return leaves /n1/info frozen on the last good
            # sample, so the overlay keeps reporting healthy FPS while nothing
            # is reaching the model at all -- the exact symptom a wedged server
            # produces, and indistinguishable from a policy that is thinking.
            #
            # And RELEASE the aircraft. A held drone waiting on a server that is
            # not answering is a hover with no end; the plan it already has is
            # the best thing available, so fly it and ask again.
            self._finish_decision()
            self.get_logger().warn(
                "no usable answer from the model server (%s); keeping the "
                "current commitment" % (result.metadata.get("error"),),
                throttle_duration_sec=5.0)
            return

        self._publish_info(result, instruction, now, pose)

        # The look-down frame reached the server, so the dip has done its job:
        # climb back. Cleared HERE and not before the transport check, because a
        # request that never arrived leaves the agent still waiting for the low
        # frame -- clearing on the attempt sends it the cruise-altitude frame
        # instead, which is the failure performing the dip exists to remove.
        #
        # Two independent `if`s, not an if/elif: System 2 answers a look-down
        # frame with another down arrow often enough, and an `elif` swallowed
        # the second request on the very step that carried it.
        if self._dip_state == "holding":
            self._dip_state = "none"
            self._set_altitude_offset(0.0)
            self.get_logger().info("look-down: frame delivered, returning to cruise")
        if result.metadata.get("look_down"):
            # THE HOLD STAYS ON THROUGH THE DIP. System 2 asked for a lower view
            # of *this* scene and will compute its pixel goal in the frame it
            # gets back; an aircraft that flies two metres while descending
            # answers with a lower view of somewhere else, which is a subtler
            # version of the same lie performing the dip exists to stop telling.
            # `_deciding` therefore stays latched -- the decision is not over,
            # it is waiting for a frame.
            self._dip_state = "descending"
            self._dip_since = now
            self._set_altitude_offset(-self._dip_m)
            self._phase = "dipping"
            self.get_logger().info(
                "look-down requested: dipping %.2f m to %.2f m"
                % (self._dip_m, self._cruise_alt - self._dip_m))
            return

        if result.metadata.get("idle") and not result.ok:
            # No new decision this tick -- System 2 is looking down, or System 1
            # returned no actions. NOT a stop: the commitment already in the air
            # is still the best plan there is, so leave it alone and let the
            # follower keep flying it. Treating this as a stop (which it was,
            # until -1 got its own name) abandoned a route mid-flight every time
            # the model tilted its camera at the floor.
            #
            # `and not result.ok` because the agent fills the trajectory BEFORE
            # it decides the action list was empty, so a real System-1 curve can
            # arrive alongside index -1. A curve always wins over an idle flag:
            # it is the thing this whole stack exists to fly.
            self._finish_decision()
            self.get_logger().info(
                "N1 idle (%s): keeping the current commitment"
                % (result.metadata.get("action"),), throttle_duration_sec=5.0)
            return

        if not result.ok:
            # A genuine STOP: the policy says the task is done. Hold by
            # publishing an empty route, which the follower reads as "stop and
            # hover".
            self._executor.reset()
            self._publish_path(self._path_pub, [], pose)
            self._finish_decision()
            self._phase = "stopped"
            self.get_logger().info("N1 STOP (%s): holding" % (result.metadata.get("action"),))
            return

        self._act(result, pose, depth, blocked, now, forced, tick)

    # ── acting on one decision ───────────────────────────────────────
    def _act(self, result, pose, depth, blocked, now, forced, tick):
        """Fly the decision: as a rotation where it is one, otherwise as a route.

        The split is the whole difference between "the model turned" and "the
        model shuffled sideways". A discrete TURN action carries no distance at
        all upstream -- it rotates and nothing else -- and a System-1 curve too
        short to be a route is the same message in continuous clothing: the
        model is asking to pivot and look again, not to travel 0.25 m.
        """
        body = np.asarray(result.trajectory, dtype=float)[:, :2]
        arc = _polyline_length(body)
        delta = self._rotation_intent(result, body, arc)

        if delta is None and blocked:
            # BLOCKED, AND ASKING TO TRANSLATE ANYWAY. The depth reflex allows
            # no forward speed, so this decision cannot be flown; committing it
            # again produces the same stationary frame and the same answer, for
            # ever. Measured before this existed: seventy of a ninety-second
            # flight, pinned 0.43 m from a wall, re-committing a 0.25 m forward
            # step every twelve seconds.
            self._blocked_forward += 1
            if self._blocked_forward >= self._escape_after:
                side = (freer_side(depth, self._brake_cfg)
                        if depth is not None else 1.0)
                delta = side * self._escape_turn_rad
                self._blocked_forward = 0
                self._escapes += 1
                self.get_logger().warn(
                    "BLOCKED ESCAPE %d: %d decisions asking to fly forward into "
                    "something the depth corridor says is impassable; turning "
                    "%.0f deg %s to look somewhere else"
                    % (self._escapes, self._escape_after,
                       np.degrees(self._escape_turn_rad),
                       "left" if side > 0 else "right"))
        else:
            self._blocked_forward = 0

        if delta is not None:
            self._begin_turn(pose, delta, now, result)
            return

        self._executor.commit(body, (pose[0], pose[1], pose[3]), now)
        self._publish_path(self._path_pub, self._executor.plan.committed_xy, pose)
        self._publish_path(self._full_path_pub, self._executor.plan.world_xy, pose)
        self._finish_decision()
        self._phase = "flying"
        # One line per commitment, because "the server answered 200" and "a route
        # was actually flown" are different claims and only this one is evidence
        # of the second. run_sjtu_n1.sh counts these to decide whether a
        # recording is a result at all.
        #
        # The [curve]/[action] tag is read off the committed shape rather than a
        # metadata flag: `trajectory_from_action` renders exactly one waypoint,
        # so an anchored commitment of two points IS the discrete fallback and
        # anything longer is System 1's integrated curve. Whether the curve is
        # being flown at all is the whole question the patched server exists to
        # answer, so this label has to be derived from something real.
        committed = self._executor.plan.committed_xy
        self._commits += 1
        self.get_logger().info(
            "committed #%d: %d pts, %.2f m, from (%.2f, %.2f) after %s [%s]"
            % (self._commits, len(committed), _polyline_length(committed),
               pose[0], pose[1],
               "look-down frame" if forced else tick.replan_reason,
               "curve" if len(committed) > 2 else "action"))

    def _rotation_intent(self, result, body, arc):
        """How far this decision wants to rotate, radians CCW, or ``None``.

        Two ways a decision is a rotation:

        * the server answered with a discrete TURN action, which upstream moves
          the heading and nothing else (``turn_delta_rad`` on the policy result);
        * System 1 returned a curve too short to be a route. Its shape still
          says where the model wants to be looking, and flying 0.2 m of it
          throws that away -- the aircraft creeps forward and the view barely
          changes, which is exactly the loop this whole node exists to break.

        Returns ``None`` in ``crab`` mode, which restores the old behaviour of
        flying every decision as a path. Kept so the two can be compared on the
        same aircraft rather than argued about.
        """
        if self._turn_mode != "rotate":
            return None
        delta = result.metadata.get("turn_delta_rad")
        if delta is not None:
            return float(delta)
        if arc >= self._min_commit_m or len(body) < 2:
            return None
        end = body[-1]
        if float(np.hypot(end[0], end[1])) < self._pivot_min_reach_m:
            # A curve with no reach at all has no bearing either -- atan2 of
            # nearly (0, 0) is noise, and turning on noise is worse than
            # standing still. Let it be committed and fail the executor's own
            # TOO_SHORT test, which is the honest report.
            return None
        bearing = float(np.arctan2(end[1], end[0]))
        if abs(bearing) < self._pivot_min_rad:
            return None
        return bearing

    def _begin_turn(self, pose, delta, now, result=None):
        """Ask the follower to rotate, and stand down until it has.

        The committed route is cleared as well as replaced: a rotation is not a
        path, and leaving the last one published would have the follower pursue
        it the instant the turn ends, from a heading it was never planned for.
        """
        target = self._turn.start(pose[3], float(delta), now)
        msg = Float32()
        msg.data = float(target)
        self._yaw_goal_pub.publish(msg)
        self._executor.reset()
        self._publish_path(self._path_pub, [], pose)
        self._publish_path(self._full_path_pub, [], pose)
        # Released, because the follower has to be free to fly the rotation --
        # the hold is "do not move", and this manoeuvre is the exception it is
        # allowed to make.
        self._finish_decision()
        self._phase = "turning"
        self._turns += 1
        action = (result.metadata.get("action") if result else None) or "curve"
        self.get_logger().info(
            "turn #%d: %+.1f deg to heading %.1f deg (%s)"
            % (self._turns, np.degrees(delta), np.degrees(target), action))

    # ── holding still while thinking ─────────────────────────────────
    def _set_hold(self, held):
        """Tell the follower to stop, or to fly again. Edge-triggered."""
        if bool(held) == self._held:
            return
        self._held = bool(held)
        msg = Bool()
        msg.data = self._held
        self._hold_pub.publish(msg)
        if not self._held:
            self._hold_since = None
            self._settle_since = None

    def _settled(self, speed, yaw_rate, now):
        """Is the aircraft stopped enough to take the observation from?

        Not "has the hold been published" -- this airframe translates by tilting
        and coasts for the better part of a second after the command goes to
        zero. A frame taken during that coast is motion-blurred and, worse,
        taken from somewhere the aircraft is no longer going to be, which is the
        stale anchor all over again in miniature.

        Times out rather than waiting for ever: a drone being pushed by the
        world, or one whose odometry is noisy, must still get to ask.
        """
        if self._hold_since is None:
            self._hold_since = now
            self._settle_since = None
        if speed <= self._settle_speed and abs(yaw_rate) <= self._settle_yaw_rate:
            if self._settle_since is None:
                self._settle_since = now
            if now - self._settle_since >= self._settle_s:
                return True
        else:
            self._settle_since = None
        if now - self._hold_since >= self._settle_timeout_s:
            self.get_logger().warn(
                "never settled (%.2f m/s, %.2f rad/s after %.1f s); observing "
                "from a moving aircraft" % (speed, yaw_rate, self._settle_timeout_s),
                throttle_duration_sec=10.0)
            return True
        return False

    def _finish_decision(self):
        """The decision is made: let the aircraft fly again."""
        self._deciding = False
        self._set_hold(False)

    def _set_altitude_offset(self, offset):
        """Ask the follower to fly this far off its cruise altitude."""
        msg = Float32()
        msg.data = float(offset)
        self._alt_offset_pub.publish(msg)

    def _publish_info(self, result, instruction, now, pose):
        """Fold the step's timings into the FPS meters and publish a JSON status.

        One topic (`/simple_drone/n1/info`) carries everything the recorder
        overlays -- action, System-1/System-2 FPS and latency, the S2 pixel goal
        -- so the recorder subscribes to topics only and never touches the model.
        """
        md = result.metadata
        self._s1_fps.update(md.get("s1_ms"))
        self._s2_fps.update(md.get("s2_ms"))
        traj = result.trajectory
        traj_m = (_polyline_length(np.asarray(traj)[:, :2])
                  if traj is not None and len(traj) else 0.0)
        wp = md.get("waypoint_px")
        # Whether THIS decision came from System 1's continuous curve or from a
        # discrete action rendered as a short step, plus the running share. It
        # is the number the whole dual-system deployment is judged on, and
        # without it on screen the only way to know is to read the log.
        if not md.get("idle"):
            self._decisions += 1
            if md.get("from_curve"):
                self._curve_decisions += 1
        share = (100.0 * self._curve_decisions / self._decisions) if self._decisions else None
        info = {
            "instruction": instruction,
            "action": md.get("action"),
            "from_curve": bool(md.get("from_curve")),
            "idle": bool(md.get("idle")),
            "curve_share_pct": share,
            "s1_ms": md.get("s1_ms"),
            "s2_ms": md.get("s2_ms"),
            "s1_fps": self._s1_fps.fps,
            "s2_fps": self._s2_fps.fps,
            "pixel_goal": list(wp) if wp else None,
            "pixel_goal_frame": [self._cam_w, self._cam_h],
            # How many decisions ago System 2 chose it, and whether this is the
            # decision that chose it. The goal is a pixel in the frame System 2
            # saw; a consumer drawing it on the live frame has to know that.
            "pixel_goal_fresh": bool(md.get("waypoint_fresh")),
            "pixel_goal_age": md.get("waypoint_age_steps"),
            # Wall clock of THIS decision. Freshness is per decision, but the
            # recorder redraws at 10 fps and a decision now lasts seconds, so
            # without a timestamp one goal flagged fresh once is painted as a
            # solid live target for a hundred consecutive frames while the
            # aircraft flies away from the pose that produced it.
            "decision_time": now,
            "stop": bool(result.stop),
            # What was actually decided, in metres and degrees, so a recording
            # answers "is it flying curves?" without anyone reading a log. The
            # prediction's own length -- before the commitment clips it -- is
            # the number that says whether System 1 ran at all.
            "traj_m": traj_m,
            "traj_pts": 0 if traj is None else int(len(traj)),
            "turn_deg": (None if md.get("turn_delta_rad") is None
                         else float(np.degrees(md["turn_delta_rad"]))),
            # How long the aircraft stood still for this decision, and whether
            # it was standing still at all. A recording that shows a motionless
            # drone is either thinking or wedged, and those are very different.
            "think_s": (now - self._hold_since) if self._hold_since else 0.0,
            "held": bool(self._held),
            "blocked": bool(self._blocked),
            "phase": self._phase,
            "commits": self._commits,
            "turns": self._turns,
            "escapes": self._escapes,
            "altitude_m": float(pose[2]),
            "yaw_deg": float(np.degrees(pose[3])),
        }
        self._info = info
        self._emit_info()

        if now - self._last_fps_log_s > 5.0:
            self._last_fps_log_s = now
            s1, s2 = self._s1_fps.fps, self._s2_fps.fps
            self.get_logger().info(
                "N1 FPS  System1=%s  System2=%s  (action=%s)"
                % (("%.1f Hz" % s1) if s1 else "--",
                   ("%.1f Hz" % s2) if s2 else "--", md.get("action")))

    def _emit_info(self):
        msg = String()
        msg.data = json.dumps(self._info)
        self._info_pub.publish(msg)

    def _status_tick(self):
        """Republish the live status. Runs while `_tick` is blocked on the model."""
        if not rclpy.ok():
            return
        with self._lock:
            pose = self._pose
        if pose is None:
            return
        self._publish_status(time.time(), pose)

    def _publish_status(self, now, pose):
        """Republish the last decision with the live phase, a few times a second.

        Everything here is what changed *since* the decision -- where the
        aircraft is, what it is doing, how long it has been thinking. The
        decision's own fields are left exactly as they were reported, because
        rewriting them would make a stale answer look like a fresh one.
        """
        if not self._info:
            return
        self._last_status_s = now
        self._info["phase"] = self._phase
        self._info["held"] = bool(self._held)
        self._info["blocked"] = bool(self._blocked)
        self._info["think_s"] = (now - self._hold_since) if self._hold_since else 0.0
        self._info["altitude_m"] = float(pose[2])
        self._info["yaw_deg"] = float(np.degrees(pose[3]))
        self._info["commits"] = self._commits
        self._info["turns"] = self._turns
        self._info["escapes"] = self._escapes
        self._emit_info()

    def _publish_path(self, publisher, world_xy, pose):
        msg = Path()
        msg.header.frame_id = self._world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for point in np.asarray(world_xy, dtype=float).reshape(-1, 2):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(point[0])
            ps.pose.position.y = float(point[1])
            ps.pose.position.z = float(pose[2])  # hold the current altitude
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = N1PolicyNode()

    def _finish(_signum, _frame):
        # Leave at once. This node spends most of its life blocked in an HTTP
        # call to the model server -- seconds at a time, since System 2 is a 7B
        # VLM -- and a signal delivered mid-request is not seen until that
        # request returns. Waiting for it means the teardown's grace expires and
        # the node is SIGKILLed instead, which the launch file then reports as
        # "this node died". Nothing here needs unwinding: the aircraft belongs
        # to the follower, and there is no file to flush.
        os._exit(0)

    signal.signal(signal.SIGTERM, _finish)
    signal.signal(signal.SIGINT, _finish)
    # Multi-threaded, because `_tick` spends seconds at a time inside one HTTP
    # call and a single-threaded spin would freeze the odometry and the status
    # for the whole of it. The callback groups above are what make that safe.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        # An orderly stop, not a fault. Letting any of these escape exits 1,
        # which the launch file reads as "this node died" and turns into an
        # ERROR and an emergency shutdown of its siblings -- so a clean Ctrl-C
        # produces a log that looks like a crash, and the one alarm that exists
        # for a node dying at import becomes noise.
        #
        # RCLError belongs here because rclpy does NOT always raise the tidy
        # ExternalShutdownException: a shutdown that lands between spin
        # iterations surfaces as `failed to initialize wait set: the given
        # context is not valid`, and one that lands inside a callback as
        # `failed to publish: publisher's context is invalid`. Both are the
        # same event.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

