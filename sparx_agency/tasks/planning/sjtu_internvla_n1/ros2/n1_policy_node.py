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
from math import atan2

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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, String

from sparx_agency.core.planning.vlas.common.plan_commit.executor import (
    CommitSpec,
    PlanCommitExecutor,
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
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


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

        self._instruction = pp.get("default_instruction", "explore the warehouse")
        self._inference_rate = float(pp.get("inference_rate_hz", 4.0))
        self._control_rate = float(cfg.get("follower", {}).get("control_rate_hz", 20.0))

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._pose = None  # (x, y, z, yaw)

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
            max_deviation_m=float(commit.get("max_deviation_m", 1.5)),
            min_period_s=float(commit.get("min_period_s", max(0.1, 1.0 / self._inference_rate))),
        ))
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

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        rgb_type = CompressedImage if self._rgb_compressed else Image
        self.create_subscription(rgb_type, self._rgb_topic, self._on_rgb, sensor_qos)
        if self._depth_enabled:
            self.create_subscription(Image, self._depth_topic, self._on_depth, sensor_qos)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(String, self._instruction_topic, self._on_instruction, 1)

        self._path_pub = self.create_publisher(Path, self._path_topic, latched)
        self._full_path_pub = self.create_publisher(Path, self._full_path_topic, latched)
        self._info_topic = topics.get("info", "/simple_drone/n1/info")
        self._info_pub = self.create_publisher(String, self._info_topic, latched)
        self._alt_offset_pub = self.create_publisher(
            Float32, topics.get("altitude_offset", "/simple_drone/n1/altitude_offset"), 1)

        # FPS meters: the model reports per-step System-1 / System-2 durations
        # (the trajectory-patched agent), which these turn into a smoothed rate.
        self._s1_fps = FpsMeter()
        self._s2_fps = FpsMeter()
        self._last_fps_log_s = 0.0
        self._cam_w = int(camera.get("width", 600))
        self._cam_h = int(camera.get("height", 600))

        self.create_timer(1.0 / max(1e-3, self._control_rate), self._tick)
        self.create_timer(1.5, self._init_once)  # one-shot server init after startup
        self._initialised = False

        self.get_logger().info(
            "n1_policy_node up: rgb=%s depth=%s odom=%s -> path=%s (server %s:%s), "
            "instruction=%r" % (self._rgb_topic, self._depth_topic if self._depth_enabled else "off",
                                self._odom_topic, self._path_topic,
                                server.get("host", "127.0.0.1"), server.get("port", 8087),
                                self._instruction))

    # ── setup ────────────────────────────────────────────────────────
    @staticmethod
    def _model_settings(camera, policy_params=None):
        """Server model settings: the SJTU camera model, and the S2 cadence.

        The FALCON/SJTU stack's single most expensive bug was a wrong camera
        intrinsic; the server projects its pixel goal with these, so they are
        passed explicitly rather than left at the 640x480 default.

        ``sys2_max_forward_step`` is the one knob that trades decision rate for
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
        with self._lock:
            self._pose = (p.position.x, p.position.y, p.position.z,
                          _yaw_from_quat(p.orientation))

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
            rgb = None if self._rgb is None else self._rgb.copy()
            depth = None if self._depth is None else self._depth.copy()
            instruction = self._instruction
        if pose is None:
            return

        now = time.time()

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
                self.get_logger().info("look-down: at %.2f m, sending the low frame" % pose[2])
            elif now - self._dip_since > self._dip_timeout_s:
                # It never got there -- blocked, or the follower is not tracking.
                # Send the frame anyway rather than wedging the episode; the
                # agent is blocked waiting for it.
                self._dip_state = "holding"
                self.get_logger().warn(
                    "look-down: never reached %.2f m (at %.2f m after %.1f s); "
                    "sending the frame from here"
                    % (self._cruise_alt - self._dip_m, pose[2], now - self._dip_since))
            else:
                return

        tick = self._executor.tick(pose[0], pose[1], now)
        # The agent is blocked waiting for the look-down frame, so the commit
        # executor does not get a vote on whether to step. A flag, not a `tick =
        # None` -- the tick object is read again further down and nulling it
        # simply moved the problem to a line that had no reason to expect it.
        forced = self._dip_state == "holding"
        if not forced and tick.replan_reason is None:
            return
        if rgb is None or now - self._last_inference_s < self._min_inference_interval:
            return

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
            self.get_logger().warn(
                "no usable answer from the model server (%s); keeping the "
                "current commitment" % (result.metadata.get("error"),),
                throttle_duration_sec=5.0)
            return

        self._publish_info(result, instruction, now)

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
            self._dip_state = "descending"
            self._dip_since = now
            self._set_altitude_offset(-self._dip_m)
            self.get_logger().info(
                "look-down requested: dipping %.2f m to %.2f m"
                % (self._dip_m, self._cruise_alt - self._dip_m))

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
            self.get_logger().info("N1 STOP (%s): holding" % (result.metadata.get("action"),))
            return

        self._executor.commit(result.trajectory[:, :2], (pose[0], pose[1], pose[3]), now)
        self._publish_path(self._path_pub, self._executor.plan.committed_xy, pose)
        self._publish_path(self._full_path_pub, self._executor.plan.world_xy, pose)
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

    def _set_altitude_offset(self, offset):
        """Ask the follower to fly this far off its cruise altitude."""
        msg = Float32()
        msg.data = float(offset)
        self._alt_offset_pub.publish(msg)

    def _publish_info(self, result, instruction, now):
        """Fold the step's timings into the FPS meters and publish a JSON status.

        One topic (`/simple_drone/n1/info`) carries everything the recorder
        overlays -- action, System-1/System-2 FPS and latency, the S2 pixel goal
        -- so the recorder subscribes to topics only and never touches the model.
        """
        md = result.metadata
        self._s1_fps.update(md.get("s1_ms"))
        self._s2_fps.update(md.get("s2_ms"))
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
        }
        msg = String()
        msg.data = json.dumps(info)
        self._info_pub.publish(msg)

        if now - self._last_fps_log_s > 5.0:
            self._last_fps_log_s = now
            s1, s2 = self._s1_fps.fps, self._s2_fps.fps
            self.get_logger().info(
                "N1 FPS  System1=%s  System2=%s  (action=%s)"
                % (("%.1f Hz" % s1) if s1 else "--",
                   ("%.1f Hz" % s2) if s2 else "--", md.get("action")))

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
    try:
        rclpy.spin(node)
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

