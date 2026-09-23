"""target_approach_node — the mission found the target; fly to it and land beside it.

The last leg of the scene-graph mission, and **the only node in this stack that
ever takes the aircraft off FALCON**. Until ``/target_seen`` latches True it is
completely inert: it publishes nothing, subscribes to nothing but the two
latched target topics, runs no timer and holds no camera. Today's exploration
flight is therefore byte-identical with this node loaded and never triggered —
that is the point, and every default below is chosen to keep it so.

What happens once the target IS seen
------------------------------------
1. **Arm.** Read the latched ``/target_seen/info`` JSON for the *class* the
   watcher matched (``matched_class``, falling back to ``target``); that is
   what the servo locks onto, not the mission's target word — the LLM matcher
   may well have mapped "bed" onto the vocabulary prompt "hospital bed".
   Subscribe the RGB + depth cameras and their ``camera_info``, start the
   control timer and the detection worker. **Still passive**: FALCON keeps
   flying.
2. **Re-acquire, visually.** A worker thread POSTs one JPEG every
   ``detect_period_s`` to the same detection server the mapper uses, and
   :class:`TargetConfirmationGate` counts consecutive frames carrying the
   class. Between POSTs the box is carried by
   :mod:`sparx_agency.core.mapping.tracking` at the full control rate —
   detect-once / track-many, which is why the servo gets a fresh box at 10 Hz
   from a 2 Hz detector. The server's vocabulary is **never** changed:
   ``/set_classes`` is shared state, and re-writing it would silently re-aim
   the object mapper that is still running.
3. **Take the aircraft.** Only when
   :class:`VisualApproachStateMachine` first says ``drive_cmd_vel`` (i.e. the
   target is confirmed *and* the tracker holds a box) does the node publish
   ``True`` on the latched ``/scene_graph/external_ctrl`` that mutes the
   FALCON b-spline follower. That flag is a **lease, not a switch**: the
   follower expires it after its ``~external_ctrl_timeout_s`` -- 5 s, whose
   default lives in ``falcon_sjtu/adapter/launch/bspline_follower.launch``
   -- unless the owner keeps renewing, so this node republishes True every
   ``external_ctrl_period_s`` (1 s) for as long as it flies. The first thing
   it then sends is a zero Twist (``ACQUIRE_STOP``), so the one tick where
   both publishers may overlap is a tick where our command is a stop.
4. **Servo in.** :class:`VisualServoController` in ``holonomic`` mode (this
   airframe takes a full body Twist, so yaw + forward + crab at once) drives
   toward the box, ramping forward on the *metric* range read off the depth
   camera.
5. **Land beside it.** ``land_range_m`` (default 1.0 m) is deliberately
   larger than the servo's ``target_range_m`` hover standoff, so the machine
   commits to its terminal ``LAND`` on the way in rather than settling into a
   hover — "land next to it", not on top of it. It takes
   ``land_confirm_ticks`` consecutive in-range measurements, so one depth
   glitch cannot land the aircraft. Then: stop the Twist, burst
   ``std_msgs/Empty`` on ``/simple_drone/land`` (a lone Empty is *silently
   ignored* from the wrong state, so it is repeated until ``/simple_drone/state``
   reports LANDED), and only then release the mute.

Giving up, and never leaving the follower muted
-----------------------------------------------
``approach_timeout_s`` (120 s from arming) and the state machine's own
recovery give-up (``reset_acquisition``, raised when the track is lost past
``recover_timeout_s``) both end the run the same way: stop commanding, release
the mute, log a WARNING, publish the terminal status, and hand FALCON back its
aircraft. Leaving the follower muted would strand the aircraft with **no**
``cmd_vel`` publisher at all, holding whatever twist the plugin last got, so
the release runs from ``destroy_node`` and from the ``finally`` in
:func:`main` too.

A signal is the case worth stating precisely, because the obvious teardown
does *not* work: ``rclpy.init`` handles SIGINT **and** SIGTERM, and both shut
the context down before ``spin()`` returns, so publishing the ``False`` from
``destroy_node`` raises ``publisher's context is invalid`` and sends nothing
(measured, both signals). ``stop_scene_graph.sh`` stops the host nodes with a
plain ``kill`` — SIGTERM — so that is the *ordinary* operator stop, not an edge
case. :func:`...target_approach_release.emergency_release` therefore re-sends
the release from a throwaway participant on a fresh context, which the signal
did not touch.

Only a hard ``kill -9`` can run none of that, and there the lease is the
backstop: the refresh stops with us and the follower takes the aircraft back
once ``~external_ctrl_timeout_s`` lapses.

The mute crosses the ROS1/ROS2 boundary, so its QoS is not a free choice —
``config/bridge.yaml`` declares this topic ``reliable`` + ``transient_local``
+ depth 1, and the publisher below matches it exactly. A volatile publisher
against that bridge entry delivers nothing at all until its next change,
which for a flag that moves twice a mission is the whole flight.

Releasing after a successful landing is safe for the same reason it is
required: the plugin will not fly a LANDED aircraft without a ``/takeoff``,
which nothing in this stack sends.

Subscribes
----------
``/target_seen``          (std_msgs/Bool, latched) — the arm trigger.
``/target_seen/info``     (std_msgs/String JSON, latched) — which class to lock.
``/simple_drone/front/image_raw``            (sensor_msgs/Image) — after arming.
``/simple_drone/front_depth/depth/image_raw``(sensor_msgs/Image) — after arming.
``/simple_drone/front/camera_info`` + the depth one — after arming.
``/simple_drone/state``   (std_msgs/Int8) — to know the land actually took.

Publishes
---------
``/scene_graph/external_ctrl`` (std_msgs/Bool, latched) — mutes the follower.
``/simple_drone/cmd_vel``      (geometry_msgs/Twist) — only while engaged.
``/simple_drone/land``         (std_msgs/Empty) — the terminal land burst.
``/target_approach/info``      (std_msgs/String JSON, latched) — the status.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.\
target_approach_node --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import json
import threading
from collections import deque
from typing import Optional

import requests
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Empty, Int8, String

from sparx_agency.core.common.types.perception import Intrinsics
from sparx_agency.core.control.velocity_servo import BodyTwistCommand
from sparx_agency.core.planning.visual_servo import (
    ACQUIRE_STOP, LAND, RECOVER, SEARCH, TargetConfirmationGate,
    VisualServoRequest, select_overlapping_target_detection)
from sparx_agency.robots.SJTU.adapters.plant_config import body_velocity_limits
from sparx_agency.robots.SJTU.adapters.velocity_command import (fill_twist,
                                                                twist_fields)
# The package's single copies of the three message decoders, imported rather
# than re-typed: `np.frombuffer` on a `sensor_msgs/Image` is exactly the kind
# of detail (row stride in BYTES, 32FC1 metres vs 16UC1 millimetres, rgb8 vs
# bgr8) that drifts silently once it exists three times. Importing a node
# module runs no ROS code -- it only defines a class.
from sparx_agency.tasks.mapping.scene_graph.ros2.detector_client_node import (
    _image_msg_to_bgr as image_msg_to_bgr)
from sparx_agency.tasks.mapping.scene_graph.ros2.object_mapper_node import (
    _depth_metres as depth_metres, _nearest as nearest_by_stamp,
    _stamp_to_sec as stamp_to_sec)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import latched_qos
from sparx_agency.tasks.mapping.scene_graph.ros2.target_approach_config import (
    PARAM_DEFAULTS, build_fsm, build_gate_config, build_recovery, build_servo,
    build_tracker)
from sparx_agency.tasks.mapping.scene_graph.ros2.target_approach_release import (
    emergency_release)
from sparx_agency.tasks.mapping.scene_graph.ros2.target_approach_payloads import (
    approach_info_payload, bbox_range_m, detections_to_core,
    target_info_from_json)
from sparx_agency.tasks.mapping.scene_graph.serve.contract import encode_frame

DEPTH_DEQUE_LEN = 30    # ~2 s of the 15 Hz front depth camera
FRAME_DEQUE_LEN = 30    # RGB ring, so a detection can be seeded on its own frame
STATE_LANDED = 0
"""``/simple_drone/state`` value for LANDED (see sjtu_internvla_n1 ensure_flying)."""


class TargetApproachNode(Node):
    """Idle until the target is seen; then lock on, fly in, and land beside it."""

    def __init__(self) -> None:
        super().__init__("target_approach")

        # Every knob, its default and the core objects it builds live in
        # target_approach_config, so the parameter list and the tuning it feeds
        # cannot drift apart the way a declare-here / read-there pair does.
        for name, default in PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)
        cfg = {name: self.get_parameter(name).value for name in PARAM_DEFAULTS}

        self._enabled = bool(cfg["enabled"])
        self._rgb_topic = str(cfg["rgb_topic"])
        self._depth_topic = str(cfg["depth_topic"])
        self._rgb_info_topic = str(cfg["rgb_info_topic"])
        self._depth_info_topic = str(cfg["depth_info_topic"])
        self._state_topic = str(cfg["state_topic"])
        self._server_url = str(cfg["server_url"]).rstrip("/")
        self._server_timeout_s = float(cfg["server_timeout_s"])
        self._detect_period_s = float(cfg["detect_period_s"])
        self._rate_hz = max(1.0, float(cfg["approach_rate_hz"]))
        self._timeout_s = float(cfg["approach_timeout_s"])
        self._ctrl_period_s = float(cfg["external_ctrl_period_s"])
        self._min_depth_m = float(cfg["min_depth_m"])
        self._max_depth_m = float(cfg["max_depth_m"])
        self._max_gap_s = float(cfg["max_stamp_gap_s"])
        self._confirm_iou = float(cfg["confirm_iou"])
        self._soft_min_score = float(cfg["soft_confirm_min_score"])
        self._land_repeat_period_s = float(cfg["land_repeat_period_s"])
        self._land_settle_s = float(cfg["land_settle_s"])

        self._tracker = build_tracker(cfg)
        self._servo = build_servo(cfg)
        self._fsm = build_fsm(cfg)
        self._recovery = build_recovery(cfg)
        self._gate_cfg = build_gate_config(cfg)
        self._gate: Optional[TargetConfirmationGate] = None
        # The airframe's own saturations, from config/airframe.yaml -- the last
        # word before the wire, under the servo's much tighter caps.
        self._limits = body_velocity_limits()

        # State
        self._lock = threading.Lock()
        self._info = None                 # TargetInfo once /target_seen/info lands
        self._seen = False                # /target_seen has latched True
        self._armed = False
        self._engaged = False             # we have muted the follower
        self._finished = False
        self._reason = "waiting for /target_seen"
        self._t0 = None                   # arm time (s)
        self._prev_tick_t = None
        self._ticks = 0
        self._frames = deque(maxlen=FRAME_DEQUE_LEN)   # (stamp, bgr)
        self._depths = deque(maxlen=DEPTH_DEQUE_LEN)   # (stamp, HxW float32 m)
        self._rgb_k = None                # (fx, fy, cx, cy)
        self._depth_k = None
        self._intr: Optional[Intrinsics] = None
        self._confirmed = False
        self._streak = 0
        self._last_range = None
        self._flight_state = None
        self._landing = False
        self._land_sent_t = None
        self._land_begin_t = None
        self._land_bursts = 0
        self._n = dict(posts=0, dets=0, conn_errors=0, bad_replies=0)
        self._stop = threading.Event()
        self._det_thread = None
        self._ctrl_timer = None
        self._mute_timer = None
        self._sess = requests.Session()

        self._latched = latched_qos()
        # depth 1, not the shared sensor_qos() default of 5: this node holds
        # only the newest frame and drops the rest, so a queue would only add
        # latency to the servo loop.
        self._sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                      durability=DurabilityPolicy.VOLATILE,
                                      history=HistoryPolicy.KEEP_LAST, depth=1)

        # Publishers exist from startup but stay SILENT until the target is
        # seen. A latched publisher that never publishes puts nothing on the
        # wire, so the running mission cannot tell this node is loaded.
        self._pub_cmd = self.create_publisher(Twist, str(cfg["cmd_topic"]), 10)
        self._pub_land = self.create_publisher(Empty, str(cfg["land_topic"]), 10)
        # Held rather than read inline: the mute topic is a parameter, and the
        # emergency release below has to publish on the one this node actually
        # took control with. Releasing the module default while the operator
        # had overridden it would report success and leave the follower muted.
        self._ctrl_topic = str(cfg["external_ctrl_topic"])
        self._pub_ctrl = self.create_publisher(Bool, self._ctrl_topic,
                                               self._latched)
        self._pub_status = self.create_publisher(
            String, str(cfg["status_topic"]), self._latched)

        self.create_subscription(String, str(cfg["info_topic"]), self._info_cb,
                                 self._latched)
        self.create_subscription(Bool, str(cfg["target_seen_topic"]),
                                 self._seen_cb, self._latched)
        self.create_timer(10.0, self._heartbeat)

        self.get_logger().info(
            "target_approach ARMED-ON-DEMAND and silent until %s: rate=%.1fHz "
            "detect=%.2fs server=%s land_range=%.2fm timeout=%.0fs enabled=%s"
            % (str(cfg["target_seen_topic"]), self._rate_hz,
               self._detect_period_s, self._server_url,
               float(cfg["land_range_m"]), self._timeout_s, self._enabled))

    # ── the arm trigger ──────────────────────────────────────────────
    def _info_cb(self, msg: String) -> None:
        """Latch which detector class to lock onto (may arrive either order)."""
        try:
            info = target_info_from_json(msg.data)
        except ValueError as exc:
            self.get_logger().error(
                "cannot read /target_seen/info, so there is no class to lock "
                "onto and the approach will NOT engage: %s" % (exc,))
            return
        with self._lock:
            self._info = info
        self.get_logger().info(
            "target info: target=%r matched_class=%r -> locking onto %r "
            "(object_id=%d at %.2f,%.2f, %d confirmations)"
            % (info.target, info.matched_class, info.lock_class,
               info.object_id, info.xy[0], info.xy[1], info.count))
        # Both topics are latched with depth 1 and arrive in no guaranteed
        # order, so whichever lands second is the one that arms. Without this
        # an info that follows the Bool would leave the node idle forever.
        if self._seen and not self._armed and not self._finished:
            self._arm()

    def _seen_cb(self, msg: Bool) -> None:
        if not msg.data or self._armed or self._finished:
            return
        self._seen = True
        self._arm()

    def _arm(self) -> None:
        """Wake up: subscribe the cameras, start the loops. Still passive.

        Reached from either latched topic, so the ``enabled`` switch is
        enforced HERE rather than at the Bool: gating it in ``_seen_cb`` alone
        would let an ``/target_seen/info`` that lands second arm a node the
        operator had switched off.
        """
        if not self._enabled:
            self.get_logger().warning(
                "the target was found but ~enabled is false -- FALCON keeps "
                "the aircraft and no approach is flown", once=True)
            return
        with self._lock:
            info = self._info
        if info is None:
            self.get_logger().warning(
                "/target_seen is True but /target_seen/info has not arrived "
                "yet; holding -- there is no class to lock onto until it does, "
                "and arming happens on whichever of the two lands second")
            return
        self._armed = True
        # NOT self._now(): under use_sim_time the ROS clock reads 0 until the
        # first /clock arrives, and /target_seen is latched, so arming happens
        # in the same breath as start-up -- before any /clock has been seen.
        # A _t0 of 0 against the next tick's real sim time made elapsed the
        # whole age of the simulation, and the approach "timed out after 240 s"
        # 0.5 s after arming, with ticks=0, having never commanded anything.
        # Latch it lazily on the first tick where the clock is actually
        # running, and only start counting from there.
        self._t0 = None
        self._gate = TargetConfirmationGate(info.lock_class, self._gate_cfg)

        self.create_subscription(Image, self._rgb_topic, self._rgb_cb,
                                 self._sensor_qos)
        self.create_subscription(Image, self._depth_topic, self._depth_cb,
                                 self._sensor_qos)
        self.create_subscription(CameraInfo, self._rgb_info_topic,
                                 self._rgb_info_cb, self._sensor_qos)
        self.create_subscription(CameraInfo, self._depth_info_topic,
                                 self._depth_info_cb, self._sensor_qos)
        self.create_subscription(Int8, self._state_topic, self._state_cb,
                                 self._sensor_qos)
        self._ctrl_timer = self.create_timer(1.0 / self._rate_hz, self._tick)
        self._det_thread = threading.Thread(target=self._detect_loop,
                                            name="target_approach_detect",
                                            daemon=True)
        self._det_thread.start()

        bar = "=" * 64
        self.get_logger().info(bar)
        self.get_logger().info(
            "  TARGET SEEN -- approaching %r (locking onto class %r)"
            % (info.target, info.lock_class))
        self.get_logger().info(
            "  FALCON still owns the aircraft; the visual take-over waits "
            "until the detector re-confirms the object in view")
        self.get_logger().info(bar)
        self._set_reason("armed; re-acquiring the target visually")
        self._publish_status()

    # ── sensor callbacks ─────────────────────────────────────────────
    def _rgb_cb(self, msg: Image) -> None:
        try:
            bgr = image_msg_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().error("bad RGB frame: %s" % (exc,),
                                    throttle_duration_sec=10.0)
            return
        with self._lock:
            self._frames.append((stamp_to_sec(msg.header.stamp), bgr))

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth = depth_metres(msg)
        except ValueError as exc:
            self.get_logger().error("depth image rejected: %s" % (exc,),
                                    throttle_duration_sec=10.0)
            return
        with self._lock:
            self._depths.append((stamp_to_sec(msg.header.stamp), depth))

    def _rgb_info_cb(self, msg: CameraInfo) -> None:
        if self._rgb_k is not None:
            return
        self._rgb_k = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]),
                       float(msg.k[5]))
        self._intr = Intrinsics(width=int(msg.width), height=int(msg.height),
                                fx=self._rgb_k[0], fy=self._rgb_k[1],
                                cx=self._rgb_k[2], cy=self._rgb_k[3])

    def _depth_info_cb(self, msg: CameraInfo) -> None:
        if self._depth_k is None:
            self._depth_k = (float(msg.k[0]), float(msg.k[4]),
                             float(msg.k[2]), float(msg.k[5]))

    def _state_cb(self, msg: Int8) -> None:
        self._flight_state = int(msg.data)

    # ── detection worker (a POST must never stall the control loop) ──
    def _detect_loop(self) -> None:
        """Post one frame every ``detect_period_s`` and feed the gate + tracker.

        Runs in its own thread because ``requests.post`` blocks for as long as
        the shared detection server takes (tens to hundreds of ms), and the
        control loop must keep flying the aircraft meanwhile. Everything it
        touches afterwards is under ``self._lock``, which the control tick also
        holds while stepping the tracker.
        """
        while not self._stop.wait(self._detect_period_s):
            if self._finished or self._intr is None:
                continue
            with self._lock:
                frame = self._frames[-1] if self._frames else None
            if frame is None:
                continue
            stamp, bgr = frame
            try:
                self._n["posts"] += 1
                reply = self._sess.post(
                    self._server_url + "/detect", data=encode_frame(bgr),
                    headers={"Content-Type": "image/jpeg"},
                    timeout=self._server_timeout_s)
                reply.raise_for_status()
                items = reply.json().get("detections", [])
            except (requests.RequestException, ValueError) as exc:
                self._n["conn_errors"] += 1
                if self._stop.is_set():
                    return
                self.get_logger().warning(
                    "POST /detect to %s failed: %s" % (self._server_url, exc),
                    throttle_duration_sec=10.0)
                continue
            if self._stop.is_set():
                # The node was torn down while the POST was in flight; the
                # logger and the tracker are on their way out.
                return
            try:
                dets = detections_to_core(items, bgr.shape[1], bgr.shape[0],
                                          self._intr.width, self._intr.height)
            except ValueError as exc:
                self._n["bad_replies"] += 1
                self.get_logger().error("malformed /detect reply: %s" % (exc,),
                                        throttle_duration_sec=10.0)
                continue
            self._n["dets"] += len(dets)
            self._on_detections(dets, stamp, bgr)

    def _on_detections(self, dets, stamp: float, bgr) -> None:
        """Advance the confirmation gate and (re)seed the tracker.

        Mirrors the FALCON reference node: a HARD match acquires the lock; once
        tracking, a *weak* detection sitting on the tracked box (IoU >=
        ``confirm_iou``) also re-seeds it, so a genuinely tracked object is not
        dropped the moment its score dips — while pure background drift, which
        has no overlapping detection, still times out.
        """
        with self._lock:
            if self._gate is None:
                return
            state = self._gate.update(dets)
            self._confirmed = state.confirmed
            self._streak = state.streak
            seed = state.best
            if (seed is None and self._tracker.has_target
                    and self._confirm_iou > 0.0
                    and self._tracker.last_track is not None):
                seed = select_overlapping_target_detection(
                    dets, self._gate.target, self._tracker.last_track.bbox_xyxy,
                    self._confirm_iou, self._soft_min_score)
            if seed is None:
                return
            # A non-propagating (detector-only) tracker has no state to coast
            # on, so it is always re-fed; the tracked path is re-seeded too, to
            # bound optical-flow drift to the detector's inter-arrival time.
            if self._tracker.has_target or state.confirmed:
                self._tracker.on_detection(bgr, seed, stamp)

    # ── control loop ─────────────────────────────────────────────────
    def _tick(self) -> None:
        """One control tick, wrapped so a bad frame cannot kill the timer."""
        try:
            self._step()
        except Exception as exc:            # noqa: BLE001 -- resilience is the point
            self.get_logger().error(
                "approach tick failed (%s: %s) -- holding"
                % (type(exc).__name__, exc), throttle_duration_sec=2.0)
            if self._engaged and not self._landing:
                self._publish_cmd(0.0, 0.0, 0.0)

    def _step(self) -> None:
        now = self._now()
        dt = 0.0 if self._prev_tick_t is None else max(0.0, now - self._prev_tick_t)
        self._prev_tick_t = now
        self._ticks += 1

        if self._finished:
            return
        if self._landing:                   # terminal: never servo again
            self._drive_land(now)
            return
        if self._t0 is None:
            if now <= 0.0:                  # /clock still absent; do not count
                return
            self._t0 = now
            self.get_logger().info(
                "approach clock started at t=%.1fs (timeout %.0fs)"
                % (now, self._timeout_s))
        if now - self._t0 > self._timeout_s:
            self._give_up("approach timed out after %.0f s without reaching "
                          "the target" % self._timeout_s)
            return
        if self._intr is None:
            self.get_logger().warning(
                "no RGB camera_info yet on %s -- the approach cannot start "
                "without the image geometry" % self._rgb_info_topic,
                throttle_duration_sec=5.0)
            return

        with self._lock:
            frame = self._frames[-1] if self._frames else None
            confirmed = self._confirmed
            track = (self._tracker.on_frame(frame[1], frame[0])
                     if (frame is not None and self._tracker.has_target)
                     else None)
            last_track = self._tracker.last_track
            depth = (nearest_by_stamp(self._depths, frame[0])
                     if frame is not None else None)
        track_valid = bool(track is not None and track.valid)

        # A depth frame is only a measurement of THIS RGB frame while the two
        # stamps agree: nearest_by_stamp always returns something, and the
        # oldest entry in a stalled ring would otherwise be read as a live
        # range and could land the aircraft on a two-second-old wall.
        if depth is not None and abs(depth[0] - frame[0]) > self._max_gap_s:
            depth = None

        res = None
        if track_valid:
            rng = self._range_to(track, None if depth is None else depth[1])
            self._last_range = rng
            res = self._servo.step(VisualServoRequest(
                track=track, intrinsics=self._intr, range_m=rng, dt=dt))

        dec = self._fsm.update(confirmed=confirmed, track_valid=track_valid,
                               at_target=bool(res is not None and res.at_target),
                               dt=dt,
                               range_m=None if res is None else res.range_m)

        if dec.mode == LAND:
            self._begin_land(now)
            return
        if dec.reset_acquisition:
            self._give_up("lost the %s and could not re-acquire it within "
                          "the recovery window" % self._gate.target)
            return
        if not dec.drive_cmd_vel:           # SEARCH: FALCON still flies
            self._set_reason("searching: confirmed=%s streak=%d tracking=%s"
                             % (confirmed, self._streak, track_valid))
            return

        # We own the aircraft from here. Mute the follower FIRST; the very
        # first command we then send is the ACQUIRE_STOP zero, so the single
        # tick where both publishers may overlap is a tick where ours is a stop.
        self._take_control()
        if dec.mode == ACQUIRE_STOP:
            self._publish_cmd(0.0, 0.0, 0.0)
            self._set_reason("settling in place before the visual approach")
        elif dec.mode == RECOVER or res is None:
            rec = self._recovery.command(last_track, dec.lost_for_s,
                                         self._intr.width, self._intr.height)
            self._publish_cmd(rec.command.x, rec.command.y,
                              rec.command.yaw_rate)
            self._set_reason("lost the box; re-searching (%s, %.1f s)"
                             % (rec.phase, dec.lost_for_s))
        else:
            self._publish_cmd(res.command.x, res.command.y,
                              res.command.yaw_rate)
            self._set_reason("closing on the %s: range=%s x_off=%+.2f"
                             % (self._gate.target,
                                "n/a" if res.range_m is None
                                else "%.2f m" % res.range_m, res.x_offset))
        if self._ticks % max(1, int(self._rate_hz)) == 0:
            self._publish_status(state=dec.mode)

    def _range_to(self, track, depth) -> Optional[float]:
        """Metric range (m) to the tracked box, or None (no depth, no measure)."""
        if depth is None or self._rgb_k is None or self._depth_k is None:
            self.get_logger().warning(
                "no metric range available (depth=%s rgb_info=%s depth_info=%s)"
                " -- the terminal LAND cannot fire without one"
                % (depth is not None, self._rgb_k is not None,
                   self._depth_k is not None),
                throttle_duration_sec=5.0)
            return None
        return bbox_range_m(depth, track.bbox_xyxy, self._rgb_k, self._depth_k,
                            min_depth_m=self._min_depth_m,
                            max_depth_m=self._max_depth_m)

    # ── owning /cmd_vel ──────────────────────────────────────────────
    def _take_control(self) -> None:
        """Mute the FALCON follower, and keep it muted (idempotent)."""
        if self._engaged:
            return
        self._engaged = True
        self._pub_ctrl.publish(Bool(data=True))
        # The mute is a LEASE: the follower expires it after its
        # ~external_ctrl_timeout_s -- 5 s, defaulted in
        # falcon_sjtu/adapter/launch/bspline_follower.launch -- unless it keeps
        # being renewed, so a dead claimant cannot strand the aircraft muted.
        # The latch alone is therefore not enough -- renew it for as long as
        # we fly.
        self._mute_timer = self.create_timer(
            self._ctrl_period_s, lambda: self._pub_ctrl.publish(Bool(data=True)))
        self.get_logger().warning(
            "TAKING THE AIRCRAFT: muting the FALCON follower on %s and flying "
            "the visual approach onto the %s"
            % (self._ctrl_topic, self._gate.target))

    def _release(self) -> None:
        """Hand the aircraft back. Safe to call any number of times, from anywhere.

        Only ever publishes ``False`` if we actually took control, so a node
        that idled through a whole flight never touches the follower's mute.
        """
        if self._mute_timer is not None:
            self._mute_timer.cancel()
            self._mute_timer = None
        if not self._engaged:
            return
        self._engaged = False
        try:
            self._pub_ctrl.publish(Bool(data=False))
        except Exception as exc:            # noqa: BLE001 -- teardown must not raise
            # The measured case, not a hypothetical one: rclpy shuts the
            # context down on SIGINT *and* SIGTERM before spin() returns, so
            # this publish raises "publisher's context is invalid" on every
            # signal teardown -- and stop_scene_graph.sh stops the host nodes
            # with a plain kill, i.e. SIGTERM. Falling back to the follower's
            # lease here would mean the most ordinary way this mission ends
            # leaves the aircraft flying its last approach twist until the
            # lease lapses. So say it again from a context the signal did not
            # touch; the lease is only the backstop for a hard kill, which
            # cannot run any of this.
            print("[target_approach] the node's own mute release failed (%s: "
                  "%s); re-publishing it from a fresh context"
                  % (type(exc).__name__, exc))
            if emergency_release(self._ctrl_topic):
                print("[target_approach] emergency mute release delivered on "
                      "%s -- FALCON owns the aircraft again"
                      % self._ctrl_topic)
            else:
                print("[target_approach] emergency mute release found no "
                      "subscriber on %s; the follower's staleness lease must "
                      "lapse it" % self._ctrl_topic)
            return
        self.get_logger().warning("released the follower mute on %s -- FALCON "
                                  "owns the aircraft again" % self._ctrl_topic)

    def _publish_cmd(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Clamp to the airframe's saturations and put one Twist on the wire.

        The clamp is the platform's, from ``config/airframe.yaml``, and it
        scales the horizontal pair together so a saturated command still flies
        the direction it was aimed.
        """
        fields = twist_fields(
            BodyTwistCommand(vx=float(vx), vy=float(vy), vz=0.0,
                             yaw_rate=float(yaw_rate)), self._limits)
        self._pub_cmd.publish(fill_twist(Twist(), fields))

    # ── terminal land ────────────────────────────────────────────────
    def _begin_land(self, now: float) -> None:
        """Commit to the land sequence (idempotent, latched, never reversed)."""
        if not self._landing:
            self._landing = True
            self._land_begin_t = now
            bar = "=" * 64
            self.get_logger().info(bar)
            self.get_logger().info(
                "  REACHED THE %s -- range %s <= land_range; landing beside it"
                % (self._gate.target.upper(),
                   "n/a" if self._last_range is None
                   else "%.2f m" % self._last_range))
            self.get_logger().info(bar)
            self._set_reason("reached the %s at %s; landing"
                             % (self._gate.target,
                                "n/a" if self._last_range is None
                                else "%.2f m" % self._last_range))
            self._publish_status(state=LAND)
        self._drive_land(now)

    def _drive_land(self, now: float) -> None:
        """Stop the aircraft and land it, then release the mute.

        The plugin holds the last twist it was given, so going quiet is not
        stopping: a zero Twist is published every tick until the aircraft is
        down. ``/simple_drone/land`` is silently ignored from any state but
        FLYING, so the Empty is repeated rather than fired once and trusted.
        """
        if self._engaged:
            self._publish_cmd(0.0, 0.0, 0.0)
        landed = self._flight_state == STATE_LANDED
        if not landed:
            if (self._land_sent_t is None
                    or now - self._land_sent_t >= self._land_repeat_period_s):
                self._pub_land.publish(Empty())
                self._land_sent_t = now
                self._land_bursts += 1
            if now - self._land_begin_t < self._land_settle_s:
                return
            self.get_logger().error(
                "the aircraft never reported LANDED after %d land requests "
                "over %.0f s (state=%s); releasing anyway so nothing is left "
                "muted" % (self._land_bursts, self._land_settle_s,
                           self._flight_state))
            self._finish("landed=unconfirmed after %d land requests"
                         % self._land_bursts)
            return
        self.get_logger().info("aircraft is LANDED beside the %s -- mission "
                               "complete" % self._gate.target)
        self._finish("landed beside the %s" % self._gate.target)

    # ── endings ──────────────────────────────────────────────────────
    def _give_up(self, reason: str) -> None:
        """Stop, hand the aircraft back, and say why (WARNING, not an error)."""
        if self._engaged:
            self._publish_cmd(0.0, 0.0, 0.0)
        self.get_logger().warning(
            "GIVING UP on the visual approach: %s -- FALCON keeps flying the "
            "mission" % reason)
        self._finish(reason)

    def _finish(self, reason: str) -> None:
        """Terminal for every exit path: release, report, stop the loops."""
        if self._finished:
            return
        self._finished = True
        self._set_reason(reason)
        self._release()
        self._stop.set()
        if self._ctrl_timer is not None:
            self._ctrl_timer.cancel()
            self._ctrl_timer = None
        self._publish_status(state="DONE")

    # ── status + heartbeat ───────────────────────────────────────────
    def _set_reason(self, reason: str) -> None:
        self._reason = reason

    def _publish_status(self, state: str = SEARCH) -> None:
        info = self._info
        payload = approach_info_payload(
            stamp=self._now(),
            state="IDLE" if not self._armed else state,
            target="" if info is None else info.target,
            lock_class="" if info is None else info.lock_class,
            engaged=self._engaged, confirmed=self._confirmed,
            streak=self._streak,
            tracking=bool(self._tracker.is_locked),
            range_m=self._last_range, ticks=self._ticks,
            elapsed_s=0.0 if self._t0 is None else self._now() - self._t0,
            ended=self._finished, reason=self._reason)
        self._pub_status.publish(String(data=json.dumps(payload)))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _heartbeat(self) -> None:
        if not self._armed:
            self.get_logger().info(
                "target_approach hb  IDLE -- silent until /target_seen "
                "(nothing published, no camera held)")
            return
        if self._finished:
            # The terminal DONE status is already latched; republishing a live
            # state over it would erase how the mission ended.
            self.get_logger().info("target_approach hb  DONE -- %s"
                                   % self._reason)
            return
        self.get_logger().info(
            "target_approach hb  state=%s engaged=%s confirmed=%s streak=%d "
            "range=%s ticks=%d posts=%d dets=%d conn_err=%d bad=%d | %s"
            % (self._fsm.state, self._engaged, self._confirmed, self._streak,
               "n/a" if self._last_range is None else "%.2f" % self._last_range,
               self._ticks, self._n["posts"], self._n["dets"],
               self._n["conn_errors"], self._n["bad_replies"], self._reason))
        self._n = dict.fromkeys(self._n, 0)
        self._publish_status(state=self._fsm.state)

    # ── teardown ─────────────────────────────────────────────────────
    def destroy_node(self) -> None:
        """Never leave the follower muted, whatever killed us."""
        self._stop.set()
        if self._det_thread is not None and self._det_thread.is_alive():
            # Give the worker a moment to fall out of its wait so it cannot log
            # through a node that no longer exists. It is a daemon, so a POST
            # still in flight does not hold the process open.
            self._det_thread.join(timeout=1.0)
        try:
            self._release()
        finally:
            super().destroy_node()


def main() -> None:
    rclpy.init()
    node = TargetApproachNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT and SIGTERM. stop_scene_graph.sh sends SIGTERM, which rclpy
        # turns into ExternalShutdownException out of spin(); uncaught it
        # printed a traceback on every clean teardown.
        pass
    finally:
        # destroy_node releases the mute. A SIGINT mid-approach that left it
        # latched would strand the aircraft with NO cmd_vel publisher at all,
        # flying whatever twist the plugin last received.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
