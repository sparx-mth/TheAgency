#!/usr/bin/env python3
"""lost_localization_node.py -- stop and recover when the drone's pose goes cold.

AprilTag localization only produces a pose while a tag is in view. When the last
tag leaves the frame ``/xtend/localization`` goes SILENT -- nothing republishes
the last pose, but every consumer in this stack caches it, so the drone keeps
flying on a pose from seconds ago and is, in the only sense that matters, lost.
Nothing today notices out loud: the follower quietly stops (and only if
``~use_pose_estimator`` happens to be on), and no one is told why.

This node watches the pose stream and, when it dies, takes the drone off the
follower and recovers it:

    /xtend/localization ─[ how old? ]─> PERSIST ──> HOLD ─> ladder ─> GIVE_UP
                                           │          │       │         │
                                     finish the     stop  back/climb   land
                                     move, or undo it      /sweep

All of the escalation logic -- the two thresholds, the persist prelude, the
ladder, the sweep, the exit debounce -- lives in the ROS-free, unit-tested
``sparx_agency.core.planning.lost_localization``. This node owns ONLY ROS and
platform concerns: rosparams -> params, the clock (the core is clock-free and is
handed an age and a ``dt``), the demo-mode claim/release handshake, observing what
the navigator was flying, and turning one ``ControlCommand`` into one
``geometry_msgs/Twist``.

  in   ~pose_topic (PoseStamped)          /xtend/localization -- the thing we watch
  in   ~bearing_topic (Float32)           /xtend/bearing -- tag-INDEPENDENT heading
  in   ~demo_mode_topic (String)          /xtend/demo_mode
  in   ~cmd_vel_topic (Twist)             what the navigator is flying -- see below
  out  ~cmd_vel_topic (Twist)             <drone_ns>/cmd_vel_raw (via the GO gate)
  out  ~demo_mode_request_topic (String)  /xtend/demo_mode_request
  out  ~status_topic (String, latched)    /recovery/status

Four things about this node are easy to get wrong and are deliberate:

* **It watches the STAMPED topic, and uses ARRIVAL time, not ``header.stamp``.**
  ``pose_adapter_node`` republishes the pose as a bare ``Pose``, throwing the
  header away, so ``<drone_ns>/gt_pose`` cannot answer "how old is this". And the
  stamp itself is the drone's camera clock, from a different machine
  (``mapping_sync_node`` keeps a whole drop-bucket for that skew), and can
  legitimately be 0. Arrival time also catches the union of failures we care
  about: tag loss, a bridge stall, a dead drone.
* **A stop is published as zeros, not as silence.** The XTEND bridge is a
  hold-style protocol: the last command runs until something replaces it. Going
  quiet would leave the drone flying the command it was already flying. (The GO
  gate's "closed = silence" rule is the opposite case -- there, silence is what
  lets a human fly.)
* **Releasing the mode is mandatory.** The follower goes passive while the
  latched demo_mode reads ``recovery``, so if we simply went quiet the follower
  would stay passive forever and the drone would never fly again -- it does not
  even re-request its own mode from that branch. Handing back therefore actively
  requests ``fly_straight`` (the rule, and the reason, are the same as
  ``object_approach_node._release``), and then holds the drone still with a short
  burst of zeros while that request round-trips through the demo manager.
* **It SUBSCRIBES to the topic it publishes on.** The persist prelude needs to
  know what the navigator was flying when the pose died, and the honest answer is
  whatever was last commanded on cmd_vel -- by the follower, by object_approach,
  by whoever had the drone. Reading the commands themselves rather than asking
  the follower keeps this node decoupled from which controller is running and
  from the demo_mode handshake's latency. The obvious hazard is hearing our OWN
  Twists back and persisting our own recovery, so observation stops the instant
  we claim the mode: from then on the context is a fact about the past, already
  latched by the core.

See the file footer for the full rosparam list.
"""
import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32, String

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.planning.lost_localization import (
    GIVE_UP,
    LostLocalizationParams,
    LostLocalizationRecovery,
    MotionContext,
)

#: The stop this node publishes when it hands control back.
_ZERO = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0)

#: Demo modes bridged over /xtend/demo_mode(_request).
MODE_RECOVERY = "recovery"        # we own cmd_vel; follower + object_approach passive
MODE_FINISH = "finish"            # terminal land: the manager does stop -> land -> disarm
MODE_RELEASE = "fly_straight"     # hand cmd_vel back to the follower


def _param_bool(name, default):
    """rosparam bool that also accepts the strings roslaunch passes."""
    v = rospy.get_param(name, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


#: Below this a commanded axis is noise, not a move (m/s and rad/s alike). Used to
#: tell a real command from the zeros the follower emits between yaw bursts -- a
#: stop expresses no intent, so it must never become the thing the persist finishes.
_CMD_EPS = 1e-3

#: |bearing| above this cannot be radians (2*pi + margin), so it is degrees.
#: Same rule as xtend_dome_main._bearing_to_rad, which is the only other consumer
#: that bothered to ask -- and it defaults to 'auto' for the same reason.
_BEARING_DEG_THRESHOLD = 2.0 * math.pi + 0.5


def _turn_dir(value):
    """'left'/'right' (or +1/-1) -> the core's +1/-1. Operator-facing."""
    s = str(value).strip().lower()
    if s in ("left", "ccw", "+1", "1"):
        return 1
    if s in ("right", "cw", "-1"):
        return -1
    raise ValueError("~turn_dir must be left|right (or +1/-1), got %r" % (value,))


class LostLocalizationNode(object):
    def __init__(self):
        rospy.init_node("lost_localization")
        G = rospy.get_param

        self.drone_ns = str(G("~drone_ns", ""))
        # Publish to cmd_vel_raw, NOT cmd_vel: recovery goes through the GO gate
        # like every other follower, so it cannot move a drone that is still in
        # someone's hands.
        self.cmd_vel_topic = str(G("~cmd_vel_topic", self.drone_ns + "/cmd_vel_raw"))
        self.pose_topic = str(G("~pose_topic", "/xtend/localization"))
        self.bearing_topic = str(G("~bearing_topic", "/xtend/bearing")).strip()
        # auto | rad | deg -- see _bearing_cb for why this is not just "rad".
        self._bearing_units = str(G("~bearing_units", "auto")).strip().lower()
        if self._bearing_units not in ("auto", "rad", "deg"):
            raise ValueError("~bearing_units must be auto|rad|deg, got %r"
                             % (self._bearing_units,))
        self._bearing_is_deg = (self._bearing_units == "deg")
        # The GO gate's latched status. Recovery must not run while the gate is
        # shut: the gate drops our Twists, so the ladder would "run" without the
        # drone ever moving, burn through every rung against a drone a human is
        # hand-flying, and then land it -- and the land is NOT gated (it goes out
        # as a demo-mode request, and the manager drives cmd_nav directly). Set ''
        # to ignore the gate.
        self.go_status_topic = str(G("~go_status_topic", "/mission/go_status")).strip()
        self.demo_mode_topic = str(G("~demo_mode_topic", "/xtend/demo_mode"))
        self.demo_req_topic = str(G("~demo_mode_request_topic",
                                    "/xtend/demo_mode_request"))
        self.status_topic = str(G("~status_topic", "/recovery/status"))

        self.rate_hz = float(G("~rate_hz", 10.0))
        # How long a navigator command stays a valid answer to "what were we
        # doing". Comfortably longer than the follower's settle between yaw
        # bursts -- that stationary gap is still part of a turn and must persist
        # as one -- but short enough that a move from a minute ago, with a whole
        # hover since, is treated as no answer at all rather than a stale one.
        self.context_max_age_s = float(G("~context_max_age_s", 3.0))
        if self.context_max_age_s <= 0.0:
            raise ValueError("~context_max_age_s must be > 0, got %r"
                             % (self.context_max_age_s,))
        # Beyond this the bearing is not trusted to close the sweep, and the sweep
        # falls back to its timeout rather than to a heading that has stopped moving.
        self.bearing_max_age_s = float(G("~bearing_max_age_s", 0.5))
        # Whether to wait for the platform to CONFIRM the recovery mode before
        # commanding. Default FALSE, and that is a safety choice, not an oversight:
        # a drone-side demo manager that has not been taught the 'recovery' mode
        # DROPS the request silently, and a recovery that waits forever for a
        # confirmation that can never come is a recovery that does not exist. When
        # unconfirmed we still command (and warn loudly): the follower may briefly
        # fight us, which is strictly better than nobody flying the drone.
        self.require_mode_confirm = _param_bool("~require_mode_confirm", False)
        self.request_repeat_s = float(G("~request_repeat_sec", 0.5))
        # Zero Twists to send, one per tick, when handing back. It must be >1 and
        # they must be on SEPARATE ticks, for two independent reasons:
        #   * the XTEND converter ignores the FIRST zero after a motion command
        #     (zero_stop_required_count = 2) because the planner emits transient
        #     zeros between yaw bursts -- so a lone zero is silently discarded and
        #     the drone HOLDS the rung it was flying (the platform is hold-style);
        #   * cmd_vel_gate subscribes queue_size=1, so two zeros published
        #     back-to-back in one tick can coalesce into one and undo the first fix.
        # Meanwhile the follower cannot take over instantly anyway: our fly_straight
        # request round-trips through the ROS2 demo manager. These ticks are what
        # actually stops the drone in that gap.
        self.release_zero_ticks = int(G("~release_zero_ticks", 3))
        if self.release_zero_ticks < 2:
            raise ValueError(
                "~release_zero_ticks must be >= 2 (got %r): the XTEND converter "
                "discards the first zero after a motion command, so a single zero "
                "would leave the drone flying the last recovery command"
                % (self.release_zero_ticks,))

        # Every default below comes from the core params, never a number retyped
        # here. Duplicating them is how a tuning fix reaches the launch but not the
        # node: the two then disagree, and which one flies depends on whether some
        # launch happens to set that param -- silently, since both values look fine
        # on their own. (Exactly the trap ~use_pose_estimator already sits in: node
        # default False, launch default true.)
        d = LostLocalizationParams()
        self.recovery = LostLocalizationRecovery(LostLocalizationParams(
            enabled=_param_bool("~enabled", d.enabled),
            stale_s=float(G("~stale_s", d.stale_s)),
            ladder_s=float(G("~ladder_s", d.ladder_s)),
            # Finish the move that was in flight before falling back to the
            # stop-and-search ladder: keep turning through a turn, give the metres
            # back after an advance. Needs ~cmd_vel_topic to carry the navigator's
            # commands (it does -- we publish there too); with nothing observed the
            # context is unknown and the prelude is skipped on its own.
            persist_enabled=_param_bool("~persist_enabled", d.persist_enabled),
            persist_turn_s=float(G("~persist_turn_s", d.persist_turn_s)),
            persist_back_s=float(G("~persist_back_s", d.persist_back_s)),
            persist_settle_s=float(G("~persist_settle_s", d.persist_settle_s)),
            exit_confirm_poses=int(G("~exit_confirm_poses", d.exit_confirm_poses)),
            back_speed=float(G("~back_speed", d.back_speed)),
            back_duration_s=float(G("~back_duration_s", d.back_duration_s)),
            back_repeats=int(G("~back_repeats", d.back_repeats)),
            dwell_s=float(G("~dwell_s", d.dwell_s)),
            # Climb needs a platform that accepts a vertical velocity. On XTEND
            # that depends on WHICH Twist converter is running: the in-process one
            # inside online_nav_bridge honours linear.z, while the standalone
            # xtend_twist_to_cmd_nav.py drops it unless started with
            # --allow-multi-axes. There is no way to detect this from here (a
            # dropped axis produces no error, the drone simply does not rise), so
            # set this false when the climb cannot work and the ladder will skip
            # those rungs rather than burn seconds going nowhere.
            climb_enabled=_param_bool("~climb_enabled", d.climb_enabled),
            climb_speed=float(G("~climb_speed", d.climb_speed)),
            climb_duration_s=float(G("~climb_duration_s", d.climb_duration_s)),
            climb_repeats=int(G("~climb_repeats", d.climb_repeats)),
            turn_enabled=_param_bool("~turn_enabled", d.turn_enabled),
            # rad/s, like every other rotation rate in this stack (the follower's
            # ~yaw_rate, the core's turn_rate). A degrees-facing knob here would be
            # the third unit convention on one airframe, and the bearing topic has
            # already shown what that costs.
            turn_rate=float(G("~turn_rate", d.turn_rate)),
            turn_dir=_turn_dir(G("~turn_dir", "left" if d.turn_dir > 0 else "right")),
            turn_target_rad=math.radians(float(G("~turn_target_deg",
                                                 math.degrees(d.turn_target_rad)))),
            turn_timeout_s=float(G("~turn_timeout_s", d.turn_timeout_s)),
        ))

        # ── Live state ──
        self._last_rx = None          # arrival time of the newest pose
        self._pose_count = 0          # monotonic; the core diffs it to spot arrivals
        self._bearing = None
        self._bearing_rx = None
        self._last_cmd = None         # (vx, wz) of the last MOVE the navigator sent
        self._last_cmd_rx = None      # ...and when it sent it
        self.current_demo_mode = None
        self._requested_mode = None
        self._last_request_pub_t = None
        self._release_ticks = 0       # zeros sent so far in the current hand-back
        self._last_tick = rospy.Time.now()
        self._last_state = None
        self._warned_no_pose = False
        # True until the gate says otherwise: an un-wired gate must not disable the
        # recovery, only a gate that actually reports itself shut.
        self._go = True

        # ── Publishers ──
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_req_topic, String,
                                            queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1,
                                          latch=True)

        # ── Subscribers ──
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb,
                         queue_size=10)
        # Yes, the topic we publish on: this is how we learn what the navigator
        # was flying. See the module docstring for why, and _cmd_cb for how our
        # own Twists are kept out of it.
        rospy.Subscriber(self.cmd_vel_topic, Twist, self._cmd_cb, queue_size=10)
        if self.bearing_topic:
            rospy.Subscriber(self.bearing_topic, Float32, self._bearing_cb,
                             queue_size=10)
        if self.go_status_topic:
            rospy.Subscriber(self.go_status_topic, String, self._go_status_cb,
                             queue_size=1)

        self._banner()
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)

    # ── Inputs ────────────────────────────────────────────────────────
    def _pose_cb(self, _msg):
        """The ONLY thing we need from a pose is that it arrived, and when.

        Not its value, and not its header.stamp -- see the module docstring.
        """
        self._last_rx = rospy.Time.now()
        self._pose_count += 1

    def _bearing_cb(self, msg):
        """Cache the platform heading, in radians, whatever units it arrives in.

        The repo does not agree with itself about ``/xtend/bearing``:
        ``xtend_online_bridge_base.bearing_to_yaw_rad`` asserts radians and passes
        the value straight through, while ``online_rgbd_localization_node`` reads
        the same topic as ``yaw_deg`` and ``estimate_bearing`` documents "compass
        heading, degrees". Getting it wrong is not a small error: fed degrees while
        believing radians, the sweep clears its 2*pi target after about 9 degrees
        of real rotation and the drone lands almost immediately -- i.e. WORSE than
        having no bearing at all. So ``~bearing_units`` defaults to auto-detection
        rather than to either belief, and the detection LATCHES: a mid-sweep change
        of mind would corrupt the accumulated angle it is feeding.
        """
        b = float(msg.data)
        if self._bearing_units == "auto" and not self._bearing_is_deg:
            if abs(b) > _BEARING_DEG_THRESHOLD:
                self._bearing_is_deg = True
                rospy.logwarn(
                    "lost_localization: %s reads %.1f, which cannot be radians -- "
                    "treating it as DEGREES from now on. Set ~bearing_units "
                    "explicitly to pin this.", self.bearing_topic, b)
        self._bearing = math.radians(b) if self._bearing_is_deg else b
        self._bearing_rx = rospy.Time.now()

    def _cmd_cb(self, msg):
        """Remember the last MOVE the navigator asked for, and when.

        Two filters, both load-bearing:

        * **Not while we own cmd_vel.** Our own Twists come back on this
          subscription, and a recovery that read them would persist its own
          output -- the back-up rung would look like "we were flying backwards",
          the sweep like "we were turning". ``_requested_mode`` is set before the
          first command goes out and cleared only after the last release zero, so
          it brackets our ownership exactly.
        * **Zeros are not moves.** The follower publishes zeros constantly: while
          it settles between yaw bursts, while it brakes, while it waits for a
          mode. Those are the moments the drone is stationary but still very much
          mid-turn, and treating them as the last command would erase the very
          intent the persist exists to finish. Keeping only real moves is what
          makes "lost during the pause between two yaw bursts" persist as a turn.
        """
        if self._requested_mode is not None:
            return
        vx, wz = float(msg.linear.x), float(msg.angular.z)
        if abs(vx) < _CMD_EPS and abs(wz) < _CMD_EPS:
            return
        self._last_cmd = (vx, wz)
        self._last_cmd_rx = rospy.Time.now()

    def _motion_context(self, now):
        """What the navigator was flying, for the core's persist prelude.

        Translation beats yaw when a controller drives both (the continuous ones
        do; the one-axis follower never does): the two responses are opposites, so
        the tie has to break toward the failure that actually hurts. Losing a tag
        while advancing usually means we advanced INTO something -- close enough
        that the marker left the frame -- and giving those metres back is both the
        fix and the safe thing to do. Over-turning is merely unhelpful.
        """
        if self._last_cmd is None or self._last_cmd_rx is None:
            return MotionContext.unknown()
        if (now - self._last_cmd_rx).to_sec() > self.context_max_age_s:
            # Nothing recent enough to call a move in progress. Say so, rather
            # than finish a manoeuvre that ended a minute ago.
            return MotionContext.unknown()
        vx, wz = self._last_cmd
        if abs(vx) >= _CMD_EPS:
            return MotionContext.forward(vx)
        return MotionContext.turning(wz)

    def _demo_mode_cb(self, msg):
        self.current_demo_mode = (msg.data or "").strip().lower()

    def _go_status_cb(self, msg):
        """Track the GO gate. Its status is prose, so match how it says GO."""
        go = (msg.data or "").strip().upper().startswith("GO")
        if go != self._go:
            rospy.logwarn("lost_localization: GO gate is now %s -- recovery is %s",
                          "OPEN" if go else "CLOSED",
                          "armed" if go else "INERT (a human has the drone)")
        self._go = go

    def _fresh_bearing(self, now):
        """The platform heading if it is live, else None (sweep goes open loop)."""
        if self._bearing is None or self._bearing_rx is None:
            return None
        if (now - self._bearing_rx).to_sec() > self.bearing_max_age_s:
            return None
        return self._bearing

    # ── Control loop ──────────────────────────────────────────────────
    def _tick(self, _evt):
        now = rospy.Time.now()
        dt = (now - self._last_tick).to_sec()
        self._last_tick = now
        if dt <= 0.0:
            return                       # a bag jumped the clock backwards

        # The gate is shut: a human is flying. Every Twist we sent would be dropped
        # by the gate, so the ladder would tick through its rungs against a drone
        # that never moved and then "give up" and land it -- and that land, unlike
        # our Twists, is NOT gated. Stay out entirely, and reset so the episode
        # starts clean whenever GO does come.
        if not self._go:
            if self._requested_mode is not None:
                self._release()
            else:
                self.recovery.reset()
            self._publish_status_line("INERT -- GO gate closed; a human has the drone")
            return

        # inf, not a big number: "no pose has EVER arrived" is a wiring fault
        # (e.g. a bridge config that does not carry the pose topic at all), and
        # the core must not fly a blind ladder over it. Say so once, loudly.
        if self._last_rx is None:
            age = float("inf")
            self._warn_never_bootstrapped()
        else:
            age = (now - self._last_rx).to_sec()

        dec = self.recovery.update(age, dt, self._pose_count,
                                   yaw=self._fresh_bearing(now),
                                   context=self._motion_context(now))
        self._log_transition(dec)
        self._publish_status(dec)

        if not dec.active:
            self._release()
            return

        # Taking over (again): any hand-back in progress is abandoned mid-way, so
        # the next one starts from a full zero burst rather than a part-spent one.
        self._release_ticks = 0
        want = MODE_FINISH if dec.give_up else MODE_RECOVERY
        self._request_mode(want)
        if self.require_mode_confirm and self.current_demo_mode != want:
            rospy.logwarn_throttle(
                2.0, "lost_localization: HOLDING recovery -- waiting for demo_mode "
                     "%r (have %r). The drone is NOT being recovered.",
                want, self.current_demo_mode)
            return
        if self.current_demo_mode != want:
            rospy.logwarn_throttle(
                5.0, "lost_localization: commanding while demo_mode is %r, not %r "
                     "-- the platform has not granted the mode, so the follower may "
                     "fight us. Teach the demo manager the %r mode.",
                self.current_demo_mode, want, want)
        self._publish_cmd(dec.command)

    # ── Outputs ───────────────────────────────────────────────────────
    def _publish_cmd(self, command):
        """One ControlCommand -> one Twist.

        linear.y is hardwired 0 (the platform does not crab on this path). Unlike
        every other Twist assembler in this stack, linear.z is NOT hardwired: the
        climb rungs are the whole reason this node exists as a separate publisher.
        """
        m = Twist()
        m.linear.x = command.x
        m.linear.y = 0.0
        m.linear.z = command.z
        m.angular.z = command.yaw_rate
        self.cmd_pub.publish(m)

    def _request_mode(self, mode):
        """Ask for a demo mode; re-publish periodically until it is confirmed."""
        now = rospy.Time.now()
        changed = mode != self._requested_mode
        stale = (self._last_request_pub_t is None
                 or (now - self._last_request_pub_t).to_sec() >= self.request_repeat_s)
        if not (changed or stale):
            return
        if changed:
            rospy.loginfo("lost_localization: demo_mode request -> %s", mode)
        self._requested_mode = mode
        self.demo_req_pub.publish(String(data=mode))
        self._last_request_pub_t = now

    def _release(self):
        """Stop the drone and hand cmd_vel back to the follower.

        Mandatory, not tidy-up, and in this order for a reason:

        * Requesting a mode back is REQUIRED. The follower stays passive for as
          long as the latched demo_mode reads 'recovery', so merely going quiet
          would leave it passive forever and the drone would never fly again.
          Asked on the first tick, because the request round-trips through the
          ROS2 demo manager and the follower needs the head start.
        * Then zeros, one per tick, ``~release_zero_ticks`` of them -- a single
          zero does NOT stop this platform (see the param's comment). Until they
          land the drone is still flying the last rung, so this is the stop.

        Only acts if we actually took over, so an idle recovery never fights the
        follower's own mode management. Never releases out of FINISH: that land is
        terminal.
        """
        if self._requested_mode is None or self._requested_mode == MODE_FINISH:
            return
        if self._release_ticks == 0:
            self._request_mode(MODE_RELEASE)
        self._release_ticks += 1
        self._publish_cmd(_ZERO)
        if self._release_ticks >= self.release_zero_ticks:
            self._requested_mode = None      # done: go silent, follower has it
            self._release_ticks = 0

    # ── Reporting ─────────────────────────────────────────────────────
    def _warn_never_bootstrapped(self):
        if not self._warned_no_pose:
            self._warned_no_pose = True
            rospy.logwarn("lost_localization: no pose has EVER arrived on %s -- "
                          "recovery is INERT until one does (it cannot tell 'lost' "
                          "from 'never connected'). If the drone is flying, the "
                          "pose topic is not reaching this container: check that "
                          "the bridge config carries %s.",
                          self.pose_topic, self.pose_topic)

    def _log_transition(self, dec):
        if dec.state == self._last_state:
            return
        self._last_state = dec.state
        if dec.state == GIVE_UP:
            rospy.logerr("lost_localization: GIVE UP after %.1fs with no "
                         "localization -- the full ladder ran and no tag was "
                         "re-acquired. Requesting %s (stop -> land -> disarm).",
                         dec.elapsed_s, MODE_FINISH)
        else:
            rospy.logwarn("lost_localization: %s  (pose age %.2fs%s)",
                          dec.state, dec.pose_age_s,
                          ", rung %s" % dec.rung_label if dec.rung_label else "")

    def _status_line(self, dec):
        if not dec.active:
            return "%s -- localization healthy" % dec.state
        return ("%s -- pose %.2fs old%s, %.1fs into recovery"
                % (dec.state, dec.pose_age_s,
                   ", rung %s" % dec.rung_label if dec.rung_label else "",
                   dec.elapsed_s))

    def _publish_status(self, dec):
        self._publish_status_line(self._status_line(dec))

    def _publish_status_line(self, line):
        self.status_pub.publish(String(data=line))

    def _banner(self):
        p = self.recovery.p
        L = rospy.loginfo
        L("=" * 64)
        L("lost_localization (stop + recover when the pose goes cold)")
        L("  watching = %s  (PoseStamped; ARRIVAL time, not header.stamp)", self.pose_topic)
        L("  cmd out  = %s  (through the GO gate)", self.cmd_vel_topic)
        L("  enabled  = %s", p.enabled)
        L("  stop at  = %.2fs cold      ladder at = %.2fs cold", p.stale_s, p.ladder_s)
        if p.persist_enabled:
            L("  persist  = ON: mid-turn -> %.1fs more turn; mid-advance -> %.1fs back,",
              p.persist_turn_s, p.persist_back_s)
            L("            then %.1fs still to look. Reads the last move off %s;",
              p.persist_settle_s, self.cmd_vel_topic)
            L("            nothing seen in %.1fs => no prelude, straight to the stop.",
              self.context_max_age_s)
        else:
            L("  persist  = OFF (a dropout goes straight to the stop)")
        if self.recovery.ladder:
            L("  ladder   = %s", " -> ".join(r.label for r in self.recovery.ladder))
        else:
            L("  ladder   = (empty -- every stage disabled; will land on a dropout)")
        if p.climb_enabled:
            # Both failure directions are silent, so say both out loud BEFORE the
            # flight: the axis can be dropped by the converter (drone never rises),
            # or it can be far stronger than the number suggests (drone hits the
            # ceiling -- nothing here measures altitude to stop it).
            L("  climb   = ON: %d rungs of %.2f x %.1fs (a THRUST dial, not a rate --",
              p.climb_repeats, p.climb_speed, p.climb_duration_s)
            L("            the real height gained is MORE, and nothing measures it.")
            L("            Check the headroom above %s. If the drone never rises at",
              "cruise" if p.climb_enabled else "-")
            L("            all instead, the Twist converter is dropping linear.z:")
            L("            xtend_twist_to_cmd_nav.py needs --allow-multi-axes.")
        else:
            L("  climb   = OFF (ladder skips the climb rungs)")
        L("  sweep    = %.2f rad/s (%.0f deg/s) to the %s, %.0f deg, %.0fs cap",
          p.turn_rate, math.degrees(p.turn_rate),
          "left" if p.turn_dir > 0 else "right",
          math.degrees(p.turn_target_rad), p.turn_timeout_s)
        L("             bearing: %s", self.bearing_topic or "none -- OPEN LOOP")
        L("=" * 64)


def main():
    try:
        LostLocalizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses)
#   IO: ~drone_ns ('') ~cmd_vel_topic (<drone_ns>/cmd_vel_raw)
#           BOTH an output and an input: we publish recovery commands here AND
#           watch it to learn what the navigator was flying (see the persist
#           block below and _cmd_cb).
#       ~pose_topic (/xtend/localization)   the STAMPED source, not /gt_pose
#       ~bearing_topic (/xtend/bearing)     '' => sweep runs open loop
#       ~bearing_units (auto)  auto|rad|deg. The repo disagrees with itself about
#           this topic's units; fed degrees as radians the sweep closes after ~9
#           deg and lands. auto latches on the first value that cannot be radians.
#       ~bearing_max_age_s (0.5)
#       ~go_status_topic (/mission/go_status)  '' => ignore the GO gate. While the
#           gate is shut a human is flying: recovery stays inert (its Twists would
#           be dropped, but its give-up LAND would not be).
#       ~demo_mode_topic (/xtend/demo_mode) ~demo_mode_request_topic
#       ~status_topic (/recovery/status)    ~rate_hz (10.0)
#   detect: ~enabled (true) ~stale_s (0.3) ~ladder_s (1.0)
#           ~exit_confirm_poses (2)
#   persist: finish the move that was in flight before falling back to the ladder.
#           ~persist_enabled (true) ~persist_turn_s (0.6) ~persist_back_s (0.6)
#           ~persist_settle_s (0.5) ~context_max_age_s (3.0)
#           Mid-turn we keep turning at the navigator's OWN rate (the tag left the
#           frame because we rotated it out; the next one is often already coming
#           in). Mid-advance we reverse instead, at the speed we advanced at capped
#           by ~back_speed (losing a tag while advancing means we flew too close to
#           it). Either way we then hold still and look. A dropout during the
#           settle BETWEEN yaw bursts counts as mid-turn -- zeros are not moves.
#   back:   ~back_speed (0.30) ~back_duration_s (1.5) ~back_repeats (2)
#   settle: ~dwell_s (1.5)
#   climb:  ~climb_enabled (true) ~climb_speed (0.08) ~climb_duration_s (0.4)
#           ~climb_repeats (2)
#           ~climb_speed is a THRUST dial, not a rate: the vertical axis reuses the
#           FORWARD calibration (|v|/0.3*400) and the drone climbs much harder at a
#           given number than it flies forward. The height actually gained exceeds
#           speed*duration (hold-style commands + inertia) and NOTHING in this stack
#           measures altitude, so the ceiling is the only limit. Raise only against
#           a measured climb.
#           NOTE: the standalone xtend_twist_to_cmd_nav.py needs --allow-multi-axes
#           or it silently drops linear.z and the rung does nothing (the in-process
#           converter in online_nav_bridge honours it already).
#   sweep:  ~turn_enabled (true) ~turn_rate (0.50 rad/s) ~turn_dir (right)
#           ~turn_target_deg (360.0) ~turn_timeout_s (40.0)
#           ~turn_rate is rad/s, the same unit as the follower's ~yaw_rate (0.7),
#           so the two rotation dials on this airframe are directly comparable.
#           (~turn_target_deg stays in degrees: it is an EXTENT, not a rate, and a
#           full sweep reads better as 360 than as 6.28.)
#           ~turn_rate has a FLOOR as well as a ceiling, and the floor is the one
#           that bites: too low and the commanded yaw never overcomes the airframe,
#           so the drone barely rotates and a sweep that never swept looks exactly
#           like a sweep that found nothing.
#   modes:  ~require_mode_confirm (false)  see the __init__ comment: false so a
#           demo manager that does not know 'recovery' cannot mute the recovery
#           ~request_repeat_sec (0.5)
#           ~release_zero_ticks (3)  zeros (one per tick) sent when handing back.
#           MUST be >= 2: the XTEND converter discards the first zero after a
#           motion command, so one zero leaves the drone flying the last rung.
# ============================================================================
