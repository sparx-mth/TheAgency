"""One writer on ``cmd_vel`` at a time: the mute lease and the gate, together.

Two continuous writers on ``/simple_drone/cmd_vel`` is last-writer-wins, not a
handover, and both writers here are continuous by construction:

* FALCON's ``bspline_follower`` publishes a Twist EVERY tick at 50 Hz in every
  state it has;
* our ``trajectory_follower_node`` publishes at 20 Hz from every branch of its
  control step -- a zero twist when capsized, odom-stale or not-flying, a
  hold-altitude twist when it has no path, a tracking twist otherwise. There
  is no "publish nothing" state in it at all.

So the handover needs BOTH halves, and this node owns both so they cannot
disagree:

* **the mute**, a lease on the latched ``/scene_graph/external_ctrl`` Bool.
  FALCON's follower checks it at ``_send()``, the single choke point every one
  of its states reaches the wire through, and while muted it publishes nothing
  -- not the command, not a zero. ``True`` is both claim and renew and must be
  republished inside ``~external_ctrl_timeout_s`` (5.0 s by default) or FALCON
  logs a warning and takes the aircraft back;
* **the gate**, forwarding ``/simple_drone/cmd_vel_raw`` to
  ``/simple_drone/cmd_vel`` only while the mute is held. The room-search
  follower config already points the follower at ``_raw``; nothing on ROS 2
  subscribes it, which is why ``fly:=true`` does nothing today. This node is
  the missing subscriber.

The two are driven by ONE latched Bool from the supervisor, so the gate is
open exactly when FALCON is muted and the invariant is structural rather than
a timing hope.

**The zero twist on both edges is not politeness.** The SJTU plugin latches
the last Twist it was handed and has no watchdog of its own, so whichever
writer falls silent leaves the aircraft flying its last command until the
other one speaks. Taking the mute without immediately publishing would fly
FALCON's last command onward; releasing the gate without one would fly ours.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.cmd_vel_arbiter_node \\
        --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool

from sparx_agency.robots.SJTU.adapters import topics
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import latched_qos
from sparx_agency.tasks.mapping.scene_graph.ros2.target_approach_release import (
    emergency_release)


def command_qos(depth: int = 10) -> QoSProfile:
    """RELIABLE + VOLATILE, matching ``bridge.yaml``'s ``cmd_vel`` entry.

    Not :func:`latched_qos`: a transient_local writer on the command topic
    would replay its last twist to any late-joining reader, which for a
    velocity command means an aircraft that starts moving when a diagnostic
    tool connects.
    """
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


class CmdVelArbiterNode(Node):
    """Holds the FALCON mute and gates our follower's twists behind it."""

    def __init__(self):
        super().__init__("cmd_vel_arbiter")

        p = self.declare_parameter
        p("active_topic", "/object_search/active")
        p("target_seen_topic", "/target_seen")
        p("cmd_in", "/simple_drone/cmd_vel_raw")
        p("cmd_out", topics.CMD_VEL)
        p("ctrl_topic", "/scene_graph/external_ctrl")
        # Well inside FALCON's 5 s lease, and on its own timer so a slow
        # planning callback in another node cannot starve the renewal.
        p("ctrl_period_s", 1.0)
        # Zero twists published after the claim before ours are forwarded, so
        # the mute has crossed the bridge and FALCON's stream has stopped
        # before we start interleaving with it.
        p("settle_ticks", 3)
        p("info_topic", "/cmd_vel_arbiter/info")

        g = lambda name: self.get_parameter(name).value

        self._cmd_out_topic = str(g("cmd_out"))
        self._ctrl_topic = str(g("ctrl_topic"))
        self._settle_ticks = max(0, int(g("settle_ticks")))

        self._engaged = False
        self._stood_down = False
        self._settling = 0
        self._forwarded = 0
        self._dropped = 0

        self._cmd_pub = self.create_publisher(Twist, self._cmd_out_topic,
                                              command_qos())
        self._ctrl_pub = self.create_publisher(Bool, self._ctrl_topic,
                                               latched_qos())
        self._info_pub = self.create_publisher(Bool, str(g("info_topic")),
                                               latched_qos())

        self.create_subscription(Bool, str(g("active_topic")),
                                 self._on_active, latched_qos())
        self.create_subscription(Bool, str(g("target_seen_topic")),
                                 self._on_target_seen, latched_qos())
        # VOLATILE and shallow: a velocity command is only ever worth
        # forwarding while it is fresh, and a queue of stale ones is a queue
        # of wrong ones.
        self.create_subscription(Twist, str(g("cmd_in")), self._on_cmd,
                                 command_qos(depth=1))

        self._renew_timer = None
        self.get_logger().info(
            "cmd_vel_arbiter up: %s -> %s, mute on %s every %.1fs, settle=%d"
            % (str(g("cmd_in")), self._cmd_out_topic, self._ctrl_topic,
               float(g("ctrl_period_s")), self._settle_ticks))
        self._ctrl_period_s = float(g("ctrl_period_s"))

    # -- the claim --------------------------------------------------------
    def _on_active(self, msg: Bool) -> None:
        """The supervisor says whether it is flying the aircraft."""
        if self._stood_down:
            return
        want = bool(msg.data)
        if want and not self._engaged:
            self._engage()
        elif not want and self._engaged:
            self._disengage("supervisor released the aircraft")

    def _engage(self) -> None:
        """Take the mute, start renewing it, and stop the aircraft first."""
        self._engaged = True
        self._settling = self._settle_ticks
        self._ctrl_pub.publish(Bool(data=True))
        # FALCON's last command is latched in the plugin. Publish a zero
        # before anything else so the aircraft is not still flying it.
        self._cmd_pub.publish(Twist())
        if self._renew_timer is None:
            self._renew_timer = self.create_timer(
                self._ctrl_period_s, self._renew)
        self._info_pub.publish(Bool(data=True))
        self.get_logger().info("engaged: FALCON muted, gate open")

    def _renew(self) -> None:
        """Republish the lease. Silence past the timeout hands FALCON back."""
        if self._engaged and not self._stood_down:
            self._ctrl_pub.publish(Bool(data=True))

    def _disengage(self, why: str, release_mute: bool = True) -> None:
        """Close the gate, stop the aircraft, and give FALCON the wire back."""
        self._engaged = False
        if self._renew_timer is not None:
            self._renew_timer.cancel()
            self._renew_timer = None
        # Our last twist is latched in the plugin exactly as FALCON's was.
        self._cmd_pub.publish(Twist())
        if release_mute:
            self._ctrl_pub.publish(Bool(data=False))
        self._info_pub.publish(Bool(data=False))
        self.get_logger().info("disengaged: %s" % (why,))

    def _on_target_seen(self, msg: Bool) -> None:
        """Stand down for good and leave the mute topic to target_approach.

        Deliberately does NOT publish ``False``. ``target_approach_node``
        claims the same lease to fly its approach, and a release from here
        after its claim would unmute FALCON underneath it -- two planners on
        one aircraft, which is the exact failure this node exists to prevent.
        Stopping the renewal is enough: if nothing else claims the lease it
        expires on its own and FALCON resumes.
        """
        if not bool(msg.data) or self._stood_down:
            return
        self._stood_down = True
        if self._engaged:
            self._disengage("target seen -- handing over to the approach",
                            release_mute=False)
        else:
            self._info_pub.publish(Bool(data=False))
            self.get_logger().info("target seen -- standing down")

    # -- the gate ---------------------------------------------------------
    def _on_cmd(self, msg: Twist) -> None:
        """Forward one twist, but only while we actually own the aircraft."""
        if not self._engaged or self._stood_down:
            self._dropped += 1
            return
        if self._settling > 0:
            self._settling -= 1
            self._cmd_pub.publish(Twist())
            return
        self._cmd_pub.publish(msg)
        self._forwarded += 1

    # -- teardown ---------------------------------------------------------
    def release_on_exit(self) -> None:
        """Best-effort release, then the out-of-context fallback.

        ``stop_scene_graph.sh`` stops host nodes with SIGTERM, and rclpy has
        already invalidated this node's context by the time teardown runs, so
        the ordinary publish raises rather than sending. The fallback builds a
        throwaway participant on a fresh context to carry the release; without
        it a stopped stack leaves FALCON muted until its lease lapses, with
        the plugin still flying our last command.
        """
        if not self._engaged or self._stood_down:
            return
        try:
            self._cmd_pub.publish(Twist())
            self._ctrl_pub.publish(Bool(data=False))
            self.get_logger().info("released the mute on exit")
        except Exception:                   # noqa: BLE001 -- teardown must finish
            emergency_release(self._ctrl_topic, node_name="cmd_vel_arbiter_release")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelArbiterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.release_on_exit()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
