#!/usr/bin/env python3
"""cmd_vel_gate_node.py -- the GO gate: nothing reaches the drone until you say so.

The whole pipeline (BEV, A*, the correctors, the detector, the mission director) can
come up and run while the drone is still on the ground or being taken off by hand --
but the moment a follower publishes /cmd_vel, it fights whoever is flying. This node
is the single choke point between "the stack decided a velocity" and "the drone gets
a velocity":

    waypoint_follower  ─┐
                        ├─>  <in_topic>  ─[ GO? ]─>  <out_topic>  ─> the drone
    object_approach    ─┘

Everything upstream publishes to ``~in_topic`` (``<drone_ns>/cmd_vel_raw``) and only
this node publishes ``~out_topic`` (``<drone_ns>/cmd_vel``). It is deliberately the
ONLY gate rather than a check inside each follower: a new publisher added later is
gated by construction, and there is exactly one place to read to know whether the
drone can move.

While CLOSED it publishes NOTHING -- it does not forward, and it does not emit zeros.
A zero stream is not neutral: it is a command to hold still, which is exactly what
fights a manual takeoff. Silence lets the pilot fly.

GO is a latched ``std_msgs/Bool`` on ``~go_topic``, so it survives a late-joining gate
and can come from the mission director's GO button or the command line::

    rostopic pub -1 /mission/go std_msgs/Bool "data: true"

``~start_go`` decides the state before any message arrives. It defaults to TRUE
(pass-through), so every existing launch behaves exactly as before; the object mission
sets it false to require an explicit GO.

Closing the gate mid-flight (GO -> false) sends ONE zero twist by default
(``~zero_on_close``), so a stack that was driving does not leave the drone coasting on
its last command. That is a stop, not a hold: nothing further is published.
"""
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

from thinking import Thinker


def _param_bool(name, default):
    """rosparam bool that also accepts the strings roslaunch passes ('true'/'false')."""
    v = rospy.get_param(name, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class CmdVelGateNode(object):
    def __init__(self):
        rospy.init_node("cmd_vel_gate")
        G = rospy.get_param

        self.drone_ns = str(G("~drone_ns", ""))
        self.in_topic = str(G("~in_topic", self.drone_ns + "/cmd_vel_raw"))
        self.out_topic = str(G("~out_topic", self.drone_ns + "/cmd_vel"))
        self.go_topic = str(G("~go_topic", "/mission/go"))
        self.status_topic = str(G("~status_topic", "/mission/go_status"))
        if self.in_topic == self.out_topic:
            raise ValueError("cmd_vel_gate: ~in_topic and ~out_topic are both %r -- "
                             "the gate would feed itself" % self.in_topic)

        # Default OPEN so every existing launch is unchanged; the object mission
        # explicitly sets start_go:=false to require a GO.
        self.go = _param_bool("~start_go", True)
        self.zero_on_close = _param_bool("~zero_on_close", True)

        self._passed = 0          # commands forwarded since the gate last opened
        self._blocked = 0         # commands dropped while closed

        self.thinker = Thinker("cmd_vel_gate")

        self.cmd_pub = rospy.Publisher(self.out_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1,
                                          latch=True)
        rospy.Subscriber(self.in_topic, Twist, self._cmd_cb, queue_size=1)
        rospy.Subscriber(self.go_topic, Bool, self._go_cb, queue_size=1)

        self._publish_status()
        self._banner()
        rospy.Timer(rospy.Duration(5.0), self._heartbeat)

    # ── Gate ──────────────────────────────────────────────────────────
    def _cmd_cb(self, msg):
        """Forward one command, or drop it. The whole point of the node."""
        if not self.go:
            self._blocked += 1
            return
        self._passed += 1
        self.cmd_pub.publish(msg)

    def _go_cb(self, msg):
        want = bool(msg.data)
        if want == self.go:
            return
        self.go = want
        # The operator's first question about a motionless drone is whether it is
        # even allowed to move; narrate the transition here, never in _cmd_cb.
        self.thinker.say(
            "GO gate is open -- the drone may fly" if want else
            "GO gate is shut -- holding the drone still, nobody can command it",
            category="mission", level="info" if want else "warn")
        if want:
            rospy.logwarn("cmd_vel_gate: GO -- commands now reach the drone "
                          "(%d were blocked while closed)", self._blocked)
            self._passed = 0
        else:
            rospy.logwarn("cmd_vel_gate: STOP -- commands blocked (%d had passed)",
                          self._passed)
            if self.zero_on_close:
                # One zero so a stack that was driving does not leave the drone
                # coasting on its last command. Then silence.
                self.cmd_pub.publish(Twist())
            self._blocked = 0
        self._publish_status()

    # ── Reporting ─────────────────────────────────────────────────────
    def _status_line(self):
        return ("GO -- commands reaching the drone" if self.go else
                "HELD -- no commands sent; publish true on %s to fly" % self.go_topic)

    def _publish_status(self):
        self.status_pub.publish(String(data=self._status_line()))

    def _heartbeat(self, _evt):
        """Say it out loud periodically: a silently-held gate looks like a hung stack."""
        if self.go:
            rospy.loginfo_throttle(30.0, "cmd_vel_gate: GO (%d commands passed)",
                                   self._passed)
        else:
            rospy.logwarn_throttle(10.0,
                                   "cmd_vel_gate: HOLDING -- %d commands blocked. "
                                   "Press GO (or: rostopic pub -1 %s std_msgs/Bool "
                                   "\"data: true\")", self._blocked, self.go_topic)

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("cmd_vel_gate (the GO gate: nothing reaches the drone until GO)")
        L("  in   = %s", self.in_topic)
        L("  out  = %s", self.out_topic)
        L("  go   = %s   (std_msgs/Bool, latched)", self.go_topic)
        L("  state= %s", self._status_line())
        L("=" * 64)


def main():
    try:
        CmdVelGateNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses)
#   ~drone_ns       ('')                  namespace the in/out defaults are built from
#   ~in_topic       (<drone_ns>/cmd_vel_raw)  what the followers publish
#   ~out_topic      (<drone_ns>/cmd_vel)      what the drone listens to
#   ~go_topic       (/mission/go)         std_msgs/Bool, latched; true = fly
#   ~status_topic   (/mission/go_status)  std_msgs/String, latched
#   ~start_go       (true)                gate state before any GO message. TRUE keeps
#                                         every existing launch behaving as before;
#                                         object_mission.launch sets it false.
#   ~zero_on_close  (true)                send one zero twist when GO goes false
#   ~thinking / ~thinking_topic / ~thinking_echo  -- inherited from thinking.Thinker;
#                                         narration on/off, its topic, rosout mirror
# ============================================================================
