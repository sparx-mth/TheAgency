#!/usr/bin/env python3
"""rooster_demo_mode_manager.py -- minimal demo-mode arbiter for Rooster/Sphera.

Confirmed missing 2026-07-28: XTEND has xtend_drone_demo_manager.py (a ROS2 node,
run alongside the XTEND stack) to turn a requested mode (~request_topic) into the
authoritative current mode (~mode_topic); Rooster never had an equivalent, so
/R1/demo_mode never actually reported "turning" during a real rotation.
waypoint_follower_node.py's own gating already tolerates this (controller=multi_axis
with ~mx_require_mode:=false), but mapping_sync_node_sphera.py's rotation freeze --
the authoritative place voxel-pair fusion is paused while the platform turns, per
its own docstring -- depends on seeing "turning" on this topic. Without it, depth
frames kept fusing with rapidly-changing pose through a real ~50 deg/s in-place
turn, smearing obstacles into a ring around the drone (confirmed live: BEV showed a
full ring of occupied cells with the planner reporting "boxed in - no A* route").

Both waypoint_follower_node.py (publishes ~demo_mode_request, subscribes
~demo_mode) and mapping_sync_node_sphera.py (subscribes ~demo_mode) are ROS1 nodes
inside the same `falcon` container -- unlike most other Rooster-only ROS1<->ROS2
naming in this stack, /R1/demo_mode never crosses ros1_bridge, so this arbiter is
ROS1/rospy, run via `rosrun falcon_adapter`, not a ROS2 node in `it`.

Deliberately NOT a copy of xtend_drone_demo_manager.py: that node also sends
stop/land/disarm cmd_nav actions on entering FINISH mode, which would make this
arbiter autonomously land the drone whenever FALCON's own state machine requests
FINISH -- landing should stay the pilot's deliberate action for Rooster (see the
fly-rooster-sphera skill), not something a mode-echo utility triggers as a side
effect. This node's only job is: echo the request back as the current mode,
immediately, plus a periodic re-publish so a late subscriber sees current state.
No state machine, no side effects, nothing else.
"""
import rospy
from std_msgs.msg import String


class RoosterDemoModeManager(object):
    def __init__(self):
        rospy.init_node("rooster_demo_mode_manager")
        G = rospy.get_param

        self.request_topic = str(G("~request_topic", "/R1/demo_mode_request"))
        self.mode_topic = str(G("~mode_topic", "/R1/demo_mode"))
        self.publish_period_sec = float(G("~publish_period_sec", 1.0))
        self.current_mode = str(G("~initial_mode", "fly_straight"))

        self.mode_pub = rospy.Publisher(self.mode_topic, String, queue_size=1,
                                        latch=True)
        rospy.Subscriber(self.request_topic, String, self._on_request, queue_size=10)
        rospy.Timer(rospy.Duration(self.publish_period_sec), self._publish_current_mode)

        rospy.loginfo(
            "rooster_demo_mode_manager ready\n"
            "  request in:  %s\n"
            "  mode out:    %s (initial=%r)",
            self.request_topic, self.mode_topic, self.current_mode)
        self._publish_current_mode()

    def _on_request(self, msg):
        requested = str(msg.data).strip()
        if not requested:
            return
        if requested != self.current_mode:
            rospy.loginfo("Mode: %r -> %r", self.current_mode, requested)
            self.current_mode = requested
        self._publish_current_mode()

    def _publish_current_mode(self, _evt=None):
        self.mode_pub.publish(String(data=self.current_mode))


def main():
    try:
        RoosterDemoModeManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
