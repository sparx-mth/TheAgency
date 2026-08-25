#!/usr/bin/env python3
"""Republish FALCON's own trajectory as ``nav_msgs/Path`` so the BEV can draw it.

WHY THIS EXISTS
---------------
``bev_click_goal_node.py`` already knows how to draw routes -- it subscribes to
about a dozen ``nav_msgs/Path`` topics and colours each one. But every single
one of those belongs to the A*/NavDP click-to-fly pipeline
(``/path/waypoints``, ``/path/waypoints_raw``, ``/path/waypoints_astar``, ...),
and NONE of them has a publisher while FALCON is exploring: ``sphera_drone.launch``
starts only ``pose_adapter``, ``mapping_sync`` and ``rooster_demo_mode_manager``.
So during exploration the BEV window shows the occupancy grid and the drone dot,
and no route at all.

FALCON does publish its trajectory -- just not as a Path, and not on any topic
the viewer watches. ``traj_server`` streams ``quadrotor_msgs/PositionCommand``
one setpoint at a time on ``/planning/pos_cmd``, and the planner publishes
``visualization_msgs`` markers for RViz (``/planning_vis/trajectory`` for the
B-spline it intends to fly, ``/planning/travel_traj`` for what it has flown).
Those markers are exactly the geometry we want; they are simply the wrong
message type for this viewer.

This node is the adapter: markers in, Paths out. It changes nothing about
planning or control and publishes no commands.

FAILING LOUDLY
--------------
The input topic type differs between FALCON builds -- some publish ``Marker``,
some ``MarkerArray`` -- and subscribing with the wrong type fails SILENTLY: you
get a subscriber, no error, and no data, which looks exactly like "the feature
does not work". So the type is resolved from the master at startup rather than
assumed, and a topic that never appears is reported once a second instead of
being passed over in silence. This repo has been bitten more than once by a
component that was wired up, produced nothing, and said nothing about it.
"""

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

#: How long to wait for an input topic to appear before saying so out loud.
DISCOVER_TIMEOUT_S = 20.0
#: How often to complain about an input that never showed up.
NAG_PERIOD_S = 30.0


def _points_of(msg):
    """Every point carried by a Marker or MarkerArray, in order.

    FALCON draws a trajectory as a LINE_STRIP (or SPHERE_LIST) whose ``points``
    are the trajectory samples in the world frame, so the marker geometry IS
    the path -- no interpolation or reconstruction is needed.

    Args:
        msg: A ``visualization_msgs/Marker`` or ``MarkerArray``.

    Returns:
        list: ``geometry_msgs/Point`` in publication order, empty if the marker
        carries none (a DELETE marker, or a type that uses ``pose`` instead).
    """
    markers = msg.markers if isinstance(msg, MarkerArray) else [msg]
    points = []
    for marker in markers:
        # DELETE (2) / DELETEALL (3) carry no geometry and must not clear the
        # path we are already showing -- a momentary delete would otherwise
        # blank the route on every republish.
        if marker.action in (Marker.DELETE, Marker.DELETEALL):
            continue
        points.extend(marker.points)
    return points


class TrajectoryToPath(object):
    """One marker topic in, one ``nav_msgs/Path`` out."""

    def __init__(self, in_topic, out_topic, frame_id, label):
        """Wire up a single conversion.

        Args:
            in_topic: FALCON marker topic to read.
            out_topic: ``nav_msgs/Path`` topic to publish.
            frame_id: Frame to stamp the Path with; must match what the BEV
                and RViz expect or the route draws in the wrong place.
            label: Human name used in log lines.
        """
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.frame_id = frame_id
        self.label = label
        self.published = 0
        self.last_points = 0
        # Latched: the BEV viewer may start after the planner, and a route that
        # only exists in a message sent before the subscriber connected would
        # never be drawn.
        self.pub = rospy.Publisher(out_topic, Path, queue_size=1, latch=True)
        self.sub = None

    def resolve_and_subscribe(self):
        """Subscribe using the type the master actually advertises.

        Returns:
            bool: True if a subscriber was created.
        """
        if self.sub is not None:
            return True
        kind = None
        for name, type_name in rospy.get_published_topics():
            if name == self.in_topic:
                kind = type_name
                break
        if kind is None:
            return False
        if kind.endswith("MarkerArray"):
            msg_type = MarkerArray
        elif kind.endswith("Marker"):
            msg_type = Marker
        else:
            rospy.logwarn("[traj2path] %s publishes %s, which is neither "
                          "Marker nor MarkerArray -- not subscribing",
                          self.in_topic, kind)
            return False
        self.sub = rospy.Subscriber(self.in_topic, msg_type, self._cb,
                                    queue_size=1)
        rospy.loginfo("[traj2path] %s: %s (%s) -> %s", self.label,
                      self.in_topic, kind.split("/")[-1], self.out_topic)
        return True

    def _cb(self, msg):
        """Convert one marker message and publish the Path."""
        points = _points_of(msg)
        if not points:
            return
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id
        for point in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub.publish(path)
        self.published += 1
        self.last_points = len(path.poses)


def main():
    """Run the conversions until shutdown."""
    rospy.init_node("falcon_traj_to_path")
    get = rospy.get_param
    frame_id = str(get("~frame_id", "world"))

    bridges = [
        TrajectoryToPath(
            str(get("~planned_in", "/planning_vis/trajectory")),
            str(get("~planned_out", "/falcon/planned_path")),
            frame_id, "planned"),
        TrajectoryToPath(
            str(get("~executed_in", "/planning/travel_traj")),
            str(get("~executed_out", "/falcon/executed_path")),
            frame_id, "executed"),
    ]

    rospy.loginfo("[traj2path] frame_id=%s; waiting for FALCON to advertise "
                  "its trajectory markers", frame_id)
    started = rospy.Time.now()
    nagged = started
    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        pending = [b for b in bridges if not b.resolve_and_subscribe()]
        now = rospy.Time.now()
        if pending and (now - started).to_sec() > DISCOVER_TIMEOUT_S \
                and (now - nagged).to_sec() > NAG_PERIOD_S:
            nagged = now
            # Loud on purpose: silence here is indistinguishable from "the
            # trajectory feature is broken", which is what sent someone looking
            # for a route in the BEV that was never being published.
            rospy.logwarn("[traj2path] still no publisher for: %s -- the BEV "
                          "will show no FALCON route until FALCON advertises "
                          "it (is exploration_node running?)",
                          ", ".join(b.in_topic for b in pending))
        if not pending:
            live = ", ".join("%s %d msgs/%d pts" % (b.label, b.published,
                                                    b.last_points)
                             for b in bridges)
            rospy.loginfo_throttle(30.0, "[traj2path] %s", live)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
