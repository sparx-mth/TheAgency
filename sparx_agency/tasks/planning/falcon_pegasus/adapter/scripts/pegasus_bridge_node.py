#!/usr/bin/env python3
"""Make Isaac Sim look, to FALCON, exactly like FALCON's own simulator.

FALCON was written against ``uav_simulator``: a geometry-only rig where a mesh
renderer produces depth images and ``poscmd_2_odom`` feeds the position command
straight back as odometry. It has no physics, so the commanded state *is* the
state and tracking error does not exist. This node replaces the two halves of
that rig with a real simulator, without touching a line of FALCON:

.. code-block:: text

    Isaac Sim (PX4 + PhysX)                    this node                FALCON
    ------------------------------------------------------------------------------
    depth image + camera pose  --TCP 5599-->  /uav_simulator/depth_image
                                              /uav_simulator/sensor_pose
    ground-truth vehicle state --TCP 5599-->  /uav_simulator/odometry
    outer-loop tracker + PX4   <--TCP 5600--  /planning/pos_cmd
                                              /planning/replan

Three things here are load-bearing and easy to get wrong:

* **One capture time, two topics.** FALCON refuses to fuse a depth image unless
  it can find a camera pose within 1 ms of the image's stamp. Both messages are
  built from a single ``FRAME`` carrying one timestamp, and the pose is published
  first -- so the tolerance is satisfied by construction, whatever the link does.
* **Wall clock, never simulated time.** ``exploration_node`` opens with
  ``CHECK(!use_sim_time)`` -- a glog fatal. Publishing ``/clock`` does not
  degrade this system, it kills it. Timestamps come from the Isaac side's
  ``time.time()``, which is the same ``CLOCK_REALTIME`` this container reads
  because both share the host kernel.
* **Intrinsics are checked, not assumed.** FALCON back-projects depth with
  rosparams, not with anything on the wire. If the aircraft renders with one
  camera and the mapper unprojects with another, nothing errors anywhere -- it
  just builds a confident map of a building the wrong size. The ``HELLO``
  handshake compares the two and refuses to run on a mismatch.

Run: ``rosrun falcon_pegasus pegasus_bridge_node.py`` (normally from
``launch/falcon_pegasus.launch``).
"""
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.falcon_pegasus.link.socket_link import (
    DOWNLINK_PORT, LinkServer, UPLINK_PORT,
)

# FALCON's own name for the world; every message it reads is in it.
WORLD_FRAME = "world"
CAMERA_FRAME = "camera"
BODY_FRAME = "base_link"

# `/planning/replan` values, from exploration_fsm.cpp. 2 means the FSM reached
# FINISH; it is also what makes traj_server exit, so it is the only reliable
# signal that the mission is over rather than merely quiet.
REPLAN_EXPLORATION_FINISHED = 2

# 1 means safetyCallback() found the EXECUTING trajectory in collision. FALCON
# replans and says nothing else about it; the aircraft is flying that trajectory
# in the meantime, so it is forwarded and the aircraft holds until a new one
# lands.
REPLAN_TRAJECTORY_UNSAFE = 1

# How long the command stream may go quiet before the aircraft is told the
# planner is gone. traj_server publishes at 100 Hz while a trajectory is live and
# stops entirely once exploration finishes.
COMMAND_TIMEOUT_S = 3.0

# How long a fresh uplink connection has to identify itself before it is treated
# as something other than the aircraft. Generous: the aircraft sends its HELLO
# in the same breath as connecting, so this only ever bounds how long a stray
# connection delays the real one.
HELLO_TIMEOUT_S = 5.0

# Intrinsics must agree to better than this. Not exact equality: the two sides
# reach the same numbers through a YAML load and a rosparam load, and a float
# that survives both is not obliged to be bit-identical.
INTRINSICS_TOLERANCE = 1e-3


def _odometry_as_transform(odometry):
    """The aircraft's pose as a ``world -> base_link`` TF, for RViz.

    Carries the same pose the planner was given, so a view attached to
    ``base_link`` in RViz follows the aircraft exactly as FALCON sees it.
    """
    transform = TransformStamped()
    transform.header = odometry.header
    transform.child_frame_id = odometry.child_frame_id
    transform.transform.translation.x = odometry.pose.pose.position.x
    transform.transform.translation.y = odometry.pose.pose.position.y
    transform.transform.translation.z = odometry.pose.pose.position.z
    transform.transform.rotation = odometry.pose.pose.orientation
    return transform


def _param(name):
    """Read a required rosparam, or explain what is missing.

    Raises:
        rospy.ROSException: If the parameter is absent. FALCON reads the same
            parameter and would silently use a default, so an absent one has to
            fail here or not at all.
    """
    if not rospy.has_param(name):
        raise rospy.ROSException(
            "required parameter %s is not set. The launch file must load "
            "FALCON's uav_model config before this node starts." % name)
    return rospy.get_param(name)


class PegasusBridge(object):
    """Bridges the Isaac Sim link to FALCON's simulator topics."""

    def __init__(self):
        self.depth_pub = rospy.Publisher("/uav_simulator/depth_image", Image, queue_size=1)
        self.pose_pub = rospy.Publisher("/uav_simulator/sensor_pose", TransformStamped,
                                        queue_size=100)
        self.odom_pub = rospy.Publisher("/uav_simulator/odometry", Odometry, queue_size=10)
        self.status_pub = rospy.Publisher("/falcon_pegasus/status", String, queue_size=10,
                                          latch=True)

        self.uplink_port = rospy.get_param("~uplink_port", UPLINK_PORT)
        self.downlink_port = rospy.get_param("~downlink_port", DOWNLINK_PORT)
        self.strict_intrinsics = rospy.get_param("~strict_intrinsics", True)

        # FALCON itself uses no TF at all -- it takes the camera pose off a topic
        # and never asks a listener for anything. RViz does: its Fixed Frame must
        # exist in the TF tree or every display reports "Fixed Frame [world] does
        # not exist" and the window stays empty. So this is published only when
        # something is watching, which in practice means when RViz is running.
        self.publish_tf = rospy.get_param("~publish_tf", False)
        self._tf = tf2_ros.TransformBroadcaster() if self.publish_tf else None

        self._downlink = None
        self._downlink_lock = threading.Lock()
        self._frames = 0
        self._odoms = 0
        self._commands = 0
        self._finished = False
        self._unsafe = False
        self._last_command_at = None
        self._planner_gone_reported = False

        rospy.Subscriber("/planning/pos_cmd", PositionCommand, self._on_position_command,
                         queue_size=10)
        rospy.Subscriber("/planning/replan", Int32, self._on_replan, queue_size=10)

    # ── the FALCON -> Isaac direction ────────────────────────────────────

    def _on_position_command(self, msg):
        """Forward one reference state to whatever is flying the aircraft."""
        self._commands += 1
        # Only a real trajectory arms the watchdog. Before FALCON has planned
        # anything, traj_server publishes a burst of ~200 commands parking the
        # aircraft at the init pose, all with trajectory_id 0; treating those as
        # "the planner is alive" makes the stream look like it died the moment
        # the burst ends, which is a normal part of start-up.
        if msg.trajectory_id >= 1:
            self._last_command_at = rospy.Time.now()
        self._send_down(protocol.position_command(
            msg.header.stamp.to_sec(), msg.trajectory_id,
            (msg.position.x, msg.position.y, msg.position.z),
            (msg.velocity.x, msg.velocity.y, msg.velocity.z),
            (msg.acceleration.x, msg.acceleration.y, msg.acceleration.z),
            msg.yaw, msg.yaw_dot))

    def _on_replan(self, msg):
        """Watch for the FSM declaring the space explored.

        ``/planning/replan == 2`` is FALCON's only announcement that it is done,
        and it is also what makes ``traj_server`` exit -- so after this the
        command stream stops for good. The aircraft has to be told, or it holds
        its last reference until a timeout it did not need to wait for.
        """
        if msg.data == REPLAN_TRAJECTORY_UNSAFE:
            # Edge-triggered: the FSM re-enters PLAN_TRAJ on every attempt and
            # this fires each time, which would be a hundred identical events
            # while the aircraft is already holding.
            if not self._unsafe:
                self._unsafe = True
                rospy.logwarn("[bridge] FALCON: obstacle on the live trajectory "
                              "-- telling the aircraft to hold")
                self._send_down(protocol.event(protocol.EVENT_TRAJECTORY_UNSAFE,
                                               "collision on the executing trajectory"))
            return
        self._unsafe = False
        if msg.data != REPLAN_EXPLORATION_FINISHED or self._finished:
            return
        self._finished = True
        rospy.logwarn("[bridge] FALCON says exploration is finished -- telling the aircraft")
        self._send_down(protocol.event(protocol.EVENT_EXPLORATION_FINISHED,
                                       "frontier set is empty"))
        self._publish_status("exploration_finished")

    def _send_down(self, message):
        """Write to the downlink, if the aircraft is still on the other end."""
        with self._downlink_lock:
            if self._downlink is None or self._downlink.closed:
                return
            self._downlink.send(message)

    # ── the Isaac -> FALCON direction ────────────────────────────────────

    def _on_hello(self, header):
        """Check that the renderer and the mapper agree about the camera.

        Raises:
            rospy.ROSException: On any disagreement, unless ``~strict_intrinsics``
                is false. A mismatch is invisible downstream -- FALCON builds a
                complete, self-consistent map of a building that is the wrong
                size -- so this is the only place it can be caught.
        """
        expected = {
            "fx": float(_param("/uav_model/sensing_parameters/camera_intrinsics/fx")),
            "fy": float(_param("/uav_model/sensing_parameters/camera_intrinsics/fy")),
            "cx": float(_param("/uav_model/sensing_parameters/camera_intrinsics/cx")),
            "cy": float(_param("/uav_model/sensing_parameters/camera_intrinsics/cy")),
            "width": int(_param("/uav_model/sensing_parameters/image_width")),
            "height": int(_param("/uav_model/sensing_parameters/image_height")),
        }
        mismatches = [
            "%s: aircraft renders %r, FALCON unprojects %r" % (key, header.get(key), value)
            for key, value in expected.items()
            if abs(float(header.get(key, 0.0)) - float(value)) > INTRINSICS_TOLERANCE
        ]
        rospy.loginfo("[bridge] aircraft: scene=%s run=%s camera=%sx%s fx=%.2f fy=%.2f "
                      "cx=%.2f cy=%.2f", header.get("scene"), header.get("run"),
                      header.get("width"), header.get("height"), header.get("fx", 0.0),
                      header.get("fy", 0.0), header.get("cx", 0.0), header.get("cy", 0.0))
        if not mismatches:
            return
        detail = "camera mismatch between Isaac Sim and FALCON:\n  " + "\n  ".join(mismatches)
        if self.strict_intrinsics:
            raise rospy.ROSException(detail)
        rospy.logerr("[bridge] %s\n(continuing because ~strict_intrinsics is false; "
                     "the map WILL be geometrically wrong)", detail)

    def _on_frame(self, header, payload):
        """Publish one depth image and the camera pose that took it.

        The pose goes first. FALCON looks a pose up by the image's stamp and, if
        every queued pose is older than the image, gives up for that call --
        publishing in this order means the matching entry is already in its queue
        by the time the image callback runs.
        """
        stamp = rospy.Time.from_sec(float(header["t"]))
        width, height = int(header["w"]), int(header["h"])

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = WORLD_FRAME
        transform.child_frame_id = CAMERA_FRAME
        position, quaternion = header["p"], header["q"]
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.pose_pub.publish(transform)
        if self._tf is not None:
            # The same transform, reused: world -> camera is exactly what the
            # mapper was given, so RViz draws the depth cloud where FALCON fused it.
            self._tf.sendTransform(transform)

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = CAMERA_FRAME
        image.height, image.width = height, width
        # "16UC1" is load-bearing. FALCON converts only when the encoding string
        # is exactly "32FC1" and otherwise reads the buffer as uint16 millimetres
        # regardless of what the label says -- so a wrong label is a wrong map,
        # not an error.
        image.encoding = "16UC1"
        image.is_bigendian = 0
        image.step = width * 2
        image.data = payload
        self.depth_pub.publish(image)
        self._frames += 1

    def _on_odometry(self, header):
        """Publish the aircraft's ground-truth state.

        The twist is world-frame, deliberately. REP-147 says a
        ``nav_msgs/Odometry`` twist is expressed in ``child_frame_id``, but
        FALCON reads ``twist.twist.linear`` straight into the initial velocity of
        its next B-spline without rotating it -- so a body-frame twist would send
        every replan off in the wrong direction. The sender documents the same
        thing from its end.
        """
        odometry = Odometry()
        odometry.header.stamp = rospy.Time.from_sec(float(header["t"]))
        odometry.header.frame_id = WORLD_FRAME
        odometry.child_frame_id = BODY_FRAME
        position, quaternion = header["p"], header["q"]
        linear, angular = header["v"], header["w"]
        odometry.pose.pose.position.x = position[0]
        odometry.pose.pose.position.y = position[1]
        odometry.pose.pose.position.z = position[2]
        odometry.pose.pose.orientation.x = quaternion[0]
        odometry.pose.pose.orientation.y = quaternion[1]
        odometry.pose.pose.orientation.z = quaternion[2]
        odometry.pose.pose.orientation.w = quaternion[3]
        odometry.twist.twist.linear.x = linear[0]
        odometry.twist.twist.linear.y = linear[1]
        odometry.twist.twist.linear.z = linear[2]
        odometry.twist.twist.angular.x = angular[0]
        odometry.twist.twist.angular.y = angular[1]
        odometry.twist.twist.angular.z = angular[2]
        self.odom_pub.publish(odometry)
        self._odoms += 1
        if self._tf is not None:
            self._tf.sendTransform(_odometry_as_transform(odometry))

    def _on_event(self, header):
        name = header.get("name")
        rospy.logwarn("[bridge] aircraft says: %s (%s)", name, header.get("detail", ""))
        if name == protocol.EVENT_MISSION_OVER:
            self._publish_status("mission_over")

    def _publish_status(self, state):
        self.status_pub.publish(String(data=state))

    # ── the loop ─────────────────────────────────────────────────────────

    def run(self):
        """Accept the aircraft's two connections and pump messages until shutdown."""
        uplink_server = LinkServer(self.uplink_port, "uplink")
        downlink_server = LinkServer(self.downlink_port, "downlink")
        rospy.loginfo("[bridge] waiting for Isaac Sim on 127.0.0.1:%d (uplink) and :%d "
                      "(downlink)", self.uplink_port, self.downlink_port)
        self._publish_status("waiting_for_aircraft")

        uplink = self._accept_aircraft(uplink_server)
        downlink = self._accept(downlink_server, "downlink")
        if uplink is None or downlink is None:
            return
        with self._downlink_lock:
            self._downlink = downlink
        # traj_server publishes ~200 setpoints at start-up, parking the aircraft
        # at the init pose, and it does that long before the aircraft connects.
        # Those count as commands but say nothing about whether the planner is
        # alive NOW, so the watchdog starts from the moment there is somebody to
        # warn -- otherwise it fires immediately on every connection.
        self._last_command_at = None
        rospy.loginfo("[bridge] aircraft connected")
        self._publish_status("connected")

        handlers = {
            protocol.KIND_HELLO: lambda header, _payload: None,   # already validated
            protocol.KIND_FRAME: self._on_frame,
            protocol.KIND_ODOM: lambda header, _payload: self._on_odometry(header),
            protocol.KIND_EVENT: lambda header, _payload: self._on_event(header),
        }
        report_at = rospy.Time.now()
        while not rospy.is_shutdown():
            for kind, header, payload in uplink.poll(timeout=0.05):
                handler = handlers.get(kind)
                if handler is None:
                    rospy.logwarn_throttle(10.0, "[bridge] ignoring message kind %s", kind)
                    continue
                handler(header, payload)
            if uplink.closed:
                rospy.logwarn("[bridge] the aircraft closed the uplink -- shutting down")
                self._publish_status("aircraft_gone")
                break
            self._watch_command_stream()
            now = rospy.Time.now()
            if (now - report_at).to_sec() >= 10.0:
                report_at = now
                rospy.loginfo("[bridge] %d depth frames, %d odom, %d commands forwarded",
                              self._frames, self._odoms, self._commands)

        uplink.close()
        downlink.close()
        uplink_server.close()
        downlink_server.close()

    def _accept(self, server, name):
        """Wait for one connection, staying responsive to Ctrl-C."""
        while not rospy.is_shutdown():
            endpoint = server.accept(timeout=1.0)
            if endpoint is not None:
                return endpoint
        return None

    def _accept_aircraft(self, server):
        """Accept uplink connections until one identifies itself as the aircraft.

        A connection that opens and closes without saying anything is not the
        aircraft -- it is a port probe, a health check, or a run that was killed
        between its two connects. Taking the first thing that connects as the
        aircraft means any of those consumes the accept, and the real aircraft
        then lands on the *downlink* socket while the bridge waits forever for a
        depth frame that a probe was never going to send. Requiring a ``HELLO``
        makes the wrong peer cost a log line instead of a run.
        """
        while not rospy.is_shutdown():
            endpoint = server.accept(timeout=1.0)
            if endpoint is None:
                continue
            header = self._await_hello(endpoint)
            if header is not None:
                self._on_hello(header)
                return endpoint
            rospy.logwarn("[bridge] a connection on the uplink port said nothing and "
                          "went away -- ignoring it and waiting for the aircraft")
            endpoint.close()
        return None

    def _await_hello(self, endpoint, timeout_s=HELLO_TIMEOUT_S):
        """The peer's ``HELLO`` header, or None if it never sent one."""
        deadline = rospy.Time.now() + rospy.Duration(timeout_s)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            for kind, header, _payload in endpoint.poll(timeout=0.2):
                if kind == protocol.KIND_HELLO:
                    return header
                rospy.logwarn("[bridge] first message on the uplink was kind %s, not a "
                              "HELLO", kind)
                return None
            if endpoint.closed:
                return None
        return None

    def _watch_command_stream(self):
        """Notice the trajectory server dying without a finish announcement.

        Reported once. The aircraft holds station on a stale reference anyway
        (its tracker times the reference out), but telling it turns a silent
        hover into a logged reason.
        """
        if self._finished or self._planner_gone_reported or self._last_command_at is None:
            return
        if (rospy.Time.now() - self._last_command_at).to_sec() < COMMAND_TIMEOUT_S:
            return
        self._planner_gone_reported = True
        rospy.logerr("[bridge] no position command for %.0f s -- the trajectory server "
                     "has stopped", COMMAND_TIMEOUT_S)
        self._send_down(protocol.event(protocol.EVENT_PLANNER_GONE,
                                       "no /planning/pos_cmd for %.0f s" % COMMAND_TIMEOUT_S))
        self._publish_status("planner_gone")


def main():
    rospy.init_node("pegasus_bridge")
    PegasusBridge().run()


if __name__ == "__main__":
    main()
