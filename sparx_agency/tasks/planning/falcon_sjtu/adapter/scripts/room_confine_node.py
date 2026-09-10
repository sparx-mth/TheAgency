#!/usr/bin/env python3
"""Turn a bridged ROS 2 confinement request into FALCON's leased keep-in boxes.

The scene-graph host on ROS 2 knows which room the aircraft is searching and
where that room's doors are. FALCON, on ROS 1, reads its confinement from three
rosparams (``/map_config/keep_in_runtime``, ``/map_config/keep_out_runtime``,
``/map_config/confine_deadline``) that the patched planner re-reads once a
second. Nothing bridges rosparams, so this node is the seam: one latched
``std_msgs/String`` of JSON in, three rosparams out, plus an acknowledgement
the host can wait on.

**Why a lease rather than a flag.** The keep-in box is the only thing stopping
the aircraft leaving a room, which means a stale one is the only thing that
could strand it there. The host sends ``lease_s`` and this node writes an
absolute ``confine_deadline`` on the ROS clock, refreshed on every message. If
the host dies, the bridge drops, or this node is killed, the deadline passes
and the planner clears its own boxes on its next read -- the fence fails OPEN,
always, without anybody having to publish a release.

**Why the deadline is written last and cleared first.** Applying a fence means
writing geometry then arming it; lifting one means disarming then dropping the
geometry. In both orders the intermediate state is "no confinement", never "a
deadline with stale geometry" -- which would fence the aircraft into whichever
room it was in a minute ago.

Wire format on ``~in_topic`` (default ``/scene_graph/confine``)::

    {"room_id": 7,
     "lease_s": 8.0,
     "keep_in":  [[xmin, ymin, zmin, xmax, ymax, zmax], ...],
     "keep_out": [[xmin, ymin, zmin, xmax, ymax, zmax], ...]}

An empty ``keep_in`` AND ``keep_out``, or a ``lease_s`` of zero, is a RELEASE.
The host sends one when a room's turn ends; it is an optimisation, not a
requirement, because simply falling silent achieves the same thing one lease
later.

Runs inside the ROS1 Noetic FALCON container, on Python 3.8.
"""
from __future__ import annotations

import json

import rospy
from std_msgs.msg import String

KEEP_IN_PARAM = "/map_config/keep_in_runtime"
KEEP_OUT_PARAM = "/map_config/keep_out_runtime"
DEADLINE_PARAM = "/map_config/confine_deadline"
ROOM_PHASE_PARAM = "/mission/room_phase"
RESUME_PARAM = "/fsm/resume_from_finish"

BOX_VALUES = 6
"""xmin ymin zmin xmax ymax zmax -- the flat form the planner parses."""


def flatten(boxes):
    # type: (object) -> list
    """Flatten ``[[6 floats], ...]`` to one list, dropping malformed boxes.

    Malformed entries are skipped rather than raised on, deliberately: this
    runs between a bridge and a planner, and a single bad box must degrade the
    fence rather than kill the node that is holding it. What is NOT tolerated
    is silence -- every drop is logged.
    """
    flat = []
    if not isinstance(boxes, (list, tuple)):
        return flat
    for i, box in enumerate(boxes):
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            rospy.logwarn("[confine] box %d is not numeric; dropped", i)
            continue
        if len(values) != BOX_VALUES:
            rospy.logwarn("[confine] box %d has %d values, expected %d; dropped",
                          i, len(values), BOX_VALUES)
            continue
        flat.extend(values)
    return flat


class RoomConfineNode(object):
    """Holds FALCON's confinement lease on behalf of the ROS 2 host."""

    def __init__(self):
        self._in_topic = rospy.get_param("~in_topic", "/scene_graph/confine")
        self._ack_topic = rospy.get_param("~ack_topic", "/scene_graph/confine_ack")
        self._max_lease_s = float(rospy.get_param("~max_lease_s", 30.0))
        self._held = False

        # The C++ resume out of FINISH is what makes room-by-room search
        # possible at all, and it is off by default so a plain survey run is
        # unchanged. This node is only ever launched for a room-by-room
        # mission, so it is the right place to turn it on.
        rospy.set_param(RESUME_PARAM, True)
        self._release("startup")

        self._ack_pub = rospy.Publisher(self._ack_topic, String, queue_size=1,
                                        latch=True)
        rospy.Subscriber(self._in_topic, String, self._on_request, queue_size=1)
        rospy.loginfo("[confine] up: %s -> %s/%s/%s (max lease %.1fs), %s=true",
                      self._in_topic, KEEP_IN_PARAM, KEEP_OUT_PARAM,
                      DEADLINE_PARAM, self._max_lease_s, RESUME_PARAM)

    # -- the request ------------------------------------------------------
    def _on_request(self, msg):
        # type: (String) -> None
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError) as exc:
            rospy.logwarn_throttle(5.0, "[confine] undecodable request: %s", exc)
            return
        if not isinstance(payload, dict):
            rospy.logwarn_throttle(5.0, "[confine] request is not an object")
            return

        keep_in = flatten(payload.get("keep_in"))
        keep_out = flatten(payload.get("keep_out"))
        try:
            lease_s = float(payload.get("lease_s", 0.0))
        except (TypeError, ValueError):
            lease_s = 0.0
        room_id = payload.get("room_id")

        if lease_s <= 0.0 or (not keep_in and not keep_out):
            self._release("request from the host (room %s)" % (room_id,))
            self._ack(room_id, False, 0.0, 0, 0)
            return

        # Bounded on purpose. The host asks for a few seconds and refreshes;
        # a bug that asked for an hour would fence the aircraft for an hour,
        # and the lease is the safety property this whole mechanism rests on.
        if lease_s > self._max_lease_s:
            rospy.logwarn_throttle(
                10.0, "[confine] lease %.1fs exceeds ~max_lease_s %.1fs; "
                "clamping", lease_s, self._max_lease_s)
            lease_s = self._max_lease_s

        now = rospy.Time.now().to_sec()
        # Geometry FIRST, then the deadline that arms it. The reverse order
        # has a window in which the previous room's boxes are live under the
        # new room's deadline.
        rospy.set_param(KEEP_IN_PARAM, keep_in)
        rospy.set_param(KEEP_OUT_PARAM, keep_out)
        rospy.set_param(DEADLINE_PARAM, now + lease_s)
        rospy.set_param(ROOM_PHASE_PARAM, True)
        if not self._held:
            rospy.logwarn("[confine] confining to room %s: %d keep-in box(es), "
                          "%d door seal(s), lease %.1fs",
                          room_id, len(keep_in) // BOX_VALUES,
                          len(keep_out) // BOX_VALUES, lease_s)
        self._held = True
        self._ack(room_id, True, now + lease_s,
                  len(keep_in) // BOX_VALUES, len(keep_out) // BOX_VALUES)

    def _release(self, why):
        # type: (str) -> None
        """Disarm, then drop the geometry. Never the other way round."""
        rospy.set_param(DEADLINE_PARAM, 0.0)
        rospy.set_param(KEEP_IN_PARAM, [])
        rospy.set_param(KEEP_OUT_PARAM, [])
        rospy.set_param(ROOM_PHASE_PARAM, False)
        if self._held:
            rospy.logwarn("[confine] released: %s", why)
        self._held = False

    def _ack(self, room_id, held, deadline, n_in, n_out):
        # type: (object, bool, float, int, int) -> None
        self._ack_pub.publish(String(data=json.dumps({
            "stamp": rospy.Time.now().to_sec(),
            "room_id": room_id,
            "held": bool(held),
            "deadline": float(deadline),
            "keep_in_boxes": int(n_in),
            "keep_out_boxes": int(n_out),
        })))

    def spin(self):
        # type: () -> None
        try:
            rospy.spin()
        finally:
            # Best effort. The lease would lapse on its own within seconds,
            # but a clean exit should not leave the aircraft fenced at all.
            try:
                self._release("node shutting down")
            except Exception:               # noqa: BLE001 -- teardown must finish
                pass


def main():
    # type: () -> None
    rospy.init_node("room_confine")
    RoomConfineNode().spin()


if __name__ == "__main__":
    main()
