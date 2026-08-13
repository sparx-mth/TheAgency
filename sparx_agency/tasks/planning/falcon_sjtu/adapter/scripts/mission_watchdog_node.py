#!/usr/bin/env python3
"""Judge the exploration while it flies, and end it when it stops being worth flying.

An exploration mission fails in ways every individual node reads as success. The
FSM is in ``EXEC_TRAJ``; the follower is tracking; the aircraft is moving; and
the map has not gained a voxel in four minutes because the drone is orbiting a
room whose only exit is a doorway it cannot thread. Nothing in the graph has the
two facts that would settle it -- where the aircraft has been, and how much map
that bought -- in the same place. This node is that place.

The judgement itself is ROS-free and unit-tested in
``core/planning/exploration/progress_monitor.py``; this file is the wiring:
subscribe, sample at 1 Hz, feed, act. Two levels of action:

* **nudge** -- confined and barren, but not yet for long enough to give up on.
  Published on ``/mission/nudge``; the follower answers it with a fresh survey
  turn, which rebuilds the local map and re-arms FALCON's frontier finder on a
  region it has stopped seeing frontiers in.
* **abort** -- published on ``/mission/abort`` with the reason, and printed with
  a ``[watchdog] MISSION ABORT`` banner that ``rig/campaign_run.sh`` greps for.
  The node does not kill anything itself: it is the judge, the harness is the
  executioner, and keeping those apart is what lets a human run the same stack
  interactively without a watchdog tearing their session down.

It also records the gap the whole task turns on: **where FALCON thinks the
aircraft is versus where it is.** FALCON starts each replan from the point its
PREVIOUS trajectory says the aircraft should have reached, not from odometry, so
every brake, hold and retreat this stack applies opens a gap between the planned
world and the real one. That gap is invisible in every existing log; it is
written to the progress trace here, per plan.

Runs inside the ROS1 Noetic FALCON container, on Python 3.8.
"""
from __future__ import annotations

import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from trajectory.msg import Bspline

from sparx_agency.core.planning.exploration.progress_monitor import (
    ExplorationProgressConfig, ExplorationProgressMonitor,
)
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory


class MissionWatchdogNode(object):
    """Sample the mission once a second and rule on it."""

    def __init__(self):
        # type: () -> None
        self._monitor = ExplorationProgressMonitor(ExplorationProgressConfig(
            time_cap_s=float(rospy.get_param("~time_cap_s", 2400.0)),
            grace_s=float(rospy.get_param("~grace_s", 90.0)),
            window_s=float(rospy.get_param("~window_s", 120.0)),
            confine_radius_m=float(rospy.get_param("~confine_radius_m", 3.0)),
            growth_m3_per_min=float(rospy.get_param("~growth_m3_per_min", 3.0)),
            confine_cap_s=float(rospy.get_param("~confine_cap_s", 240.0)),
            nudge_every_s=float(rospy.get_param("~nudge_every_s", 45.0)),
            barren_cap_s=float(rospy.get_param("~barren_cap_s", 300.0)),
            no_move_m=float(rospy.get_param("~no_move_m", 0.6)),
            no_move_cap_s=float(rospy.get_param("~no_move_cap_s", 120.0)),
            min_growth_m3=float(rospy.get_param("~min_growth_m3", 0.5))))
        # The trace is written INSIDE the container, to /tmp, and collected by
        # campaign_run.sh with `docker cp`. Writing to the read-only repo mount
        # would fail, and a host bind-mount would make an interactive run
        # depend on a directory the operator has to remember to create.
        self._trace_path = str(rospy.get_param("~trace", "/tmp/mission_progress.jsonl"))
        self._trace = None
        try:
            self._trace = open(self._trace_path, "a", 1)
        except IOError as exc:
            rospy.logwarn("[watchdog] no progress trace (%s): %s",
                          self._trace_path, exc)

        self._odom = None               # type: object
        self._coverage = 0.0
        self._finished = False
        self._aborted = False
        self._plans = 0
        # The plan-origin gap, latched per plan and reported with the next
        # sample: FALCON's start point against the aircraft's real position.
        self._plan_gap_m = None         # type: object
        self._plan_gap_max = 0.0
        self._plan_gap_sum = 0.0
        self._plan_gap_n = 0
        self._pose_lag_s = None         # type: object

        # ── keep the dead-end blacklist from outliving the map it describes ──
        # falcon_deadend_guard.patch persists physics-vetoed viewpoints in the
        # rosparam /frontier_finder/blocked_regions_runtime so they survive an
        # exploration_node respawn. Within one incarnation that list is exactly
        # right. ACROSS a respawn it is not, because the respawn also wipes the
        # MAP: the rebuilt world has those regions already shadowed, in places
        # the aircraft has not re-explored, so the frontier finder runs out
        # early and FALCON declares the mission complete on a partial map.
        #
        # Measured, and it is the sharpest correlation in the campaign: the run
        # that mapped the whole hospital had ZERO planner crashes and finished
        # at 760 m3; every crashed run finished early -- 355.9, 260.8, 116.1 m3.
        #
        # Deleting the param each tick leaves the in-memory blacklist untouched
        # (it is a C++ member, not this param), so blocking still works exactly
        # as designed while the node lives. It only stops the list being
        # INHERITED by a successor whose map no longer justifies it. At most one
        # second of blocks can survive a crash, which is the width of this loop.
        self._clear_blocked = bool(
            rospy.get_param("~clear_blocked_regions_on_respawn", True))
        self._blocked_param = "/frontier_finder/blocked_regions_runtime"

        self._nudge_pub = rospy.Publisher("/mission/nudge", String, queue_size=1)
        self._abort_pub = rospy.Publisher("/mission/abort", String, queue_size=1,
                                          latch=True)
        self._status_pub = rospy.Publisher("/mission/progress", String, queue_size=1)

        rospy.Subscriber(rospy.get_param("~odom_topic", "/simple_drone/odom"),
                         Odometry, self._on_odom, queue_size=4)
        rospy.Subscriber("/voxel_mapping/map_coverage", Float32,
                         self._on_coverage, queue_size=4)
        rospy.Subscriber("/planning/bspline", Bspline, self._on_bspline,
                         queue_size=4)
        rospy.Subscriber("/map_ros/pose", PoseStamped, self._on_sensor_pose,
                         queue_size=4)
        rospy.Subscriber("/planning/replan", rospy.AnyMsg, self._on_replan,
                         queue_size=10)

        rospy.Timer(rospy.Duration(1.0), self._tick)
        cfg = self._monitor.config
        rospy.loginfo("[watchdog] armed: cap %.0fs, confined under %.1f m gaining "
                      "under %.1f m3/min for %.0fs is an abort; no growth for "
                      "%.0fs or no movement for %.0fs also ends it",
                      cfg.time_cap_s, cfg.confine_radius_m,
                      cfg.growth_m3_per_min, cfg.confine_cap_s,
                      cfg.barren_cap_s, cfg.no_move_cap_s)

    # ── inputs ───────────────────────────────────────────────────────────

    def _on_odom(self, msg):
        # type: (Odometry) -> None
        self._odom = msg

    def _on_coverage(self, msg):
        # type: (Float32) -> None
        self._coverage = float(msg.data)

    def _on_sensor_pose(self, msg):
        # type: (PoseStamped) -> None
        """How stale the pose the MAPPER fuses depth against is.

        A mapper pairing a depth frame with a pose from a different instant
        smears every surface it reconstructs along the flight path, and the
        planner then routes through walls that are not where it thinks. The
        transformer pairs by stamp with a 0.05 s tolerance, so this is a check
        on the tolerance being met, not on the pairing being done.
        """
        if self._odom is None:
            return
        self._pose_lag_s = abs((msg.header.stamp
                                - self._odom.header.stamp).to_sec())

    def _on_bspline(self, msg):
        # type: (Bspline) -> None
        """Measure how far FALCON's plan starts from where the aircraft is.

        This is the number the whole "FALCON thinks it is somewhere else"
        question reduces to. FALCON derives its replan start by evaluating the
        PREVIOUS trajectory at the current time rather than reading odometry
        (``exploration_fsm.cpp``), so every second this stack spends braked,
        held or retreating opens a gap between the planned world and the real
        one -- and the new curve is then anchored in the planned one. In a
        1.4 m warehouse aisle a metre of that is survivable. In a 0.9 m
        hospital doorway it is the whole clearance budget, spent before the
        aircraft has moved.
        """
        self._plans += 1
        if self._odom is None or msg.order != 3:
            return
        try:
            trajectory = BsplineTrajectory.from_falcon(
                order=msg.order,
                knots=list(msg.knots),
                position_points=[(p.x, p.y, p.z) for p in msg.pos_pts],
                yaw_points=list(msg.yaw_pts),
                yaw_dt=msg.yaw_dt,
                start_time_s=msg.start_time.to_sec(),
                traj_id=msg.traj_id)
            start = trajectory.position_at(0.0)
        except Exception as exc:                    # noqa: BLE001 - diagnostics only
            rospy.logwarn_throttle(30.0, "[watchdog] could not evaluate the plan "
                                         "start: %s", exc)
            return
        p = self._odom.pose.pose.position
        gap = math.sqrt((start[0] - p.x) ** 2 + (start[1] - p.y) ** 2
                        + (start[2] - p.z) ** 2)
        self._plan_gap_m = gap
        self._plan_gap_max = max(self._plan_gap_max, gap)
        self._plan_gap_sum += gap
        self._plan_gap_n += 1

    def _on_replan(self, _msg):
        # type: (object) -> None
        """Any replan traffic proves FALCON is alive; ``2`` means it is done.

        Subscribed as ``AnyMsg`` so this node never needs the message package
        to be importable just to notice the mission ended.
        """
        return

    # ── the loop ─────────────────────────────────────────────────────────

    def _clear_blocked_regions(self):
        # type: () -> None
        """Drop the persisted dead-end blacklist; see the note in ``__init__``."""
        if not self._clear_blocked:
            return
        try:
            if rospy.has_param(self._blocked_param):
                rospy.delete_param(self._blocked_param)
        except Exception:                       # noqa: BLE001 - never kill the timer
            pass

    def _tick(self, _event):
        # type: (object) -> None
        self._clear_blocked_regions()
        if self._aborted or self._odom is None:
            return
        now = rospy.Time.now().to_sec()
        p = self._odom.pose.pose.position
        verdict = self._monitor.update(now, (p.x, p.y, p.z), self._coverage)
        record = verdict.as_dict()
        record["plans"] = self._plans
        record["plan_origin_gap_m"] = (None if self._plan_gap_m is None
                                       else round(self._plan_gap_m, 3))
        record["plan_origin_gap_max_m"] = round(self._plan_gap_max, 3)
        record["plan_origin_gap_mean_m"] = (
            None if not self._plan_gap_n
            else round(self._plan_gap_sum / self._plan_gap_n, 3))
        record["sensor_pose_lag_s"] = (None if self._pose_lag_s is None
                                       else round(self._pose_lag_s, 4))
        record["x"], record["y"], record["z"] = (round(p.x, 3), round(p.y, 3),
                                                 round(p.z, 3))
        if self._trace is not None:
            self._trace.write(json.dumps(record) + "\n")
        self._status_pub.publish(String(data=json.dumps(record)))

        if verdict.is_nudge:
            rospy.logwarn("[watchdog] NUDGE %s; asking for a fresh survey",
                          verdict.reason)
            self._nudge_pub.publish(String(data=verdict.reason))
            return
        if verdict.is_abort:
            self._abort(verdict)

    def _abort(self, verdict):
        # type: (object) -> None
        """Declare the mission over, loudly and once.

        The banner is deliberately greppable and deliberately repeated a few
        times: it has to survive a log the FSM is writing forty lines a second
        into, and the harness polls at 10 s.
        """
        self._aborted = True
        self._abort_pub.publish(String(data="%s: %s" % (verdict.state,
                                                        verdict.reason)))
        for _ in range(3):
            rospy.logerr("[watchdog] MISSION ABORT (%s): %s -- coverage %.1f m3 "
                         "after %.0fs, %d plans, plan-origin gap mean %.2f m "
                         "max %.2f m",
                         verdict.state, verdict.reason, verdict.coverage_m3,
                         verdict.elapsed_s, self._plans,
                         (self._plan_gap_sum / self._plan_gap_n)
                         if self._plan_gap_n else 0.0, self._plan_gap_max)
        if self._trace is not None:
            self._trace.flush()


def main():
    # type: () -> None
    """Entry point."""
    rospy.init_node("mission_watchdog")
    MissionWatchdogNode()
    rospy.spin()


if __name__ == "__main__":
    main()
