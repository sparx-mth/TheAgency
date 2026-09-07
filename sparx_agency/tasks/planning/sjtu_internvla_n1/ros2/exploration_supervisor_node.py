#!/usr/bin/env python3
"""Hold the survey, and keep handing the policy one small order it can fly.

The layer above System 1 and System 2, and its entire interface to them is one
string on one topic. It subscribes to odometry and to the policy's own status,
decides which room to go to next, and republishes
``/simple_drone/navigation/instruction``. **The flight loop is not touched, not
imported, and does not know this node exists** -- the policy node re-reads its
instruction on every decision and substitutes it fresh into the model's prompt,
so changing it mid-flight is already supported and already used by hand.

Why it exists is in ``core/planning/exploration/mission.py``: under one
open-ended order, five recorded flights saw 9-16 % of this building and four of
them stopped themselves part-way through. Under a sequence of bounded ones,
``STOP`` stops meaning "the flight is over" and starts meaning "that one is
done", which is a claim the policy is far better placed to make.

It keeps its own coverage tracker rather than borrowing the recorder's. That is
a deliberate second copy of a cheap computation -- 1.8 ms per pose at 5 Hz -- and
it is the right one: the recorder only runs with ``record:=true``, the two
processes are independent, and a supervisor whose survey state arrived over a
topic would go blind whenever the recorder was not started.

CPU-only, like every node here.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.core.planning.environment.occupancy_io import occupancy_from_mask
from sparx_agency.core.planning.exploration.briefing import BriefingStyle, brief
from sparx_agency.core.planning.exploration.mission import (
    SURVEY_COMPLETE,
    ExplorationSupervisor,
    SupervisorParams,
)
from sparx_agency.core.planning.exploration.region_coverage import RegionCoverage
from sparx_agency.core.planning.exploration.region_map import load_region_map
from sparx_agency.core.planning.exploration.survey_state import (
    load_survey,
    save_survey,
)
from sparx_agency.core.planning.exploration.visibility_coverage import (
    VisibilityCoverage,
    cone_from_intrinsics,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
    load_map_backdrop,
)

# <repo>/sparx_agency/tasks/planning/sjtu_internvla_n1/ros2/<this file>
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))


def _load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(path):
    """A config path, relative to the repo root like every other path here."""
    if path and not os.path.isabs(path):
        return os.path.join(_REPO_ROOT, path)
    return path


class ExplorationSupervisorNode(Node):
    """Drive the survey by rewriting the instruction the policy is flying."""

    def __init__(self):
        super().__init__("exploration_supervisor_node")
        self.declare_parameter("config_file", "")
        # The experiment, as launch arguments: same supervisor, less context.
        self.declare_parameter("include_location", True)
        self.declare_parameter("goal_only", False)
        cfg = _load_config(self.get_parameter("config_file").value)
        topics = cfg.get("topics", {})
        rec = cfg.get("recorder", {})
        sup = cfg.get("supervisor", {})

        self._style = BriefingStyle(
            include_location=bool(self.get_parameter("include_location").value),
            goal_only=bool(self.get_parameter("goal_only").value))

        self._lock = threading.Lock()
        self._pose = None
        self._stop_hint = False
        self._busy = False
        self._published = ""
        self._last_publish_s = 0.0
        self._republish_s = float(sup.get("republish_s", 5.0))
        self._observe_period_s = 1.0 / max(1e-3, float(sup.get("observe_rate_hz", 5.0)))
        self._last_observe_s = 0.0
        self._finished_logged = False
        # WHERE THE SURVEY LIVES BETWEEN FLIGHTS. A hospital is bigger than one
        # flight and a capsize ends a flight outright, so without this a run
        # that flips at minute nine throws nine minutes of building away. The
        # campaign harness restarts the world and re-ferries between runs; this
        # is what makes run n+1 continue run n.
        self._state_file = _resolve(sup.get("state_file", ""))
        self._save_every_s = float(sup.get("save_every_s", 20.0))
        self._last_save_s = 0.0

        camera = cfg.get("camera", {})
        coverage_cfg = rec.get("coverage", {})
        self._supervisor, self._coverage, self._region_map = self._build(
            rec, sup, camera, coverage_cfg)

        instruction_topic = topics.get("instruction",
                                       "/simple_drone/navigation/instruction")
        odom_topic = topics.get("odom", "/simple_drone/odom")
        info_topic = topics.get("info", "/simple_drone/n1/info")
        mission_topic = topics.get("mission", "/simple_drone/n1/mission")
        nudge_topic = topics.get("nudge", "/simple_drone/n1/nudge_back")

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        # The policy node subscribes to the instruction VOLATILE with depth 1,
        # so a message published before it is listening is simply gone. Publish
        # RELIABLE and re-send periodically rather than trusting one shot.
        self._instruction_pub = self.create_publisher(String, instruction_topic, 10)
        self._mission_pub = self.create_publisher(String, mission_topic, latched)
        # Asking the follower to break contact. It runs the manoeuvre it already
        # has for the depth reflex; this is a second trigger, not a second
        # escape, and it is the only thing this node ever asks of the aircraft
        # other than by rewriting the instruction.
        self._nudge_pub = self.create_publisher(Empty, nudge_topic, 1)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(String, info_topic, self._on_info, latched)

        if self._supervisor is not None and self._state_file:
            try:
                load_survey(self._state_file, self._coverage, self._supervisor,
                            logger=self.get_logger())
            except ValueError as exc:
                # Refuse rather than start again quietly: a mask laid over the
                # wrong building would make every number downstream wrong.
                self.get_logger().error("cannot resume the survey: %s" % (exc,))
                raise
        self.create_timer(1.0 / max(1e-3, float(sup.get("rate_hz", 2.0))), self._tick)
        self.get_logger().info(
            "exploration supervisor up: %d rooms to clear, publishing %s"
            % (len(self._region_map.rooms()) if self._region_map else 0,
               instruction_topic))

    # ── construction ─────────────────────────────────────────────────
    def _build(self, rec, sup, camera, coverage_cfg):
        """The region map, the coverage tracker and the supervisor over them.

        Returns ``(None, None, None)`` when the building has not been decomposed:
        without a region map there is no checklist, and a supervisor that
        invented one would be worse than none at all.
        """
        region_path = _resolve(sup.get("region_map", ""))
        self._region_map_path = region_path
        region_map = load_region_map(region_path, logger=self.get_logger())
        if region_map is None:
            self.get_logger().warn(
                "no region map configured (supervisor.region_map); this node "
                "will not touch the instruction")
            return (None, None, None)
        backdrop = load_map_backdrop(_resolve(rec.get("map", "")),
                                     logger=self.get_logger())
        if backdrop is None:
            self.get_logger().warn("no occupancy map; the supervisor cannot measure")
            return (None, None, None)
        grid = occupancy_from_mask(backdrop.occupied_mask, backdrop.resolution,
                                   backdrop.origin_x, backdrop.origin_y,
                                   known=backdrop.known_mask)
        cone = cone_from_intrinsics(
            width=int(camera.get("width", 600)),
            fx=float(camera.get("fx", 390.642735)),
            max_range_m=float(coverage_cfg.get("max_range_m", 10.0)),
            forward_offset_m=float(coverage_cfg.get("camera_forward_m", 0.2)))
        coverage = VisibilityCoverage(grid, cone)
        params = SupervisorParams(
            scanned_fraction=float(sup.get("scanned_fraction", 0.60)),
            doorway_radius_m=float(sup.get("doorway_radius_m", 1.2)),
            min_portal_m=float(sup.get("min_portal_m", 0.80)),
            mission_timeout_s=float(sup.get("mission_timeout_s", 75.0)),
            defer_s=float(sup.get("defer_s", 180.0)),
            scan_stall_s=float(sup.get("scan_stall_s", 25.0)),
            stop_hint_min_fraction=float(sup.get("stop_hint_min_fraction", 0.35)),
            max_attempts=int(sup.get("max_attempts", 3)),
            approach_offset_m=float(sup.get("approach_offset_m", 1.6)),
            bearing_hold_s=float(sup.get("bearing_hold_s", 20.0)),
            rescan_radius_m=float(sup.get("rescan_radius_m", 9.0)),
            travel_step_m=float(sup.get("travel_step_m", 10.0)),
            travel_arrive_m=float(sup.get("travel_arrive_m", 2.0)),
            travel_timeout_s=float(sup.get("travel_timeout_s", 60.0)),
            refuse_after_s=float(sup.get("refuse_after_s", 12.0)),
            refuse_radius_m=float(sup.get("refuse_radius_m", 4.0)),
            in_frame_deg=float(sup.get("in_frame_deg", 35.0)),
            turn_cost_m_per_deg=float(
                sup.get("turn_cost_m_per_deg", 0.4)),
            max_issues_multiple=int(sup.get("max_issues_multiple", 4)),
            nudge_after_s=float(sup.get("nudge_after_s", 35.0)),
            nudge_min_move_m=float(sup.get("nudge_min_move_m", 0.6)),
            nudge_cooldown_s=float(sup.get("nudge_cooldown_s", 25.0)))
        region_coverage = RegionCoverage(region_map, coverage.countable_mask,
                                         scanned_fraction=params.scanned_fraction)
        return (ExplorationSupervisor(region_map, region_coverage, params),
                coverage, region_map)

    # ── subscriptions ────────────────────────────────────────────────
    def _on_odom(self, msg):
        p = msg.pose.pose
        q = p.orientation
        pose = (p.position.x, p.position.y,
                yaw_from_quaternion((q.x, q.y, q.z, q.w)))
        with self._lock:
            self._pose = pose

    def _on_info(self, msg):
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            # A STOP is the policy's claim that the CURRENT order is finished.
            # It is a hint only: the supervisor corroborates it against the map
            # before ticking anything off.
            self._stop_hint = bool(data.get("stop"))
            # The aircraft is meant to be still in these, and a nudge would
            # interrupt the very thing the mission needs: it is rotating to
            # look, dipping for a look-down, or settling out of one.
            #
            # "THINKING" IS DELIBERATELY NOT ON THIS LIST, and it used to be.
            # At 0.2-0.3 Hz, thinking is not a state the aircraft passes
            # through -- it is the state it is normally in. Measured over one
            # 90-minute flight it was 79.2% of every status published, and
            # during the 50-minute stall it was 92.1%: counting it as busy
            # suppressed the nudge for 99.7% of exactly the stretch the nudge
            # exists for, and two nudges fired in fifty motionless minutes.
            #
            # Nor does backing off during a think undo anything. The other
            # three are changing the view on purpose and a reverse would waste
            # that; a think is the model staring at a frame, and a slightly
            # different frame is the point.
            self._busy = str(data.get("phase") or "") in (
                "settling", "turning", "dipping")

    # ── the loop ─────────────────────────────────────────────────────
    def _tick(self):
        if self._supervisor is None:
            return
        try:
            self._step()
        except Exception as exc:  # noqa: BLE001
            # Never take the flight down. The aircraft keeps flying whatever
            # order it last had, which is a degraded run rather than a lost one.
            self.get_logger().error("supervisor stopped after an error: %s" % (exc,))
            self._supervisor = None

    def _step(self):
        with self._lock:
            pose = self._pose
            stop_hint = self._stop_hint
            busy = self._busy
            self._stop_hint = False
        if pose is None:
            return
        now = time.monotonic()
        if (now - self._last_observe_s) >= self._observe_period_s:
            self._last_observe_s = now
            self._coverage.observe(pose[0], pose[1], pose[2])

        state = self._supervisor.update(pose[0], pose[1], pose[2],
                                        self._coverage.seen_mask, now,
                                        stop_hint=stop_hint, busy=busy)
        if state.completed is not None:
            verdict = self._supervisor.history[-1][1]
            self.get_logger().info(
                "MISSION %s: %s (%s)"
                % (state.completed.kind, verdict, state.completed.note))
        if state.nudge:
            self.get_logger().warn(
                "NUDGE: %s has not moved for %.0f s; asking for a short reverse"
                % (state.mission.kind if state.mission else "the mission",
                   self._supervisor.params.nudge_after_s))
            self._nudge_pub.publish(Empty())
        if state.mission is None:
            return

        text = brief(state, self._region_map, self._style)
        if not text:
            return
        if state.changed:
            self.get_logger().info(
                "MISSION %s -> %s | %s | %d/%d rooms, %.1f%% seen"
                % (state.mission.kind, state.mission.note, state.topo,
                   state.rooms_scanned, state.rooms_total,
                   100.0 * state.fraction_seen))
            self.get_logger().info("INSTRUCTION %s" % (text,))
        self._publish_mission(state, text)

        if state.mission.kind == SURVEY_COMPLETE and not self._finished_logged:
            self._finished_logged = True
            self.get_logger().info(
                "SURVEY COMPLETE: %d of %d rooms, %.1f%% of the building seen"
                % (state.rooms_scanned, state.rooms_total,
                   100.0 * state.fraction_seen))

        if text != self._published or (now - self._last_publish_s) >= self._republish_s:
            self._published = text
            self._last_publish_s = now
            self._instruction_pub.publish(String(data=text))

        if self._state_file and (now - self._last_save_s) >= self._save_every_s:
            self._last_save_s = now
            self._save()

    def _save(self):
        """Write the survey out. Never fatal -- a lost save is a lost segment."""
        try:
            save_survey(self._state_file, self._coverage, self._supervisor,
                        extra={"region_map": self._region_map_path})
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn("could not save the survey: %s" % (exc,))

    def _publish_mission(self, state, text):
        """What the supervisor is doing, for the recorder and the log."""
        payload = {
            "instruction": text,
            "topo": state.topo,
            "region": state.region.name if state.region else None,
            "mission": state.mission.kind,
            "target": (self._region_map.regions[state.mission.target_id].name
                       if state.mission.target_id in self._region_map.regions
                       else None),
            "bearing": state.bearing,
            "range_m": state.range_m,
            "rooms_scanned": state.rooms_scanned,
            "rooms_total": state.rooms_total,
            "fraction_seen": state.fraction_seen,
        }
        self._mission_pub.publish(String(data=json.dumps(payload)))


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationSupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # BOTH, and the second is the one that matters. `ros2 launch` stops its
        # children by shutting the context down under them, which rclpy raises
        # out of spin as ExternalShutdownException -- not KeyboardInterrupt. Left
        # uncaught it exits 1, and the launch file's `on_exit=Shutdown()` then
        # reports a perfectly orderly teardown as "process has died", which is
        # indistinguishable in the log from the node crashing mid-flight.
        pass
    finally:
        # One last write, so the segment the aircraft just flew is not lost to
        # whichever teardown got here first.
        try:
            if getattr(node, "_supervisor", None) is not None and node._state_file:
                node._save()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
