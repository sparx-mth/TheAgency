"""target_watcher_node — watches confirmed objects, latches /target_seen.

ROS wiring around :class:`sparx_agency.core.mapping.topology.TargetMatcher`
(ported from the SJTU ``target_watcher_node.py``). The old node also blasted
a zero-Twist ``cmd_vel`` burst to halt the drone; that is deliberately
DROPPED here — FALCON owns flight in this deployment, and the latched
``/target_seen`` Bool is the only stop signal. Whoever wants the aircraft to
react subscribes to it.

Subscribes
----------
``/perception/objects``  (std_msgs/String JSON, latched)
    Confirmed landmarks ``{"objects": [{"id", "class", "xy", "count"}]}``.
    Every object id is checked exactly ONCE against the target (the match
    ladder in core caches (target, class) verdicts, so repeated classes
    never re-query the LLM).

Publishes
---------
``/target_seen``       (std_msgs/Bool, latched) — False at startup; flips
    True permanently on the first match. Changing ``target_object`` at
    runtime re-checks existing objects but never un-latches.
``/target_seen/info``  (std_msgs/String JSON, latched)
    ``{"stamp", "target", "matched_class", "object_id", "xy", "count",
    "reason"}`` — emitted once, when the latch flips.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.\
target_watcher_node --ros-args -p use_sim_time:=true -p target_object:=wheelchair
"""
from __future__ import annotations

import json
from typing import Set

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from sparx_agency.core.mapping.topology import (
    LLMClient,
    MatchResult,
    TargetMatcher,
)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import latched_qos


class TargetWatcherNode(Node):
    """First confirmed object matching the target latches /target_seen."""

    def __init__(self) -> None:
        super().__init__("target_watcher")

        self.declare_parameter("target_object", "wheelchair")
        self.declare_parameter("objects_topic", "/perception/objects")
        self.declare_parameter("use_llm", True)

        gp = self.get_parameter
        self._target = str(gp("target_object").value)
        self._objects_topic = str(gp("objects_topic").value)
        self._use_llm = bool(gp("use_llm").value)

        self.add_on_set_parameters_callback(self._on_param_set)

        client = None
        if self._use_llm:
            client = LLMClient.from_env()
            self.get_logger().info(
                "target_watcher LLM backend=%s model=%s url=%s"
                % (client.cfg.backend, client.cfg.model, client.cfg.base_url))
            if not client.ping():
                self.get_logger().warning(
                    "LLM at %s unreachable; the matcher will use the "
                    "offline token-overlap fallback." % client.cfg.base_url)
        self._matcher = TargetMatcher(client=client, use_llm=self._use_llm)

        self._seen = False
        self._checked_ids: Set[int] = set()
        self._n = dict(objects=0, checks=0)

        latched = latched_qos()
        self._pub_seen = self.create_publisher(Bool, "/target_seen", latched)
        self._pub_info = self.create_publisher(String, "/target_seen/info",
                                               latched)
        # Deterministic startup state for late (latched) subscribers.
        self._pub_seen.publish(Bool(data=False))

        self.create_subscription(String, self._objects_topic,
                                 self._objects_cb, latched)
        self.create_timer(10.0, self._heartbeat)

        self.get_logger().info(
            "target_watcher params  target=%r  objects_topic=%s  use_llm=%s"
            % (self._target, self._objects_topic, self._use_llm))

    # -- runtime param updates -----------------------------------------
    def _on_param_set(self, params) -> SetParametersResult:
        for p in params:
            if p.name == "target_object":
                new_target = str(p.value)
                if new_target != self._target:
                    self.get_logger().info(
                        "target_object: %r -> %r  (re-checking objects; "
                        "the seen latch does NOT reset)"
                        % (self._target, new_target))
                    self._target = new_target
                    self._checked_ids.clear()
        return SetParametersResult(successful=True)

    # -- objects subscription ------------------------------------------
    def _objects_cb(self, msg: String) -> None:
        if self._seen:
            return  # idempotent — first match wins, latch never resets
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning("bad objects JSON: %s" % exc,
                                      throttle_duration_sec=5.0)
            return
        objects = data.get("objects", []) or []
        self._n["objects"] = len(objects)
        for obj in objects:
            try:
                oid = int(obj.get("id", -1))
            except (TypeError, ValueError):
                continue
            cname = str(obj.get("class", "")).strip().lower()
            if oid < 0 or not cname or oid in self._checked_ids:
                continue
            self._checked_ids.add(oid)
            self._n["checks"] += 1
            result = self._matcher.matches(self._target, cname)
            self.get_logger().info(
                "match check  target=%r class=%r -> %s  (%s)"
                % (self._target, cname, result.match, result.reason))
            if result.match:
                self._on_match(obj, cname, result)
                return

    # -- match handler -------------------------------------------------
    def _on_match(self, obj: dict, cname: str, result: MatchResult) -> None:
        self._seen = True
        xy = obj.get("xy", [0.0, 0.0])
        info = {
            "stamp": self.get_clock().now().nanoseconds * 1e-9,
            "target": self._target,
            "matched_class": cname,
            "object_id": int(obj.get("id", -1)),
            "xy": [float(xy[0]), float(xy[1])],
            "count": int(obj.get("count", 0)),
            "reason": result.reason,
        }

        bar = "=" * 64
        self.get_logger().info(bar)
        self.get_logger().info(
            "  TARGET FOUND   target=%r  matched class=%r  obj_id=%d"
            % (self._target, cname, info["object_id"]))
        self.get_logger().info(
            "  world XY = (%.2f, %.2f)   confirmations=%d"
            % (info["xy"][0], info["xy"][1], info["count"]))
        if result.reason:
            self.get_logger().info("  reason: %s" % result.reason)
        self.get_logger().info(
            "  -> latching /target_seen=True (FALCON owns flight; "
            "no halt is commanded here)")
        self.get_logger().info(bar)

        self._pub_seen.publish(Bool(data=True))
        self._pub_info.publish(String(data=json.dumps(info)))

    # -- heartbeat -----------------------------------------------------
    def _heartbeat(self) -> None:
        self.get_logger().info(
            "target_watcher hb  %s  target=%r  objects=%d checks=%d "
            "checked_ids=%d match_cache=%d"
            % ("SEEN" if self._seen else "watching", self._target,
               self._n["objects"], self._n["checks"],
               len(self._checked_ids), self._matcher.cache_size))
        self._n = dict(objects=self._n["objects"], checks=0)


def main() -> None:
    rclpy.init()
    node = TargetWatcherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT and SIGTERM. stop_scene_graph.sh sends SIGTERM,
        # which rclpy turns into ExternalShutdownException out of
        # spin() -- uncaught it printed a traceback on every clean
        # teardown and exited non-zero, so a normal stop read as a
        # crash in the node log.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
