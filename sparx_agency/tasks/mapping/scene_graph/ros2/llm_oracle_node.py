"""llm_oracle_node — target + labeled rooms -> per-room probabilities.

ROS wiring around :class:`sparx_agency.core.mapping.topology.SearchOracle`
(ported from the SJTU ``llm_oracle_node.py``; the LLM plane — prompt,
clamping, normalization, uniform fallback — lives in core).

Subscribes
----------
``/scene_graph``                 (std_msgs/String JSON, latched)
``/semantic_mapper/room_labels`` (std_msgs/String JSON, latched)
    Rooms missing a label are presented to the oracle as ``"unknown"``.
``/target_seen``                 (std_msgs/Bool, latched)
    True pauses the tick loop permanently — no more LLM spend once found.

Publishes
---------
``/llm_oracle/probabilities``  (std_msgs/String JSON, latched)
    ``{"stamp", "target", "model", "source": "llm"|"uniform_fallback",
    "rooms": [{"id", "label", "prob", "reason", "time_in_room_s",
    "frontier_clusters"}]}`` — the probs sum to 1 (the core oracle
    normalizes, and uniform-fallbacks internally on any LLM trouble).

``target_object`` is runtime-settable::

    ros2 param set /llm_oracle target_object "iv stand"

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.\
llm_oracle_node --ros-args -p use_sim_time:=true -p target_object:=wheelchair
"""
from __future__ import annotations

import json
from typing import Dict, List

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from sparx_agency.core.mapping.topology import (
    LLMClient,
    OracleRoom,
    SearchOracle,
)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import latched_qos


class LLMOracleNode(Node):
    """Periodically asks the search oracle where the target likely is."""

    def __init__(self) -> None:
        super().__init__("llm_oracle")

        self.declare_parameter("target_object", "wheelchair")
        self.declare_parameter("tick_period_s", 10.0)
        self.declare_parameter("scene_graph_topic", "/scene_graph")
        self.declare_parameter("labels_topic", "/semantic_mapper/room_labels")
        self.declare_parameter("out_topic", "/llm_oracle/probabilities")

        gp = self.get_parameter
        self._target = str(gp("target_object").value)
        tick_period_s = max(1.0, float(gp("tick_period_s").value))
        self._scene_topic = str(gp("scene_graph_topic").value)
        self._labels_topic = str(gp("labels_topic").value)
        self._out_topic = str(gp("out_topic").value)

        # Retarget at runtime without relaunching.
        self.add_on_set_parameters_callback(self._on_param_set)

        self._llm = LLMClient.from_env()
        self.get_logger().info(
            "llm_oracle LLM backend=%s model=%s url=%s"
            % (self._llm.cfg.backend, self._llm.cfg.model,
               self._llm.cfg.base_url))
        if not self._llm.ping():
            self.get_logger().warning(
                "LLM server at %s did not answer a ping; the oracle will "
                "uniform-fallback until it comes up." % self._llm.cfg.base_url)
        else:
            self.get_logger().info("LLM ping OK")
        self._oracle = SearchOracle(self._llm)

        self._latest_sg = None
        self._latest_labels: Dict[str, Dict] = {}
        self._stopped = False  # latched by /target_seen=True, permanent
        self._n = dict(ticks=0, publishes=0, fallbacks=0, skips=0)

        latched = latched_qos()
        self.create_subscription(String, self._scene_topic, self._sg_cb,
                                 latched)
        self.create_subscription(String, self._labels_topic, self._labels_cb,
                                 latched)
        self.create_subscription(Bool, "/target_seen", self._target_seen_cb,
                                 latched)
        self._pub = self.create_publisher(String, self._out_topic, latched)

        self.create_timer(tick_period_s, self._tick)
        self.create_timer(10.0, self._heartbeat)

        self.get_logger().info(
            "llm_oracle params  target=%r  tick_period_s=%.1f  in=(%s, %s)  "
            "out=%s"
            % (self._target, tick_period_s, self._scene_topic,
               self._labels_topic, self._out_topic))

    # -- runtime param updates -----------------------------------------
    def _on_param_set(self, params) -> SetParametersResult:
        for p in params:
            if p.name == "target_object":
                new_target = str(p.value)
                if new_target != self._target:
                    self.get_logger().info(
                        "target_object: %r -> %r"
                        % (self._target, new_target))
                    self._target = new_target
        return SetParametersResult(successful=True)

    # -- subscriptions -------------------------------------------------
    def _sg_cb(self, msg: String) -> None:
        try:
            self._latest_sg = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning("bad scene graph JSON: %s" % exc,
                                      throttle_duration_sec=5.0)

    def _labels_cb(self, msg: String) -> None:
        try:
            self._latest_labels = json.loads(msg.data).get("labels", {}) or {}
        except json.JSONDecodeError as exc:
            self.get_logger().warning("bad room labels JSON: %s" % exc,
                                      throttle_duration_sec=5.0)

    def _target_seen_cb(self, msg: Bool) -> None:
        if msg.data and not self._stopped:
            self.get_logger().info(
                "/target_seen=True — pausing LLM oracle permanently.")
            self._stopped = True

    # -- tick ----------------------------------------------------------
    def _tick(self) -> None:
        if self._stopped:
            return
        self._n["ticks"] += 1
        rooms = self._merged_rooms()
        if not rooms:
            self._n["skips"] += 1
            self.get_logger().info(
                "llm_oracle waiting — no rooms in the scene graph yet",
                throttle_duration_sec=30.0)
            return

        oracle_rooms = [
            OracleRoom(
                id=r["id"],
                label=r["label"],
                searched_s=r["time_in_room_s"],
                frontier_clusters=r["frontier_clusters"],
                observed_classes=tuple(r["observed_classes"]),
            )
            for r in rooms
        ]
        result = self._oracle.probabilities(self._target, oracle_rooms)
        if result.source != "llm":
            self._n["fallbacks"] += 1
            self.get_logger().warning(
                "oracle degraded to uniform fallback (LLM down or "
                "unusable reply)", throttle_duration_sec=30.0)

        payload = {
            "stamp": self.get_clock().now().nanoseconds * 1e-9,
            "target": self._target,
            "model": self._llm.cfg.model,
            "source": result.source,
            "rooms": [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "prob": result.probs[r["id"]],
                    "reason": result.reasons.get(r["id"], ""),
                    "time_in_room_s": r["time_in_room_s"],
                    "frontier_clusters": r["frontier_clusters"],
                }
                for r in rooms
            ],
        }
        self._pub.publish(String(data=json.dumps(payload)))
        self._n["publishes"] += 1

        top = sorted(payload["rooms"], key=lambda r: -r["prob"])[:3]
        self.get_logger().info(
            "probs  target=%r source=%s  top: %s"
            % (self._target, result.source,
               ", ".join("R%d(%s)=%.2f" % (r["id"], r["label"], r["prob"])
                         for r in top)))

    def _merged_rooms(self) -> List[Dict]:
        """Latest scene-graph rooms merged with their LLM labels."""
        if self._latest_sg is None:
            return []
        merged = []
        for room in self._latest_sg.get("rooms", []) or []:
            pid = room.get("id")
            if pid is None:
                continue
            label_entry = self._latest_labels.get(str(pid), {})
            objs = room.get("objects", []) or []
            merged.append({
                "id": int(pid),
                "label": str(label_entry.get("label", "unknown")),
                "time_in_room_s": float(room.get("time_in_room_s", 0.0)),
                "frontier_clusters": int(room.get("frontier_clusters", 0)),
                "observed_classes": [str(o.get("class", "")) for o in objs
                                     if o.get("class")],
            })
        return merged

    # -- heartbeat -----------------------------------------------------
    def _heartbeat(self) -> None:
        self.get_logger().info(
            "llm_oracle hb  %s  target=%r  ticks=%d publishes=%d "
            "fallbacks=%d skips=%d"
            % ("PAUSED" if self._stopped else "running", self._target,
               self._n["ticks"], self._n["publishes"], self._n["fallbacks"],
               self._n["skips"]))
        self._n = dict(ticks=0, publishes=0, fallbacks=0, skips=0)


def main() -> None:
    rclpy.init()
    node = LLMOracleNode()
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
