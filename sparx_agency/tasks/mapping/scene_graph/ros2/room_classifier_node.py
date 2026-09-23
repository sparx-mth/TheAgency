"""room_classifier_node — objects per room -> LLM -> room type labels.

ROS wiring around :class:`sparx_agency.core.mapping.topology.
RoomTypeClassifier` (ported from the SJTU ``room_classifier_node.py``; the
LLM plane — prompts, signature cache, label coercion — lives in core).

Subscribes
----------
``/scene_graph``  (std_msgs/String JSON, latched)
    Each room carries its objects list ``{"id", "objects": [{"class", ...}]}``.
``/target_seen``  (std_msgs/Bool, latched)
    True stops the tick loop permanently — no more LLM spend once found.

Publishes
---------
``/semantic_mapper/room_labels``  (std_msgs/String JSON, latched)
    ``{"stamp": float, "labels": {"<room id>": {"label", "confidence",
    "reasoning"}}}`` — re-published only when the label map changes.
    Labels of vanished rooms are dropped.

The classifier's signature cache (frozenset of a room's object classes)
makes repeat ticks free; an LLM failure for a room is logged and the
previous label kept (raise-then-catch, never a silent stale cache entry).

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.\
room_classifier_node --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import json
from typing import Dict

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from sparx_agency.core.mapping.topology import (
    DEFAULT_LABEL_SET,
    LLMClient,
    RoomTypeClassifier,
)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import latched_qos


class RoomClassifierNode(Node):
    """Ticks over the latest scene graph and labels each room via the LLM."""

    def __init__(self) -> None:
        super().__init__("room_classifier")

        self.declare_parameter("scene_graph_topic", "/scene_graph")
        self.declare_parameter("out_topic", "/semantic_mapper/room_labels")
        self.declare_parameter("tick_rate_hz", 1.0)
        self.declare_parameter("min_objects_for_call", 1)
        self.declare_parameter("room_label_set", DEFAULT_LABEL_SET)

        gp = self.get_parameter
        self._scene_topic = str(gp("scene_graph_topic").value)
        self._out_topic = str(gp("out_topic").value)
        tick_rate_hz = max(0.05, float(gp("tick_rate_hz").value))
        min_objects = int(gp("min_objects_for_call").value)
        label_set = [str(s) for s in gp("room_label_set").value]

        llm = LLMClient.from_env()
        self.get_logger().info(
            "room_classifier LLM backend=%s model=%s url=%s"
            % (llm.cfg.backend, llm.cfg.model, llm.cfg.base_url))
        if not llm.ping():
            # Liveness note, not an exit — the server may come up later
            # and every tick retries.
            self.get_logger().warning(
                "LLM server at %s did not answer a ping; will keep "
                "retrying on ticks." % llm.cfg.base_url)
        else:
            self.get_logger().info("LLM ping OK")
        self._classifier = RoomTypeClassifier(llm, label_set=label_set,
                                              min_objects=min_objects)

        self._latest_sg = None
        self._stopped = False  # latched by /target_seen=True, permanent
        self._labels: Dict[str, Dict] = {}
        self._n = dict(ticks=0, errors=0, publishes=0)

        latched = latched_qos()
        self.create_subscription(String, self._scene_topic, self._sg_cb,
                                 latched)
        self.create_subscription(Bool, "/target_seen", self._target_seen_cb,
                                 latched)
        self._pub = self.create_publisher(String, self._out_topic, latched)

        self.create_timer(1.0 / tick_rate_hz, self._tick)
        self.create_timer(10.0, self._heartbeat)

        self.get_logger().info(
            "room_classifier params  in=%s  out=%s  tick_rate_hz=%.2f  "
            "min_objects=%d  labels=%d"
            % (self._scene_topic, self._out_topic, tick_rate_hz,
               min_objects, len(label_set)))

    # -- subscriptions -------------------------------------------------
    def _sg_cb(self, msg: String) -> None:
        try:
            self._latest_sg = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning("bad scene graph JSON: %s" % exc,
                                      throttle_duration_sec=5.0)

    def _target_seen_cb(self, msg: Bool) -> None:
        if msg.data and not self._stopped:
            self.get_logger().info(
                "/target_seen=True — stopping room classifier permanently.")
            self._stopped = True

    # -- tick ----------------------------------------------------------
    def _tick(self) -> None:
        if self._stopped or self._latest_sg is None:
            return
        self._n["ticks"] += 1
        rooms = self._latest_sg.get("rooms", []) or []
        dirty = False
        for room in rooms:
            pid = str(room.get("id"))
            objs = room.get("objects", []) or []
            classes = [str(o.get("class", "")) for o in objs
                       if o.get("class")]
            try:
                verdict = self._classifier.classify(classes)
            except Exception as exc:  # LLM transport/parse failure
                self._n["errors"] += 1
                self.get_logger().error(
                    "LLM classify failed for room %s (%d objs): %s — "
                    "keeping previous label" % (pid, len(classes), exc),
                    throttle_duration_sec=5.0)
                continue
            new = {"label": verdict.label,
                   "confidence": verdict.confidence,
                   "reasoning": verdict.reasoning}
            if self._labels.get(pid) != new:
                self._labels[pid] = new
                dirty = True

        # Drop labels for rooms that vanished (registry resets etc.).
        live_ids = {str(r.get("id")) for r in rooms}
        for gone in [k for k in self._labels if k not in live_ids]:
            self._labels.pop(gone, None)
            dirty = True

        if dirty:
            payload = {
                "stamp": self.get_clock().now().nanoseconds * 1e-9,
                "labels": self._labels,
            }
            self._pub.publish(String(data=json.dumps(payload)))
            self._n["publishes"] += 1

    # -- heartbeat -----------------------------------------------------
    def _heartbeat(self) -> None:
        self.get_logger().info(
            "room_classifier hb  %s  ticks=%d publishes=%d errors=%d  "
            "labeled_rooms=%d sig_cache=%d"
            % ("STOPPED" if self._stopped else "running",
               self._n["ticks"], self._n["publishes"], self._n["errors"],
               len(self._labels), self._classifier.cache_size))
        self._n = dict(ticks=0, errors=0, publishes=0)


def main() -> None:
    rclpy.init()
    node = RoomClassifierNode()
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
