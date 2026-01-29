#!/usr/bin/env python3
"""
Interactive Instruction Console Node

Loads a YAML file containing navigation instructions and provides a simple
interactive CLI:

- Type an integer ID (e.g., 1..20) and press Enter -> publishes that instruction
  text to the instruction topic (std_msgs/String).
- Type 0 and press Enter -> prompts for a free-text instruction, press Enter to
  publish it.
- Type 'q' (or 'quit') -> exits.

Why a queue + timer?
- rclpy is happiest when ROS publishes happen on the ROS thread.
- We read stdin in a background thread and enqueue publish requests.
- A ROS timer flushes the queue and publishes in the main ROS executor thread.

YAML format expected:

scenario: "prison"
language: "en"
instructions:
  - id: 1
    text: "Enter the prison cell on your left and look for the prisoner."
  - id: 2
    text: "Look for a nearby door."

Run (ROS2 Foxy, in container):
  source /opt/ros/foxy/setup.bash
  python3 -m internnav_bridge.instruction_console_node --ros-args \
    -p yaml_path:=config/prison_instructions.yaml \
    -p instruction_topic:=/R1/navigation/instruction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(frozen=True)
class Instruction:
    """Single instruction entry."""
    id: int
    text: str


def _load_yaml(path: str) -> Dict:
    """Load YAML file into a Python dict using PyYAML."""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required. Install with: pip3 install pyyaml") from e

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_instructions(data: Dict) -> Dict[int, Instruction]:
    """Parse {id: Instruction} from YAML dict."""
    items = data.get("instructions", [])
    if not isinstance(items, list):
        raise ValueError("YAML 'instructions' must be a list")

    out: Dict[int, Instruction] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError(f"Instruction entry #{i} must contain 'id' and 'text'")
        ins_id = int(item["id"])
        text = str(item["text"]).strip()
        if not text:
            raise ValueError(f"Instruction id={ins_id} has empty text")
        out[ins_id] = Instruction(id=ins_id, text=text)

    if not out:
        raise ValueError("No instructions found in YAML (instructions list is empty)")
    return out


class InstructionConsoleNode(Node):
    """Interactive console that publishes instructions to a ROS2 String topic."""

    def __init__(self) -> None:
        super().__init__("instruction_console")

        # Parameters
        self.declare_parameter("yaml_path", "")
        self.declare_parameter("instruction_topic", "/R1/navigation/instruction")
        self.declare_parameter("print_menu_every_sec", 0.0)  # 0 = only once

        yaml_path = str(self.get_parameter("yaml_path").value).strip()
        self._instruction_topic = str(self.get_parameter("instruction_topic").value).strip()
        self._menu_repeat = float(self.get_parameter("print_menu_every_sec").value)

        if not yaml_path:
            raise RuntimeError(
                "Missing required param 'yaml_path'. Example:\n"
                "  --ros-args -p yaml_path:=config/prison_instructions.yaml"
            )

        data = _load_yaml(yaml_path)
        self._instructions = _parse_instructions(data)

        self._pub = self.create_publisher(String, self._instruction_topic, 10)

        # Thread-safe queue for publish requests
        self._queue_lock = threading.Lock()
        self._publish_queue: List[str] = []

        # Timer to flush queue (publish from ROS thread)
        self._flush_timer = self.create_timer(0.05, self._flush_queue)

        # Start console thread
        self._stop_event = threading.Event()
        self._console_thread = threading.Thread(target=self._console_loop, daemon=True)
        self._console_thread.start()

        ids_sorted = sorted(self._instructions.keys())
        self.get_logger().info(f"Loaded {len(ids_sorted)} instructions.")
        self.get_logger().info(f"Publishing to: {self._instruction_topic}")
        self._print_menu(ids_sorted)

    def _enqueue_publish(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._queue_lock:
            self._publish_queue.append(text)

    def _flush_queue(self) -> None:
        """Publish queued messages (ROS thread)."""
        batch: List[str] = []
        with self._queue_lock:
            if self._publish_queue:
                batch = self._publish_queue[:]
                self._publish_queue.clear()

        for text in batch:
            msg = String()
            msg.data = text
            self._pub.publish(msg)
            self.get_logger().info(f"Published instruction: {text}")

    def _print_menu(self, ids_sorted: List[int]) -> None:
        # Print to stdout (not ROS logger) so it looks like a clean CLI.
        print("\n=== Instruction Console ===")
        print("Type an instruction ID and press Enter (e.g., 1).")
        print("Type 0 to enter free-text instruction.")
        print("Type q to quit.\n")
        print(f"Available IDs: {ids_sorted}\n")

    def _console_loop(self) -> None:
        """Blocking stdin loop (background thread)."""
        ids_sorted = sorted(self._instructions.keys())
        last_menu_t = 0.0

        # Print menu periodically if requested
        def maybe_repeat_menu():
            nonlocal last_menu_t
            if self._menu_repeat <= 0:
                return
            now = time.time()
            if now - last_menu_t >= self._menu_repeat:
                self._print_menu(ids_sorted)
                last_menu_t = now

        last_menu_t = time.time()

        while not self._stop_event.is_set():
            try:
                maybe_repeat_menu()
                s = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                s = "q"

            if not s:
                continue

            lower = s.lower()
            if lower in ("q", "quit", "exit"):
                print("Exiting instruction console...")
                self._stop_event.set()
                # Trigger ROS shutdown from thread-safe context
                rclpy.try_shutdown()
                return

            # Free text mode
            if s == "0":
                try:
                    txt = input("Enter instruction text (then press Enter):\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                if txt:
                    self._enqueue_publish(txt)
                continue

            # ID mode
            try:
                ins_id = int(s)
            except ValueError:
                print("Invalid input. Enter an ID number (e.g., 1), 0 for free-text, or q to quit.")
                continue

            ins: Optional[Instruction] = self._instructions.get(ins_id)
            if ins is None:
                print(f"Unknown ID={ins_id}. Available IDs: {ids_sorted}")
                continue

            self._enqueue_publish(ins.text)

    def destroy_node(self) -> None:
        self._stop_event.set()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = InstructionConsoleNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
