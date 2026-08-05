#!/usr/bin/env python3
"""manual_flight_logger.py

Logs every /{rooster_id}/cmd_nav command (whatever issues it -- the UI, a
script, FALCON's twist adapter, anything) alongside the ground-truth pose
stream, to build a command->actual-path dataset for debugging. Three JSONL
files per run, one row per event, so everything is trivially joinable on
timestamp afterward:

  commands.jsonl  {t, action, value, axes}            -- one row per cmd_nav message
  pose.jsonl      {t, x, y, z, yaw}                    -- one row per /localization tick
                  (our own transformed pose -- sign-flipped from Sphera's raw
                  telemetry, see rooster_ground_truth_localization.py)
  raw_pose.jsonl  {t, x, y, z, vx, vy, vz, yaw}         -- one row per
                  /{rooster_id}/sphera/state tick: Sphera's OWN ground-truth
                  telemetry, untransformed. vx/vy/vz are true physics-engine
                  linear velocity (not derived by differencing position),
                  useful for isolating control-vs-perception bugs without
                  going through our own sign-flip pipeline at all.

Read-only with respect to flight: never publishes anything, only subscribes.
"""
import argparse
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sphera_common_interfaces.msg import SpheraPawnState


def _yaw_from_quat(q):
    # planar rotation only (x=y=0), matches rooster_ground_truth_localization.py's encoding
    return 2.0 * math.atan2(q.z, q.w)


class ManualFlightLogger(Node):
    def __init__(self, rooster_id, out_dir):
        super().__init__("manual_flight_logger")
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._cmd_f = open(os.path.join(out_dir, "commands.jsonl"), "a", buffering=1)
        self._pose_f = open(os.path.join(out_dir, "pose.jsonl"), "a", buffering=1)
        self._raw_f = open(os.path.join(out_dir, "raw_pose.jsonl"), "a", buffering=1)

        self.create_subscription(
            String, f"/{rooster_id}/cmd_nav", self._on_cmd, 10)
        self.create_subscription(
            PoseStamped, f"/{rooster_id}/localization", self._on_pose, 10)
        self.create_subscription(
            SpheraPawnState, f"/{rooster_id}/sphera/state", self._on_raw, 10)

        self.get_logger().info(
            f"manual_flight_logger ready for {rooster_id}\n"
            f"  commands -> {self._cmd_f.name}\n"
            f"  pose     -> {self._pose_f.name}\n"
            f"  raw_pose -> {self._raw_f.name}"
        )

    def _on_cmd(self, msg: String):
        row = {"t": round(time.time(), 3)}
        try:
            row.update(json.loads(msg.data))
        except (json.JSONDecodeError, TypeError):
            row["raw"] = msg.data
        self._cmd_f.write(json.dumps(row) + "\n")

    def _on_pose(self, msg: PoseStamped):
        p, o = msg.pose.position, msg.pose.orientation
        row = {
            "t": round(time.time(), 3),
            "x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4),
            "yaw": round(_yaw_from_quat(o), 5),
        }
        self._pose_f.write(json.dumps(row) + "\n")

    def _on_raw(self, msg: SpheraPawnState):
        loc, vel = msg.location, msg.velocity
        row = {
            "t": round(time.time(), 3),
            "x": round(loc.x, 4), "y": round(loc.y, 4), "z": round(loc.z, 4),
            "vx": round(vel.x, 4), "vy": round(vel.y, 4), "vz": round(vel.z, 4),
            "yaw": round(msg.rotation.yaw, 5),
        }
        self._raw_f.write(json.dumps(row) + "\n")

    def destroy_node(self):
        self._cmd_f.close()
        self._pose_f.close()
        self._raw_f.close()
        super().destroy_node()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--out-dir", default=os.path.expanduser("~/rooster_manual_flight_logs"))
    args = p.parse_args()

    rclpy.init()
    node = ManualFlightLogger(args.rooster_id, args.out_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
