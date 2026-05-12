#!/usr/bin/env python3
"""
twist_replayer.py — ROS2 node that replays a recorded JSONL log of
geometry_msgs/Twist commands onto the same topic name, preserving the
original inter-message timing.

The log format is the one produced by waypoint_follower.py — one JSON
object per line, e.g.:

    {"t": 1778582400.123,
     "linear":  {"x": 0.30, "y": 0.0, "z": 0.0},
     "angular": {"x": 0.0,  "y": 0.0, "z": 0.0}}

Parameters (all ROS2 params):
    log_path   (str)    required — path to the .jsonl file
    topic      (str)    default "/cmd_vel"
    speed      (double) default 1.0   (2.0 = play at 2x real-time)
    loop       (bool)   default False (restart at EOF if True)

Run examples:
    python3 twist_replayer.py --ros-args \
        -p log_path:=/data/cmd_log_20260512_104530.jsonl

    python3 twist_replayer.py --ros-args \
        -p log_path:=/data/cmd_log.jsonl  -p speed:=2.0  -p loop:=true
"""

import json
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy


class TwistReplayer(Node):
    def __init__(self):
        super().__init__('twist_replayer')

        self.declare_parameter('log_path', 'cmd_log_20260512_113700 1.jsonl')
        self.declare_parameter('topic',    '/cmd_vel')
        self.declare_parameter('speed',    1.0) # 1= 20Hz 0.25 = 5Hz
        self.declare_parameter('loop',     False)

        log_path  = self.get_parameter('log_path').value
        topic     = self.get_parameter('topic').value
        self.spd  = float(self.get_parameter('speed').value)
        self.loop = bool (self.get_parameter('loop').value)

        if not log_path:
            self.get_logger().error("required parameter 'log_path' is empty")
            raise RuntimeError("log_path required")
        if self.spd <= 0:
            raise RuntimeError("speed must be > 0")

        # Load all entries
        with open(log_path) as f:
            self.entries = [json.loads(l) for l in f if l.strip()]
        if not self.entries:
            raise RuntimeError(f"no entries in {log_path}")

        self.log_t0  = float(self.entries[0]['t'])
        duration_s   = float(self.entries[-1]['t']) - self.log_t0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.pub = self.create_publisher(Twist, topic, qos)
        self.idx = 0
        self.wall_t0 = None  # set on first tick

        # 200 Hz polling — finer than the 20 Hz source data so timing
        # is accurate even at 2-3x speed.
        self._timer = self.create_timer(0.005, self._tick)

        self.get_logger().info(
            f"loaded {len(self.entries)} Twists from {log_path}")
        self.get_logger().info(
            f"original duration: {duration_s:.2f}s  speed: {self.spd}x  "
            f"loop: {self.loop}  publishing on: {topic}")

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.wall_t0 is None:
            self.wall_t0 = now

        # log-time elapsed since playback started
        elapsed = (now - self.wall_t0) * self.spd

        # Publish every entry whose log-relative timestamp has passed
        while self.idx < len(self.entries):
            entry_dt = float(self.entries[self.idx]['t']) - self.log_t0
            if entry_dt > elapsed:
                break
            self._publish_entry(self.entries[self.idx])
            self.idx += 1

        if self.idx >= len(self.entries):
            if self.loop:
                self.idx = 0
                self.wall_t0 = now
                self.get_logger().info("loop: restarting")
            else:
                self.get_logger().info(
                    f"done — published {len(self.entries)} Twists")
                self._timer.cancel()

    def _publish_entry(self, e):
        m = Twist()
        m.linear.x  = float(e['linear']['x'])
        m.linear.y  = float(e['linear']['y'])
        m.linear.z  = float(e['linear']['z'])
        m.angular.x = float(e['angular']['x'])
        m.angular.y = float(e['angular']['y'])
        m.angular.z = float(e['angular']['z'])
        self.pub.publish(m)


def main():
    rclpy.init()
    try:
        node = TwistReplayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, FileNotFoundError) as e:
        print(f"twist_replayer: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
