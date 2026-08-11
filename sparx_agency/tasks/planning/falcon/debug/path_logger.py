#!/usr/bin/env python3
"""path_logger.py -- logs every A* path FALCON publishes, plus every click goal.

Companion to manual_flight_logger.py (which logs cmd_nav + localization). This
one runs inside the `falcon` container (ROS1) since /path/waypoints_astar and
/waypoint_nav/goal never cross ros1_bridge. Read-only: never publishes.

  paths.jsonl  {t, goal_x, goal_y, waypoints: [[x,y], ...]}  -- one row per
               new A* path (nav_msgs/Path on ~path_topic)
  goals.jsonl  {t, x, y}                                     -- one row per
               click/goal message on ~goal_topic

Built for the specific question "when the path asks the drone to turn right,
why does it turn left" -- pairing paths.jsonl's first waypoint bearing against
manual_flight_logger's pose.jsonl yaw trend at the same timestamp answers it
directly, without re-deriving bearing math from scratch each time.
"""
import argparse
import json
import os
import time

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Path


class PathLogger(object):
    def __init__(self, path_topic, goal_topic, out_dir):
        rospy.init_node("path_logger")
        os.makedirs(out_dir, exist_ok=True)
        self._paths_f = open(os.path.join(out_dir, "paths.jsonl"), "a", buffering=1)
        self._goals_f = open(os.path.join(out_dir, "goals.jsonl"), "a", buffering=1)
        self._last_goal = None

        rospy.Subscriber(path_topic, Path, self._on_path, queue_size=5)
        rospy.Subscriber(goal_topic, Point, self._on_goal, queue_size=5)

        rospy.loginfo(
            "path_logger ready\n  paths -> %s\n  goals -> %s",
            self._paths_f.name, self._goals_f.name)

    def _on_goal(self, msg):
        self._last_goal = (msg.x, msg.y)
        row = {"t": round(time.time(), 3), "x": round(msg.x, 4), "y": round(msg.y, 4)}
        self._goals_f.write(json.dumps(row) + "\n")

    def _on_path(self, msg):
        wps = [[round(p.pose.position.x, 4), round(p.pose.position.y, 4)]
               for p in msg.poses]
        row = {
            "t": round(time.time(), 3),
            "goal_x": self._last_goal[0] if self._last_goal else None,
            "goal_y": self._last_goal[1] if self._last_goal else None,
            "waypoints": wps,
        }
        self._paths_f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path-topic", default="/path/waypoints_astar")
    p.add_argument("--goal-topic", default="/waypoint_nav/goal")
    p.add_argument("--out-dir", default="/tmp/path_logger_out")
    args = p.parse_args()

    PathLogger(args.path_topic, args.goal_topic, args.out_dir)
    rospy.spin()


if __name__ == "__main__":
    main()
