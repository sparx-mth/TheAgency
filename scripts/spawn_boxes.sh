#!/usr/bin/env bash
set -euo pipefail

ros2 run gazebo_ros spawn_entity.py -entity box1 -file /workspace/turtlebot3_modifications/box.sdf -x 2 -y 0 -z 0

ros2 run gazebo_ros spawn_entity.py -entity box2 -file /workspace/turtlebot3_modifications/box.sdf -x -2 -y 0 -z 0

ros2 run gazebo_ros spawn_entity.py -entity box3 -file /workspace/turtlebot3_modifications/box.sdf -x 2 -y 2 -z 0

