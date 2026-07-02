#!/usr/bin/env bash
set -euo pipefail


# echo "Setting up turtlebot3_burger with depth camera..."
# bash /workspace/turtlebot3_modifications/setup_camera.sh 

echo "Spawning turtlebot3_burger..."
ros2 launch turtlebot3_gazebo empty_world.launch.py   

# ros2 launch turtlebot3_gazebo empty_world.launch.py x_pos:=1.0 y_pos:=2.0 z_pos:=0.0   

# ros2 run gazebo_ros spawn_entity.py -topic /robot_description -entity turtlebot3_burger -x 1.0 -y 2.0 -z 0.0   