#!/bin/bash
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
while true; do
  running=$(ps aux | grep 'devel/lib/exploration_manager/exploration_node' | grep -v grep)
  if [ -z "$running" ]; then
    echo "[watchdog] $(date): exploration_node missing, relaunching" >> /tmp/exploration_watchdog.log
    rosrun exploration_manager exploration_node \
      /voxel_mapping/depth_image:=/map_ros/depth \
      /transformer/sensor_pose_topic:=/map_ros/pose \
      /odom_world:=/odom_world \
      __name:=exploration_node >> /tmp/exploration_node_watchdog_stdout.log 2>&1 &
    sleep 5
  fi
  sleep 2
done
