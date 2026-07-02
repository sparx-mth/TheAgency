#!/usr/bin/env bash
set -euo pipefail

gazebo --verbose /opt/ros/humble/share/gazebo_ros/worlds/empty.world -s libgazebo_ros_init.so -s libgazebo_ros_factory.so
