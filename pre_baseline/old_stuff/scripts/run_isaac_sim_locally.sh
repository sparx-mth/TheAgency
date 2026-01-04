#!/usr/bin/env bash

workdir=~/GIT/IsaacSim-ros_workspaces/humble_ws

cd "$workdir"
echo "$PWD"
source install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE="$workdir"/fastdds.xml
