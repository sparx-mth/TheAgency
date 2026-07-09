#!/bin/bash
# Wrapper for running calibrate_camera.py with the ROS-enabled venv, same
# pattern as run_ui.sh - sources the real ROS env in a real shell, then execs
# the venv's python with whatever args were passed through.
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/calibrate_camera.py "$@"
