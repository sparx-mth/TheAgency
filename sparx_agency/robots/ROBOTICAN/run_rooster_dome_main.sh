#!/bin/bash
# Wrapper for running rooster_dome_main.py from PyCharm (or any launcher)
# without fighting its environment-variable UI - sources the real ROS env in
# a real shell, then execs the venv's python with whatever args were passed
# through. Same pattern as run_ui.sh/run_calibrate_camera.sh.
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_dome_main.py "$@"
