#!/bin/bash
# Wrapper for running rooster_offline_frame_dir_publisher.py from PyCharm (or
# any launcher) without fighting its environment-variable UI - sources the
# real ROS env in a real shell, then execs the venv's python with whatever
# args were passed through. Same pattern as run_ui.sh/run_rooster_frame_dir_publisher.sh.
#
# Usage: ./run_rooster_offline_replay.sh --session-dir ~/rooster_dome_capture/latest [--loop]
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=6
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_offline_frame_dir_publisher.py "$@"
