#!/bin/bash
# Wrapper for running rooster_frame_dir_publisher.py from PyCharm (or any
# launcher) without fighting its environment-variable UI - sources the real
# ROS env in a real shell, then execs the venv's python with whatever args
# were passed through. Same pattern as run_ui.sh/run_calibrate_camera.sh.
#
# Running this script directly from PyCharm's run config (instead of pointing
# it at the .py file) is required, not optional: without ROS's environment
# (GI_TYPELIB_PATH/GST_PLUGIN_PATH/LD_LIBRARY_PATH etc. set by setup.bash),
# the gi/GStreamer bindings this script loads can be ABI-mismatched against
# whatever's on the default system path, which crashes as a hard SIGSEGV
# rather than a Python exception.
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_frame_dir_publisher.py "$@"
