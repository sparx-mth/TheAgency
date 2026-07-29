#!/bin/bash
# Runs rooster_frame_dir_publisher.py inside the `robotican_dev` container
# (docker-compose.robotican.yml / theagency:robotican), not the host venv.
#
# 2026-07-29: moved off the bare host venv (see docs/progress/entries/
# 002-rooster-full-containerize.md) -- the ABI-sensitive GStreamer/PyGObject
# bindings this script needs (see the old comment history in git blame if
# curious) are already built into theagency:robotican's Humble+GStreamer
# base image, verified live: `import gi; Gst.init(None)` works cleanly
# there, so the host-venv workaround is no longer needed. Requires
# `robotican_dev` already running (see docker-compose.robotican.yml) --
# same container `run_depth_processor.sh`/`run_twist_control_adapter.sh` use.
docker exec \
  -e ROS_DOMAIN_ID=9 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml" \
  robotican_dev bash -lc "
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=\$PYTHONPATH:/home/$USER/GIT/TheAgency
    python3 /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_frame_dir_publisher.py $*
  "
