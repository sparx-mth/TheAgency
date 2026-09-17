#!/bin/bash
# No GStreamer needed here, but still runs in robotican_dev for the same
# ROS2 Humble/domain-id/CycloneDDS env every other Rooster node depends on.
# Requires robotican_dev already running.
docker exec \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-9}" \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml" \
  robotican_dev bash -lc "
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=\$PYTHONPATH:/home/$USER/GIT/TheAgency
    python3 /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/rooster_v4l2_frame_publisher.py $*
  "
