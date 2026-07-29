#!/bin/bash
# Runs rooster_twist_control_adapter.py inside the `robotican_dev` container
# (docker-compose.robotican.yml / theagency:robotican).
#
# New 2026-07-29 (see docs/progress/entries/002-rooster-full-containerize.md)
# -- this adapter previously had no run_*.sh wrapper at all and was launched
# ad hoc via the host venv. Bridges FALCON's /cmd_vel into cmd_nav "move"
# commands for click-to-fly; requires `robotican_dev` and
# rooster_command_unit.py (inside `it`) already running. Requires
# `robotican_dev` already running (see docker-compose.robotican.yml).
docker exec \
  -e ROS_DOMAIN_ID=9 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI="file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml" \
  robotican_dev bash -lc "
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=\$PYTHONPATH:/home/$USER/GIT/TheAgency
    python3 -m sparx_agency.robots.ROBOTICAN.adapters.rooster_twist_control_adapter $*
  "
