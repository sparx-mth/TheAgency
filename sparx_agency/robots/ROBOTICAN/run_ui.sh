#!/bin/bash
# Wrapper for running ui.py from PyCharm (or any launcher) without fighting
# its environment-variable UI - sources the real ROS env in a real shell,
# then execs the venv's python with whatever args were passed through.
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Without this, CycloneDDS silently ignores /etc/cyclonedds.xml and picks a
# network interface "arbitrarily" instead of the Rooster network - which can
# end up on a different interface than the Sphera/Rooster side and crash its
# older CycloneDDS's discovery parser.
export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
exec /home/user1/GIT/TheAgency/venv/bin/python \
  /home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/ui.py "$@"
