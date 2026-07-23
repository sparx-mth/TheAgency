#!/bin/bash
# Sends JPEGs from --frames-dir as a UDP/RTP-H264 stream on port 5001.
# No ROS needed — pure GStreamer via the project venv.
# Usage: ./run_fake_stream.sh --frames-dir /path/to/jpgs [--fps 5] [--port 5001]
set -e
exec /home/$USER/GIT/TheAgency/venv/bin/python \
  /home/$USER/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/fake_udp_stream.py "$@"
