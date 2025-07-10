#!/usr/bin/env bash

docker build \
  -t isaac-sim-4.0.0-matrix:1.0 \
  -f $(dirname "$0")/../docker/Dockerfile-nvidia-isaac-ros $(dirname "$0")/..