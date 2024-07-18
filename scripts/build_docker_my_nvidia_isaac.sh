#!/usr/bin/env bash

docker build \
  -t nvidia/isaac-sim-4.0.0-matrix:1.0 \
  -f $(dirname "$0")/../docker/Dockerfile-my-nvidia-isaac $(dirname "$0")/..