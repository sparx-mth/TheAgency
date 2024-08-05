#!/usr/bin/env bash

docker build \
  -t nvidia-base-cuda-11.7.1-ubuntu-22.04:1.0 \
  -f $(dirname "$0")/../docker/Dockerfile-nvidia-base $(dirname "$0")/..