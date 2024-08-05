#!/usr/bin/env bash

docker build \
  -t IsaacGym-AerialGym:1.0 \
  -f $(dirname "$0")/../docker/Dockerfile-IsaacGym-AerialGym $(dirname "$0")/..