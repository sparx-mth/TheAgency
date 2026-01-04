#!/usr/bin/env bash

docker_image=nvidia-base-cuda-11.7.1-ubuntu-22.04:1.0

RED='\033[0;31m'
RED_BOLD='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
BROWN='\033[0;33m'
CYAN_BOLD='\033[1;36m'
# ==================================

if [[ -z "$1" ]]; then
  name="nvidia_base"
elif [[ "$1" == "?" ]]; then
  name="" # Used randon name
else
  name="$1"
fi

# Allow X Server Access on the Host to display simulation window
xhost +local:docker

echo -e "${RED}Launching docker image ${RED_BOLD}$docker_image"${NC}

run_docker_cmd="docker run --name ${name}"

    # --runtime=nvidia \
    # --gpus all \
    
run_docker_cmd+="  --rm \
    --net=host \
    --ipc=host \
    -it \
    --entrypoint bash \
    -e ACCEPT_EULA=Y\
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    -v $(dirname -- "$( readlink -f -- "$0"; )")/../scripts/bashrc:/root/.bashrc:ro \
    -v $(dirname -- "$( readlink -f -- "$0"; )")/../scripts/tmux.conf:/root/.tmux.conf:ro \
    -v $(dirname -- "$( readlink -f -- "$0"; )")/..:/workspace"

run_docker_cmd+=" $docker_image"

echo -e "${CYAN_BOLD}$run_docker_cmd${NC}"
eval "$run_docker_cmd"
