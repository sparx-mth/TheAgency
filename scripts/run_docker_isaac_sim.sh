#!/usr/bin/env bash

docker_image=nvcr.io/nvidia/isaac-sim:4.0.0


RED='\033[0;31m'
RED_BOLD='\033[1;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
BROWN='\033[0;33m'
CYAN_BOLD='\033[1;36m'
# ==================================

if [[ -z "$1" ]]; then
  name="isaac_sim"
elif [[ "$1" == "?" ]]; then
  name="" # Used randon name
else
  name="$1"
fi

# Allow X Server Access on the Host to display simulation window
xhost +local:docker

echo -e "${RED}Launching docker image ${RED_BOLD}$docker_image"${NC}

run_docker_cmd="docker run --name ${name}"

run_docker_cmd+="  --rm \
    --net=host \
    --ipc=host \
    -it \
    --entrypoint bash \
    --runtime=nvidia \
    --gpus all \
    -e ACCEPT_EULA=Y\
    -v ./isaac_sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ./isaac_sim/cache/ov:/root/.cache/ov:rw \
    -v ./isaac_sim/cache/pip:/root/.cache/pip:rw \
    -v ./isaac_sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ./isaac_sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ./isaac_sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ./isaac_sim/data:/root/.local/share/ov/data:rw \
    -v ./isaac_sim/documents:/root/Documents:rw \
    -v $(dirname -- "$( readlink -f -- "$0"; )")/../scripts/bashrc:/root/.bashrc:ro \
    -v $(dirname -- "$( readlink -f -- "$0"; )")/..:/workspace"

run_docker_cmd+=" $docker_image"

echo -e "${CYAN_BOLD}$run_docker_cmd${NC}"
eval "$run_docker_cmd"