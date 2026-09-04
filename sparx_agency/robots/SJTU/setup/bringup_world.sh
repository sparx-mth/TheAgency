#!/usr/bin/env bash
# Bring one SJTU world up in docker, with nothing else attached to /cmd_vel.
#
# Why this exists rather than the external repo's run.sh:
#
#   * run.sh launches `sjtu_drone_bringup.launch.py`, which also starts rviz2, a
#     joy node and an xterm teleop -- and the teleop PUBLISHES TO
#     /simple_drone/cmd_vel. Two publishers on the only control input make every
#     control experiment unrepeatable, and nothing warns you. This script
#     launches the inner `sjtu_drone_gazebo.launch.py` instead:
#     robot_state_publisher + gzserver + optional gzclient + spawn_drone + the
#     world->odom static TF, and nothing else.
#   * run.sh appends '.world' to its argument, so `./run.sh hospital.world`
#     becomes 'hospital.world.world' and aborts with "world file not found".
#     This script takes a bare NAME, tolerates a name that already ends in
#     .world, and resolves the file itself.
#   * run.sh also clones and builds gazebo_ros_2d_map on the way past.
#
# Everything about the external checkout comes from setup/env.sh; no path to the
# simulator is written down here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `|| exit` and not `set -e` alone: a sourced script that hits its own failure
# path `return`s, which would drop straight back into this one with the paths
# unset.
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh" || exit 1

# Repos that may contribute worlds and models, searched in this order. Only
# aws-robomaker-hospital-world and sjtu_drone_description are present in the
# checkout today; the rest are listed so that cloning one is all it takes.
WORLD_REPOS=(
  "aws-robomaker-hospital-world"
  "aws-robomaker-small-house-world"
  "aws-robomaker-bookstore-world"
  "aws-robomaker-small-warehouse-world"
  "sjtu_drone/sjtu_drone_description"
)

USE_GUI="false"
SKIP_BUILD="false"
CONTAINER_NAME=""
WORLD_NAME=""
# Empty means "decide from what is installed"; --rmw makes it a demand, and a
# demand that cannot be met is an error rather than a quiet downgrade.
RMW_CHOICE=""

usage() {
  cat <<EOF
Usage: bringup_world.sh [options] <world>

Options:
  --gui             also run gzclient, to watch the simulation
  --headless        gzserver only (default). Still REQUIRES a working \$DISPLAY --
                    see the note below; headless here means "no viewer", not
                    "no X".
  --domain <N>      ROS_DOMAIN_ID for the container (default ${ROS_DOMAIN_ID})
  --rmw <IMPL>      cyclonedds | fastrtps. Default cyclonedds when
                    ${SJTU_CYCLONE_IMAGE} is built, else fastrtps with a
                    warning. Only cyclonedds can reach the ROS 1 bridge.
  --name <NAME>     container name (default sjtu_drone_<world>)
  --skip-build      reuse the workspace already built under
                    \$SJTU_PROJECT_DIR/install (fast; wrong after a plugin edit)
  -h, --help        this message

A world is named, not pathed: 'hospital', 'playground'. A trailing '.world' is
accepted and stripped.

Worlds available in ${SJTU_PROJECT_DIR}:
EOF
  local found="false"
  local repo dir world
  for repo in "${WORLD_REPOS[@]}"; do
    dir="${SJTU_PROJECT_DIR}/${repo}/worlds"
    [[ -d "${dir}" ]] || continue
    for world in "${dir}"/*.world; do
      [[ -e "${world}" ]] || continue
      printf '  %-26s %s\n' "$(basename "${world}" .world)" "${dir}"
      found="true"
    done
  done
  if [[ "${found}" == "false" ]]; then
    echo "  (none -- no <repo>/worlds directory found)"
  fi
  cat <<EOF

Not present in this checkout: small_house, bookstore, small_warehouse. They are
separate aws-robomaker repositories; clone one next to sjtu_drone/ and it will
appear in the list above with no change to this script.

Gazebo Classic disables EVERY camera sensor when it cannot open a display --
with the GUI off as well -- logging "Can't open display" and then "Unable to
create CameraSensor. Rendering is disabled." The drone still flies and still
publishes odom, so the failure looks like a camera bug rather than a display
one. \$DISPLAY must be set and /tmp/.X11-unix must exist; :1 (XWayland) works on
this machine.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gui) USE_GUI="true"; shift ;;
    --headless) USE_GUI="false"; shift ;;
    --domain) ROS_DOMAIN_ID="${2:?--domain needs a value}"; export ROS_DOMAIN_ID; shift 2 ;;
    --rmw) RMW_CHOICE="${2:?--rmw needs a value}"; shift 2 ;;
    --name) CONTAINER_NAME="${2:?--name needs a value}"; shift 2 ;;
    --skip-build) SKIP_BUILD="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "[sjtu/bringup] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    *)
      [[ -z "${WORLD_NAME}" ]] || { echo "[sjtu/bringup] ERROR: more than one world named" >&2; exit 2; }
      WORLD_NAME="$1"; shift ;;
  esac
done

if [[ -z "${WORLD_NAME}" ]]; then
  echo "[sjtu/bringup] ERROR: no world named." >&2
  usage >&2
  exit 2
fi

# The run.sh bug, avoided rather than reproduced: strip a '.world' the caller
# supplied instead of appending a second one.
WORLD_NAME="${WORLD_NAME%.world}"

HOST_WORLD=""
for repo in "${WORLD_REPOS[@]}"; do
  candidate="${SJTU_PROJECT_DIR}/${repo}/worlds/${WORLD_NAME}.world"
  if [[ -f "${candidate}" ]]; then
    HOST_WORLD="${candidate}"
    break
  fi
done

if [[ -z "${HOST_WORLD}" ]]; then
  echo "[sjtu/bringup] ERROR: no world file '${WORLD_NAME}.world' under ${SJTU_PROJECT_DIR}." >&2
  usage >&2
  exit 1
fi

CONTAINER_WORLD="${HOST_WORLD/${SJTU_PROJECT_DIR}/${SJTU_CONTAINER_WS}}"
CONTAINER_NAME="${CONTAINER_NAME:-sjtu_drone_${WORLD_NAME}}"
WORLD_REPOS_STR="${WORLD_REPOS[*]}"

# --- DDS ------------------------------------------------------------------
#
# The middleware and the image are one decision, not two: only
# $SJTU_CYCLONE_IMAGE carries rmw_cyclonedds_cpp, so asking for CycloneDDS is
# asking for that image. Fast DDS is perfectly fine for anything that stays
# inside ROS 2 -- it is the ROS 1 bridge, which runs Foxy and therefore Fast DDS
# 2.1 against Humble's 2.6, that cannot decode it.
#
# Unlike the external run.sh, nothing is apt-installed at launch here. That
# install needed the network on every single run and hard-exited on failure,
# which turned a flaky mirror into "the simulator will not start".
IMAGE="${SJTU_IMAGE}"
HAVE_CYCLONE_IMAGE="false"
if docker image inspect "${SJTU_CYCLONE_IMAGE}" >/dev/null 2>&1; then
  HAVE_CYCLONE_IMAGE="true"
fi

case "${RMW_CHOICE}" in
  cyclonedds|fastrtps) ;;
  "")
    # Default: CycloneDDS when it is available, because a world brought up on
    # Fast DDS is invisible to FALCON and the symptom -- a bridge that lists
    # nothing -- looks like a bridge bug for as long as you care to look.
    if [[ "${HAVE_CYCLONE_IMAGE}" == "true" ]]; then
      RMW_CHOICE="cyclonedds"
    else
      RMW_CHOICE="fastrtps"
      echo "[sjtu/bringup] WARNING: '${SJTU_CYCLONE_IMAGE}' is not built, so this world comes up on" >&2
      echo "  Fast DDS. ROS 2 tooling will work; the ROS 1 bridge will see NOTHING, because" >&2
      echo "  Foxy's Fast DDS 2.1 is not wire-compatible with Humble's 2.6." >&2
      echo "  Build it (~10 s, thin layer over ${SJTU_IMAGE}):" >&2
      echo "    docker build -t ${SJTU_CYCLONE_IMAGE} ${SCRIPT_DIR}" >&2
    fi
    ;;
  *)
    echo "[sjtu/bringup] ERROR: --rmw '${RMW_CHOICE}' is not one of cyclonedds, fastrtps." >&2
    exit 2 ;;
esac

if [[ "${RMW_CHOICE}" == "cyclonedds" ]]; then
  if [[ "${HAVE_CYCLONE_IMAGE}" != "true" ]]; then
    echo "[sjtu/bringup] ERROR: --rmw cyclonedds, but '${SJTU_CYCLONE_IMAGE}' is not built." >&2
    echo "  Only that image carries ros-humble-rmw-cyclonedds-cpp; the base image would" >&2
    echo "  start, log 'failed to load required rmw library' and die." >&2
    echo "    docker build -t ${SJTU_CYCLONE_IMAGE} ${SCRIPT_DIR}" >&2
    exit 1
  fi
  if [[ "${SJTU_IMAGE_EXPLICIT:-false}" == "true" ]]; then
    # A pinned SJTU_IMAGE is a decision; honour it and say what it costs rather
    # than swapping the image out from under the caller.
    echo "[sjtu/bringup] NOTE: SJTU_IMAGE is pinned to '${SJTU_IMAGE}', so that is what runs." >&2
    echo "  It must provide rmw_cyclonedds_cpp itself or the container will not start." >&2
  else
    IMAGE="${SJTU_CYCLONE_IMAGE}"
  fi
  RMW_IMPL="rmw_cyclonedds_cpp"
  DDS_URI="${SJTU_CYCLONEDDS_URI}"
else
  RMW_IMPL="rmw_fastrtps_cpp"
  DDS_URI=""
fi

# Printed before the environment checks below, so a run that is about to be
# refused still says what it resolved -- which is usually the thing in dispute.
echo "[sjtu/bringup] world     ${HOST_WORLD}"
echo "[sjtu/bringup]           -> ${CONTAINER_WORLD} (in container)"
echo "[sjtu/bringup] container ${CONTAINER_NAME} (${IMAGE})"
echo "[sjtu/bringup] rmw       ${RMW_IMPL}${DDS_URI:+   config ${DDS_URI}}"
echo "[sjtu/bringup] gui       ${USE_GUI}   domain ${ROS_DOMAIN_ID}   display ${DISPLAY:-<unset>}"

if [[ -z "${DISPLAY:-}" || ! -d /tmp/.X11-unix ]]; then
  echo "[sjtu/bringup] ERROR: no X display (DISPLAY='${DISPLAY:-}', /tmp/.X11-unix $( [[ -d /tmp/.X11-unix ]] && echo present || echo missing ))." >&2
  echo "  Gazebo Classic disables all camera sensors without one, GUI or not, so the" >&2
  echo "  drone would fly blind and publish nothing on /simple_drone/front/*." >&2
  echo "  On this machine: export DISPLAY=:1" >&2
  exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[sjtu/bringup] ERROR: docker image '${IMAGE}' not found." >&2
  echo "  Build it: docker build -t ${IMAGE} ${SJTU_DRONE_DIR}" >&2
  exit 1
fi

# Let the container's root talk to the host X server, and take it back on exit
# however the script ends.
xhost +local:docker >/dev/null 2>&1 || true
trap 'xhost -local:docker >/dev/null 2>&1 || true' EXIT

# Allocate a TTY only when there is one to allocate. `docker run -it` from a
# script with a redirected stdin -- a soak loop, a CI job, a `nohup ... &` --
# dies with "cannot attach stdin to a TTY-enabled container", and it does so
# AFTER printing the whole banner, so it reads as the simulator failing to
# start rather than as an argument problem. The interactive case still gets its
# TTY, which is what makes Ctrl-C reach gzserver.
TTY_ARGS=()
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

# THE SIMULATOR DOES NOT GET THE GPU.
#
# Gazebo Classic renders its camera sensors through OGRE, and with `--gpus all`
# that rendering context lands on the same card a VLA server is holding. On an
# 8 GB card with InternVLA-N1 resident at ~7.2 GB there is under a gigabyte
# left, and Gazebo asking for a GL context in what remains takes the whole
# machine down -- not the container, the machine. Software rendering costs
# frames on a box with 32 threads to spare; the alternative costs the session.
#
# So: no `--gpus`, and llvmpipe forced below, unless someone deliberately asks
# for the card with SJTU_SIM_GPU=1 (a machine with a second GPU, or no VLA
# running). The physics never touched the GPU in the first place -- Gazebo
# Classic has no GPU physics -- so this only moves the rendering.
GPU_ARGS=()
RENDER_ENV=(-e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe -e MESA_LOADER_DRIVER_OVERRIDE=llvmpipe)
if [[ "${SJTU_SIM_GPU:-0}" == "1" ]]; then
  GPU_ARGS=(--gpus all)
  RENDER_ENV=()
  echo "[sjtu/bringup] SJTU_SIM_GPU=1: the simulator is taking the GPU." >&2
  echo "  If a VLA server is resident on the same card this can hard-lock the host." >&2
else
  echo "[sjtu/bringup] rendering on the CPU (llvmpipe); the GPU is left free."
fi

docker run "${TTY_ARGS[@]}" --rm \
  "${GPU_ARGS[@]}" \
  "${RENDER_ENV[@]}" \
  --privileged \
  --net=host \
  --ipc=host \
  -v /dev/shm:/dev/shm \
  --name "${CONTAINER_NAME}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${SJTU_PROJECT_DIR}:${SJTU_CONTAINER_WS}:rw" \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPL}" \
  -e CYCLONEDDS_URI="${DDS_URI}" \
  -e SJTU_CONTAINER_WS="${SJTU_CONTAINER_WS}" \
  -e WORLD_REPOS="${WORLD_REPOS_STR}" \
  -e CONTAINER_WORLD="${CONTAINER_WORLD}" \
  -e USE_GUI="${USE_GUI}" \
  -e SKIP_BUILD="${SKIP_BUILD}" \
  -e SJTU_DRONE_SPAWN="${SJTU_DRONE_SPAWN:-}" \
  "${IMAGE}" \
  bash -c '
    set -eo pipefail
    source /opt/ros/humble/setup.bash

    # An empty CYCLONEDDS_URI is not the same as an unset one: CycloneDDS reads
    # it as "a config file at the empty path" and aborts participant creation.
    # This is the Fast DDS path, where the variable is deliberately blank.
    [[ -n "${CYCLONEDDS_URI:-}" ]] || unset CYCLONEDDS_URI

    # Fail here, loudly, rather than inside a launch file: a missing rmw library
    # surfaces as a Python traceback from rclpy three screens into the colcon
    # output, long after the interesting lines have scrolled away.
    if [[ "$RMW_IMPLEMENTATION" == "rmw_cyclonedds_cpp" && ! -f /opt/ros/humble/lib/librmw_cyclonedds_cpp.so ]]; then
      echo "[sjtu/bringup] ERROR: this image has no librmw_cyclonedds_cpp.so." >&2
      echo "  It is not the sjtu_drone_sparx image; rebuild from robots/SJTU/setup." >&2
      exit 1
    fi
    echo "[sjtu/bringup] rmw in container: $RMW_IMPLEMENTATION  ${CYCLONEDDS_URI:-<no dds config>}"

    # Also for `docker exec`ed shells: a debug session that lands on the wrong
    # middleware sees an empty topic list and blames the simulator.
    echo "export RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION" >> /root/.bashrc
    if [[ -n "${CYCLONEDDS_URI:-}" ]]; then
      echo "export CYCLONEDDS_URI=$CYCLONEDDS_URI" >> /root/.bashrc
    fi
    echo "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID" >> /root/.bashrc
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
    echo "[[ -f $SJTU_CONTAINER_WS/install/setup.bash ]] && source $SJTU_CONTAINER_WS/install/setup.bash" >> /root/.bashrc

    # Gazebo finds models and meshes by path, not by package. Every world repo
    # that is present contributes its models/, fuel_models/ and worlds/ --
    # aws-robomaker worlds reference their furniture as plain model:// URIs and
    # come up as an empty room without this.
    export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models
    export GAZEBO_RESOURCE_PATH=/usr/share/gazebo-11
    for repo in $WORLD_REPOS; do
      if [[ -d "$SJTU_CONTAINER_WS/$repo/models" ]]; then
        export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$SJTU_CONTAINER_WS/$repo/models"
      fi
      if [[ -d "$SJTU_CONTAINER_WS/$repo/fuel_models" ]]; then
        export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$SJTU_CONTAINER_WS/$repo/fuel_models"
      fi
      if [[ -d "$SJTU_CONTAINER_WS/$repo/worlds" ]]; then
        export GAZEBO_RESOURCE_PATH="$GAZEBO_RESOURCE_PATH:$SJTU_CONTAINER_WS/$repo/worlds"
      fi
      if [[ -d "$SJTU_CONTAINER_WS/$repo" ]]; then
        export GAZEBO_RESOURCE_PATH="$GAZEBO_RESOURCE_PATH:$SJTU_CONTAINER_WS/$repo"
      fi
    done
    export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$SJTU_CONTAINER_WS/sjtu_drone/sjtu_drone_description:$SJTU_CONTAINER_WS/sjtu_drone/models"
    # Empty on purpose: without it Gazebo stalls for minutes trying to fetch
    # missing models from the online database before giving up.
    export GAZEBO_MODEL_DATABASE_URI=
    export GAZEBO_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:${GAZEBO_PLUGIN_PATH:-}"
    export GAZEBO_AUDIO_DEVICE=null

    cd "$SJTU_CONTAINER_WS"
    if [[ "$SKIP_BUILD" != "true" ]]; then
      echo "[sjtu/bringup] building sjtu_drone_{bringup,description,control} ..."
      colcon build --packages-select sjtu_drone_bringup sjtu_drone_description sjtu_drone_control \
        --cmake-args -DBUILD_TESTING=OFF
    elif [[ ! -f install/setup.bash ]]; then
      echo "[sjtu/bringup] ERROR: --skip-build, but $SJTU_CONTAINER_WS/install does not exist." >&2
      echo "  Nothing has been built in this workspace yet. Drop --skip-build." >&2
      exit 1
    fi
    source install/setup.bash

    echo "[sjtu/bringup] launching gzserver on $CONTAINER_WORLD"
    exec ros2 launch sjtu_drone_bringup sjtu_drone_gazebo.launch.py \
      world:="$CONTAINER_WORLD" use_gui:="$USE_GUI"
  '
