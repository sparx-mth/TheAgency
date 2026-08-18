#!/usr/bin/env bash
# Apply one patch script to a throwaway container and COMPILE the result.
#
# Applying cleanly and compiling are different things, and only the second one
# matters. fix_falcon_finish_reopen.sh applied without a murmur, printed its own
# success line, passed its grep verification -- and then failed the real build
# 12 minutes later on a one-line redefinition, because a helper it introduced
# already existed in the source. A full `docker build` is the wrong feedback
# loop for that: it recompiles the world to tell you about one symbol.
#
# This builds only the package the patch touched, inside the current image, and
# throws the container away afterwards. Roughly a minute instead of twelve.
#
# Usage:
#   bash verify_patch.sh fix_falcon_finish_reopen.sh [package ...]
#
# With no package given it builds the two that every FALCON patch here touches.
set -euo pipefail

PATCH="${1:?usage: verify_patch.sh <patch-script.sh> [package ...]}"
shift || true
PKGS=("$@")
if [ ${#PKGS[@]} -eq 0 ]; then
  PKGS=(exploration_manager plan_manage)
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${FALCON_IMAGE:-falcon-ros:noetic}"

test -f "$HERE/$PATCH" || { echo "no such patch: $HERE/$PATCH" >&2; exit 1; }

echo "[verify] applying $PATCH in a throwaway $IMAGE container, then building: ${PKGS[*]}"

docker run --rm -v "$HERE:/p:ro" --entrypoint bash "$IMAGE" -c "
set -e
bash /p/$PATCH
echo '[verify] --- patch applied, now compiling ---'
source /opt/ros/noetic/setup.bash
cd /catkin_ws
catkin_make $(printf -- '--pkg %s ' "${PKGS[@]}") \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j\$(nproc) \
    2>&1 | grep -E 'error:|Error [0-9]|Built target' || true
echo '[verify] --- second application must be a no-op ---'
bash /p/$PATCH
" 2>&1 | tail -25
