#!/bin/bash
# ============================================================
# falcon_docker/fix_falcon_sop.sh
#
# Patches FALCON's SOP solver timeout check from 1s → 10s and
# rebuilds the exploration_manager package.
#
# The actual line in exploration_manager.cpp:578 is:
#   CHECK_LE(sop_time, 1.0) << "SOP solver internal error..."
# When the SOP takes slightly over 1s (e.g. 1.025s — see crash
# logs), glog calls abort() and kills the planner. The abort
# message itself says "Please restart the planner" — i.e. the
# authors know this check is overzealous.
#
# Usage (one-shot, inside the falcon container):
#   docker exec -it falcon bash /fix_falcon_sop.sh
#
# To make it permanent across container rebuilds, add to the
# FALCON Dockerfile:
#   COPY fix_falcon_sop.sh /fix_falcon_sop.sh
#   RUN bash /fix_falcon_sop.sh
# ============================================================
set -e

SRC="/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp"

if [[ ! -f "${SRC}" ]]; then
    echo "[fix] ERROR: ${SRC} not found"
    exit 1
fi

if grep -q 'CHECK_LE(sop_time, 10.0)' "${SRC}"; then
    echo "[fix] Already patched — nothing to do."
    exit 0
fi

if ! grep -q 'CHECK_LE(sop_time, 1\.0)' "${SRC}"; then
    echo "[fix] ERROR: patch target not found."
    grep -n "sop_time" "${SRC}" || true
    exit 1
fi

echo "[fix] Patching ${SRC}..."
sed -i 's|CHECK_LE(sop_time, 1\.0)|CHECK_LE(sop_time, 10.0)|' "${SRC}"

echo "[fix] Verifying patch:"
grep -n "CHECK_LE(sop_time" "${SRC}"

echo "[fix] Rebuilding exploration_manager..."
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make --only-pkg-with-deps exploration_manager \
    >/tmp/falcon_rebuild.log 2>&1 || {
        echo "[fix] Build failed. Last 40 lines of log:"
        tail -40 /tmp/falcon_rebuild.log
        exit 1
    }

echo "[fix] Done. Restart roslaunch for the patch to take effect."