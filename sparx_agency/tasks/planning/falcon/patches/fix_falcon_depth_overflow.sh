#!/bin/bash
# ============================================================
# fix_falcon_depth_overflow.sh
#
# Fixes a heap-corruption crash (malloc(): invalid size /
# SIGABRT / roslaunch exit -6) in FALCON's voxel_mapping that
# fires inside MapServer::depthToPointcloud during depth
# integration.
#
# Root cause:
#   depthToPointcloud (and depthToPointcloudDecimation) pre-size
#   the output point cloud with FLOOR division:
#       resize(cols * rows / (skip * skip));
#   but the projection loop actually writes
#       ceil(rows/skip) * ceil(cols/skip)
#   points. When the image height/width is not divisible by skip
#   (e.g. 504x294 with skip=4 -> alloc 9261 but writes up to 9324)
#   AND the depth frame is densely valid (a real camera fills the
#   frame, so the `depth==0` skip rarely fires), points_num runs
#   past the end of the vector -> out-of-bounds write -> heap
#   corruption. It is intermittent because the ~1KB overrun only
#   sometimes lands on a live chunk header.
#
# Fix:
#   Allocate a safe upper bound, (rows/skip + 1) * (cols/skip + 1),
#   which is always >= the loop count. The function's trailing
#   resize(points_num) trims it back, so over-allocation is free.
#
# Idempotent + self-verifying. Run AFTER cloning FALCON, BEFORE
# catkin_make.
# ============================================================
set -euo pipefail

MS="$(find /catkin_ws/src/FALCON -path '*voxel_mapping*/map_server.cpp' | head -n1)"
if [ -z "${MS}" ]; then
  echo "[fix_depth_overflow] ERROR: map_server.cpp not found" >&2
  exit 1
fi
echo "[fix_depth_overflow] Patching ${MS}"

if grep -q "FALCON patch: depth cloud overflow" "${MS}"; then
  echo "[fix_depth_overflow] Already patched, skipping."
  exit 0
fi
cp "${MS}" "${MS}.orig.bak"

# Fix 1: depthToPointcloud (skip_pixel).
perl -0777 -i -pe 's{pointcloud\.points\.resize\(cols \* rows / \(config_\.skip_pixel_ \* config_\.skip_pixel_\)\);}{pointcloud.points.resize((rows / config_.skip_pixel_ + 1) * (cols / config_.skip_pixel_ + 1)); // FALCON patch: depth cloud overflow}' "${MS}"

# Fix 2: depthToPointcloudDecimation (decimation_magnitude_).
perl -0777 -i -pe 's{pointcloud\.points\.resize\(cols \* rows / \(decimation_magnitude_ \* decimation_magnitude_\)\);}{pointcloud.points.resize((rows / decimation_magnitude_ + 1) * (cols / decimation_magnitude_ + 1)); // FALCON patch: depth cloud overflow}' "${MS}"

N=$(grep -c "FALCON patch: depth cloud overflow" "${MS}" || true)
echo "[fix_depth_overflow] patches applied: ${N} (expect 2)"
if [ "${N}" -lt 1 ]; then
  echo "[fix_depth_overflow] ERROR: patch target not found; upstream code may have changed." >&2
  mv "${MS}.orig.bak" "${MS}"
  exit 1
fi
echo "[fix_depth_overflow] OK: depth point-cloud allocation is now a safe upper bound."