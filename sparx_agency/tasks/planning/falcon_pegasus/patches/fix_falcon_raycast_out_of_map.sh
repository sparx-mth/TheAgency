#!/bin/bash
# Stop the mapper aborting when a depth ray leaves the voxel map.
#
# `TSDF::inputPointCloud` walks every voxel between the sensor and each returned
# point with a DDA raycaster, and turns each one into a flat array address:
#
#     while (raycaster_->nextId(voxel_idx)) {
#       voxel_addr = indexToAddress(voxel_idx);        // no bounds check
#       updateTSDFVoxel(voxel_addr, value, weight);    // writes data[addr]
#       occupancy_grid_->updateOccupancyVoxel(voxel_addr);
#     }
#
# The ray's endpoints are clamped into the map -- `closestPointInMap` -- but the
# DDA itself can still step one voxel past the edge on the way there, because
# the clamp is in metres and the stepping is in voxels. Neither consumer of the
# address checks it. `updateTSDFVoxel` writes `map_data_->data[addr]` straight
# out; `updateOccupancyVoxel` reads through `getVoxel(addr)`, which carries a
# glog CHECK, and glog's CHECK failure is `abort()`:
#
#     F0806 21:12:34 map_base_inl.h:173] Check failed:
#       addr < map_data_->data.size() (6334695 vs. 6334614)
#       Address out of range: 6334695
#
# Eighty-one addresses past the end -- a single index step off one face of the
# grid. It killed a 6_whole_office flight at 595 m3 of coverage.
#
# The same file already knows this can happen and handles it correctly one
# function away, in `OccupancyGrid::setOccupancy`:
#
#     if (addr < 0 || addr >= (int)map_data_->data.size()) return false;
#
# so this is not a new policy, it is the existing one applied on the path that
# was missed.
#
# GUARD THE LOOP, NOT THE TWO CALLEES. Skipping the voxel where the index is
# produced fixes both consumers with one condition, keeps the check in voxel
# space where `isInMap(VoxelIndex)` already exists, and costs one comparison per
# raycast step. A voxel outside the map is not somewhere the map can record
# anything about, so there is nothing to lose by skipping it.
set -e

TSDF=/catkin_ws/src/FALCON/falcon_planner/voxel_mapping/src/tsdf.cpp
[ -f "$TSDF" ] || { echo "tsdf.cpp not found at $TSDF"; exit 1; }

python3 - "$TSDF" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()

original = """    while (raycaster_->nextId(voxel_idx)) {
      voxel_pos = indexToPosition(voxel_idx);
      voxel_addr = indexToAddress(voxel_idx);"""

patched = """    while (raycaster_->nextId(voxel_idx)) {
      // The ray's endpoints were clamped into the map in metres, but the DDA
      // steps in voxels and can still land one index past an edge. Neither
      // updateTSDFVoxel nor updateOccupancyVoxel checks the address it is
      // given: the first writes the array directly, the second reads through a
      // glog CHECK that calls abort(). A voxel outside the map holds nothing
      // worth recording, so skip it.
      if (!isInMap(voxel_idx))
        continue;
      voxel_pos = indexToPosition(voxel_idx);
      voxel_addr = indexToAddress(voxel_idx);"""

if source.count(original) != 1:
    raise SystemExit("patch: expected exactly 1 raycast fuse loop, found %d"
                     % source.count(original))

open(path, "w").write(source.replace(original, patched))
print("patched: raycast voxels outside the map are skipped, not fused")
PYEOF
