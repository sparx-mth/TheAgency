#!/bin/bash
# ============================================================
# fix_falcon_visgrid_cadence.sh
#
# Stops occupancy-grid visualisation publishes from scaling with the explored
# volume. MapServer::publishOccupancyGrid() sweeps the ENTIRE voxel box every
# 0.5 s and serialises occupied, FREE and UNKNOWN point clouds whenever anyone
# subscribes (RViz, BEV tooling, brake gates). The free/unknown clouds grow
# with explored volume -- hundreds of thousands to millions of points -- so
# publish AND subscriber-parse cost grow linearly with mapping progress: the
# "the further the mission maps, the slower everything runs" failure, measured
# and fixed on the SJTU deployment 2026-08-12. Occupied keeps 2 Hz (small,
# safety-relevant); free/unknown drop to every 10th cycle.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail
SRC="/catkin_ws/src/FALCON/falcon_planner/voxel_mapping/src/map_server.cpp"
[ -f "${SRC}" ] || { echo "[fix_visgrid] ERROR: ${SRC} not found" >&2; exit 1; }

python3 - "$SRC" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
if "publish_bulk" in s:
    print("[fix_visgrid] already applied"); sys.exit(0)

anchor = """void MapServer::publishOccupancyGrid() {
  if (occupancy_grid_occupied_pub_.getNumSubscribers() == 0 &&
      occupancy_grid_free_pub_.getNumSubscribers() == 0 &&
      occupancy_grid_unknown_pub_.getNumSubscribers() == 0)
    return;
"""
replacement = anchor + """
  // Cadence split: occupied is small and goes every cycle; free/unknown grow
  // with explored volume and only serve visualisation -- every 10th cycle.
  static int slow_cycle_counter = 0;
  const bool publish_bulk = (slow_cycle_counter++ % 10 == 0);
"""
assert anchor in s, "anchor not found -- has upstream moved?"
s = s.replace(anchor, replacement, 1)

for kind in ("free", "unknown"):
    old = ("  pointcloud_%s.width = pointcloud_%s.points.size();\n"
           "  pointcloud_%s.height = 1;\n"
           "  pointcloud_%s.is_dense = true;\n"
           "  pointcloud_%s.header.frame_id = config_.world_frame_;\n"
           "  pcl::toROSMsg(pointcloud_%s, pointcloud_msg);\n"
           "  occupancy_grid_%s_pub_.publish(pointcloud_msg);") % ((kind,) * 7)
    new = ("  if (publish_bulk) {\n  " + old.replace("\n", "\n  ") + "\n  }")
    idx = s.find("void MapServer::publishOccupancyGrid()")
    seg_end = s.find("void MapServer::", idx + 10)
    seg = s[idx:seg_end]
    assert old in seg, "%s block not found in publishOccupancyGrid" % kind
    seg = seg.replace(old, new, 1)
    s = s[:idx] + seg + s[seg_end:]

open(p, "w").write(s)
print("[fix_visgrid] patched")
EOF

ok=1
grep -q 'publish_bulk' "${SRC}" || ok=0
[ "$(awk '/void MapServer::publishOccupancyGrid\(\)/,/^void MapServer::publishOccupancyGridHighResolution/' "${SRC}" | grep -c 'if (publish_bulk)')" = "2" ] || ok=0
[ "${ok}" = "1" ] || { echo "[fix_visgrid] ERROR: verification failed" >&2; exit 1; }
echo "[fix_visgrid] OK: occupied 2 Hz; free/unknown every 10th cycle."
