#!/bin/bash
# ============================================================
# sparx_map_stats.sh
#
# Publishes the mapping-RATE counters the campaign needs on
# /voxel_mapping/map_stats (std_msgs/String, JSON, 2 Hz).
#
# FALCON already exposes one mapping number: /voxel_mapping/map_coverage, a
# Float32 of explored volume. That cannot answer the questions the campaign
# actually asks -- how long a square metre takes, how long a cubic metre takes,
# and how the NEW-voxel rate decays as the flight revisits ground it has already
# seen. The occupancy clouds carry the raw data but are 14-450 MB per message,
# so sampling them is not an option.
#
# MapServer::publishMapCoverage() already walks the whole occupancy box every
# 0.5 s to count non-UNKNOWN voxels. This adds the other counters to THAT walk,
# so the cost is a handful of increments and two bitmaps, not a second pass:
#   known / free / occupied voxel counts   (the free-vs-occupied split)
#   known + occupied 2D footprint, m2      (columns with any known voxel)
#   known 1x1x1 m cells                    (the operator's 3D cell question)
#   the box totals, so every count has a denominator
# The new-voxel rate is the difference between consecutive samples and is
# computed offline; nothing needs to be tracked in C++.
#
# std_msgs/String + JSON on purpose: a new .msg would have to be taught to the
# ROS1<->ROS2 bridge and to every consumer. map_coverage is left untouched, so
# nothing downstream changes behaviour.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail
ROOT="/catkin_ws/src/FALCON/falcon_planner/voxel_mapping"
SRC="${ROOT}/src/map_server.cpp"
HDR="${ROOT}/include/voxel_mapping/map_server.h"
for f in "${SRC}" "${HDR}"; do
  [ -f "${f}" ] || { echo "[sparx_map_stats] ERROR: ${f} not found" >&2; exit 1; }
done

python3 - "$SRC" "$HDR" <<'EOF'
import sys

src_path, hdr_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()
if "map_stats_pub_" in s:
    print("[sparx_map_stats] already applied to map_server.cpp")
else:
    old_inc = '#include "voxel_mapping/map_server.h"'
    new_inc = (old_inc + "\n\n"
               "// SPARX: publishMapCoverage() formats its counters as JSON.\n"
               "#include <std_msgs/String.h>\n"
               "#include <cmath>\n"
               "#include <iomanip>\n"
               "#include <sstream>\n"
               "#include <vector>")
    assert old_inc in s, "map_server.h include anchor missing"
    s = s.replace(old_inc, new_inc, 1)

    old_adv = ('  map_coverage_pub_ = nh.advertise<std_msgs::Float32>'
               '("/voxel_mapping/map_coverage", 10);')
    assert old_adv in s, "map_coverage advertise anchor missing"
    s = s.replace(old_adv, old_adv + "\n"
                  "  // SPARX: per-sample mapping-rate counters, see publishMapCoverage().\n"
                  '  map_stats_pub_ = nh.advertise<std_msgs::String>("/voxel_mapping/map_stats", 10);',
                  1)

    start = s.index("void MapServer::publishMapCoverage() {")
    end = s.index("void MapServer::depthToPointcloud(")
    body = r'''void MapServer::publishMapCoverage() {
  std_msgs::Float32 msg;
  int map_coverage_num = 0;
  // SPARX: mapping-rate counters, computed inside the walk this function
  // already does, so they cost a few increments rather than a second pass.
  int free_num = 0, occupied_num = 0;
  const Eigen::Vector3i bmin = occupancy_grid_->map_config_.box_min_idx_;
  const Eigen::Vector3i bmax = occupancy_grid_->map_config_.box_max_idx_;
  const int nx = std::max(1, bmax.x() - bmin.x());
  const int ny = std::max(1, bmax.y() - bmin.y());
  const int nz = std::max(1, bmax.z() - bmin.z());
  const double res = tsdf_->map_config_.resolution_;
  // Voxels per metre, so a 1x1x1 m cell is one bucket at any resolution.
  const int per_m = std::max(1, static_cast<int>(std::lround(1.0 / res)));
  const int cx = nx / per_m + 1, cy = ny / per_m + 1, cz = nz / per_m + 1;
  std::vector<uint8_t> column_known(static_cast<size_t>(nx) * ny, 0);
  std::vector<uint8_t> column_occupied(static_cast<size_t>(nx) * ny, 0);
  std::vector<uint8_t> cell_known(static_cast<size_t>(cx) * cy * cz, 0);
  for (int x = bmin.x(); x < bmax.x(); x++) {
    for (int y = bmin.y(); y < bmax.y(); y++) {
      for (int z = bmin.z(); z < bmax.z(); z++) {
        VoxelIndex idx(x, y, z);
        const OccupancyType value = occupancy_grid_->getVoxel(idx).value;
        if (value == OccupancyType::UNKNOWN)
          continue;
        map_coverage_num++;
        const bool occupied = (value == OccupancyType::OCCUPIED);
        if (occupied)
          occupied_num++;
        else
          free_num++;
        const size_t col = static_cast<size_t>(x - bmin.x()) * ny + (y - bmin.y());
        column_known[col] = 1;
        if (occupied)
          column_occupied[col] = 1;
        const size_t cell =
            (static_cast<size_t>((x - bmin.x()) / per_m) * cy + ((y - bmin.y()) / per_m)) *
                cz + ((z - bmin.z()) / per_m);
        cell_known[cell] = 1;
      }
    }
  }
  long columns_known = 0, columns_occupied = 0, cells_known = 0;
  for (size_t i = 0; i < column_known.size(); i++) {
    columns_known += column_known[i];
    columns_occupied += column_occupied[i];
  }
  for (size_t i = 0; i < cell_known.size(); i++)
    cells_known += cell_known[i];

  map_coverage_ = map_coverage_num * tsdf_->map_config_.resolution_ *
                  tsdf_->map_config_.resolution_ * tsdf_->map_config_.resolution_;
  msg.data = map_coverage_;
  map_coverage_pub_.publish(msg);

  // SPARX: the same walk's counters as JSON on a side topic. A String needs no
  // new .msg, so neither the bridge nor any consumer has to learn a type.
  std_msgs::String stats;
  std::ostringstream out;
  out << std::fixed
      << "{\"stamp\":" << std::setprecision(3) << ros::Time::now().toSec()
      << ",\"resolution\":" << std::setprecision(4) << res
      << ",\"known_voxels\":" << map_coverage_num
      << ",\"free_voxels\":" << free_num
      << ",\"occupied_voxels\":" << occupied_num
      << ",\"known_volume_m3\":" << std::setprecision(3) << map_coverage_
      << ",\"occupied_volume_m3\":" << occupied_num * res * res * res
      << ",\"known_area_m2\":" << columns_known * res * res
      << ",\"occupied_area_m2\":" << columns_occupied * res * res
      << ",\"known_cells_1m3\":" << cells_known
      << ",\"box_voxels\":" << static_cast<long>(nx) * ny * nz
      << ",\"box_columns\":" << static_cast<long>(nx) * ny
      << ",\"box_cells_1m3\":" << static_cast<long>(cx) * cy * cz << "}";
  stats.data = out.str();
  map_stats_pub_.publish(stats);
}

'''
    s = s[:start] + body + s[end:]
    open(src_path, "w").write(s)
    print("[sparx_map_stats] map_server.cpp patched")

h = open(hdr_path).read()
if "map_stats_pub_" in h:
    print("[sparx_map_stats] already applied to map_server.h")
else:
    old_h = "  ros::Publisher map_coverage_pub_;"
    assert old_h in h, "map_coverage_pub_ declaration anchor missing"
    h = h.replace(old_h, old_h + "\n"
                  "  //! SPARX: mapping-rate counters as JSON, see publishMapCoverage().\n"
                  "  ros::Publisher map_stats_pub_;", 1)
    open(hdr_path, "w").write(h)
    print("[sparx_map_stats] map_server.h patched")
EOF

grep -q "map_stats_pub_" "${SRC}" || { echo "[sparx_map_stats] ERROR: verification failed in ${SRC}" >&2; exit 1; }
grep -q "map_stats_pub_" "${HDR}" || { echo "[sparx_map_stats] ERROR: verification failed in ${HDR}" >&2; exit 1; }
echo "[sparx_map_stats] OK"
