#!/bin/bash
# Let a run choose its mapping resolution, and say what that costs.
#
# Three problems in the same twenty lines of map_server.cpp, all about the size
# of the voxel grid, which is allocated in full on the first tick and never
# grows:
#
#   1. A typo makes one of the two resolutions unsettable. The node reads
#      `/voxel_mapping/resolutionf_fine` -- note the transposed f -- while
#      voxel_mapping.yaml writes `resolution_fine`. The lookup always falls back
#      to its hard-coded 0.1. Today the default happens to equal the YAML value,
#      so nothing looks wrong; anyone trying to solve a memory problem by editing
#      that line finds it does nothing, and concludes the map is not the problem.
#
#   2. Resolution is chosen by the exploration box's VOLUME: under 4000 m3 the
#      fine one, at or above it the coarse one. That couples two things that
#      should not be coupled, in the surprising direction -- shrinking your
#      exploration box can multiply memory by eight by dropping under the
#      threshold. Our whole-office runs sit at 2222 m3, just inside it: 267 MB
#      at 10 cm against 33 MB at 20 cm, for exactly the same flight.
#
#   3. Nothing ever says how big the grid is. map_server does log its resolution
#      and dimensions -- behind `if (config_.verbose_)`, and voxel_mapping.yaml
#      ships `verbose: false`, so in practice it says nothing and never mentions
#      bytes at all. The failure mode is a container that dies or thrashes during
#      startup allocation, which looks like anything at all.
#
# So: fix the key, honour an explicit `/map_config/map_size/resolution` when one
# is set, and print the grid dimensions and the bill. The volume rule stays as
# the fallback, so an unpatched config behaves exactly as before.
set -e

SRC="${FALCON_SRC:-/catkin_ws/src/FALCON}"
MAP_SERVER="${SRC}/falcon_planner/voxel_mapping/src/map_server.cpp"
[ -f "$MAP_SERVER" ] || { echo "map_server.cpp not found at $MAP_SERVER"; exit 1; }

python3 - "$MAP_SERVER" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()

TYPO = '"/voxel_mapping/resolutionf_fine"'
FIXED = '"/voxel_mapping/resolution_fine"'

VOLUME_RULE = """  double box_volume = (map_config.box_max_ - map_config.box_min_).prod();
  if (box_volume < 4000.0) {
    map_config.resolution_ = map_config.resolution_fine_;
  } else {
    map_config.resolution_ = map_config.resolution_coarse_;
  }
"""

EXPLICIT_RULE = """  // An explicit resolution from the run's `area` block wins. The volume rule
  // below stays as the fallback, so a config that does not set one is unchanged.
  double resolution_explicit = 0.0;
  nh.param("/map_config/map_size/resolution", resolution_explicit, 0.0);
  double box_volume = (map_config.box_max_ - map_config.box_min_).prod();
  if (resolution_explicit > 0.0) {
    map_config.resolution_ = resolution_explicit;
  } else if (box_volume < 4000.0) {
    map_config.resolution_ = map_config.resolution_fine_;
  } else {
    map_config.resolution_ = map_config.resolution_coarse_;
  }
"""

GRID_SIZED = """  map_config.map_size_ = map_config.map_max_ - map_config.map_min_;
  for (int i = 0; i < 3; ++i)
    map_config.map_size_idx_(i) = ceil(map_config.map_size_(i) / map_config.resolution_);
"""

ANNOUNCE = """
  // Six arrays are sized to this grid and allocated up front: occupancy (4 B),
  // TSDF (16 B), ESDF (8 B) and the ESDF's two scratch buffers (8 B each), plus
  // one bit per voxel for the frontier finder's flags. Nothing here is sparse
  // and nothing grows later, so this number is the whole bill and it is knowable
  // now rather than when the allocator gives up.
  {
    const double voxels = (double)map_config.map_size_idx_(0) *
                          (double)map_config.map_size_idx_(1) *
                          (double)map_config.map_size_idx_(2);
    ROS_WARN("[MapServer] voxel grid %d x %d x %d = %.0f voxels at %.3f m -- "
             "%.0f MB allocated up front (resolution %s)",
             map_config.map_size_idx_(0), map_config.map_size_idx_(1),
             map_config.map_size_idx_(2), voxels, map_config.resolution_,
             voxels * 44.125 / (1024.0 * 1024.0),
             resolution_explicit > 0.0 ? "set explicitly" : "from box volume");
  }
"""

changed = 0

if TYPO in source:
    source = source.replace(TYPO, FIXED, 1)
    changed += 1
elif FIXED not in source:
    raise SystemExit("neither the typo nor its fix is present: upstream has "
                     "changed how the fine resolution is read")

if EXPLICIT_RULE not in source:
    if VOLUME_RULE not in source:
        raise SystemExit("could not find the box-volume resolution rule; "
                         "upstream may have changed map_server.cpp")
    source = source.replace(VOLUME_RULE, EXPLICIT_RULE, 1)
    changed += 1

if ANNOUNCE not in source:
    if GRID_SIZED not in source:
        raise SystemExit("could not find where the grid is sized; upstream may "
                         "have changed map_server.cpp")
    source = source.replace(GRID_SIZED, GRID_SIZED + ANNOUNCE, 1)
    changed += 1

open(path, "w").write(source)
print("map_server.cpp: applied %d/3 map-sizing changes" % changed)
PYEOF
