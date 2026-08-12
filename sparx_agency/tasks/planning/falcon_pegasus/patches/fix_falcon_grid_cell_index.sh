#!/bin/bash
# Stop exploration_node segfaulting when the aircraft leaves the exploration box.
#
# `UniformGrid::positionToGridCellId` turns a world position into a flat cell
# index with no bounds check at all:
#
#     int x = std::floor((pos.x() - config_.bbox_min_.x()) / config_.cell_size_.x());
#     int y = std::floor((pos.y() - config_.bbox_min_.y()) / config_.cell_size_.y());
#     int z = std::floor((pos.z() - config_.bbox_min_.z()) / config_.cell_size_.z());
#     return x + y * num_cells_x_ + z * num_cells_x_ * num_cells_y_;
#
# Every term can go negative or run past the end. `positionToGridCellCenterId`
# then does `uniform_grid_[cell_id].bbox_min_` on the result, and
# `std::vector::operator[]` with a negative index is undefined behaviour, not an
# exception. Measured: `SIGSEGV (Address not mapped to object [0x20ea301])`
# inside `positionToGridCellCenterId`, 43 seconds into a healthy flight, with
# `traj_server` outliving the planner so the aircraft saw only a frozen plan.
#
# WHY IT HAPPENS HERE AND NOT UPSTREAM. The exploration box is not the flyable
# space. In these runs it is inset from the building -- `box_min_x` -23.0
# against a `map_min_x` of -25.0 -- and its floor is z = 1.0 while the aircraft
# cruises at 1.4 m. So an aircraft exploring an outer wall, or dipping under the
# band during a manoeuvre (the previous flight logged a collision check at
# z = 0.63), is legitimately OUTSIDE the box while still flying perfectly. That
# is routine for us and rare in the small single-room setups FALCON ships
# configured for.
#
# THE FIX IS TO CLAMP, NOT TO REJECT. "Which cell is this position in" has a
# sensible answer for a position just outside the grid -- the nearest edge cell
# -- and every caller wants that answer rather than a failure. Two of them
# (`getLayerCellId`) only compare the id against another id, and a clamped
# value compares correctly; the third indexes the vector, and a clamped value is
# in range by construction. Rejecting instead would mean teaching four call
# sites what -1 means, for no gain.
#
# Upstream knew: `positionToGridCellCenterId` carries the assertion that would
# have caught this, commented out --
#
#     // CHECK_GE(cell_id, 0) << "Invalid cell id in positionToGridCellCenterId";
#
# and `-1` ids are normal in operation (the FSM logs "Current cell id: 30,
# center id: -1" on healthy ticks), so the check could not simply be switched
# back on.
#
# Same shape as patches/fix_falcon_viewpoint_index.sh: an unchecked index into a
# std::vector, in code that only meets the bad input on a building-sized map.
set -e

GRID=/catkin_ws/src/FALCON/falcon_planner/exploration_preprocessing/src/hierarchical_grid.cpp
[ -f "$GRID" ] || { echo "hierarchical_grid.cpp not found at $GRID"; exit 1; }

python3 - "$GRID" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()

original = """int UniformGrid::positionToGridCellId(const Position &pos) {
  int x = std::floor((pos.x() - config_.bbox_min_.x()) / config_.cell_size_.x());
  int y = std::floor((pos.y() - config_.bbox_min_.y()) / config_.cell_size_.y());
  int z = std::floor((pos.z() - config_.bbox_min_.z()) / config_.cell_size_.z());

  return x + y * config_.num_cells_x_ + z * config_.num_cells_x_ * config_.num_cells_y_;
}"""

patched = """int UniformGrid::positionToGridCellId(const Position &pos) {
  int x = std::floor((pos.x() - config_.bbox_min_.x()) / config_.cell_size_.x());
  int y = std::floor((pos.y() - config_.bbox_min_.y()) / config_.cell_size_.y());
  int z = std::floor((pos.z() - config_.bbox_min_.z()) / config_.cell_size_.z());

  // Clamp to the grid. The exploration box is not the flyable space -- it is
  // inset from the building and its floor sits above the ground -- so an
  // aircraft working an outer wall, or dipping below the band, is routinely
  // outside this grid while flying perfectly well. Without the clamp the
  // expression below returns a negative or past-the-end index, and
  // positionToGridCellCenterId indexes uniform_grid_ with it, which is
  // undefined behaviour rather than an error. The nearest edge cell is the
  // right answer for every caller: two only compare the id, and the third
  // subscripts a vector that this now keeps in range.
  x = std::max(0, std::min(x, config_.num_cells_x_ - 1));
  y = std::max(0, std::min(y, config_.num_cells_y_ - 1));
  z = std::max(0, std::min(z, config_.num_cells_z_ - 1));

  return x + y * config_.num_cells_x_ + z * config_.num_cells_x_ * config_.num_cells_y_;
}"""

if source.count(original) != 1:
    raise SystemExit("patch: expected exactly 1 positionToGridCellId body, found %d"
                     % source.count(original))
source = source.replace(original, patched)

# Belt and braces at the site that actually crashed. The clamp above makes an
# out-of-range id impossible, so this can only fire if the grid is empty or its
# cell counts disagree with its contents -- both of which are configuration
# faults worth a message rather than a signal.
guard_at = """void UniformGrid::positionToGridCellCenterId(const Position &pos, int &cell_id, int &center_id) {
  cell_id = -1;
  center_id = -1;

  cell_id = positionToGridCellId(pos);
"""
guarded = """void UniformGrid::positionToGridCellCenterId(const Position &pos, int &cell_id, int &center_id) {
  cell_id = -1;
  center_id = -1;

  cell_id = positionToGridCellId(pos);

  // Unreachable once positionToGridCellId clamps, and cheap. A grid that is
  // empty or whose cell counts disagree with its contents is a configuration
  // fault; saying so beats indexing past the end of the vector.
  if (cell_id < 0 || cell_id >= (int)uniform_grid_.size()) {
    ROS_ERROR("[UniformGrid] position (%.2f, %.2f, %.2f) maps to cell %d of %zu -- "
              "refusing to index the grid; check the hgrid configuration",
              pos.x(), pos.y(), pos.z(), cell_id, uniform_grid_.size());
    cell_id = -1;
    return;
  }
"""

if source.count(guard_at) != 1:
    raise SystemExit("patch: expected exactly 1 positionToGridCellCenterId head, found %d"
                     % source.count(guard_at))
source = source.replace(guard_at, guarded)

# No #include is added on purpose. The file already calls std::max and std::min
# four times, so <algorithm> is reaching it transitively -- and its first
# include is <pcl/kdtree/kdtree_flann.h>, which is exactly the wrong place to
# insert a header. Doing so made gcc 9 die with an "internal compiler error:
# Segmentation fault ... during RTL pass: fwprop1" inside the UniformGrid
# constructor: PCL and Eigen are include-order sensitive, and the template
# instantiation that follows is heavy enough to turn that into a compiler crash
# rather than a diagnostic.
if "std::max" not in source or "std::min" not in source:
    raise SystemExit("patch: hierarchical_grid.cpp no longer uses std::max/std::min, so "
                     "<algorithm> may not be reaching it transitively any more -- add the "
                     "include AFTER the PCL headers, never before")

open(path, "w").write(source)
print("patched: grid cell ids are clamped to the grid")
PYEOF
