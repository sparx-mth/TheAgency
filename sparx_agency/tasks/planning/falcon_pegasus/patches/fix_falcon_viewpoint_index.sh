#!/bin/bash
# Stop exploration_node segfaulting when a grid cell has no reachable viewpoint.
#
# `planExploreMotionHGrid` picks the cheapest viewpoint with the usual
# initialise-to--1-and-scan idiom:
#
#     int min_cost_id = -1;
#     for (int i = 0; i < candidates.size(); ++i)
#       if (cost < min_cost) { min_cost = cost; min_cost_id = i; }
#     next_pos = candidates[min_cost_id];      // <-- candidates[-1] if none won
#
# and never checks the outcome. `min_cost_id` stays -1 whenever the candidate
# list is empty or every candidate costs more than the initial bound, and
# `std::vector::operator[](-1)` is undefined behaviour, not an exception. There
# are three such sites.
#
# It survives upstream because it needs a cell whose only frontier has no
# reachable viewpoint -- rare in the small rooms FALCON ships configured for, and
# routine at the end of a real building, when the last frontiers are the awkward
# ones. Measured here: a clean 85-second exploration, 177 trajectories, then
# `exit code -11` mid-flight with no message. The aircraft holds its last
# reference and the run is lost.
#
# The fix is the missing check. An unusable candidate set is a planning failure,
# which the FSM already knows how to handle: it stays in PLAN_TRAJ, re-runs the
# frontier update, and tries again with a map that has since grown.
set -e

MANAGER=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp
[ -f "$MANAGER" ] || { echo "exploration_manager.cpp not found at $MANAGER"; exit 1; }

python3 - "$MANAGER" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()


def guard(condition, what):
    return ("      if (%s) {\n"
            "        ROS_ERROR(\"[ExplorationManager] %s -- treating it as a planning \"\n"
            "                  \"failure instead of indexing out of bounds\");\n"
            "        return FAIL;\n"
            "      }\n" % (condition, what))


SITES = [
    # 1. Reachability check on the best frontier near the next cell.
    ("      // Check reachability from current position\n"
     "      vector<Eigen::Vector3d> path;\n",
     guard("min_cost_id < 0 || min_cost_id >= (int)ed_->points_.size()",
           "no viewpoint near the next grid cell")),
    # 2. The nearest-viewpoint fallback, when that one was unreachable.
    ("      next_pos = ed_->points_[min_cost_id];\n"
     "      next_yaw = ed_->yaws_[min_cost_id];\n",
     guard("min_cost_id < 0 || min_cost_id >= (int)ed_->points_.size()",
           "no reachable viewpoint anywhere in the frontier set")),
    # 3. The single-frontier branch.
    ("      next_pos = ed_->n_points_[0][min_cost_id];\n"
     "      next_yaw = ed_->n_yaws_[0][min_cost_id];\n",
     guard("ed_->n_points_.empty() || min_cost_id < 0 || "
           "min_cost_id >= (int)ed_->n_points_[0].size()",
           "the single frontier in the next grid cell has no usable viewpoint")),
]

patched = 0
for anchor, inserted in SITES:
    if anchor not in source:
        print("WARNING: could not find one of the viewpoint-index sites; upstream "
              "may have changed. Skipping it.")
        continue
    if inserted in source:
        continue
    source = source.replace(anchor, inserted + anchor, 1)
    patched += 1

open(path, "w").write(source)
print("guarded %d/%d unchecked viewpoint indices" % (patched, len(SITES)))
if patched != len(SITES):
    raise SystemExit("refusing to build with an unguarded viewpoint index: the "
                     "crash this patch exists to prevent would still happen")
PYEOF
