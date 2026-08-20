#!/usr/bin/env bash
# Commit to a coverage-tour target instead of re-picking it every iteration.
#
# ExplorationManager re-solves the HGrid TSP on every planning cycle and takes
# indices[0] as the next cell. Small cost changes flip that first element, so
# measured 2026-08-20 over one flight: the next-cell target changed 190 times in
# 517 s -- 22 per minute, median dwell 0.3 s -- oscillating between three cells
# that took two thirds of the samples. A target that survives 0.3 s cannot be
# driven to, and the aircraft ends up steering toward an average of several
# cells rather than progressing through a tour. It shows up as transit
# inefficiency: 0.41-0.72 distinct 1 m cells per metre flown against ~1.0 for a
# non-repeating sweep.
#
# So hold the chosen (cell, center) pair until one of three things is true:
#
#   * it drops out of the tour entirely -- its frontiers are gone, which is also
#     what happens once the aircraft arrives and clears them;
#   * the commitment has stood longer than /exploration/tour_commit_max_s. This
#     bound is not optional: committing to a cell the aircraft cannot reach is
#     precisely the lock this campaign already fixed once (a viewpoint chosen
#     21315 times in a row while A* failed on every one), and a timeout is what
#     keeps a bad commitment cheap;
#   * there is no commitment yet.
#
# Everything else about the tour is untouched -- the TSP still runs, the tour is
# still published and visualised, and only which element is handed downstream as
# "next" is held steady.
set -euo pipefail

SRC=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp
test -f "$SRC" || { echo "[tour-commit] no $SRC" >&2; exit 1; }

if grep -q "sparx_tour_commit" "$SRC"; then
  echo "[tour-commit] already applied"
  exit 0
fi

python3 - "$SRC" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = "    tsp_indices = indices;"
if anchor not in src:
    sys.exit("[tour-commit] anchor 'tsp_indices = indices;' not found")

block = anchor + """

    // ── SPARX: hold the tour target steady ──────────────────────────────────
    // See patches/fix_falcon_tour_commit.sh for the measurement behind this.
    {
      static int sparx_tour_commit_cell = -1;
      static int sparx_tour_commit_center = -1;
      static double sparx_tour_commit_since = -1.0;
      static double sparx_tour_commit_max_s = -1.0;
      if (sparx_tour_commit_max_s < 0.0)
        ros::param::param("/exploration/tour_commit_max_s",
                          sparx_tour_commit_max_s, 8.0);

      const double sparx_now = ros::Time::now().toSec();
      bool sparx_still_offered = false;
      if (sparx_tour_commit_cell >= 0) {
        for (size_t k = 0; k < indices.size(); ++k) {
          const std::pair<int, int> &pr = cost_mat_id_to_cell_center_id[indices[k]];
          if (pr.first == sparx_tour_commit_cell &&
              pr.second == sparx_tour_commit_center) {
            sparx_still_offered = true;
            break;
          }
        }
      }
      const bool sparx_expired =
          sparx_tour_commit_since > 0.0 &&
          sparx_now - sparx_tour_commit_since > sparx_tour_commit_max_s;

      if (sparx_tour_commit_cell < 0 || !sparx_still_offered || sparx_expired) {
        sparx_tour_commit_cell = next_cell_id;
        sparx_tour_commit_center = next_center_id;
        sparx_tour_commit_since = sparx_now;
      } else if (sparx_tour_commit_cell != next_cell_id) {
        ROS_INFO("[ExplorationManager] Holding tour target cell %d (tour now "
                 "offers %d) for %.1fs of %.1fs",
                 sparx_tour_commit_cell, next_cell_id,
                 sparx_now - sparx_tour_commit_since, sparx_tour_commit_max_s);
        next_cell_id = sparx_tour_commit_cell;
        next_center_id = sparx_tour_commit_center;
        next_cell_id_grid_tour2 = sparx_tour_commit_cell;
      }
    }
"""
open(path, "w").write(src.replace(anchor, block, 1))
print("[tour-commit] applied")
PY

grep -q "sparx_tour_commit" "$SRC" || { echo "[tour-commit] verification failed" >&2; exit 1; }
echo "[tour-commit] OK"
