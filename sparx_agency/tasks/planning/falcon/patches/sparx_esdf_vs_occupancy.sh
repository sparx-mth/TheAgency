#!/bin/bash
# ============================================================
# sparx_esdf_vs_occupancy.sh
#
# Reports the ESDF's opinion at the exact point the pre-publish collision check
# rejects a trajectory.
#
# WHY. 41% of everything the planner produces is thrown away by that check
# (LOOP_BUGS.md B37/B38), and B40 established the optimiser is NOT to blame for
# lack of effort: 70 of 70 solves terminated on xtol, using under 1 ms of their
# 10 ms budget. It converges -- and its converged answer still passes through
# voxels the map calls OCCUPIED.
#
# That leaves two candidates, and they call for opposite fixes:
#   1. the soft clearance penalty loses the trade against the other cost terms;
#   2. the two halves disagree about the world -- the optimiser minimises against
#      the ESDF while the checker tests raw occupancy, so a stale, smoothed or
#      truncated distance field would let it believe a curve is clear that the
#      checker knows is not.
#
# The optimiser reads clearance as
#   map_server_->getESDF()->getDistanceAndGradient(q, dist, grad)
# and the checker holds the same map_server_. Asking the ESDF what it thinks at
# the rejecting point separates the two outright: comfortable clearance reported
# where occupancy says OCCUPIED is candidate 2, and nothing else is.
#
# Diagnostic only: one extra ESDF query on the rejection path, which already
# logs and returns false. No control flow changed.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/fast_planner/src/planner_manager.cpp
[ -f "$F" ] || { echo "[sparx-esdf] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = """    if (map_server_->getOccupancy(fut_pt) == voxel_mapping::OccupancyType::OCCUPIED) {
      ROS_ERROR("[FastPlannerManager] Collision detected at (%.2f, %.2f, %.2f)", fut_pt.x(),
                fut_pt.y(), fut_pt.z());
      return false;
    }"""
new = """    if (map_server_->getOccupancy(fut_pt) == voxel_mapping::OccupancyType::OCCUPIED) {
      // SPARX: ask the ESDF what IT thinks here. The optimiser minimises
      // against the distance field while this test reads raw occupancy; if the
      // two disagree, the 41% rejection rate is a map-consistency problem and
      // not a planning one. See LOOP_BUGS.md B40.
      double sparx_d = -1.0;
      Eigen::Vector3d sparx_g;
      map_server_->getESDF()->getDistanceAndGradient(fut_pt, sparx_d, sparx_g);
      ROS_ERROR("[FastPlannerManager] Collision detected at (%.2f, %.2f, %.2f)", fut_pt.x(),
                fut_pt.y(), fut_pt.z());
      ROS_WARN_THROTTLE(1.0, "[sparx-esdf] reject at (%.2f, %.2f, %.2f) esdf_dist=%.3f t=%.2f",
                        fut_pt.x(), fut_pt.y(), fut_pt.z(), sparx_d, t_now + fut_t);
      return false;
    }"""

if "[sparx-esdf] reject at" in s:
    print("[sparx-esdf] already applied")
else:
    assert s.count(old) == 1, "anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "\[sparx-esdf\] reject at" "$F" || { echo "[sparx-esdf] ERROR: marker missing" >&2; exit 1; }
echo "[sparx-esdf] OK: ESDF distance logged at the rejection point"
