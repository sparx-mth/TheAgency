#!/bin/bash
# ============================================================
# sparx_viewpoint_log.sh
#
# Logs the viewpoint the exploration manager actually SELECTS.
#
# WHY (LOOP_FIXES.md F21). Seeding FALCON's blocked regions is meant to stop the
# planner *targeting* the dozen cells that wedge the aircraft (LOOP_BUGS.md B33)
# and produce half its trajectory rejections (B37). F21 could not be measured
# because nothing logs the quantity the mechanism changes:
#   - time inside the cells is mostly TRANSIT, which blocked regions do not touch;
#   - dormant-frontier counts run 106-157 per flight from ordinary runtime
#     blocking, so twelve seeded regions are invisible in them.
# Both were chosen without checking what the mechanism could move.
#
# The quantity that matters is simply: where is the next viewpoint? One throttled
# line at the selection site makes "viewpoints selected inside the seeded cells"
# countable offline, which is exactly the near-binary check F21 wants -- a
# working seed should drive it to zero while the control keeps selecting there.
#
# Diagnostic only: one ROS_INFO on a path that already logs. No control flow
# changed, nothing gated -- it is useful on every flight, not just an experiment.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp
[ -f "$F" ] || { echo "[sparx-vp] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = """  // Plan trajectory (position and yaw) to the next viewpoint
  auto t1 = ros::Time::now();"""
new = """  // Plan trajectory (position and yaw) to the next viewpoint
  // SPARX: log WHERE the planner chose to go. Blocked regions change which
  // viewpoints are offered, and nothing else in this planner records that --
  // see LOOP_FIXES.md F21, where the absence of this line made the experiment
  // unmeasurable.
  ROS_INFO_THROTTLE(0.5, "[sparx-vp] next_viewpoint (%.2f, %.2f, %.2f) yaw %.2f",
                    next_pos.x(), next_pos.y(), next_pos.z(), next_yaw);
  auto t1 = ros::Time::now();"""

if "[sparx-vp] next_viewpoint" in s:
    print("[sparx-vp] already applied")
else:
    assert s.count(old) == 1, "anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "\[sparx-vp\] next_viewpoint" "$F" || { echo "[sparx-vp] ERROR: marker missing" >&2; exit 1; }
echo "[sparx-vp] OK: selected viewpoint is logged"
