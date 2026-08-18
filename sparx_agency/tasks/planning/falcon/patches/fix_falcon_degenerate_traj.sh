#!/usr/bin/env bash
# Never fly a trajectory with no duration -- and never hand one downstream.
#
# planTrajToView() checks whether the position trajectory it just optimised meets
# the time lower bound, logs an ERROR when it does not, and then carries on and
# uses it anyway. When the optimiser returns a DEGENERATE trajectory -- duration
# exactly 0.00, which happens when the viewpoint is essentially where the
# aircraft already is -- everything downstream divides by that duration.
# `dt_yaw = duration_ / seg_num` becomes 0, and the yaw stage then spins.
#
# Measured live 2026-08-18 (sphera_jail, run 8): the last log line in the whole
# mission was
#
#   [ExplorationManager] Time lower bound not satified in planTrajToView,
#                        time_lb: 3.72, traj_time: 0.00
#
# after which exploration_node stopped emitting anything at all for five
# minutes, pinned one core at 94%, and grew from a normal footprint to 35 GB RSS
# at ~90 MB/s -- an allocating infinite loop on the FSM's own callback thread. It
# had to be killed to keep the host from running out of memory. The mission was
# dead from that instant, with the aircraft holding position and the follower
# reporting a perfectly healthy 0.17 m tracking error the whole time.
#
# FAIL is already a first-class outcome here -- the "No path to next viewpoint
# using coarse A*" branch a few lines earlier returns it, and the FSM responds by
# replanning. So return FAIL rather than flying a non-trajectory.
#
# Anchor-based script, not a .patch: exploration_manager.cpp is already rewritten
# by falcon_deadend_guard and falcon_hgrid_clamp. See patches/README.md.
set -euo pipefail

SRC=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp
test -f "$SRC" || { echo "[degenerate-traj] ERROR: $SRC not found" >&2; exit 1; }

python3 - "$SRC" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

if "degenerate trajectory" in src:
    print("[degenerate-traj] already applied, nothing to do")
    sys.exit(0)

OLD = '''  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb)
    ROS_ERROR("[ExplorationManager] Time lower bound not satified in planTrajToView, "
              "time_lb: %.2lf, traj_time: %.2lf",
              time_lb, planner_manager_->local_data_.position_traj_.getTimeSum());'''

NEW = '''  const double sparx_traj_time = planner_manager_->local_data_.position_traj_.getTimeSum();
  if (sparx_traj_time < time_lb)
    ROS_ERROR("[ExplorationManager] Time lower bound not satified in planTrajToView, "
              "time_lb: %.2lf, traj_time: %.2lf", time_lb, sparx_traj_time);

  // A trajectory with no usable duration is not a trajectory. Upstream logs the
  // line above and flies it anyway; everything downstream divides by this
  // duration (dt_yaw = duration_ / seg_num), and the yaw stage then spins
  // forever allocating -- measured pinning a core at 94% and reaching 35 GB RSS
  // in six minutes, with no further log output of any kind. Written as
  // !(t > min) so a NaN duration takes this branch too.
  static double sparx_min_traj_time = -1.0;
  static double sparx_max_traj_time = -1.0;
  if (sparx_min_traj_time < 0.0) {
    ros::param::param("/exploration/min_traj_time", sparx_min_traj_time, 0.05);
    ros::param::param("/exploration/max_traj_time", sparx_max_traj_time, 600.0);
  }
  if (!(sparx_traj_time > sparx_min_traj_time) || sparx_traj_time > sparx_max_traj_time) {
    ROS_ERROR("[ExplorationManager] planTrajToView produced a degenerate trajectory "
              "(duration %.3lf s, allowed %.3lf..%.0lf) -- failing this plan instead "
              "of flying it", sparx_traj_time, sparx_min_traj_time, sparx_max_traj_time);
    return FAIL;
  }'''

n = src.count(OLD)
if n != 1:
    raise SystemExit(
        "[degenerate-traj] ERROR: anchor matched %d times (want 1)" % n)
open(path, "w").write(src.replace(OLD, NEW))
print("[degenerate-traj] applied: a zero-duration plan now FAILs instead of hanging")
PYEOF

grep -q "degenerate trajectory" "$SRC" || {
  echo "[degenerate-traj] ERROR: verification grep failed" >&2; exit 1; }
echo "[degenerate-traj] OK"
