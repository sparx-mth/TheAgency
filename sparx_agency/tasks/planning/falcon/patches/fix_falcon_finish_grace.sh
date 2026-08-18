#!/usr/bin/env bash
# Don't let ONE empty frontier verdict end the mission irreversibly.
#
# Upstream's EXEC_TRAJ handler transits straight to FINISH the first time
# updateFrontierStruct() returns 0, and that is terminal: the resulting replan
# type 2 makes traj_server log "Task finished, traj server shutdown" and stop
# publishing, so nothing short of relaunching the node explores again.
#
# Measured live 2026-08-18 (sphera_jail, with fix_falcon_frontier_visibility.sh
# already in place): a run that had flown 90.2 m in 137.8 s at 0.65 m/s mean
# quit on a single empty verdict immediately after "Replan: cluster covered" --
# and 43 s later the frontier finder was reporting four recoverable clusters,
# every 0.5 s, to a state machine that had already shut down. The map is still
# being fused while the aircraft hovers, and clusters keep forming behind it, so
# an empty verdict is a statement about this instant, not about the world.
#
# So the verdict has to hold before it is believed: for a number of consecutive
# checks AND for a wall-clock span. Both thresholds at 0 restore upstream
# exactly. While the window is open the FSM simply does not transit, which
# leaves the aircraft holding its last setpoint and re-checking on the next tick
# -- the same thing it does between replans.
#
# Anchor-based script, not a .patch: exploration_fsm.cpp is already rewritten by
# falcon_slow_traj_rescale, falcon_replan_from_pose and fix_falcon_fsm_logspam,
# so a context diff against upstream will not apply. See patches/README.md.
set -euo pipefail

SRC=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_fsm.cpp
test -f "$SRC" || { echo "[finish-grace] ERROR: $SRC not found" >&2; exit 1; }

python3 - "$SRC" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

if "finishVerdictHolds" in src:
    print("[finish-grace] already applied, nothing to do")
    sys.exit(0)


def replace_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            "[finish-grace] ERROR: %s anchor matched %d times (want 1)" % (what, n))
    return text.replace(old, new)


HELPERS = '''namespace fast_planner {

namespace {

// How long an "exploration finished" verdict must hold before it is acted on.
// File-statics rather than members so this fix stays in one translation unit;
// there is one FSM per node and it is single-threaded through its timer.
int g_finish_empty_checks = 0;
double g_finish_first_empty_time = 0.0;

void resetFinishVerdict() {
  g_finish_empty_checks = 0;
  g_finish_first_empty_time = 0.0;
}

double finishGraceElapsed() {
  if (g_finish_first_empty_time <= 0.0)
    return 0.0;
  return ros::Time::now().toSec() - g_finish_first_empty_time;
}

// Call once per empty verdict. Returns true when the verdict has held long
// enough, in both checks and seconds, to be believed.
bool finishVerdictHolds() {
  static bool loaded = false;
  static int min_checks = 0;
  static double min_seconds = 0.0;
  if (!loaded) {
    ros::param::param("/fsm/finish_grace_checks", min_checks, 0);
    ros::param::param("/fsm/finish_grace_sec", min_seconds, 0.0);
    loaded = true;
    ROS_WARN("[finish_grace] an empty frontier verdict must hold for %d check(s) "
             "and %.0fs before exploration is declared finished (0/0 restores "
             "upstream: finish on the first empty verdict)",
             min_checks, min_seconds);
  }
  ++g_finish_empty_checks;
  if (g_finish_first_empty_time <= 0.0)
    g_finish_first_empty_time = ros::Time::now().toSec();
  return g_finish_empty_checks >= min_checks && finishGraceElapsed() >= min_seconds;
}

}  // namespace
'''
src = replace_once(src, "namespace fast_planner {\n", HELPERS, "namespace open")

OLD = '''      if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
        // Update frontier and plan new motion
        thread vis_thread(&ExplorationFSM::visualize, this);
        vis_thread.detach();

        transitState(PLAN_TRAJ, "FSM");

        // Use following code can debug the planner step by step
        // transitState(WAIT_TRIGGER, "FSM");
        // fd_->static_state_ = true;

      } else {
        // No frontier detected, finish exploration
        transitState(FINISH, "FSM");
        ROS_WARN("[FSM] Finish exploration: No frontier detected");
        // clearVisMarker();
        // visualize();
      }'''
NEW = '''      if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
        // Update frontier and plan new motion
        thread vis_thread(&ExplorationFSM::visualize, this);
        vis_thread.detach();

        resetFinishVerdict();
        transitState(PLAN_TRAJ, "FSM");

        // Use following code can debug the planner step by step
        // transitState(WAIT_TRIGGER, "FSM");
        // fd_->static_state_ = true;

      } else if (finishVerdictHolds()) {
        // No frontier detected, and the verdict has held long enough to be
        // believed. This is terminal -- traj_server shuts down on the resulting
        // replan type 2 -- so it is deliberately hard to reach.
        transitState(FINISH, "FSM");
        ROS_WARN("[FSM] Finish exploration: No frontier detected (verdict held "
                 "%d check(s) over %.0fs)", g_finish_empty_checks,
                 finishGraceElapsed());
        // clearVisMarker();
        // visualize();
      } else {
        // Hold and re-check. The map is still being fused while the aircraft
        // hovers, and clusters keep forming behind it -- measured recovering
        // four clusters 43s after a verdict that upstream acted on instantly.
        ROS_WARN_THROTTLE(2.0, "[FSM] No frontier, but the finish grace window is "
                          "still open (%d check(s), %.0fs) -- holding and "
                          "re-checking", g_finish_empty_checks,
                          finishGraceElapsed());
      }'''
src = replace_once(src, OLD, NEW, "EXEC_TRAJ finish branch")

open(path, "w").write(src)
print("[finish-grace] applied: an empty frontier verdict must now hold to be believed")
PYEOF

grep -q "finishVerdictHolds" "$SRC" || {
  echo "[finish-grace] ERROR: verification grep failed" >&2; exit 1; }
echo "[finish-grace] OK"
