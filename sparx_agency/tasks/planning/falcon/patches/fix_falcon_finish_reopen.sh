#!/usr/bin/env bash
# Make "exploration finished" recoverable instead of terminal.
#
# fix_falcon_finish_grace.sh made FINISH hard to REACH. It is still impossible
# to LEAVE, and that is the bigger problem: FINISH is an absorbing state with no
# outgoing transition anywhere in exploration_fsm.cpp, and entering it publishes
# replan type 2, which sets task_finished_ in traj_server and ENDS THAT PROCESS.
#
# Measured live 2026-08-18 (sphera_jail, 610 s run, grace patch already in):
# the follower tracked normally for the first 106 s, FALCON logged "Exploration
# finished" once, and the follower then reported holding=True for the remaining
# 560 s -- 84% of the flight -- because traj_server had exited and its last
# /planning/pos_cmd sample aged past the tracker's staleness timeout. The
# aircraft spent nine minutes station-keeping while its own frontier finder,
# which keeps running in FINISH (frontierCallback explicitly handles
# state_ == FINISH), went on finding frontiers for a state machine that had
# already stopped listening.
#
# An empty frontier verdict is a statement about one instant. The map is still
# being fused while the aircraft hovers, clusters keep forming behind it, and a
# region that was unreachable at 108 s is often reachable once a door has been
# mapped. So:
#
#   1. exploration_fsm.cpp -- FINISH re-opens to PLAN_TRAJ as soon as the
#      frontier finder reports work again, after a settling cooldown. The grace
#      counters are reset on the way out, or the next empty verdict would
#      re-finish instantly since they are already past their thresholds.
#   2. traj_server.cpp -- replan type 2 still stops the current trajectory but
#      no longer terminates the process, so there is something alive to fly the
#      re-opened plan. Set /traj_server/exit_on_finish true to restore upstream.
#
# /fsm/finish_reopen_cooldown_sec < 0 restores upstream's terminal FINISH.
#
# Anchor-based script, not a .patch: exploration_fsm.cpp is already rewritten by
# falcon_slow_traj_rescale, falcon_replan_from_pose, fix_falcon_fsm_logspam and
# fix_falcon_finish_grace, so a context diff against upstream will not apply.
# See patches/README.md.
set -euo pipefail

FSM=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_fsm.cpp
test -f "$FSM" || { echo "[finish-reopen] ERROR: $FSM not found" >&2; exit 1; }

python3 - "$FSM" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

if "FSM-finish-reopen" in src:
    print("[finish-reopen] fsm already patched, nothing to do")
    sys.exit(0)

def need(marker, what):
    if marker not in src:
        sys.exit("[finish-reopen] ERROR: %s anchor not found" % what)

need("bool finishVerdictHolds() {", "finishVerdictHolds")
need("    ROS_INFO_ONCE(\"[FSM] Exploration finished\");", "FINISH entry")
need("    static bool clear_vis = false;", "clear_vis")

# 1. Helpers, in the same anonymous namespace as the grace globals so they can
#    reset them. clear_vis is hoisted here too: the re-open has to clear it, and
#    a function-local static cannot be reached from the guard above it.
helpers = '''double g_finish_entered_time = 0.0;
bool g_finish_clear_vis = false;

// How long FINISH must settle before it may re-open, and how many frontier
// clusters must be present. Re-opening on the tick the FSM finished would just
// thrash between the two states while the map catches up.
bool finishReopenAllowed(int frontier_count) {
  static bool loaded = false;
  static double cooldown_s = 0.0;
  static int min_frontiers = 0;
  if (!loaded) {
    ros::param::param("/fsm/finish_reopen_cooldown_sec", cooldown_s, 5.0);
    ros::param::param("/fsm/finish_reopen_min_frontiers", min_frontiers, 1);
    loaded = true;
    ROS_WARN("[finish_reopen] FINISH re-opens to PLAN_TRAJ once >=%d frontier(s) "
             "are known and %.0fs have passed (cooldown <0 disables, restoring "
             "upstream's terminal FINISH)",
             min_frontiers, cooldown_s);
  }
  if (cooldown_s < 0.0 || min_frontiers <= 0)
    return false;
  if (frontier_count < min_frontiers)
    return false;
  if (g_finish_entered_time <= 0.0)
    return false;
  return (ros::Time::now().toSec() - g_finish_entered_time) >= cooldown_s;
}

// Forget this FINISH episode. resetFinishVerdict() already exists (added by
// fix_falcon_finish_grace.sh) and clears the empty-verdict history; this clears
// the re-open state that sits alongside it, so a re-opened mission gets the
// full grace window again rather than finishing on its next empty tick.
void resetFinishReopen() {
  g_finish_entered_time = 0.0;
  g_finish_clear_vis = false;
}

'''
src = src.replace("bool finishVerdictHolds() {", helpers + "bool finishVerdictHolds() {", 1)

# 2. clear_vis becomes a reference to the hoisted flag; every existing use of
#    the name keeps working unchanged.
src = src.replace("    static bool clear_vis = false;",
                  "    bool &clear_vis = g_finish_clear_vis;", 1)

# 3. Stamp when FINISH was entered, for the cooldown.
src = src.replace('    ROS_INFO_ONCE("[FSM] Exploration finished");',
                  '    ROS_INFO_ONCE("[FSM] Exploration finished");\n'
                  '    if (g_finish_entered_time <= 0.0)\n'
                  '      g_finish_entered_time = ros::Time::now().toSec();', 1)

# 4. The re-open, checked on every FINISH tick and before the "finished" verdict
#    is published to traj_server.
probe = '''    std_msgs::Int32 replan_msg;
    replan_msg.data = 2;
    replan_pub_.publish(replan_msg);'''
need(probe, "FINISH replan publish")
src = src.replace(probe, '''    {
      // The frontier finder keeps running in FINISH (see frontierCallback), so
      // ed_->frontiers_ is live here. If it has found work again, go back to
      // planning rather than publish "finished" forever at a traj_server that
      // stops flying on hearing it.
      const int frontier_count = (int)expl_manager_->ed_->frontiers_.size();
      if (finishReopenAllowed(frontier_count)) {
        resetFinishVerdict();
        resetFinishReopen();
        fd_->static_state_ = true;
        ROS_WARN("[FSM] Re-opening exploration: %d frontier cluster(s) found "
                 "after finishing", frontier_count);
        transitState(PLAN_TRAJ, "FSM-finish-reopen");
        break;
      }
    }

''' + probe, 1)

open(path, "w").write(src)
print("[finish-reopen] exploration_fsm.cpp patched")
PYEOF

grep -q "FSM-finish-reopen" "$FSM" || {
  echo "[finish-reopen] ERROR: verification failed, re-open transition absent" >&2
  exit 1
}

# ── traj_server: survive a finish, so there is something to fly afterwards ──
TS=$(ls /catkin_ws/src/FALCON/falcon_planner/*/src/traj_server.cpp 2>/dev/null | head -1)
test -n "$TS" || { echo "[finish-reopen] ERROR: traj_server.cpp not found" >&2; exit 1; }

python3 - "$TS" <<'PYEOF'
import sys

path = sys.argv[1]
src = open(path).read()

if "exit_on_finish" in src:
    print("[finish-reopen] traj_server already patched, nothing to do")
    sys.exit(0)

anchor = '''    t_stop = (time_now - start_time_).toSec();
    task_finished_ = true;'''
if anchor not in src:
    sys.exit("[finish-reopen] ERROR: traj_server replan-type-2 anchor not found")

src = src.replace(anchor, '''    t_stop = (time_now - start_time_).toSec();
    // Stopping the trajectory is right; ending the PROCESS is not. The FSM can
    // leave FINISH again (fix_falcon_finish_reopen.sh), and a dead traj_server
    // means its re-opened plan has nothing to fly it -- the follower just
    // watches its last setpoint go stale and holds station for the rest of the
    // flight. Measured: 560 s of a 610 s run.
    {
      static bool loaded = false;
      static bool exit_on_finish = false;
      if (!loaded) {
        ros::param::param("/traj_server/exit_on_finish", exit_on_finish, false);
        loaded = true;
        ROS_WARN("[TrajServer] exit_on_finish=%s", exit_on_finish ? "true" : "false");
      }
      if (exit_on_finish)
        task_finished_ = true;
    }''', 1)
open(path, "w").write(src)
print("[finish-reopen] traj_server.cpp patched")
PYEOF

grep -q "exit_on_finish" "$TS" || {
  echo "[finish-reopen] ERROR: verification failed, traj_server not patched" >&2
  exit 1
}

echo "[finish-reopen] OK: FINISH is recoverable and traj_server survives it"
