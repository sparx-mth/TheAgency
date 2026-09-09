#!/bin/bash
# ============================================================
# sparx_astar_legible.sh
#
# Makes A*'s three failure exits distinguishable in the log.
#
# Two of the three are silent upstream: the early-termination warning is
# commented out (astar.cpp, "Early terminated") and the open-set-empty
# diagnostics are commented out at the end of the search; only node-pool
# exhaustion logs. So a 1 ms search timeout and a genuinely enclosed aircraft
# produce the identical "[FSM] Plan fail" line -- and they call for OPPOSITE
# fixes (give the search more time vs. retire the target).
#
# This matters because A*-plan-fail flooding is the single largest sink of
# flight time measured on this platform: 91.9% of all no-moving-reference time,
# 21.7% of every flight, worst case 254 s locked on one cell emitting 17,925
# plan fails. No fix for that can be chosen while the failure mode is unknown.
#
# SUCCESS is logged too, not just failure. The one number that decides whether
# any budget change is worth flying -- how many iterations a search that
# SUCCEEDS actually needs -- was invisible while only NO_PATH was instrumented.
# A timeout at 238 iterations means nothing until you know whether success takes
# 200 or 2000.
#
# Diagnostic only: ROS_WARN_THROTTLE lines, no control flow touched, no return
# value changed. Throttled at 2 s because the caller retries at ~70 Hz.
#
# ALL SIX search overloads are instrumented, not just the first. The first
# version replaced only the first occurrence of each anchor, which landed every
# marker in `search(start, end, MODE, bbox_min, bbox_max)` -- while
# exploration_manager calls the two-argument `search(start, end)`. The result
# was a patch that verified, compiled, shipped, and printed nothing. Each
# message carries __FUNCTION__ so the overload is identifiable.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail
SRC="/catkin_ws/src/FALCON/falcon_planner/pathfinding/src/astar.cpp"
[ -f "${SRC}" ] || { echo "[sparx_astar] ERROR: ${SRC} not found" >&2; exit 1; }

python3 - "$SRC" <<'EOF'
import sys

path = sys.argv[1]
s = open(path).read()
if "sparx-astar" in s:
    print("[sparx_astar] already applied")
    sys.exit(0)

# 1. Early termination on the search-time budget.
old_timeout = '''    if ((ros::Time::now() - t1).toSec() > config_.max_search_time_) {
      // ROS_WARN("[AStar] Early terminated, duration: %f", (ros::Time::now() - t1).toSec());'''
new_timeout = '''    if ((ros::Time::now() - t1).toSec() > config_.max_search_time_) {
      ROS_WARN_THROTTLE(2.0, "[AStar] sparx-astar %s NO_PATH: TIMEOUT after %.4f s "
                        "(budget %.4f), iters %d, nodes %d -- the search ran out "
                        "of time, the space is not necessarily blocked",
                        __FUNCTION__, (ros::Time::now() - t1).toSec(),
                        config_.max_search_time_, iter_num_, use_node_num_);'''
assert old_timeout in s, "timeout exit anchor missing"
s = s.replace(old_timeout, new_timeout)          # ALL overloads

# 2. Node-pool exhaustion: name it the same way so the three read as a set.
old_pool = '''              ROS_WARN("[AStar] Run out of node pool. Duration: %f",
                       (ros::Time::now() - t1).toSec());'''
new_pool = '''              ROS_WARN_THROTTLE(2.0, "[AStar] sparx-astar %s NO_PATH: NODE_POOL "
                                "exhausted (%d) after %.4f s, iters %d",
                                __FUNCTION__, config_.allocate_num_,
                                (ros::Time::now() - t1).toSec(), iter_num_);'''
assert old_pool in s, "node pool anchor missing"
s = s.replace(old_pool, new_pool)                # ALL overloads

# 3. Open set empty -- the genuinely-enclosed case, and the one that was silent.
old_empty = '''  // cout << "open set empty, no path!" << endl;
  // cout << "use node num: " << use_node_num_ << endl;
  // cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;'''
new_empty = '''  ROS_WARN_THROTTLE(2.0, "[AStar] sparx-astar %s NO_PATH: OPEN_SET_EMPTY after "
                    "%.4f s, iters %d, nodes %d -- every reachable cell was "
                    "expanded, so the goal is genuinely unreachable from here",
                    __FUNCTION__, (ros::Time::now() - t1).toSec(), iter_num_,
                    use_node_num_);
  return NO_PATH;'''
assert old_empty in s, "open-set-empty anchor missing"
s = s.replace(old_empty, new_empty)              # ALL overloads

# 4. The SUCCESS path, in every overload. Without it the failure iteration
#    counts cannot be interpreted.
old_ok = """      if (safe) {
        backtrack(cur_node, end_pt);
        return REACH_END;
      }"""
new_ok = """      if (safe) {
        ROS_WARN_THROTTLE(2.0, "[AStar] sparx-astar %s REACH_END after %.4f s, "
                          "iters %d, nodes %d", __FUNCTION__,
                          (ros::Time::now() - t1).toSec(), iter_num_,
                          use_node_num_);
        backtrack(cur_node, end_pt);
        return REACH_END;
      }"""
if old_ok in s:
    s = s.replace(old_ok, new_ok)
old_ok2 = """      backtrack(cur_node, end_pt);
      return REACH_END;"""
new_ok2 = """      ROS_WARN_THROTTLE(2.0, "[AStar] sparx-astar %s REACH_END after %.4f s, "
                        "iters %d, nodes %d", __FUNCTION__,
                        (ros::Time::now() - t1).toSec(), iter_num_,
                        use_node_num_);
      backtrack(cur_node, end_pt);
      return REACH_END;"""
if old_ok2 in s:
    s = s.replace(old_ok2, new_ok2)

open(path, "w").write(s)
print("[sparx_astar] astar.cpp patched")
EOF

grep -q "sparx-astar" "${SRC}" || { echo "[sparx_astar] ERROR: verification failed" >&2; exit 1; }
# Six search overloads x three exits. Anything less means an anchor drifted and
# some overload is silent again -- which is exactly how the first version shipped.
# Six overloads x three failure exits, plus six success exits.
COUNT="$(grep -c 'sparx-astar' "${SRC}")"
test "${COUNT}" -ge 24 || { echo "[sparx_astar] ERROR: expected 24 markers, found ${COUNT}" >&2; exit 1; }
echo "[sparx_astar] OK"
